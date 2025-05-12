import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm

world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads,
                                                        self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads,
                                                        self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
                                 persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim),
                                 persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        多注意力层(Multi-head Attention Layer)的前向传播函数

        参数:
            x: 输入张量，形状为 [batch_size, seq_len, dim]，即 [2, 128, 2048]
            start_pos: 序列的起始位置，用于缓存管理
            freqs_cis: 预计算的旋转位置编码的复数指数值，用于RoPE位置编码
            mask: 可选的注意力掩码张量，用于控制注意力计算范围
        """
        # 获取输入张量的维度信息
        bsz, seqlen, _ = x.size()  # bsz=2, seqlen=128, _=2048

        # 计算序列的结束位置，用于缓存管理
        end_pos = start_pos + seqlen  # end_pos = start_pos + 128

        # 处理查询(Q)投影
        if self.q_lora_rank == 0:
            # 如果没有使用LoRA，直接进行线性变换
            # 将输入从[2, 128, 2048]投影到[2, 128, n_heads * qk_head_dim]
            q = self.wq(x)
        else:
            # 使用LoRA进行查询投影
            # 1. 首先通过低秩矩阵A: [2, 128, 2048] -> [2, 128, q_lora_rank]
            # 2. 应用RMSNorm归一化
            # 3. 最后通过低秩矩阵B: [2, 128, q_lora_rank] -> [2, 128, n_heads * qk_head_dim]
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        # 重塑查询张量以分离注意力头
        # 从[2, 128, n_heads * qk_head_dim]变为[2, 128, n_local_heads, qk_head_dim]
        # n_local_heads是每个GPU上的注意力头数量
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)

        # 将查询分为非位置编码部分和位置编码部分
        # q_nope: [2, 128, n_local_heads, qk_nope_head_dim]
        # q_pe: [2, 128, n_local_heads, qk_rope_head_dim]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # 对位置编码部分应用旋转位置编码(RoPE)
        # 输入: [2, 128, n_local_heads, qk_rope_head_dim]
        # 输出: [2, 128, n_local_heads, qk_rope_head_dim]
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # 处理键值(KV)投影
        # 将输入从[2, 128, 2048]投影到[2, 128, kv_lora_rank + qk_rope_head_dim]
        kv = self.wkv_a(x)

        # 分离键值投影和位置编码
        # kv: [2, 128, kv_lora_rank]
        # k_pe: [2, 128, qk_rope_head_dim]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # 对键的位置编码部分应用旋转位置编码
        # 1. 增加一个维度: [2, 128, qk_rope_head_dim] -> [2, 128, 1, qk_rope_head_dim]
        # 2. 应用RoPE: [2, 128, 1, qk_rope_head_dim] -> [2, 128, 1, qk_rope_head_dim]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        if attn_impl == "naive":
            # 使用朴素实现方式
            # 合并查询的非位置编码和位置编码部分
            # 从[2, 128, n_local_heads, qk_head_dim]变为[2, 128, n_local_heads, qk_head_dim]
            q = torch.cat([q_nope, q_pe], dim=-1)

            # 对键值进行投影和归一化
            # 1. 应用RMSNorm: [2, 128, kv_lora_rank] -> [2, 128, kv_lora_rank]
            # 2. 线性投影: [2, 128, kv_lora_rank] -> [2, 128, n_local_heads * (qk_nope_head_dim + v_head_dim)]
            # 3. 重塑: [2, 128, n_local_heads, qk_nope_head_dim + v_head_dim]
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)

            # 分离键和值
            # k_nope: [2, 128, n_local_heads, qk_nope_head_dim]
            # v: [2, 128, n_local_heads, v_head_dim]
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            # 合并键的非位置编码和位置编码部分
            # 1. 扩展k_pe: [2, 128, 1, qk_rope_head_dim] -> [2, 128, n_local_heads, qk_rope_head_dim]
            # 2. 合并: [2, 128, n_local_heads, qk_head_dim]
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

            # 更新键值缓存
            # k_cache: [2, max_seq_len, n_local_heads, qk_head_dim]
            # v_cache: [2, max_seq_len, n_local_heads, v_head_dim]
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v

            # 计算注意力分数
            # 1. 矩阵乘法: [2, 128, n_local_heads, qk_head_dim] @ [2, end_pos, n_local_heads, qk_head_dim]^T
            # 2. 结果: [2, 128, n_local_heads, end_pos]
            # 3. 应用缩放因子
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # 使用吸收实现方式
            # 获取权重并进行反量化（如果需要）
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight,
                                                                                      self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)

            # 计算非位置编码部分的注意力分数
            # 1. 矩阵乘法: [2, 128, n_local_heads, qk_nope_head_dim] @ [n_local_heads, qk_nope_head_dim, kv_lora_rank]
            # 2. 结果: [2, 128, n_local_heads, kv_lora_rank]
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])

            # 更新键值缓存和位置编码缓存
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

            # 计算总注意力分数
            # 1. 非位置编码部分: [2, 128, n_local_heads, kv_lora_rank] @ [2, end_pos, kv_lora_rank]^T
            # 2. 位置编码部分: [2, 128, n_local_heads, qk_rope_head_dim] @ [2, end_pos, qk_rope_head_dim]^T
            # 3. 相加并应用缩放因子
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

        # 应用注意力掩码（如果存在）
        if mask is not None:
            # 扩展掩码维度以匹配注意力头: [seqlen, seqlen] -> [1, seqlen, 1, seqlen]
            scores += mask.unsqueeze(1)

        # 应用softmax获取注意力权重
        # 1. 在最后一个维度上应用softmax
        # 2. 转换为float32以提高数值稳定性
        # 3. 转回原始数据类型
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        if attn_impl == "naive":
            # 使用朴素实现方式计算输出
            # 1. 注意力权重与值相乘: [2, 128, n_local_heads, end_pos] @ [2, end_pos, n_local_heads, v_head_dim]
            # 2. 结果: [2, 128, n_local_heads, v_head_dim]
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # 使用吸收实现方式计算输出
            # 1. 注意力权重与键值缓存相乘: [2, 128, n_local_heads, end_pos] @ [2, end_pos, kv_lora_rank]
            # 2. 结果: [2, 128, n_local_heads, kv_lora_rank]
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            # 3. 与权重矩阵相乘: [2, 128, n_local_heads, kv_lora_rank] @ [n_local_heads, kv_lora_rank, v_head_dim]
            # 4. 结果: [2, 128, n_local_heads, v_head_dim]
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        # 通过输出投影层
        # 1. 展平注意力头: [2, 128, n_local_heads, v_head_dim] -> [2, 128, n_local_heads * v_head_dim]
        # 2. 线性投影: [2, 128, n_local_heads * v_head_dim] -> [2, 128, dim]
        x = self.wo(x.flatten(2))

        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gate的前向传播函数，用于计算路由权重和选择专家

        参数:
            x: 输入张量，形状为 [batch_size * seq_len, dim]，即 [256, 2048]

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - weights: 路由权重，形状为 [256, n_activated_experts]
                - indices: 选择的专家索引，形状为 [256, n_activated_experts]
        """
        # 计算每个token对每个专家的得分
        # 通过线性层将输入映射到专家数量的维度
        # scores形状: [256, n_routed_experts]
        scores = linear(x, self.weight)

        # 根据score_func选择不同的激活函数
        if self.score_func == "softmax":
            # 使用softmax将得分转换为概率分布
            # 确保所有专家的权重和为1
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            # 使用sigmoid将得分转换为0-1之间的值
            scores = scores.sigmoid()

        # 保存原始得分，用于后续计算最终权重
        original_scores = scores

        # 如果存在偏置项，将其加到得分上
        if self.bias is not None:
            scores = scores + self.bias

        # 如果使用专家分组
        if self.n_groups > 1:
            # 将得分重塑为分组形式
            # 形状从 [256, n_routed_experts] 变为 [256, n_groups, experts_per_group]
            scores = scores.view(x.size(0), self.n_groups, -1)

            # 计算每个组的得分
            if self.bias is None:
                # 如果没有偏置，使用每个组中的最大得分
                group_scores = scores.amax(dim=-1)
            else:
                # 如果有偏置，使用每个组中前两个最高得分的和
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

            # 选择得分最高的topk_groups个组
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]

            # 创建掩码，将未选择的组的得分设为负无穷
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)

        # 选择得分最高的topk个专家
        # indices形状: [256, n_activated_experts]
        indices = torch.topk(scores, self.topk, dim=-1)[1]

        # 从原始得分中获取选中专家的权重
        # weights形状: [256, n_activated_experts]
        weights = original_scores.gather(1, indices)

        # 如果使用sigmoid激活函数，需要重新归一化权重
        if self.score_func == "sigmoid":
            # 确保每个token的专家权重和为1
            weights /= weights.sum(dim=-1, keepdim=True)

        # 应用路由缩放因子
        weights *= self.route_scale

        # 将权重转换为与输入相同的数据类型
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
             for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE (Mixture of Experts) 的前向传播函数

        参数:
            x: 输入张量，形状为 [batch_size, seq_len, dim]，即 [2, 128, 2048]

        返回:
            输出张量，形状与输入相同 [2, 128, 2048]
        """
        # 保存原始输入形状，用于最后恢复
        shape = x.size()  # shape = (2, 128, 2048)

        # 将输入张量重塑为2D，便于后续处理
        # 从 [2, 128, 2048] 变为 [256, 2048]
        # 256 = 2 * 128，即所有序列位置展平
        x = x.view(-1, self.dim)

        # 通过门控网络获取路由权重和专家索引
        # weights: 形状为 [256, n_activated_experts]，表示每个token被分配给每个激活专家的权重
        # indices: 形状为 [256, n_activated_experts]，表示每个token被分配给的专家索引
        weights, indices = self.gate(x)

        # 初始化输出张量，形状与输入相同 [256, 2048]
        y = torch.zeros_like(x)

        # 统计每个专家被分配到的token数量
        # counts: 长度为n_routed_experts的列表，表示每个专家被分配到的token数量
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        # 遍历当前进程负责的专家
        for i in range(self.experts_start_idx, self.experts_end_idx):
            # 如果当前专家没有被分配到任何token，跳过
            if counts[i] == 0:
                continue

            # 获取当前专家模块
            expert = self.experts[i]

            # 找出被分配到当前专家的所有token的索引
            # idx: 表示token在展平后的序列中的位置
            # top: 表示该token在当前专家的激活专家列表中的位置
            idx, top = torch.where(indices == i)

            # 对分配到当前专家的token进行处理
            # 1. 通过专家网络处理这些token
            # 2. 将结果乘以对应的路由权重
            # 3. 将结果累加到输出张量中
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        # 处理共享专家
        # 对所有token应用共享专家网络
        z = self.shared_experts(x)  # 形状: [256, 2048]

        # 如果是分布式训练，需要同步所有进程的结果
        if world_size > 1:
            dist.all_reduce(y)

        # 将输出重塑回原始形状并返回
        # 1. 将y和z相加得到最终输出
        # 2. 将形状从 [256, 2048] 变回 [2, 128, 2048]
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

        # 初始化权重
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.head.weight)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


def make_tokenizer():
    from transformers import AutoTokenizer
    # 指定仓库路径
    ckpt_path = "/data/quickllm/DeepSeek-V3/deepseek-V3"
    ckpt_path = "/data2/users/yszhang/quickllm/DeepSeek-V3/deepseek-V3"
    # 只加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    return tokenizer


def make_input_x(text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)  # 编码文本，不加特殊 token
    print("Original text:", text)
    print("Encoded token IDs:", token_ids)

    # 调整为 (2, 128) 的形状
    seq_len = 128
    if len(token_ids) > seq_len:
        token_ids = token_ids[:seq_len]  # 截断
    else:
        token_ids = token_ids + [tokenizer.pad_token_id] * (seq_len - len(token_ids))  # 填充
    x = torch.tensor([token_ids, token_ids], device="cuda")  # 复制为 (2, 128)
    print("Input x shape:", x.shape)

    # 解码 x 的每个样本，验证输入
    for i in range(x.size(0)):
        decoded_text = tokenizer.decode(x[i].tolist(), skip_special_tokens=True)
        print(f"Sample {i + 1} decoded text: {decoded_text}")
    return x, token_ids


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 指定只有第二张GPU对PyTorch可见

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(42)
    args = ModelArgs()

    tokenizer = make_tokenizer()
    # 用真实文本生成输入 x
    text = "Hello, how are you today? I hope you're doing well and enjoying this beautiful day."
    x, token_ids = make_input_x(text)

    # 初始化并运行模型
    model = Transformer(args)
    logits = model(x)  # (2, 102400)
    print("Logits sample (first 10 values):", logits[0, :10].tolist())  # 查看第一个样本的前 10 个值
    print("Logits max indices:", torch.argmax(logits, dim=-1).tolist())

    token_ids = torch.argmax(logits, dim=-1)  # (2,)
    print("Predicted token IDs:", token_ids.tolist())

    # 解码预测的 token IDs
    decoded_tokens = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids.tolist()]
    print("Predicted tokens for each sample:", decoded_tokens)
