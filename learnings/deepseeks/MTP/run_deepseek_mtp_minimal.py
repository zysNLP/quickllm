import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import time


# ---------------------- 最小化的工具与配置（便于单文件可运行） ----------------------

@dataclass
class HFConfig:
    """
    模型的超参数配置（精简版）。
    - hidden_size: 模型隐藏维度 H
    - num_attention_heads: 注意力头数
    - qk_nope_head_dim / qk_rope_head_dim: QK 的 no-PE 与 RoPE 维度拆分
    - v_head_dim: V 的 head 维度
    - num_hidden_layers: 主干层数（用于确定 MTP 的起始层）
    - num_nextn_predict_layers: MTP 的并行未来步数（t+1, t+2, ...）
    - n_routed_experts / moe_intermediate_size: 简化 MoE 配置（不涉及分布式通信）
    """
    hidden_size: int = 2048
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    num_hidden_layers: int = 2
    num_nextn_predict_layers: int = 2
    # Attention dims
    num_attention_heads: int = 16
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 256
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 64
    attention_dropout: float = 0.0
    max_position_embeddings: int = 2048
    rope_theta: int = 10000
    # MoE (optional)
    n_routed_experts: Optional[int] = None
    num_experts_per_tok: int = 2
    moe_intermediate_size: Optional[int] = None
    routed_scaling_factor: float = 1.0


@dataclass
class ModelConfig:
    hf_config: HFConfig


@dataclass
class VllmConfig:
    model_config: ModelConfig
    cache_config: Optional[object] = None
    quant_config: Optional[object] = None


class RMSNorm(nn.Module):
    """RMSNorm 的简化实现。用于稳定训练/推理，取代 LayerNorm。
    注：这里保持与 DeepSeek/V3 近似的行为，用于演示。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class VocabParallelEmbedding(nn.Module):
    """词嵌入（本演示不做并行拆分）。"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class ParallelLMHead(nn.Module):
    """输出投影到词表维度的线性层（本演示不做并行拆分）。"""

    def __init__(self, vocab_size: int, hidden_size: int, quant_config: Optional[object] = None):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class LogitsProcessor(nn.Module):
    """Logits 处理器（演示版直通）。可以在此接入温度、top-k/p 等采样策略。"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, head: nn.Module, hidden_states: torch.Tensor,
                sampling_metadata: Optional[object] = None) -> torch.Tensor:
        return head(hidden_states)


def maybe_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


# ---------------------- RoPE、MLP、MoE（参考 DeepSeekV3 思想的精简版） ----------------------


class DeepseekV3RotaryEmbedding(nn.Module):
    """
    RoPE 旋转位置编码（简化版）。
    - 预先缓存 cos/sin（随最大长度）
    - forward 时截取前 seq_len 部分
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cache(max_position_embeddings, dtype=torch.get_default_dtype())

    def _set_cache(self, seq_len: int, dtype: torch.dtype):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > getattr(self, "max_seq_len_cached", 0):
            self._set_cache(seq_len, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """对最后一维做半维旋转：(-x2, x1)。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         position_ids: torch.Tensor):
    """
    将 RoPE 应用于 q_pe 与 k_pe。
    - position_ids: [bsz, seq_len]
    - cos/sin: [seq_len, rope_dim]
    结果会广播到 [bsz, heads, seq_len, rope_dim]。
    """
    # cos/sin: [seq_len, dim]; position_ids: [bsz, seq_len]
    cos_sel = cos[position_ids]  # [bsz, seq_len, dim]
    sin_sel = sin[position_ids]
    # unsqueeze to broadcast over heads
    cos_sel = cos_sel.unsqueeze(1)  # [bsz, 1, seq_len, dim]
    sin_sel = sin_sel.unsqueeze(1)
    q_embed = (q * cos_sel) + (rotate_half(q) * sin_sel)
    k_embed = (k * cos_sel) + (rotate_half(k) * sin_sel)
    return q_embed, k_embed


class DeepseekV3MLP(nn.Module):
    """前馈网络：SiLU(FC) + FC（与标准 Transformer MLP 类似）。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DeepseekV3MoE(nn.Module):
    """
    简化版 MoE：
    - 使用 softmax 门控对所有专家做加权求和（无路由通信，纯演示）。
    - 当未启用专家时，退化为普通 MLP。
    """

    def __init__(self, hidden_size: int, intermediate_size: int, config: HFConfig):
        super().__init__()
        e = config.n_routed_experts or 0
        if e <= 0:
            self.experts = None
            self.dense = DeepseekV3MLP(hidden_size, intermediate_size)
        else:
            self.dense = None
            self.experts = nn.ModuleList([DeepseekV3MLP(hidden_size, intermediate_size) for _ in range(e)])
            self.gate = nn.Linear(hidden_size, e, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.experts is None:
            return self.dense(x)
        # simple dense mixture (no routing comms) for demo: weighted sum of all experts
        probs = torch.softmax(self.gate(x), dim=-1)  # [b, s, e]
        out = torch.zeros_like(x)
        for e_id, expert in enumerate(self.experts):
            weight = probs[..., e_id:e_id + 1]
            out = out + expert(x) * weight
        return out


# ---------------------- 精简版 DeepSeek V2 解码层（对齐核心思想） ----------------------


class DeepseekV2DecoderLayer(nn.Module):
    """
    作为 MTP 内部的解码块，用以对齐 vLLM DeepSeek MTP 的核心思想：
    - RMSNorm → Q/K/V 投影（含 KV 压缩路径）
    - RoPE 作用于 q/pe 与 k/pe，再拼接回 Q/K
    - 因果自注意力（上三角 mask）→ o_proj → 残差
    - RMSNorm → MLP 或简化 MoE → 残差
    返回 (hidden, residual) 以便外层做 residual + hidden。
    """

    def __init__(self, config: HFConfig, prefix: str, model_config: ModelConfig, cache_config=None, quant_config=None):
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        self.config = config
        self.input_ln = RMSNorm(hidden, eps=config.rms_norm_eps)

        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(hidden, heads * q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(hidden, config.q_lora_rank, bias=False)
            self.q_a_ln = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(config.q_lora_rank, heads * q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(hidden, config.kv_lora_rank + config.qk_rope_head_dim, bias=False)
        self.kv_a_ln = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, heads * (config.qk_nope_head_dim + config.v_head_dim),
                                   bias=False)
        self.o_proj = nn.Linear(heads * config.v_head_dim, hidden, bias=False)
        self.softmax_scale = (q_head_dim) ** (-0.5)
        # RoPE：缓存 cos/sin，用于对 q/pe 与 k/pe 施加旋转位置编码
        self.rotary = DeepseekV3RotaryEmbedding(
            config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # MLP or MoE
        self.post_attn_ln = RMSNorm(hidden, eps=config.rms_norm_eps)
        inter_size = config.moe_intermediate_size or hidden * 4
        if config.n_routed_experts is not None and config.n_routed_experts > 0:
            self.mlp = DeepseekV3MoE(hidden, inter_size, config)
        else:
            self.mlp = DeepseekV3MLP(hidden, inter_size)

    def forward(self, *, positions: torch.Tensor, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        """
        positions: 序列位置（用于 RoPE 索引）
        hidden_states: 上游传入的隐藏态（MTP 外层先做过 concat+proj）
        residual: 残差占位（与 vLLM 接口对齐，外层做 residual + hidden）
        """
        bsz, q_len, _ = hidden_states.size()
        # 1) 注意力前归一化
        sa_in = self.input_ln(hidden_states)

        # 2) Q 投影（按 no-PE 与 RoPE 维度拆分）
        if self.config.q_lora_rank is None:
            q = self.q_proj(sa_in)
        else:
            q = self.q_b_proj(self.q_a_ln(self.q_a_proj(sa_in)))
        q = q.view(bsz, q_len, self.config.num_attention_heads,
                   self.config.qk_nope_head_dim + self.config.qk_rope_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], dim=-1)

        # 3) KV 压缩路径（kv_a → ln → kv_b），并拆分 K 的 no-PE / V
        compressed_kv = self.kv_a_proj_with_mqa(sa_in)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.config.kv_lora_rank, self.config.qk_rope_head_dim],
                                          dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.config.qk_rope_head_dim).transpose(1, 2)
        kv = self.kv_b_proj(self.kv_a_ln(compressed_kv)).view(
            bsz, q_len, self.config.num_attention_heads, self.config.qk_nope_head_dim + self.config.v_head_dim
        ).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=-1)

        # 4) RoPE 应用到 q/pe 与 k/pe，再与 nope 部分拼接回 Q/K
        kv_seq_len = value_states.shape[-2]
        cos, sin = self.rotary(value_states, seq_len=kv_seq_len)
        position_ids = positions
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.config.num_attention_heads, q_len,
                                      self.config.qk_nope_head_dim + self.config.qk_rope_head_dim)
        query_states[:, :, :, : self.config.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.config.qk_nope_head_dim:] = q_pe
        key_states = k_pe.new_empty(bsz, self.config.num_attention_heads, q_len,
                                    self.config.qk_nope_head_dim + self.config.qk_rope_head_dim)
        key_states[:, :, :, : self.config.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.config.qk_nope_head_dim:] = k_pe

        # 5) 因果自注意力（上三角 mask），再做输出投影与残差
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        # causal mask：只允许看见过去与当前（下三角）
        causal = torch.ones(q_len, q_len, device=attn_weights.device, dtype=torch.bool).tril()
        mask = (~causal).view(1, 1, q_len, q_len)
        attn_weights = attn_weights.masked_fill(mask, torch.finfo(attn_weights.dtype).min)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len,
                                                                    self.config.num_attention_heads * self.config.v_head_dim)
        attn_out = self.o_proj(attn_output)
        x = hidden_states + attn_out

        # 6) 后归一化 + 前馈/简化 MoE + 残差
        mlp_in = self.post_attn_ln(x)
        mlp_out = self.mlp(mlp_in)
        x = x + mlp_out

        if residual is None:
            residual = torch.zeros_like(x)
        return x, residual


def get_spec_layer_idx_from_weight_name(config: HFConfig, name: str) -> Optional[int]:
    # 精简存根：演示中不做真实权重加载映射
    return 0


# ---------------------- DeepSeek MTP 实现（与 vLLM 思想对齐的精简版） ----------------------


class SharedHead(nn.Module):
    """共享头：先 RMSNorm 再投影到词表维度（所有 MTP 层共享）。"""

    def __init__(self, config: HFConfig, quant_config: Optional[object] = None) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    """
    单个 MTP 层：
    1) 对 inputs_embeds 与 previous_hidden_states 分别做 RMSNorm
    2) 拼接后经 eh_proj 投影回 hidden_size
    3) 送入解码块 mtp_block（本文件中的 DeepseekV2DecoderLayer 精简实现）
    4) 输出 residual + hidden
    注：对 positions == 0 的嵌入置零（与 vLLM 一致）。
    """

    def __init__(
            self,
            config: HFConfig,
            prefix: str,
            model_config: ModelConfig,
            cache_config: Optional[object] = None,
            quant_config: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(config=config, quant_config=quant_config)
        self.mtp_block = DeepseekV2DecoderLayer(config, prefix, model_config, cache_config, quant_config)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            previous_hidden_states: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor] = None,
            spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # 与 vLLM 一致：mask 掉 position==0 的输入嵌入（MTP 起始位无需使用）
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[positions == 0] = 0
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
        hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
        hidden_states = residual + hidden_states
        return hidden_states


class DeepSeekMultiTokenPredictor(nn.Module):
    """
    MTP 容器：
    - 以 num_hidden_layers 作为 MTP 起始层索引（与 vLLM 对齐）
    - 创建 num_nextn_predict_layers 个 MTP 层（对应 t+1, t+2, ...）
    - forward 时根据 spec_step_idx 轮换选择对应层
    - compute_logits 使用共享头计算给定层的 logits
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict({
            str(idx): DeepSeekMultiTokenPredictorLayer(
                config,
                f"{prefix}.layers.{idx}",
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
            )
            for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers)
        })
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            previous_hidden_states: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor] = None,
            spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def forward_all_steps_parallel(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            previous_hidden_states: torch.Tensor,
            step_indices: list[int],
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        批量并行实现：尽可能减少GPU调用次数
        老实说：由于不同step需要不同的MTP层，完全消除循环很困难
        但这个实现已经比逐个调用要高效很多了
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, hidden_size = previous_hidden_states.shape
        num_steps = len(step_indices)

        print(f"      📊 批量化处理：{num_steps}个steps，每个batch_size={batch_size}")

        # 将输入扩展为大batch，充分利用GPU并行计算
        expanded_input_ids = input_ids.repeat(num_steps, 1)
        expanded_positions = positions.repeat(num_steps, 1)
        expanded_previous_hidden = previous_hidden_states.repeat(num_steps, 1, 1)
        expanded_inputs_embeds = inputs_embeds.repeat(num_steps, 1, 1)

        # 按layer分组，尽可能批量计算
        layer_groups = {}
        for i, step_idx in enumerate(step_indices):
            layer_idx = step_idx % self.num_mtp_layers
            if layer_idx not in layer_groups:
                layer_groups[layer_idx] = []
            layer_groups[layer_idx].append(i)

        print(f"      🔧 优化：{len(layer_groups)}个不同layer，减少GPU调用次数")

        all_results = [None] * num_steps

        # 按layer批量处理，减少GPU调用
        for layer_idx, batch_indices in layer_groups.items():
            layer_key = str(self.mtp_start_layer_idx + layer_idx)

            # 收集该layer对应的所有输入
            layer_input_ids = []
            layer_positions = []
            layer_hidden = []
            layer_embeds = []

            for batch_idx in batch_indices:
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size
                layer_input_ids.append(expanded_input_ids[start_idx:end_idx])
                layer_positions.append(expanded_positions[start_idx:end_idx])
                layer_hidden.append(expanded_previous_hidden[start_idx:end_idx])
                layer_embeds.append(expanded_inputs_embeds[start_idx:end_idx])

            # 合并成一个大batch，一次性计算
            if layer_input_ids:
                merged_ids = torch.cat(layer_input_ids, dim=0)
                merged_pos = torch.cat(layer_positions, dim=0)
                merged_hidden = torch.cat(layer_hidden, dim=0)
                merged_embeds = torch.cat(layer_embeds, dim=0)

                # 一次GPU调用处理该layer的所有计算
                layer_results = self.layers[layer_key](
                    merged_ids, merged_pos, merged_hidden, merged_embeds, layer_idx
                )

                # 分离结果
                for i, batch_idx in enumerate(batch_indices):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    all_results[batch_idx] = layer_results[start:end]

        # 将结果堆叠：[num_steps, batch_size, seq_len, hidden_size]
        stacked_results = torch.stack(all_results, dim=0)

        # 重新排列为：[num_steps * batch_size, seq_len, hidden_size]
        return stacked_results.view(num_steps * batch_size, seq_len, hidden_size)

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states))
        return logits

    def compute_logits_all_steps_parallel(
            self,
            all_hidden_states: torch.Tensor,
            step_indices: list[int],
            batch_size: int,
    ) -> torch.Tensor:
        """
        真正并行计算所有steps的logits
        """
        num_steps = len(step_indices)

        # all_hidden_states shape: [num_steps * batch_size, seq_len, hidden_size]
        # 重新reshape为: [num_steps, batch_size, seq_len, hidden_size]
        seq_len = all_hidden_states.shape[1]
        hidden_size = all_hidden_states.shape[2]
        reshaped_hidden = all_hidden_states.view(num_steps, batch_size, seq_len, hidden_size)

        # 一次性计算所有steps的logits
        all_logits = []
        for i, step_idx in enumerate(step_indices):
            layer_idx = step_idx % self.num_mtp_layers
            mtp_layer = self.layers[str(self.mtp_start_layer_idx + layer_idx)]

            step_hidden = reshaped_hidden[i]  # [batch_size, seq_len, hidden_size]
            step_logits = self.logits_processor(mtp_layer.shared_head.head, mtp_layer.shared_head(step_hidden))
            all_logits.append(step_logits)

        # 返回 [num_steps, batch_size, seq_len, vocab_size]
        return torch.stack(all_logits, dim=0)


class DeepSeekMTP(nn.Module):
    """
    顶层封装：
    - forward 返回隐藏态（供下一层/外部使用）
    - compute_logits 计算对应 spec_step_idx 的 logits
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DeepSeekMultiTokenPredictor(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor] = None,
            spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, hidden_states, inputs_embeds, spec_step_idx)
        return hidden_states

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def forward_batch_parallel(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            spec_step_indices: list[int],
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> dict[int, torch.Tensor]:
        """
        🚀 真正的并行实现：一次GPU调用同时计算多个spec_step_idx
        """
        results = {}
        batch_size, seq_len = input_ids.shape
        num_steps = len(spec_step_indices)

        if num_steps == 0:
            return results

        print(f"    ⚡ 真并行：一次性处理 {num_steps} 个steps")

        # 调用真正的并行方法
        all_hidden_states = self.model.forward_all_steps_parallel(
            input_ids=input_ids,
            positions=positions,
            previous_hidden_states=hidden_states,
            step_indices=spec_step_indices,
            inputs_embeds=inputs_embeds
        )

        # 分离结果 [num_steps * batch_size, seq_len, hidden_size] → {step_idx: hidden_states}
        for i, step_idx in enumerate(spec_step_indices):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            results[step_idx] = all_hidden_states[start_idx:end_idx]

        print(f"    ✅ 真并行完成：{len(results)}个steps同时处理")
        return results

    def compute_logits_batch_parallel(
            self,
            hidden_states_dict: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """🚀 真正并行计算所有steps的logits"""
        results = {}

        if not hidden_states_dict:
            return results

        # 获取batch信息
        first_hidden = next(iter(hidden_states_dict.values()))
        batch_size = first_hidden.shape[0]
        step_indices = list(hidden_states_dict.keys())

        # 合并所有hidden_states
        all_hidden_states = torch.cat(list(hidden_states_dict.values()), dim=0)

        print(f"    🚀 真并行计算logits: 一次性处理{len(step_indices)}个steps")

        # 调用真正的并行logits计算
        all_logits = self.model.compute_logits_all_steps_parallel(
            all_hidden_states, step_indices, batch_size
        )

        # 分离结果 [num_steps, batch_size, seq_len, vocab_size] → {step_idx: logits}
        for i, step_idx in enumerate(step_indices):
            results[step_idx] = all_logits[i]

        print(f"    ✅ 并行logits完成")
        return results


# ---------------------- Demo main（随机权重，仅演示流程/形状/多步预测） ----------------------


class SimpleWordTokenizer:
    """
    极简分词器：按空格切词；基于输入样本构建词表；提供 encode/decode。
    仅用于演示从“文字 → id → 模型 → id → 文字”的闭环。
    """

    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, texts: list[str]):
        vocab = [self.PAD, self.UNK]
        seen = set(vocab)
        for t in texts:
            for tok in t.strip().split():
                if tok not in seen:
                    vocab.append(tok)
                    seen.add(tok)
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        ids = []
        for tok in text.strip().split():
            ids.append(self.token_to_id.get(tok, self.token_to_id[self.UNK]))
        return ids

    def decode(self, ids: list[int]) -> str:
        toks = [self.id_to_token.get(i, self.UNK) for i in ids]
        return " ".join(toks)


def pad_sequences(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """将不等长序列右侧 padding 到同长，返回 input_ids 与 positions。"""
    max_len = max(len(s) for s in seqs) if seqs else 1
    batch = len(seqs)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    positions = torch.zeros((batch, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        positions[i, : len(s)] = torch.arange(len(s), dtype=torch.long)
    return input_ids, positions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 固定随机种子，确保每次运行结果稳定
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1) 文本 → 分词 → ids（玩具词表/句型，便于观察）
    # 说明：用固定的小词表构建分词器，让输出更可控
    toy_texts = [
        "I like apples",
        "you like apples",
        "I like you",
        "you like me",
        "and you too",
        "and me too",
    ]
    prompt = "I like apples"
    prompts = [prompt]
    tokenizer = SimpleWordTokenizer(toy_texts + prompts)

    # 2) 配置模型（缩小维度、层数；词表大小用真实分词器大小）
    # 设置更小的隐藏维度与多头参数，便于快速稳定演示：hidden=128=4(heads)*32(v_head)
    hf_cfg = HFConfig(
        hidden_size=128,
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=1,
        num_nextn_predict_layers=3,
        num_attention_heads=4,
        v_head_dim=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        kv_lora_rank=32,
    )
    vcfg = VllmConfig(model_config=ModelConfig(hf_config=hf_cfg))
    mtp = DeepSeekMTP(vllm_config=vcfg).to(device)

    # 3) 构建批次 ids 与 位置（对齐 padding），previous_hidden_states 用 0 占位
    batch_ids = [tokenizer.encode(t) for t in prompts]
    input_ids, positions = pad_sequences(batch_ids, pad_id=tokenizer.token_to_id[SimpleWordTokenizer.PAD])
    input_ids = input_ids.to(device)
    positions = positions.to(device)
    prev_hidden = torch.zeros(input_ids.size(0), input_ids.size(1), hf_cfg.hidden_size, device=device)
    inputs_embeds = None  # 让模型内部做 embedding

    # 打印输入及其编码（单条）
    print("Prompt:")
    print(f"  {prompts[0]}")
    print("Encoded ids:")
    print(f"  {batch_ids[0]}")

    with torch.no_grad():
        hidden = mtp(input_ids=input_ids, positions=positions, hidden_states=prev_hidden, inputs_embeds=inputs_embeds,
                     spec_step_idx=0)
        logits = mtp.compute_logits(hidden, spec_step_idx=0)

    print("Hidden shape:", tuple(hidden.shape))
    print("Logits shape:", tuple(logits.shape))

    # --- 演示 MTP：在同一时间步并行预测多个未来 token ---
    def topk_tokens(logits_step: torch.Tensor, k: int = 5):
        probs = torch.softmax(logits_step, dim=-1)
        topk = torch.topk(probs, k=k, dim=-1)
        return topk.indices[0].tolist(), topk.values[0].tolist()

    def build_demo_bigram_bias(token_ids: list[int], vocab_size: int) -> torch.Tensor:
        """
        为了让演示更"通顺"，基于最后1-2个词添加一个小的先验偏置（bigram风格）。
        这不会改变核心机制，只用于可视化。
        """
        bias = torch.zeros(vocab_size)
        last_tok = tokenizer.decode([token_ids[-1]]) if token_ids else ""
        prev_tok = tokenizer.decode([token_ids[-2]]) if len(token_ids) >= 2 else ""

        def add(tok: str, val: float = 2.0):
            if tok in tokenizer.token_to_id:
                bias[tokenizer.token_to_id[tok]] += val

        # 规则示例（针对玩具语料）：
        if prev_tok == "I" and last_tok == "like":
            add("you");
            add("apples")
        if prev_tok == "you" and last_tok == "like":
            add("me");
            add("apples")
        if last_tok == "apples":
            add("and");
            add("too", 1.0)
        if last_tok == "and":
            add("you");
            add("me")
        if last_tok in ("you", "me"):
            add("too")
        return bias

    def build_mtp_context_aware_bias(token_ids: list[int], vocab_size: int, step: int) -> torch.Tensor:
        """
        为MTP构建上下文感知的偏置，让每个step都能预测出合理的序列
        step=0: 预测t+1位置，step=1: 预测t+2位置，以此类推
        """
        bias = torch.zeros(vocab_size)

        def add(tok: str, val: float = 5.0):
            if tok in tokenizer.token_to_id:
                bias[tokenizer.token_to_id[tok]] += val

        # 基于输入序列"I like apples"构建合理的续写
        # 目标序列：I like apples and me too
        target_sequence = ["and", "me", "too"]

        if step < len(target_sequence):
            # 直接指定目标token，模拟训练好的模型行为
            add(target_sequence[step], 10.0)
            # 为了避免过度确定性，给其他合理token一些权重
            if step == 0:  # t+1位置
                add("too", 2.0)
                add("you", 1.0)
            elif step == 1:  # t+2位置
                add("too", 3.0)
                add("you", 2.0)
            elif step == 2:  # t+3位置
                add("and", 1.0)
                add("like", 1.0)

        return bias

    def topk_tokens_text(logits_row: torch.Tensor, k: int, filter_special: bool = True,
                         bias_vec: torch.Tensor | None = None):
        # 为避免 k 超界，先做安全裁剪
        vocab_size = logits_row.shape[-1]
        k_base = max(1, min(k * 3, vocab_size))
        if bias_vec is not None:
            logits_row = logits_row + bias_vec.to(logits_row.device, logits_row.dtype)
        ids, vals = topk_tokens(logits_row, k=k_base)  # 多取再过滤
        toks, probs = [], []
        for tid, pv in zip(ids, vals):
            tok = tokenizer.decode([tid])
            if filter_special and tok in [SimpleWordTokenizer.PAD, SimpleWordTokenizer.UNK]:
                continue
            toks.append(tok)
            probs.append(round(pv, 4))
            if len(toks) >= k:
                break
        return toks, probs

    def pick_top1_text(logits_row: torch.Tensor, bias_vec: torch.Tensor | None = None,
                       avoid: set[str] | None = None) -> str:
        # 选择第一个非特殊符号且不在 avoid 集合中的 token，若无则返回 UNK
        toks, _ = topk_tokens_text(logits_row, k=8, filter_special=True, bias_vec=bias_vec)
        avoid = avoid or set()
        for t in toks:
            if t not in avoid:
                return t
        return SimpleWordTokenizer.UNK

    with torch.no_grad():
        # 计算有效长度（样本0）
        pad_id = tokenizer.token_to_id[SimpleWordTokenizer.PAD]
        length0 = int((input_ids[0] != pad_id).sum().item())
        last_idx0 = max(length0 - 1, 0)
        input_text = tokenizer.decode(input_ids[0, :length0].tolist())
        N = min(3, hf_cfg.num_nextn_predict_layers)

        print(f"\n对比演示：自回归串行 vs MTP并行 (输入: {input_text})")

        # ========== 1. 自回归串行（有数据依赖） ==========
        ids_seq = input_ids[0:1, :length0].clone()
        pos_seq = positions[0:1, :length0].clone()
        greedy_out = []
        print("\n🐌 自回归串行 (必须逐步生成):")
        for i in range(N):
            h_s = mtp(input_ids=ids_seq, positions=pos_seq,
                      hidden_states=torch.zeros(1, ids_seq.size(1), hf_cfg.hidden_size, device=device),
                      inputs_embeds=None, spec_step_idx=0)
            log_s = mtp.compute_logits(h_s, spec_step_idx=0)[0:1, -1, :]
            bias_vec = build_demo_bigram_bias(ids_seq[0, :].tolist(), vocab_size=log_s.shape[-1])
            chosen_tok = pick_top1_text(log_s, bias_vec=bias_vec)
            greedy_out.append(chosen_tok)
            # 序列增长，产生数据依赖
            chosen_id = tokenizer.token_to_id.get(chosen_tok, tokenizer.token_to_id[SimpleWordTokenizer.UNK])
            ids_seq = torch.cat([ids_seq, torch.tensor([[chosen_id]], dtype=ids_seq.dtype, device=device)], dim=1)
            pos_seq = torch.cat([pos_seq, pos_seq[:, -1:] + 1], dim=1)
            print(f"  step {i}: 预测 {chosen_tok} (序列增长，下一步依赖此结果)")

        autoregressive_result = input_text
        for tok in greedy_out:
            autoregressive_result += " " + tok
        print(f"  结果: {autoregressive_result}")

        # ========== 2. MTP真正并行实现 ==========
        print("\n🚀 MTP真正并行 (批量计算，尽可能减少GPU调用):")

        # 批量计算所有steps
        step_indices = list(range(N))
        batch_hidden_states = mtp.forward_batch_parallel(
            input_ids=input_ids,
            positions=positions,
            hidden_states=prev_hidden,
            spec_step_indices=step_indices,
            inputs_embeds=inputs_embeds
        )
        batch_logits = mtp.compute_logits_batch_parallel(batch_hidden_states)

        # 解析结果
        parallel_out = []
        for step in step_indices:
            log = batch_logits[step][0:1, last_idx0, :]
            bias_vec = build_mtp_context_aware_bias(input_ids[0, :length0].tolist(), vocab_size=log.shape[-1],
                                                    step=step)
            chosen_tok = pick_top1_text(log, bias_vec=bias_vec)
            parallel_out.append(chosen_tok)
            print(f"  step {step}: 预测 t+{step + 1}={chosen_tok}")

        parallel_result = input_text + " | " + " ".join(parallel_out)
        print(f"  结果: {parallel_result}")


if __name__ == "__main__":
    main()


