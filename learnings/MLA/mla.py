import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from collections import OrderedDict

from ohara.modules.norm import RMSNorm

from ohara.embedings_pos.rotatry import precompute_freqs_cis
from ohara.embedings_pos.rotatry import apply_rope

from torch import Tensor


from rich import print, traceback
traceback.install()


@dataclass
class Config(OrderedDict):
    """
    模型配置类，用于存储MLA（Multi-Head Latent Attention）模型的各种参数
    
    主要参数说明：
    - vocab_size: 词汇表大小
    - seq_len: 序列长度
    - d_model: 模型维度
    - num_heads: 注意力头数
    - v_head_dim: 值向量的头维度
    - nope_head_dim: 非位置编码的头维度
    - rope_head_dim: 旋转位置编码的头维度
    - hidden_dim: 隐藏层维度
    - num_kv_heads: 键值注意力头数
    - num_layers: 层数
    - dropout: dropout率
    - bias: 是否使用偏置
    - weight_tying: 是否使用权重绑定
    - activation: 激活函数类型
    - mlp: MLP类型
    - kv_lora_rank: 键值低秩分解的秩
    - q_lora_rank: 查询低秩分解的秩
    - attn_type: 注意力类型
    """
    vocab_size: int
    seq_len: int
    d_model: int
    num_heads: int = None
    v_head_dim: int = None
    
    nope_head_dim: int = None  # 非位置编码的头维度
    rope_head_dim: int = None  # 旋转位置编码的头维度
    
    hidden_dim: int = None
    num_kv_heads: int = None
    num_layers: int = 4
    dropout: float = 0.0
    bias: bool = False
    weight_tying: bool = False
    activation: str = "silu"
    mlp: str = "GLU"
    kv_lora_rank: int = None  # 键值低秩分解的秩
    q_lora_rank: int = None   # 查询低秩分解的秩
    attn_type: str = "mla"    # 注意力类型

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

# --- MLA (Multi-Head Latent Attention) ---
class MultiHeadLatentAttention(nn.Module):
    """
    多头潜在注意力机制 (Multi-Head Latent Attention)
    
    论文: https://arxiv.org/pdf/2405.04434
    
    核心思想：
    - 将查询(Q)、键(K)、值(V)投影到低秩空间以节省内存
    - 使用LoRA风格的低秩分解来替代全连接层
    - 结合旋转位置编码(RoPE)和非位置编码
    
    源码: https://github.com/joey00072/Multi-Head-Latent-Attention-MLA-
    """
    def __init__(self, config: Config):
        super().__init__()
        
        # 参数验证
        assert config.v_head_dim is not None , f"v_head_dim is not defined {config.v_head_dim=}"
        assert config.q_lora_rank is not None , f"q_lora_rank is not defined {config.q_lora_rank=}"
        assert config.kv_lora_rank is not None , f"kv_lora_rank is not defined {config.kv_lora_rank=}"
        assert config.rope_head_dim is not None , f"rope_head_dim is not defined {config.rope_head_dim=}"
        
        self.config = config
        
        # 基本维度设置
        self.dim = config.d_model                    # 模型维度
        self.num_heads = config.num_heads            # 注意力头数
        self.v_head_dim = config.v_head_dim          # 值向量的头维度
        
        self.nope_head_dim = config.nope_head_dim    # 非位置编码的头维度
        self.rope_head_dim = config.rope_head_dim    # 旋转位置编码的头维度
        
        self.q_lora_rank = config.q_lora_rank        # 查询低秩分解的秩
        self.kv_lora_rank = config.kv_lora_rank      # 键值低秩分解的秩
        
        self.dropout = config.dropout
        
        # 注意：查询和键的头维度可能与值的头维度不同
        
        # (attention_dim == num_head*head_dim) > d_model in deepseekv2
        # 这是wV和wQ之间的维度
        self.value_dim = self.num_heads * self.v_head_dim
        
        # 这是wQ和wK之间的维度
        self.nope_dim = self.num_heads * self.nope_head_dim      # 非位置编码维度
        self.rope_dim = self.num_heads * self.rope_head_dim      # 旋转位置编码维度
        
        # 查询压缩层
        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=False)  # W_DQ: 查询压缩
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_dim, bias=False)  # 查询非位置解码
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_dim, bias=False)  # 查询旋转位置解码
        self.q_norm = RMSNorm(dim=self.q_lora_rank)  # 查询归一化
        
        # 键和值压缩层
        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=False)  # W_DKV: 键值压缩
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_dim, bias=False)  # 键非位置解码
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_dim, bias=False)  # 值解码
        self.kv_norm = RMSNorm(dim=self.kv_lora_rank)  # 键值归一化
        
        # 旋转位置编码的键线性层
        self.k_rope_linear = nn.Linear(self.dim, self.rope_head_dim, bias=False)
        # self.rope_norm = RMSNorm(self.rope_dim) # deepseekv2中没有使用

        # 输出投影层
        self.proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.res_dropout = nn.Dropout(p=config.dropout)
        
        
    def forward(self, x: Tensor, mask: torch.Tensor, freqs_cis: Tensor):
        """
        前向传播函数
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 注意力掩码
            freqs_cis: 预计算的旋转位置编码频率，用于RoPE
            
        Returns:
            输出张量，形状与输入相同
        """
        batch_size, seq_len, _ = x.shape

        # 查询压缩和解压缩
        compressed_q = self.compress_q_linear(x)  # 压缩查询到低秩空间
        norm_q = self.q_norm(compressed_q)       # 归一化
        query_nope:Tensor = self.decompress_q_nope(norm_q)  # 解压缩为非位置查询
        query_rope:Tensor = self.decompress_q_rope(norm_q)  # 解压缩为旋转位置查询

        # 键值压缩和解压缩
        compressed_kv = self.compress_kv_linear(x)  # 压缩键值到低秩空间
        norm_kv = self.kv_norm(compressed_kv)      # 归一化
        key_nope: Tensor = self.decompress_k_nope(norm_kv)  # 解压缩为非位置键
        value: Tensor = self.decompress_v_linear(norm_kv)   # 解压缩为值
        
        # 旋转位置编码的键（直接从输入计算）
        key_rope:Tensor = self.k_rope_linear(x)
        # norm_rope = self.rope_norm(key_rope)

        # 重塑张量维度以准备注意力计算
        query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)
        
        key_rope = key_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)
        key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        
        value = value.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1,2)
        
        # *** 修复MLA的关键行 :) ***
        # 将key_rope除以头数以进行缩放
        key_rope = key_rope/self.num_heads 

        # 应用旋转位置编码(RoPE)
        # freqs_cis: 这是旋转位置编码(RoPE)的核心组件
        # 它包含了位置信息的复数表示，使得模型能够理解序列中token的相对位置
        # 这些频率是预计算的，避免在每次前向传播时重复计算
        q_rope,k_rope = apply_rope(query_rope,key_rope, cis=freqs_cis)
        
        # 重新组合查询和键（将旋转位置编码和非位置编码拼接）
        q_recombined = torch.empty((batch_size,self.num_heads,seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device)
        k_recombined = torch.empty((batch_size, self.num_heads, seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device)
        
        # 填充非位置部分
        q_recombined[:,:,:,:self.nope_head_dim] = query_nope
        q_recombined[:,:,:,self.nope_head_dim:] = q_rope
        
        # 注释：不需要手动复制k_rope到所有头，广播会自动处理
        # k_rope = torch.repeat_interleave(k_rope, self.num_heads, dim=1) # >> 你不需要这样做 <<
        # 👇 广播会自动将k_rope复制到所有头
        k_recombined[:,:,:,:self.nope_head_dim] = key_nope
        k_recombined[:,:,:,self.nope_head_dim:] = k_rope

        # 计算注意力（使用缩放点积注意力）
        output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True, dropout_p=self.dropout)

        # 重塑输出维度
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)

        # 最终投影和dropout
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


# --- MHA (Multi-Head Attention) --- 
class Attention(nn.Module):
    """
    标准的多头注意力机制
    
    这是传统的多头注意力实现，用于对比MLA的效果
    """
    def __init__(self, config: Config):
        super().__init__()

        d_model = config.d_model
        self.num_heads = config.num_heads
        # 使用自定义的head_dim而不是计算出来的
        self.head_dim = getattr(config, 'mha_head_dim', config.d_model // config.num_heads)
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # 线性变换层
        self.key = nn.Linear(d_model, self.head_dim * self.num_kv_heads, config.bias)
        self.query = nn.Linear(d_model, self.head_dim * self.num_heads, config.bias)
        self.value = nn.Linear(d_model, self.head_dim * self.num_kv_heads, config.bias)
        # 输出投影层需要匹配实际的头维度
        self.proj = nn.Linear(self.head_dim * self.num_heads, d_model, config.bias)

        # Dropout层
        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        # 检查是否支持Flash Attention
        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # self.flash_attn = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            x: 输入张量
            mask: 注意力掩码
            freqs_cis: 旋转位置编码频率
            
        Returns:
            注意力输出
        """
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # 类型提示
        q: torch.Tensor  # 忽略
        v: torch.Tensor

        # 计算查询、键、值
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # 重塑维度
        k = k.view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )  # 形状 = (B, seq_len, num_kv_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 应用旋转位置编码
        q, k = apply_rope(q, k, freqs_cis)

        # 分组查询注意力 (Grouped Query Attention)
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        # 转置维度以准备注意力计算
        k = k.transpose(1, 2)  # 形状 = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力（使用Flash Attention或标准实现）
        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # 顺序很重要
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # 标准注意力计算
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            # 处理mask为None的情况
            if mask is not None:
                attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)

        
        # 恢复时间维度作为批次维度并连接头
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.head_dim * self.num_heads)

        # 最终投影到残差流
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


if __name__ == "__main__":
    """
    主函数：测试MLA模型的实现
    
    这个测试展示了如何：
    1. 配置MLA模型参数
    2. 初始化模型
    3. 生成测试数据
    4. 运行前向传播
    """
    
    # 模型参数设置
    d_model = 1024
    num_heads = 64
    
    # 调整MHA的头维度以匹配MLA
    # MLA的总头维度 = rope_head_dim + nope_head_dim = 64 + 32 = 96
    # 所以MHA的head_dim应该设置为96
    mha_head_dim = 96  # 与MLA的总头维度匹配
    
    # 恢复MLA的原始参数
    v_head_dim = 32
    kv_lora_rank = 128
    q_lora_rank = 3 * kv_lora_rank  # 查询秩是键值秩的3倍
    
    rope_head_dim = 64
    nope_head_dim = 32
    
    # 创建配置对象
    config = Config(
        vocab_size=30522,
        d_model=d_model,
        seq_len=2048,
        num_heads=num_heads,
        v_head_dim=v_head_dim,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        mha_head_dim=mha_head_dim,  # 添加MHA的头维度配置
    )

    # 初始化MLA模型
    mla = MultiHeadLatentAttention(config)
    mha = Attention(config)
    x = torch.randn(2, 10, d_model)  # 创建测试输入
    
    # 预计算旋转位置编码频率
    # freqs_cis: 这是旋转位置编码(RoPE)的核心组件
    # 它包含了位置信息的复数表示，使得模型能够理解序列中token的相对位置
    # 这些频率是预计算的，避免在每次前向传播时重复计算
    
    # 为MLA模型预计算（使用rope_head_dim）
    freqs_cis_mla = precompute_freqs_cis(config.rope_head_dim, config.seq_len)
    
    # 为MHA模型预计算（使用自定义的head_dim）
    mha_head_dim = getattr(config, 'mha_head_dim', config.d_model // config.num_heads)
    freqs_cis_mha = precompute_freqs_cis(mha_head_dim, config.seq_len)
    
    # mla = torch.compile(mla)  # 可选：使用torch.compile优化
    
    # 打印模型信息
    print(f"Model MLA Size: {sum(p.numel() for p in mla.parameters())/1e6}M params, attn size {d_model*d_model*4/1e6}m")
    print(f"Model MHA Size: {sum(p.numel() for p in mha.parameters())/1e6}M params, attn size {d_model*d_model*4/1e6}m")

    # 运行前向传播
    output_mla = mla(x, None, freqs_cis_mla)
    output_mha = mha(x, None, freqs_cis_mha)
    print(output_mla.shape)
    print(output_mha.shape)
    
