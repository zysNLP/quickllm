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
    vocab_size: int
    seq_len: int
    d_model: int
    num_heads: int = None
    v_head_dim: int = None
    
    nope_head_dim: int = None
    rope_head_dim: int = None
    
    hidden_dim: int = None
    num_kv_heads: int = None
    num_layers: int = 4
    dropout: float = 0.0
    bias: bool = False
    weight_tying: bool = False
    activation: str = "silu"
    mlp: str = "GLU"
    kv_lora_rank: int = None
    q_lora_rank: int = None
    attn_type: str = "mla"

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

# --- MLA ---
class MultiHeadLatentAttention(nn.Module):
    """
    Multi Head Latent Attention 
    paper: https://arxiv.org/pdf/2405.04434
    
    TLDR: 
    kv are low ranks, this verient of attention project q,k,v to low rank to save memory,
    replace linear with lora(ish) layers

    source: https://github.com/joey00072/Multi-Head-Latent-Attention-MLA-
    """
    def __init__(self, config: Config):
        super().__init__()
        
        assert config.v_head_dim is not None , f"v_head_dim is not defined {config.v_head_dim=}"
        assert config.q_lora_rank is not None , f"q_lora_rank is not defined {config.q_lora_rank=}"
        assert config.kv_lora_rank is not None , f"kv_lora_rank is not defined {config.kv_lora_rank=}"
        assert config.rope_head_dim is not None , f"rope_head_dim is not defined {config.rope_head_dim=}"
        
        self.config = config
        
        self.dim = config.d_model
        self.num_heads = config.num_heads
        self.v_head_dim = config.v_head_dim
        
        self.nope_head_dim = config.nope_head_dim
        self.rope_head_dim = config.rope_head_dim
        
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        
        self.dropout = config.dropout
        
        # note: head dim of query and key if different from head dim of value
        
        # (attention_dim == num_head*head_dim) > d_model in deepseekv2
        # this is dim between wV and wQ
        self.value_dim = self.num_heads * self.v_head_dim
        
        # this is dims between wQ and wK
        self.nope_dim = self.num_heads * self.nope_head_dim
        self.rope_dim = self.num_heads * self.rope_head_dim  
        
        # query compression
        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=False)  # W_DQ
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_dim, bias=False)
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_dim, bias=False)
        self.q_norm = RMSNorm(dim=self.q_lora_rank)
        
        
        # key and value compression
        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=False)  # W_DKV
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_dim, bias=False)
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_dim, bias=False)
        self.kv_norm = RMSNorm(dim=self.kv_lora_rank)
        
        
        self.k_rope_linear = nn.Linear(self.dim, self.rope_head_dim  , bias=False)
        # self.rope_norm = RMSNorm(self.rope_dim) # not in deepseekv2

        self.proj = nn.Linear(self.value_dim , self.dim, bias=False)
        self.res_dropout = nn.Dropout(p=config.dropout)
        
        
    def forward(self, x: Tensor,mask: torch.Tensor, freqs_cis: Tensor):
        batch_size, seq_len, _ = x.shape

        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope:Tensor = self.decompress_q_nope(norm_q)
        query_rope:Tensor = self.decompress_q_rope(norm_q)

        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope: Tensor = self.decompress_k_nope(norm_kv)
        value: Tensor = self.decompress_v_linear(norm_kv)
        
        key_rope:Tensor = self.k_rope_linear(x)
        # norm_rope = self.rope_norm(key_rope)

        query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)
        
        key_rope = key_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)
        key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        
        value = value.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1,2)
        
        # *** the line that fixes MLA :) ***
        key_rope = key_rope/self.num_heads 

        q_rope,k_rope = apply_rope(query_rope,key_rope, cis=freqs_cis)
        
        q_recombined = torch.empty((batch_size,self.num_heads,seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device)
        k_recombined = torch.empty((batch_size, self.num_heads, seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device)
        
        q_recombined[:,:,:,:self.nope_head_dim] = query_nope
        q_recombined[:,:,:,self.nope_head_dim:] = q_rope
        
        # k_rope = torch.repeat_interleave(k_rope, self.num_heads, dim=1) # >> you dont need to do this <<
        # 👇 broadcasting will do replication krope to all heads automagically
        k_recombined[:,:,:,:self.nope_head_dim] = key_nope
        k_recombined[:,:,:,self.nope_head_dim:] = k_rope

        output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True, dropout_p=self.dropout)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)

        output = self.proj(output)
        output = self.res_dropout(output)
        return output


# --- MHA --- 
class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.num_kv_heads = config.num_heads if config.num_kv_heads == 0 else config.num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.key = nn.Linear(d_model, self.head_dim * self.num_kv_heads, config.bias)
        self.query = nn.Linear(d_model, self.head_dim * self.num_heads, config.bias)
        self.value = nn.Linear(d_model, self.head_dim * self.num_kv_heads, config.bias)
        self.proj = nn.Linear(d_model, d_model, config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )  # shape = (B, seq_len, num_kv_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # Grouped Query Attention
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order important
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)

        
        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


if __name__ == "__main__":
    
    d_model = 1024
    num_heads = 46
    
    v_head_dim = 32
    kv_lora_rank = 128
    q_lora_rank = 3 * kv_lora_rank
    
    rope_head_dim = 64
    nope_head_dim = 32
    
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
    )

    mla = MultiHeadLatentAttention(config)
    x = torch.randn(2, 10, d_model)
    freqs_cis = precompute_freqs_cis(config.rope_head_dim, config.seq_len)
    # mla = torch.compile(mla)
    print(f"Model Size: {sum(p.numel() for p in mla.parameters())/1e6}M params, attn size {d_model*d_model*4/1e6}m")
    output = mla(x,None, freqs_cis)
    print(output.shape)
    
