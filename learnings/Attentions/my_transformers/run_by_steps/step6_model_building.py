# -*- coding: utf-8 -*-
"""
Step 6: 构建Transformer模型
构建完整的Transformer模型结构，包括Encoder、Decoder、MultiHeadAttention等组件
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from loguru import logger

# 检测可用设备（优先GPU，其次MPS，最后CPU）
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🎯 使用GPU加速")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 使用MPS加速（Mac）")
else:
    device = torch.device("cpu")
    print("⚙️ 使用CPU")

print(f"使用设备: {device}")


def get_position_embedding(sentence_length: int, d_model: int, device=None, dtype=torch.float32):
    """生成位置编码"""
    def get_angles(pos: torch.Tensor, i: torch.Tensor, d_model: int):
        angle_rates = 1.0 / torch.pow(
            10000,
            (2 * torch.div(i, 2, rounding_mode='floor')).float() / d_model
        )
        return pos.float() * angle_rates

    if device is None:
        device = torch.device("cpu")

    pos = torch.arange(sentence_length, device=device).unsqueeze(1)  # [L, 1]
    i = torch.arange(d_model, device=device).unsqueeze(0)  # [1, D]

    angle_rads = get_angles(pos, i, d_model)  # [L, D]

    # 偶数下标：sin
    sines = torch.sin(angle_rads[:, 0::2])
    # 奇数下标：cos
    cosines = torch.cos(angle_rads[:, 1::2])

    # 拼接还原成 [L, D]
    position_embedding = torch.zeros((sentence_length, d_model), device=device, dtype=dtype)
    position_embedding[:, 0::2] = sines
    position_embedding[:, 1::2] = cosines

    # 增加 batch 维度 [1, L, D]
    position_embedding = position_embedding.unsqueeze(0)

    return position_embedding

def create_padding_mask(batch_data: torch.Tensor, pad_token_id: int = 0):
    """创建padding mask"""
    mask = (batch_data == pad_token_id).float()
    return mask[:, None, None, :]  # [B, 1, 1, L]

def create_look_ahead_mask(size: int):
    """生成Look-ahead mask (上三角矩阵)"""
    ones = torch.ones((size, size))
    # 参数 diagonal 控制对角线的偏移量：
    # - diagonal=0 表示主对角线及其以上为1
    # - diagonal=1 表示主对角线以上（不含主对角线）为1
    # 这里用 diagonal=1，生成主对角线以上为1，其余为0的上三角矩阵，用于屏蔽未来信息
    mask = torch.triu(ones, diagonal=1)
    return mask

def create_masks(
    inp_ids: torch.Tensor,  # [B, L_src]
    tar_ids: torch.Tensor,  # [B, L_tgt] —— 通常是 decoder 输入（已左移）
    src_pad_id: int = 0,
    tgt_pad_id: int = 0,
):
    """
    返回:
      encoder_padding_mask         : [B, 1, 1, L_src]  (给 EncoderLayer self-attn)
      decoder_mask (LA + padding)  : [B, 1, L_tgt, L_tgt]  (给 DecoderLayer 自注意力)
      encoder_decoder_padding_mask : [B, 1, 1, L_src]  (给 DecoderLayer cross-attn)
    语义:
      1 = 屏蔽（masked），0 = 保留
    """
    # 1) Encoder 端 padding mask
    encoder_padding_mask = create_padding_mask(inp_ids, pad_token_id=src_pad_id)  # [B,1,1,L_src]
    encoder_decoder_padding_mask = create_padding_mask(inp_ids, pad_token_id=src_pad_id)  # [B,1,1,L_src]

    # 2) Decoder 端 look-ahead + padding 合并
    L_tgt = tar_ids.size(1)

    # [L_tgt, L_tgt] → [1,1,L_tgt,L_tgt]，放到与输入相同 device/dtype
    look_ahead = create_look_ahead_mask(L_tgt).to(
        device=tar_ids.device, dtype=encoder_padding_mask.dtype
    ).unsqueeze(0).unsqueeze(1)  # [1,1,L_tgt,L_tgt]

    # 目标端 padding： [B,1,1,L_tgt] → 扩到 [B,1,L_tgt,L_tgt]
    decoder_padding_mask = create_padding_mask(tar_ids, pad_token_id=tgt_pad_id)  # [B,1,1,L_tgt]
    decoder_padding_mask = decoder_padding_mask.expand(-1, -1, L_tgt, -1)  # [B,1,L_tgt,L_tgt]

    # 合并（任一为 1 即屏蔽）
    decoder_mask = torch.maximum(decoder_padding_mask, look_ahead)  # [B,1,L_tgt,L_tgt]

    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask

def scaled_dot_product_attention(q, k, v, mask=None):
    """缩放点积注意力机制"""
    # (..., seq_len_q, seq_len_k)
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))

    # 缩放
    dk = q.size()[-1]
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=q.device))

    # 加上 mask
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 1, -1e9)

    # softmax 得到注意力权重
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    # 加权求和
    output = torch.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # 每个头的维度 Dh

        # 对应 Keras 的 Dense(d_model)
        self.WQ = nn.Linear(d_model, d_model, bias=True)
        self.WK = nn.Linear(d_model, d_model, bias=True)
        self.WV = nn.Linear(d_model, d_model, bias=True)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor):
        """x: [B, L, d_model] -> [B, num_heads, L, depth]"""
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.depth)  # [B, L, H, Dh]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, L, Dh]
        return x

    def _combine_heads(self, x: torch.Tensor):
        """x: [B, num_heads, L, depth] -> [B, L, d_model]"""
        B, H, L, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, H, Dh]
        x = x.view(B, L, H * Dh)  # [B, L, d_model]
        return x

    def forward(self, q, k, v, mask=None, return_attn: bool = True):
        B = q.size(0)

        # 线性映射
        q = self.WQ(q)  # [B, Lq, d_model]
        k = self.WK(k)  # [B, Lk, d_model]
        v = self.WV(v)  # [B, Lv, d_model]

        # 分头
        q = self._split_heads(q)  # [B, H, Lq, Dh]
        k = self._split_heads(k)  # [B, H, Lk, Dh]
        v = self._split_heads(v)  # [B, H, Lv, Dh]

        # 处理 mask：广播到 [B, H, Lq, Lk]
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B,1,Lq,Lk]
            elif mask.dim() == 4 and mask.size(1) == 1:
                pass  # 已是 [B,1,Lq,Lk]
            else:
                raise ValueError("mask 形状需为 [B, Lq, Lk] 或 [B, 1, Lq, Lk]")
            mask = mask.expand(B, self.num_heads, mask.size(-2), mask.size(-1))

        # 注意力
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask)  # [B,H,Lq,Dh], [B,H,Lq,Lk]

        # 合并头
        attn_out = self._combine_heads(attn_out)  # [B, Lq, d_model]

        # 输出线性层
        output = self.out_proj(attn_out)  # [B, Lq, d_model]

        if return_attn:
            return output, attn_weights
        return output

def feed_forward_network(d_model, dff):
    """前馈网络 FFN"""
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )

class EncoderLayer(nn.Module):
    """编码器层"""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        # Self-Attention
        attn_out, _ = self.mha(x, x, x, mask=src_mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.norm1(x + attn_out)  # 残差 + LayerNorm

        # Feed Forward
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.norm2(out1 + ffn_out)

        return out2

class DecoderLayer(nn.Module):
    """解码器层"""

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # masked self-attn
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # cross-attn

        self.ffn = feed_forward_network(d_model, dff)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(
            self,
            x: torch.Tensor,
            enc_out: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            enc_dec_mask: torch.Tensor = None,
    ):
        # 1) Masked Self-Attention
        attn1_out, attn_weights1 = self.mha1(x, x, x, mask=tgt_mask)
        attn1_out = self.dropout1(attn1_out)
        out1 = self.norm1(x + attn1_out)

        # 2) Cross-Attention
        attn2_out, attn_weights2 = self.mha2(out1, enc_out, enc_out, mask=enc_dec_mask)
        attn2_out = self.dropout2(attn2_out)
        out2 = self.norm2(out1 + attn2_out)

        # 3) FFN
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out)
        out3 = self.norm3(out2 + ffn_out)

        return out3, attn_weights1, attn_weights2

class EncoderModel(nn.Module):
    """编码器模型"""
    
    def __init__(self, num_layers: int, input_vocab_size: int, max_length: int,
                 d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 padding_idx: int = None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        # Embedding
        self.embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=padding_idx)

        # 位置编码：注册为 buffer（不参与训练/优化器）
        pe = get_position_embedding(max_length, d_model, device="cpu")  # 在CPU上创建
        self.register_buffer("position_embedding", pe, persistent=False)

        self.dropout = nn.Dropout(rate)

        # 堆叠 EncoderLayer
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )

        # 预存缩放因子
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        B, L = x.shape
        if L > self.max_length:
            raise ValueError(f"input_seq_len ({L}) should be ≤ max_length ({self.max_length})")

        # [B, L, D]
        x = self.embedding(x)
        # 缩放：使 embedding 的尺度与位置编码相近（论文做法）
        x = x * self.scale
        # 加位置编码（按实际序列长度切片）
        x = x + self.position_embedding[:, :L, :].to(x.device)  # 确保位置编码在正确的设备上

        x = self.dropout(x)

        # 逐层 Encoder
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

class DecoderModel(nn.Module):
    """解码器模型"""
    
    def __init__(self, num_layers: int, target_vocab_size: int, max_length: int,
                 d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 padding_idx: int = None):
        super().__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model

        # 词嵌入
        self.embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=padding_idx)

        # 位置编码（注册为 buffer，不参与训练）
        pe = get_position_embedding(max_length, d_model, device="cpu")  # 在CPU上创建
        self.register_buffer("position_embedding", pe, persistent=False)

        self.dropout = nn.Dropout(rate)

        # 堆叠解码层
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )

        self.scale = math.sqrt(d_model)

    def forward(
            self,
            x: torch.Tensor,  # [B, L_tgt] 目标端 token ids
            enc_out: torch.Tensor,  # [B, L_src, D] 编码器输出
            tgt_mask: torch.Tensor = None,  # [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]（look-ahead+padding）
            enc_dec_mask: torch.Tensor = None,  # [B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]（对 encoder 的 padding）
    ):
        B, Lt = x.shape
        if Lt > self.max_length:
            raise ValueError(f"output_seq_len ({Lt}) should be ≤ max_length ({self.max_length})")

        # (B, Lt, D)
        x = self.embedding(x) * self.scale
        x = x + self.position_embedding[:, :Lt, :].to(x.device)  # 确保位置编码在正确的设备上
        x = self.dropout(x)

        attention_weights = {}

        for i, layer in enumerate(self.decoder_layers, start=1):
            x, attn1, attn2 = layer(x, enc_out, tgt_mask=tgt_mask, enc_dec_mask=enc_dec_mask)
            attention_weights[f"decoder_layer{i}_att1"] = attn1  # [B, H, Lt, Lt]
            attention_weights[f"decoder_layer{i}_att2"] = attn2  # [B, H, Lt, Ls]

        # x: (B, Lt, D)
        return x, attention_weights

class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, num_layers, input_vocab_size, target_vocab_size,
                 max_length, d_model, num_heads, dff, rate=0.1,
                 src_padding_idx: int = None, tgt_padding_idx: int = None):
        super().__init__()
        self.encoder_model = EncoderModel(
            num_layers=num_layers,
            input_vocab_size=input_vocab_size,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=rate,
            padding_idx=src_padding_idx,
        )
        self.decoder_model = DecoderModel(
            num_layers=num_layers,
            target_vocab_size=target_vocab_size,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=rate,
            padding_idx=tgt_padding_idx,
        )
        # 等价于 Keras 的 Dense(target_vocab_size)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp_ids, tgt_ids, src_mask=None, tgt_mask=None, enc_dec_mask=None):
        """
        inp_ids: [B, L_src]  源端 token ids
        tgt_ids: [B, L_tgt]  目标端 token ids（训练时通常是 shift 后的 decoder 输入）
        src_mask:    [B, 1, L_src, L_src] 或 [B, L_src, L_src]（1=屏蔽）
        tgt_mask:    [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]（look-ahead+padding）
        enc_dec_mask:[B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]
        返回:
          logits: [B, L_tgt, target_vocab_size]
          attention_weights: dict，包含每层的 attn
        """
        enc_out = self.encoder_model(inp_ids, src_mask=src_mask)  # [B, L_src, D]
        dec_out, attention_weights = self.decoder_model(
            tgt_ids, enc_out, tgt_mask=tgt_mask, enc_dec_mask=enc_dec_mask
        )  # [B, L_tgt, D], dict
        logits = self.final_layer(dec_out)  # [B, L_tgt, V_tgt]
        return logits, attention_weights

def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("=" * 60)
    print("Step 6: 构建Transformer模型")
    print("=" * 60)
    
    # 模型结构参数
    num_layers = 4
    d_model = 128  # hidden-size
    dff = 512
    num_heads = 8
    dropout_rate = 0.25
    max_length = 30
    
    # 词表大小（示例值）
    input_vocab_size = 8192  # 葡萄牙语词表大小
    target_vocab_size = 8192  # 英语词表大小
    
    print(f"🔧 模型结构参数:")
    print(f"   编码器层数: {num_layers}")
    print(f"   解码器层数: {num_layers}")
    print(f"   嵌入维度: {d_model}")
    print(f"   前馈网络维度: {dff}")
    print(f"   注意力头数: {num_heads}")
    print(f"   Dropout率: {dropout_rate}")
    print(f"   最大序列长度: {max_length}")
    print(f"   输入词表大小: {input_vocab_size}")
    print(f"   目标词表大小: {target_vocab_size}")
    
    try:
        # 1. 构建Transformer模型
        print(f"\n🔨 构建Transformer模型...")
        model = Transformer(
            num_layers=num_layers,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=dropout_rate,
            src_padding_idx=0,  # 假设pad_token_id=0
            tgt_padding_idx=0,
        ).to(device)  # 将模型移动到正确的设备

        print(f"✅ Transformer模型构建完成！")
        
        # 2. 统计模型参数
        total_params = count_parameters(model)
        print(f"📊 模型参数统计:")
        print(f"   总参数数量: {total_params:,}")
        print(f"   可训练参数: {total_params:,}")

        # 3. 测试模型前向传播（对齐 train.py/step4 的真实 mask 逻辑）
        print(f"\n🧪 测试模型前向传播...")
        batch_size = 2
        src_len = 10
        tgt_len = 8

        # 创建示例输入（在正确的设备上）
        inp_ids = torch.randint(0, input_vocab_size, (batch_size, src_len)).to(device)
        tgt_ids = torch.randint(0, target_vocab_size, (batch_size, tgt_len)).to(device)

        print(f"   输入形状: {inp_ids.shape}")
        print(f"   目标形状: {tgt_ids.shape}")
        
        # 构造与真实训练一致的 mask
        # decoder 输入/输出拆分（teacher forcing）
        tar_inp = tgt_ids[:, :-1]
        tar_real = tgt_ids[:, 1:]

        # 根据 pad_id 构造三类 mask
        SRC_PAD_ID = 0
        TGT_PAD_ID = 0
        enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
            inp_ids, tar_inp, src_pad_id=SRC_PAD_ID, tgt_pad_id=TGT_PAD_ID
        )
        print(f"enc_pad_mask: {enc_pad_mask.shape}")
        print(f"dec_mask: {dec_mask.shape}")
        print(f"enc_dec_pad_mask: {enc_dec_pad_mask.shape}")
        
        # 将 enc_dec_pad_mask 扩展为 [B,1,L_tgt,L_src]
        enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, tar_inp.size(1), -1)

        # 前向传播（显式传入 mask）
        with torch.no_grad():
            logits, attention_weights = model(
                inp_ids, tar_inp,
                src_mask=enc_pad_mask,
                tgt_mask=dec_mask,
                enc_dec_mask=enc_dec_mask
            )
        
        print(f"   输出logits形状: {logits.shape}")
        print(f"   注意力权重数量: {len(attention_weights)}")
        
        # 4. 显示模型结构
        print(f"\n📋 模型结构:")
        print(model)
        
        print(f"\n✅ Transformer模型构建和测试完成！")
        
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback

        traceback.print_exc()
        print("💡 请检查参数设置是否正确")
