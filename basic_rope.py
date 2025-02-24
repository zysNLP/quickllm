# -*- coding: utf-8 -*-
"""
    @Project : blue-vs-red
    @File    : learning.py
    @Author  : ys
    @Time    : 2025/2/24 9:15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RoPEAttention(nn.Module):
    def __init__(self, input_dim, output_dim, nums_head, max_len, batch_size=16, dropout_rate=0.1, device="cuda:0"):
        super(RoPEAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nums_head = nums_head
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 定义独立的线性层，分别生成 q、k 和 v
        self.q_projection = nn.Linear(input_dim, nums_head * output_dim).to(self.device)
        self.k_projection = nn.Linear(input_dim, nums_head * output_dim).to(self.device)
        self.v_projection = nn.Linear(input_dim, nums_head * output_dim).to(self.device)
        # 输出投影层，将注意力结果映射回原始维度
        self.o_projection = nn.Linear(nums_head * output_dim, input_dim).to(self.device)

        # 规范化层
        self.layer_norm1 = nn.LayerNorm(input_dim).to(self.device)  # 输入规范化
        self.layer_norm2 = nn.LayerNorm(input_dim).to(self.device)  # 输出规范化

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def sinusoidal_position_embedding(self, batch_size, nums_head, max_len, output_dim):
        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)

        # (output_dim//2)
        # 即公式里的i, i的范围是 [0,d/2]
        ids = torch.arange(0, output_dim // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / output_dim)

        # (max_len, output_dim//2)
        # 即公式里的：pos / (10000^(2i/d))
        embeddings = position * theta

        # (max_len, output_dim//2, 2)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # (bs, head, max_len, output_dim//2, 2)
        # 在bs维度重复，其他维度都是1不重复
        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

        # (bs, head, max_len, output_dim)
        # reshape后就是：偶数sin, 奇数cos了
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        return embeddings.to(self.device)

    def _rope(self, q, k):
        # q,k: (bs, head, max_len, output_dim)
        batch_size = q.shape[0]
        nums_head = q.shape[1]
        max_len = q.shape[2]
        output_dim = q.shape[-1]

        # (bs, head, max_len, output_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim)

        # cos_pos,sin_pos: (bs, head, max_len, output_dim)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        # q,k: (bs, head, max_len, output_dim)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape)
        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
        k2 = k2.reshape(k.shape)
        k = k * cos_pos + k2 * sin_pos

        return q, k

    def _generate_qkv(self, x):
        q_proj = self.q_projection(x)
        k_proj = self.k_projection(x)
        v_proj = self.v_projection(x)
        q = q_proj.view(self.batch_size, self.max_len, self.nums_head, self.output_dim).transpose(1, 2)
        k = k_proj.view(self.batch_size, self.max_len, self.nums_head, self.output_dim).transpose(1, 2)
        v = v_proj.view(self.batch_size, self.max_len, self.nums_head, self.output_dim).transpose(1, 2)
        return q, k, v

    def attention(self, q, k, v, mask=None, use_RoPE=True):
        # q.shape: (bs, head, seq_len, dk)
        # k.shape: (bs, head, seq_len, dk)
        # v.shape: (bs, head, seq_len, dk)

        if use_RoPE:
            q, k = self._rope(q, k)

        d_k = k.size()[-1]

        # 计算注意力权重
        att_logits = torch.matmul(q, k.transpose(-2, -1))  # (bs, head, seq_len, seq_len)
        att_logits /= math.sqrt(d_k)

        if mask is not None:
            att_logits = att_logits.masked_fill(mask == 0, -1e9)

        # Softmax 归一化
        att_scores = F.softmax(att_logits, dim=-1)
        att_scores = self.dropout(att_scores)  # 添加 dropout

        # 加权求和
        output = torch.matmul(att_scores, v)  # (bs, head, seq_len, dk)
        return output, att_scores

    def forward(self, x, mask=None):
        # 输入规范化
        x_norm = self.layer_norm1(x)

        # 生成 q、k、v
        q, k, v = self._generate_qkv(x_norm)

        # 注意力机制
        attn_output, att_scores = self.attention(q, k, v, mask=mask, use_RoPE=True)

        # 将注意力输出转换回原始形状
        attn_output = attn_output.transpose(1, 2).reshape(self.batch_size, self.max_len, self.nums_head * self.output_dim)
        attn_output = self.o_projection(attn_output)  # 映射回 input_dim

        # 残差连接
        output = x + self.dropout(attn_output)

        # 输出规范化
        output = self.layer_norm2(output)

        return output, att_scores


if __name__ == '__main__':
    # 设置随机数种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 定义超参数
    batch_size = 16
    nums_head = 8
    max_len = 256
    output_dim = 512
    input_dim = 512
    dropout_rate = 0.1

    # 初始化模型
    model = RoPEAttention(input_dim, output_dim, nums_head, max_len, batch_size, dropout_rate)

    # 生成输入张量 x 和掩码
    x = torch.randn(batch_size, max_len, input_dim).to(model.device)
    # 示例掩码：模拟因果注意力（下三角矩阵）
    mask = torch.tril(torch.ones(max_len, max_len)).expand(batch_size, nums_head, max_len, max_len).to(model.device)

    # 前向传播
    output, att_scores = model(x, mask)

    # 输出结果
    print("x shape:", x.shape)
    print("output shape:", output.shape)
    print("att_scores shape:", att_scores.shape)
    print("output (first few elements):", output[0, 0, :5])
    print("att_scores (first few elements):", att_scores[0, 0, 0, :5])