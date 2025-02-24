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
    def __init__(self, input_dim, output_dim, nums_head, max_len, batch_size=16, device="cuda:0"):
        super(RoPEAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nums_head = nums_head
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 定义独立的线性层，分别生成 q 和 k
        self.q_projection = nn.Linear(input_dim, nums_head * output_dim).to(self.device)
        self.k_projection = nn.Linear(input_dim, nums_head * output_dim).to(self.device)

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
        embeddings = embeddings.to(self.device)
        return embeddings

    def _rope(self, q, k):
        # q,k: (bs, head, max_len, output_dim)
        batch_size = q.shape[0]
        nums_head = q.shape[1]
        max_len = q.shape[2]
        output_dim = q.shape[-1]

        # (bs, head, max_len, output_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim)

        # cos_pos,sin_pos: (bs, head, max_len, output_dim)
        # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

        # q,k: (bs, head, max_len, output_dim)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape)  # reshape后就是正负交替了

        # 更新qw, *对应位置相乘
        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
        k2 = k2.reshape(k.shape)
        # 更新kw, *对应位置相乘
        k = k * cos_pos + k2 * sin_pos

        return q, k

    def _generate_qk(self, x):
        # 通过线性层投影 x，分别得到 q 和 k
        q_proj = self.q_projection(x)  # (batch_size, max_len, nums_head * output_dim)
        k_proj = self.k_projection(x)  # (batch_size, max_len, nums_head * output_dim)

        # 调整形状为 (batch_size, nums_head, max_len, output_dim)
        q = q_proj.view(self.batch_size, self.max_len, self.nums_head, self.output_dim).transpose(1, 2)
        k = k_proj.view(self.batch_size, self.max_len, self.nums_head, self.output_dim).transpose(1, 2)
        return q, k

    def forward(self, x):
        # 生成 q 和 k
        q, k = self._generate_qk(x)
        # 应用 RoPE
        q_rope, k_rope = self._rope(q, k)

        return q_rope, k_rope


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
    input_dim = 512  # 假设输入维度与输出维度相同

    # 初始化模型
    model = RoPEAttention(input_dim, output_dim, nums_head, max_len, batch_size)

    # 生成输入张量 x，形状为 (batch_size, max_len, input_dim)
    x = torch.randn(batch_size, max_len, input_dim).to(model.device)

    # 前向传播
    q, k = model(x)

    # 输出结果
    print("q shape:", q.shape)
    print("k shape:", k.shape)
    print("q (first few elements):", q[0, 0, 0, :5])  # 打印部分元素，避免输出过长
    print("k (first few elements):", k[0, 0, 0, :5])  # 打印部分元素，避免输出过长