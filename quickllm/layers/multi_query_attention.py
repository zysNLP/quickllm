# -*- coding: utf-8 -*-
"""
    @Project ：quickllm
    @File    ：multi_query_attention.py
    @Author  ：ys
    @Time    ：2023/12/21 21:21
"""

import torch
import torch.nn as nn
import math


class VanillaAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(VanillaAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None):
        query, key, value = x, x, x
        batch_size, seq_length, _ = query.shape
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attention_scores += attn_mask.unsqueeze(1).unsqueeze(2)  # Add the mask to the attention scores

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        attention_output = self.out_proj(attention_output)

        return attention_output


class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries):
        super(MultiQueryAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.num_queries = num_queries

        self.query_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_queries)])
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None):
        query, key, value = x, x, x
        batch_size, seq_length, _ = query.shape
        queries = [proj(query) for proj in self.query_projs]

        key = self.key_proj(key)
        value = self.value_proj(value)

        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)

        attention_outputs = []
        for q in queries:
            q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
            attention_scores = torch.matmul(q, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attn_mask is not None:
                attention_scores += attn_mask.unsqueeze(1).unsqueeze(2)  # Add the mask to the attention scores

            attention_probs = torch.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_probs, value)
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length,
                                                                                  self.embed_dim)
            attention_outputs.append(attention_output)

        attention_output = torch.sum(torch.stack(attention_outputs), dim=0)
        attention_output = self.out_proj(attention_output)

        return attention_output