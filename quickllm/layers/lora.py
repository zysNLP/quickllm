# -*- coding: utf-8 -*- 
"""
    @Project ：quickllm-loc 
    @File    ：lora.py
    @Author  ：ys
    @Time    ：2024/1/10 15:05 
"""

import torch
from torch import nn


class Lora(nn.Module):
    def __init__(self, r, output_size, alpha, dropout_prob, bias=False, name='lora'):
        super(Lora, self).__init__()
        self.name = name
        self.r = r
        self.output_size = output_size
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout_prob)
        self.use_bias = bias

        # 初始化权重层
        self.low = nn.Linear(inputs.shape[1], self.r, bias=self.use_bias)
        self.up = nn.Linear(self.r, self.output_size, bias=self.use_bias)

    def forward(self, inputs, training=False):
        x = self.low(inputs)
        x = self.up(x)
        if training:
            x = self.dropout(x)
        x = x * self.scaling
        return x
