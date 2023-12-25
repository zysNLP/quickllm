# -*- coding: utf-8 -*-
"""
    @Project ：quickllm
    @File    ：basic_language_model_moe_transformers.py
    @Author  ：ys
    @Time    ：2023/12/21 18:10
    Mixtral-8x7b 模型中的moe部分，以下代码来自官方transformers库
"""

import torch
torch.manual_seed(123)

from quickllm.layers.moe_by_transformers import MixtralConfig
from quickllm.layers.moe_by_transformers import MixtralSparseMoeBlock

config = MixtralConfig()
moe = MixtralSparseMoeBlock(config)

hidden_states = torch.randn(4, 71, 4096)
hidden_states, router_logits = moe(hidden_states)

print(hidden_states.shape, router_logits.shape)