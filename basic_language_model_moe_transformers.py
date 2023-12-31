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
from quickllm.layers.moe_by_transformers import load_balancing_loss_func

config = MixtralConfig()
moe = MixtralSparseMoeBlock(config)

hidden_states = torch.randn(4, 71, 4096)      # (batch_size, seq_len, hidden_size=4096)
hidden_states, router_logits = moe(hidden_states)

# outputs = (hidden_states,)
# outputs += (router_logits,)

aux_loss = load_balancing_loss_func(router_logits,
                                    config.num_local_experts,
                                    config.num_experts_per_tok)

# (4, 71, 4096) * （4096, 8）= (4, 71, 8)
print(hidden_states.shape, router_logits.shape)
print(aux_loss)