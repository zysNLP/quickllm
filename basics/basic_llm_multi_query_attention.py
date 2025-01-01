# -*- coding: utf-8 -*-
"""
    @Project ：quickllm
    @File    ：basic_llm_multi_query_attention.py
    @Author  ：ys
    @Time    ：2023/12/22 21:12
    官方项目：https://github.com/THUDM/ChatGLM2-6B
"""


import torch

from quickllm.transformers_models.configuration_chatglm import ChatGLMConfig
from quickllm.transformers_models.modeling_chatglm import SelfAttention

x = torch.rand(4, 10, 4096)
glm_config = ChatGLMConfig()

attn_transformers = SelfAttention(config=glm_config, layer_number=1)
output_transformers, _ = attn_transformers(x)

print(output_transformers.shape)
