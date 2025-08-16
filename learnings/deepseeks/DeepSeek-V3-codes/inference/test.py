# -*- coding: utf-8 -*-
"""
    @Project : learnings
    @File    : test.py
    @Author  : ys
    @Time    : 2025/2/25 10:29
"""

from transformers import AutoTokenizer

# 指定仓库路径
#ckpt_path = "/data2/users/yszhang/learnings/DeepSeek-V3/deepseek-V3"
ckpt_path = "/data/quickllm/DeepSeek-V3/deepseek-V3"

# 只加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

# 测试 tokenizer
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(tokens)
token_ids = tokenizer.encode(text)
print(token_ids)
