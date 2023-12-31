# -*- coding: utf-8 -*- 
"""
    @Time : 2023/12/31 18:21 
    @Author : ys 
    @File : basic_run_moe_flash_attention.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer

model_id = "casperhansen/mixtral-instruct-awq"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)

streamer = TextStreamer(tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True)

text = "[INST] How to make a cake? [/INST]"

tokens = tokenizer(
    text,
    return_tensors="pt"
).input_ids.to("cuda:0")

output = model.generate(
    tokens,
    streamer=streamer,
    max_new_tokens=512
)