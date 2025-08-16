# -*- coding: utf-8 -*-
"""
配置文件
"""

# 模型路径
TRAINED_MODEL_PATH = '/data2/users/yszhang/quickllm/outputs/tiny_grpo/output/step_232'
FALLBACK_TOKENIZER_PATH = "/data2/users/yszhang/quickllm/models/Qwen2.5-0.5B-Instruct"

# 服务端口
TRANSFORMERS_PORT = 16020
VLLM_PORT = 16021

# 系统提示
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

# 生成配置 - Transformers
GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 1.0,
    "top_p": 1.0,
    "do_sample": True,
    "repetition_penalty": 1.0,
}

# 生成配置 - vLLM
VLLM_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "repetition_penalty": 1.0,
    "stop_token_ids": [],
    "seed": 42,
}

# vLLM模型配置
VLLM_MODEL_CONFIG = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.8,
    "trust_remote_code": True,
}

# MCP配置
SERVICE_NAME = "math-reasoning-service"

# 日志配置
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_LEVEL = "INFO"

# 随机种子
RANDOM_SEED = 42 