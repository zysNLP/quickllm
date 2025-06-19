#!/bin/bash

# 启动 Transformers 服务
nohup python server_transformers.py > logs/trans.log 2>&1 &

# 启动 vLLM 服务  
nohup python server_vllm.py > logs/vllm.log 2>&1 &

echo "服务已启动:"
echo "- Transformers: 端口 16020"
echo "- vLLM: 端口 16021" 