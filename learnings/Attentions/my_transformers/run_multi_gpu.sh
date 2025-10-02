#!/bin/bash

# 多卡训练启动脚本
# 使用显存较少的GPU: 2,4,5,6,7

echo "启动多卡Transformer训练..."
echo "使用的GPU: 2,4,5,6,7"
echo "批次大小: 128 (多卡训练)"
echo "混合精度: 启用"
echo "================================"

# 激活环境
source .venv/bin/activate

# 运行训练
python learnings/Attentions/my_transformers/train_transformer.py

echo "训练完成！"


