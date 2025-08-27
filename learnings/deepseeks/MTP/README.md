## DeepSeek MTP（最小可运行演示）

该目录提供一个最小、自包含的 MTP（Multi-Token Prediction，多 Token 并行预测）演示：在一次解码步内，通过多层/多头的 MTP 模块并行预测未来多个位置的 token 分布。

### 演示内容
- 按 `spec_step_idx % num_mtp_layers` 选择当前 MTP 层，实现 t+1、t+2、… 的并行预测。
- 每个 MTP 层：对 `inputs_embeds` 与 `previous_hidden_states` 分别归一化 → 拼接 → 线性映射 → 解码块。
- 解码块（借鉴 DeepSeekV3 思路）：RoPE 旋转位置编码、KV 压缩路径、因果自注意力、MLP/可选简化 MoE、残差连接。
- 使用共享头（RMSNorm + 线性层）计算 logits。

### 文件
- `run_deepseek_mtp_minimal.py`：可直接运行的脚本，包含一个小模型与 MTP 多步 top-k 预测示例。

### 快速开始
1)（可选）创建虚拟环境并安装 torch
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
```

2) 运行示例
```bash
python learnings/deepseeks/MTP/run_deepseek_mtp_minimal.py
```

预期输出包含 hidden/logits 的形状和多步 MTP 的 top-5：
```
Hidden shape: (B, T, H)
Logits shape: (B, T, V)

MTP demo: top-5 predictions for future tokens from last position
  step 0 (t+1): ids=[...] probs=[...]
  step 1 (t+2): ids=[...] probs=[...]
  ...
```

### 流程（简版）
```text
inputs → (可选)embed → 选择MTP层 → norm+concat+linear → 解码块
                                     │
                                     └→ 注意力(RoPE+KV) + MLP/MoE + 残差
                                              │
                                              └→ 共享头(norm+lm head) → logits

循环 spec_step_idx ∈ [0..num_mtp_layers-1]，得到 t+1、t+2、… 的预测。
```

### 说明
- 为了清晰易跑，本演示省略了分布式路由/通信与权重加载等工程细节。
- 如需得到更有意义的预测，需要加载真实权重与 tokenizer，并对齐各维度配置。


