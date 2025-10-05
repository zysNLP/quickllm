

# Step 6: Transformer 模型构建

## 概述

本步骤实现了完整的 Transformer 模型架构，包括核心的注意力机制、编码器、解码器以及完整的 Transformer 模型。这是整个训练流程的核心组件，为后续的训练和评估提供模型基础。

## 文件内容

### 核心组件

1. **位置编码 (Positional Encoding)**
   - `get_position_embedding()`: 生成正弦余弦位置编码

2. **注意力机制 (Attention Mechanism)**
   - `scaled_dot_product_attention()`: 缩放点积注意力
   - `MultiHeadAttention`: 多头注意力机制

3. **前馈网络 (Feed Forward Network)**
   - `feed_forward_network()`: 两层全连接网络

4. **掩码机制 (Masking)**
   - `create_padding_mask()`: 创建填充掩码
   - `create_look_ahead_mask()`: 创建前瞻掩码

5. **模型层 (Model Layers)**
   - `EncoderLayer`: 编码器层
   - `DecoderLayer`: 解码器层
   - `EncoderModel`: 编码器模型
   - `DecoderModel`: 解码器模型
   - `Transformer`: 完整的 Transformer 模型

## 面试题解答

### 1. 什么是注意力机制？

**核心概念**：注意力机制是一种让模型能够"关注"输入序列中不同位置信息的机制，模拟人类在处理信息时的选择性注意过程。

**数学原理**：
```python
def scaled_dot_product_attention(q, k, v, mask=None):
    # Q, K, V 分别表示查询、键、值矩阵
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # 计算相似度
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk))  # 缩放
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # 归一化
    output = torch.matmul(attention_weights, v)  # 加权求和
```

**工作原理**：
1. **相似度计算**：通过 Q 和 K 的点积计算查询与键的相似度
2. **缩放**：除以 √d_k 防止梯度消失
3. **归一化**：使用 softmax 将相似度转换为概率分布
4. **加权求和**：用注意力权重对 V 进行加权求和

**优势**：
- 能够捕捉长距离依赖关系
- 计算可以并行化
- 提供可解释性（注意力权重可视化）

### 2. 什么是 Transformer？

**定义**：Transformer 是一种基于注意力机制的seq2seq的模型架构，完全摒弃了循环和卷积结构，仅使用注意力机制来处理序列信息。

**核心架构**：
```
Transformer = Encoder + Decoder

Encoder:
  - Multi-Head Self-Attention
  - Feed Forward Network
  - Residual Connection + Layer Normalization

Decoder:
  - Masked Multi-Head Self-Attention
  - Multi-Head Cross-Attention (with Encoder output)
  - Feed Forward Network
  - Residual Connection + Layer Normalization
```

**关键特点**：
1. **纯注意力架构**：不使用 RNN 或 CNN
2. **并行计算**：所有位置可以同时计算
3. **多头注意力**：从多个子空间学习不同的表示
4. **位置编码**：为序列提供位置信息
5. **残差连接**：缓解梯度消失问题

**实现细节**：
```python
class Transformer(nn.Module):
    def __init__(self, num_layers, input_vocab_size, target_vocab_size, ...):
        self.encoder_model = EncoderModel(...)
        self.decoder_model = DecoderModel(...)
        self.final_layer = nn.Linear(d_model, target_vocab_size)
    
    def forward(self, inp_ids, tgt_ids, src_mask=None, tgt_mask=None, enc_dec_mask=None):
        enc_out = self.encoder_model(inp_ids, src_mask=src_mask)
        dec_out, attention_weights = self.decoder_model(tgt_ids, enc_out, ...)
        logits = self.final_layer(dec_out)
        return logits, attention_weights
```

### 3. Mask 的作用是什么？

**Mask 的核心作用**：控制注意力机制中哪些位置应该被忽略，防止模型看到不应该看到的信息。

**三种主要 Mask**：

#### 3.1 Padding Mask（填充掩码）
```python
def create_padding_mask(batch_data: torch.Tensor, pad_token_id: int = 0):
    mask = (batch_data == pad_token_id).float()
    return mask[:, None, None, :]  # [B, 1, 1, L]
```
- **作用**：忽略填充位置，防止模型关注无意义的填充 token
- **应用场景**：处理不同长度的序列时，短序列会被填充到相同长度

#### 3.2 Look-ahead Mask（前瞻掩码）
```python
def create_look_ahead_mask(size: int):
    ones = torch.ones((size, size))
    mask = torch.triu(ones, diagonal=1)  # 上三角矩阵
    return mask
```
- **作用**：防止解码器在生成时看到未来的 token
- **应用场景**：训练时使用 Teacher Forcing，但需要防止信息泄露

#### 3.3 Cross-attention Mask（交叉注意力掩码）
- **作用**：控制解码器对编码器输出的注意力
- **应用场景**：确保解码器只关注编码器的有效位置

**Mask 的数学实现**：
```python
# 在注意力计算中应用 mask
if mask is not None:
    scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 1, -1e9)
attention_weights = F.softmax(scaled_attention_logits, dim=-1)
```
- 将 mask 位置设置为 -∞，softmax 后变为 0
- 实现"忽略"的效果

### 4. Dropout 内部是怎么做的？

**Dropout 的核心思想**：在训练过程中随机将一部分神经元输出设置为 0，防止过拟合。

**数学原理**：
```python
# 训练时
def dropout_forward(x, p=0.1, training=True):
    if training:
        # 生成随机掩码
        mask = torch.rand_like(x) > p
        # 应用掩码并缩放
        return x * mask / (1 - p)
    else:
        # 推理时不改变
        return x
```

**实现细节**：

#### 4.1 随机掩码生成
```python
# PyTorch 内部实现（简化版）
mask = torch.rand_like(input) > dropout_prob
output = input * mask / (1 - dropout_prob)
```

#### 4.2 缩放因子
- **训练时**：除以 `(1 - dropout_prob)` 保持期望值不变
- **推理时**：直接输出，不进行缩放

#### 4.3 在 Transformer 中的应用
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        self.dropout1 = nn.Dropout(rate)  # 注意力后 dropout
        self.dropout2 = nn.Dropout(rate)  # 前馈网络后 dropout
    
    def forward(self, x, src_mask=None):
        # Self-Attention + Dropout
        attn_out, _ = self.mha(x, x, x, mask=src_mask)
        attn_out = self.dropout1(attn_out)  # 随机置零
        out1 = self.norm1(x + attn_out)
        
        # Feed Forward + Dropout
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)  # 随机置零
        out2 = self.norm2(out1 + ffn_out)
        return out2
```

**Dropout 的优势**：
1. **防止过拟合**：强制模型不依赖特定的神经元
2. **提高泛化能力**：增强模型的鲁棒性
3. **正则化效果**：相当于集成多个子模型

**注意事项**：
- 只在训练时生效，推理时自动关闭
- 缩放因子确保训练和推理的期望值一致
- 位置很重要：通常在激活函数之后、残差连接之前

## 模型参数

### 默认配置
```python
num_layers = 4        # 编码器/解码器层数
d_model = 128         # 模型维度
dff = 512            # 前馈网络维度
num_heads = 8        # 注意力头数
dropout_rate = 0.25  # Dropout 概率
max_length = 30      # 最大序列长度
```

### 参数统计
- **总参数数量**：约 2.1M 参数
- **可训练参数**：全部参数都可训练
- **内存占用**：约 8.4MB（FP32）

## 使用方法

### 运行脚本
```bash
python step6_model_building.py
```

### 输出内容
1. **模型结构展示**：打印完整的模型架构
2. **参数统计**：显示模型参数数量和内存占用
3. **前向传播测试**：验证模型输入输出形状
4. **注意力权重**：展示注意力机制的输出

## 技术亮点

1. **完整的 Transformer 实现**：包含所有核心组件
2. **模块化设计**：每个组件都可以独立使用
3. **GPU 加速支持**：支持 CUDA 计算
4. **可解释性**：提供注意力权重输出
5. **内存优化**：使用 buffer 存储位置编码

## 扩展阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化解释
- [Transformer 详解](https://zhuanlan.zhihu.com/p/338817680) - 中文详细解析


## 附：为什么需要三个 mask？如何根据输入形状生成？

下面结合本步骤脚本中的实际张量形状进行说明。示例中：
- 输入（源端）形状: `inp_ids = torch.Size([2, 10])`，其中 `B=2`、`L_src=10`
- 目标（解码端）形状: `tgt_ids = torch.Size([2, 8])`，其中 `B=2`、`L_tgt_raw=8`
- 注意：训练时会使用 Teacher Forcing，将目标序列左移一位作为解码器输入 `tar_inp = tgt_ids[:, :-1]`，因此 `L_tgt = 7`
- `create_masks(inp_ids, tar_inp, src_pad_id=0, tgt_pad_id=0)` 返回三个张量：
  1) `encoder_padding_mask: [B, 1, 1, L_src] = [2, 1, 1, 10]`
  2) `decoder_mask:         [B, 1, L_tgt, L_tgt] = [2, 1, 7, 7]`
  3) `encoder_decoder_padding_mask: [B, 1, 1, L_src] = [2, 1, 1, 10]`

注意：在解码器的交叉注意力中，第三个 mask 会被扩展为 `[B, 1, L_tgt, L_src] = [2, 1, 7, 10]` 以与注意力分数张量对齐（此处的 `L_tgt` 指 `tar_inp` 的长度 7）。

### 这三个 mask 是如何生成的？

1) Encoder Padding Mask（编码器填充掩码）
- 目的：在编码器自注意力中忽略源序列中的 padding 位置
- 生成：
  - 先通过 `create_padding_mask(inp_ids, pad_token_id=src_pad_id)` 得到形状 `[B, 1, 1, L_src]`
  - 其中 mask 值为 1 的地方表示“屏蔽/忽略”，0 表示“保留”

2) Decoder Mask（解码器合并掩码 = Look-ahead + Padding）
- 目的：
  - Look-ahead 部分防止解码器在第 t 个位置看到 t 之后的“未来”信息（避免信息泄露）
  - Padding 部分忽略目标序列中的 padding 位置
- 生成：
  - `look_ahead = create_look_ahead_mask(L_tgt)` 得到形状 `[L_tgt, L_tgt]`（此处 `L_tgt=7`）
    - 实现为 `torch.triu(torch.ones(L_tgt, L_tgt), diagonal=1)`，对角线上方为 1，其余为 0
    - 语义：第 i 行第 j 列，若 j>i（未来位置）则置 1 表示屏蔽
  - `decoder_padding_mask = create_padding_mask(tar_ids, pad_token_id=tgt_pad_id)` 得到 `[B, 1, 1, L_tgt]`
  - 将其扩展为 `[B, 1, L_tgt, L_tgt]`，与 look-ahead 对齐
  - 最后合并：`decoder_mask = max(decoder_padding_mask, look_ahead)` → `[B, 1, L_tgt, L_tgt]`

3) Encoder-Decoder Padding Mask（交叉注意力掩码）
- 目的：在解码器的交叉注意力中，让每个目标位置仅关注源序列的有效（非 padding）位置
- 生成：
  - 先用 `create_padding_mask(inp_ids, pad_token_id=src_pad_id)` 得到 `[B, 1, 1, L_src]`
  - 在交叉注意力计算前扩展为 `[B, 1, L_tgt, L_src]`，使 batch 内每个目标位置都能屏蔽源端的 padding 位置

### 为什么要这么生成？（与注意力计算的形状对齐）

以缩放点积注意力为例：注意力 logits 的形状通常为 `[B, num_heads, L_q, L_k]`。mask 需要能按广播规则与该形状对齐，因此：
- 编码器自注意力：`L_q = L_k = L_src`，故使用 `[B, 1, 1, L_src]`，可广播为 `[B, H, L_src, L_src]`
- 解码器自注意力：`L_q = L_k = L_tgt`，且需要 look-ahead，故使用 `[B, 1, L_tgt, L_tgt]`
- 交叉注意力：`L_q = L_tgt, L_k = L_src`，故最终需要 `[B, 1, L_tgt, L_src]`

### 掩码的数值约定与作用位置

- 数值约定：本项目中约定 1 表示屏蔽（masked），0 表示保留（keep）
- 作用位置：在注意力 logits 上用 `masked_fill(mask == 1, -1e9)` 将被屏蔽位置的分数置为极小值，softmax 后权重趋近 0，从而不会对输出产生贡献

### 与示例形状的对应关系（B=2, L_src=10, L_tgt_raw=8, L_tgt=7）

- `encoder_padding_mask`: `[2, 1, 1, 10]` → 编码器自注意力屏蔽源端 padding
- `decoder_mask`: `[2, 1, 7, 7]` → 解码器自注意力同时实现 look-ahead 与 padding 屏蔽
- `encoder_decoder_padding_mask` 扩展后：`[2, 1, 7, 10]` → 解码器交叉注意力屏蔽源端 padding

这样设计能确保：
1. 模型不会把概率“浪费”在 padding 上
2. 解码阶段不泄露未来信息（保证自回归训练/推理的一致性）
3. 三类注意力（enc self-attn、dec self-attn、cross-attn）都能拿到形状匹配的掩码张量
