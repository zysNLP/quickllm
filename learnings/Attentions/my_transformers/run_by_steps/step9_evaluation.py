# -*- coding: utf-8 -*-
"""
Step 9: 模型评估和翻译测试
实现模型推理、翻译功能和注意力可视化
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 导入必要的组件（避免循环导入，直接包含函数）
import math

# 必要的辅助函数
def create_padding_mask(batch_data: torch.Tensor, pad_token_id: int = 0):
    """创建padding mask"""
    mask = (batch_data == pad_token_id).float()
    return mask[:, None, None, :]  # [B, 1, 1, L]

def create_look_ahead_mask(size: int):
    """生成Look-ahead mask (上三角矩阵)"""
    ones = torch.ones((size, size))
    mask = torch.triu(ones, diagonal=1)
    return mask

def create_masks(
        inp_ids: torch.Tensor,  # [B, L_src]
        tar_ids: torch.Tensor,  # [B, L_tgt] —— 通常是 decoder 输入（已左移）
        src_pad_id: int = 0,
        tgt_pad_id: int = 0,
):
    """
    创建各种mask
    返回:
      encoder_padding_mask         : [B, 1, 1, L_src]  (给 EncoderLayer self-attn)
      decoder_mask (LA + padding)  : [B, 1, L_tgt, L_tgt]  (给 DecoderLayer 自注意力)
      encoder_decoder_padding_mask : [B, 1, 1, L_src]  (给 DecoderLayer cross-attn)
    语义:
      1 = 屏蔽（masked），0 = 保留
    """
    # 1) Encoder 端 padding mask
    encoder_padding_mask = create_padding_mask(inp_ids, pad_token_id=src_pad_id)  # [B,1,1,L_src]
    encoder_decoder_padding_mask = create_padding_mask(inp_ids, pad_token_id=src_pad_id)  # [B,1,1,L_src]

    # 2) Decoder 端 look-ahead + padding 合并
    B, L_tgt = tar_ids.size(0), tar_ids.size(1)

    # [L_tgt, L_tgt] → [1,1,L_tgt,L_tgt]，放到与输入相同 device/dtype
    look_ahead = create_look_ahead_mask(L_tgt).to(
        device=tar_ids.device, dtype=encoder_padding_mask.dtype
    ).unsqueeze(0).unsqueeze(1)  # [1,1,L_tgt,L_tgt]

    # 目标端 padding： [B,1,1,L_tgt] → 扩到 [B,1,L_tgt,L_tgt]
    decoder_padding_mask = create_padding_mask(tar_ids, pad_token_id=tgt_pad_id)  # [B,1,1,L_tgt]
    decoder_padding_mask = decoder_padding_mask.expand(-1, -1, L_tgt, -1)  # [B,1,L_tgt,L_tgt]

    # 合并（任一为 1 即屏蔽）
    decoder_mask = torch.maximum(decoder_padding_mask, look_ahead)  # [B,1,L_tgt,L_tgt]

    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask

# 简单的Transformer模拟类（用于演示）
class Transformer(nn.Module):
    """简单的Transformer模拟类"""
    def __init__(self, num_layers, input_vocab_size, target_vocab_size,
                 max_length, d_model, num_heads, dff, rate=0.1,
                 src_padding_idx: int = None, tgt_padding_idx: int = None):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=src_padding_idx)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=tgt_padding_idx)
        self.final_layer = nn.Linear(d_model, target_vocab_size)
        
    def forward(self, inp_ids, tgt_ids, src_mask=None, tgt_mask=None, enc_dec_mask=None):
        # 简单的模拟前向传播
        enc_out = self.encoder_embedding(inp_ids)
        dec_out = self.decoder_embedding(tgt_ids)
        logits = self.final_layer(dec_out)
        return logits, {}

def encode_with_bos_eos(tokenizer, text: str):
    """编码文本并添加BOS/EOS标记"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("请确保 tokenizer 设置了 bos_token/eos_token")
    return [bos_id] + ids + [eos_id]

@torch.no_grad()
def evaluate(
        inp_sentence: str,
        transformer: Transformer,
        pt_tokenizer,
        en_tokenizer,
        max_length: int,
        device: str = None):
    """
    评估模型翻译能力
    inp_sentence: 输入的源语言字符串 (pt)
    transformer: 已训练的 Transformer
    pt_tokenizer, en_tokenizer: 分别是葡萄牙语和英语 tokenizer
    max_length: 最大生成长度
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer.eval()
    transformer.to(device)

    # 1. 编码输入，加 <s> 和 </s>
    inp_ids = encode_with_bos_eos(pt_tokenizer, inp_sentence)
    encoder_input = torch.tensor(inp_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, Ls)

    # 2. decoder 起始符 <s>
    start_id = en_tokenizer.bos_token_id
    end_id = en_tokenizer.eos_token_id
    decoder_input = torch.tensor([[start_id]], dtype=torch.long, device=device)  # (1, 1)

    # 3. 循环预测
    attention_weights = {}
    for _ in range(max_length):
        enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
            encoder_input, decoder_input,
            src_pad_id=pt_tokenizer.pad_token_id,
            tgt_pad_id=en_tokenizer.pad_token_id,
        )
        enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, decoder_input.size(1), -1)

        logits, attn = transformer(
            encoder_input, decoder_input,
            src_mask=enc_pad_mask,
            tgt_mask=dec_mask,
            enc_dec_mask=enc_dec_mask,
        )

        # 取最后一步预测
        next_token_logits = logits[:, -1, :]  # (1, V)
        predicted_id = torch.argmax(next_token_logits, dim=-1)  # (1,)

        if predicted_id.item() == end_id:
            break

        # 拼接到 decoder_input
        decoder_input = torch.cat(
            [decoder_input, predicted_id.unsqueeze(0)], dim=-1
        )  # (1, Lt+1)
        attention_weights = attn

    return decoder_input.squeeze(0).tolist(), attention_weights

def plot_encoder_decoder_attention(attention, input_sentence, result, layer_name, 
                                  pt_tokenizer, en_tokenizer):
    """
    可视化编码器-解码器注意力矩阵
    attention: 来自 forward 返回的 attention_weights dict
               形状 [B, num_heads, tgt_len, src_len]
    input_sentence: 源语言字符串
    result: 目标句子 token id 列表 (decoder 输出)
    layer_name: 指定可视化的层 key，比如 "decoder_layer1_att2"
    """
    fig = plt.figure(figsize=(16, 8))

    # 源句子编码
    input_id_sentence = pt_tokenizer.encode(input_sentence, add_special_tokens=False)

    # 取 batch 维度 squeeze，并转 numpy
    attn = attention[layer_name].squeeze(0)  # [num_heads, tgt_len, src_len]
    attn = attn.detach().cpu().numpy()

    for head in range(attn.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # 只取 result[:-1] 的注意力 (去掉最后 <eos>)
        ax.matshow(attn[head][:-1, :], cmap="viridis")

        fontdict = {"fontsize": 10}

        # X 轴: 输入 token (<s> + sentence + </s>)
        ax.set_xticks(range(len(input_id_sentence) + 2))
        ax.set_xticklabels(
            ["<s>"] + [pt_tokenizer.decode([i]) for i in input_id_sentence] + ["</s>"],
            fontdict=fontdict, rotation=90,
        )

        # Y 轴: decoder 输出 token
        ax.set_yticks(range(len(result)))
        ax.set_yticklabels(
            [en_tokenizer.decode([i]) for i in result if i < en_tokenizer.vocab_size],
            fontdict=fontdict,
        )

        ax.set_ylim(len(result) - 1.5, -0.5)
        ax.set_xlabel(f"Head {head + 1}")

    plt.tight_layout()
    plt.show()

def translate(input_sentence, transformer, pt_tokenizer, en_tokenizer,
              max_length=64, device=None, layer_name="", show_attention=False):
    """
    翻译函数
    """
    # 调用我们改好的 evaluate (PyTorch 版)
    result, attention_weights = evaluate(
        inp_sentence=input_sentence,
        transformer=transformer,
        pt_tokenizer=pt_tokenizer,
        en_tokenizer=en_tokenizer,
        max_length=max_length,
        device=device,
    )

    # 把 token id 转回句子
    predicted_sentence = en_tokenizer.decode(
        [i for i in result if i < en_tokenizer.vocab_size],
        skip_special_tokens=True
    )

    logger.info("Input: {}".format(input_sentence))
    logger.info(f"Predicted translation: {predicted_sentence}")

    # 如果传入了 layer_name，就画注意力图
    if show_attention and layer_name:
        plot_encoder_decoder_attention(
            attention_weights,
            input_sentence,
            result,
            layer_name,
            pt_tokenizer,
            en_tokenizer
        )

    return predicted_sentence

def batch_translate(input_sentences, transformer, pt_tokenizer, en_tokenizer,
                   max_length=64, device=None):
    """
    批量翻译函数
    """
    translations = []
    
    for sentence in input_sentences:
        try:
            translation = translate(
                sentence, transformer, pt_tokenizer, en_tokenizer,
                max_length=max_length, device=device
            )
            translations.append(translation)
        except Exception as e:
            logger.error(f"翻译失败: {sentence}, 错误: {e}")
            translations.append("翻译失败")
    
    return translations

def evaluate_translation_quality(translations, references=None):
    """
    评估翻译质量（简单的BLEU计算）
    """
    if references is None:
        print("📊 翻译结果:")
        for i, trans in enumerate(translations):
            print(f"   {i+1}. {trans}")
        return
    
    # 简单的BLEU计算（这里只是示例，实际应该使用更复杂的BLEU计算）
    print("📊 翻译质量评估:")
    for i, (trans, ref) in enumerate(zip(translations, references)):
        print(f"   {i+1}. 翻译: {trans}")
        print(f"      参考: {ref}")
        # 这里可以添加更复杂的BLEU计算
        print()

def analyze_attention_patterns(attention_weights, layer_name="decoder_layer1_att2"):
    """
    分析注意力模式
    """
    if layer_name not in attention_weights:
        print(f"⚠️ 未找到注意力层: {layer_name}")
        return
    
    attn = attention_weights[layer_name].squeeze(0)  # [num_heads, tgt_len, src_len]
    attn = attn.detach().cpu().numpy()
    
    print(f"📊 注意力模式分析 ({layer_name}):")
    print(f"   注意力头数: {attn.shape[0]}")
    print(f"   目标长度: {attn.shape[1]}")
    print(f"   源长度: {attn.shape[2]}")
    
    # 分析每个头的注意力分布
    for head in range(attn.shape[0]):
        head_attn = attn[head]
        max_attn = np.max(head_attn)
        mean_attn = np.mean(head_attn)
        print(f"   头 {head+1}: 最大注意力={max_attn:.4f}, 平均注意力={mean_attn:.4f}")

def create_mock_tokenizers():
    """创建模拟的tokenizer用于演示"""
    class MockTokenizer:
        def __init__(self, vocab_size, pad_token_id=0, bos_token_id=1, eos_token_id=2):
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
        
        def encode(self, text, add_special_tokens=False):
            # 简单的模拟编码
            words = text.split()
            ids = [hash(word) % (self.vocab_size - 10) + 10 for word in words]
            return ids
        
        def decode(self, ids, skip_special_tokens=True):
            # 简单的模拟解码
            if skip_special_tokens:
                ids = [i for i in ids if i >= 10]
            return " ".join([f"word_{i}" for i in ids])
    
    pt_tokenizer = MockTokenizer(8192, 0, 1, 2)
    en_tokenizer = MockTokenizer(8192, 0, 1, 2)
    
    return pt_tokenizer, en_tokenizer

if __name__ == "__main__":
    print("=" * 60)
    print("Step 9: 模型评估和翻译测试")
    print("=" * 60)
    
    # 参数设置
    max_length = 30
    d_model = 128
    num_layers = 4
    num_heads = 8
    dff = 512
    dropout_rate = 0.25
    
    # 词表大小
    input_vocab_size = 8192
    target_vocab_size = 8192
    
    print(f"🔧 评估参数:")
    print(f"   最大生成长度: {max_length}")
    print(f"   模型维度: {d_model}")
    print(f"   注意力头数: {num_heads}")
    
    try:
        # 1. 创建模型
        print(f"\n🔨 创建Transformer模型...")
        model = Transformer(
            num_layers=num_layers,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=dropout_rate,
            src_padding_idx=0,
            tgt_padding_idx=0,
        )
        
        # 2. 创建模拟tokenizer
        print(f"\n🔨 创建模拟tokenizer...")
        pt_tokenizer, en_tokenizer = create_mock_tokenizers()
        
        # 3. 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        print(f"   使用设备: {device}")
        
        # 4. 测试翻译
        print(f"\n🌍 测试翻译功能...")
        test_sentences = [
            "Olá, como você está?",
            "Obrigado pela sua ajuda.",
            "Este é um teste de tradução.",
        ]
        
        translations = batch_translate(
            test_sentences, model, pt_tokenizer, en_tokenizer,
            max_length=max_length, device=device
        )
        
        evaluate_translation_quality(translations)
        
        # 5. 测试单个翻译并分析注意力
        print(f"\n🔍 分析注意力模式...")
        test_sentence = "Olá mundo"
        
        result, attention_weights = evaluate(
            inp_sentence=test_sentence,
            transformer=model,
            pt_tokenizer=pt_tokenizer,
            en_tokenizer=en_tokenizer,
            max_length=max_length,
            device=device,
        )
        
        predicted_sentence = en_tokenizer.decode(
            [i for i in result if i < en_tokenizer.vocab_size],
            skip_special_tokens=True
        )
        
        print(f"   输入: {test_sentence}")
        print(f"   输出: {predicted_sentence}")
        
        # 分析注意力模式
        if attention_weights:
            analyze_attention_patterns(attention_weights, "decoder_layer1_att2")
        
        # 6. 可视化注意力（如果有matplotlib支持）
        print(f"\n📊 可视化注意力...")
        try:
            if attention_weights and "decoder_layer1_att2" in attention_weights:
                plot_encoder_decoder_attention(
                    attention_weights, test_sentence, result, "decoder_layer1_att2",
                    pt_tokenizer, en_tokenizer
                )
                print("   注意力可视化完成")
            else:
                print("   无法进行注意力可视化（缺少注意力权重）")
        except Exception as e:
            print(f"   注意力可视化失败: {e}")
        
        # 7. 模型性能分析
        print(f"\n📈 模型性能分析...")
        
        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   总参数数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        # 测试推理速度
        import time
        start_time = time.time()
        for _ in range(10):
            _ = evaluate(
                inp_sentence=test_sentence,
                transformer=model,
                pt_tokenizer=pt_tokenizer,
                en_tokenizer=en_tokenizer,
                max_length=max_length,
                device=device,
            )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"   平均推理时间: {avg_time:.4f}秒")
        
        print(f"\n✅ 模型评估和翻译测试完成！")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        print("💡 请检查模型和参数设置是否正确")
