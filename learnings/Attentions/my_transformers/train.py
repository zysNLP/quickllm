# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_cosine_schedule_with_warmup
from datetime import datetime
from loguru import logger

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def check_env():
    """
    检查 PyTorch 环境信息、GPU 状态，以及常用依赖库版本。
    返回推荐的 device ('cuda' 或 'cpu')。
    """
    logger.info("===== PyTorch & 系统信息 =====")
    logger.info("torch.__version__:", torch.__version__)
    logger.info("python version:", sys.version_info)

    logger.info("\n===== 常用库版本 =====")
    for module in (mpl, np, pd, torch):
        logger.info(module.__name__, module.__version__)

    logger.info("\n===== GPU 检查 =====")
    logger.info("torch.cuda.is_available():", torch.cuda.is_available())
    logger.info("torch.version.cuda:", torch.version.cuda)
    try:
        logger.info("cudnn version:", torch.backends.cudnn.version())
    except Exception as e:
        logger.info("cudnn version: N/A", e)

    if torch.cuda.is_available():
        logger.info("GPU count:", torch.cuda.device_count())
        logger.info("Current device id:", torch.cuda.current_device())
        logger.info("GPU name:", torch.cuda.get_device_name(0))
        logger.info("bfloat16 supported:", torch.cuda.is_bf16_supported())

        # 启用 TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        device = "cuda"
    else:
        logger.info("⚠️ 没检测到 CUDA，可强制 device='cpu' 运行，但速度会慢")
        device = "cpu"

    logger.info("\n推荐使用 device: Cuda;")
    return device


def load_translation_dataset(train_path: str, val_path: str, delimiter: str = "\t"):
    """
    加载葡萄牙语-英语翻译数据集 (TED Talks)

    参数:
        train_path: 训练集 CSV 文件路径
        val_path: 验证集 CSV 文件路径
        delimiter: 分隔符，默认制表符 '\t'

    返回:
        train_dataset, val_dataset
    """
    logger.info("开始加载数据...")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_path,
            "validation": val_path
        },
        column_names=["pt", "en"],
        delimiter=delimiter
    )

    logger.info("数据集类型:", type(dataset))
    logger.info(dataset)

    # 打印一个样本
    sample = dataset["train"][0]
    logger.info(f"示例数据 -> pt: {sample['pt']} | en: {sample['en']}")

    return dataset["train"], dataset["validation"]


def train_and_load_tokenizers(
    train_dataset,
    pt_key="pt",
    en_key="en",
    vocab_size=2 ** 13,
    min_freq=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    save_dir_pt="tok_pt",
    save_dir_en="tok_en",
    max_length=1024
):
    """
    训练并加载葡萄牙语和英语的 ByteLevel BPE Tokenizer

    参数:
        train_dataset: 数据集 (需包含 pt_key 和 en_key 两列)
        pt_key: 源语言字段名 (默认 "pt")
        en_key: 目标语言字段名 (默认 "en")
        vocab_size: 词表大小
        min_freq: 最小词频
        special_tokens: 特殊符号
        save_dir_pt: 葡语 tokenizer 保存路径
        save_dir_en: 英语 tokenizer 保存路径
        max_length: 模型最大序列长度

    返回:
        pt_tokenizer, en_tokenizer
    """

    def iter_lang(ds, key):
        for ex in ds:
            txt = ex[key]
            if isinstance(txt, bytes):
                txt = txt.decode("utf-8")
            yield txt

    # 初始化 tokenizer
    pt_bbpe = ByteLevelBPETokenizer(add_prefix_space=True)
    en_bbpe = ByteLevelBPETokenizer(add_prefix_space=True)

    # 训练 tokenizer
    pt_bbpe.train_from_iterator(
        iter_lang(train_dataset, pt_key),
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )
    en_bbpe.train_from_iterator(
        iter_lang(train_dataset, en_key),
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )

    # 保存 vocab/merges + tokenizer.json
    Path(save_dir_pt).mkdir(exist_ok=True)
    Path(save_dir_en).mkdir(exist_ok=True)
    pt_bbpe.save_model(save_dir_pt)
    en_bbpe.save_model(save_dir_en)
    pt_bbpe._tokenizer.save(f"{save_dir_pt}/tokenizer.json")
    en_bbpe._tokenizer.save(f"{save_dir_en}/tokenizer.json")

    # 用 PreTrainedTokenizerFast 加载
    pt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{save_dir_pt}/tokenizer.json")
    en_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{save_dir_en}/tokenizer.json")

    # 设置特殊符号
    for tok in (pt_tokenizer, en_tokenizer):
        tok.pad_token = "<pad>"
        tok.unk_token = "<unk>"
        tok.bos_token = "<s>"
        tok.eos_token = "</s>"
        tok.mask_token = "<mask>"
        tok.model_max_length = max_length
        tok.padding_side = "right"

    logger.info("pt vocab size:", len(pt_tokenizer))
    logger.info("en vocab size:", len(en_tokenizer))

    return pt_tokenizer, en_tokenizer


def test_tokenizers(en_tokenizer, pt_tokenizer,
                    en_sample: str = "Transformer is awesome.",
                    pt_sample: str = "Transformers são incríveis."):
    """
    测试英文和葡萄牙语的 tokenizer 编码/解码是否正确，
    并打印 token IDs、单个 ID 对应的 token 结果。

    参数:
        en_tokenizer: 英语 tokenizer
        pt_tokenizer: 葡语 tokenizer
        en_sample: 英文测试句子
        pt_sample: 葡文测试句子
    """

    # --- English ---
    logger.info("=== English Tokenizer Test ===")
    en_ids = en_tokenizer.encode(en_sample, add_special_tokens=False)
    logger.info(f"[EN] Tokenized IDs: {en_ids}")

    en_decoded = en_tokenizer.decode(
        en_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    logger.info(f"[EN] Decoded string: {en_decoded}")
    assert en_decoded == en_sample, "EN decode != original input!"

    logger.info("[EN] id --> decoded([id])  |  id --> token(str)")
    for tid in en_ids:
        single_decoded = en_tokenizer.decode(
            [tid],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        token_str = en_tokenizer.convert_ids_to_tokens(tid)
        logger.info(f"{tid:>6} --> {single_decoded!r}  |  {tid:>6} --> {token_str!r}")

    logger.info("\n" + "-" * 60 + "\n")

    # --- Portuguese ---
    logger.info("=== Portuguese Tokenizer Test ===")
    pt_ids = pt_tokenizer.encode(pt_sample, add_special_tokens=False)
    logger.info(f"[PT] Tokenized IDs: {pt_ids}")

    pt_decoded = pt_tokenizer.decode(
        pt_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    logger.info(f"[PT] Decoded string: {pt_decoded}")
    assert pt_decoded == pt_sample, "PT decode != original input!"

    logger.info("[PT] id --> decoded([id])  |  id --> token(str)")
    for tid in pt_ids:
        single_decoded = pt_tokenizer.decode(
            [tid],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        token_str = pt_tokenizer.convert_ids_to_tokens(tid)
        logger.info(f"{tid:>6} --> {single_decoded!r}  |  {tid:>6} --> {token_str!r}")


def build_dataloaders(
    train_dataset,
    val_dataset,
    pt_tokenizer,
    en_tokenizer,
    batch_size: int = 64,
    max_length: int = 48,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    """
    构建训练和验证 DataLoader（等价 TF 的 filter_by_max_length + padded_batch）

    参数:
        train_dataset: HuggingFace Dataset (训练集)
        val_dataset: HuggingFace Dataset (验证集)
        pt_tokenizer: 葡语 tokenizer (源语言)
        en_tokenizer: 英语 tokenizer (目标语言)
        batch_size: 批大小
        max_length: 样本最大长度（超过则过滤）
        num_workers: DataLoader worker 数量
        shuffle_train: 是否打乱训练集

    返回:
        train_loader, val_loader
    """

    # 1) 小工具：编码 + 添加 BOS/EOS
    def encode_with_bos_eos(tokenizer, text: str):
        ids = tokenizer.encode(text, add_special_tokens=False)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError("请确保 tokenizer 设置了 bos_token/eos_token")
        return [bos_id] + ids + [eos_id]

    # 2) 构造已过滤的样本对
    def build_filtered_pairs(hf_split, pt_tok, en_tok, max_len: int):
        pairs, kept, skipped = [], 0, 0
        for ex in hf_split:
            pt_ids = encode_with_bos_eos(pt_tok, ex["pt"])
            en_ids = encode_with_bos_eos(en_tok, ex["en"])
            if len(pt_ids) <= max_len and len(en_ids) <= max_len:
                pairs.append((pt_ids, en_ids));
                kept += 1
            else:
                skipped += 1
        logger.info(f"[filter] kept={kept}, skipped={skipped}, max_length={max_len}")
        return pairs

    train_pairs = build_filtered_pairs(train_dataset, pt_tokenizer, en_tokenizer, max_length)
    val_pairs = build_filtered_pairs(val_dataset, pt_tokenizer, en_tokenizer, max_length)

    # 3) Dataset 类
    class PairsDataset(Dataset):
        def __init__(self, pairs): self.pairs = pairs

        def __len__(self): return len(self.pairs)

        def __getitem__(self, idx):
            pt_ids, en_ids = self.pairs[idx]
            return {"pt_input_ids": pt_ids, "en_input_ids": en_ids}

    # 4) Collate 函数（动态 padding）
    def collate_padded(batch, pad_id_pt: int, pad_id_en: int):
        def pad_block(seqs, pad_value):
            max_len = max(len(s) for s in seqs)
            out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
            attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                L = len(s)
                out[i, :L] = torch.tensor(s, dtype=torch.long)
                attn[i, :L] = 1
            return out, attn

        pt_ids_list = [ex["pt_input_ids"] for ex in batch]
        en_ids_list = [ex["en_input_ids"] for ex in batch]
        pt_input_ids, pt_attention_mask = pad_block(pt_ids_list, pt_tokenizer.pad_token_id)
        en_input_ids, en_attention_mask = pad_block(en_ids_list, en_tokenizer.pad_token_id)

        return {
            "pt_input_ids": pt_input_ids,
            "pt_attention_mask": pt_attention_mask,
            "en_input_ids": en_input_ids,
            "en_attention_mask": en_attention_mask,
        }

    # 5) DataLoader
    train_loader = DataLoader(
        PairsDataset(train_pairs),
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=lambda b: collate_padded(b, pt_tokenizer.pad_token_id, en_tokenizer.pad_token_id),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        PairsDataset(val_pairs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_padded(b, pt_tokenizer.pad_token_id, en_tokenizer.pad_token_id),
        num_workers=num_workers,
    )

    return train_loader, val_loader


def test_dataloaders(train_loader, val_loader, show_val: bool = True):
    """
    检查 DataLoader 的 batch 输出，打印张量形状和一个示例

    参数:
        train_loader: 训练 DataLoader
        val_loader: 验证 DataLoader
        show_val: 是否展示验证集的一个样本（默认 True）
    """
    # 1. 拿一个训练 batch 看 shape
    batch = next(iter(train_loader))
    logger.info("=== Train Loader Batch Shapes ===")
    for k, v in batch.items():
        logger.info(f"{k:20s} {tuple(v.shape)}")

    # 2. 验证集样本
    if show_val:
        logger.info("\n=== Validation Loader Example ===")
        for i in val_loader:
            logger.info("pt_input_ids:     ", i["pt_input_ids"][0])
            logger.info("pt_attention_mask:", i["pt_attention_mask"][0])
            break


def get_position_embedding(sentence_length: int, d_model: int, device="cuda", dtype=torch.float32):
    """
    返回 position 对应的 embedding 矩阵
    形状: [1, sentence_length, d_model]
    """

    def get_angles(pos: torch.Tensor, i: torch.Tensor, d_model: int):
        """
        获取单词 pos 对应 embedding 的角度
        pos: [sentence_length, 1]
        i  : [1, d_model]
        return: [sentence_length, d_model]
        """
        angle_rates = 1.0 / torch.pow(
            10000,
            (2 * torch.div(i, 2, rounding_mode='floor')).float() / d_model
        )
        return pos.float() * angle_rates

    if device is None:
        device = torch.device("cpu")

    pos = torch.arange(sentence_length, device=device).unsqueeze(1)  # [L, 1]
    i = torch.arange(d_model, device=device).unsqueeze(0)  # [1, D]

    angle_rads = get_angles(pos, i, d_model)  # [L, D]

    # 偶数下标：sin
    sines = torch.sin(angle_rads[:, 0::2])
    # 奇数下标：cos
    cosines = torch.cos(angle_rads[:, 1::2])

    # 拼接还原成 [L, D]
    position_embedding = torch.zeros((sentence_length, d_model), device=device, dtype=dtype)
    position_embedding[:, 0::2] = sines
    position_embedding[:, 1::2] = cosines

    # 增加 batch 维度 [1, L, D]
    position_embedding = position_embedding.unsqueeze(0)

    return position_embedding


def plot_position_embedding(position_embedding: torch.Tensor):
    """
    可视化位置编码矩阵
    参数:
        position_embedding: [1, L, D] 的张量
    """
    # 转到 CPU，并转成 numpy
    pe = position_embedding.detach().cpu().numpy()[0]  # [L, D]

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pe, cmap='RdBu')  # L × D 矩阵
    plt.xlabel("Depth (d_model)")
    plt.xlim((0, pe.shape[1]))
    plt.ylabel("Position (pos)")
    plt.colorbar()
    plt.title("Positional Encoding Visualization")
    plt.show()


def create_padding_mask(batch_data: torch.Tensor, pad_token_id: int = 0):
    """
    输入:
        batch_data: [batch_size, seq_len]，填充位置用 pad_token_id 表示
        pad_token_id: 默认是 0
    输出:
        padding_mask: [batch_size, 1, 1, seq_len]
    """
    # 等价于 tf.math.equal(batch_data, 0)
    mask = (batch_data == pad_token_id).float()
    # 插入维度
    return mask[:, None, None, :]  # [B, 1, 1, L]


def create_look_ahead_mask(size: int):
    """
    生成 Look-ahead mask (上三角矩阵)
    参数:
        size: 序列长度 (seq_len)
    返回:
        mask: [seq_len, seq_len]，上三角为 1，其他为 0
    """
    # ones: [size, size]
    ones = torch.ones((size, size))
    # 取上三角（不含对角线）=1，下三角和对角线=0
    mask = torch.triu(ones, diagonal=1)
    return mask


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
        q: (..., seq_len_q, depth)
        k: (..., seq_len_k, depth)
        v: (..., seq_len_v, depth_v)  (seq_len_k == seq_len_v)
        mask: (..., seq_len_q, seq_len_k)，
              mask里1表示要忽略的位置，0表示保留。

    Returns:
        output: (..., seq_len_q, depth_v) 加权和
        attention_weights: (..., seq_len_q, seq_len_k) 注意力权重
    """
    # (..., seq_len_q, seq_len_k)
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))

    # 缩放
    dk = q.size()[-1]
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=q.device))

    # 加上 mask
    if mask is not None:
        # 在 mask==1 的位置加上 -1e9，使 softmax 后趋近于0
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 1, -1e9)

    # softmax 得到注意力权重
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    # 加权求和
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    PyTorch 版 MHA，与 Keras 版本对应：
      q -> WQ -> 分头
      k -> WK -> 分头
      v -> WV -> 分头
      计算 scaled dot-product attention
      合并 -> 线性层
    期望输入形状：
      q, k, v: [B, L, d_model]
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # 每个头的维度 Dh

        # 对应 Keras 的 Dense(d_model)
        self.WQ = nn.Linear(d_model, d_model, bias=True)
        self.WK = nn.Linear(d_model, d_model, bias=True)
        self.WV = nn.Linear(d_model, d_model, bias=True)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor):
        """
        x: [B, L, d_model] -> [B, num_heads, L, depth]
        """
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.depth)  # [B, L, H, Dh]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, L, Dh]
        return x

    def _combine_heads(self, x: torch.Tensor):
        """
        x: [B, num_heads, L, depth] -> [B, L, d_model]
        """
        B, H, L, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, H, Dh]
        x = x.view(B, L, H * Dh)  # [B, L, d_model]
        return x

    def forward(self, q, k, v, mask=None, return_attn: bool = True):
        """
        q, k, v: [B, Lq/Lk/Lv, d_model]
        mask: 期望形状为 [B, 1, Lq, Lk] 或 [B, Lq, Lk]；值为1表示屏蔽，0表示保留
        return:
          output: [B, Lq, d_model]
          attention_weights (可选): [B, num_heads, Lq, Lk]
        """
        B = q.size(0)

        # 线性映射
        q = self.WQ(q)  # [B, Lq, d_model]
        k = self.WK(k)  # [B, Lk, d_model]
        v = self.WV(v)  # [B, Lv, d_model]

        # 分头
        q = self._split_heads(q)  # [B, H, Lq, Dh]
        k = self._split_heads(k)  # [B, H, Lk, Dh]
        v = self._split_heads(v)  # [B, H, Lv, Dh]

        # 处理 mask：广播到 [B, H, Lq, Lk]
        if mask is not None:
            # 允许 [B, 1, Lq, Lk] 或 [B, Lq, Lk]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B,1,Lq,Lk]
            elif mask.dim() == 4 and mask.size(1) == 1:
                pass  # 已是 [B,1,Lq,Lk]
            else:
                raise ValueError("mask 形状需为 [B, Lq, Lk] 或 [B, 1, Lq, Lk]")
            mask = mask.expand(B, self.num_heads, mask.size(-2), mask.size(-1))

        # 注意力
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask)  # [B,H,Lq,Dh], [B,H,Lq,Lk]

        # 合并头
        attn_out = self._combine_heads(attn_out)  # [B, Lq, d_model]

        # 输出线性层
        output = self.out_proj(attn_out)  # [B, Lq, d_model]

        if return_attn:
            return output, attn_weights
        return output


def feed_forward_network(d_model, dff):
    """
    前馈网络 FFN
    Args:
        d_model: 输出维度 (embedding 维度)
        dff: 内部隐层维度 (feed-forward 网络的中间层大小)
    Returns:
        nn.Sequential 模型
    """
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )


class EncoderLayer(nn.Module):
    """
    x -> self-attention -> add & norm & dropout
      -> feed-forward   -> add & norm & dropout
    期望输入:
      x: [B, L, d_model]
      src_mask: [B, 1, L, L] 或 [B, L, L]，其中 1 表示屏蔽，0 表示保留
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)  # 前面已实现
        self.ffn = feed_forward_network(d_model, dff)  # 前面已实现

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        """
        返回:
          out: [B, L, d_model]
        """
        # Self-Attention
        attn_out, _ = self.mha(x, x, x, mask=src_mask)  # [B, L, d_model], [B, H, L, L]
        attn_out = self.dropout1(attn_out)  # 训练模式下生效
        out1 = self.norm1(x + attn_out)  # 残差 + LayerNorm

        # Feed Forward
        ffn_out = self.ffn(out1)  # [B, L, d_model]
        ffn_out = self.dropout2(ffn_out)
        out2 = self.norm2(out1 + ffn_out)

        return out2


class DecoderLayer(nn.Module):
    """
    x -> masked self-attention -> add & norm & dropout -> out1
    out1, enc_out -> cross-attention -> add & norm & dropout -> out2
    out2 -> FFN -> add & norm & dropout -> out3
    期望输入:
      x: [B, L_tgt, d_model]
      enc_out: [B, L_src, d_model]
      tgt_mask: [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]  (look-ahead + padding 的合并掩码，1=屏蔽)
      enc_dec_mask: [B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]  (decoder 对 encoder 的 padding 掩码，1=屏蔽)
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # masked self-attn
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # cross-attn

        self.ffn = feed_forward_network(d_model, dff)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(
            self,
            x: torch.Tensor,
            enc_out: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            enc_dec_mask: torch.Tensor = None,
    ):
        # 1) Masked Self-Attention (decoder 自注意力，使用 look-ahead+padding 的合并 mask)
        attn1_out, attn_weights1 = self.mha1(x, x, x, mask=tgt_mask)  # [B,Lt,D], [B,H,Lt,Lt]
        attn1_out = self.dropout1(attn1_out)
        out1 = self.norm1(x + attn1_out)

        # 2) Cross-Attention (query=out1, key/value=enc_out)，使用 encoder padding 掩码
        attn2_out, attn_weights2 = self.mha2(out1, enc_out, enc_out, mask=enc_dec_mask)  # [B,Lt,D], [B,H,Lt,Ls]
        attn2_out = self.dropout2(attn2_out)
        out2 = self.norm2(out1 + attn2_out)

        # 3) FFN
        ffn_out = self.ffn(out2)  # [B,Lt,D]
        ffn_out = self.dropout3(ffn_out)
        out3 = self.norm3(out2 + ffn_out)  # [B,Lt,D]

        return out3, attn_weights1, attn_weights2


class EncoderModel(nn.Module):
    def __init__(self, num_layers: int, input_vocab_size: int, max_length: int,
                 d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 padding_idx: int = None):
        """
        参数与 Keras 版本对齐；额外提供 padding_idx 以便 Embedding 忽略 pad 的梯度。
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        # Embedding
        self.embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=padding_idx)

        # 位置编码：注册为 buffer（不参与训练/优化器）
        pe = get_position_embedding(max_length, d_model)  # [1, max_len, d_model]
        self.register_buffer("position_embedding", pe, persistent=False)

        self.dropout = nn.Dropout(rate)

        # 堆叠 EncoderLayer（前面我们已实现过）
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )

        # 预存缩放因子
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        """
        x: [B, L]  （token ids）
        src_mask: [B, 1, L, L] 或 [B, L, L]，1=屏蔽，0=保留（与前文一致）
        return: 编码结果 [B, L, d_model]
        """
        B, L = x.shape
        # 等价于 tf.debugging.assert_less_equal
        if L > self.max_length:
            raise ValueError(f"input_seq_len ({L}) should be ≤ max_length ({self.max_length})")

        # [B, L, D]
        x = self.embedding(x)
        # 缩放：使 embedding 的尺度与位置编码相近（论文做法）
        x = x * self.scale
        # 加位置编码（按实际序列长度切片）
        x = x + self.position_embedding[:, :L, :]

        x = self.dropout(x)

        # 逐层 Encoder
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x


class DecoderModel(nn.Module):
    """
    x -> masked self-attn -> add & norm & dropout
      -> cross-attn(enc_out) -> add & norm & dropout
      -> FFN -> add & norm & dropout
    """

    def __init__(self, num_layers: int, target_vocab_size: int, max_length: int,
                 d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 padding_idx: int = None):
        super().__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model

        # 词嵌入
        self.embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=padding_idx)

        # 位置编码（注册为 buffer，不参与训练）
        pe = get_position_embedding(max_length, d_model)
        self.register_buffer("position_embedding", pe, persistent=False)

        self.dropout = nn.Dropout(rate)

        # 堆叠解码层
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )

        self.scale = math.sqrt(d_model)

    def forward(
            self,
            x: torch.Tensor,  # [B, L_tgt] 目标端 token ids
            enc_out: torch.Tensor,  # [B, L_src, D] 编码器输出
            tgt_mask: torch.Tensor = None,  # [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]（look-ahead+padding）
            enc_dec_mask: torch.Tensor = None,  # [B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]（对 encoder 的 padding）
    ):
        B, Lt = x.shape
        if Lt > self.max_length:
            raise ValueError(f"output_seq_len ({Lt}) should be ≤ max_length ({self.max_length})")

        # (B, Lt, D)
        x = self.embedding(x) * self.scale
        x = x + self.position_embedding[:, :Lt, :]
        x = self.dropout(x)

        attention_weights = {}

        for i, layer in enumerate(self.decoder_layers, start=1):
            x, attn1, attn2 = layer(x, enc_out, tgt_mask=tgt_mask, enc_dec_mask=enc_dec_mask)
            attention_weights[f"decoder_layer{i}_att1"] = attn1  # [B, H, Lt, Lt]
            attention_weights[f"decoder_layer{i}_att2"] = attn2  # [B, H, Lt, Ls]

        # x: (B, Lt, D)
        return x, attention_weights


class Transformer(nn.Module):
    def __init__(self, num_layers, input_vocab_size, target_vocab_size,
                 max_length, d_model, num_heads, dff, rate=0.1,
                 src_padding_idx: int = None, tgt_padding_idx: int = None):
        super().__init__()
        self.encoder_model = EncoderModel(
            num_layers=num_layers,
            input_vocab_size=input_vocab_size,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=rate,
            padding_idx=src_padding_idx,
        )
        self.decoder_model = DecoderModel(
            num_layers=num_layers,
            target_vocab_size=target_vocab_size,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=rate,
            padding_idx=tgt_padding_idx,
        )
        # 等价于 Keras 的 Dense(target_vocab_size)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp_ids, tgt_ids, src_mask=None, tgt_mask=None, enc_dec_mask=None):
        """
        inp_ids: [B, L_src]  源端 token ids
        tgt_ids: [B, L_tgt]  目标端 token ids（训练时通常是 shift 后的 decoder 输入）
        src_mask:    [B, 1, L_src, L_src] 或 [B, L_src, L_src]（1=屏蔽）
        tgt_mask:    [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]（look-ahead+padding）
        enc_dec_mask:[B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]
        返回:
          logits: [B, L_tgt, target_vocab_size]
          attention_weights: dict，包含每层的 attn
        """
        enc_out = self.encoder_model(inp_ids, src_mask=src_mask)  # [B, L_src, D]
        dec_out, attention_weights = self.decoder_model(
            tgt_ids, enc_out, tgt_mask=tgt_mask, enc_dec_mask=enc_dec_mask
        )  # [B, L_tgt, D], dict
        logits = self.final_layer(dec_out)  # [B, L_tgt, V_tgt]
        return logits, attention_weights


class CustomizedSchedule(_LRScheduler):
    """
    Noam / Transformer LR:
      lr = d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)  # 确保从 1 开始
        scale = self.d_model ** -0.5
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = scale * min(arg1, arg2)
        return [lr for _ in self.base_lrs]


def plot_customized_lr_curve(optimizer, scheduler, total_steps: int, label: str = None):
    """
    绘制学习率曲线（支持传入已有 optimizer 和 scheduler）

    Args:
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器
        total_steps (int): 总训练步数
        label (str): 图例标签，默认使用 scheduler 配置
    """
    lrs = []
    for step in range(total_steps):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)

    # 绘制曲线
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, total_steps + 1), lrs, label=label or "LR Curve")
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.show()


def loss_function(real, pred):
    """
    Args:
        real: (B, L) target ids (shift 后)
        pred: (B, L, V) logits
    Returns:
        loss (float): 平均有效 token 的交叉熵损失
    """
    B, L, V = pred.shape

    # 展平
    pred = pred.reshape(-1, V)  # (B*L, V)
    real = real.reshape(-1)  # (B*L,)

    # token 级别交叉熵 (padding 已被 ignore_index 屏蔽)
    loss_ = loss_object(pred, real)  # (B*L,)

    # # 统计有效 token
    # valid = (real != PAD_ID_TGT).float()

    # # 均值损失（只对有效 token 求平均）
    # loss = (loss_ * valid).sum() / valid.sum()
    return loss_.mean()


def create_masks(
        inp_ids: torch.Tensor,  # [B, L_src]
        tar_ids: torch.Tensor,  # [B, L_tgt] —— 通常是 decoder 输入（已左移）
        src_pad_id: int = 0,
        tgt_pad_id: int = 0,
):
    """
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


@torch.no_grad()
def token_accuracy(real, pred, pad_id):
    pred_ids = pred.argmax(dim=-1)  # (B, L)
    mask = (real != pad_id)
    correct = ((pred_ids == real) & mask).sum().item()
    denom = mask.sum().item()
    return correct / max(1, denom)


class AverageMeter:
    def __init__(self, name="meter"): self.name = name; self.reset()

    def reset(self): self.sum = 0.0; self.n = 0

    def update(self, val, count=1): self.sum += float(val) * count; self.n += count

    @property
    def avg(self): return self.sum / max(1, self.n)


def train_step(batch, transformer, optimizer, scheduler=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer.train()

    inp = batch["pt_input_ids"].to(device)
    tar = batch["en_input_ids"].to(device)

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    SRC_PAD_ID = pt_tokenizer.pad_token_id
    TGT_PAD_ID = en_tokenizer.pad_token_id

    enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
        inp, tar_inp, src_pad_id=SRC_PAD_ID, tgt_pad_id=TGT_PAD_ID
    )
    enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, tar_inp.size(1), -1)

    logits, _ = transformer(
        inp, tar_inp,
        src_mask=enc_pad_mask,
        tgt_mask=dec_mask,
        enc_dec_mask=enc_dec_mask
    )

    loss = loss_function(tar_real, logits)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    acc = token_accuracy(tar_real, logits, pad_id=TGT_PAD_ID)
    return loss.item(), acc


def train_model(
        epochs: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        scheduler=None,
        device: str = None,
        log_every: int = 100,
        ckpt_dir: str = "checkpoints",
        ckpt_prefix: str = "ckpt",
):
    os.makedirs(ckpt_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loss_meter = AverageMeter("train_loss")
    train_acc_meter = AverageMeter("train_accuracy")
    global_step = 0

    for epoch in range(epochs):
        try:
            start = time.time()
            train_loss_meter.reset()
            train_acc_meter.reset()
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                loss_val, acc_val = train_step(
                    batch=batch, transformer=model, optimizer=optimizer, scheduler=scheduler, device=device
                )
                train_loss_meter.update(loss_val, 1)
                train_acc_meter.update(acc_val, 1)

                global_step += 1
                if batch_idx % log_every == 0:
                    logger.info(
                        f"Epoch {epoch + 1} Batch {batch_idx} global_step {global_step}"
                        f"Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f}"
                    )
                    save_ckpt(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        step=global_step,
                        ckpt_dir=ckpt_dir,
                        tag="latest"
                    )

            logger.info(f"Epoch {epoch + 1} Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f}")
            logger.info(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

            # 每个epoch结束后进行验证集评测
            validate_loss, validate_acc = evaluate_on_val(model, val_loader, device)
            logger.info(f"Validation - Epoch {epoch + 1} Loss: {validate_loss:.4f}, Accuracy: {validate_acc:.4f}\n")

        except Exception as e:
            logger.info(f"报错啦!!! 报错信息: {e}")
            save_ckpt(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                tag="error"
            )


@torch.no_grad()
def evaluate(
        inp_sentence: str,
        transformer: Transformer,
        pt_tokenizer,
        en_tokenizer,
        max_length: int,
        device: str = None):
    """
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


@torch.no_grad()
def evaluate_on_val(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_count = 0

    for batch in val_loader:
        inp = batch["pt_input_ids"].to(device)
        tar = batch["en_input_ids"].to(device)

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
            inp, tar_inp, src_pad_id=pt_tokenizer.pad_token_id, tgt_pad_id=en_tokenizer.pad_token_id
        )
        enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, tar_inp.size(1), -1)

        logits, _ = model(
            inp, tar_inp,
            src_mask=enc_pad_mask,
            tgt_mask=dec_mask,
            enc_dec_mask=enc_dec_mask
        )

        loss = loss_function(tar_real, logits)
        acc = token_accuracy(tar_real, logits, pad_id=en_tokenizer.pad_token_id)

        total_loss += loss.item() * inp.size(0)
        total_acc += acc * inp.size(0)
        total_count += inp.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc


def plot_encoder_decoder_attention(attention, input_sentence, result, layer_name):
    """
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
              max_length=64, device=None, layer_name=""):
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
    if layer_name:
        plot_encoder_decoder_attention(
            attention_weights,
            input_sentence,
            result,
            layer_name
        )

    return predicted_sentence


def save_ckpt(model, optimizer, scheduler, epoch, step, ckpt_dir="checkpoints", tag="latest"):
    """
    保存 checkpoint
    Args:
        model: nn.Module
        optimizer: torch.optim
        scheduler: torch.optim.lr_scheduler (可选)
        epoch: 当前 epoch
        step: 全局 step
        ckpt_dir: 保存目录
        tag: 保存标识 ("latest", "error", "custom" 等)
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict() if scheduler else None,
    }

    latest_path = os.path.join(ckpt_dir, "latest.pt")
    torch.save(ckpt, latest_path)
    # logger.info(f"✅ checkpoint updated: {latest_path}")

    # 1. 默认保存 latest
    if tag == "latest":
        path = os.path.join(ckpt_dir, f"mid_e{epoch}_s{step}.pt")

    elif tag == "error":
        # 避免覆盖，用时间戳
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ckpt_dir, f"error_e{epoch}_s{step}_{ts}.pt")
    else:
        path = os.path.join(ckpt_dir, f"{tag}_e{epoch}_s{step}.pt")

    torch.save(ckpt, path)
    # logger.info(f"✅ checkpoint saved: {path}")
    return path


def load_ckpt(model, optimizer=None, scheduler=None, ckpt_dir="checkpoints", device="cpu"):
    """
    加载最新 checkpoint
    """
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(latest):
        logger.info("⚠️ No checkpoint found, training from scratch.")
        return 0, 0
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer: optimizer.load_state_dict(ckpt["optim"])
    if scheduler and ckpt["sched"]: scheduler.load_state_dict(ckpt["sched"])
    logger.info(f"✅ checkpoint loaded (epoch={ckpt['epoch']}, step={ckpt['step']})")
    return ckpt["epoch"], ckpt["step"]


if __name__ == "__main__":
    # 0. 常量定义

    # 数据文件地址
    train_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_train.csv"
    val_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_test.csv"
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    checkpoint_dir = './checkpoints-tmp6'

    # 构建词表参数
    vocab_size = 2 ** 13  # 词表大小
    min_freq = 2  # 最小词频
    special_tokens = special_tokens  # 特殊符号
    max_length = 30  # 最大序列长度

    # 模型训练超参数
    batch_size = 32  # 批处理数
    # warmup_steps = 4000           # warmup steps数
    epochs = 600  # 训练轮数
    # learning_rate = 1.0           # 学习率
    # betas = (0.9, 0.98)           # Adam 的一阶矩（梯度均值）；二阶矩（梯度平方的均值）
    # eps = 1e-9                    # 防止除零错误的小常数
    learning_rate = 5e-4
    betas = (0.9, 0.98)
    eps = 1e-8
    weight_decay = 2e-5  # L2正则化(（权重衰减）) - 0.01

    # 模型结构
    # num_layers = 8
    # d_model = 512                 # hidden-size
    # dff = 2048
    # num_heads = 8
    # dropout_rate = 0.1

    num_layers = 4
    d_model = 128  # hidden-size
    dff = 512
    num_heads = 8
    dropout_rate = 0.25

    # 1. 检查 PyTorch 环境信息、GPU 状态，以及常用依赖库版本；
    device = check_env()
    logger.info("实际使用设备:", device)

    # 2. 加载葡萄牙语-英语翻译数据集
    train_dataset, val_dataset = load_translation_dataset(
        train_path=train_path,
        val_path=val_path
    )
    logger.info("训练集样本数:", len(train_dataset))
    logger.info("验证集样本数:", len(val_dataset))

    # 3. 构建 Tokenizer
    # 3.1 构建 Tokenizer
    logger.info("开始构建 Tokenizer...")
    pt_tokenizer, en_tokenizer = train_and_load_tokenizers(
        train_dataset=train_dataset,  # 数据集
        pt_key="pt",  # 葡语列名
        en_key="en",  # 英语列名
        vocab_size=vocab_size,  # 词表大小
        min_freq=min_freq,  # 最小词频
        special_tokens=special_tokens,  # 特殊符号
        save_dir_pt="tok_pt",  # 保存目录 (pt)
        save_dir_en="tok_en",  # 保存目录 (en)
        max_length=max_length  # 最大序列长度
    )

    # 3.2 【测试】 Tokenizer 代码
    test_tokenizers(en_tokenizer=en_tokenizer, pt_tokenizer=pt_tokenizer)

    # 3.3 构建 batch data loader
    logger.info("开始构建 batch data loader...")
    train_loader2, val_loader2 = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pt_tokenizer=pt_tokenizer,
        en_tokenizer=en_tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=0,
        shuffle_train=True
    )

    # 3.4 【测试】 batch data loader
    test_dataloaders(train_loader2, val_loader2)

    # 4. 位置编码
    # 4.2 【测试】 - 打印位置编码矩阵图形
    position_embedding = get_position_embedding(max_length, d_model)
    plot_position_embedding(position_embedding)

    # 5. 构建 model 模型 Transformer 结构
    input_vocab_size = pt_tokenizer.vocab_size
    target_vocab_size = en_tokenizer.vocab_size

    model = Transformer(
        num_layers=num_layers,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        rate=dropout_rate,
        src_padding_idx=pt_tokenizer.pad_token_id if hasattr(pt_tokenizer, "pad_token_id") else None,
        tgt_padding_idx=en_tokenizer.pad_token_id if hasattr(en_tokenizer, "pad_token_id") else None,
    )

    ##############################【Test - optimizer | scheduler 】##############################
    # # 6. 自定义学习率和优化器
    # optimizer = optim.Adam(model.parameters(),
    #                    lr=learning_rate,
    #                    betas=betas,
    #                    eps=eps)
    # # 自定义学习率
    # scheduler = CustomizedSchedule(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    # 6. 自定义学习率和优化器
    num_training_steps = len(train_loader2) * epochs

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )

    warmup_steps = int(0.1 * num_training_steps)  # 10% 步数用作 warmup
    # 获取学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )

    # 自定义学习率
    # num_training_steps = len(train_loader2) * epochs
    # # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    # #     optimizer,
    # #     T_max=num_training_steps,
    # #     eta_min=1e-6
    # # )
    # # 设置 warmup steps
    # warmup_steps = int(0.1 * num_training_steps)  # 10% 步数用作 warmup
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=num_training_steps,
    # )

    # 6.2 【测试】 打印自定义学习率曲线
    plot_customized_lr_curve(optimizer, scheduler, total_steps=num_training_steps,
                             label=f"d_model={d_model}, warmup={warmup_steps}")

    ##############################【Test - optimizer | scheduler 】##############################

    # 7. 自定义损失函数
    # PyTorch 的 CrossEntropyLoss 默认就支持 from_logits=True
    PAD_ID_TGT = en_tokenizer.pad_token_id
    loss_object = nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_ID_TGT)

    # 8. 训练模型 && checkpoints
    logger.info(f"learning_rate:{learning_rate}")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

        train_model(
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader2,
            val_loader=val_loader2,
            scheduler=scheduler,  # Noam 调度
            device=device,  # 自动选 GPU/CPU
            log_every=100,
            ckpt_dir="checkpoints",
            ckpt_prefix="transformer",
        )
    else:
        start_epoch, global_step = load_ckpt(model, optimizer, scheduler, device=device)
        logger.info("Checkpoint loaded successfully!")

