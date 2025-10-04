# -*- coding: utf-8 -*-
"""
Step 3: 训练和测试Tokenizer
训练并加载葡萄牙语和英语的 ByteLevel BPE Tokenizer
"""

import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from loguru import logger

def load_translation_dataset(train_path: str, val_path: str, delimiter: str = "\t"):
    """加载数据集"""
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_path,
            "validation": val_path
        },
        column_names=["pt", "en"],
        delimiter=delimiter
    )
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
    logger.info("开始训练葡萄牙语 tokenizer...")
    pt_bbpe.train_from_iterator(
        iter_lang(train_dataset, pt_key),
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )
    
    logger.info("开始训练英语 tokenizer...")
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

    logger.info(f"pt vocab size: {len(pt_tokenizer)}")
    logger.info(f"en vocab size: {len(en_tokenizer)}")

    return pt_tokenizer, en_tokenizer

def test_tokenizers(en_tokenizer, pt_tokenizer,
                    en_sample: str = "Transformer is awesome.",
                    pt_sample: str = "Transformers são incríveis."):
    """
    测试英文和葡萄牙语的 tokenizer 编码/解码是否正确，
    并打印 token IDs、单个 ID 对应的 token 结果。
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

if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: 训练和测试Tokenizer")
    print("=" * 60)
    
    # 数据文件地址
    train_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_train.csv"
    val_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_test.csv"
    
    # 构建词表参数
    vocab_size = 2 ** 13  # 词表大小
    min_freq = 2  # 最小词频
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]  # 特殊符号
    max_length = 30  # 最大序列长度
    
    print(f"🔧 Tokenizer参数:")
    print(f"   词表大小: {vocab_size}")
    print(f"   最小词频: {min_freq}")
    print(f"   最大序列长度: {max_length}")
    print(f"   特殊符号: {special_tokens}")
    
    try:
        # 1. 加载数据集
        print(f"\n📁 加载数据集...")
        train_dataset, val_dataset = load_translation_dataset(
            train_path=train_path,
            val_path=val_path
        )
        
        # 2. 构建 Tokenizer
        print(f"\n🔨 开始构建 Tokenizer...")
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
        
        print(f"\n✅ Tokenizer构建完成！")
        print(f"📊 葡萄牙语词表大小: {len(pt_tokenizer)}")
        print(f"📊 英语词表大小: {len(en_tokenizer)}")
        
        # 3. 测试 Tokenizer
        print(f"\n🧪 测试 Tokenizer...")
        test_tokenizers(en_tokenizer=en_tokenizer, pt_tokenizer=pt_tokenizer)
        
        print(f"\n✅ Tokenizer测试完成！")
        
    except Exception as e:
        print(f"❌ Tokenizer构建失败: {e}")
        print("💡 请检查数据文件路径是否正确")

"""
(train_transformers) root@iv-ydg6wcq3ggay8n6dmn75:/data2/workspace/yszhang/train_transformers/run_by_steps/run_by_steps# python step3_tokenizer.py 
============================================================
Step 3: 训练和测试Tokenizer
============================================================
🔧 Tokenizer参数:
   词表大小: 8192
   最小词频: 2
   最大序列长度: 30
   特殊符号: ['<s>', '<pad>', '</s>', '<unk>', '<mask>']

📁 加载数据集...

🔨 开始构建 Tokenizer...
2025-10-04 17:42:38.086 | INFO     | __main__:train_and_load_tokenizers:68 - 开始训练葡萄牙语 tokenizer...
[00:00:00] Pre-processing sequences       ██████████████████████████████████████████████████████████████████████████████ 0        /        0
[00:00:00] Tokenize words                 ██████████████████████████████████████████████████████████████████████████████ 38341    /    38341
[00:00:00] Count pairs                    ██████████████████████████████████████████████████████████████████████████████ 38341    /    38341
[00:00:03] Compute merges                 ██████████████████████████████████████████████████████████████████████████████ 7931     /     7931
2025-10-04 17:42:42.074 | INFO     | __main__:train_and_load_tokenizers:76 - 开始训练英语 tokenizer...
[00:00:00] Pre-processing sequences       ██████████████████████████████████████████████████████████████████████████████ 0        /        0
[00:00:00] Tokenize words                 ██████████████████████████████████████████████████████████████████████████████ 27827    /    27827
[00:00:00] Count pairs                    ██████████████████████████████████████████████████████████████████████████████ 27827    /    27827
[00:00:02] Compute merges                 ██████████████████████████████████████████████████████████████████████████████ 7931     /     7931
2025-10-04 17:42:45.475 | INFO     | __main__:train_and_load_tokenizers:106 - pt vocab size: 8192
2025-10-04 17:42:45.476 | INFO     | __main__:train_and_load_tokenizers:107 - en vocab size: 8192

✅ Tokenizer构建完成！
📊 葡萄牙语词表大小: 8192
📊 英语词表大小: 8192

🧪 测试 Tokenizer...
2025-10-04 17:42:45.481 | INFO     | __main__:test_tokenizers:120 - === English Tokenizer Test ===
2025-10-04 17:42:45.481 | INFO     | __main__:test_tokenizers:122 - [EN] Tokenized IDs: [56, 7815, 1224, 274, 317, 4029, 18]
2025-10-04 17:42:45.481 | INFO     | __main__:test_tokenizers:129 - [EN] Decoded string: Transformer is awesome.
2025-10-04 17:42:45.481 | INFO     | __main__:test_tokenizers:132 - [EN] id --> decoded([id])  |  id --> token(str)
2025-10-04 17:42:45.481 | INFO     | __main__:test_tokenizers:140 -     56 --> 'T'  |      56 --> 'T'
2025-10-04 17:42:45.481 | INFO     | __main__:test_tokenizers:140 -   7815 --> 'rans'  |    7815 --> 'rans'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:140 -   1224 --> 'form'  |    1224 --> 'form'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:140 -    274 --> 'er'  |     274 --> 'er'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:140 -    317 --> ' is'  |     317 --> 'Ġis'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:140 -   4029 --> ' awesome'  |    4029 --> 'Ġawesome'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:140 -     18 --> '.'  |      18 --> '.'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:142 - 
------------------------------------------------------------

2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:145 - === Portuguese Tokenizer Test ===
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:147 - [PT] Tokenized IDs: [56, 433, 87, 3004, 3365, 453, 4083, 18]
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:154 - [PT] Decoded string: Transformers são incríveis.
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:157 - [PT] id --> decoded([id])  |  id --> token(str)
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:165 -     56 --> 'T'  |      56 --> 'T'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:165 -    433 --> 'ran'  |     433 --> 'ran'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:165 -     87 --> 's'  |      87 --> 's'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:165 -   3004 --> 'form'  |    3004 --> 'form'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:165 -   3365 --> 'ers'  |    3365 --> 'ers'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:165 -    453 --> ' são'  |     453 --> 'ĠsÃ£o'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:165 -   4083 --> ' incríveis'  |    4083 --> 'ĠincrÃŃveis'
2025-10-04 17:42:45.482 | INFO     | __main__:test_tokenizers:165 -     18 --> '.'  |      18 --> '.'

✅ Tokenizer测试完成！
(train_transformers) root@iv-ydg6wcq3ggay8n6dmn75:/data2/workspace/yszhang/train_transformers/run_by_steps/run_by_steps# 
"""