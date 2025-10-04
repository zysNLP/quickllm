# -*- coding: utf-8 -*-
"""
Step 2: 数据加载和数据集信息
加载葡萄牙语-英语翻译数据集 (TED Talks)
"""

import os
from datasets import load_dataset
from loguru import logger

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

    logger.info(f"数据集类型: {type(dataset)}")
    logger.info(dataset)

    # 打印一个样本
    sample = dataset["train"][0]
    logger.info(f"示例数据 -> pt: {sample['pt']} | en: {sample['en']}")

    return dataset["train"], dataset["validation"]

if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: 数据加载和数据集信息")
    print("=" * 60)
    
    # 数据文件地址
    train_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_train.csv"
    val_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_test.csv"
    
    print(f"📁 训练集路径: {train_path}")
    print(f"📁 验证集路径: {val_path}")
    
    try:
        # 加载数据集
        train_dataset, val_dataset = load_translation_dataset(
            train_path=train_path,
            val_path=val_path
        )
        
        print(f"\n✅ 数据加载完成！")
        print(f"📊 训练集样本数: {len(train_dataset)}")
        print(f"📊 验证集样本数: {len(val_dataset)}")
        
        # 显示更多样本
        print(f"\n📝 数据集样本预览:")
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            print(f"  样本 {i+1}:")
            print(f"    葡萄牙语: {sample['pt']}")
            print(f"    英语: {sample['en']}")
            print()
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("💡 请检查数据文件路径是否正确")

"""
(train_transformers) root@iv-ydg6wcq3ggay8n6dmn75:/data2/workspace/yszhang/train_transformers/run_by_steps/run_by_steps# python step2_data_loading.py 
============================================================
Step 2: 数据加载和数据集信息
============================================================
📁 训练集路径: /data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_train.csv
📁 验证集路径: /data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_test.csv
2025-10-04 17:41:07.332 | INFO     | __main__:load_translation_dataset:23 - 开始加载数据...
2025-10-04 17:41:10.454 | INFO     | __main__:load_translation_dataset:34 - 数据集类型: <class 'datasets.dataset_dict.DatasetDict'>
2025-10-04 17:41:10.454 | INFO     | __main__:load_translation_dataset:35 - DatasetDict({
    train: Dataset({
        features: ['pt', 'en'],
        num_rows: 51786
    })
    validation: Dataset({
        features: ['pt', 'en'],
        num_rows: 1194
    })
})
2025-10-04 17:41:10.455 | INFO     | __main__:load_translation_dataset:39 - 示例数据 -> pt: pt | en: en

✅ 数据加载完成！
📊 训练集样本数: 51786
📊 验证集样本数: 1194

📝 数据集样本预览:
  样本 1:
    葡萄牙语: pt
    英语: en

  样本 2:
    葡萄牙语: e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
    英语: and when you improve searchability , you actually take away the one advantage of print , which is serendipity .

  样本 3:
    葡萄牙语: mas e se estes fatores fossem ativos ?
    英语: but what if it were active ?

(train_transformers) root@iv-ydg6wcq3ggay8n6dmn75:/data2/workspace/yszhang/train_transformers/run_by_steps/run_by_steps# 
"""