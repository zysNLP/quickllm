# pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install tensorflow-datasets

import tensorflow_datasets as tfds
import csv
import os

# 设置数据目录 - 指向新的位置
DATA_DIR = '/data3/users/yszhang/mr_project/train_transformers/tensorflow_datasets'

print("正在加载数据集...")

try:
    # 加载数据集
    ds = tfds.load(
        'ted_hrlr_translate/pt_to_en',
        split=['train', 'validation'],
        as_supervised=True,
        download=False,
        data_dir=DATA_DIR
    )

    # 修正：ds 是一个列表，不是字典
    train_ds = ds[0]  # 训练集
    val_ds = ds[1]  # 验证集

    print("数据集加载成功！")

    # 统计样本数量
    train_count = sum(1 for _ in train_ds)
    val_count = sum(1 for _ in val_ds)

    print(f"训练集样本数: {train_count}")
    print(f"验证集样本数: {val_count}")

    # 保存训练集为 CSV
    print("正在保存训练集...")
    with open("./ted_pt_en_train.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["pt", "en"])  # 添加表头

        for i, (pt, en) in enumerate(tfds.as_numpy(train_ds)):
            if i % 1000 == 0:
                print(f"已处理训练样本: {i}/{train_count}")
            writer.writerow([pt.decode("utf-8"), en.decode("utf-8")])

    print("训练集保存完成！")

    # 保存验证集为 CSV
    print("正在保存验证集...")
    with open("./ted_pt_en_test.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["pt", "en"])  # 添加表头

        for i, (pt, en) in enumerate(tfds.as_numpy(val_ds)):
            if i % 100 == 0:
                print(f"已处理验证样本: {i}/{val_count}")
            writer.writerow([pt.decode("utf-8"), en.decode("utf-8")])

    print("验证集保存完成！")

    # 验证保存的文件
    print("\n文件保存验证:")
    if os.path.exists("./ted_pt_en_train.csv"):
        train_size = os.path.getsize("./ted_pt_en_train.csv")
        print(f"ted_pt_en_train.csv: {train_size} bytes")

    if os.path.exists("./ted_pt_en_test.csv"):
        test_size = os.path.getsize("./ted_pt_en_test.csv")
        print(f"ted_pt_en_test.csv: {test_size} bytes")

    # 显示前几行内容
    print("\n训练集前3行预览:")
    with open("./ted_pt_en_train.csv", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 4:  # 表头 + 3行数据
                print(f"第{i}行: {line.strip()}")
            else:
                break

except Exception as e:
    print(f"错误: {e}")
    print("请检查数据目录和文件是否存在")
