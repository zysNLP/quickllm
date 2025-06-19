import json
import pandas as pd
import os

def print_math_tasks_sample():
    """打印math_tasks.jsonl中的一条数据"""
    jsonl_path = 'data/math_tasks.jsonl'
    
    print("=== math_tasks.jsonl 数据示例 ===")
    
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            # 读取第一行
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                print(f"问题: {data.get('question', 'N/A')}")
                print(f"答案: {data.get('answer', 'N/A')}")
                print(f"ID: {data.get('id', 'N/A')}")
                print(f"项数: {data.get('num_terms', 'N/A')}")
                print(f"位数: {data.get('num_digits', 'N/A')}")
                print(f"完整数据: {data}")
    else:
        print(f"文件不存在: {jsonl_path}")
    
    print("\n")

def print_gsm8k_chinese_sample():
    """打印gsm8k_chinese数据集中的一条数据"""
    parquet_path = '/data2/users/yszhang/quickllm/rl/llm_related-main/grpo_from_scratch/datasets/gsm8k_chinese/data/train-00000-of-00001.parquet'
    
    print("=== gsm8k_chinese 数据示例 ===")
    
    if os.path.exists(parquet_path):
        try:
            # 读取parquet文件
            df = pd.read_parquet(parquet_path)
            
            if len(df) > 0:
                # 获取第一行数据
                first_row = df.iloc[0]
                print(f"数据集大小: {len(df)} 条")
                print(f"列名: {list(df.columns)}")
                print("\n第一条数据:")
                for col in df.columns:
                    value = first_row[col]
                    if isinstance(value, str) and len(value) > 200:
                        # 如果文本太长，只显示前200个字符
                        print(f"{col}: {value[:200]}...")
                    else:
                        print(f"{col}: {value}")
                        
                print(f"\n完整第一行数据:\n{first_row.to_dict()}")
            else:
                print("数据集为空")
                
        except Exception as e:
            print(f"读取parquet文件时出错: {e}")
    else:
        print(f"文件不存在: {parquet_path}")

if __name__ == "__main__":
    print_math_tasks_sample()
    print_gsm8k_chinese_sample()
