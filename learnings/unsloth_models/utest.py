#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 导入必要的库
import os  # 用于文件和目录操作
import torch  # 用于张量计算和设备管理
from unsloth import FastLanguageModel  # 加载 unsloth 模型
from transformers import AutoTokenizer  # 加载分词器
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print(f"当前GPU设备：{os.environ['CUDA_VISIBLE_DEVICES']}")

# 1. 配置与加载模型
merged_model_dir = "DeepSeek-R1-Legal-COT-merged_test"  # 指定训练后模型的本地目录
max_seq_length = 2048  # 设置最大序列长度，与训练保持一致

if not os.path.exists(merged_model_dir):  # 检查模型目录是否存在
    raise FileNotFoundError(f"错误：目录 {merged_model_dir} 不存在")

model, tokenizer = FastLanguageModel.from_pretrained(  # 加载合并后的模型和分词器
    merged_model_dir,  # 使用本地模型路径
    max_seq_length=max_seq_length,  # 设置最大序列长度
    dtype=None,  # 自动选择数据类型
    load_in_4bit=True  # 启用 4-bit 量化
)
print("模型和分词器加载成功！")  # 输出加载成功的提示

device = "cuda" if torch.cuda.is_available() else "cpu"  # 动态选择设备
print(f"使用设备：{device}")  # 输出当前设备
FastLanguageModel.for_inference(model)  # 切换模型至推理模式

# 2. 定义推理提示模板
prompt_template = (
    "你是一位法律专家，具备高级法律推理、案例分析和法律解释能力。\n"
    "请根据以下问题，生成逐步的思路链并回答。\n\n"
    "问题：{}\n"
    "回答：\n"
    "<思路>{}</思路>\n"
)

def print_response(response):
    print("\n【模型回答】")
    response_text = response.strip()
    if "<思路>" in response_text:
        print(response_text)
    else:
        print("警告：未生成完整思路链，直接输出：")
        print(response_text)

# 3. 推理函数
def perform_inference(question):
    input_text = prompt_template.format(question, "")
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        use_cache=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(input_text):].strip()
    print_response(response)

# 4. 主程序：自动问一个问题并进入交互
if __name__ == "__main__":
    try:
        print("欢迎使用 DeepSeek-R1-Legal-COT 法律推理模型！")
        print("自动测试问题：什么是著作权？\n")
        perform_inference("什么是著作权？")
        print("\n你可以继续输入中文法律问题（输入 '退出' 或 'exit' 结束）：")
        while True:
            question = input("> ").strip()
            if question.lower() in ["退出", "exit"]:
                print("程序已退出。")
                break
            if not question:
                print("请输入有效问题！")
                continue
            print(f"\n处理问题：{question}")
            perform_inference(question)
    except Exception as e:
        print(f"程序出错：{e}") 