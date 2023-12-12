#! -*- coding: utf-8 -*-
# 基本测试：chatglm的对话测试
# quickllm_config.json文件请访问readme

# 官方项目：https://github.com/THUDM/ChatGLM-6B
# hf链接：https://huggingface.co/THUDM/chatglm-6b
# fp16半精度下显存占用14G
# 20230406 官方项目对20000个和图像相关的进行的裁剪，因此本项目之前裁剪及tokenize的作废，使用最新的tokenize不需要进行offset

import torch
from quickllm.models import build_transformer_model
from transformers import AutoTokenizer
from quickllm.generation import AutoRegressiveDecoder, SeqGeneration
import platform
import os
import re


choice = 'default'  # default, int4, int8
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B"
elif choice == 'int4':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int4"
elif choice == 'int8':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int8"
else:
    raise ValueError(f'{choice} not in pre maintained choices')

config_path = dir_path + '/quickllm_config.json'
checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)
# 建立模型，加载权重
if choice == 'default':
    encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).half()
    encoder = encoder.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)
else:
    # 在config中已经写入了量化的配置参数
    encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).to(device)

# 第一种方式: 自定义解码
# class Chat(AutoRegressiveDecoder):
#     @AutoRegressiveDecoder.wraps(default_rtype='logits')
#     def predict(self, inputs, output_ids, states):
#         token_ids = torch.cat([inputs[0], output_ids], 1)
#         logits = encoder.predict([token_ids])
#         return logits[:, -1, :]

#     def generate(self, text, n=1, topk=50, topp=0.7, temperature=0.95):
#         token_ids = tokenizer.encode(text)
#         results = self.random_sample([token_ids], n=n, topk=topk, topp=topp,  temperature=temperature)  # 基于随机采样
#         return tokenizer.decode(results[0].cpu().numpy())
# generation = Chat(start_id=None, end_id=tokenizer.eos_token_id, maxlen=2048, device=device)

# 第二种方式：调用封装好的接口，可使用cache
generation = SeqGeneration(encoder, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample',
                           maxlen=2048, default_rtype='logits', use_states=True)

def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response

def chat(query, history=[]):
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

    response = generation.generate(prompt, topk=50, topp=0.7, temperature=0.95)
    response = process_response(response)
    history = history + [(query, response)]
    return response, history


if __name__ == '__main__':
    os_name = platform.system()
    history = []
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        response, history = chat(query, history=history)
        print(f"ChatGLM-6B：{response}")
        torch.cuda.empty_cache()  # 清理显存

