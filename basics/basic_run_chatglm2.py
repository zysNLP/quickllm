# -*- coding: utf-8 -*-
"""
    @Project ：quickllm
    @File    ：basic_run_chatglm2.py
    @Author  ：ys
    @Time    ：2023/12/12 12:12
    官方项目：https://github.com/THUDM/ChatGLM2-6B
"""

import os
import torch
from loguru import logging
from transformers import AutoTokenizer

from quickllm.models import build_transformer_model
from quickllm.generation import SeqGeneration


class ExpertModel:

    def __init__(self):
        self.prompt = "请以一位医疗领域知识图谱专家的角色回答以下问题："
        self.choice = 'default'  # default, int4, 32k
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 来自https://huggingface.co/THUDM/chatglm2-6b；
        self.dir_path = "/path/to/my/pretrain_ckpt/glm/chatglm2-6B"
        self.checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
        # 来自项目中的：examples/basic/glm/chatglm2-6B/quickllm_config.json
        self.config_path = dir_path + '/quickllm_config.json'

    def build_prompt(self, history):
        for query, response in history:
            self.prompt += f"\n\n用户：{query}"
            self.prompt += f"\n\nChatGLM-6B：{response}"
        return self.prompt

    def build_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.dir_path.replace('/', '\\'), trust_remote_code=True)
        if self.choice in {'default', '32k'}:
            encoder = build_transformer_model(config_path=self.config_path,
                                              checkpoint_path=self.checkpoint_path).half().to(device)
        else:
            encoder = build_transformer_model(config_path=self.config_path,
                                              checkpoint_path=self.checkpoint_path).to(device)

        model = SeqGeneration(encoder, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample',
                              maxlen=2048, default_rtype='logits', use_states=True)
        return model

    def chat(self, query, history):
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)

        for response in self.build_model().stream_generate(prompt, topk=50, topp=0.7, temperature=0.95):
            new_history = history + [(query, response)]
            yield response, new_history

    def main(self):
        history = []
        logging.info("----欢迎使用ChatGLM2-6B模型，修改prompt输入内容进行对话，clear清空对话历史，stop终止程序")
        while True:
            query = input("\nQuestion：")
            if query.strip() == "stop":
                break
            if query.strip() == "clear":
                history = []
                print("----已清空历史对话----")
                continue
            for response, history in self.chat(query, history=history):
                print(build_prompt(history), flush=True)

            print(build_prompt(history), flush=True)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    expert_bot = ExpertModel()
    expert_bot.main()