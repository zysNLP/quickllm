# -*- coding: utf-8 -*- 
"""
    @Project ：quickllm 
    @File    ：triton_client_glm3.py
    @Author  ：ys
    @Time    ：2023/12/13 11:20 
"""

from typing import List
import numpy as np
import time
from copy import deepcopy

from triton_client import TritonHttpClient


class ChatGLMTokenizer(PreTrainedTokenizer):

    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, **kwargs):
        self.name = "GLMTokenizer"
        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }
        super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)

    def build_prompt(self, prompt, query, history=None):
        input_prompt= self.tokenizer(prompt)
        input_query = self.tokenizer(query)
        input_history = self.tokenizer(history)
        input_ids = [input_prompt, input_query, input_history]
        return input_ids

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens


class ChatGLM3TritonClient:

    def __init__(self, name, model_dir, batch_size, max_output, triton_client) -> None:
        self.tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
        self.batch_size = batch_size
        self.max_output = max_output
        self.triton = triton_client
        self.end_id = self.tokenizer.special_tokens['<eos>']
        self.pad_id = self.tokenizer.special_tokens['<pad>']
        self.name = name

    def pre_process(self, prompt: str, query: str, history=None):
        """将text转化为token
        """
        input_ids = self.tokenizer.build_prompt(prompt, query, history)[1]
        input_ids = [input_ids]

        # 构建输入
        batch_data = []
        size = len(input_ids) // self.batch_size + 1
        for i in range(size):
            batch_input = input_ids[i * self.batch_size:i * self.batch_size + self.batch_size]
            length = max([len(_) for _ in batch_input])
            for bi, line in enumerate(batch_input):
                if len(line) < length:
                    line.extend([self.pad_id] * length)
                batch_input[bi] = batch_input[bi][:length]

            batch_data.append({
                "input_ids": np.array(batch_input, dtype=np.int32),
                "request_output_len": np.array([[self.max_output] for _ in batch_input], dtype=np.uint32),
            })
            return batch_data

    def post_process(self, context):
        rt = []
        for record in context:
            rt.append(self.tokenizer.decode(record[0]))
        return rt

    def chat(self, query: str, history=None):
        """根据提供的问题和问答历史进行回答
        """
        if history is None:
            history = []
        pre_process_data = self.pre_process(query, history)

        results = []
        for index, record in enumerate(pre_process_data):
            results.append(self.triton.run(self.name, record))

        return results