# -*- coding: utf-8 -*-
"""
    @Project : quickllm
    @File    : chat.py
    @Author  : sunday
    @Time    : 2025/5/15 09:59
"""
import requests
import concurrent.futures

API_URL = "http://localhost:8100/v1/completions"
HEADERS = {"Content-Type": "application/json"}

def request_vllm(prompt):
    data = {
        "model": "R1-14B",
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "n": 1,
        "stream": False
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    return response.json()

prompts = [
    "请用一句话介绍中国。",
    "请用一句话介绍中国。",
    "请用一句话介绍中国。",
]

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(request_vllm, prompt) for prompt in prompts]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())