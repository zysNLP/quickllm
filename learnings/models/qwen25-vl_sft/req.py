# -*- coding: utf-8 -*-
"""
    @Project : quickllm
    @File    : req.py
    @Author  : sunday
    @Time    : 2025/5/16 13:25
"""

import requests
import base64
import time

def build_message(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_b64,
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28
                },
                {
                    "type": "text",
                    "text": "请描述这张图片"
                }
            ]
        },
        {
            "role": "assistant",
            "content": "中国宇航员桂海潮正在讲话。"
        },
        {
            "role": "user",
            "content": "他取得过哪些成就？"
        }
    ]
    return messages

if __name__ == "__main__":
    t1 = time.time()
    image_paths = [
        "/quickllm/LLaMA-Factory/data/mllm_demo_data/3.jpg",
    ] * 1000  # 10个batch

    batch_messages = [build_message(path) for path in image_paths]

    url = "http://localhost:7868/chat"
    headers = {"Content-Type": "application/json"}
    data = {"batch_messages": batch_messages}

    response = requests.post(url, headers=headers, json=data, timeout=120)
    if response.status_code == 200:
        result = response.json()
        for i, resp in enumerate(result["responses"]):
            print(f"第{i+1}个响应: {resp}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("错误信息:", response.text)
    t2 = time.time()
    print(f"总时间: {t2 - t1}秒")
