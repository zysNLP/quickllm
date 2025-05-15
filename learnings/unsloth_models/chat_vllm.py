# -*- coding: utf-8 -*-
"""
    @Project : quickllm
    @File    : chat_vllm.py
    @Author  : sunday
    @Time    : 2025/5/15 09:59
"""
import asyncio
import aiohttp

url = "http://localhost:8101/generate"
headers = {"Content-Type": "application/json"}

async def chat(prompt, session):
    data = {"prompt": prompt}
    try:
        async with session.post(url, headers=headers, json=data) as resp:
            res = await resp.json()
            return res.get("result", "无结果")
    except Exception as e:
        return f"请求异常: {e}"

async def batch_chat(prompts):
    async with aiohttp.ClientSession() as session:
        tasks = [chat(prompt, session) for prompt in prompts]
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    prompts = [
        "请用一句话介绍中国。",
        "请用一句话介绍中国。",
        "请用一句话介绍中国。"
    ]
    results = asyncio.run(batch_chat(prompts))
    for i, res in enumerate(results):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Result: {res}\n")