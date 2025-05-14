# -*- coding: utf-8 -*-
"""
    @Project : quickllm
    @File    : tool_us.py
    @Author  : ys
    @Time    : 2025/3/1 16:00
"""

import sys
import asyncio
import time
import datetime

from typing import Any, Optional, List
import aiohttp


async def legal_cot_batch_tool(questions, url):
    start_time = datetime.datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S.%f')}] 向 {url} 发送请求，问题: {questions}")
    payload = {"questions": questions}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                end_time = datetime.datetime.now()
                print(f"[{end_time.strftime('%H:%M:%S.%f')}] {url} 返回，耗时: {(end_time - start_time).total_seconds():.2f}s")
                return result
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')}] {url} 请求异常: {str(e)}")
        return {
            'code': 400,
            'elapsed': 0.0,
            'data': {},
            'message': f"Tool error: {str(e)}"
        }

async def main():
    all_start = datetime.datetime.now()
    print(f"主程序开始时间: {all_start.strftime('%H:%M:%S.%f')}")
    test_cases = [
        ["什么是著作权？", "什么是专利？", "什么是商标？", "什么是知识产权侵权？"],
        ["什么是著作权？", "什么是专利？", "什么是商标？", "什么是知识产权侵权？"],
        ["什么是著作权？", "什么是专利？", "什么是商标？", "什么是知识产权侵权？"]
    ]
    urls = [
        "http://localhost:8100/legal_cot_batch",
        "http://localhost:8101/legal_cot_batch",
        "http://localhost:8102/legal_cot_batch"
    ]
    tasks = [legal_cot_batch_tool(qs, url) for qs, url in zip(test_cases, urls)]
    results = await asyncio.gather(*tasks)
    all_end = datetime.datetime.now()
    print(f"主程序结束时间: {all_end.strftime('%H:%M:%S.%f')}")
    print(f"总耗时: {(all_end - all_start).total_seconds():.2f}s")
    for i, result in enumerate(results):
        print(f"\nResult from {urls[i]}: code={result.get('code')}, elapsed={result.get('elapsed')}ms")
        print(f"  responses: {result.get('data', {}).get('responses', result.get('data'))}")

if __name__ == "__main__":
    asyncio.run(main()) 