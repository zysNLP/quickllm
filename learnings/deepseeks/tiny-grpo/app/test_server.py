# -*- coding: utf-8 -*-
"""
测试推理服务器
"""
import asyncio
import httpx
import json
import sys
import os

# 处理相对导入问题
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.config import TRANSFORMERS_PORT, VLLM_PORT
else:
    from .config import TRANSFORMERS_PORT, VLLM_PORT


async def test_server(port: int, server_name: str):
    """测试单个服务器"""
    url = f"http://localhost:{port}/solve_question"
    
    test_question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    payload = {
        "question": test_question
    }
    
    print(f"\n=== 测试 {server_name} (端口 {port}) ===")
    print(f"问题: {test_question}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"状态码: {result['code']}")
                print(f"处理时间: {result['elapsed_milliseconds']}ms")
                print(f"消息: {result['message']}")
                print("\n--- 结果数据 ---")
                for key, value in result["data"].items():
                    print(f"{key}: {value}")
                return result
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return None
                
    except Exception as e:
        print(f"连接错误: {e}")
        return None


async def compare_servers():
    """比较两个服务器的结果"""
    print("开始比较两个服务器...")
    
    # 测试两个服务器
    transformers_result = await test_server(TRANSFORMERS_PORT, "Transformers")
    vllm_result = await test_server(VLLM_PORT, "vLLM")
    
    if transformers_result and vllm_result:
        print("\n=== 结果比较 ===")
        
        # 比较最终答案
        trans_answer = transformers_result["data"].get("final_answer", "")
        vllm_answer = vllm_result["data"].get("final_answer", "")
        
        print(f"Transformers 最终答案: {trans_answer}")
        print(f"vLLM 最终答案: {vllm_answer}")
        print(f"答案一致: {'是' if trans_answer == vllm_answer else '否'}")
        
        # 比较处理时间
        trans_time = transformers_result["elapsed_milliseconds"]
        vllm_time = vllm_result["elapsed_milliseconds"]
        speedup = trans_time / vllm_time if vllm_time > 0 else 0
        
        print(f"\n--- 性能比较 ---")
        print(f"Transformers 处理时间: {trans_time}ms")
        print(f"vLLM 处理时间: {vllm_time}ms")
        print(f"vLLM 加速比: {speedup:.2f}x")
        
        # 比较推理过程
        trans_reasoning = transformers_result["data"].get("reasoning_process", "")
        vllm_reasoning = vllm_result["data"].get("reasoning_process", "")
        
        print(f"\n--- 推理过程比较 ---")
        print(f"推理过程长度 - Transformers: {len(trans_reasoning)} 字符")
        print(f"推理过程长度 - vLLM: {len(vllm_reasoning)} 字符")
        
        # 显示详细的助手回复
        print(f"\n--- Transformers 完整回复 ---")
        print(transformers_result["data"].get("assistant_response", ""))
        
        print(f"\n--- vLLM 完整回复 ---")
        print(vllm_result["data"].get("assistant_response", ""))


if __name__ == "__main__":
    asyncio.run(compare_servers()) 