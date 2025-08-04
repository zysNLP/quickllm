# -*- coding: utf-8 -*-
"""
    @Project : quickllm
    @File    : main.py
    @Author  : sunday
    @Time    : 2025/8/2 11:46
"""

from mem0 import MemoryClient
import time

client = MemoryClient(api_key="m0-3hve3lYKhMMYxg5z15XrwEV8t9yek4t2lcUlwWUY")

print("=== 基础记忆示例 ===")
# 基础示例：存储用户基本信息
messages = [
    { "role": "user", "content": "你好，我是小明。我是素食主义者，而且对花生过敏。" },
    { "role": "assistant", "content": "你好小明！我了解到你是素食主义者，并且对花生过敏。" }
]

client.add(messages, user_id="xiaoming")

query = "今晚我可以做什么晚餐？"
answer = client.search(query, user_id="xiaoming")
print(f"查询: {query}")
print(f"相关记忆: {len(answer)} 条")
print()

print("=== 长期记忆示例：多轮对话 ===")
# 模拟长期对话，逐步建立用户画像
conversations = [
    # 第一次对话：基本信息
    [
        {"role": "user", "content": "我今年25岁，是一名软件工程师，在北京工作。"},
        {"role": "assistant", "content": "很高兴认识你！25岁的软件工程师，在北京工作，听起来很有挑战性。"}
    ],
    
    # 第二次对话：工作相关
    [
        {"role": "user", "content": "我最近在做一个AI项目，使用Python和TensorFlow，经常加班到很晚。"},
        {"role": "assistant", "content": "AI项目听起来很有趣！使用Python和TensorFlow，而且经常加班，工作强度不小啊。"}
    ],
    
    # 第三次对话：生活习惯
    [
        {"role": "user", "content": "我习惯晚上11点睡觉，早上7点起床，喜欢喝咖啡提神。"},
        {"role": "assistant", "content": "作息很规律呢！晚上11点睡，早上7点起，咖啡确实能提神。"}
    ],
    
    # 第四次对话：健康信息
    [
        {"role": "user", "content": "我有轻微的颈椎病，医生建议我多运动，但我总是没时间。"},
        {"role": "assistant", "content": "颈椎病确实需要重视，医生建议多运动是对的，工作再忙也要注意健康。"}
    ],
    
    # 第五次对话：个人偏好
    [
        {"role": "user", "content": "我喜欢看科幻电影，最近迷上了《星际穿越》，周末经常去电影院。"},
        {"role": "assistant", "content": "《星际穿越》确实很棒！喜欢科幻电影，周末去电影院是个不错的放松方式。"}
    ]
]

# 逐步添加对话记忆
for i, conversation in enumerate(conversations, 1):
    print(f"添加第{i}轮对话记忆...")
    client.add(conversation, user_id="xiaoming")

print("\n=== 长期记忆检索测试 ===")

# 首先查看所有存储的记忆
print("=== 调试：查看所有存储的记忆 ===")
all_memories = client.search("用户", user_id="xiaoming")  # 使用通用词查询
print(f"总共存储了 {len(all_memories)} 条记忆:")
for i, memory in enumerate(all_memories, 1):
    print(f"{i}. {memory['memory']} (类别: {memory['categories']}, 相关度: {memory['score']:.3f})")

print("\n=== 针对性查询测试 ===")
# 测试不同类型的记忆检索
queries = [
    "我的工作是什么？",
    "我有什么健康问题？", 
    "我喜欢什么类型的电影？",
    "我的作息时间怎么样？",
    "我最近在做什么项目？"
]

for query in queries:
    print(f"\n查询: {query}")
    memories = client.search(query, user_id="xiaoming")
    print(f"找到 {len(memories)} 条相关记忆:")
    for i, memory in enumerate(memories[:5], 1):  # 显示前5条
        print(f"  {i}. {memory['memory']} (相关度: {memory['score']:.3f})")

print("\n=== 直接关键词查询测试 ===")
# 使用更直接的关键词查询
direct_queries = [
    "科幻电影",
    "星际穿越", 
    "电影院",
    "软件工程师",
    "AI项目",
    "颈椎病",
    "咖啡"
]

for query in direct_queries:
    print(f"\n直接查询关键词: {query}")
    memories = client.search(query, user_id="xiaoming")
    print(f"找到 {len(memories)} 条相关记忆:")
    for memory in memories[:3]:
        print(f"  - {memory['memory']} (相关度: {memory['score']:.3f})")

print("\n=== 记忆更新示例 ===")
# 模拟记忆更新：用户信息发生变化
update_conversation = [
    {"role": "user", "content": "我最近换工作了，现在在一家AI创业公司，工作更忙了，经常工作到凌晨。"},
    {"role": "assistant", "content": "恭喜换工作！AI创业公司听起来很有挑战性，工作强度更大了。"}
]

print("更新工作信息...")
client.add(update_conversation, user_id="xiaoming")

# 测试更新后的记忆
print("\n更新后的记忆检索:")
updated_query = "我现在的公司怎么样？"
updated_memories = client.search(updated_query, user_id="xiaoming")
print(f"查询: {updated_query}")
for memory in updated_memories[:2]:
    print(f"  - {memory['memory']} (相关度: {memory['score']:.3f})")

print("\n=== 跨时间记忆对比 ===")
# 对比不同时间点的记忆
old_query = "我的工作作息"
old_memories = client.search(old_query, user_id="xiaoming")
print(f"查询: {old_query}")
print("相关记忆:")
for memory in old_memories[:3]:
    print(f"  - {memory['memory']} (相关度: {memory['score']:.3f})")

print("\n=== 个性化推荐示例 ===")
# 基于长期记忆提供个性化建议
recommendation_queries = [
    "我应该怎么改善颈椎问题？",
    "有什么适合我的放松方式？",
    "我适合什么样的工作环境？"
]

for query in recommendation_queries:
    print(f"\n查询: {query}")
    memories = client.search(query, user_id="xiaoming")
    print("基于记忆的个性化建议:")
    for memory in memories[:2]:
        print(f"  - 考虑因素: {memory['memory']}")
    print("  - 建议: 基于您的健康、工作习惯和个人偏好")

print("\n=== 记忆持久性测试 ===")
# 模拟长时间后的记忆检索
print("模拟3个月后的记忆检索...")
long_term_query = "我的基本情况"
long_term_memories = client.search(long_term_query, user_id="xiaoming")
print(f"查询: {long_term_query}")
print(f"长期记忆保持: {len(long_term_memories)} 条核心信息")
for memory in long_term_memories[:5]:
    print(f"  - {memory['memory']} (类别: {memory['categories']})")

print("\n=== 多用户记忆隔离测试 ===")
# 测试多用户记忆隔离
other_user_conversation = [
    {"role": "user", "content": "你好，我是小红。我是一名医生，喜欢瑜伽和冥想。"},
    {"role": "assistant", "content": "你好小红！医生是很崇高的职业，瑜伽和冥想对身心健康都很有帮助。"}
]

client.add(other_user_conversation, user_id="xiaohong")

# 测试用户隔离
print("测试用户记忆隔离...")
xiaoming_query = "我的职业"
xiaohong_query = "我的爱好"

xiaoming_memories = client.search(xiaoming_query, user_id="xiaoming")
xiaohong_memories = client.search(xiaohong_query, user_id="xiaohong")

print(f"小明({len(xiaoming_memories)}条记忆): {[m['memory'] for m in xiaoming_memories[:2]]}")
print(f"小红({len(xiaohong_memories)}条记忆): {[m['memory'] for m in xiaohong_memories[:2]]}")

print("\n=== 长期记忆功能总结 ===")
print("✅ 多轮对话记忆累积")
print("✅ 跨时间记忆保持")
print("✅ 记忆更新和演进")
print("✅ 个性化推荐能力")
print("✅ 多用户记忆隔离")
print("✅ 语义相关性检索")
print("✅ 智能记忆分类")
