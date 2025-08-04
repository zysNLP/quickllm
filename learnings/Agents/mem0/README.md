# Memory0 AI记忆系统示例

## 项目简介

这是一个使用 [Memory0](https://mem0.ai/) 服务实现AI助手记忆功能的示例项目。Memory0是一个专门为AI应用提供记忆能力的平台，让AI助手能够记住用户的重要信息，提供个性化的服务体验。

## 功能特性

- 🧠 **智能记忆存储**：自动提取和存储用户的重要信息
- 🔍 **语义搜索**：基于自然语言查询检索相关记忆
- 🏷️ **智能分类**：自动将信息归类到不同类别（健康、偏好、个人信息等）
- 📊 **相关性评分**：为每条记忆计算与查询的相关性分数
- 👤 **用户隔离**：支持多用户，每个用户的记忆独立存储

## 安装依赖

```bash
pip install mem0
```

## 快速开始

### 1. 获取API密钥

首先需要在 [Memory0官网](https://mem0.ai/) 注册账号并获取API密钥。

### 2. 基本使用

```python
from mem0 import MemoryClient

# 初始化客户端
client = MemoryClient(api_key="your-api-key-here")

# 存储用户记忆
messages = [
    { "role": "user", "content": "你好，我是小明。我是素食主义者，而且对花生过敏。" },
    { "role": "assistant", "content": "你好小明！我了解到你是素食主义者，并且对花生过敏。" }
]

client.add(messages, user_id="xiaoming")

# 查询记忆
query = "今晚我可以做什么晚餐？"
answer = client.search(query, user_id="xiaoming")
print(answer)
```

## 代码示例详解

### 记忆存储

```python
messages = [
    { "role": "user", "content": "你好，我是小明。我是素食主义者，而且对花生过敏。" },
    { "role": "assistant", "content": "你好小明！我了解到你是素食主义者，并且对花生过敏。" }
]

client.add(messages, user_id="xiaoming")
```

**功能说明：**
- 将对话历史存储到Memory0服务
- 系统自动提取关键信息并分类
- 使用`user_id`标识不同用户的记忆

### 记忆检索

```python
query = "今晚我可以做什么晚餐？"
answer = client.search(query, user_id="xiaoming")
```

**功能说明：**
- 基于自然语言查询检索相关记忆
- 返回按相关性排序的记忆列表
- 每条记忆包含分类标签和相关性分数

## 运行结果示例

运行代码后，系统会返回类似以下的结构化记忆：

```json
[
  {
    "memory": "用户名字是小明",
    "categories": ["personal_details"],
    "score": 0.42634574410430304
  },
  {
    "memory": "对花生过敏",
    "categories": ["health"],
    "score": 0.39457240072324273
  },
  {
    "memory": "是素食主义者",
    "categories": ["user_preferences", "food"],
    "score": 0.38318318574278243
  }
]
```

## 记忆分类说明

Memory0会自动将用户信息分类到以下类别：

- **personal_details**：个人信息（姓名、年龄等）
- **health**：健康信息（过敏、疾病等）
- **user_preferences**：用户偏好（饮食、爱好等）
- **food**：食物相关偏好
- **location**：位置信息
- **work**：工作相关信息

## 实际应用场景

### 1. 个性化推荐
- 基于用户饮食偏好推荐菜品
- 考虑健康限制避免过敏食物
- 根据用户习惯提供定制化建议

### 2. 客服助手
- 记住用户的购买历史
- 了解用户的偏好和需求
- 提供更精准的客户服务

### 3. 健康管理
- 记录用户的健康信息
- 提醒用药和检查
- 提供个性化的健康建议

### 4. 学习助手
- 记住用户的学习进度
- 了解用户的强项和弱项
- 提供个性化的学习计划

## 高级功能

### 批量记忆管理

```python
# 批量添加记忆
memories = [
    {"content": "用户喜欢咖啡", "user_id": "xiaoming"},
    {"content": "用户住在北京", "user_id": "xiaoming"},
    {"content": "用户是程序员", "user_id": "xiaoming"}
]

for memory in memories:
    client.add([{"role": "system", "content": memory["content"]}], user_id=memory["user_id"])
```

### 记忆更新和删除

```python
# 更新记忆
client.update(memory_id, new_content)

# 删除记忆
client.delete(memory_id)
```

## 注意事项

1. **API密钥安全**：请妥善保管API密钥，不要将其提交到公开代码库
2. **用户隐私**：确保遵守相关隐私法规，合理使用用户数据
3. **记忆限制**：注意Memory0服务的存储限制和API调用频率限制
4. **数据备份**：重要数据建议进行本地备份

## 故障排除

### 常见问题

1. **API密钥错误**
   - 检查API密钥是否正确
   - 确认账户是否已激活

2. **网络连接问题**
   - 检查网络连接
   - 确认防火墙设置

3. **记忆检索为空**
   - 检查user_id是否正确
   - 确认是否已成功存储记忆

## 相关资源

- [Memory0官方文档](https://docs.mem0.ai/)
- [Memory0 API参考](https://docs.mem0.ai/platform/quickstart)
- [GitHub仓库](https://github.com/memory0/memory0-python)

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**注意**：这是一个示例项目，用于演示Memory0服务的使用方法。在实际应用中，请根据具体需求调整代码结构和功能实现。 