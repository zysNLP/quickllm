# OpenManus Agent 模块

OpenManus Agent模块是一个基于ReAct（Reasoning + Acting）模式的智能代理系统，提供了完整的AI Agent架构，支持多轮对话、工具调用、任务规划和状态管理。

## 目录结构

```
app/agent/
├── __init__.py          # 模块初始化
├── base.py              # 基础Agent抽象类
├── react.py             # ReAct模式抽象类
├── toolcall.py          # 工具调用Agent实现
├── manus.py             # 通用多功能Agent
├── browser.py           # 浏览器自动化Agent
├── swe.py               # 软件工程Agent
├── mcp.py               # MCP协议Agent
└── README.md            # 本文档
```

## 核心架构

### 1. 基础架构层次

```
BaseAgent (抽象基类)
    ↓
ReActAgent (ReAct模式抽象类)
    ↓
ToolCallAgent (工具调用实现)
    ↓
专用Agent (ManusAgent, BrowserAgent, SWEAgent, MCPAgent)
```

### 2. 核心组件

#### Memory（内存管理）
- **文件位置**: `app/schema.py`
- **核心类**: `Memory`, `Message`, `Role`
- **功能**: 管理对话历史、上下文信息和消息类型

#### Planning（任务规划）
- **文件位置**: `app/flow/planning.py`
- **核心类**: `PlanningFlow`, `PlanStepStatus`
- **功能**: 任务分解、步骤管理和执行状态跟踪

#### Tools（工具系统）
- **文件位置**: `app/tool/`
- **核心类**: `BaseTool`, `ToolCollection`, `ToolResult`
- **功能**: 提供具体执行能力，支持文件操作、编程、浏览器自动化等

#### ReAct（推理+行动）
- **文件位置**: `app/agent/`
- **核心类**: `ReActAgent`, `ToolCallAgent`
- **功能**: 实现"思考-行动"循环，智能决策和任务执行

## 主要类详解

### BaseAgent

**文件**: `app/agent/base.py`

Agent的抽象基类，提供核心功能：

```python
class BaseAgent(BaseModel, ABC):
    name: str                    # Agent名称
    description: Optional[str]   # Agent描述
    system_prompt: Optional[str] # 系统提示词
    next_step_prompt: Optional[str] # 下一步提示词
    llm: LLM                    # 语言模型实例
    memory: Memory              # 内存管理
    state: AgentState           # 当前状态
    max_steps: int              # 最大执行步数
    current_step: int           # 当前步数
```

**核心方法**:
- `run(request: str) -> str`: 执行Agent主循环
- `step() -> str`: 执行单个步骤（抽象方法）
- `update_memory()`: 更新内存
- `is_stuck()`: 检测是否陷入循环

### ReActAgent

**文件**: `app/agent/react.py`

实现ReAct模式的抽象类：

```python
class ReActAgent(BaseAgent, ABC):
    @abstractmethod
    async def think(self) -> bool:
        """处理当前状态并决定下一步行动"""

    @abstractmethod
    async def act(self) -> str:
        """执行决定的行动"""

    async def step(self) -> str:
        """执行单个步骤：思考然后行动"""
```

### ToolCallAgent

**文件**: `app/agent/toolcall.py`

支持工具调用的具体实现：

```python
class ToolCallAgent(ReActAgent):
    available_tools: ToolCollection    # 可用工具集合
    tool_choices: TOOL_CHOICE_TYPE     # 工具选择模式
    special_tool_names: List[str]      # 特殊工具名称
    tool_calls: List[ToolCall]         # 当前工具调用
```

**工具选择模式**:
- `ToolChoice.AUTO`: 自动选择是否使用工具
- `ToolChoice.REQUIRED`: 必须使用工具
- `ToolChoice.NONE`: 不使用工具，仅文本回复

## 专用Agent

### ManusAgent

**文件**: `app/agent/manus.py`

通用多功能Agent，支持多种工具：

```python
class Manus(ToolCallAgent):
    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools"

    # 可用工具
    available_tools: ToolCollection = ToolCollection(
        PythonExecute(),      # Python代码执行
        BrowserUseTool(),     # 浏览器自动化
        StrReplaceEditor(),   # 文件编辑
        Terminate()           # 终止执行
    )
```

**特性**:
- 支持Python代码执行
- 浏览器自动化操作
- 文件编辑和操作
- 智能任务规划

### BrowserAgent

**文件**: `app/agent/browser.py`

专用浏览器自动化Agent：

```python
class BrowserAgent(ToolCallAgent):
    name: str = "browser"
    description: str = "A browser agent that can control a browser to accomplish tasks"

    available_tools: ToolCollection = ToolCollection(
        BrowserUseTool(),     # 浏览器工具
        Terminate()           # 终止工具
    )
```

**特性**:
- 网页导航和交互
- 表单填写和提交
- 内容提取和分析
- 截图和状态监控

### SWEAgent

**文件**: `app/agent/swe.py`

软件工程Agent，专注于代码开发：

```python
class SWEAgent(ToolCallAgent):
    name: str = "swe"
    description: str = "an autonomous AI programmer that interacts directly with the computer to solve tasks"

    available_tools: ToolCollection = ToolCollection(
        Bash(),               # 命令行执行
        StrReplaceEditor(),   # 文件编辑
        Terminate()           # 终止工具
    )
```

**特性**:
- 代码编写和修改
- 命令行操作
- 文件系统操作
- 软件工程任务

### MCPAgent

**文件**: `app/agent/mcp.py`

MCP（Model Context Protocol）协议Agent：

```python
class MCPAgent(ToolCallAgent):
    name: str = "mcp_agent"
    description: str = "An agent that connects to an MCP server and uses its tools"

    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    connection_type: str = "stdio"  # "stdio" or "sse"
```

**特性**:
- 连接MCP服务器
- 动态工具发现
- 多媒体响应处理
- 协议通信管理

## 使用示例

### 基本使用

```python
from app.agent.manus import Manus

# 创建Agent实例
agent = Manus()

# 执行任务
result = await agent.run("请帮我分析这个网站的内容")
print(result)
```

### 自定义Agent

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection, Bash, Terminate

class CustomAgent(ToolCallAgent):
    name = "custom_agent"
    description = "自定义Agent"

    available_tools = ToolCollection(
        Bash(),
        Terminate()
    )

    system_prompt = "你是一个自定义的AI助手"
    next_step_prompt = "请分析当前情况并采取行动"

# 使用自定义Agent
agent = CustomAgent()
result = await agent.run("执行自定义任务")
```

### 浏览器Agent使用

```python
from app.agent.browser import BrowserAgent

# 创建浏览器Agent
browser_agent = BrowserAgent()

# 执行浏览器任务
result = await browser_agent.run("请访问百度并搜索'OpenAI'")
print(result)
```

## 状态管理

Agent支持以下状态：

```python
class AgentState(str, Enum):
    IDLE = "IDLE"           # 空闲状态
    RUNNING = "RUNNING"     # 运行中
    FINISHED = "FINISHED"   # 已完成
    ERROR = "ERROR"         # 错误状态
```

## 错误处理

### 循环检测

Agent会自动检测是否陷入循环：

```python
def is_stuck(self) -> bool:
    """检测Agent是否陷入循环"""
    # 检查最近的消息是否有重复内容
    # 如果重复次数超过阈值，则认为陷入循环
```

### 错误恢复

```python
def handle_stuck_state(self):
    """处理陷入循环的状态"""
    stuck_prompt = "检测到重复响应，请考虑新的策略"
    self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
```

## 配置选项

### 基本配置

```python
agent = Manus(
    max_steps=20,           # 最大执行步数
    max_observe=10000,      # 最大观察长度
    duplicate_threshold=2   # 重复检测阈值
)
```

### 工具配置

```python
# 自定义工具集合
custom_tools = ToolCollection(
    PythonExecute(),
    BrowserUseTool(),
    # 添加更多工具...
)

agent = ToolCallAgent(
    available_tools=custom_tools,
    tool_choices=ToolChoice.AUTO
)
```

## 扩展开发

### 创建新工具

```python
from app.tool.base import BaseTool, ToolResult

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "自定义工具描述"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "输入参数"}
        }
    }

    async def execute(self, **kwargs) -> ToolResult:
        # 实现工具逻辑
        input_data = kwargs.get("input", "")
        # 处理逻辑...
        return ToolResult(output="处理结果")
```

### 创建新Agent

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection

class NewAgent(ToolCallAgent):
    name = "new_agent"
    description = "新Agent描述"

    system_prompt = "系统提示词"
    next_step_prompt = "下一步提示词"

    available_tools = ToolCollection(
        # 添加工具...
    )

    async def think(self) -> bool:
        # 自定义思考逻辑
        return await super().think()

    async def act(self) -> str:
        # 自定义行动逻辑
        return await super().act()
```

## 最佳实践

### 1. 提示词设计

- 使用清晰的系统提示词定义Agent角色
- 提供具体的下一步指导
- 包含错误处理和边界情况

### 2. 工具选择

- 根据任务类型选择合适的工具
- 避免工具功能重复
- 考虑工具的执行效率

### 3. 状态管理

- 合理设置最大步数限制
- 监控Agent执行状态
- 及时处理错误和异常

### 4. 内存管理

- 控制内存大小，避免上下文过长
- 合理使用消息类型
- 定期清理无用信息

## 故障排除

### 常见问题

1. **Agent陷入循环**
   - 检查提示词是否过于模糊
   - 调整重复检测阈值
   - 增加更多上下文信息

2. **工具执行失败**
   - 检查工具参数格式
   - 验证工具依赖环境
   - 查看错误日志

3. **内存溢出**
   - 减少最大消息数量
   - 优化提示词长度
   - 使用更高效的工具

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查Agent状态
print(f"Agent状态: {agent.state}")
print(f"当前步数: {agent.current_step}")
print(f"内存消息数: {len(agent.memory.messages)}")

# 查看工具调用历史
for msg in agent.memory.messages:
    if msg.tool_calls:
        print(f"工具调用: {msg.tool_calls}")
```

## 贡献指南

1. 遵循现有代码风格
2. 添加适当的文档和注释
3. 编写单元测试
4. 更新README文档

## 许可证

本项目遵循项目根目录的许可证条款。
