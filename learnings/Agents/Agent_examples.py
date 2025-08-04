# AI智能体框架核心技术示例代码

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from datetime import datetime

# ============================================================================
# 1. ICL (In-Context Learning) - 上下文学习示例
# ============================================================================

class ICLAgent:
    """基于ICL的智能体示例"""
    
    def __init__(self):
        self.task_templates = {
            "sentiment_analysis": {
                "examples": [
                    {"text": "这个产品很棒！", "sentiment": "positive"},
                    {"text": "质量太差了", "sentiment": "negative"},
                    {"text": "一般般吧", "sentiment": "neutral"}
                ],
                "prompt_template": "分析以下文本的情感倾向：\n{examples}\n\n待分析文本：{input_text}\n情感倾向："
            },
            "translation": {
                "examples": [
                    {"en": "Hello world", "zh": "你好世界"},
                    {"en": "Good morning", "zh": "早上好"},
                    {"en": "Thank you", "zh": "谢谢"}
                ],
                "prompt_template": "将以下英文翻译成中文：\n{examples}\n\n待翻译：{input_text}\n翻译："
            }
        }
    
    def create_icl_prompt(self, task_type: str, input_text: str, num_examples: int = 3) -> str:
        """创建ICL prompt"""
        if task_type not in self.task_templates:
            raise ValueError(f"未知任务类型: {task_type}")
        
        template = self.task_templates[task_type]
        examples = template["examples"][:num_examples]
        
        # 构建示例字符串
        example_str = ""
        for example in examples:
            if task_type == "sentiment_analysis":
                example_str += f"文本：{example['text']}\n情感：{example['sentiment']}\n\n"
            elif task_type == "translation":
                example_str += f"英文：{example['en']}\n中文：{example['zh']}\n\n"
        
        return template["prompt_template"].format(
            examples=example_str.strip(),
            input_text=input_text
        )
    
    def execute_icl_task(self, task_type: str, input_text: str) -> str:
        """执行ICL任务"""
        prompt = self.create_icl_prompt(task_type, input_text)
        # 这里应该调用实际的LLM API
        # return llm.generate(prompt)
        return f"基于ICL的{task_type}任务结果: {input_text}"

# 使用示例
def demo_icl():
    agent = ICLAgent()
    
    # 情感分析示例
    sentiment_result = agent.execute_icl_task("sentiment_analysis", "这个电影太精彩了！")
    print("ICL情感分析:", sentiment_result)
    
    # 翻译示例
    translation_result = agent.execute_icl_task("translation", "I love this movie")
    print("ICL翻译:", translation_result)

# ============================================================================
# 2. CoT (Chain of Thought) - 思维链推理示例
# ============================================================================

class CoTAgent:
    """基于CoT的智能体示例"""
    
    def __init__(self):
        self.reasoning_templates = {
            "math_problem": "让我们一步步解决这个问题：\n{problem}\n\n思考过程：",
            "logical_reasoning": "让我们分析这个逻辑问题：\n{problem}\n\n推理步骤：",
            "code_generation": "让我们分析需求并生成代码：\n{requirement}\n\n分析过程："
        }
    
    def solve_math_problem(self, problem: str) -> Dict[str, str]:
        """解决数学问题的CoT示例"""
        prompt = self.reasoning_templates["math_problem"].format(problem=problem)
        
        # 模拟CoT推理过程
        reasoning_steps = [
            "1. 首先理解题目要求",
            "2. 识别已知条件和未知量",
            "3. 选择合适的解题方法",
            "4. 逐步计算得出结果",
            "5. 验证答案的正确性"
        ]
        
        return {
            "problem": problem,
            "reasoning_steps": reasoning_steps,
            "final_answer": "基于CoT推理的答案",
            "confidence": 0.95
        }
    
    def logical_reasoning(self, problem: str) -> Dict[str, Any]:
        """逻辑推理的CoT示例"""
        prompt = self.reasoning_templates["logical_reasoning"].format(problem=problem)
        
        reasoning_steps = [
            "1. 分析前提条件",
            "2. 识别逻辑关系",
            "3. 应用推理规则",
            "4. 得出结论",
            "5. 检查推理的有效性"
        ]
        
        return {
            "problem": problem,
            "reasoning_steps": reasoning_steps,
            "conclusion": "基于逻辑推理的结论",
            "validity": "有效"
        }

# 使用示例
def demo_cot():
    agent = CoTAgent()
    
    # 数学问题求解
    math_result = agent.solve_math_problem("如果x + 2y = 10，且2x - y = 5，求x和y的值")
    print("CoT数学求解:", json.dumps(math_result, indent=2, ensure_ascii=False))
    
    # 逻辑推理
    logic_result = agent.logical_reasoning("如果所有A都是B，所有B都是C，那么所有A都是C吗？")
    print("CoT逻辑推理:", json.dumps(logic_result, indent=2, ensure_ascii=False))

# ============================================================================
# 3. 记忆机制设计示例
# ============================================================================

@dataclass
class MemoryItem:
    """记忆项数据结构"""
    content: str
    memory_type: str  # 'conversation', 'user_preference', 'knowledge', 'task'
    timestamp: datetime
    importance: float  # 0-1，重要性评分
    access_count: int = 0
    last_accessed: datetime = None

class MemorySystem:
    """记忆系统实现"""
    
    def __init__(self, max_memory_size: int = 1000):
        self.memories: List[MemoryItem] = []
        self.max_memory_size = max_memory_size
        self.memory_index = defaultdict(list)  # 简单的关键词索引
        
    def add_memory(self, content: str, memory_type: str, importance: float = 0.5):
        """添加新记忆"""
        memory_item = MemoryItem(
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance
        )
        
        self.memories.append(memory_item)
        self._update_index(memory_item)
        
        # 如果记忆过多，进行清理
        if len(self.memories) > self.max_memory_size:
            self._cleanup_memories()
    
    def retrieve_memories(self, query: str, memory_type: str = None, limit: int = 5) -> List[MemoryItem]:
        """检索相关记忆"""
        relevant_memories = []
        
        for memory in self.memories:
            # 类型过滤
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # 简单的关键词匹配
            if any(keyword in memory.content.lower() for keyword in query.lower().split()):
                relevant_memories.append(memory)
                memory.access_count += 1
                memory.last_accessed = datetime.now()
        
        # 按重要性和访问频率排序
        relevant_memories.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
        return relevant_memories[:limit]
    
    def update_memory_importance(self, memory_id: int, new_importance: float):
        """更新记忆重要性"""
        if 0 <= memory_id < len(self.memories):
            self.memories[memory_id].importance = new_importance
    
    def _update_index(self, memory_item: MemoryItem):
        """更新索引"""
        words = memory_item.content.lower().split()
        for word in words:
            self.memory_index[word].append(memory_item)
    
    def _cleanup_memories(self):
        """清理不重要的记忆"""
        # 按重要性和访问频率排序，保留重要的
        self.memories.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
        self.memories = self.memories[:self.max_memory_size // 2]

# 使用示例
def demo_memory():
    memory_system = MemorySystem()
    
    # 添加不同类型的记忆
    memory_system.add_memory("用户喜欢简洁的回答", "user_preference", 0.8)
    memory_system.add_memory("Python是编程语言", "knowledge", 0.9)
    memory_system.add_memory("上次对话讨论了机器学习", "conversation", 0.6)
    memory_system.add_memory("成功完成了数据分析任务", "task", 0.7)
    
    # 检索记忆
    user_memories = memory_system.retrieve_memories("用户偏好", "user_preference")
    print("用户偏好记忆:", [m.content for m in user_memories])
    
    knowledge_memories = memory_system.retrieve_memories("编程", "knowledge")
    print("知识记忆:", [m.content for m in knowledge_memories])

# ============================================================================
# 4. 自进化机制示例
# ============================================================================

@dataclass
class PerformanceMetric:
    """性能指标"""
    task_type: str
    success_rate: float
    response_time: float
    user_satisfaction: float
    timestamp: datetime

class SelfEvolutionAgent:
    """自进化智能体"""
    
    def __init__(self):
        self.performance_history: List[PerformanceMetric] = []
        self.current_strategies = {
            "response_style": "concise",  # concise, detailed, friendly
            "reasoning_depth": "medium",  # shallow, medium, deep
            "memory_usage": "moderate",   # minimal, moderate, extensive
            "confidence_threshold": 0.7
        }
        self.evolution_rules = {
            "success_rate_threshold": 0.8,
            "satisfaction_threshold": 0.7,
            "adaptation_rate": 0.1
        }
    
    def record_performance(self, task_type: str, success: bool, response_time: float, 
                          user_satisfaction: float):
        """记录性能指标"""
        metric = PerformanceMetric(
            task_type=task_type,
            success_rate=1.0 if success else 0.0,
            response_time=response_time,
            user_satisfaction=user_satisfaction,
            timestamp=datetime.now()
        )
        self.performance_history.append(metric)
    
    def analyze_performance(self, task_type: str = None) -> Dict[str, float]:
        """分析性能表现"""
        if task_type:
            metrics = [m for m in self.performance_history if m.task_type == task_type]
        else:
            metrics = self.performance_history
        
        if not metrics:
            return {}
        
        recent_metrics = metrics[-10:]  # 最近10次
        
        return {
            "avg_success_rate": np.mean([m.success_rate for m in recent_metrics]),
            "avg_response_time": np.mean([m.response_time for m in recent_metrics]),
            "avg_satisfaction": np.mean([m.user_satisfaction for m in recent_metrics])
        }
    
    def evolve_strategies(self):
        """根据性能表现进化策略"""
        performance = self.analyze_performance()
        
        if not performance:
            return
        
        # 根据成功率调整策略
        if performance["avg_success_rate"] < self.evolution_rules["success_rate_threshold"]:
            # 降低置信度阈值，增加推理深度
            self.current_strategies["confidence_threshold"] *= (1 - self.evolution_rules["adaptation_rate"])
            self.current_strategies["reasoning_depth"] = "deep"
        
        # 根据用户满意度调整响应风格
        if performance["avg_satisfaction"] < self.evolution_rules["satisfaction_threshold"]:
            if self.current_strategies["response_style"] == "concise":
                self.current_strategies["response_style"] = "detailed"
            elif self.current_strategies["response_style"] == "detailed":
                self.current_strategies["response_style"] = "friendly"
        
        # 根据响应时间调整记忆使用
        if performance["avg_response_time"] > 2.0:  # 超过2秒
            self.current_strategies["memory_usage"] = "minimal"
    
    def get_current_strategy(self) -> Dict[str, str]:
        """获取当前策略"""
        return self.current_strategies.copy()

# 使用示例
def demo_self_evolution():
    agent = SelfEvolutionAgent()
    
    # 模拟性能记录
    agent.record_performance("qa", True, 1.5, 0.8)
    agent.record_performance("qa", False, 2.1, 0.6)
    agent.record_performance("qa", True, 1.8, 0.9)
    
    # 分析性能
    performance = agent.analyze_performance("qa")
    print("性能分析:", performance)
    
    # 进化策略
    agent.evolve_strategies()
    current_strategy = agent.get_current_strategy()
    print("进化后的策略:", current_strategy)

# ============================================================================
# 5. 多智能体协同机制示例
# ============================================================================

@dataclass
class Agent:
    """智能体基类"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    current_task: Optional[str] = None
    status: str = "idle"  # idle, busy, completed, failed

class MultiAgentSystem:
    """多智能体协同系统"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.task_queue: List[Dict] = []
        self.communication_log: List[Dict] = []
        self.collaboration_patterns = {
            "master_slave": self._master_slave_collaboration,
            "peer_to_peer": self._peer_to_peer_collaboration,
            "competitive": self._competitive_collaboration
        }
    
    def add_agent(self, agent_id: str, agent_type: str, capabilities: List[str]):
        """添加智能体"""
        self.agents[agent_id] = Agent(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities
        )
    
    def submit_task(self, task: Dict):
        """提交任务"""
        self.task_queue.append(task)
    
    def execute_collaboration(self, task_id: str, pattern: str = "master_slave"):
        """执行协作任务"""
        if pattern not in self.collaboration_patterns:
            raise ValueError(f"未知的协作模式: {pattern}")
        
        return self.collaboration_patterns[pattern](task_id)
    
    def _master_slave_collaboration(self, task_id: str) -> Dict:
        """主从协作模式"""
        # 选择主智能体（通常是通用型）
        master_agent = None
        slave_agents = []
        
        for agent in self.agents.values():
            if "general" in agent.agent_type:
                master_agent = agent
            else:
                slave_agents.append(agent)
        
        if not master_agent:
            raise ValueError("未找到主智能体")
        
        # 主智能体分解任务
        subtasks = self._decompose_task(task_id)
        
        # 分配子任务给从智能体
        task_assignments = {}
        for i, subtask in enumerate(subtasks):
            if i < len(slave_agents):
                slave_agents[i].current_task = subtask
                slave_agents[i].status = "busy"
                task_assignments[slave_agents[i].agent_id] = subtask
        
        # 收集结果
        results = {}
        for agent_id, subtask in task_assignments.items():
            results[agent_id] = f"完成子任务: {subtask}"
            self.agents[agent_id].status = "completed"
        
        # 主智能体整合结果
        final_result = master_agent.agent_id + "整合结果: " + str(results)
        
        return {
            "task_id": task_id,
            "pattern": "master_slave",
            "master_agent": master_agent.agent_id,
            "slave_agents": list(task_assignments.keys()),
            "results": results,
            "final_result": final_result
        }
    
    def _peer_to_peer_collaboration(self, task_id: str) -> Dict:
        """对等协作模式"""
        # 重置所有智能体状态为idle
        for agent in self.agents.values():
            agent.status = "idle"
            agent.current_task = None
        
        available_agents = [agent for agent in self.agents.values() if agent.status == "idle"]
        
        if len(available_agents) < 2:
            raise ValueError("对等协作需要至少2个可用智能体")
        
        # 智能体平等协作
        collaboration_result = {}
        for agent in available_agents:
            agent.status = "busy"
            agent.current_task = task_id
            collaboration_result[agent.agent_id] = f"协作完成: {task_id}"
            agent.status = "completed"
        
        return {
            "task_id": task_id,
            "pattern": "peer_to_peer",
            "participants": [agent.agent_id for agent in available_agents],
            "results": collaboration_result
        }
    
    def _competitive_collaboration(self, task_id: str) -> Dict:
        """竞争协作模式"""
        # 重置所有智能体状态为idle
        for agent in self.agents.values():
            agent.status = "idle"
            agent.current_task = None
        
        available_agents = [agent for agent in self.agents.values() if agent.status == "idle"]
        
        if len(available_agents) < 2:
            raise ValueError("竞争协作需要至少2个可用智能体")
        
        # 智能体竞争解决同一任务
        solutions = {}
        for agent in available_agents:
            agent.status = "busy"
            agent.current_task = task_id
            # 模拟不同智能体给出不同解决方案
            solutions[agent.agent_id] = f"解决方案{agent.agent_id}: {task_id}"
            agent.status = "completed"
        
        # 选择最佳解决方案
        best_solution = max(solutions.items(), key=lambda x: len(x[1]))
        
        return {
            "task_id": task_id,
            "pattern": "competitive",
            "participants": [agent.agent_id for agent in available_agents],
            "solutions": solutions,
            "best_solution": best_solution
        }
    
    def _decompose_task(self, task_id: str) -> List[str]:
        """任务分解"""
        return [f"子任务{i}" for i in range(3)]  # 简化为3个子任务
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "total_agents": len(self.agents),
            "idle_agents": len([a for a in self.agents.values() if a.status == "idle"]),
            "busy_agents": len([a for a in self.agents.values() if a.status == "busy"]),
            "completed_agents": len([a for a in self.agents.values() if a.status == "completed"]),
            "pending_tasks": len(self.task_queue)
        }

# 使用示例
def demo_multi_agent():
    system = MultiAgentSystem()
    
    # 添加不同类型的智能体
    system.add_agent("agent_1", "general", ["planning", "coordination"])
    system.add_agent("agent_2", "specialist", ["data_analysis"])
    system.add_agent("agent_3", "specialist", ["code_generation"])
    system.add_agent("agent_4", "specialist", ["visualization"])
    
    # 提交任务
    system.submit_task({"id": "task_1", "description": "数据分析项目"})
    
    # 执行主从协作
    result = system.execute_collaboration("task_1", "master_slave")
    print("主从协作结果:", json.dumps(result, indent=2, ensure_ascii=False))
    
    # 执行对等协作
    result2 = system.execute_collaboration("task_1", "peer_to_peer")
    print("对等协作结果:", json.dumps(result2, indent=2, ensure_ascii=False))
    
    # 查看系统状态
    status = system.get_system_status()
    print("系统状态:", status)

# ============================================================================
# 主函数：运行所有示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("AI智能体框架核心技术示例")
    print("=" * 50)
    
    print("\n1. ICL (In-Context Learning) 示例:")
    demo_icl()
    
    print("\n2. CoT (Chain of Thought) 示例:")
    demo_cot()
    
    print("\n3. 记忆机制示例:")
    demo_memory()
    
    print("\n4. 自进化机制示例:")
    demo_self_evolution()
    
    print("\n5. 多智能体协同机制示例:")
    demo_multi_agent()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")
    print("=" * 50) 