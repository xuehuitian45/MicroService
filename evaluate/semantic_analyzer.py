"""
语义分析模块

使用大模型分析微服务划分的语义相关性和业务一致性
"""

import asyncio
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import json

from split.utils.data_processor import load_json
from evaluate.config import EvaluationConfig

# AgentScope 相关导入（可选）
AGENTSCOPE_AVAILABLE = True
try:
    from agentscope.agent import ReActAgent
    from agentscope.formatter import DashScopeChatFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.message import Msg
    from agentscope.model import DashScopeChatModel
    from agentscope.tool import Toolkit
except Exception:
    AGENTSCOPE_AVAILABLE = False
    # 定义占位符类型，避免类型检查错误
    class ReActAgent: ...
    class DashScopeChatFormatter: ...
    class InMemoryMemory: ...
    class Msg: ...
    class DashScopeChatModel: ...
    class Toolkit: ...


@dataclass
class SemanticMetrics:
    """语义相关指标"""
    # 语义一致性指标
    semantic_coherence: float = 0.0  # 0-1，服务内部的语义一致性
    semantic_coupling: float = 0.0   # 0-1，服务间的语义耦合度
    business_alignment: float = 0.0  # 0-1，与业务领域的对齐度
    
    # 详细分析（仅指标相关）
    service_semantic_analysis: Dict[int, Dict] = field(default_factory=dict)  # 各服务的语义分析
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'semantic_coherence': round(self.semantic_coherence, 4),
            'semantic_coupling': round(self.semantic_coupling, 4),
            'business_alignment': round(self.business_alignment, 4),
            'service_semantic_analysis': self.service_semantic_analysis
        }


class AgentFactory:
    """ReActAgent 工厂类"""
    
    @staticmethod
    def create_sync_react_agent(
            agent_name: str = "SemanticAnalyzer",
            sys_prompt: str = "你是一个专业的微服务架构分析助手，擅长分析代码的语义关系和业务一致性",
            model_name: str = "qwen-max",
            api_key: str = None,
            stream: bool = False,
            enable_thinking: bool = False,
            toolkit: Toolkit = None,
    ) -> ReActAgent:
        """
        同步创建 ReAct 智能体（内部处理异步初始化）

        Args:
            agent_name: 智能体名称
            sys_prompt: 系统提示词
            model_name: 通义模型名称
            api_key: 通义 API-KEY
            stream: 是否流式输出
            enable_thinking: 是否开启思考过程输出
            toolkit: 工具集实例

        Returns:
            初始化完成的 ReActAgent 实例
        """
        # 处理 API-KEY
        if not api_key:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("请传入 api_key 或设置环境变量 DASHSCOPE_API_KEY")

        # 创建通义模型实例
        model = DashScopeChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=stream,
            enable_thinking=enable_thinking,
        )

        if toolkit is None:
            toolkit = Toolkit()

        # 创建并返回 ReAct 智能体
        return ReActAgent(
            name=agent_name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )

    @staticmethod
    def run_sync_agent(agent: ReActAgent, user_query: str) -> str:
        """
        同步调用智能体处理用户请求

        Args:
            agent: 已初始化的 ReActAgent 实例
            user_query: 用户输入的查询/任务

        Returns:
            智能体返回的响应内容（字符串）
        """
        # 构造用户消息
        user_msg = Msg(
            name="user",
            content=user_query,
            role="user",
        )

        # 同步运行异步的 agent 调用
        response = asyncio.run(agent(user_msg))

        # 提取响应内容并返回字符串
        if hasattr(response, 'content'):
            return str(response.content)
        elif isinstance(response, str):
            return response
        else:
            return str(response)


class SemanticAnalyzer:
    """语义分析器"""
    
    def __init__(self, config: EvaluationConfig = None, agent: ReActAgent = None):
        """
        初始化语义分析器
        
        Args:
            config: 评估配置
            agent: ReActAgent 实例，如果为 None 则自动创建
        """
        self.config = config or EvaluationConfig()
        self.data = None
        self.partition = None
        self.class_id_to_name = {}
        self.class_name_to_id = {}
        
        # 初始化或使用提供的 agent
        if agent is None:
            try:
                self.agent = AgentFactory.create_sync_react_agent()
            except ValueError as e:
                print(f"⚠️  无法初始化 Agent: {e}")
                print("   将使用本地分析模式（不调用大模型）")
                self.agent = None
        else:
            self.agent = agent
    
    def load_data(self) -> None:
        """加载数据和划分结果"""
        # 加载数据
        self.data = load_json(self.config.data_path)
        
        # 构建ID到名称的映射
        for node in self.data:
            self.class_id_to_name[node['id']] = node['name']
            self.class_name_to_id[node['name']] = node['id']
        
        # 加载划分结果
        with open(self.config.result_path, 'r') as f:
            partition_dict = json.load(f)
        
        # 转换划分结果为 {class_id: service_id}
        self.partition = {}
        for service_id, class_names in partition_dict.items():
            for class_name in class_names:
                if class_name in self.class_name_to_id:
                    self.partition[self.class_name_to_id[class_name]] = int(service_id)
    
    def analyze_semantic_coherence(self) -> SemanticMetrics:
        """
        分析语义一致性
        
        Returns:
            SemanticMetrics: 语义指标
        """
        if self.data is None:
            self.load_data()
        
        metrics = SemanticMetrics()
        
        # 获取各服务的类和描述
        services = {}
        for node_id, service_id in self.partition.items():
            if service_id not in services:
                services[service_id] = []
            
            class_info = next((n for n in self.data if n['id'] == node_id), None)
            if class_info:
                services[service_id].append({
                    'name': class_info['name'],
                    'description': class_info['description']
                })
        
        # 分析每个服务的语义一致性
        total_coherence = 0
        for service_id, classes in services.items():
            coherence = self._analyze_service_semantic_coherence(service_id, classes)
            metrics.service_semantic_analysis[service_id] = {
                'semantic_coherence': coherence,
                'class_count': len(classes),
                'classes': [c['name'] for c in classes]
            }
            total_coherence += coherence
        
        # 计算平均语义一致性
        if services:
            metrics.semantic_coherence = total_coherence / len(services)
        
        # 分析语义耦合度和业务对齐度
        metrics.semantic_coupling = self._analyze_semantic_coupling(services)
        metrics.business_alignment = self._analyze_business_alignment(services)
        
        return metrics
    
    def _analyze_service_semantic_coherence(self, service_id: int, classes: List[Dict]) -> float:
        """
        分析单个服务的语义一致性
        
        Args:
            service_id: 服务ID
            classes: 服务中的类列表
        
        Returns:
            语义一致性分数 (0-1)
        """
        if not classes or len(classes) < 2:
            return 1.0
        
        if self.agent is None:
            # 本地分析模式：基于类名相似性的简单启发式方法
            return self._local_semantic_coherence_analysis(classes)
        
        # 使用大模型进行语义分析
        class_descriptions = "\n".join([
            f"- {c['name']}: {c['description']}"
            for c in classes[:10]  # 限制数量以避免 token 过多
        ])
        
        query = f"""
        分析以下微服务中的类是否在语义上一致（即它们是否属于同一个业务领域或功能模块）。
        
        服务中的类：
        {class_descriptions}
        
        请评估这些类的语义一致性，返回一个 0-1 之间的分数，其中：
        - 1.0 表示完全一致（所有类都属于同一业务领域）
        - 0.5 表示中等一致（大部分类相关，但有一些不相关）
        - 0.0 表示完全不一致（类来自不同的业务领域）
        
        只返回数字，不要返回其他内容。
        """
        
        try:
            response = AgentFactory.run_sync_agent(self.agent, query)
            # 提取数字
            score = float(''.join(c for c in response if c.isdigit() or c == '.'))
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️  大模型分析失败: {e}，使用本地分析")
            return self._local_semantic_coherence_analysis(classes)
    
    def _local_semantic_coherence_analysis(self, classes: List[Dict]) -> float:
        """
        本地语义一致性分析（不依赖大模型）
        
        基于类名和描述的关键词相似性
        """
        if len(classes) < 2:
            return 1.0
        
        # 提取所有描述中的关键词
        keywords_list = []
        for cls in classes:
            name = cls['name'].lower()
            desc = cls['description'].lower()
            
            # 简单的关键词提取（基于常见的业务术语）
            keywords = set()
            
            # 从类名提取
            for part in name.split('_'):
                if len(part) > 3:
                    keywords.add(part)
            
            # 从描述提取关键词
            common_keywords = [
                'ping', 'servlet', 'ejb', 'cdi', 'bean', 'session',
                'trade', 'quote', 'market', 'account', 'order', 'holding',
                'json', 'websocket', 'database', 'jdbc', 'jndi',
                'executor', 'thread', 'timer', 'mdb', 'message'
            ]
            
            for keyword in common_keywords:
                if keyword in desc or keyword in name:
                    keywords.add(keyword)
            
            keywords_list.append(keywords)
        
        # 计算关键词重叠度
        if not keywords_list:
            return 0.5
        
        # 计算所有类对之间的相似性
        similarities = []
        for i in range(len(keywords_list)):
            for j in range(i + 1, len(keywords_list)):
                if keywords_list[i] or keywords_list[j]:
                    intersection = len(keywords_list[i] & keywords_list[j])
                    union = len(keywords_list[i] | keywords_list[j])
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
        
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.5
    
    def _analyze_semantic_coupling(self, services: Dict) -> float:
        """
        分析服务间的语义耦合度
        
        Args:
            services: 服务字典
        
        Returns:
            语义耦合度 (0-1)
        """
        if len(services) < 2:
            return 0.0
        
        if self.agent is None:
            # 本地分析：基于类名相似性
            return self._local_semantic_coupling_analysis(services)
        
        # 使用大模型分析
        service_summaries = []
        for service_id, classes in list(services.items()):  # 限制数量
            class_names = [c['name'] for c in classes]
            service_summaries.append(f"Service-{service_id}: {', '.join(class_names)}")
        
        query = f"""
        分析以下微服务之间的语义耦合度（即它们在业务上的关联程度）。
        
        服务列表：
        {chr(10).join(service_summaries)}
        
        请评估这些服务之间的语义耦合度，返回一个 0-1 之间的分数，其中：
        - 1.0 表示高度耦合（服务之间有很强的业务关联）
        - 0.5 表示中等耦合（有一些业务关联）
        - 0.0 表示完全解耦（服务之间没有业务关联）
        
        只返回数字，不要返回其他内容。
        """
        
        try:
            response = AgentFactory.run_sync_agent(self.agent, query)
            score = float(''.join(c for c in response if c.isdigit() or c == '.'))
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️  大模型分析失败: {e}，使用本地分析")
            return self._local_semantic_coupling_analysis(services)
    
    def _local_semantic_coupling_analysis(self, services: Dict) -> float:
        """本地语义耦合度分析"""
        if len(services) < 2:
            return 0.0
        
        # 计算服务间的关键词重叠
        service_keywords = {}
        for service_id, classes in services.items():
            keywords = set()
            for cls in classes:
                name = cls['name'].lower()
                for part in name.split('_'):
                    if len(part) > 3:
                        keywords.add(part)
            service_keywords[service_id] = keywords
        
        # 计算服务对之间的相似性
        similarities = []
        service_ids = list(service_keywords.keys())
        for i in range(len(service_ids)):
            for j in range(i + 1, len(service_ids)):
                kw1 = service_keywords[service_ids[i]]
                kw2 = service_keywords[service_ids[j]]
                if kw1 or kw2:
                    intersection = len(kw1 & kw2)
                    union = len(kw1 | kw2)
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
        
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0
    
    def _analyze_business_alignment(self, services: Dict) -> float:
        """
        分析与业务领域的对齐度
        
        Args:
            services: 服务字典
        
        Returns:
            业务对齐度 (0-1)
        """
        if self.agent is None:
            return self._local_business_alignment_analysis(services)
        
        # 使用大模型分析
        service_summaries = []
        for service_id, classes in list(services.items()):
            class_names = [c['name'] for c in classes]
            service_summaries.append(f"Service-{service_id}: {', '.join(class_names)}")
        
        query = f"""
        分析以下微服务划分是否与业务领域对齐（即每个服务是否代表一个清晰的业务功能或领域）。
        
        服务列表：
        {chr(10).join(service_summaries)}
        
        请评估这个划分与业务领域的对齐度，返回一个 0-1 之间的分数，其中：
        - 1.0 表示完全对齐（每个服务都代表一个清晰的业务功能）
        - 0.5 表示部分对齐（大多数服务对齐，但有一些混乱）
        - 0.0 表示完全不对齐（服务划分与业务无关）
        
        只返回数字，不要返回其他内容。
        """
        
        try:
            response = AgentFactory.run_sync_agent(self.agent, query)
            score = float(''.join(c for c in response if c.isdigit() or c == '.'))
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️  大模型分析失败: {e}，使用本地分析")
            return self._local_business_alignment_analysis(services)
    
    def _local_business_alignment_analysis(self, services: Dict) -> float:
        """本地业务对齐度分析"""
        # 识别常见的业务领域
        business_domains = {
            'ping': ['ping', 'servlet', 'interceptor', 'websocket'],
            'trade': ['trade', 'quote', 'market', 'order', 'account', 'holding'],
            'json': ['json', 'encoder', 'decoder', 'message'],
            'database': ['jdbc', 'database', 'bean', 'data'],
            'execution': ['executor', 'thread', 'timer', 'mdb']
        }
        
        # 为每个服务分配业务领域
        service_domains = {}
        for service_id, classes in services.items():
            domain_scores = {domain: 0 for domain in business_domains}
            
            for cls in classes:
                name = cls['name'].lower()
                for domain, keywords in business_domains.items():
                    for keyword in keywords:
                        if keyword in name:
                            domain_scores[domain] += 1
            
            # 找到主要领域
            if max(domain_scores.values()) > 0:
                main_domain = max(domain_scores, key=domain_scores.get)
                service_domains[service_id] = main_domain
            else:
                service_domains[service_id] = 'unknown'
        
        # 计算对齐度（有多少服务有明确的业务领域）
        aligned_services = sum(1 for domain in service_domains.values() if domain != 'unknown')
        return aligned_services / len(services) if services else 0.0

    def print_semantic_report(self, metrics: SemanticMetrics) -> None:
        """打印语义分析报告"""
        print("\n" + "=" * 80)
        print("微服务划分语义分析报告")
        print("=" * 80)
        
        print("\n【语义指标】")
        print(f"  语义一致性: {metrics.semantic_coherence:.4f} ({metrics.semantic_coherence*100:.2f}%)")
        print(f"  语义耦合度: {metrics.semantic_coupling:.4f} ({metrics.semantic_coupling*100:.2f}%)")
        print(f"  业务对齐度: {metrics.business_alignment:.4f} ({metrics.business_alignment*100:.2f}%)")
        
        print("\n【各服务语义分析】")
        for service_id in sorted(metrics.service_semantic_analysis.keys()):
            info = metrics.service_semantic_analysis[service_id]
            print(f"\n  Service-{service_id}:")
            print(f"    语义一致性: {info['semantic_coherence']:.4f}")
            print(f"    类数: {info['class_count']}")
            print(f"    包含的类: {', '.join(info['classes'][:3])}")
            if len(info['classes']) > 3:
                print(f"              ... 还有 {len(info['classes']) - 3} 个类")
        
        print("\n" + "=" * 80)
    
    def save_semantic_report(self, metrics: SemanticMetrics, output_path: str) -> None:
        """保存语义分析报告"""
        report = {
            'semantic_metrics': metrics.to_dict(),
            'summary': {
                'semantic_coherence_level': 'Excellent' if metrics.semantic_coherence >= 0.8 else
                                           'Good' if metrics.semantic_coherence >= 0.6 else
                                           'Fair' if metrics.semantic_coherence >= 0.4 else
                                           'Poor',
                'semantic_coupling_level': 'Low' if metrics.semantic_coupling <= 0.3 else
                                          'Medium' if metrics.semantic_coupling <= 0.6 else
                                          'High',
                'business_alignment_level': 'Excellent' if metrics.business_alignment >= 0.8 else
                                           'Good' if metrics.business_alignment >= 0.6 else
                                           'Fair'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 语义分析报告已保存到: {output_path}")


def analyze_semantic(
    result_path: str = None,
    data_path: str = None,
    output_path: str = None,
    print_report: bool = True,
    agent: ReActAgent = None
) -> SemanticMetrics:
    """
    快速进行语义分析
    
    Args:
        result_path: 划分结果文件路径
        data_path: 数据文件路径
        output_path: 报告输出路径
        print_report: 是否打印报告
        agent: ReActAgent 实例
    
    Returns:
        SemanticMetrics: 语义指标
    """
    config = EvaluationConfig()
    if result_path:
        config.result_path = result_path
    if data_path:
        config.data_path = data_path
    
    analyzer = SemanticAnalyzer(config, agent)
    analyzer.load_data()
    metrics = analyzer.analyze_semantic_coherence()
    
    if print_report:
        analyzer.print_semantic_report(metrics)
    
    if output_path:
        analyzer.save_semantic_report(metrics, output_path)
    
    return metrics

# 兼容用户提供的调用方式：在模块层暴露相同命名的函数

def create_sync_react_agent(
        agent_name: str = "Jarvis",
        sys_prompt: str = "你是一个名为 Jarvis 的助手，擅长完成用户的各类需求",
        model_name: str = "qwen-max",
        api_key: str = None,
        stream: bool = False,
        enable_thinking: bool = False,
        toolkit: Toolkit = None,
) -> ReActAgent:
    return AgentFactory.create_sync_react_agent(
        agent_name=agent_name,
        sys_prompt=sys_prompt,
        model_name=model_name,
        api_key=api_key,
        stream=stream,
        enable_thinking=enable_thinking,
        toolkit=toolkit,
    )


def run_sync_agent(agent: ReActAgent, user_query: str) -> str:
    return AgentFactory.run_sync_agent(agent, user_query)

