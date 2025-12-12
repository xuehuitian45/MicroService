"""
微服务划分质量评估模块

包含以下评估指标：
1. 内聚度 (Cohesion): 衡量服务内部的紧密程度
2. 耦合度 (Coupling): 衡量服务之间的依赖程度
3. 模块性 (Modularity): 衡量整体的模块化程度
4. 平衡度 (Balance): 衡量各服务规模的均衡程度
5. 可维护性指数 (Maintainability Index): 综合评分
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import json
import math
from collections import defaultdict

from split.utils.data_processor import load_json
from .config import EvaluationConfig
from .semantic_analyzer import SemanticMetrics


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    # 基础指标
    cohesion: float = 0.0  # 内聚度 (0-1)
    coupling: float = 0.0  # 耦合度 (0-1)
    modularity: float = 0.0  # 模块性 (-0.5-1)
    balance: float = 0.0  # 平衡度 (0-1)
    
    # 详细指标
    intra_edges: int = 0  # 服务内部边数
    inter_edges: int = 0  # 服务间边数
    total_edges: int = 0  # 总边数
    
    # 服务规模指标
    avg_service_size: float = 0.0  # 平均服务规模
    std_service_size: float = 0.0  # 服务规模标准差
    max_service_size: int = 0  # 最大服务规模
    min_service_size: int = 0  # 最小服务规模
    
    # 综合指标
    maintainability_index: float = 0.0  # 可维护性指数 (0-100)
    quality_score: float = 0.0  # 综合质量分数 (0-100)

    # 语义指标
    semantic_coherence: float = 0.0  # 服务内语义一致性 (0-1)
    semantic_coupling: float = 0.0   # 服务间语义耦合度 (0-1)
    business_alignment: float = 0.0  # 业务对齐度 (0-1)

    # 详细分析
    service_metrics: Dict[int, Dict] = field(default_factory=dict)  # 各服务的详细指标
    service_semantic_analysis: Dict[int, Dict] = field(default_factory=dict)  # 各服务的语义分析
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'cohesion': round(self.cohesion, 4),
            'coupling': round(self.coupling, 4),
            'modularity': round(self.modularity, 4),
            'balance': round(self.balance, 4),
            'intra_edges': self.intra_edges,
            'inter_edges': self.inter_edges,
            'total_edges': self.total_edges,
            'avg_service_size': round(self.avg_service_size, 2),
            'std_service_size': round(self.std_service_size, 2),
            'max_service_size': self.max_service_size,
            'min_service_size': self.min_service_size,
            'maintainability_index': round(self.maintainability_index, 2),
            'quality_score': round(self.quality_score, 2),
            # 语义指标
            'semantic_coherence': round(self.semantic_coherence, 4),
            'semantic_coupling': round(self.semantic_coupling, 4),
            'business_alignment': round(self.business_alignment, 4),
            'service_semantic_analysis': self.service_semantic_analysis,
            # 服务指标
            'service_metrics': self.service_metrics
        }


class Evaluator:
    """微服务划分评估器"""
    
    def __init__(self, config: EvaluationConfig = None):
        """
        初始化评估器
        
        Args:
            config: 评估配置
        """
        self.config = config or EvaluationConfig()
        self.data = None
        self.partition = None
        self.class_id_to_name = {}
        self.class_name_to_id = {}
        self.dependency_graph = None
        self.metrics: EvaluationMetrics | None = None  # 存储完整指标（含语义）
        
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
        
        # 构建依赖图
        self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> None:
        """构建依赖图"""
        self.dependency_graph = defaultdict(set)
        
        for node in self.data:
            node_id = node['id']
            for dep_id in node['dependencies']:
                self.dependency_graph[node_id].add(dep_id)
    
    def evaluate(self) -> EvaluationMetrics:
        """
        执行完整的评估
        
        Returns:
            EvaluationMetrics: 评估结果
        """
        if self.data is None:
            self.load_data()
        
        metrics = EvaluationMetrics()
        
        # 计算各项指标
        self._calculate_edge_metrics(metrics)
        self._calculate_cohesion(metrics)
        self._calculate_coupling(metrics)
        self._calculate_modularity(metrics)
        self._calculate_balance(metrics)
        self._calculate_service_metrics(metrics)
        self._calculate_maintainability_index(metrics)
        self._calculate_quality_score(metrics)
        
        # 存入实例，便于外部模块或后续步骤补充语义指标
        self.metrics = metrics
        return metrics
    
    def _calculate_edge_metrics(self, metrics: EvaluationMetrics) -> None:
        """计算边相关的指标"""
        intra_edges = 0  # 服务内部边
        inter_edges = 0  # 服务间边
        
        for node_id, dependencies in self.dependency_graph.items():
            if node_id not in self.partition:
                continue
            
            source_service = self.partition[node_id]
            
            for dep_id in dependencies:
                if dep_id not in self.partition:
                    continue
                
                target_service = self.partition[dep_id]
                
                if source_service == target_service:
                    intra_edges += 1
                else:
                    inter_edges += 1
        
        metrics.intra_edges = intra_edges
        metrics.inter_edges = inter_edges
        metrics.total_edges = intra_edges + inter_edges
    
    def _calculate_cohesion(self, metrics: EvaluationMetrics) -> None:
        """
        计算内聚度
        
        内聚度 = 服务内部边数 / 总边数
        范围: 0-1，越高越好
        """
        if metrics.total_edges == 0:
            metrics.cohesion = 0.0
        else:
            metrics.cohesion = metrics.intra_edges / metrics.total_edges
    
    def _calculate_coupling(self, metrics: EvaluationMetrics) -> None:
        """
        计算耦合度
        
        耦合度 = 1 - (服务间边数 / 总边数)
        范围: 0-1，越高越好（耦合度低）
        """
        if metrics.total_edges == 0:
            metrics.coupling = 1.0
        else:
            metrics.coupling = 1.0 - (metrics.inter_edges / metrics.total_edges)
    
    def _calculate_modularity(self, metrics: EvaluationMetrics) -> None:
        """
        计算模块性（Newman modularity）
        
        Q = (1/2m) * Σ(e_ii - (a_i)^2)
        其中：
        - m: 总边数
        - e_ii: 服务i内部的边数
        - a_i: 与服务i相连的边数
        """
        if metrics.total_edges == 0:
            metrics.modularity = 0.0
            return
        
        # 获取所有服务
        services = set(self.partition.values())
        
        # 计算每个服务的内部边数和总关联边数
        modularity = 0.0
        
        for service_id in services:
            # 该服务的所有节点
            service_nodes = [nid for nid, sid in self.partition.items() if sid == service_id]
            
            # 服务内部边数
            e_ii = 0
            for node_id in service_nodes:
                for dep_id in self.dependency_graph[node_id]:
                    if self.partition.get(dep_id) == service_id:
                        e_ii += 1
            
            # 与该服务相连的总边数
            a_i = 0
            for node_id in service_nodes:
                a_i += len(self.dependency_graph[node_id])
            
            # 计算该服务对模块性的贡献
            if metrics.total_edges > 0:
                modularity += (e_ii / metrics.total_edges) - ((a_i / (2 * metrics.total_edges)) ** 2)
        
        metrics.modularity = modularity
    
    def _calculate_balance(self, metrics: EvaluationMetrics) -> None:
        """
        计算平衡度
        
        平衡度 = 1 - (std_dev / mean)
        范围: 0-1，越高越好（各服务规模越均衡）
        """
        services = set(self.partition.values())
        service_sizes = defaultdict(int)
        
        for node_id, service_id in self.partition.items():
            service_sizes[service_id] += 1
        
        sizes = list(service_sizes.values())
        
        if len(sizes) == 0:
            metrics.balance = 0.0
            metrics.avg_service_size = 0.0
            metrics.std_service_size = 0.0
            return
        
        # 计算平均值和标准差
        avg_size = sum(sizes) / len(sizes)
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        std_dev = math.sqrt(variance)
        
        metrics.avg_service_size = avg_size
        metrics.std_service_size = std_dev
        metrics.max_service_size = max(sizes)
        metrics.min_service_size = min(sizes)
        
        # 计算平衡度
        if avg_size == 0:
            metrics.balance = 0.0
        else:
            # 使用变异系数的倒数作为平衡度
            cv = std_dev / avg_size
            metrics.balance = max(0.0, 1.0 - min(cv, 1.0))
    
    def _calculate_service_metrics(self, metrics: EvaluationMetrics) -> None:
        """计算每个服务的详细指标"""
        services = set(self.partition.values())
        
        for service_id in services:
            service_nodes = [nid for nid, sid in self.partition.items() if sid == service_id]
            
            # 服务规模
            size = len(service_nodes)
            
            # 内部边数
            intra = 0
            for node_id in service_nodes:
                for dep_id in self.dependency_graph[node_id]:
                    if self.partition.get(dep_id) == service_id:
                        intra += 1
            
            # 外部边数
            inter = 0
            for node_id in service_nodes:
                for dep_id in self.dependency_graph[node_id]:
                    if self.partition.get(dep_id) != service_id and dep_id in self.partition:
                        inter += 1
            
            # 内聚度
            total = intra + inter
            cohesion = intra / total if total > 0 else 0.0
            
            metrics.service_metrics[service_id] = {
                'size': size,
                'intra_edges': intra,
                'inter_edges': inter,
                'cohesion': round(cohesion, 4),
                'classes': [self.class_id_to_name.get(nid, f'Unknown_{nid}') for nid in service_nodes]
            }
    
    def _calculate_maintainability_index(self, metrics: EvaluationMetrics) -> None:
        """
        计算可维护性指数
        
        综合考虑内聚度、耦合度、模块性和平衡度
        范围: 0-100
        """
        # 将模块性从 (-0.5, 1) 映射到 (0, 1)
        modularity_normalized = (metrics.modularity + 0.5) / 1.5
        modularity_normalized = max(0.0, min(1.0, modularity_normalized))
        
        # 综合指数
        maintainability = (
            metrics.cohesion * 0.3 +
            metrics.coupling * 0.3 +
            modularity_normalized * 0.2 +
            metrics.balance * 0.2
        ) * 100
        
        metrics.maintainability_index = maintainability
    
    def _calculate_quality_score(self, metrics: EvaluationMetrics) -> None:
        """
        计算综合质量分数
        
        基于配置的权重计算加权平均分
        """
        quality = (
            metrics.cohesion * self.config.cohesion_weight * 100 +
            metrics.coupling * self.config.coupling_weight * 100 +
            metrics.balance * self.config.balance_weight * 100 +
            metrics.maintainability_index * self.config.modularity_weight
        ) / (
            self.config.cohesion_weight +
            self.config.coupling_weight +
            self.config.balance_weight +
            self.config.modularity_weight
        )
        
        metrics.quality_score = quality
    
    def print_report(self, metrics: EvaluationMetrics) -> None:
        """打印评估报告"""
        print("\n" + "=" * 80)
        print("微服务划分质量评估报告")
        print("=" * 80)
        
        print("\n【核心指标】")
        print(f"  内聚度 (Cohesion):        {metrics.cohesion:.4f} ({metrics.cohesion*100:.2f}%)")
        print(f"  耦合度 (Coupling):        {metrics.coupling:.4f} ({metrics.coupling*100:.2f}%)")
        print(f"  模块性 (Modularity):      {metrics.modularity:.4f}")
        print(f"  平衡度 (Balance):         {metrics.balance:.4f} ({metrics.balance*100:.2f}%)")
        
        print("\n【边数统计】")
        print(f"  服务内部边数:             {metrics.intra_edges}")
        print(f"  服务间边数:               {metrics.inter_edges}")
        print(f"  总边数:                   {metrics.total_edges}")
        if metrics.total_edges > 0:
            print(f"  内部边占比:               {metrics.intra_edges/metrics.total_edges*100:.2f}%")
            print(f"  跨服务边占比:             {metrics.inter_edges/metrics.total_edges*100:.2f}%")
        
        print("\n【服务规模】")
        print(f"  平均服务规模:             {metrics.avg_service_size:.2f}")
        print(f"  服务规模标准差:           {metrics.std_service_size:.2f}")
        print(f"  最大服务规模:             {metrics.max_service_size}")
        print(f"  最小服务规模:             {metrics.min_service_size}")
        
        print("\n【综合评分】")
        print(f"  可维护性指数:             {metrics.maintainability_index:.2f}/100")
        print(f"  综合质量分数:             {metrics.quality_score:.2f}/100")
        
        print("\n【各服务详细指标】")
        for service_id in sorted(metrics.service_metrics.keys()):
            svc = metrics.service_metrics[service_id]
            print(f"\n  Service-{service_id}:")
            print(f"    规模:                   {svc['size']} 个类")
            print(f"    内部边:                 {svc['intra_edges']}")
            print(f"    外部边:                 {svc['inter_edges']}")
            print(f"    内聚度:                 {svc['cohesion']:.4f}")
            print(f"    包含的类 ({len(svc['classes'])}): {', '.join(svc['classes'][:5])}")
            if len(svc['classes']) > 5:
                print(f"                          ... 还有 {len(svc['classes']) - 5} 个类")
        
        print("\n" + "=" * 80)
    
    def attach_semantic_metrics(self, semantic: SemanticMetrics) -> None:
        """将语义指标合并到 Evaluator 的指标中并保存在 self.metrics
        需要在 self.evaluate() 之后调用。
        """
        if self.metrics is None:
            self.metrics = EvaluationMetrics()
        self.metrics.semantic_coherence = semantic.semantic_coherence
        self.metrics.semantic_coupling = semantic.semantic_coupling
        self.metrics.business_alignment = semantic.business_alignment
        self.metrics.service_semantic_analysis = semantic.service_semantic_analysis

    def save_report(self, metrics: EvaluationMetrics, output_path: str) -> None:
        """
        保存评估报告为JSON文件
        
        Args:
            metrics: 评估指标
            output_path: 输出文件路径
        """
        report = {
            'metrics': metrics.to_dict(),
            'summary': {
                'total_services': len(metrics.service_metrics),
                'total_classes': sum(len(svc['classes']) for svc in metrics.service_metrics.values()),
                'total_edges': metrics.total_edges,
                'intra_service_edges': metrics.intra_edges,
                'inter_service_edges': metrics.inter_edges,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 评估报告已保存到: {output_path}")


def evaluate_partition(
    result_path: str = None,
    data_path: str = None,
    output_path: str = None,
    print_report: bool = True
) -> EvaluationMetrics:
    """
    快速评估微服务划分
    
    Args:
        result_path: 划分结果文件路径
        data_path: 数据文件路径
        output_path: 报告输出路径
        print_report: 是否打印报告
    
    Returns:
        EvaluationMetrics: 评估结果
    """
    config = EvaluationConfig()
    if result_path:
        config.result_path = result_path
    if data_path:
        config.data_path = data_path
    
    evaluator = Evaluator(config)
    evaluator.load_data()
    metrics = evaluator.evaluate()
    
    if print_report:
        evaluator.print_report(metrics)
    
    if output_path:
        evaluator.save_report(metrics, output_path)
    
    return metrics

