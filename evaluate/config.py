"""
评估模块配置文件
"""

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 结果文件路径
    result_path: str = "/Users/xht/Downloads/MicroService/result/result.json"
    data_path: str = "/Users/xht/Downloads/MicroService/data/data.json"
    
    # 评估指标权重
    cohesion_weight: float = 0.3  # 内聚度权重
    coupling_weight: float = 0.3  # 耦合度权重
    modularity_weight: float = 0.2  # 模块性权重
    balance_weight: float = 0.2  # 平衡度权重
    
    # 阈值配置
    min_service_size: int = 1  # 最小服务规模
    max_service_size: int = 100  # 最大服务规模

