"""
配置文件：定义系统的各种参数

该模块包含以下配置类：
1. DataConfig: 数据路径配置
2. PartitionConfig: 微服务划分配置
3. StructuralEncoderConfig: 结构编码器配置
4. SemanticEncoderConfig: 语义编码器配置
5. FusionConfig: 融合模块配置
6. HierarchicalEncoderConfig: 分层编码器配置
7. CodeGraphEncoderConfig: 完整编码器配置（包含上述三个编码器）

预定义配置模板：
- SMALL_GRAPH_CONFIG: 小规模图 (< 100 节点)
- MEDIUM_GRAPH_CONFIG: 中规模图 (100-1000 节点)
- LARGE_GRAPH_CONFIG: 大规模图 (1000-10000 节点)
- XLARGE_GRAPH_CONFIG: 超大规模图 (> 10000 节点)
- LIGHTWEIGHT_CONFIG: 轻量级配置（快速推理）
- HIGHPERFORMANCE_CONFIG: 高性能配置（最佳效果）

使用方式：
1. 自动选择: config = get_config_by_graph_size(num_nodes)
2. 手动选择: config = MEDIUM_GRAPH_CONFIG
3. 自定义: config = CodeGraphEncoderConfig(...)
"""

from dataclasses import dataclass


@dataclass
class DataConfig:
    """数据配置"""
    dataset_path: str = "C:/Users/lenovo/Desktop/MicroService/data/data.json"
    result_path: str = "C:/Users/lenovo/Desktop/MicroService/result/constraint_solving/result.json"

@dataclass
class EdgeTypeWeightConfig:
    """边类型权重配置"""
    type_weights: dict = None  # 边类型到权重的映射，如 {"call": 1.0, "import": 0.8}
    
    def __post_init__(self):
        if self.type_weights is None:
            # 默认权重配置
            self.type_weights = {
                "call": 1.0,      # 方法调用：最高权重
                "import": 0.8,    # 导入关系：中等权重
                "inherit": 0.9,   # 继承关系：较高权重
                "implement": 0.85, # 实现接口：中等偏高权重
                "depend": 0.7,    # 依赖关系：较低权重
            }
    
    def get_weight(self, edge_type: str) -> float:
        """获取指定边类型的权重，默认为1.0"""
        return self.type_weights.get(edge_type, 1.0)



@dataclass
class PartitionConfig:
    """划分配置"""
    num_communities: int = 7
    random_seed: int = 42
    alpha = 5.0  # 结构内聚权重
    beta = 1  # 语义内聚权重
    gamma = 3  # 跨服务耦合惩罚
    size_lower = [int(5) for _ in range(num_communities)]
    size_upper = [int(20) for _ in range(num_communities)]
    pair_threshold = 0.0
    time_limit_sec = 60
    # 迭代/Agent 优化配置（默认开启迭代 + Agent，可在此修改）

    enable_agent_optimization = True
    max_iterations = 1
    edge_type_weights: EdgeTypeWeightConfig = None  # 边类型权重配置
    
    def __post_init__(self):
        if self.edge_type_weights is None:
            self.edge_type_weights = EdgeTypeWeightConfig()


@dataclass
class StructuralEncoderConfig:
    """结构编码器配置"""
    node_feature_dim: int = 1
    hidden_dim: int = 256
    output_dim: int = 256
    num_edge_types: int = 5
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class SemanticEncoderConfig:
    """语义编码器配置"""
    model_name: str = "BAAI/bge-m3"
    output_dim: int = 256
    freeze_encoder: bool = False
    max_length: int = 512


@dataclass
class FusionConfig:
    """融合模块配置"""
    structural_dim: int = 256
    semantic_dim: int = 256
    output_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1

@dataclass
class CodeGraphEncoderConfig:
    """完整编码器配置"""
    structural: StructuralEncoderConfig = None
    semantic: SemanticEncoderConfig = None
    fusion: FusionConfig = None
    
    def __post_init__(self):
        if self.structural is None:
            self.structural = StructuralEncoderConfig()
        if self.semantic is None:
            self.semantic = SemanticEncoderConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()


# 预定义的配置模板

# 小规模图配置（< 100 节点）
SMALL_GRAPH_CONFIG = CodeGraphEncoderConfig(
    structural=StructuralEncoderConfig(
        hidden_dim=256,
        output_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0.1
    ),
    semantic=SemanticEncoderConfig(
        model_name="BAAI/bge-m3",
        output_dim=256,
        freeze_encoder=False
    ),
    fusion=FusionConfig(
        structural_dim=256,
        semantic_dim=256,
        output_dim=512,
        num_heads=8,
        dropout=0.1
    )
)

# 中规模图配置（100-1000 节点）
MEDIUM_GRAPH_CONFIG = CodeGraphEncoderConfig(
    structural=StructuralEncoderConfig(
        hidden_dim=256,
        output_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    ),
    semantic=SemanticEncoderConfig(
        model_name="BAAI/bge-m3",
        output_dim=256,
        freeze_encoder=True
    ),
    fusion=FusionConfig(
        structural_dim=256,
        semantic_dim=256,
        output_dim=512,
        num_heads=4,
        dropout=0.1
    )
)

def get_config_by_graph_size(num_nodes: int) -> CodeGraphEncoderConfig:
    """
    根据图的大小自动选择配置
    
    Args:
        num_nodes: 图中的节点数
    
    Returns:
        推荐的配置对象
    """
    if num_nodes < 100:
        return SMALL_GRAPH_CONFIG
    else:
        return MEDIUM_GRAPH_CONFIG
