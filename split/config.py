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
    dataset_path: str = "/Users/xht/Downloads/MicroService/data/data.json"
    result_path: str = "/Users/xht/Downloads/MicroService/result/result.json"

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
    
    def apply_weight(self, similarity_matrix, edge_types_list):
        """
        将边类型权重应用到相似度矩阵
        
        Args:
            similarity_matrix: [N, N] 相似度矩阵
            edge_types_list: 边类型列表，edge_types_list[i] 表示边 i 的类型
        
        Returns:
            加权后的相似度矩阵
        """
        weighted_matrix = similarity_matrix.copy()
        
        # 如果有边类型信息，应用权重
        if edge_types_list and len(edge_types_list) > 0:
            for edge_type in edge_types_list:
                # 获取边类型的权重
                _ = self.get_weight(edge_type)
                # 这里需要知道边的源和目标节点，通常在调用处处理
                # 此处仅作为示例
        
        return weighted_matrix


@dataclass
class PartitionConfig:
    """划分配置"""
    num_communities: int = 7
    random_seed: int = 42
    alpha = 1.0  # 结构内聚权重
    beta = 1.0  # 语义内聚权重
    gamma = 1.0  # 跨服务耦合惩罚
    size_lower = [int(5) for _ in range(num_communities)]
    size_upper = [int(20) for _ in range(num_communities)]
    pair_threshold = 0.0
    time_limit_sec = 20
    # 迭代/Agent 优化配置（默认开启迭代 + Agent，可在此修改）
    enable_agent_optimization = True
    max_iterations = 5
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
    model_name: str = "microsoft/codebert-base"
    output_dim: int = 256
    freeze_encoder: bool = False
    max_length: int = 256


@dataclass
class FusionConfig:
    """融合模块配置"""
    structural_dim: int = 256
    semantic_dim: int = 256
    output_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class HierarchicalEncoderConfig:
    """分层编码器配置"""
    hidden_dim: int = 256
    output_dim: int = 512
    num_edge_types: int = 5
    num_communities: int = 5
    num_local_layers: int = 2
    num_global_layers: int = 2
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
        model_name="microsoft/codebert-base",
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
        model_name="microsoft/codebert-base",
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

# 大规模图配置（1000-10000 节点）
LARGE_GRAPH_CONFIG = HierarchicalEncoderConfig(
    hidden_dim=256,
    output_dim=512,
    num_communities=10,
    num_local_layers=2,
    num_global_layers=1,
    num_heads=4,
    dropout=0.1
)

# 超大规模图配置（> 10000 节点）
XLARGE_GRAPH_CONFIG = HierarchicalEncoderConfig(
    hidden_dim=128,
    output_dim=256,
    num_communities=20,
    num_local_layers=1,
    num_global_layers=1,
    num_heads=2,
    dropout=0.1
)

# 轻量级配置（用于快速推理）
LIGHTWEIGHT_CONFIG = CodeGraphEncoderConfig(
    structural=StructuralEncoderConfig(
        hidden_dim=128,
        output_dim=128,
        num_layers=1,
        num_heads=4,
        dropout=0.1
    ),
    semantic=SemanticEncoderConfig(
        model_name="microsoft/codebert-base",
        output_dim=128,
        freeze_encoder=True,
        max_length=128
    ),
    fusion=FusionConfig(
        structural_dim=128,
        semantic_dim=128,
        output_dim=256,
        num_heads=4,
        dropout=0.1
    )
)

# 高性能配置（用于最佳效果）
HIGHPERFORMANCE_CONFIG = CodeGraphEncoderConfig(
    structural=StructuralEncoderConfig(
        hidden_dim=512,
        output_dim=512,
        num_layers=4,
        num_heads=16,
        dropout=0.1
    ),
    semantic=SemanticEncoderConfig(
        model_name="microsoft/graphcodebert-base",
        output_dim=512,
        freeze_encoder=False,
        max_length=512
    ),
    fusion=FusionConfig(
        structural_dim=512,
        semantic_dim=512,
        output_dim=1024,
        num_heads=16,
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
    elif num_nodes < 1000:
        return MEDIUM_GRAPH_CONFIG
    else:
        return LARGE_GRAPH_CONFIG


def get_hierarchical_config_by_graph_size(num_nodes: int) -> HierarchicalEncoderConfig:
    """
    根据图的大小自动选择分层编码器配置
    
    Args:
        num_nodes: 图中的节点数
    
    Returns:
        推荐的分层编码器配置对象
    """
    if num_nodes < 1000:
        return HierarchicalEncoderConfig(
            hidden_dim=256,
            output_dim=512,
            num_communities=5,
            num_local_layers=2,
            num_global_layers=2
        )
    elif num_nodes < 10000:
        return LARGE_GRAPH_CONFIG
    else:
        return XLARGE_GRAPH_CONFIG


