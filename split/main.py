"""
完整的使用示例：展示如何使用代码图编码系统
"""

import torch
from split.encoder.code_graph_encoder import (
    CodeClass, CodeGraphDataBuilder, CodeGraphEncoder
)
from split.partition.microservice_partition import partition_from_embeddings
from typing import List

from split.utils.data_processor import load_json, save_json
import split.config as config
from split.config import (
    CodeGraphEncoderConfig, get_config_by_graph_size
)

def load_data(data_path: str) -> List[CodeClass]:
    nodes = load_json(data_path)
    classes = [
        CodeClass(
            id=node['id'],
            name=node['name'],
            description=node['description'],
            methods=node['methods'],
            dependencies=node['dependencies'],
            edge_types=node['edge_types']
        )
        for node in nodes
    ]

    return classes


def create_encoder_from_config(
    encoder_config: CodeGraphEncoderConfig,
    num_edge_types: int,
    device: torch.device
) -> CodeGraphEncoder:
    """
    从配置对象创建编码器
    
    Args:
        encoder_config: 编码器配置对象
        num_edge_types: 边类型数量
        device: 计算设备
    
    Returns:
        初始化好的 CodeGraphEncoder 模型
    """
    model = CodeGraphEncoder(
        structural_hidden_dim=encoder_config.structural.hidden_dim,
        structural_output_dim=encoder_config.structural.output_dim,
        semantic_output_dim=encoder_config.semantic.output_dim,
        final_output_dim=encoder_config.fusion.output_dim,
        num_edge_types=num_edge_types,
        num_structural_layers=encoder_config.structural.num_layers,
        num_heads=encoder_config.structural.num_heads,
        dropout=encoder_config.structural.dropout,
        code_encoder_model=encoder_config.semantic.model_name,
        freeze_code_encoder=encoder_config.semantic.freeze_encoder
    ).to(device)
    
    return model


def example_basic_encoder():
    """
    示例1：基础编码器
    """

    classes = load_data(config.DataConfig.dataset_path)
    class_names = [cls.name for cls in classes]

    # 构建图数据
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, texts = builder.build_graph_data()

    print(f"\n项目信息:")
    print(f"  类数: {len(classes)}")
    print(f"  边数: {edge_index.size(1)}")
    print(f"  边类型: {list(builder.edge_type_to_idx.keys())}")

    # 根据数据规模自动选择配置
    encoder_config = get_config_by_graph_size(len(classes))
    print(f"  自动选择的配置: {encoder_config.__class__.__name__}")

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    model = create_encoder_from_config(
        encoder_config=encoder_config,
        num_edge_types=len(builder.edge_type_to_idx),
        device=device
    )

    # 移动数据到设备
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)
    pos_encoding = pos_encoding.to(device)

    # 前向传播
    print("\n执行编码...")
    with torch.no_grad():
        embeddings = model(x, edge_index, edge_types, pos_encoding, texts)

    print(f"✓ 编码完成！")
    print(f"  输出形状: {embeddings.shape}")

    # ========== 微服务划分（MILP） ==========
    # 类规模：用方法数近似（也可换成 LOC/复杂度）
    sizes = [max(1, len(c.methods)) for c in classes]

    print("\n开始微服务划分 (MILP)...")
    part_res = partition_from_embeddings(
        embeddings=embeddings,
        edge_index=edge_index,
        K=config.PartitionConfig.num_communities,
        alpha=config.PartitionConfig.alpha,  # 结构内聚权重
        beta=config.PartitionConfig.beta,  # 语义内聚权重
        gamma=config.PartitionConfig.gamma,  # 跨服务耦合惩罚
        sizes=sizes,
        size_lower=config.PartitionConfig.size_lower,
        size_upper=config.PartitionConfig.size_upper,
        pair_threshold=config.PartitionConfig.pair_threshold,
        time_limit_sec=config.PartitionConfig.time_limit_sec,
    )

    print(f"划分求解状态: {part_res.solver_status}")
    print(f"目标值: {part_res.objective_value:.4f}")
    # 打印服务分组
    groups = {k: [] for k in range(config.PartitionConfig.num_communities)}
    for i, k in enumerate(part_res.assignments):
        if k >= 0:
            groups[k].append(classes[i].name)

    save_json(groups, config.DataConfig.result_path)
    print(f"✓ 微服务划分结果已保存到 {config.DataConfig.result_path}")
    print("服务分组结果：")
    for k in range(config.PartitionConfig.num_communities):
        print(f"  Service-{k}: {groups[k]}")

    return embeddings, class_names


def example_custom_config():
    """
    示例2：使用自定义配置
    """
    from split.config import (
        LIGHTWEIGHT_CONFIG, HIGHPERFORMANCE_CONFIG, MEDIUM_GRAPH_CONFIG
    )

    classes = load_data(config.DataConfig.dataset_path)
    class_names = [cls.name for cls in classes]

    # 构建图数据
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, texts = builder.build_graph_data()

    print(f"\n项目信息:")
    print(f"  类数: {len(classes)}")
    print(f"  边数: {edge_index.size(1)}")

    # 使用自定义配置（例如轻量级配置）
    custom_config = LIGHTWEIGHT_CONFIG
    print(f"\n使用配置: LIGHTWEIGHT_CONFIG")
    print(f"  结构编码器隐层维度: {custom_config.structural.hidden_dim}")
    print(f"  结构编码器输出维度: {custom_config.structural.output_dim}")
    print(f"  语义编码器输出维度: {custom_config.semantic.output_dim}")
    print(f"  融合输出维度: {custom_config.fusion.output_dim}")

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_encoder_from_config(
        encoder_config=custom_config,
        num_edge_types=len(builder.edge_type_to_idx),
        device=device
    )

    # 移动数据到设备
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)
    pos_encoding = pos_encoding.to(device)

    # 前向传播
    print("\n执行编码...")
    with torch.no_grad():
        embeddings = model(x, edge_index, edge_types, pos_encoding, texts)

    print(f"✓ 编码完成！")
    print(f"  输出形状: {embeddings.shape}")

    return embeddings, class_names


def main():
    example_basic_encoder()


if __name__ == "__main__":
    main()
