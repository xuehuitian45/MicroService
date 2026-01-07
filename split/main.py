"""
主入口：使用 full encoder 产生结构 / 语义 / 融合三种 embedding，并用三种 embedding 共同进行微服务划分。
"""

import os
import asyncio
from typing import List

import torch
import torch.nn.functional as F
from split.encoder.code_graph_encoder import (
    CodeClass,
    CodeGraphDataBuilder,
    CodeGraphEncoder,
)
from split.partition.microservice_partition import partition_from_multi_embeddings_iterative
from split.utils.data_processor import load_json, save_json
import split.config as config
from split.config import get_config_by_graph_size


def load_data(data_path: str) -> List[CodeClass]:
    nodes = load_json(data_path)
    classes = [
        CodeClass(
            id=node["id"],
            name=node["name"],
            description=node["description"],
            methods=node["methods"],
            dependencies=node["dependencies"],
            edge_types=node["edge_types"],
        )
        for node in nodes
    ]
    return classes


def example_full_encoder():
    """
    示例2：结构 + 语义 + 融合编码器（使用 cross-attn，并加入结构-语义对齐）

    依赖：
        - 先运行：python -m split.train_full_encoder
        - 加载：split/result/full_encoder_best.pt
    """

    classes = load_data(config.DataConfig.dataset_path)
    class_names = [cls.name for cls in classes]

    # 构建图数据（使用边类型权重）
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, texts, edge_weights = builder.build_graph_data(
        edge_type_weights=config.PartitionConfig().edge_type_weights
    )

    print(f"\n[FullEncoder] 项目信息:")
    print(f"  类数: {len(classes)}")
    print(f"  边数: {edge_index.size(1)}")
    print(f"  边类型: {list(builder.edge_type_to_idx.keys())}")

    # 根据数据规模自动选择配置
    encoder_config = get_config_by_graph_size(len(classes))
    print(f"  自动选择的配置: {encoder_config.__class__.__name__}")

    # 初始化 full encoder（结构 + 语义 + 融合）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    model = CodeGraphEncoder(
        structural_hidden_dim=encoder_config.structural.hidden_dim,
        structural_output_dim=encoder_config.structural.output_dim,
        semantic_output_dim=encoder_config.semantic.output_dim,
        final_output_dim=encoder_config.fusion.output_dim,
        num_edge_types=len(builder.edge_type_to_idx),
        num_structural_layers=encoder_config.structural.num_layers,
        num_heads=encoder_config.structural.num_heads,
        dropout=encoder_config.structural.dropout,
        code_encoder_model=encoder_config.semantic.model_name,
        freeze_code_encoder=encoder_config.semantic.freeze_encoder,
        structural_only=False,  # 关键：启用语义 + cross-attn 融合
    ).to(device)

    # 加载 full encoder 预训练权重
    pretrained_path = "split/result/full_encoder_best.pt"
    if os.path.exists(pretrained_path):
        print(f"\n[FullEncoder] 检测到预训练模型: {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            print("✓ full encoder 权重加载成功")
        except Exception as e:
            print(f"! 加载 full encoder 预训练权重失败: {e}，将使用随机初始化")
    else:
        print(f"\n[FullEncoder] 未找到预训练模型 ({pretrained_path})，使用随机初始化")
        print("  提示：先运行 'python -m split.train_full_encoder' 来训练 full encoder")


    # 移动数据到设备
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)
    pos_encoding = pos_encoding.to(device)
    edge_weights = edge_weights.to(device) if edge_weights is not None and edge_weights.numel() > 0 else None

    # 前向传播，获得三种 embedding
    print("\n[FullEncoder] 执行编码（结构 + 语义 + 融合）...")
    model.eval()
    with torch.no_grad():
        z_struct, z_text, z_fused = model(
            x=x,
            edge_index=edge_index,
            edge_types=edge_types,
            pos_encoding=pos_encoding,
            texts=texts,
            edge_weights=edge_weights,
        )

    print("✓ 编码完成！")
    print(f"  结构 embedding 形状: {z_struct.shape}")
    print(f"  语义 embedding 形状: {z_text.shape}")
    print(f"  融合 embedding 形状: {z_fused.shape}")

    # ========== 调试：计算不同连接关系的节点对的语义相似度 ==========
    print("\n[调试] 计算语义向量相似度（按连接关系分类）...")
    
    num_nodes = z_struct.size(0)
    
    # 1. 构建无向邻接矩阵
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32, device=z_struct.device)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # 无向图
    adj.fill_diagonal_(0.0)  # 排除自环
    
    # 2. 计算结构相似度矩阵
    z_struct_norm = F.normalize(z_struct, p=2, dim=-1)  # [N, D]
    sim_matrix = torch.matmul(z_struct_norm, z_struct_norm.t())  # [N, N]
    
    # 3. 分类节点对
    # 直接相连的节点对（距离=1）
    direct_connected = (adj > 0).float()
    direct_mask = direct_connected.bool()
    
    # 距离为2的节点对（通过一个中间节点连接）
    # 计算 A^2，然后减去直接相连的和自环
    adj_squared = torch.matmul(adj, adj)
    adj_squared.fill_diagonal_(0.0)  # 排除自环
    distance_2 = (adj_squared > 0).float() * (1 - direct_connected)  # 距离为2且不是直接相连
    distance_2_mask = distance_2.bool()
    
    # 不相连的节点对（距离>2或没有路径）
    not_connected = (1 - direct_connected - distance_2).float()
    not_connected.fill_diagonal_(0.0)  # 排除自环
    not_connected_mask = not_connected.bool()
    
    # 4. 计算各组的平均相似度
    # 直接相连的节点对
    direct_sims = sim_matrix[direct_mask]
    avg_direct = direct_sims.mean().item() if direct_sims.numel() > 0 else 0.0
    count_direct = direct_sims.numel()
    
    # 距离为2的节点对
    distance_2_sims = sim_matrix[distance_2_mask]
    avg_distance_2 = distance_2_sims.mean().item() if distance_2_sims.numel() > 0 else 0.0
    count_distance_2 = distance_2_sims.numel()
    
    # 不相连的节点对
    not_connected_sims = sim_matrix[not_connected_mask]
    avg_not_connected = not_connected_sims.mean().item() if not_connected_sims.numel() > 0 else 0.0
    count_not_connected = not_connected_sims.numel()
    
    # 5. 输出结果
    print(f"\n[调试] 语义相似度统计（按连接关系）：")
    print(f"{'连接关系':<25} {'节点对数量':<15} {'平均相似度':<15} {'标准差':<15}")
    print("-" * 70)
    print(f"{'直接相连（距离=1）':<25} {count_direct:<15} {avg_direct:<15.6f} {direct_sims.std().item():<15.6f}" if count_direct > 0 else f"{'直接相连（距离=1）':<25} {count_direct:<15} {'N/A':<15} {'N/A':<15}")
    print(f"{'中转1节点（距离=2）':<25} {count_distance_2:<15} {avg_distance_2:<15.6f} {distance_2_sims.std().item():<15.6f}" if count_distance_2 > 0 else f"{'中转1节点（距离=2）':<25} {count_distance_2:<15} {'N/A':<15} {'N/A':<15}")
    print(f"{'不相连（距离>2）':<25} {count_not_connected:<15} {avg_not_connected:<15.6f} {not_connected_sims.std().item():<15.6f}" if count_not_connected > 0 else f"{'不相连（距离>2）':<25} {count_not_connected:<15} {'N/A':<15} {'N/A':<15}")
    
    # 计算总节点对数量（用于验证）
    total_pairs = num_nodes * (num_nodes - 1) // 2  # 无向图，排除自环
    print(f"\n[调试] 验证：")
    print(f"  总节点数: {num_nodes}")
    print(f"  理论节点对总数（无向图）: {total_pairs}")
    print(f"  实际统计节点对总数: {count_direct + count_distance_2 + count_not_connected}")
    print(f"  差异: {total_pairs - (count_direct + count_distance_2 + count_not_connected)}")
    
    print("\n" + "=" * 70)

    # ========== 输出语义 / 融合向量最相似的节点对 ==========
    def print_top_similar_pairs(embeddings, names, title, top_k=100):
        """打印最相似的前 top_k 个节点对及其余弦相似度。"""
        if embeddings is None or embeddings.numel() == 0:
            print(f"[调试] {title}：无可用向量，跳过。")
            return

        z_norm = F.normalize(embeddings, p=2, dim=-1)
        sim_full = torch.matmul(z_norm, z_norm.t())

        # 仅取上三角（排除自环），避免重复
        upper_idx = torch.triu_indices(num_nodes, num_nodes, offset=1, device=embeddings.device)
        if upper_idx.numel() == 0:
            print(f"[调试] {title}：节点不足，跳过。")
            return

        pair_scores = sim_full[upper_idx[0], upper_idx[1]]
        k = min(top_k, pair_scores.numel())
        if k == 0:
            print(f"[调试] {title}：无可用节点对，跳过。")
            return

        top_vals, top_indices = torch.topk(pair_scores, k=k, largest=True)
        print(f"\n[调试] {title} 最相似的前 {k} 个节点对（余弦相似度）：")
        for rank in range(k):
            i = upper_idx[0][top_indices[rank]].item()
            j = upper_idx[1][top_indices[rank]].item()
            sim_score = top_vals[rank].item()
            print(f"  #{rank + 1:<3} ({names[i]} , {names[j]}) -> {sim_score:.6f}")

    print_top_similar_pairs(z_text, class_names, "语义向量", top_k=100)
    print_top_similar_pairs(z_fused, class_names, "融合向量", top_k=1000)

    # ========== 使用三种 embedding 进行微服务划分（迭代版） ==========
    # 类规模：用方法数近似（也可换成 LOC/复杂度）
    sizes = [max(1, len(c.methods)) for c in classes]

    print("\n[FullEncoder] 开始基于三种 embedding 的微服务划分 (迭代优化)...")

    # 从配置读取迭代与 Agent 开关
    max_iterations = int(getattr(config.PartitionConfig, "max_iterations", 1))
    enable_agent_cfg = bool(getattr(config.PartitionConfig, "enable_agent_optimization", False))

    # 尝试加载 Agent 优化函数；若环境未配置（如缺少 DASHSCOPE_API_KEY），则回退为禁用 Agent
    agent_optimize_fn = None
    agent_analyze_fn = None
    try:
        from split.partition.agent_optimize import agent_optimize as _agent_optimize, agent_analyze as _agent_analyze

        agent_optimize_fn = _agent_optimize
        agent_analyze_fn = _agent_analyze
    except Exception as e:
        print(f"[FullEncoder] 未启用 Agent 优化（原因：{e}），将仅执行迭代求解而不调用 Agent")

    enable_agent = enable_agent_cfg and (agent_optimize_fn is not None)
    print(
        f"[FullEncoder] 配置：max_iterations={max_iterations}, "
        f"enable_agent_optimization={enable_agent_cfg}, 启用Agent={enable_agent}"
    )

    # 使用三种 embedding 的组合相似度进行划分：
    #   - beta_struct：结构 embedding 在“语义内聚度”中的权重（一般可设 0 或较小值）
    #   - beta_sem：语义 embedding 的权重
    #   - beta_fused：融合 embedding 的权重（通常可以设为 1，作为主要相似度来源）
    part_res = asyncio.run(
        partition_from_multi_embeddings_iterative(
            emb_struct=z_struct,
            emb_sem=z_text,
            emb_fused=z_fused,
            edge_index=edge_index,
            K=config.PartitionConfig.num_communities,
            alpha=config.PartitionConfig.alpha,
            beta_struct=1.0,
            beta_sem=2.0,
            beta_fused=1.0,
            gamma=config.PartitionConfig.gamma,
            must_link=None,
            cannot_link=None,
            sizes=sizes,
            size_lower=config.PartitionConfig.size_lower,
            size_upper=config.PartitionConfig.size_upper,
            min_service_size=None,
            max_service_size=None,
            pair_threshold=config.PartitionConfig.pair_threshold,
            time_limit_sec=config.PartitionConfig.time_limit_sec,
            symmetric_struc=True,
            edge_weights=edge_weights,
            max_iterations=max(1, max_iterations),
            enable_agent_optimization=enable_agent,
            agent_optimize_fn=agent_optimize_fn,
            agent_analyze_fn=agent_analyze_fn,
            node_names=class_names,
        )
    )

    print(f"[FullEncoder] 划分求解状态: {part_res.solver_status}")
    print(f"[FullEncoder] 目标值: {part_res.objective_value:.4f}")

    # 打印服务分组并保存
    groups = {k: [] for k in range(config.PartitionConfig.num_communities)}
    for i, k in enumerate(part_res.assignments):
        if k >= 0:
            groups[k].append(classes[i].name)

    save_json(groups, config.DataConfig.result_path)
    print(f"[FullEncoder] ✓ 微服务划分结果已保存到 {config.DataConfig.result_path}")
    print("[FullEncoder] 服务分组结果：")
    for k in range(config.PartitionConfig.num_communities):
        print(f"  Service-{k}: {groups[k]}")

    return z_struct, z_text, z_fused, class_names


def main():
    example_full_encoder()


if __name__ == "__main__":
    main()
