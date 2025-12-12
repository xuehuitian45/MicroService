"""
分层图编码器：用于处理超大规模代码依赖图
支持社区检测和分层注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from sklearn.cluster import SpectralClustering
from code_graph_encoder import (
    GraphormerLayer,
    SemanticEncoder, CrossAttentionFusion, CodeClass, CodeGraphDataBuilder
)


class CommunityDetector:
    """
    社区检测：将大图分解为多个局部子图
    """

    def __init__(self, num_communities: int = 5):
        self.num_communities = num_communities

    def detect_communities(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        使用谱聚类检测社区

        Args:
            adj_matrix: [num_nodes, num_nodes] 邻接矩阵（有向或无向均可）

        Returns:
            [num_nodes] 每个节点的社区ID
        """
        # 确保社区数不超过节点数
        num_nodes = adj_matrix.shape[0]
        num_communities = min(self.num_communities, num_nodes)

        # 1) 对称化邻接矩阵，避免 sklearn 的非对称警告
        adj = adj_matrix.astype(float)
        adj = np.maximum(adj, adj.T)

        # 2) 归一化到 [0,1]，并加入极小自环，提升数值稳定性
        max_val = adj.max()
        if max_val > 0:
            adj = adj / max_val
        adj = adj + np.eye(num_nodes) * 1e-6

        clustering = SpectralClustering(
            n_clusters=num_communities,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )

        # 使用处理后的邻接矩阵作为相似度矩阵
        community_ids = clustering.fit_predict(adj)

        return community_ids

    def get_subgraph_nodes(
            self,
            community_ids: np.ndarray,
            community_id: int
    ) -> np.ndarray:
        """获取指定社区的节点"""
        return np.where(community_ids == community_id)[0]


class LocalSubgraphEncoder(nn.Module):
    """
    局部子图编码器：在单个社区内进行全注意力
    """

    def __init__(
            self,
            hidden_dim: int,
            num_edge_types: int,
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        super().__init__()

        self.graphormer_layer = GraphormerLayer(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_edge_types=num_edge_types,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_types: torch.Tensor,
            pos_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        在子图内进行编码
        """
        x = self.graphormer_layer(x, edge_index, edge_types, pos_encoding)
        x = self.norm(x)
        return x


class GlobalInteractionModule(nn.Module):
    """
    全局交互模块：连接不同社区的表示
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 多头注意力
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            community_reps: torch.Tensor,
            community_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            community_reps: [num_nodes, hidden_dim] 社区表示
            community_ids: [num_nodes] 节点的社区ID

        Returns:
            [num_nodes, hidden_dim] 更新后的表示
        """
        num_nodes = community_reps.size(0)

        Q = self.query(community_reps)
        K = self.key(community_reps)
        V = self.value(community_reps)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # 添加社区掩码：只允许跨社区的注意力
        community_mask = (community_ids.unsqueeze(0) != community_ids.unsqueeze(1)).float()
        scores = scores + (1 - community_mask).unsqueeze(1) * float('-inf')

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = self.norm(out + community_reps)

        return out


class HierarchicalGraphEncoder(nn.Module):
    """
    分层图编码器：用于超大规模图
    """

    def __init__(
            self,
            hidden_dim: int = 256,
            output_dim: int = 512,
            num_edge_types: int = 5,
            num_communities: int = 5,
            num_local_layers: int = 2,
            num_global_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_communities = num_communities

        # 初始嵌入
        self.node_embedding = nn.Linear(1, hidden_dim)

        # 局部子图编码器
        self.local_encoders = nn.ModuleList([
            LocalSubgraphEncoder(
                hidden_dim=hidden_dim,
                num_edge_types=num_edge_types,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_local_layers)
        ])

        # 全局交互模块
        self.global_modules = nn.ModuleList([
            GlobalInteractionModule(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_global_layers)
        ])

        # 社区池化
        self.community_pool = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_types: torch.Tensor,
            pos_encoding: torch.Tensor,
            community_ids: torch.Tensor,
            adj_matrix: np.ndarray
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, 1] 初始特征
            edge_index: [2, num_edges] 边索引
            edge_types: [num_edges] 边类型
            pos_encoding: [num_nodes, pos_dim] 位置编码
            community_ids: [num_nodes] 社区ID
            adj_matrix: [num_nodes, num_nodes] 邻接矩阵

        Returns:
            [num_nodes, output_dim] 节点表示
        """
        # 初始嵌入
        x = self.node_embedding(x)
        x = self.dropout(x)

        # 局部编码：在每个社区内进行
        for local_encoder in self.local_encoders:
            x_local = x.clone()

            for community_id in range(self.num_communities):
                # 获取该社区的节点
                community_mask = (community_ids == community_id)
                community_nodes = torch.where(community_mask)[0]

                if len(community_nodes) == 0:
                    continue

                # 提取子图
                subgraph_x = x[community_nodes]

                # 提取子图的边
                edge_mask = (
                        community_mask[edge_index[0]] &
                        community_mask[edge_index[1]]
                )
                subgraph_edge_index = edge_index[:, edge_mask]
                subgraph_edge_types = edge_types[edge_mask]
                subgraph_pos_encoding = pos_encoding[community_nodes]

                # 重新映射边索引
                node_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(community_nodes)}
                subgraph_edge_index_mapped = torch.zeros_like(subgraph_edge_index)
                for i in range(subgraph_edge_index.size(1)):
                    old_src = subgraph_edge_index[0, i].item()
                    old_dst = subgraph_edge_index[1, i].item()
                    subgraph_edge_index_mapped[0, i] = node_mapping[old_src]
                    subgraph_edge_index_mapped[1, i] = node_mapping[old_dst]

                # 局部编码
                subgraph_x_encoded = local_encoder(
                    subgraph_x,
                    subgraph_edge_index_mapped,
                    subgraph_edge_types,
                    subgraph_pos_encoding
                )

                # 更新全局表示
                x_local[community_nodes] = subgraph_x_encoded

            x = x_local

        # 全局交互：连接社区
        for global_module in self.global_modules:
            x = global_module(x, community_ids)

        # 输出投影
        x = self.output_proj(x)

        return x


class HierarchicalCodeGraphEncoder(nn.Module):
    """
    完整的分层代码图编码系统
    """

    def __init__(
            self,
            hidden_dim: int = 256,
            structural_output_dim: int = 256,
            semantic_output_dim: int = 256,
            final_output_dim: int = 512,
            num_edge_types: int = 5,
            num_communities: int = 5,
            num_local_layers: int = 2,
            num_global_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.1,
            code_encoder_model: str = "microsoft/codebert-base",
            freeze_code_encoder: bool = False
    ):
        super().__init__()

        # 分层结构编码器
        self.hierarchical_encoder = HierarchicalGraphEncoder(
            hidden_dim=hidden_dim,
            output_dim=structural_output_dim,
            num_edge_types=num_edge_types,
            num_communities=num_communities,
            num_local_layers=num_local_layers,
            num_global_layers=num_global_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # 语义编码器
        self.semantic_encoder = SemanticEncoder(
            model_name=code_encoder_model,
            output_dim=semantic_output_dim,
            freeze_encoder=freeze_code_encoder
        )

        # 跨模态融合
        self.fusion = CrossAttentionFusion(
            structural_dim=structural_output_dim,
            semantic_dim=semantic_output_dim,
            output_dim=final_output_dim,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_types: torch.Tensor,
            pos_encoding: torch.Tensor,
            community_ids: torch.Tensor,
            adj_matrix: np.ndarray,
            texts: List[str]
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, 1] 初始特征
            edge_index: [2, num_edges] 边索引
            edge_types: [num_edges] 边类型
            pos_encoding: [num_nodes, pos_dim] 位置编码
            community_ids: [num_nodes] 社区ID
            adj_matrix: [num_nodes, num_nodes] 邻接矩阵
            texts: 节点文本

        Returns:
            [num_nodes, final_output_dim] 最终节点向量
        """
        # 分层结构编码
        structural_repr = self.hierarchical_encoder(
            x, edge_index, edge_types, pos_encoding, community_ids, adj_matrix
        )

        # 语义编码
        semantic_repr = self.semantic_encoder(texts)

        # 跨模态融合
        fused_repr = self.fusion(structural_repr, semantic_repr)

        return fused_repr


# 使用示例
if __name__ == "__main__":
    # 创建更大规模的示例数据
    num_classes = 20
    classes = []

    for i in range(num_classes):
        # 随机生成依赖关系
        num_deps = np.random.randint(0, 4)
        deps = np.random.choice(num_classes, size=num_deps, replace=False)
        deps = [d for d in deps if d != i]

        edge_types = ["import"] * len(deps)

        classes.append(CodeClass(
            id=i,
            name=f"Class_{i}",
            description=f"类 {i} 的描述信息",
            methods=[f"method_{j}" for j in range(np.random.randint(1, 5))],
            dependencies=list(deps),
            edge_types=edge_types
        ))

    # 构建图数据
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, texts = builder.build_graph_data()

    # 构建邻接矩阵
    adj_matrix = np.zeros((len(classes), len(classes)))
    for src_id, cls in enumerate(classes):
        for dst_id in cls.dependencies:
            adj_matrix[src_id, dst_id] = 1

    # 社区检测
    detector = CommunityDetector(num_communities=5)
    community_ids = detector.detect_communities(adj_matrix)
    community_ids = torch.from_numpy(community_ids).long()

    print("=" * 80)
    print("分层代码图编码系统演示")
    print("=" * 80)
    print(f"\n节点数: {x.size(0)}")
    print(f"边数: {edge_index.size(1)}")
    print(f"社区数: {len(np.unique(community_ids.numpy()))}")

    print("\n社区分配:")
    for community_id in range(max(community_ids) + 1):
        nodes_in_community = torch.where(community_ids == community_id)[0]
        print(f"  社区 {community_id}: {len(nodes_in_community)} 个节点")

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    model = HierarchicalCodeGraphEncoder(
        hidden_dim=256,
        structural_output_dim=256,
        semantic_output_dim=256,
        final_output_dim=512,
        num_edge_types=len(builder.edge_type_to_idx),
        num_communities=5,
        num_local_layers=2,
        num_global_layers=2,
        num_heads=8,
        dropout=0.1
    ).to(device)

    # 移动数据到设备
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)
    pos_encoding = pos_encoding.to(device)
    community_ids = community_ids.to(device)

    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        node_embeddings = model(
            x, edge_index, edge_types, pos_encoding,
            community_ids, adj_matrix, texts
        )

    print(f"\n最终节点向量形状: {node_embeddings.shape}")
    print(f"每个节点的向量维度: {node_embeddings.size(1)}")

    print("\n节点向量统计:")
    for i in range(min(5, len(classes))):
        embedding = node_embeddings[i]
        print(f"  {classes[i].name}:")
        print(f"    L2范数: {torch.norm(embedding).item():.4f}")
        print(f"    均值: {embedding.mean().item():.4f}")
        print(f"    标准差: {embedding.std().item():.4f}")

    print("\n✓ 分层系统运行成功!")

