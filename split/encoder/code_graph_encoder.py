"""
将图节点进行编码，分别采用结构和语义信息，并使用交叉注意力进行融合，采用结构向量作为查询向量，更加突出结构信息重要性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


@dataclass
class CodeClass:
    """代码类的信息"""
    id: int
    name: str
    description: str
    methods: List[str]
    dependencies: List[int]  # 依赖的其他类的ID
    edge_types: List[str]  # 边的类型（如 "import", "inherit", "call" 等）


class PositionalEncoding:
    """图位置编码：基于最短路径和PageRank（支持有向图）"""

    def __init__(self, num_nodes: int, max_distance: int = 10, directed: bool = True):
        self.num_nodes = num_nodes
        self.max_distance = max_distance
        self.directed = directed

    def shortest_path_encoding(self, adj_matrix: np.ndarray) -> torch.Tensor:
        """
        计算最短路径距离矩阵作为位置编码
        """
        # 计算所有节点对之间的最短路径
        dist_matrix = shortest_path(
            csr_matrix(adj_matrix),
            directed=self.directed,
            return_predecessors=False
        )

        # 将无穷大替换为max_distance
        dist_matrix[np.isinf(dist_matrix)] = self.max_distance

        # 归一化到 [0, 1]
        dist_matrix = dist_matrix / self.max_distance

        return torch.from_numpy(dist_matrix).float()

    def pagerank_encoding(self, adj_matrix: np.ndarray, alpha: float = 0.85) -> torch.Tensor:
        """
        基于PageRank的位置编码
        """
        # 使用有向图，PageRank 将体现有向依赖
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        pr = nx.pagerank(G, alpha=alpha)

        # 转换为向量
        pr_values = np.array([pr[i] for i in range(self.num_nodes)])

        # 归一化
        pr_values = (pr_values - pr_values.min()) / (pr_values.max() - pr_values.min() + 1e-8)

        return torch.from_numpy(pr_values).float().unsqueeze(1)

    def degree_encoding(self, adj_matrix: np.ndarray) -> torch.Tensor:
        """
        基于节点度数的位置编码
        """
        degrees = np.sum(adj_matrix, axis=1)
        degrees = degrees / (degrees.max() + 1e-8)

        return torch.from_numpy(degrees).float().unsqueeze(1)


class EdgeTypeEmbedding(nn.Module):
    """异构边类型嵌入"""

    def __init__(self, num_edge_types: int, embedding_dim: int):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(num_edge_types, embedding_dim)

    def forward(self, edge_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_types: [num_edges] 边的类型索引

        Returns:
            [num_edges, embedding_dim] 边的嵌入
        """
        return self.edge_type_embedding(edge_types)


class GraphormerLayer(MessagePassing):
    """
    标准 Graphormer 层（全局注意力 + 注意力偏置 + 中心性编码，Pre-LN 结构）

    关键点：
    - 使用最短路径距离（SPD）的可学习嵌入作为注意力偏置
    - 使用边类型（直接相邻）嵌入作为注意力偏置
    - 使用入度/出度中心性嵌入，叠加到节点表示
    - 预归一化 Transformer 结构（LN -> MHA -> 残差 -> LN -> FFN -> 残差）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_dist: int = 10,
        max_degree: int = 512,
    ):
        super().__init__(aggr='add')
        assert in_channels == out_channels, "GraphormerLayer 需 in_channels == out_channels"
        assert out_channels % num_heads == 0, "out_channels 必须能被 num_heads 整除"

        self.hidden_dim = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.max_dist = max_dist
        self.max_degree = max_degree

        # QKV 投影
        self.q_proj = nn.Linear(out_channels, out_channels)
        self.k_proj = nn.Linear(out_channels, out_channels)
        self.v_proj = nn.Linear(out_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)

        # 注意力偏置：SPD 与 边类型（直接边）
        # SPD 偏置：每个距离一个可学习的每头偏置（num_heads 维向量）
        self.spd_bias_table = nn.Embedding(self.max_dist + 1, num_heads)
        # 边类型偏置：每种边类型一个每头偏置
        self.edge_type_bias = nn.Embedding(num_edge_types, num_heads)

        # 中心性编码（加到节点表示上）
        self.in_degree_emb = nn.Embedding(self.max_degree + 1, out_channels)
        self.out_degree_emb = nn.Embedding(self.max_degree + 1, out_channels)

        # 预归一化 + FFN
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _compute_degrees(num_nodes: int, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = edge_index.device if edge_index.numel() > 0 else torch.device('cpu')
        deg_out = torch.zeros(num_nodes, dtype=torch.long, device=device)
        deg_in = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            deg_out.scatter_add_(0, src, torch.ones_like(src))
            deg_in.scatter_add_(0, dst, torch.ones_like(dst))
        return deg_in, deg_out

    def _build_attention_bias(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        pos_encoding: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        返回形状 [num_heads, N, N] 的注意力偏置矩阵。
        - SPD 偏置：从 pos_encoding 的前 N 列提取（N x N 的最短路径编码，已归一化到 [0,1]），
          映射到 0..max_dist 的整数桶后查表；
        - 边类型偏置：对直接边 (i->j) 为每个 head 加上对应的偏置。
        """
        attn_bias = torch.zeros(self.num_heads, num_nodes, num_nodes, device=device)

        # SPD 偏置
        if pos_encoding is not None and pos_encoding.size(1) >= num_nodes:
            spd_norm = pos_encoding[:, :num_nodes]  # [N, N], 行 i 表示 i 到所有 j 的距离（归一化）
            # 反归一化回 0..max_dist 的整数桶
            spd_bucket = (spd_norm * self.max_dist + 0.5).clamp(0, self.max_dist).long()  # [N, N]
            # 查表得到每个 pair 的每头偏置 -> [N, N, H]
            spd_bias = self.spd_bias_table(spd_bucket)  # [N, N, H]
            attn_bias = attn_bias + spd_bias.permute(2, 0, 1)  # -> [H, N, N]

        # 直接边类型偏置
        if edge_index.numel() > 0 and edge_types.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            # 每条边一个每头偏置：[E, H]
            e_bias = self.edge_type_bias(edge_types)  # [E, H]
            # 累加到 [H, N, N]
            for h in range(self.num_heads):
                attn_bias[h].index_put_((src, dst), e_bias[:, h], accumulate=True)

        return attn_bias

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        pos_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, D] 节点特征
            edge_index: [2, E] 有向边索引
            edge_types: [E] 边类型索引
            pos_encoding: [N, N + k]，前 N 列为 i->j 的最短路径距离编码（归一化到 [0,1]）

        Returns:
            [N, D] 更新后的节点表示
        """
        N, D = x.size()
        device = x.device

        # 1) 中心性编码（入度/出度）加到表示上
        deg_in, deg_out = self._compute_degrees(N, edge_index)
        deg_in = deg_in.clamp(max=self.max_degree)
        deg_out = deg_out.clamp(max=self.max_degree)
        x = x + self.in_degree_emb(deg_in) + self.out_degree_emb(deg_out)

        # 2) Pre-LN + 多头注意力
        x_norm = self.ln1(x)

        q = self.q_proj(x_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Hd]
        k = self.k_proj(x_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Hd]
        v = self.v_proj(x_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Hd]

        # 构建注意力偏置（可作为 additive mask 传入 SDPA）
        attn_bias = self._build_attention_bias(N, edge_index, edge_types, pos_encoding, device)  # [H, N, N]

        # 使用 PyTorch 2.x 的 Flash-Attention 接口（若可用）
        q_t = q.unsqueeze(0)  # [1, H, N, Hd]
        k_t = k.unsqueeze(0)
        v_t = v.unsqueeze(0)
        attn_mask = attn_bias.unsqueeze(0)  # [1, H, N, N]

        if hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )  # [1, H, N, Hd]
            attn_out = attn_out.squeeze(0)  # [H, N, Hd]
        else:
            # 兼容旧版：手动计算
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [H, N, N]
            scores = scores + attn_bias
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            attn_out = torch.matmul(attn, v)  # [H, N, Hd]

        # 合并 heads -> [N, D]
        attn_out = attn_out.transpose(0, 1).contiguous().view(N, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_dropout(attn_out)

        x = x + attn_out  # 残差

        # 3) FFN（Pre-LN）
        y = self.ln2(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)

        out = x + y  # 残差
        return out


class StructuralEncoder(nn.Module):
    """
    结构编码器：使用Graphormer编码代码依赖图
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_edge_types: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 初始节点特征投影
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        # Graphormer层
        self.graphormer_layers = nn.ModuleList([
            GraphormerLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_edge_types=num_edge_types,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        pos_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, node_feature_dim] 初始节点特征
            edge_index: [2, num_edges] 边索引
            edge_types: [num_edges] 边类型
            pos_encoding: [num_nodes, pos_dim] 位置编码（前 N 列为 SPD）

        Returns:
            [num_nodes, output_dim] 节点的结构表示
        """
        # 嵌入初始特征
        x = self.node_embedding(x)
        x = self.dropout(x)

        # 通过Graphormer层
        for layer in self.graphormer_layers:
            x = layer(x, edge_index, edge_types, pos_encoding)

        # 输出投影
        x = self.output_proj(x)

        return x


class SemanticEncoder(nn.Module):
    """
    语义编码器：使用预训练的Code-Text Encoder
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        output_dim: int = 256,
        freeze_encoder: bool = False
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        encoder_dim = self.encoder.config.hidden_size

        # 投影层
        self.projection = nn.Linear(encoder_dim, output_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: 文本列表（类名、注释、方法签名等）

        Returns:
            [num_texts, output_dim] 文本的语义表示
        """
        # 分词和编码
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        # 移到同一设备
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 获取编码
        with torch.no_grad():
            outputs = self.encoder(**inputs)

        # 使用[CLS]令牌的表示
        cls_output = outputs.last_hidden_state[:, 0, :]

        # 投影
        semantic_repr = self.projection(cls_output)

        return semantic_repr


class CrossAttentionFusion(nn.Module):
    """
    跨模态融合模块：使用Cross-Attention融合结构和语义信息

    优化点：
    1. 自适应注意力缩放：学习注意力缩放因子，提高数值稳定性
    2. 注意力掩码支持：处理无效节点或填充值
    3. 更优的激活函数：使用GELU替代ReLU
    4. 兼容PyTorch 2.0+的Flash Attention：提高性能
    """

    def __init__(
        self,
        structural_dim: int,
        semantic_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_adaptive_scale: bool = True
    ):
        super().__init__()

        self.structural_dim = structural_dim
        self.semantic_dim = semantic_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_adaptive_scale = use_adaptive_scale

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        # Query来自结构表示
        self.query_proj = nn.Linear(structural_dim, output_dim)

        # Key和Value来自语义表示
        self.key_proj = nn.Linear(semantic_dim, output_dim)
        self.value_proj = nn.Linear(semantic_dim, output_dim)

        # 输出投影
        self.out_proj = nn.Linear(output_dim, output_dim)

        # 残差投影（将结构表示投到output_dim以便做残差相加）
        self.residual_proj = nn.Linear(structural_dim, output_dim) if structural_dim != output_dim else nn.Identity()

        # 归一化层
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        # 前馈网络（使用GELU激活函数）
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),  # 使用GELU替代ReLU，提高性能
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim)
        )

        # 自适应注意力缩放因子
        if self.use_adaptive_scale:
            self.attention_scale = nn.Parameter(torch.log(torch.tensor(self.head_dim ** 0.5)))
        else:
            self.attention_scale = torch.tensor(self.head_dim ** 0.5)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数，确保训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier均匀初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm权重初始化为1，偏置初始化为0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        structural_repr: torch.Tensor,
        semantic_repr: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        跨模态融合前向传播

        Args:
            structural_repr: [num_nodes, structural_dim] 节点的结构表示
            semantic_repr: [num_nodes, semantic_dim] 节点的语义表示
            attention_mask: 可选，[num_nodes, num_nodes] 或 [num_heads, num_nodes, num_nodes]
                注意力掩码，值为True或1的位置表示需要被屏蔽，将被设置为负无穷

        Returns:
            [num_nodes, output_dim] 融合后的节点表示
        """
        num_nodes = structural_repr.size(0)

        # 1. 线性投影
        Q = self.query_proj(structural_repr)  # [num_nodes, output_dim]
        K = self.key_proj(semantic_repr)  # [num_nodes, output_dim]
        V = self.value_proj(semantic_repr)  # [num_nodes, output_dim]

        # 2. 重塑为多头格式：[num_nodes, num_heads, head_dim]
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)

        # 3. 处理注意力掩码
        if attention_mask is not None:
            # 将掩码转换为float类型，并将True/1的值设置为负无穷
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.float().masked_fill(attention_mask, float('-inf'))
            else:
                attention_mask = attention_mask.masked_fill(attention_mask != 0, float('-inf'))

            # 适配多头维度：[num_heads, num_nodes, num_nodes]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).expand(self.num_heads, -1, -1)

        # 4. 计算注意力权重
        if hasattr(F, 'scaled_dot_product_attention'):
            scale = torch.exp(self.attention_scale) if self.use_adaptive_scale else self.attention_scale

            # 使用Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q.unsqueeze(0).transpose(1, 2),
                K.unsqueeze(0).transpose(1, 2),
                V.unsqueeze(0).transpose(1, 2),
                attn_mask=attention_mask.unsqueeze(0) if attention_mask is not None else None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
                scale=scale  # 传入自适应缩放因子
            ).transpose(1, 2).squeeze(0)
        else:
            # 兼容旧版本PyTorch：手动计算注意力
            # 计算注意力分数：[num_nodes, num_heads, num_nodes]
            Qh = Q.transpose(0, 1)  # [H, N, Hd]
            Kh = K.transpose(0, 1)  # [H, N, Hd]
            Vh = V.transpose(0, 1)  # [H, N, Hd]
            scores = torch.matmul(Qh, Kh.transpose(-2, -1))

            # 应用自适应缩放因子
            scale = torch.exp(self.attention_scale) if hasattr(self, 'attention_scale') else np.sqrt(self.head_dim)
            scores = scores / scale

            # 应用注意力掩码
            if attention_mask is not None:
                mask = attention_mask if attention_mask.dim() == 3 else attention_mask.unsqueeze(0)
                scores = scores + mask

            # Softmax归一化
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 应用注意力到值：[H, N, Hd]
            attn_output = torch.matmul(attn_weights, Vh)  # [H, N, Hd]
            attn_output = attn_output.transpose(0, 1)  # [N, H, Hd]

        # 5. 合并多头：[num_nodes, output_dim]
        attn_output = attn_output.reshape(num_nodes, self.output_dim)

        # 6. 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # 7. 残差连接和层归一化
        out = self.norm1(attn_output + self.residual_proj(structural_repr))

        # 8. 前馈网络
        ffn_output = self.ffn(out)
        ffn_output = self.dropout(ffn_output)

        # 9. 残差连接和层归一化
        out = self.norm2(ffn_output + out)

        return out


class CodeGraphEncoder(nn.Module):
    """
    完整的代码图编码系统
    """

    def __init__(
        self,
        structural_hidden_dim: int = 256,
        structural_output_dim: int = 256,
        semantic_output_dim: int = 256,
        final_output_dim: int = 512,
        num_edge_types: int = 5,
        num_structural_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        code_encoder_model: str = "microsoft/codebert-base",
        freeze_code_encoder: bool = False
    ):
        super().__init__()

        self.structural_output_dim = structural_output_dim
        self.semantic_output_dim = semantic_output_dim
        self.final_output_dim = final_output_dim

        # 结构编码器
        self.structural_encoder = StructuralEncoder(
            node_feature_dim=1,  # 初始特征维度
            hidden_dim=structural_hidden_dim,
            output_dim=structural_output_dim,
            num_edge_types=num_edge_types,
            num_layers=num_structural_layers,
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
        texts: List[str]
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, 1] 初始节点特征
            edge_index: [2, num_edges] 边索引
            edge_types: [num_edges] 边类型
            pos_encoding: [num_nodes, pos_dim] 位置编码
            texts: 节点对应的文本（类名+注释+方法签名）

        Returns:
            [num_nodes, final_output_dim] 最终节点向量
        """
        # 结构编码
        structural_repr = self.structural_encoder(x, edge_index, edge_types, pos_encoding)

        # 语义编码
        semantic_repr = self.semantic_encoder(texts)

        # 跨模态融合
        fused_repr = self.fusion(structural_repr, semantic_repr)

        return fused_repr


class CodeGraphDataBuilder:
    """
    从代码类信息构建图数据
    """

    def __init__(self, classes: List[CodeClass]):
        self.classes = classes
        self.num_nodes = len(classes)
        self.edge_type_to_idx = {}
        self.build_edge_type_mapping()

    def build_edge_type_mapping(self):
        """构建边类型到索引的映射"""
        edge_types_set = set()
        for cls in self.classes:
            edge_types_set.update(cls.edge_types)

        for i, edge_type in enumerate(sorted(edge_types_set)):
            self.edge_type_to_idx[edge_type] = i

    def build_graph_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        构建图数据

        Returns:
            x: 节点特征
            edge_index: 边索引
            edge_types: 边类型
            pos_encoding: 位置编码
            texts: 节点文本
        """
        # 构建邻接矩阵
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        edge_list = []
        edge_type_list = []

        for src_id, cls in enumerate(self.classes):
            for dst_id, edge_type in zip(cls.dependencies, cls.edge_types):
                adj_matrix[src_id, dst_id] = 1
                edge_list.append([src_id, dst_id])
                edge_type_list.append(self.edge_type_to_idx[edge_type])

        # 转换为张量
        x = torch.ones(self.num_nodes, 1, dtype=torch.float32)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_types = torch.tensor(edge_type_list, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros(0, dtype=torch.long)

        # 计算位置编码
        pos_encoder = PositionalEncoding(self.num_nodes)

        # 组合多种位置编码
        sp_encoding = pos_encoder.shortest_path_encoding(adj_matrix)
        pr_encoding = pos_encoder.pagerank_encoding(adj_matrix)
        deg_encoding = pos_encoder.degree_encoding(adj_matrix)

        pos_encoding = torch.cat([sp_encoding, pr_encoding, deg_encoding], dim=1)

        # 构建文本
        texts = []
        for cls in self.classes:
            text = f"{cls.name}. {cls.description}. Methods: {', '.join(cls.methods)}"
            texts.append(text)

        return x, edge_index, edge_types, pos_encoding, texts


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    classes = [
        CodeClass(
            id=0,
            name="UserService",
            description="用户服务类，处理用户相关的业务逻辑",
            methods=["get_user", "create_user", "update_user", "delete_user"],
            dependencies=[1, 2],
            edge_types=["import", "call"]
        ),
        CodeClass(
            id=1,
            name="UserRepository",
            description="用户数据访问层，与数据库交互",
            methods=["query", "insert", "update", "delete"],
            dependencies=[3],
            edge_types=["import"]
        ),
        CodeClass(
            id=2,
            name="AuthService",
            description="认证服务，处理用户认证和授权",
            methods=["authenticate", "authorize", "validate_token"],
            dependencies=[1],
            edge_types=["call"]
        ),
        CodeClass(
            id=3,
            name="Database",
            description="数据库连接和操作类",
            methods=["connect", "execute", "close"],
            dependencies=[],
            edge_types=[]
        ),
        CodeClass(
            id=4,
            name="CacheService",
            description="缓存服务，提高性能",
            methods=["get", "set", "delete", "clear"],
            dependencies=[3],
            edge_types=["call"]
        ),
    ]

    # 构建图数据
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, texts = builder.build_graph_data()

    print("=" * 80)
    print("代码图编码系统演示")
    print("=" * 80)
    print(f"\n节点数: {x.size(0)}")
    print(f"边数: {edge_index.size(1)}")
    print(f"边类型数: {len(builder.edge_type_to_idx)}")
    print(f"位置编码维度: {pos_encoding.size(1)}")

    print("\n节点信息:")
    for i, cls in enumerate(classes):
        print(f"  {i}: {cls.name} - {cls.description}")

    print("\n边类型映射:")
    for edge_type, idx in builder.edge_type_to_idx.items():
        print(f"  {edge_type}: {idx}")

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    model = CodeGraphEncoder(
        structural_hidden_dim=256,
        structural_output_dim=256,
        semantic_output_dim=256,
        final_output_dim=512,
        num_edge_types=len(builder.edge_type_to_idx),
        num_structural_layers=3,
        num_heads=8,
        dropout=0.1
    ).to(device)

    # 移动数据到设备
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)
    pos_encoding = pos_encoding.to(device)

    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        node_embeddings = model(x, edge_index, edge_types, pos_encoding, texts)

    print(f"\n最终节点向量形状: {node_embeddings.shape}")
    print(f"每个节点的向量维度: {node_embeddings.size(1)}")

    print("\n节点向量统计:")
    for i, cls in enumerate(classes):
        embedding = node_embeddings[i]
        print(f"  {cls.name}:")
        print(f"    L2范数: {torch.norm(embedding).item():.4f}")
        print(f"    均值: {embedding.mean().item():.4f}")
        print(f"    标准差: {embedding.std().item():.4f}")

    print("\n节点间相似度 (余弦相似度):")
    node_embeddings_normalized = F.normalize(node_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(node_embeddings_normalized, node_embeddings_normalized.t())

    for i in range(min(3, len(classes))):
        for j in range(i + 1, min(3, len(classes))):
            sim = similarity_matrix[i, j].item()
            print(f"  {classes[i].name} <-> {classes[j].name}: {sim:.4f}")

    print("\n✓ 系统运行成功!")
