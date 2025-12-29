"""
将图节点进行编码，分别采用结构和语义信息，并使用交叉注意力进行融合，采用结构向量作为查询向量，更加突出结构信息重要性
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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


class GraphormerLayer(nn.Module):
    """
    修复版 Graphormer 层（增强结构编码能力）
    关键改进：
    - 增强注意力偏置的强度和可学习性
    - 修正边权重偏置逻辑
    - 添加自环偏置和注意力温度系数
    - 优化 SPD 编码映射方式
    - 添加结构归一化
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
            bias_scale: float = 10.0,  # 注意力偏置缩放系数（增强结构权重）
            attn_temp: float = 0.5,  # 注意力温度系数（控制分布尖锐度）
    ):
        super().__init__()
        assert in_channels == out_channels, "GraphormerLayer 需 in_channels == out_channels"
        assert out_channels % num_heads == 0, "out_channels 必须能被 num_heads 整除"

        self.hidden_dim = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.max_dist = max_dist
        self.max_degree = max_degree
        self.bias_scale = bias_scale  # 新增：偏置强度缩放
        self.attn_temp = attn_temp  # 新增：注意力温度

        # QKV 投影
        self.q_proj = nn.Linear(out_channels, out_channels)
        self.k_proj = nn.Linear(out_channels, out_channels)
        self.v_proj = nn.Linear(out_channels, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)

        # 注意力偏置（增强版）
        self.spd_bias_table = nn.Embedding(self.max_dist + 2, num_heads)  # +2 预留自环/超长距离
        self.edge_type_bias = nn.Embedding(num_edge_types, num_heads)
        self.self_loop_bias = nn.Parameter(torch.zeros(num_heads))  # 新增：自环偏置

        # 中心性编码（添加可学习缩放）
        self.in_degree_emb = nn.Embedding(self.max_degree + 1, out_channels)
        self.out_degree_emb = nn.Embedding(self.max_degree + 1, out_channels)
        self.degree_scale = nn.Parameter(torch.ones(1))  # 新增：中心性编码缩放

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
        # 初始化自环偏置为正值（增强自注意力）
        nn.init.constant_(self.self_loop_bias, 1.0)

    @staticmethod
    def _compute_degrees(num_nodes: int, edge_index: torch.Tensor) -> Tuple[Tensor, Tensor]:
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
            pos_encoding: Tensor,
            device: torch.device,
            edge_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        修复版注意力偏置构建：
        1. 增强偏置强度
        2. 修正边权重逻辑
        3. 添加自环偏置
        4. 优化 SPD 映射
        """
        # 初始化偏置矩阵 [H, N, N]
        attn_bias = torch.zeros(self.num_heads, num_nodes, num_nodes, device=device)

        # 1. 自环偏置（节点对自身的注意力增强）
        self_loop_mask = torch.eye(num_nodes, dtype=torch.bool, device=device)  # [N,N]
        for h in range(self.num_heads):
            attn_bias[h][self_loop_mask] += self.self_loop_bias[h]

        # 2. SPD 偏置（优化映射方式）
        if pos_encoding is not None and pos_encoding.size(1) >= num_nodes:
            spd_norm = pos_encoding[:, :num_nodes]  # [N, N]
            # 优化：分段映射 SPD，保留更多细节
            spd_bucket = torch.where(
                spd_norm == 0,  # 自环
                torch.tensor(0, device=device),
                torch.where(
                    spd_norm > 1.0,  # 超长距离
                    torch.tensor(self.max_dist + 1, device=device),
                    (spd_norm * self.max_dist).clamp(1, self.max_dist).long()
                )
            )
            spd_bias = self.spd_bias_table(spd_bucket)  # [N, N, H]
            # 缩放偏置强度
            attn_bias = attn_bias + spd_bias.permute(2, 0, 1) * self.bias_scale

        # 3. 边类型 + 边权重偏置（修正逻辑）
        if edge_index.numel() > 0 and edge_types.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            e_bias = self.edge_type_bias(edge_types)  # [E, H]

            # 修复：边权重应增强偏置（而非缩放），权重越高偏置越大
            if edge_weights is not None:
                # 将权重从 [0,1] 映射到 [0, 2*bias_scale]，增强高权重边的注意力
                weight_scale = edge_weights.unsqueeze(1) * 2 * self.bias_scale  # [E,1]
                e_bias = e_bias + weight_scale  # 加法增强而非乘法缩放

            # 累加到偏置矩阵（带缩放）
            for h in range(self.num_heads):
                attn_bias[h].index_put_((src, dst), e_bias[:, h] * self.bias_scale, accumulate=True)

        return attn_bias

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_types: Tensor,
            pos_encoding: Tensor,
            edge_weights: Optional[Tensor] = None,
    ) -> Tensor:
        N, D = x.size()
        device = x.device

        # 1) 中心性编码（添加可学习缩放）
        deg_in, deg_out = self._compute_degrees(N, edge_index)
        deg_in = deg_in.clamp(max=self.max_degree)
        deg_out = deg_out.clamp(max=self.max_degree)
        degree_emb = (self.in_degree_emb(deg_in) + self.out_degree_emb(deg_out)) * self.degree_scale
        x = x + degree_emb

        # 2) Pre-LN + 多头注意力
        x_norm = self.ln1(x)

        # QKV 投影 + 重塑
        q = self.q_proj(x_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Hd]
        k = self.k_proj(x_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Hd]
        v = self.v_proj(x_norm).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Hd]

        # 构建注意力偏置
        attn_bias = self._build_attention_bias(N, edge_index, edge_types, pos_encoding, device, edge_weights)

        # 计算注意力分数（添加温度系数）
        if hasattr(F, 'scaled_dot_product_attention'):
            # Flash Attention 路径（带温度和偏置）
            q_t = q.unsqueeze(0) / self.attn_temp  # 温度系数缩放
            k_t = k.unsqueeze(0) / self.attn_temp
            v_t = v.unsqueeze(0)
            attn_mask = attn_bias.unsqueeze(0)

            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
            attn_out = attn_out.squeeze(0)
        else:
            # 手动计算路径（带温度和偏置）
            scores = torch.matmul(q, k.transpose(-2, -1)) / (np.sqrt(self.head_dim) * self.attn_temp)
            scores = scores + attn_bias  # 偏置叠加
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            attn_out = torch.matmul(attn, v)

        # 合并 heads + 投影
        attn_out = attn_out.transpose(0, 1).contiguous().view(N, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_dropout(attn_out)

        # 残差连接
        x = x + attn_out

        # 3) FFN
        y = self.ln2(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        out = x + y

        return out


class StructuralEncoder(nn.Module):
    """
    增强版结构编码器：添加层归一化和输出结构约束
    """

    def __init__(
            self,
            node_feature_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_edge_types: int,
            num_layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.1,
            bias_scale: float = 10.0,
            attn_temp: float = 0.5,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 初始节点特征投影（添加偏置和归一化）
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Graphormer层（使用增强版）
        self.graphormer_layers = nn.ModuleList([
            GraphormerLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_edge_types=num_edge_types,
                num_heads=num_heads,
                dropout=dropout,
                bias_scale=bias_scale,
                attn_temp=attn_temp
            )
            for _ in range(num_layers)
        ])

        # 输出投影（添加结构归一化）
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_types: Tensor,
            pos_encoding: Tensor,
            edge_weights: Optional[Tensor] = None
    ) -> Tensor:
        # 嵌入初始特征
        x = self.node_embedding(x)

        # 通过Graphormer层
        for layer in self.graphormer_layers:
            x = layer(x, edge_index, edge_types, pos_encoding, edge_weights)

        # 输出投影 + L2归一化（增强余弦相似度的可解释性）
        x = self.output_proj(x)
        x = F.normalize(x, p=2, dim=-1)  # 新增：输出向量L2归一化

        return x

    def load_pretrained(self, checkpoint_path: str, device: torch.device | None = None):
        """
        加载预训练的结构编码器权重
        
        Args:
            checkpoint_path: 预训练权重文件路径
            device: 加载到的设备（默认为当前模型所在设备）
        """
        if device is None:
            device = next(self.parameters()).device
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(state_dict, strict=True)
        print(f"✓ 已加载预训练结构编码器: {checkpoint_path}")



class SemanticEncoder(nn.Module):
    """
    语义编码器：使用 BGE-M3 模型进行语义编码
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        output_dim: int = 256,
        freeze_encoder: bool = False,
        batch_size: int = 32
    ):
        super().__init__()

        self.model_name = model_name
        self.batch_size = batch_size

        # 加载 BGE-M3 模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # 获取编码器维度
        encoder_dim = self.encoder.config.hidden_size

        # 投影层：将 BGE-M3 的输出维度投影到目标维度
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
        device = next(self.encoder.parameters()).device
        all_embeddings = []

        # 批量处理文本
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # 分词和编码
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,  # BGE-M3 支持更长的序列
                return_tensors="pt"
            )

            # 移到同一设备
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

            # 获取编码
            with torch.no_grad() if not self.training else torch.enable_grad():
                model_output = self.encoder(**encoded_input)

                # BGE-M3 使用 mean pooling
                # 检查是否有 pooler_output
                if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
                    # 如果有 pooler_output，直接使用
                    embeddings = model_output.pooler_output
                else:
                    # 否则使用 mean pooling
                    embeddings = model_output.last_hidden_state
                    # 对 padding 位置进行掩码
                    attention_mask = encoded_input['attention_mask']
                    # 扩展 attention_mask 的维度以匹配 embeddings
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    # 计算有效 token 的平均值
                    sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

            # 投影到目标维度
            semantic_repr = self.projection(embeddings)

            all_embeddings.append(semantic_repr)

        # 合并所有批次
        return torch.cat(all_embeddings, dim=0)


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
            # PyTorch 要求 scale 为 float，这里将自适应缩放转换为 Python float
            if self.use_adaptive_scale:
                scale = float(torch.exp(self.attention_scale).item())
            else:
                scale = float(self.attention_scale)

            # 使用Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q.unsqueeze(0).transpose(1, 2),
                K.unsqueeze(0).transpose(1, 2),
                V.unsqueeze(0).transpose(1, 2),
                attn_mask=attention_mask.unsqueeze(0) if attention_mask is not None else None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
                scale=scale  # 传入自适应缩放因子（float）
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
        code_encoder_model: str = "BAAI/bge-m3",
        freeze_code_encoder: bool = False,
        structural_only: bool = False
    ):
        super().__init__()

        self.structural_output_dim = structural_output_dim
        self.semantic_output_dim = semantic_output_dim
        self.final_output_dim = final_output_dim
        self.structural_only = structural_only

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

        if not self.structural_only:
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
        else:
            self.semantic_encoder = None
            self.fusion = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        pos_encoding: torch.Tensor,
        texts: Optional[List[str]] = None,
        edge_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: [num_nodes, 1] 初始节点特征
            edge_index: [2, num_edges] 边索引
            edge_types: [num_edges] 边类型
            pos_encoding: [num_nodes, pos_dim] 位置编码
            texts: 节点对应的文本（类名+注释+方法签名）
            edge_weights: [num_edges] 边权重（可选），基于边类型的权重

        Returns:
            [num_nodes, final_output_dim] 最终节点向量
        """
        # 结构编码（使用边权重）
        structural_repr = self.structural_encoder(x, edge_index, edge_types, pos_encoding, edge_weights)

        # 仅结构模式：保持原有行为，返回结构 embedding
        if self.structural_only:
            return structural_repr

        # 语义编码
        assert texts is not None, "texts 不能为空，当 structural_only=False 时需要文本输入"
        semantic_repr = self.semantic_encoder(texts)

        # 跨模态融合，得到融合表示
        fused_repr = self.fusion(structural_repr, semantic_repr)

        # 返回三种表示，便于训练和下游使用：
        # - structural_repr: 结构 embedding
        # - semantic_repr: 语义 embedding
        # - fused_repr: 融合 embedding（结构 + 语义）
        return structural_repr, semantic_repr, fused_repr


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

    def build_graph_data(self, edge_type_weights=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
        """
        构建图数据

        Args:
            edge_type_weights: 边类型权重配置，如果提供则应用权重到边

        Returns:
            x: 节点特征
            edge_index: 边索引
            edge_types: 边类型
            pos_encoding: 位置编码
            texts: 节点文本
            edge_weights: 边权重（基于类型）
        """
        # 构建邻接矩阵
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        edge_list = []
        edge_type_list = []
        edge_weight_list = []

        for src_id, cls in enumerate(self.classes):
            for dst_id, edge_type in zip(cls.dependencies, cls.edge_types):
                # 获取边的权重
                weight = 1.0
                if edge_type_weights is not None:
                    weight = edge_type_weights.get_weight(edge_type)

                adj_matrix[src_id, dst_id] = weight
                edge_list.append([src_id, dst_id])
                edge_type_list.append(self.edge_type_to_idx[edge_type])
                edge_weight_list.append(weight)

        # 转换为张量
        x = torch.ones(self.num_nodes, 1, dtype=torch.float32)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_types = torch.tensor(edge_type_list, dtype=torch.long)
            edge_weights = torch.tensor(edge_weight_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros(0, dtype=torch.long)
            edge_weights = torch.zeros(0, dtype=torch.float32)

        # 计算位置编码
        pos_encoder = PositionalEncoding(self.num_nodes)

        # 组合多种位置编码
        sp_encoding = pos_encoder.shortest_path_encoding(adj_matrix)
        pr_encoding = pos_encoder.pagerank_encoding(adj_matrix)
        deg_encoding = pos_encoder.degree_encoding(adj_matrix)

        pos_encoding = torch.cat([sp_encoding, pr_encoding, deg_encoding], dim=1)

        # 构建文本（优化后的格式，更适合 BGE-M3 模型）
        texts = []
        for cls in self.classes:
            # 类名（最重要的特征）
            class_name = cls.name

            # 描述（核心功能）
            description = cls.description if cls.description else "无描述"

            # 方法列表（限制数量，避免文本过长）
            if cls.methods:
                # 只取前5个方法，避免文本过长
                methods_list = cls.methods[:5]
                methods_text = f"主要方法: {', '.join(methods_list)}"
                if len(cls.methods) > 5:
                    methods_text += f" 等共{len(cls.methods)}个方法"
            else:
                methods_text = ""

            # 组合文本：类名 + 描述 + 方法
            if methods_text:
                text = f"{class_name}. {description}. {methods_text}"
            else:
                text = f"{class_name}. {description}"

            texts.append(text)

        return x, edge_index, edge_types, pos_encoding, texts, edge_weights