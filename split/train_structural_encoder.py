"""
训练仅结构编码器，使直接相连的节点在embedding空间更接近。
- 训练目标：基于图的多正样本InfoNCE + 拉普拉斯平滑（可选）
- 仅使用结构信息（无文本、无跨注意力）
- 训练完成后保存权重到 split/result/structural_best.pt

运行：
  python -m split.train_structural_encoder
"""

import os
import torch
import torch.nn.functional as F
from typing import List

from split.encoder.code_graph_encoder import (
    CodeClass, CodeGraphDataBuilder, StructuralEncoder
)
from split.utils.data_processor import load_json
import split.config as config


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


def build_graph(classes: List[CodeClass]):
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, _, edge_weights = builder.build_graph_data(
        edge_type_weights=config.PartitionConfig().edge_type_weights
    )

    # 构建无向邻接（正样本掩码）与对应的权重矩阵，忽略自环
    num_nodes = len(classes)
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    pos_weight = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # 将连接视为无向正样本
        if edge_weights is not None and edge_weights.numel() > 0:
            # 根据边类型权重构建加权正样本矩阵
            pos_weight[src, dst] = edge_weights
            pos_weight[dst, src] = edge_weights
        else:
            pos_weight[src, dst] = 1.0
            pos_weight[dst, src] = 1.0
    adj.fill_diagonal_(0.0)
    pos_weight.fill_diagonal_(0.0)

    return builder, x, edge_index, edge_types, pos_encoding, adj, pos_weight, edge_weights


class StructuralTrainer:
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_edge_types: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.2,
        lambda_lap: float = 0.1,  # 拉普拉斯平滑系数（>0 即启用）
        device: torch.device | None = None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StructuralEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_edge_types=num_edge_types,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        ).to(self.device)

        self.temperature = temperature
        self.lambda_lap = lambda_lap

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.amp.GradScaler('cpu')

    def multi_positive_infonce(self, z: torch.Tensor, pos_mask: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        多正样本InfoNCE（支持加权正样本）：
        对于每个节点 i，正样本为其邻居集合 P(i)。
        loss_i = -log( sum_{j in P(i)} w_ij * exp(sim(i,j)/tau) / sum_{k != i} exp(sim(i,k)/tau) )
        仅对 P(i) 非空的节点计入损失。
        """
        z = F.normalize(z, p=2, dim=-1)
        sim = torch.matmul(z, z.t())  # [N,N], 余弦相似度
        logits = sim / self.temperature

        # 排除自身
        N = z.size(0)
        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        denom_mask = ~eye  # k != i

        # 仅对有正样本的行计算
        pos_mask = pos_mask.to(z.device).bool()
        weight = pos_weight.to(z.device) if pos_weight is not None else pos_mask.float()
        has_pos = (weight > 0).any(dim=1)
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # 数值稳定：减去每行最大值
        row_max = logits.max(dim=1, keepdim=True).values
        logits = logits - row_max

        exp_logits = torch.exp(logits)
        denom = (exp_logits * denom_mask.float()).sum(dim=1)  # [N]
        numer = (exp_logits * weight).sum(dim=1)  # [N]

        # 仅保留有正样本的项
        numer = numer[has_pos]
        denom = denom[has_pos] + 1e-12

        loss = -torch.log(numer / denom + 1e-12).mean()
        return loss

    def laplacian_smoothing(self, z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        拉普拉斯平滑项：sum_{(i,j) in E} ||z_i - z_j||^2
        """
        if adj.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        D = torch.diag(adj.sum(dim=1))
        L = D - adj
        loss = torch.trace(z.t() @ L @ z) / (adj.sum() + 1e-12)
        return loss

    def train(self,
              x: torch.Tensor,
              edge_index: torch.Tensor,
              edge_types: torch.Tensor,
              pos_encoding: torch.Tensor,
              pos_mask: torch.Tensor,
              pos_weight: torch.Tensor | None = None,
              edge_weights: torch.Tensor | None = None,
              epochs: int = 50,
              ckpt_path: str = "split/result/structural_best.pt"):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_types = edge_types.to(self.device)
        pos_encoding = pos_encoding.to(self.device)
        pos_mask = pos_mask.to(self.device)
        pos_weight = pos_weight.to(self.device) if pos_weight is not None else None
        edge_weights = edge_weights.to(self.device) if edge_weights is not None and edge_weights.numel() > 0 else None

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        best_loss = float('inf')
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                z = self.model(x, edge_index, edge_types, pos_encoding, edge_weights)
                loss_contrast = self.multi_positive_infonce(z, pos_mask, pos_weight=pos_weight)
                loss_lap = self.laplacian_smoothing(z, pos_mask) if self.lambda_lap > 0 else 0.0
                loss = loss_contrast + self.lambda_lap * loss_lap

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                cur_loss = float(loss.detach().cpu().item())
                print(f"[Epoch {epoch:03d}] loss={cur_loss:.6f} (contrast={float(loss_contrast):.6f}{f', lap={float(loss_lap):.6f}' if self.lambda_lap>0 else ''})")
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            torch.save(best_state, ckpt_path)
            print(f"✓ 最优模型已保存: {ckpt_path}  | best_loss={best_loss:.6f}")
        else:
            print("! 未保存模型（未找到更优状态）")


def main():
    # 读取数据
    classes = load_data(config.DataConfig.dataset_path)

    # 默认采用 SMALL/MEDIUM 图的结构编码配置
    from split.config import get_config_by_graph_size
    enc_cfg = get_config_by_graph_size(len(classes)).structural

    # 构建图
    builder, x, edge_index, edge_types, pos_encoding, pos_mask, pos_weight, edge_weights = build_graph(classes)

    trainer = StructuralTrainer(
        node_feature_dim=enc_cfg.node_feature_dim,
        hidden_dim=enc_cfg.hidden_dim,
        output_dim=enc_cfg.output_dim,
        num_edge_types=len(builder.edge_type_to_idx) if len(builder.edge_type_to_idx) > 0 else enc_cfg.num_edge_types,
        num_layers=enc_cfg.num_layers,
        num_heads=enc_cfg.num_heads,
        dropout=enc_cfg.dropout,
        lr=1e-3,
        weight_decay=1e-4,
        temperature=0.2,
        lambda_lap=0.1,  # 开启拉普拉斯平滑，数值可按需要调整
    )

    trainer.train(
        x=x,
        edge_index=edge_index,
        edge_types=edge_types,
        pos_encoding=pos_encoding,
        pos_mask=pos_mask,
        pos_weight=pos_weight,
        edge_weights=edge_weights,
        epochs=200,
        ckpt_path="split/result/structural_best.pt",
    )


if __name__ == "__main__":
    main()
