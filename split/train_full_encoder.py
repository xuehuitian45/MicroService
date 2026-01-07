"""
联合训练结构编码器 + 语义编码器 + 跨注意力融合：
- 结构部分：沿用当前 Graphormer 结构编码 + 多正样本 InfoNCE + （可选）拉普拉斯平滑
- 语义部分：沿用当前 BGE-M3 语义编码
- 跨注意力：使用 CrossAttentionFusion，将结构 / 文本对齐，使 cross-attn 有实际作用

训练目标：
    loss = loss_struct + lambda_lap * loss_lap + lambda_align * loss_align

其中：
- loss_struct: 和结构专训脚本相同的多正样本 InfoNCE
- loss_lap:   拉普拉斯平滑项（可选）
- loss_align: 结构 – 文本 对齐损失（每个类的结构向量和对应文本向量对齐）

运行：
  python -m split.train_full_encoder
"""

import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from split.encoder.code_graph_encoder import (
    CodeClass,
    CodeGraphDataBuilder,
    CodeGraphEncoder,
)
from split.utils.data_processor import load_json
import split.config as config


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


def build_graph(classes: List[CodeClass]):
    """
    构建图 + 正样本掩码 / 权重矩阵：
    - adj: 无向 0/1 邻接矩阵，用于拉普拉斯平滑
    - pos_mask: 同 adj，用于多正样本 InfoNCE
    - pos_weight: 按边类型权重加权的正样本权重矩阵
    """
    builder = CodeGraphDataBuilder(classes)
    x, edge_index, edge_types, pos_encoding, texts, edge_weights = builder.build_graph_data(
        edge_type_weights=config.PartitionConfig().edge_type_weights
    )

    num_nodes = len(classes)
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    pos_weight = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    if edge_index.numel() > 0:
        src, dst = edge_index
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0

        if edge_weights is not None and edge_weights.numel() > 0:
            pos_weight[src, dst] = edge_weights
            pos_weight[dst, src] = edge_weights
        else:
            pos_weight[src, dst] = 1.0
            pos_weight[dst, src] = 1.0

    adj.fill_diagonal_(0.0)
    pos_weight.fill_diagonal_(0.0)

    # pos_mask：无向邻接
    pos_mask = (adj > 0).float()

    return (
        builder,
        x,
        edge_index,
        edge_types,
        pos_encoding,
        texts,
        adj,
        pos_mask,
        pos_weight,
        edge_weights,
    )


def multi_positive_infonce(
    z: torch.Tensor,
    pos_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    多正样本 InfoNCE（与结构专训版本保持一致）：
        对于每个节点 i，正样本为其邻居集合 P(i)。
        loss_i = -log( sum_{j in P(i)} w_ij * exp(sim(i,j)/tau) / sum_{k != i} exp(sim(i,k)/tau) )
    """
    z = F.normalize(z, p=2, dim=-1)
    sim = torch.matmul(z, z.t())
    logits = sim / temperature

    N = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    denom_mask = ~eye

    pos_mask = pos_mask.to(z.device).bool()
    weight = pos_weight.to(z.device) if pos_weight is not None else pos_mask.float()
    has_pos = (weight > 0).any(dim=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    row_max = logits.max(dim=1, keepdim=True).values
    logits = logits - row_max

    exp_logits = torch.exp(logits)
    denom = (exp_logits * denom_mask.float()).sum(dim=1)
    numer = (exp_logits * weight).sum(dim=1)

    numer = numer[has_pos]
    denom = denom[has_pos] + 1e-12

    loss = -torch.log(numer / denom + 1e-12).mean()
    return loss


def laplacian_smoothing(z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    拉普拉斯平滑项：sum_{(i,j) in E} ||z_i - z_j||^2
    """
    if adj.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    D = torch.diag(adj.sum(dim=1))
    L = D - adj
    loss = torch.trace(z.t() @ L @ z) / (adj.sum() + 1e-12)
    return loss


def struct_text_alignment_loss(
    z_struct: torch.Tensor,
    z_text: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    结构 – 文本 对齐损失：
    - z_struct[i] 应该和 z_text[i] 对齐
    - 使用 InfoNCE / 分类形式的对齐：
        sim = z_struct @ z_text^T / tau
        labels = arange(N)
        loss = CrossEntropy(sim, labels)
    """
    # 归一化后做对齐，更稳定
    z_s = F.normalize(z_struct, p=2, dim=-1)
    z_t = F.normalize(z_text, p=2, dim=-1)

    sim = torch.matmul(z_s, z_t.t()) / temperature  # [N, N]
    labels = torch.arange(z_struct.size(0), device=z_struct.device)
    return F.cross_entropy(sim, labels)


def fused_alignment_loss(
    z_fused: torch.Tensor,
    z_target: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    融合表示对齐损失（支持不同维度）：
    - z_fused[i] 应该和 z_target[i] 对齐
    - 使用逐样本余弦相似度计算对齐损失
    - 支持 z_fused 和 z_target 维度不同的情况
    """
    # 归一化
    z_f = F.normalize(z_fused, p=2, dim=-1)  # [N, D1]
    z_t = F.normalize(z_target, p=2, dim=-1)  # [N, D2]

    # 对齐维度：使用较小的维度，直接截取前 min_dim 维
    min_dim = min(z_f.size(-1), z_t.size(-1))

    # 直接截取前 min_dim 维（简单且有效）
    z_f_aligned = z_f[:, :min_dim]
    z_t_aligned = z_t[:, :min_dim]

    # 确保维度完全一致（处理边界情况）
    final_dim = min(z_f_aligned.size(-1), z_t_aligned.size(-1))
    z_f_aligned = z_f_aligned[:, :final_dim]
    z_t_aligned = z_t_aligned[:, :final_dim]

    # 重新归一化（池化后可能需要）
    z_f_aligned = F.normalize(z_f_aligned, p=2, dim=-1)
    z_t_aligned = F.normalize(z_t_aligned, p=2, dim=-1)

    # 现在维度相同，计算余弦相似度矩阵
    sim = torch.matmul(z_f_aligned, z_t_aligned.t()) / temperature  # [N, N]
    labels = torch.arange(z_fused.size(0), device=z_fused.device)
    return F.cross_entropy(sim, labels)


class FullEncoderTrainer:
    def __init__(
        self,
        encoder: CodeGraphEncoder,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        temperature_struct: float = 0.2,
        lambda_lap: float = 0.1,
        # 对齐损失权重：1.0 时对齐项过强，容易和结构对比目标“拉扯”
        # 这里调低到 0.3，让结构对比 loss 主导，alignment 作为辅助
        lambda_align: float = 0.3,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = encoder.to(self.device)

        self.temperature_struct = temperature_struct
        self.lambda_lap = lambda_lap
        self.lambda_align = lambda_align

        # 只优化需要训练的参数（例如结构编码器可以选择冻结）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.amp.GradScaler("cpu")

    def train(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        pos_encoding: torch.Tensor,
        texts: List[str],
        adj: torch.Tensor,
        pos_mask: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        edge_weights: torch.Tensor | None = None,
        epochs: int = 100,
        ckpt_path: str = "split/result/full_encoder_best.pt",
    ):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_types = edge_types.to(self.device)
        pos_encoding = pos_encoding.to(self.device)
        adj = adj.to(self.device)
        pos_mask = pos_mask.to(self.device)
        pos_weight = pos_weight.to(self.device) if pos_weight is not None else None
        edge_weights = (
            edge_weights.to(self.device) if edge_weights is not None and edge_weights.numel() > 0 else None
        )

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        best_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            # 使用推荐的 torch.amp.autocast API（兼容 CPU / CUDA）
            amp_device = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(amp_device, enabled=True):
                # 得到三种 embedding
                z_struct, z_text, z_fused = self.model(
                    x=x,
                    edge_index=edge_index,
                    edge_types=edge_types,
                    pos_encoding=pos_encoding,
                    texts=texts,
                    edge_weights=edge_weights,
                )

                # 注意：由于结构编码器和语义编码器被冻结，它们的输出没有梯度
                # 因此所有损失必须基于融合后的输出 z_fused 来计算

                # 结构对比损失：使用融合后的表示，让融合模块学习保持结构信息
                loss_struct = multi_positive_infonce(
                    z_fused,
                    pos_mask=pos_mask,
                    pos_weight=pos_weight,
                    temperature=self.temperature_struct,
                )

                # 拉普拉斯平滑：使用融合后的表示
                loss_lap = laplacian_smoothing(z_fused, adj) if self.lambda_lap > 0 else 0.0

                # 对齐损失：让融合后的表示对齐到结构表示和语义表示
                # 使用 detach() 避免对冻结模块的梯度计算
                # 使用 fused_alignment_loss 处理不同维度的情况
                loss_align_struct = fused_alignment_loss(z_fused, z_struct.detach())
                loss_align_text = fused_alignment_loss(z_fused, z_text.detach())
                loss_align = (loss_align_struct + loss_align_text) / 2.0

                loss = loss_struct + self.lambda_lap * loss_lap + self.lambda_align * loss_align

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                cur_loss = float(loss.detach().cpu().item())
                print(
                    f"[Epoch {epoch:03d}] "
                    f"loss={cur_loss:.6f} "
                    f"(struct={float(loss_struct):.6f}"
                    f"{f', lap={float(loss_lap):.6f}' if self.lambda_lap > 0 else ''}"
                    f", align={float(loss_align):.6f})"
                )
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            torch.save(best_state, ckpt_path)
            print(f"✓ 最优 full encoder 模型已保存: {ckpt_path}  | best_loss={best_loss:.6f}")
        else:
            print("! 未保存模型（未找到更优状态）")


def main():
    # 1. 读取数据
    classes = load_data(config.DataConfig.dataset_path)

    # 2. 构图 + 正样本 / 权重
    (
        builder,
        x,
        edge_index,
        edge_types,
        pos_encoding,
        texts,
        adj,
        pos_mask,
        pos_weight,
        edge_weights,
    ) = build_graph(classes)

    # 3. 根据图规模选择配置
    from split.config import get_config_by_graph_size

    encoder_cfg = get_config_by_graph_size(len(classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. 创建完整编码器（结构 + 语义 + 融合）
    model = CodeGraphEncoder(
        structural_hidden_dim=encoder_cfg.structural.hidden_dim,
        structural_output_dim=encoder_cfg.structural.output_dim,
        semantic_output_dim=encoder_cfg.semantic.output_dim,
        final_output_dim=encoder_cfg.fusion.output_dim,
        num_edge_types=len(builder.edge_type_to_idx) if len(builder.edge_type_to_idx) > 0 else encoder_cfg.structural.num_edge_types,
        num_structural_layers=encoder_cfg.structural.num_layers,
        num_heads=encoder_cfg.structural.num_heads,
        dropout=encoder_cfg.structural.dropout,
        code_encoder_model=encoder_cfg.semantic.model_name,
        freeze_code_encoder=encoder_cfg.semantic.freeze_encoder,
        structural_only=False,  # 关键：启用语义 + 融合
    ).to(device)

    # 4.1 结构预训练加载：先用结构专训得到的 structural_best.pt 初始化结构编码器
    #      这样：
    #      - 结构 embedding 直接继承结构预训练效果
    #      - full 训练阶段主要训练语义编码器 + 融合模块，让它们对齐到结构空间
    struct_ckpt = "split/result/structural_best.pt"
    if os.path.exists(struct_ckpt):
        try:
            print(f"\n[FullEncoder] 检测到结构预训练模型: {struct_ckpt}")
            model.structural_encoder.load_pretrained(struct_ckpt, device=device)
            print("✓ 结构编码器已用 structural_best.pt 初始化")
        except Exception as e:
            print(f"! 加载结构预训练模型失败: {e}，将使用随机初始化结构编码器")
    else:
        print(f"\n[FullEncoder] 未找到结构预训练模型 ({struct_ckpt})，将使用随机初始化结构编码器")

    # 4.2 冻结结构编码器和语义编码器，只训练融合模块
    for p in model.structural_encoder.parameters():
        p.requires_grad = False

    # 冻结语义编码器
    for p in model.semantic_encoder.parameters():
        p.requires_grad = False

    # 打印可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[FullEncoder] 参数统计:")
    print(f"  - 总参数数: {total_params:,}")
    print(f"  - 可训练参数数: {trainable_params:,} (仅融合模块)")
    print(f"  - 冻结参数数: {total_params - trainable_params:,} (结构+语义编码器)")

    trainer = FullEncoderTrainer(
        encoder=model,
        lr=1e-4,
        weight_decay=1e-4,
        temperature_struct=0.2,
        lambda_lap=0,
        lambda_align=0,
        device=device,
    )

    trainer.train(
        x=x,
        edge_index=edge_index,
        edge_types=edge_types,
        pos_encoding=pos_encoding,
        texts=texts,
        adj=adj,
        pos_mask=pos_mask,
        pos_weight=pos_weight,
        edge_weights=edge_weights,
        epochs=30,
        ckpt_path="split/result/full_encoder_best.pt",
    )

    # 5. 训练后调试：详细的节点对相似度统计分析
    model.eval()
    with torch.no_grad():
        z_struct, z_text, z_fused = model(
            x=x.to(device),
            edge_index=edge_index.to(device),
            edge_types=edge_types.to(device),
            pos_encoding=pos_encoding.to(device),
            texts=texts,
            edge_weights=edge_weights.to(device) if edge_weights is not None and edge_weights.numel() > 0 else None,
        )
        
        # 归一化并计算相似度矩阵
        z_fused_norm = F.normalize(z_fused, p=2, dim=-1)
        z_text_norm = F.normalize(z_text, p=2, dim=-1)
        
        sim_fused = z_fused_norm @ z_fused_norm.t()  # [N, N] 融合向量相似度
        sim_text = z_text_norm @ z_text_norm.t()  # [N, N] 语义向量相似度
        
        N = sim_fused.size(0)
        
        # ========== 1. 按相似度从大到小排序的前100个节点对 ==========
        print("\n" + "="*80)
        print("[Debug 1] 融合向量相似度最高的前100个节点对（按相似度降序）:")
        print("="*80)
        
        # 获取上三角矩阵的所有节点对（避免重复和自环）
        all_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                all_pairs.append((i, j, float(sim_fused[i, j].detach().cpu().item())))
        
        # 按相似度降序排序
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 打印前100个
        for idx, (i, j, score) in enumerate(all_pairs[:100], 1):
            name_i = classes[i].name if i < len(classes) else f"Node_{i}"
            name_j = classes[j].name if j < len(classes) else f"Node_{j}"
            print(f"  {idx:3d}. ({i:04d}) {name_i:30s} <-> ({j:04d}) {name_j:30s} | cos={score:.6f}")
        
        # ========== 2. 直接相连、两跳相连以及其它节点对的统计 ==========
        print("\n" + "="*80)
        print("[Debug 2] 按图结构距离分组的融合向量相似度统计:")
        print("="*80)
        
        # 构建邻接矩阵（无向图）
        adj_matrix = torch.zeros(N, N, dtype=torch.bool, device=device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            adj_matrix[src, dst] = True
            adj_matrix[dst, src] = True
        
        # 计算两跳邻接矩阵（通过矩阵乘法）
        adj_2hop = (adj_matrix.float() @ adj_matrix.float()) > 0
        adj_2hop = adj_2hop & (~adj_matrix)  # 排除一跳的边
        adj_2hop.fill_diagonal_(False)  # 排除自环
        
        # 分类节点对
        pairs_1hop = []  # 直接相连
        pairs_2hop = []  # 两跳相连
        pairs_other = []  # 其它（三跳及以上或不相连）
        
        for i in range(N):
            for j in range(i + 1, N):
                score = float(sim_fused[i, j].detach().cpu().item())
                if adj_matrix[i, j]:
                    pairs_1hop.append(score)
                elif adj_2hop[i, j]:
                    pairs_2hop.append(score)
                else:
                    pairs_other.append(score)
        
        def compute_stats(scores):
            if len(scores) == 0:
                return 0.0, 0.0, 0
            mean_val = sum(scores) / len(scores)
            variance = sum((x - mean_val) ** 2 for x in scores) / len(scores)
            return mean_val, variance, len(scores)
        
        mean_1hop, var_1hop, count_1hop = compute_stats(pairs_1hop)
        mean_2hop, var_2hop, count_2hop = compute_stats(pairs_2hop)
        mean_other, var_other, count_other = compute_stats(pairs_other)
        
        print(f"  直接相连（一跳）:")
        print(f"    节点对数: {count_1hop}")
        print(f"    平均相似度: {mean_1hop:.6f}")
        print(f"    方差: {var_1hop:.6f}")
        print(f"    标准差: {var_1hop**0.5:.6f}")
        
        print(f"\n  两跳相连:")
        print(f"    节点对数: {count_2hop}")
        print(f"    平均相似度: {mean_2hop:.6f}")
        print(f"    方差: {var_2hop:.6f}")
        print(f"    标准差: {var_2hop**0.5:.6f}")
        
        print(f"\n  其它（三跳及以上或不相连）:")
        print(f"    节点对数: {count_other}")
        print(f"    平均相似度: {mean_other:.6f}")
        print(f"    方差: {var_other:.6f}")
        print(f"    标准差: {var_other**0.5:.6f}")
        
        # ========== 3. 按语义相似度分组的融合向量相似度统计 ==========
        print("\n" + "="*80)
        print("[Debug 3] 按语义相似度分组的融合向量相似度统计:")
        print("="*80)
        
        # 按语义相似度分组
        pairs_sem_high = []    # 语义相似度 >= 0.7
        pairs_sem_med_high = []  # 0.5 <= 语义相似度 < 0.7
        pairs_sem_med_low = []   # 0.3 <= 语义相似度 < 0.5
        pairs_sem_low = []       # 语义相似度 < 0.3
        
        for i in range(N):
            for j in range(i + 1, N):
                sem_score = float(sim_text[i, j].detach().cpu().item())
                fused_score = float(sim_fused[i, j].detach().cpu().item())
                
                if sem_score >= 0.7:
                    pairs_sem_high.append(fused_score)
                elif sem_score >= 0.5:
                    pairs_sem_med_high.append(fused_score)
                elif sem_score >= 0.3:
                    pairs_sem_med_low.append(fused_score)
                else:
                    pairs_sem_low.append(fused_score)
        
        mean_high, var_high, count_high = compute_stats(pairs_sem_high)
        mean_med_high, var_med_high, count_med_high = compute_stats(pairs_sem_med_high)
        mean_med_low, var_med_low, count_med_low = compute_stats(pairs_sem_med_low)
        mean_low, var_low, count_low = compute_stats(pairs_sem_low)
        
        print(f"  语义相似度 >= 0.7:")
        print(f"    节点对数: {count_high}")
        print(f"    融合向量平均相似度: {mean_high:.6f}")
        print(f"    方差: {var_high:.6f}")
        print(f"    标准差: {var_high**0.5:.6f}")
        
        print(f"\n  语义相似度 [0.5, 0.7):")
        print(f"    节点对数: {count_med_high}")
        print(f"    融合向量平均相似度: {mean_med_high:.6f}")
        print(f"    方差: {var_med_high:.6f}")
        print(f"    标准差: {var_med_high**0.5:.6f}")
        
        print(f"\n  语义相似度 [0.3, 0.5):")
        print(f"    节点对数: {count_med_low}")
        print(f"    融合向量平均相似度: {mean_med_low:.6f}")
        print(f"    方差: {var_med_low:.6f}")
        print(f"    标准差: {var_med_low**0.5:.6f}")
        
        print(f"\n  语义相似度 < 0.3:")
        print(f"    节点对数: {count_low}")
        print(f"    融合向量平均相似度: {mean_low:.6f}")
        print(f"    方差: {var_low:.6f}")
        print(f"    标准差: {var_low**0.5:.6f}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()


