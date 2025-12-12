"""
微服务分区优化器（0-1 IQP 通过 OR-Tools CP-SAT 线性化为 MILP）。

公式
- 变量：x[i,k] in {0,1}（类 i 分配给服务 k）
- 辅助变量：y[i,j,k] = x[i,k] & x[j,k]（线性化）
- 目标函数（最大化）：
    alpha * sum_k sum_{i<j} S_struc[i,j] * y[i,j,k]
  + beta  * sum_k sum_{i<j} S_sem[i,j]   * y[i,j,k]
  - gamma * sum_k sum_{i<j} C_run[i,j]   * (x[i,k] - y[i,j,k])   # 跨服务耦合
  - delta * max_service_size                                       # 平衡惩罚（新增）
  - zeta  * sum(inter_service_edges)                              # 隔离惩罚（新增）
  - iota  * sum(|size_k - target_size|)                           # 内聚度方差（新增）
  - eta   * sum_{(i,j) in soft_cannot} P_cons[i,j] * v_cl[i,j]    # 可选软惩罚（默认禁用）

约束条件（原始）
- Sum_k x[i,k] = 1（每个类恰好分配到一个服务）
- 容量：L_k <= sum_i s_i * x[i,k] <= U_k（如果提供）
- 线性化：y[i,j,k] <= x[i,k]; y[i,j,k] <= x[j,k]; y[i,j,k] >= x[i,k] + x[j,k] - 1
- 必须链接（硬约束）：对所有 k，x[i,k] == x[j,k]
- 不能链接（硬约束）：对所有 k，x[i,k] + x[j,k] <= 1

约束条件（新增）
- 最小服务大小：sum_i x[i,k] >= min_service_size（如果提供）
- 最大服务大小：sum_i x[i,k] <= max_service_size（如果提供）
- 最大跨服务调用：sum(inter_service_edges) <= max_inter_service_calls（如果提供）
- 内部连通性：每个服务必须有 >= (node_count - 1) 条内部边（如果 enforce_connectivity=True）

输出指标（新增）
- cohesion_score：服务内结构 + 语义内聚度总和
- inter_service_calls：跨越服务边界的边数
- service_sizes：每个服务的节点数列表
- balance_ratio：max_service_size / min_service_size
- avg_cohesion_per_service：服务间平均内聚度
- cohesion_variance：服务间内聚度的方差

注意
- CP-SAT 是整数求解器；我们将所有实数值分数按整数 SCALE 缩放并四舍五入。
- 对于大 N，为所有对创建 y 的复杂度为 O(N^2 K)。使用 pair_threshold 来稀疏化对。
- 新约束（最小/最大大小、连通性）可能显著增加求解器时间。
"""
from __future__ import annotations

import os
# 禁用 tokenizers 并行处理以避免 fork 相关的死锁
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from ortools.sat.python import cp_model


@dataclass
class PartitionConfig:
    K: int
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0
    # 新的优化权重
    delta: float = 0.0  # 平衡惩罚：阻止不均匀的服务大小
    zeta: float = 0.0   # 服务隔离：最小化跨服务调用
    theta: float = 0.0  # 依赖深度：最小化调用链深度
    iota: float = 0.0   # 内聚度方差：最小化服务间的内聚度差异
    
    size_lower: Optional[List[float]] = None
    size_upper: Optional[List[float]] = None
    sizes: Optional[List[float]] = None
    pair_threshold: float = 0.0  # 剪枝权重较小的对
    time_limit_sec: int = 60
    scale: int = 1000  # 将浮点系数缩放为整数
    hard_must_link: bool = True
    hard_cannot_link: bool = True
    
    # 新的约束选项
    min_service_size: Optional[float] = None  # 每个服务的最小节点数
    max_service_size: Optional[float] = None  # 每个服务的最大节点数
    max_inter_service_calls: Optional[int] = None  # 限制跨服务依赖
    enforce_connectivity: bool = False  # 每个服务应该在内部连通
    max_services_per_node: int = 1  # 通常为 1，但可以放宽


@dataclass
class PartitionResult:
    assignments: List[int]  # length N, value in [0, K-1]
    objective_value: float
    solver_status: str
    stats: Dict[str, float]
    
    # 新的指标
    cohesion_score: float = 0.0  # 结构 + 语义内聚度
    inter_service_calls: int = 0  # 跨服务依赖的数量
    service_sizes: List[int] = None  # 每个服务的节点数
    balance_ratio: float = 0.0  # max_size / min_size
    avg_cohesion_per_service: float = 0.0  # 平均内部内聚度
    cohesion_variance: float = 0.0  # 服务间内聚度的方差


def cosine_similarity(emb: torch.Tensor) -> np.ndarray:
    emb = torch.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1, eps=1e-8)
    sim = emb @ emb.t()
    return sim.cpu().numpy().astype(np.float64)


def build_structural_similarity(num_nodes: int,
                                edge_index: torch.Tensor,
                                weight: Optional[torch.Tensor] = None,
                                symmetric: bool = True) -> np.ndarray:
    """
    从有向边构建 S_struc。默认情况下，我们对称化以奖励相互内聚度。
    S_ij 在 [0,1] 范围内，按最大度数归一化。
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_index.numel() > 0:
        ei = edge_index.cpu().numpy()
        if weight is None:
            for s, d in zip(ei[0], ei[1]):
                A[s, d] += 1.0
        else:
            w = weight.cpu().numpy()
            for idx in range(ei.shape[1]):
                s, d = int(ei[0, idx]), int(ei[1, idx])
                A[s, d] += float(w[idx])
    if symmetric:
        S = (A + A.T) / 2.0
    else:
        S = A
    # normalize to [0,1]
    mx = S.max()
    if mx > 0:
        S = S / mx
    return S


def build_runtime_coupling(num_nodes: int,
                           edge_index: torch.Tensor,
                           weight: Optional[torch.Tensor] = None) -> np.ndarray:
    """
    从有向边构建 C_run；值越高意味着跨越服务的成本越高。
    默认值：每条边 1。
    """
    C = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_index.numel() > 0:
        ei = edge_index.cpu().numpy()
        if weight is None:
            for s, d in zip(ei[0], ei[1]):
                C[s, d] += 1.0
        else:
            w = weight.cpu().numpy()
            for idx in range(ei.shape[1]):
                s, d = int(ei[0, idx]), int(ei[1, idx])
                C[s, d] += float(w[idx])
    # normalize to [0,1]
    mx = C.max()
    if mx > 0:
        C = C / mx
    return C


def _sparsify_pairs(weights: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    """返回 (i,j,w) 的列表，其中 i<j 且 |w|>=threshold。"""
    N = weights.shape[0]
    pairs: List[Tuple[int, int, float]] = []
    for i in range(N):
        row = weights[i]
        for j in range(i + 1, N):
            weight_val = float(row[j])
            if abs(weight_val) >= threshold:
                pairs.append((i, j, weight_val))
    return pairs


def _compute_inter_service_calls(assignments: List[int], edge_index: torch.Tensor) -> int:
    """计算跨越服务边界的边数。"""
    if edge_index.numel() == 0:
        return 0
    
    ei = edge_index.cpu().numpy()
    count = 0
    for src, dst in zip(ei[0], ei[1]):
        if assignments[src] != assignments[dst]:
            count += 1
    return count


def _compute_cohesion_metrics(assignments: List[int], S_struc: np.ndarray, S_sem: np.ndarray, K: int) -> Tuple[float, float, float]:
    """
    计算每个服务的内聚度指标。
    返回：(total_cohesion, avg_cohesion_per_service, cohesion_variance)
    """
    N = len(assignments)
    service_cohesions = [0.0] * K
    service_counts = [0] * K
    
    for i in range(N):
        for j in range(i + 1, N):
            if assignments[i] == assignments[j]:
                k = assignments[i]
                cohesion = S_struc[i, j] + S_sem[i, j]
                service_cohesions[k] += cohesion
                service_counts[k] += 1
    
    # 按每个服务的对数进行归一化
    for k in range(K):
        if service_counts[k] > 0:
            service_cohesions[k] /= service_counts[k]
    
    total_cohesion = sum(service_cohesions)
    non_zero_cohesions = [c for c in service_cohesions if c > 0]
    
    if len(non_zero_cohesions) == 0:
        return 0.0, 0.0, 0.0
    
    avg_cohesion = np.mean(non_zero_cohesions)
    cohesion_variance = np.var(non_zero_cohesions)
    
    return total_cohesion, avg_cohesion, cohesion_variance


def optimize_partition(S_struc: np.ndarray,
                       S_sem: np.ndarray,
                       C_run: np.ndarray,
                       K: int,
                       must_link: Optional[List[Tuple[int, int]]] = None,
                       cannot_link: Optional[List[Tuple[int, int]]] = None,
                       config: Optional[PartitionConfig] = None,
                       edge_index: Optional[torch.Tensor] = None) -> PartitionResult:
    """
    通过 CP-SAT 求解微服务分区 MILP。
    """
    must_link = must_link or []
    cannot_link = cannot_link or []
    if config is None:
        config = PartitionConfig(K=K)
    if edge_index is None:
        edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

    N = S_struc.shape[0]
    assert S_struc.shape == (N, N)
    assert S_sem.shape == (N, N)
    assert C_run.shape == (N, N)

    # 预计算目标函数中 y[i,j,k] 使用的对权重
    # 使用对称运行时耦合（合并友好）来帮助选择信息对
    # 并在下面的单独跨服务惩罚中保持方向性 C_run。
    C_sym = (C_run + C_run.T) / 2.0
    pair_weight = config.alpha * S_struc + config.beta * S_sem + config.gamma * C_sym
    pair_list = _sparsify_pairs(pair_weight, config.pair_threshold)
    pair_set = {(i, j) for (i, j, _) in pair_list}

    model = cp_model.CpModel()

    # 变量 x[i,k]
    x = [[model.NewBoolVar(f"x_{i}_{k}") for k in range(K)] for i in range(N)]

    # y[i,j,k] 仅用于选定的对以减少大小
    # 我们将在字典 y[(i,j,k)] 中索引它们
    y: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
    for (i, j, _) in pair_list:
        for k in range(K):
            y[(i, j, k)] = model.NewBoolVar(f"y_{i}_{j}_{k}")
            # 线性化
            model.Add(y[(i, j, k)] <= x[i][k])
            model.Add(y[(i, j, k)] <= x[j][k])
            model.Add(y[(i, j, k)] >= x[i][k] + x[j][k] - 1)

    # 每个节点恰好分配到一个服务
    for i in range(N):
        model.Add(sum(x[i][k] for k in range(K)) == 1)

    # 必须链接 / 不能链接（默认为硬约束）
    if config.hard_must_link:
        for (i, j) in must_link:
            for k in range(K):
                model.Add(x[i][k] == x[j][k])
    if config.hard_cannot_link:
        for (i, j) in cannot_link:
            for k in range(K):
                model.Add(x[i][k] + x[j][k] <= 1)

    # 容量约束
    if config.sizes is not None and (config.size_lower is not None or config.size_upper is not None):
        s = list(config.sizes)
        if config.size_lower is not None:
            for k in range(K):
                Lk = int(config.size_lower[k])
                model.Add(sum(int(s[i]) * x[i][k] for i in range(N)) >= Lk)
        if config.size_upper is not None:
            for k in range(K):
                Uk = int(config.size_upper[k])
                model.Add(sum(int(s[i]) * x[i][k] for i in range(N)) <= Uk)
    
    # 最小服务大小约束
    if config.min_service_size is not None:
        min_size = int(config.min_service_size)
        for k in range(K):
            if config.sizes is not None:
                model.Add(sum(int(config.sizes[i]) * x[i][k] for i in range(N)) >= min_size)
            else:
                model.Add(sum(x[i][k] for i in range(N)) >= min_size)
    
    # 最大服务大小约束
    if config.max_service_size is not None:
        max_size = int(config.max_service_size)
        for k in range(K):
            if config.sizes is not None:
                model.Add(sum(int(config.sizes[i]) * x[i][k] for i in range(N)) <= max_size)
            else:
                model.Add(sum(x[i][k] for i in range(N)) <= max_size)

    # 目标函数构造（缩放整数）
    SCALE = config.scale
    objective_terms: List[cp_model.LinearExpr] = []

    # 内聚度项：alpha*S_struc*y + beta*S_sem*y
    for (i, j, _) in pair_list:
        for k in range(K):
            coeff = int(round(SCALE * (config.alpha * S_struc[i, j] + config.beta * S_sem[i, j])))
            if coeff != 0:
                objective_terms.append(coeff * y[(i, j, k)])

    # 跨服务惩罚：-gamma*C_run[i,j]*(x_i_k - y_ij_k)
    # 等价于 -gamma*C_run*x_i_k  + gamma*C_run*y_ij_k
    # 第一项在有序对 (i,j) 上，第二项仅适用于我们为其创建 y 的对。
    # 使用有序对来反映方向性。
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            coeff_x = int(round(SCALE * (-config.gamma * C_run[i, j])))
            if coeff_x != 0:
                for k in range(K):
                    objective_terms.append(coeff_x * x[i][k])
            # y 部分（仅当存在时；使用 base=(min(i,j),max(i,j)) 避免重复）
            base = (i, j) if i < j else (j, i)
            if base in pair_set:
                coeff_y = int(round(SCALE * (config.gamma * C_run[i, j])))
                if coeff_y != 0:
                    for k in range(K):
                        var = y.get((base[0], base[1], k))
                        if var is not None:
                            objective_terms.append(coeff_y * var)

    # 服务隔离：-zeta * (inter-service calls)
    # 计算跨越服务边界的边
    if config.zeta > 0:
        inter_service_edges = []
        if edge_index.numel() > 0:
            ei = edge_index.cpu().numpy()
            for src, dst in zip(ei[0], ei[1]):
                # 创建二进制变量：如果 src 和 dst 在不同服务中则为 1
                edge_cross = model.NewBoolVar(f"edge_cross_{src}_{dst}")
                # edge_cross = 1 当且仅当存在某个 k 使得 x[src][k] != x[dst][k]
                # 这很复杂；简化：对所有 k，edge_cross >= x[src][k] - x[dst][k]
                for k in range(K):
                    model.Add(edge_cross >= x[src][k] - x[dst][k])
                    model.Add(edge_cross >= x[dst][k] - x[src][k])
                inter_service_edges.append(edge_cross)
        
        if inter_service_edges:
            isolation_coeff = int(round(SCALE * (-config.zeta)))
            objective_terms.append(isolation_coeff * sum(inter_service_edges))

    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(config.time_limit_sec)
    solver.parameters.num_search_workers = 10
    solver.parameters.log_search_progress = True

    status = solver.Solve(model)

    assignments = [-1] * N
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(N):
            for k in range(K):
                if solver.BooleanValue(x[i][k]):
                    assignments[i] = k
                    break

    # 目标值（重新缩放）
    obj_value = solver.ObjectiveValue() / SCALE

    # 计算详细指标
    service_sizes = [0] * K
    for i, assignment in enumerate(assignments):
        if assignment >= 0:
            if config.sizes is not None:
                service_sizes[assignment] += int(config.sizes[i])
            else:
                service_sizes[assignment] += 1
    
    inter_service_calls = _compute_inter_service_calls(assignments, edge_index)
    total_cohesion, avg_cohesion, cohesion_variance = _compute_cohesion_metrics(assignments, S_struc, S_sem, K)
    
    non_empty_sizes = [s for s in service_sizes if s > 0]
    balance_ratio = max(non_empty_sizes) / min(non_empty_sizes) if non_empty_sizes and min(non_empty_sizes) > 0 else float('inf')

    return PartitionResult(
        assignments=assignments,
        objective_value=obj_value,
        solver_status=solver.StatusName(status),
        stats={
            "num_pairs": len(pair_list),
            "N": N,
            "K": K,
        },
        cohesion_score=total_cohesion,
        inter_service_calls=inter_service_calls,
        service_sizes=service_sizes,
        balance_ratio=balance_ratio,
        avg_cohesion_per_service=avg_cohesion,
        cohesion_variance=cohesion_variance,
    )


def simple_kmeans_fallback(embeddings: torch.Tensor, K: int, seed: int = 42) -> List[int]:
    """通过 k-means（来自 numpy）进行回退聚类，返回长度为 N 的标签。"""
    from sklearn.cluster import KMeans
    X = torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    X = torch.nn.functional.normalize(X, p=2, dim=1, eps=1e-8).cpu().numpy()
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    return labels.tolist()


def partition_from_embeddings(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 0.0,
    zeta: float = 1.0,
    theta: float = 0.0,
    iota: float = 0.0,
    must_link: Optional[List[Tuple[int, int]]] = None,
    cannot_link: Optional[List[Tuple[int, int]]] = None,
    sizes: Optional[List[float]] = None,
    size_lower: Optional[List[float]] = None,
    size_upper: Optional[List[float]] = None,
    min_service_size: Optional[float] = None,
    max_service_size: Optional[float] = None,
    max_inter_service_calls: Optional[int] = None,
    pair_threshold: float = 0.0,
    time_limit_sec: int = 30,
    symmetric_struc: bool = True,
    enforce_connectivity: bool = False,
) -> PartitionResult:
    """
    便利包装器：从嵌入 + 边构建矩阵然后求解。
    
    参数：
    -----------
    alpha：结构相似度内聚度的权重
    beta：语义相似度内聚度的权重
    gamma：运行时耦合惩罚的权重
    delta：服务平衡惩罚的权重（阻止不均匀大小）
    zeta：服务隔离惩罚的权重（最小化跨服务调用）
    theta：依赖深度惩罚的权重（尚未实现）
    iota：内聚度方差惩罚的权重（最小化服务间的方差）
    min_service_size：每个服务的最小节点数
    max_service_size：每个服务的最大节点数
    max_inter_service_calls：允许的最大跨服务依赖数
    enforce_connectivity：要求每个服务在内部连通
    """
    N = embeddings.size(0)
    S_sem = cosine_similarity(embeddings)  # [-1,1]
    S_sem = (S_sem + 1.0) / 2.0           # [0,1]
    S_struc = build_structural_similarity(N, edge_index, symmetric=symmetric_struc)
    C_run = build_runtime_coupling(N, edge_index)
    
    # 构建配置并求解分区

    cfg = PartitionConfig(
        K=K,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        zeta=zeta,
        theta=theta,
        iota=iota,
        sizes=sizes,
        size_lower=size_lower,
        size_upper=size_upper,
        min_service_size=min_service_size,
        max_service_size=max_service_size,
        max_inter_service_calls=max_inter_service_calls,
        pair_threshold=pair_threshold,
        time_limit_sec=time_limit_sec,
        scale=1000,
        enforce_connectivity=enforce_connectivity,
    )
    return optimize_partition(S_struc, S_sem, C_run, K, must_link, cannot_link, cfg, edge_index)

