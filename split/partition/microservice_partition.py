"""
微服务分区优化器（0-1 IQP 通过 OR-Tools CP-SAT 线性化为 MILP）。

公式
- 变量：x[i,k] in {0,1}（类 i 分配给服务 k）
- 辅助变量：y[i,j,k] = x[i,k] & x[j,k]（线性化）
- 目标函数（最大化）：
    alpha * sum_k sum_{i<j} S_struc[i,j] * y[i,j,k]
  + beta  * sum_k sum_{i<j} S_sem[i,j]   * y[i,j,k]
  - gamma * sum_k sum_{i<j} C_run[i,j]   * (x[i,k] - y[i,j,k])   # 跨服务耦合

约束条件
- Sum_k x[i,k] = 1（每个类恰好分配到一个服务）
- 容量：L_k <= sum_i s_i * x[i,k] <= U_k（如果提供）
- 线性化：y[i,j,k] <= x[i,k]; y[i,j,k] <= x[j,k]; y[i,j,k] >= x[i,k] + x[j,k] - 1
- 必须链接（硬约束）：对所有 k，x[i,k] == x[j,k]
- 不能链接（硬约束）：对所有 k，x[i,k] + x[j,k] <= 1

注意
- CP-SAT 是整数求解器；我们将所有实数值分数按整数 SCALE 缩放并四舍五入。
- 对于大 N，为所有对创建 y 的复杂度为 O(N^2 K)。使用 pair_threshold 来稀疏化对。
"""
from __future__ import annotations

import os

# 禁用 tokenizers 并行处理以避免 fork 相关的死锁
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable, Any
import numpy as np
import torch
import asyncio
from ortools.sat.python import cp_model


@dataclass
class PartitionConfig:
    K: int
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0

    size_lower: Optional[List[float]] = None
    size_upper: Optional[List[float]] = None
    sizes: Optional[List[float]] = None
    min_service_size: Optional[float] = None  # 每个服务的最小节点数
    max_service_size: Optional[float] = None  # 每个服务的最大节点数
    pair_threshold: float = 0.0  # 剪枝权重较小的对
    time_limit_sec: int = 60
    scale: int = 1000  # 将浮点系数缩放为整数
    hard_must_link: bool = True
    hard_cannot_link: bool = True

    # 迭代优化参数
    max_iterations: int = 2  # 最大迭代次数（1 表示无迭代，仅约束求解）
    enable_agent_optimization: bool = True  # 是否启用 Agent 优化


@dataclass
class PartitionResult:
    assignments: List[int]  # length N, value in [0, K-1]
    objective_value: float
    solver_status: str
    stats: Dict[str, float]
    iteration: int = 0  # 当前迭代次数
    total_iterations: int = 1  # 总迭代次数
    agent_feedback: Optional[Dict] = None  # Agent 优化的反馈信息


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
    
    Args:
        num_nodes: 节点数
        edge_index: [2, num_edges] 边索引
        weight: [num_edges] 边权重（可选），如果提供则使用这些权重
        symmetric: 是否对称化相似度矩阵
    
    Returns:
        [num_nodes, num_nodes] 结构相似度矩阵
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
    
    Args:
        num_nodes: 节点数
        edge_index: [2, num_edges] 边索引
        weight: [num_edges] 边权重（可选），如果提供则使用这些权重
    
    Returns:
        [num_nodes, num_nodes] 运行时耦合矩阵
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

    # 最小服务节点数约束
    if config.min_service_size is not None:
        min_size = int(config.min_service_size)
        for k in range(K):
            if config.sizes is not None:
                model.Add(sum(int(config.sizes[i]) * x[i][k] for i in range(N)) >= min_size)
            else:
                model.Add(sum(x[i][k] for i in range(N)) >= min_size)

    # 最大服务节点数约束
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

    return PartitionResult(
        assignments=assignments,
        objective_value=obj_value,
        solver_status=solver.StatusName(status),
        stats={
            "num_pairs": len(pair_list),
            "N": N,
            "K": K,
        },
        iteration=0,
        total_iterations=1,
        agent_feedback=None,
    )


def _convert_assignments_to_partitions(assignments: List[int], num_nodes: int) -> Dict[int, List[int]]:
    """
    将分配结果转换为分区字典格式。
    
    Args:
        assignments: [num_nodes] 每个节点的服务分配
        num_nodes: 节点总数
    
    Returns:
        {service_id: [node_ids]} 分区字典
    """
    partitions = {}
    for node_id, service_id in enumerate(assignments):
        if service_id not in partitions:
            partitions[service_id] = []
        partitions[service_id].append(node_id)
    return partitions


def _merge_constraints(
        must_link: Optional[List[Tuple[int, int]]],
        cannot_link: Optional[List[Tuple[int, int]]],
        agent_must_link: Optional[List[Tuple[int, int]]],
        agent_cannot_link: Optional[List[Tuple[int, int]]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    合并来自 Agent 的约束和原有约束。
    
    Args:
        must_link: 原有的必须链接约束
        cannot_link: 原有的不能链接约束
        agent_must_link: Agent 建议的必须链接约束
        agent_cannot_link: Agent 建议的不能链接约束
    
    Returns:
        (merged_must_link, merged_cannot_link)
    """
    merged_must = list(must_link or [])
    merged_cannot = list(cannot_link or [])

    # 添加 Agent 的建议，避免重复
    if agent_must_link:
        for constraint in agent_must_link:
            normalized = tuple(sorted(constraint))
            if normalized not in {tuple(sorted(c)) for c in merged_must}:
                merged_must.append(constraint)

    if agent_cannot_link:
        for constraint in agent_cannot_link:
            normalized = tuple(sorted(constraint))
            if normalized not in {tuple(sorted(c)) for c in merged_cannot}:
                merged_cannot.append(constraint)

    return merged_must, merged_cannot


async def iterative_optimize_partition(
        S_struc: np.ndarray,
        S_sem: np.ndarray,
        C_run: np.ndarray,
        K: int,
        must_link: Optional[List[Tuple[int, int]]] = None,
        cannot_link: Optional[List[Tuple[int, int]]] = None,
        config: Optional[PartitionConfig] = None,
        edge_index: Optional[torch.Tensor] = None,
        agent_optimize_fn: Optional[Callable[[PartitionResult, str], Any]] = None,
        agent_analyze_fn: Optional[Callable[[PartitionResult], Any]] = None,
        node_names: Optional[List[str]] = None,
) -> PartitionResult:
    """
    迭代优化微服务分区：约束求解 → Agent 优化 → 约束求解 → ...
    
    Args:
        S_struc: 结构相似度矩阵
        S_sem: 语义相似度矩阵
        C_run: 运行时耦合矩阵
        K: 服务数量
        must_link: 必须链接约束
        cannot_link: 不能链接约束
        config: 分区配置
        edge_index: 边索引
        agent_optimize_fn: Agent 优化函数（异步），签名为 async fn(partitions: Dict) -> Dict
        node_names: 节点名称列表（用于 Agent 理解）
    
    Returns:
        最终的分区结果
    """
    if config is None:
        config = PartitionConfig(K=K)

    if edge_index is None:
        edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)

    N = S_struc.shape[0]
    current_must_link = list(must_link or [])
    current_cannot_link = list(cannot_link or [])

    max_iterations = max(1, config.max_iterations)
    enable_agent = config.enable_agent_optimization and agent_optimize_fn is not None

    print(f"开始迭代优化：最大迭代次数={max_iterations}，启用Agent优化={enable_agent}")

    for iteration in range(max_iterations):
        print(f"\n{'=' * 60}")
        print(f"迭代 {iteration + 1}/{max_iterations}")
        print(f"{'=' * 60}")

        # 第一步：约束求解
        print(f"[迭代 {iteration + 1}] 执行约束求解...")
        result = optimize_partition(
            S_struc=S_struc,
            S_sem=S_sem,
            C_run=C_run,
            K=K,
            must_link=current_must_link,
            cannot_link=current_cannot_link,
            config=config,
            edge_index=edge_index,
        )

        result.iteration = iteration
        result.total_iterations = max_iterations

        print(f"[迭代 {iteration + 1}] 约束求解完成")
        print(f"  - 目标值: {result.objective_value:.4f}")
        print(f"  - 求解器状态: {result.solver_status}")

        # 如果是最后一次迭代或不启用 Agent 优化，直接返回
        if iteration == max_iterations - 1 or not enable_agent:
            print(f"\n优化完成（迭代 {iteration + 1}/{max_iterations}）")
            return result

        # 第二步：Agent 优化
        print(f"[迭代 {iteration + 1}] 调用 Agent 进行优化...")
        partitions = _convert_assignments_to_partitions(result.assignments, N)

        # 如果提供了节点名称，构建更友好的分区表示
        if node_names:
            partitions_with_names = {
                k: [node_names[i] for i in v]
                for k, v in partitions.items()
            }
        else:
            partitions_with_names = partitions

        try:
            analyze_result = await agent_analyze_fn(partitions_with_names)
            if analyze_result is None:
                print(f"[迭代 {iteration + 1}] Agent 分析返回 None，结束迭代")
                return result
            if hasattr(analyze_result, 'needs_optimization'):
                if not analyze_result.needs_optimization:
                    print(f"[迭代 {iteration + 1}] Agent 分析不需要优化，结束迭代")
                    return result
            suggestions = getattr(analyze_result, 'suggestions', '') or ''
            optimize_result = await agent_optimize_fn(partitions_with_names, suggestions)

            if optimize_result is None:
                print(f"[迭代 {iteration + 1}] Agent 返回 None，结束迭代")
                return result

            # 提取 Agent 的建议
            agent_must_link = getattr(optimize_result, 'must_links', []) or []
            agent_cannot_link = getattr(optimize_result, 'cannot_link', []) or []

            print(f"[迭代 {iteration + 1}] Agent 建议:")
            print(f"  - 必须链接: {len(agent_must_link)} 个约束")
            print(f"  - 必须链接: {agent_must_link}")
            print(f"  - 不能链接: {len(agent_cannot_link)} 个约束")
            print(f"  - 不能链接: {agent_cannot_link}")

            agent_must_link_list = agent_must_link
            agent_must_link = []
            for link_list in agent_must_link_list:
                for i in range(len(link_list)):
                    for j in range(i + 1, len(link_list)):
                        if link_list[i] != link_list[j]:
                            agent_must_link.append((link_list[i], link_list[j]))

            # 如果节点名称被使用，需要转换回节点索引
            if node_names:
                name_to_idx = {name: idx for idx, name in enumerate(node_names)}
                agent_must_link = [
                    (name_to_idx[m[0]], name_to_idx[m[1]])
                    for m in agent_must_link
                    if m[0] in name_to_idx and m[1] in name_to_idx
                ]
                agent_cannot_link = [
                    (name_to_idx[c[0]], name_to_idx[c[1]])
                    for c in agent_cannot_link
                    if c[0] in name_to_idx and c[1] in name_to_idx
                ]

            # 合并约束
            current_must_link, current_cannot_link = _merge_constraints(
                current_must_link,
                current_cannot_link,
                agent_must_link,
                agent_cannot_link,
            )

            # 保存 Agent 反馈
            result.agent_feedback = {
                "iteration": iteration,
                "must_links": agent_must_link,
                "cannot_link": agent_cannot_link,
            }

            print(f"[迭代 {iteration + 1}] 约束已更新，准备下一轮求解")

        except Exception as e:
            print(f"[迭代 {iteration + 1}] Agent 优化失败: {e}")
            print(f"[迭代 {iteration + 1}] 返回当前最佳结果")
            return result

    return result


async def partition_from_embeddings_iterative(
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        K: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        must_link: Optional[List[Tuple[int, int]]] = None,
        cannot_link: Optional[List[Tuple[int, int]]] = None,
        sizes: Optional[List[float]] = None,
        size_lower: Optional[List[float]] = None,
        size_upper: Optional[List[float]] = None,
        min_service_size: Optional[float] = None,
        max_service_size: Optional[float] = None,
        pair_threshold: float = 0.0,
        time_limit_sec: int = 30,
        symmetric_struc: bool = True,
        edge_weights: Optional[torch.Tensor] = None,
        max_iterations: int = 1,
        enable_agent_optimization: bool = False,
        agent_optimize_fn: Optional[Callable[[PartitionResult, str], Any]] = None,
        agent_analyze_fn: Optional[Callable[[PartitionResult], Any]] = None,
        node_names: Optional[List[str]] = None,
) -> PartitionResult:
    """
    便利包装器：从嵌入 + 边构建矩阵然后进行迭代优化求解。

    参数：
    -----------
    embeddings: [num_nodes, embedding_dim] 节点嵌入
    edge_index: [2, num_edges] 边索引
    K: 服务数量
    alpha：结构相似度内聚度的权重
    beta：语义相似度内聚度的权重
    gamma：运行时耦合惩罚的权重
    must_link：必须链接的节点对列表
    cannot_link：不能链接的节点对列表
    sizes：节点大小列表
    size_lower：每个服务的最小大小
    size_upper：每个服务的最大大小
    min_service_size：每个服务的最小节点数
    max_service_size：每个服务的最大节点数
    pair_threshold：对权重的剪枝阈值
    time_limit_sec：求解器时间限制
    symmetric_struc：是否对称化结构相似度
    edge_weights: [num_edges] 边权重（基于类型），如果提供则使用这些权重
    max_iterations: 最大迭代次数（1 表示仅约束求解）
    enable_agent_optimization: 是否启用 Agent 优化
    agent_optimize_fn: Agent 优化函数（异步）
    node_names: 节点名称列表
    """
    N = embeddings.size(0)
    S_sem = cosine_similarity(embeddings)  # [-1,1]
    S_sem = (S_sem + 1.0) / 2.0  # [0,1]
    S_struc = build_structural_similarity(N, edge_index, weight=edge_weights, symmetric=symmetric_struc)
    C_run = build_runtime_coupling(N, edge_index, weight=edge_weights)

    # 构建配置
    cfg = PartitionConfig(
        K=K,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        sizes=sizes,
        size_lower=size_lower,
        size_upper=size_upper,
        min_service_size=min_service_size,
        max_service_size=max_service_size,
        pair_threshold=pair_threshold,
        time_limit_sec=time_limit_sec,
        scale=1000,
        max_iterations=max_iterations,
        enable_agent_optimization=enable_agent_optimization,
    )

    return await iterative_optimize_partition(
        S_struc=S_struc,
        S_sem=S_sem,
        C_run=C_run,
        K=K,
        must_link=must_link,
        cannot_link=cannot_link,
        config=cfg,
        edge_index=edge_index,
        agent_optimize_fn=agent_optimize_fn,
        agent_analyze_fn = agent_analyze_fn,
        node_names=node_names,
    )


def partition_from_embeddings(
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        K: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        must_link: Optional[List[Tuple[int, int]]] = None,
        cannot_link: Optional[List[Tuple[int, int]]] = None,
        sizes: Optional[List[float]] = None,
        size_lower: Optional[List[float]] = None,
        size_upper: Optional[List[float]] = None,
        min_service_size: Optional[float] = None,
        max_service_size: Optional[float] = None,
        pair_threshold: float = 0.0,
        time_limit_sec: int = 30,
        symmetric_struc: bool = True,
        edge_weights: Optional[torch.Tensor] = None,
) -> PartitionResult:
    """
    便利包装器：从嵌入 + 边构建矩阵然后求解。
    
    参数：
    -----------
    embeddings: [num_nodes, embedding_dim] 节点嵌入
    edge_index: [2, num_edges] 边索引
    K: 服务数量
    alpha：结构相似度内聚度的权重
    beta：语义相似度内聚度的权重
    gamma：运行时耦合惩罚的权重
    must_link：必须链接的节点对列表
    cannot_link：不能链接的节点对列表
    sizes：节点大小列表
    size_lower：每个服务的最小大小
    size_upper：每个服务的最大大小
    min_service_size：每个服务的最小节点数
    max_service_size：每个服务的最大节点数
    pair_threshold：对权重的剪枝阈值
    time_limit_sec：求解器时间限制
    symmetric_struc：是否对称化结构相似度
    edge_weights: [num_edges] 边权重（基于类型），如果提供则使用这些权重
    """
    N = embeddings.size(0)
    S_sem = cosine_similarity(embeddings)  # [-1,1]
    S_sem = (S_sem + 1.0) / 2.0  # [0,1]
    S_struc = build_structural_similarity(N, edge_index, weight=edge_weights, symmetric=symmetric_struc)
    C_run = build_runtime_coupling(N, edge_index, weight=edge_weights)

    # 构建配置并求解分区
    cfg = PartitionConfig(
        K=K,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        sizes=sizes,
        size_lower=size_lower,
        size_upper=size_upper,
        min_service_size=min_service_size,
        max_service_size=max_service_size,
        pair_threshold=pair_threshold,
        time_limit_sec=time_limit_sec,
        scale=1000,
    )
    return optimize_partition(S_struc, S_sem, C_run, K, must_link, cannot_link, cfg, edge_index)
