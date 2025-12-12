"""
简化版分析模块（仅产出数值型指标）

功能：
- 计算并返回评估指标（结构/依赖 + 可选语义）
- 可选择保存为单一 JSON 文件
"""

from typing import Optional
import json
import os

from evaluate.evaluator import Evaluator, EvaluationMetrics
from evaluate.semantic_analyzer import SemanticAnalyzer, SemanticMetrics
from evaluate.config import EvaluationConfig


def compute_metrics(
    result_path: Optional[str] = None,
    data_path: Optional[str] = None,
    include_semantic: bool = True,
    agent=None,
) -> EvaluationMetrics:
    """
    计算并返回评估指标（仅数值）。

    Args:
        result_path: 划分结果文件路径
        data_path: 数据文件路径
        include_semantic: 是否计算语义指标
        agent: 可选的大模型 Agent，传入则用于语义分析

    Returns:
        EvaluationMetrics: 含结构/依赖与可选语义指标的完整指标对象
    """
    config = EvaluationConfig()
    if result_path:
        config.result_path = result_path
    if data_path:
        config.data_path = data_path

    # 结构/依赖指标
    evaluator = Evaluator(config)
    evaluator.load_data()
    metrics = evaluator.evaluate()

    # 语义指标（可选）
    if include_semantic:
        sem = SemanticAnalyzer(config, agent=agent)
        sem.load_data()
        sem_metrics: SemanticMetrics = sem.analyze_semantic_coherence()
        evaluator.attach_semantic_metrics(sem_metrics)
        metrics = evaluator.metrics  # 合并后的完整指标

    return metrics


def save_metrics(metrics: EvaluationMetrics, output_path: str) -> None:
    """将指标保存为 JSON 文件（仅包含顶层数值型指标）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    minimal = {
        'cohesion': round(metrics.cohesion, 4),
        'coupling': round(metrics.coupling, 4),
        'modularity': round(metrics.modularity, 4),
        'balance': round(metrics.balance, 4),
        'maintainability_index': round(metrics.maintainability_index, 2),
        'quality_score': round(metrics.quality_score, 2),
        'semantic_coherence': round(getattr(metrics, 'semantic_coherence', 0.0), 4),
        'semantic_coupling': round(getattr(metrics, 'semantic_coupling', 0.0), 4),
        'business_alignment': round(getattr(metrics, 'business_alignment', 0.0), 4),
    }
    with open(output_path, 'w') as f:
        json.dump(minimal, f, indent=2, ensure_ascii=False)
    print(f"✓ 指标已保存到: {output_path}")


if __name__ == "__main__":
    # 命令行快速生成指标（默认路径）
    cfg = EvaluationConfig()
    m = compute_metrics(
        result_path=cfg.result_path,
        data_path=cfg.data_path,
        include_semantic=True,
        agent=None,
    )
    save_metrics(m, os.path.join(os.path.dirname(cfg.result_path), 'metrics.json'))
