"""
只计算并输出评价指标的数值，不做其它分析。
"""

import os
from evaluate.config import EvaluationConfig
from evaluate.analyzer import compute_metrics, save_metrics


def main():
    cfg = EvaluationConfig()
    output_dir = "/Users/xht/Downloads/MicroService/result"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "report.json")

    # 计算指标：包含结构/依赖与语义三个指标
    metrics = compute_metrics(
        result_path=cfg.result_path,
        data_path=cfg.data_path,
        include_semantic=True,
        agent=None,
    )

    save_metrics(metrics, output_path)

    # 控制台简要提示
    print("\n生成完成：")
    print(f"  指标文件: {output_path}")


if __name__ == "__main__":
    main()
