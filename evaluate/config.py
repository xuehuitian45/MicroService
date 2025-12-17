from dataclasses import dataclass


@dataclass
class DataConfig:
    """数据配置"""
    dataset_path: str = "/Users/xht/Downloads/MicroService/data/data.json"
    result_path: str = "/Users/xht/Downloads/MicroService/result/result.json"
    evaluate_result_path: str = "/Users/xht/Downloads/MicroService/result/report.json"

@dataclass
class EvaluateConfig:
    """评估配置"""
    repeat_times: int = 1   # 重复使用 LLM 评估的次数，避免不确定性
    llm_model: str = "qwen3-max"
