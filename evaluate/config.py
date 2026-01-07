from dataclasses import dataclass


@dataclass
class DataConfig:
    """数据配置"""
    dataset_path: str = "../data/data.json"
    result_path: str = "../result"
    evaluate_result_path: str = "C:/Users/lenovo/Desktop/MicroService/result/report.json"

@dataclass
class EvaluateConfig:
    """评估配置"""
    repeat_times: int = 3   # 重复使用 LLM 评估的次数，避免不确定性