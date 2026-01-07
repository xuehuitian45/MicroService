# 将多个微服务划分结果同时输入到LLM中进行评估和比较
import asyncio
import json
from typing import List, Dict
import os

from evaluate.config import DataConfig
from evaluate.model import CompareResult
from evaluate.semantic.agent_factory import run_evaluate_agent


def load_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_comparative_evaluation_data(split_paths: List[str]) -> List[Dict]:
    splits = []
    for path in split_paths:
        split = load_json(os.path.join(path, "result.json"))
        splits.append(split)

    return splits


async def calculate_comparative_evaluation_metrics(data_path, repeat_times) -> Dict:
    split_paths = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            split_paths.append(folder_path)

    splits = prepare_comparative_evaluation_data(split_paths)
    results = []
    for i in range(repeat_times):
        print(f"Running {i+1} in {repeat_times} repeats")
        result = await run_evaluate_agent(splits)
        if isinstance(result, CompareResult):
            results.append({
                "SC": result.SC,
                "SCP": result.SCP,
                "SBC": result.SBC,
                "judge_result": result.judge_result,
            })

    # 2. 定义需要按位置求平均的三个指标
    metrics = ["SC", "SCP", "SBC"]
    final_avg_metrics = {"judge_result": results[0]["judge_result"] if results else None}

    for metric in metrics:
        metric_lists = [item[metric] for item in results]
        position_averages = [sum(elements) / len(elements) for elements in zip(*metric_lists)]
        final_avg_metrics[metric] = position_averages

    return final_avg_metrics


def main(result_path, repeat_times):
    results = asyncio.run(calculate_comparative_evaluation_metrics(result_path,repeat_times))
    return results
