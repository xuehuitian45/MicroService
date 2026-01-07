import os.path

from evaluate.config import DataConfig
from evaluator import calculate_evaluation_metrics
import json

def main():
    results = calculate_evaluation_metrics()
    results = {
        key: value.__dict__ for key, value in results.items()
    }
    report_path = DataConfig().evaluate_result_path
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()