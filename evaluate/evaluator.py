import asyncio

from evaluate.agent_evaluate import agent_cohesion, agent_coupling, agent_boundary
from evaluate.config import DataConfig, EvaluateConfig
import json


class Evaluator:
    def __init__(self, data_config: DataConfig, evaluate_config: EvaluateConfig):
        self.data_config = data_config
        self.evaluate_config = evaluate_config
        self.data = self.load_data(self.data_config.dataset_path)
        self.partitions = self.load_data(self.data_config.result_path)
        self.name_id_map = {item['name']: item['id'] for item in self.data}
        self.id_name_map = {item['id']: item['name'] for item in self.data}
        self.name_service_map = {node: service for service in self.partitions for node in self.partitions[service]}
        self.thread_num = evaluate_config.thread_num if hasattr(evaluate_config, 'thread_num') else 5

    @staticmethod
    def load_data(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    async def calculate_SC(self):
        """
        多线程计算语义内聚性（Semantic Cohesion, SC）
        核心优化：用线程池并行计算所有节点对的相似度，替代串行双层循环
        """
        total_count = 0
        total_SC = 0.0
        for service in self.partitions.values():
            total_SC += len(service) * (await agent_cohesion(service)).score
            total_count += len(service)
        return total_SC / total_count if total_count > 0 else 0.0

    async def calculate_SCP(self):
        services = list(self.partitions.values())
        total_count = 0
        total_SCP = 0.0
        for i in range(len(services)):
            for j in range(i + 1, len(services)):
                service_a = services[i]
                service_b = services[j]
                total_SCP += len(service_a) * (await agent_coupling(service_a, service_b)).score
                total_count += 1
        return total_SCP / total_count if total_count > 0 else 0.0

    async def calculate_SBC(self):
        """
        计算服务边界清晰度（Service Boundary Clarity, SBC）
        """
        return (await agent_boundary(self.partitions)).score


    async def evaluate(self):
        """
        执行评估，返回评估报告
        """
        report = {
            # "语义内聚性（SC）": await self.calculate_SC(),
            "服务耦合度（SCP）": await self.calculate_SCP(),
            # "服务边界清晰度（SBC）": await self.calculate_SBC(),
        }
        return report


async def main():
    data_config = DataConfig()
    evaluate_config = EvaluateConfig()
    evaluator = Evaluator(data_config, evaluate_config)

    evaluation_report = await evaluator.evaluate()
    print("微服务划分评估报告：")
    for metric, score in evaluation_report.items():
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
