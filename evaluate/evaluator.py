import asyncio
import math
import statistics

from evaluate.agent_evaluate import agent_cohesion, agent_coupling, agent_boundary, bge_cohesion, bge_coupling, bge_boundary
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
        self.dependencies = {item['id']: item['dependencies'] for item in self.data}
        self.name_service_map = {node: service for service in self.partitions for node in self.partitions[service]}
        # 节点描述映射（用于语义评估）
        self.node_descriptions = {item['name']: item.get('description', item['name']) for item in self.data}

    @staticmethod
    def load_data(file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def calculate_SC(self):
        """
        计算语义内聚性（Semantic Cohesion, SC）
        使用 BGE-M3 模型计算向量，通过余弦相似度计算节点对的相似度
        """
        total_count = 0
        total_SC = 0.0
        for service_nodes in self.partitions.values():
            cohesion_score = bge_cohesion(self.node_descriptions, service_nodes)
            total_SC += len(service_nodes) * cohesion_score
            total_count += len(service_nodes)
        return total_SC / total_count if total_count > 0 else 0.0

    def calculate_SCP(self):
        """
        计算服务耦合度（Semantic Coupling, SCP）
        使用 BGE-M3 模型计算向量，通过余弦相似度计算服务间节点对的相似度
        """
        services = list(self.partitions.values())
        total_count = 0
        total_SCP = 0.0
        for i in range(len(services)):
            for j in range(i + 1, len(services)):
                service_a = services[i]
                service_b = services[j]
                coupling_score = bge_coupling(self.node_descriptions, service_a, service_b)
                total_SCP += coupling_score
                total_count += 1
        return total_SCP / total_count if total_count > 0 else 0.0

    def calculate_SBC(self):
        """
        计算服务边界清晰度（Service Boundary Clarity, SBC）
        使用 BGE-M3 模型计算向量，通过服务内相似度与服务间相似度的差值来衡量边界清晰度
        """
        return bge_boundary(self.node_descriptions, self.partitions)

    def calculate_ISDD(self):
        """
        计算服务的内部服务依赖度（Internal Service Dependency Degree, ISDD）
        """
        total_ISDD = 0.0
        total_count = 0
        for service_nodes in self.partitions.values():
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}
            internal_dependencies = 0
            if len(service_node_ids) <= 1:
                total_ISDD += 1
            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is None:
                    continue
                deps = self.dependencies.get(node_id, [])
                internal_dependencies += sum(1 for dep in deps if dep in service_node_ids)
            total_ISDD += internal_dependencies / (len(service_node_ids) - 1) if len(
                service_node_ids) > 1 else 1
            total_count += len(service_nodes)

        return total_ISDD / total_count if total_count > 0 else 0.0

    def calculate_SDE(self):
        """
        计算服务依赖熵（Service Dependency Entropy, SDE）
        核心逻辑：
        1. 对每个服务，统计其调用的所有外部服务及调用次数
        2. 计算每个外部服务的调用占比（p_kl）
        3. 代入熵公式：SDE = -Σ(p_kl × log2(p_kl))
        4. 返回所有服务的平均SDE
        """
        total_sde = 0.0
        # 遍历每个服务（partition）
        for service_name, service_nodes in self.partitions.items():
            # 步骤1：获取当前服务内的节点ID集合
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}
            if len(service_node_ids) == 0:
                continue

            # 步骤2：统计当前服务对每个外部服务的调用次数
            external_call_count = {}  # key: 外部服务名称, value: 调用次数
            total_external_calls = 0  # 当前服务的总外部调用次数

            # 遍历服务内每个节点，统计对外依赖
            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is None:
                    continue
                # 获取当前节点的所有依赖节点ID
                dep_node_ids = self.dependencies.get(node_id, [])
                # 遍历依赖节点，筛选出外部节点，并找到其所属服务
                for dep_node_id in dep_node_ids:
                    if dep_node_id in service_node_ids:
                        continue  # 内部依赖，跳过
                    # 找到依赖节点对应的服务名称
                    dep_node_name = self.id_name_map.get(dep_node_id)
                    if dep_node_name is None:
                        continue
                    dep_service_name = self.name_service_map.get(dep_node_name)
                    if dep_service_name is None:
                        continue

                    # 累计对该外部服务的调用次数
                    external_call_count[dep_service_name] = external_call_count.get(dep_service_name, 0) + 1
                    total_external_calls += 1

            # 步骤3：计算当前服务的SDE（信息熵）
            service_sde = 0.0
            if total_external_calls > 0:
                for call_count in external_call_count.values():
                    # 计算调用占比 p_kl
                    p_kl = call_count / total_external_calls
                    # 熵公式：-p×log2(p)（p=0时该项为0，无需处理）
                    service_sde -= p_kl * math.log2(p_kl) if p_kl > 0 else 0.0

            # 累加当前服务的SDE
            total_sde += service_sde * len(service_nodes)

        # 步骤4：返回所有服务的平均SDE
        num_services = sum([len(nodes) for nodes in self.partitions.values()])
        return total_sde / num_services if num_services > 0 else 0.0

    def calculate_SSB(self):
        """
        计算服务规模平衡性（Service Size Balance, SSB）
        公式：SSB = 标准差(服务节点数) / 均值(服务节点数)
        """
        # 统计每个服务的节点数
        service_sizes = [len(nodes) for nodes in self.partitions.values()]
        if len(service_sizes) <= 1:
            return 0.0  # 只有1个服务时无失衡问题

        # 计算均值和标准差
        mean_size = statistics.mean(service_sizes)
        std_size = statistics.stdev(service_sizes) if len(service_sizes) > 1 else 0.0

        # 计算SSB（避免除零错误）
        ssb = std_size / mean_size if mean_size > 0 else 0.0
        return ssb

    def calculate_Nano_Mega_Ratio(self):
        """
        计算Nano/Mega服务占比
        Nano：节点数 < 5；Mega：节点数 > 50
        返回：(R_nano, R_mega)
        """
        total_services = len(self.partitions)
        if total_services == 0:
            return 0.0, 0.0

        nano_count = 0
        mega_count = 0

        for service_nodes in self.partitions.values():
            size = len(service_nodes)
            if size < 5:
                nano_count += 1
            elif size > 50:
                mega_count += 1

        # 计算占比
        r_nano = nano_count / total_services
        r_mega = mega_count / total_services
        return r_nano, r_mega

    def calculate_SII(self):
        """
        计算结构不稳定性指数（Structural Instability Index, SII）
        公式：SII(S_k) = Fan_out / (Fan_in + Fan_out)
        Fan_out：服务对外调用数；Fan_in：服务被外部调用数
        返回：所有服务的平均SII
        """
        total_sii = 0.0
        num_services = 0

        # 预计算每个服务的Fan_in和Fan_out
        service_fan_out = {s: 0 for s in self.partitions.keys()}
        service_fan_in = {s: 0 for s in self.partitions.keys()}

        # 1. 计算Fan_out（服务对外调用数）
        for service_name, service_nodes in self.partitions.items():
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}
            fan_out = 0
            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is None:
                    continue
                dep_node_ids = self.dependencies.get(node_id, [])
                for dep_node_id in dep_node_ids:
                    if dep_node_id not in service_node_ids:
                        dep_node_name = self.id_name_map.get(dep_node_id)
                        if dep_node_name:
                            dep_service = self.name_service_map.get(dep_node_name)
                            if dep_service:
                                fan_out += 1
            service_fan_out[service_name] = fan_out

        # 2. 计算Fan_in（服务被外部调用数）
        for service_name, service_nodes in self.partitions.items():
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}
            fan_in = 0
            # 遍历所有节点，统计调用当前服务的外部节点数
            for node_id, dep_node_ids in self.dependencies.items():
                node_name = self.id_name_map.get(node_id)
                if not node_name:
                    continue
                caller_service = self.name_service_map.get(node_name)
                if caller_service == service_name:
                    continue  # 内部调用，跳过
                # 检查是否调用当前服务的节点
                for dep_node_id in dep_node_ids:
                    if dep_node_id in service_node_ids:
                        fan_in += 1
            service_fan_in[service_name] = fan_in

        # 3. 计算每个服务的SII
        for service_name in self.partitions.keys():
            fan_in = service_fan_in[service_name]
            fan_out = service_fan_out[service_name]
            if fan_in + fan_out == 0:
                sii = 0.0
            else:
                sii = fan_out / (fan_in + fan_out) * len(self.partitions[service_name])
            total_sii += sii
            num_services += len(self.partitions[service_name])

        return total_sii / num_services if num_services > 0 else 0.0

    def calculate_modularity(self):
        """
        计算模块度（Modularity）
        将依赖图视为无向图，基于 Newman-Girvan 模块度定义：
        Q = Σ_k [ (l_k / m) - (d_k / (2m))^2 ]
        其中：
            - m：图中无向边的数量
            - l_k：社区（服务）k 内部的边数
            - d_k：社区（服务）k 中所有节点度数之和
        """
        # 构建 id -> service_name 映射
        id_service_map = {}
        for service_name, service_nodes in self.partitions.items():
            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is not None:
                    id_service_map[node_id] = service_name

        # 统计总边数 m、每个节点度数以及每个服务内部边数
        m = 0  # 无向边数量（以依赖关系作无向边）
        node_degree = {node_id: 0 for node_id in self.dependencies.keys()}
        service_internal_edges = {service_name: 0 for service_name in self.partitions.keys()}

        for src_id, dep_ids in self.dependencies.items():
            for dst_id in dep_ids:
                m += 1
                # 将依赖视为无向边，出入度都 +1
                if src_id not in node_degree:
                    node_degree[src_id] = 0
                if dst_id not in node_degree:
                    node_degree[dst_id] = 0
                node_degree[src_id] += 1
                node_degree[dst_id] += 1

                src_service = id_service_map.get(src_id)
                dst_service = id_service_map.get(dst_id)
                if src_service is not None and src_service == dst_service:
                    service_internal_edges[src_service] = service_internal_edges.get(src_service, 0) + 1

        if m == 0:
            return 0.0

        # 计算每个服务的 d_k（社区内节点度数之和）
        service_degree_sum = {service_name: 0 for service_name in self.partitions.keys()}
        for node_id, degree in node_degree.items():
            service_name = id_service_map.get(node_id)
            if service_name is not None:
                service_degree_sum[service_name] = service_degree_sum.get(service_name, 0) + degree

        modularity = 0.0
        for service_name in self.partitions.keys():
            l_k = service_internal_edges.get(service_name, 0)
            d_k = service_degree_sum.get(service_name, 0)
            modularity += (l_k / m) - (d_k / (2 * m)) ** 2

        return modularity

    def calculate_service_cycle_ratio(self):
        """
        计算服务级循环依赖比例（Service Cyclic Dependency Ratio）
        步骤：
        1. 构建服务之间的有向依赖图（节点为服务，边为服务间调用）
        2. 使用 Tarjan 算法求强连通分量（SCC）
        3. 所有大小 > 1 的 SCC，以及存在自环的单节点 SCC，都视为存在循环依赖
        4. 返回参与循环依赖的服务数 / 总服务数
        """
        services = list(self.partitions.keys())
        num_services = len(services)
        if num_services == 0:
            return 0.0

        # 1. 构建服务依赖图
        service_graph = {s: set() for s in services}

        for service_name, service_nodes in self.partitions.items():
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}
            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is None:
                    continue
                dep_node_ids = self.dependencies.get(node_id, [])
                for dep_node_id in dep_node_ids:
                    if dep_node_id in service_node_ids:
                        continue  # 内部调用不计入服务间依赖
                    dep_node_name = self.id_name_map.get(dep_node_id)
                    if not dep_node_name:
                        continue
                    dep_service = self.name_service_map.get(dep_node_name)
                    if dep_service and dep_service != service_name:
                        service_graph[service_name].add(dep_service)

        # 2. Tarjan 算法求 SCC
        index = 0
        indices = {}
        lowlink = {}
        stack = []
        on_stack = set()
        sccs = []

        def strongconnect(v):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in service_graph[v]:
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            # 如果 v 是一个 SCC 的根节点
            if lowlink[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in services:
            if v not in indices:
                strongconnect(v)

        # 3. 统计参与循环依赖的服务
        cyclic_services = set()
        for scc in sccs:
            if len(scc) > 1:
                cyclic_services.update(scc)
            else:
                # 单节点 SCC，如果存在自环也认为有循环依赖
                v = scc[0]
                if v in service_graph[v]:
                    cyclic_services.add(v)

        return len(cyclic_services) / num_services if num_services > 0 else 0.0

    def calculate_ICP(self):
        """
        计算内部调用占比（Internal Call Proportion, ICP）
        公式：ICP = 内部调用数 / 总调用数
        返回：所有服务的平均ICP
        """
        internal_calls = 0
        total_calls = 0

        for service_name, service_nodes in self.partitions.items():
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}


            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is None:
                    continue
                dep_node_ids = self.dependencies.get(node_id, [])
                for dep_node_id in dep_node_ids:
                    total_calls += 1
                    if dep_node_id in service_node_ids:
                        internal_calls += 1

        return internal_calls / total_calls if total_calls > 0 else 0.0

    def evaluate(self):
        """
        执行评估，返回评估报告
        """
        r_nano, r_mega = self.calculate_Nano_Mega_Ratio()

        report = {
            "语义内聚性（SC）": self.calculate_SC(),
            "服务耦合度（SCP）": self.calculate_SCP(),
            "服务边界清晰度（SBC）": self.calculate_SBC(),
            "内部服务依赖密度（ISDD）": self.calculate_ISDD(),
            "服务依赖熵（SDE）": self.calculate_SDE(),
            "服务规模平衡性（SSB）": self.calculate_SSB(),
            "Nano服务占比（R_nano）": r_nano,
            "Mega服务占比（R_mega）": r_mega,
            "结构不稳定性指数（SII）": self.calculate_SII(),
            "内部调用占比（ICP）": self.calculate_ICP(),
            "模块度（Modularity）": self.calculate_modularity(),
            "服务循环依赖比例（SCDR）": self.calculate_service_cycle_ratio()
        }
        return report


def main():
    data_config = DataConfig()
    evaluate_config = EvaluateConfig()
    evaluator = Evaluator(data_config, evaluate_config)

    evaluation_report = evaluator.evaluate()
    print("微服务划分评估报告：")
    for metric, score in evaluation_report.items():
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()
