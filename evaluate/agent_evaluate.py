import json
import os
from typing import Optional, List, Dict
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from pydantic import BaseModel, Field, ValidationError

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel

# 初始化 agent
cohesion_agent = ReActAgent(
    name="Evaluator",
    sys_prompt="""你是一个有用的软件工程师，我将要提供给你一个服务的额信息，我希望你告诉我这个服务在语义内聚度上表现如何，请根据服务的业务功能和职责进行判断，
    给出你对于该服务语义内聚度的评分，范围是0到1，0表示完全不内聚，1表示高度内聚。在回答的时候不需要给出理由，直接给出评分即可.
""",
    model=DashScopeChatModel(
        model_name="qwen3-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),
)


coupling_agent = ReActAgent(
    name="CouplingEvaluator",
    sys_prompt="""你是一个有用的软件工程师，我将要提供给你两个服务的信息，我希望你能够告诉我这两个服务微服务耦合度上状况。请基于服务的业务功能和职责进行判断。
    给出你对于这两个服务在业务上是否存在耦合关系的认可度评分，范围是0到1，0表示完全没有耦合关系，1表示高度耦合。在回答的时候不需要给出理由，直接给出评分即可.
""",
    model=DashScopeChatModel(
        model_name="qwen3-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),
)

boundary_agent = ReActAgent(
    name="BoundaryEvaluator",
    sys_prompt="""你是一个有用的软件工程师，我将要提供给你一个微服务划分的结果，我希望你能够分析该划分结果的语义边界清晰度，并给出评分，范围是0到1，0表示边界非常模糊，1表示边界非常清晰。在回答的时候不需要给出理由，直接给出评分即可.
""",
    model=DashScopeChatModel(
        model_name="qwen3-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),
)

class EvaluateResult(BaseModel):
    score: float = Field(
        default=0.5,
        description="评分指标，范围为0到1")


# BGE-M3 模型全局变量（延迟加载）
_bge_tokenizer = None
_bge_model = None
_bge_device = None


def _get_bge_model():
    """获取或初始化 BGE-M3 模型（单例模式）"""
    global _bge_tokenizer, _bge_model, _bge_device
    
    if _bge_model is None:
        model_name = "BAAI/bge-m3"
        _bge_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在加载 BGE-M3 模型到 {_bge_device}...")
        _bge_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _bge_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _bge_model.to(_bge_device)
        _bge_model.eval()
        print("BGE-M3 模型加载完成")
    
    return _bge_tokenizer, _bge_model, _bge_device


def compute_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    使用 BGE-M3 模型计算文本向量
    
    Args:
        texts: 文本列表
        batch_size: 批处理大小
    
    Returns:
        [num_texts, embedding_dim] 的 numpy 数组
    """
    tokenizer, model, device = _get_bge_model()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 分词和编码
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 移到设备
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # 获取编码
            model_output = model(**encoded_input)
            
            # BGE-M3 使用 mean pooling
            if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
                embeddings = model_output.pooler_output
            else:
                embeddings = model_output.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            
            # 归一化（余弦相似度需要归一化向量）
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def compute_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """
    计算两组向量之间的平均余弦相似度
    
    Args:
        embeddings1: [n1, dim] 的向量数组
        embeddings2: [n2, dim] 的向量数组
    
    Returns:
        平均余弦相似度
    """
    if len(embeddings1) == 0 or len(embeddings2) == 0:
        return 0.0
    
    # 计算所有向量对之间的余弦相似度
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    
    # 返回平均值
    return float(np.mean(similarity_matrix))


async def agent_cohesion(service: List[str]) -> EvaluateResult:
    try:
        res = await cohesion_agent(
            Msg(
                "user",
                f"服务信息：{service}。请基于服务的业务功能和职责，给出你对于该服务语义内聚度的评分，范围是0到1，0表示完全不内聚，1表示高度内聚。",
                "user"
            ),
            structured_model=EvaluateResult,
        )

        if not hasattr(res, "metadata") or res.metadata is None:
            print("警告：模型返回结果中无有效 metadata 数据，返回空结果")
            return EvaluateResult(score=0.5)

        try:
            evaluate_result = EvaluateResult.model_validate(res.metadata)
            return evaluate_result
        except ValidationError as e:
            print(f"警告：结构化数据转换失败（模型返回格式不合法）：{e}")
            print(f"模型原始返回的 metadata：{json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
            return EvaluateResult(score=0.5)

    except Exception as e:
        print(f"错误：Agent 分析过程中发生异常：{e}")
        import traceback
        traceback.print_exc()
        return EvaluateResult(score=0.5)


async def agent_coupling(service1: List[str], service2: List[str]) -> EvaluateResult:
    try:
        res = await coupling_agent(
            Msg(
                "user",
                f"服务1：{service1}，服务2：{service2}。请基于服务的业务功能和职责，给出你对于这两个服务在微服务耦合度上的认可度评分，范围是0到1，0表示完全没有耦合关系，1表示高度耦合。",
                "user"
            ),
            structured_model=EvaluateResult,
        )

        if not hasattr(res, "metadata") or res.metadata is None:
            print("警告：模型返回结果中无有效 metadata 数据，返回空结果")
            return EvaluateResult(score=0.5)

        try:
            evaluate_result = EvaluateResult.model_validate(res.metadata)
            return evaluate_result
        except ValidationError as e:
            print(f"警告：结构化数据转换失败（模型返回格式不合法）：{e}")
            print(f"模型原始返回的 metadata：{json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
            return EvaluateResult(score=0.5)

    except Exception as e:
        print(f"错误：Agent 分析过程中发生异常：{e}")
        import traceback
        traceback.print_exc()
        return EvaluateResult(score=0.5)


async def agent_boundary(partitions: Dict[str, List[str]]) -> EvaluateResult:
    try:
        res = await boundary_agent(
            Msg(
                "user",
                f"微服务划分结果：{json.dumps(partitions, ensure_ascii=False)}。请分析该划分结果的语义边界清晰度，并给出评分，范围是0到1，0表示边界非常模糊，1表示边界非常清晰。",
                "user"
            ),
            structured_model=EvaluateResult,
        )

        if not hasattr(res, "metadata") or res.metadata is None:
            print("警告：模型返回结果中无有效 metadata 数据，返回空结果")
            return EvaluateResult(score=0.5)

        try:
            evaluate_result = EvaluateResult.model_validate(res.metadata)
            return evaluate_result
        except ValidationError as e:
            print(f"警告：结构化数据转换失败（模型返回格式不合法）：{e}")
            print(f"模型原始返回的 metadata：{json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
            return EvaluateResult(score=0.5)

    except Exception as e:
        print(f"错误：Agent 分析过程中发生异常：{e}")
        import traceback
        traceback.print_exc()
        return EvaluateResult(score=0.5)


# ========== 基于 BGE-M3 的语义评估函数 ==========

def bge_cohesion(node_descriptions: Dict[str, str], service_nodes: List[str]) -> float:
    """
    使用 BGE-M3 模型计算服务内语义内聚性（Semantic Cohesion）
    
    Args:
        node_descriptions: 节点名称到描述的映射
        service_nodes: 服务内的节点名称列表
    
    Returns:
        语义内聚性分数（0-1之间）
    """
    if len(service_nodes) <= 1:
        return 1.0  # 单个节点或空服务，内聚性为1
    
    # 获取服务内节点的描述
    texts = []
    for node in service_nodes:
        desc = node_descriptions.get(node, node)  # 如果没有描述，使用节点名称
        texts.append(desc)
    
    # 计算向量
    embeddings = compute_embeddings(texts)
    
    # 计算服务内所有节点对的平均余弦相似度
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = float(np.dot(embeddings[i], embeddings[j]))  # 已归一化，直接点积
            similarities.append(sim)
    
    # 将相似度从 [-1, 1] 映射到 [0, 1]
    avg_similarity = np.mean(similarities) if similarities else 0.0
    cohesion_score = (avg_similarity + 1.0) / 2.0  # 归一化到 [0, 1]
    
    return float(cohesion_score)


def bge_coupling(node_descriptions: Dict[str, str], service1_nodes: List[str], service2_nodes: List[str]) -> float:
    """
    使用 BGE-M3 模型计算两个服务之间的语义耦合度（Semantic Coupling）
    
    Args:
        node_descriptions: 节点名称到描述的映射
        service1_nodes: 服务1的节点名称列表
        service2_nodes: 服务2的节点名称列表
    
    Returns:
        语义耦合度分数（0-1之间）
    """
    if len(service1_nodes) == 0 or len(service2_nodes) == 0:
        return 0.0
    
    # 获取两个服务的节点描述
    texts1 = [node_descriptions.get(node, node) for node in service1_nodes]
    texts2 = [node_descriptions.get(node, node) for node in service2_nodes]
    
    # 计算向量
    embeddings1 = compute_embeddings(texts1)
    embeddings2 = compute_embeddings(texts2)
    
    # 计算服务间节点对的平均余弦相似度
    coupling_score = compute_cosine_similarity(embeddings1, embeddings2)
    
    # 将相似度从 [-1, 1] 映射到 [0, 1]
    coupling_score = (coupling_score + 1.0) / 2.0
    
    return float(coupling_score)


def bge_boundary(node_descriptions: Dict[str, str], partitions: Dict[str, List[str]]) -> float:
    """
    使用 BGE-M3 模型计算服务边界清晰度（Service Boundary Clarity）
    
    边界清晰度 = 平均(服务内相似度) - 平均(服务间相似度)
    值越大，边界越清晰
    
    Args:
        node_descriptions: 节点名称到描述的映射
        partitions: 服务划分结果，{服务名: [节点列表]}
    
    Returns:
        服务边界清晰度分数（归一化到 0-1 之间）
    """
    services = list(partitions.values())
    if len(services) <= 1:
        return 1.0  # 只有一个服务或没有服务，边界清晰度为1
    
    # 计算所有服务内的平均相似度
    intra_similarities = []
    for service_nodes in services:
        if len(service_nodes) > 1:
            cohesion = bge_cohesion(node_descriptions, service_nodes)
            intra_similarities.append(cohesion)
    
    avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0.0
    
    # 计算所有服务间的平均相似度
    inter_similarities = []
    for i in range(len(services)):
        for j in range(i + 1, len(services)):
            coupling = bge_coupling(node_descriptions, services[i], services[j])
            inter_similarities.append(coupling)
    
    avg_inter_similarity = np.mean(inter_similarities) if inter_similarities else 0.0
    
    # 边界清晰度 = 服务内相似度 - 服务间相似度
    # 值域可能在 [-1, 1]，需要归一化到 [0, 1]
    boundary_score = avg_intra_similarity - avg_inter_similarity
    boundary_score = (boundary_score + 1.0) / 2.0  # 归一化到 [0, 1]
    
    return float(boundary_score)