import json
import os
from typing import Optional, List, Dict

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
    sys_prompt="""你是一个有用的软件工程师，我将要提供给你两个服务的信息，我希望你能够告诉我这两个服务在业务上是否存在耦合关系。请基于服务的业务功能和职责进行判断。
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
                f"服务1：{service1}，服务2：{service2}。请基于服务的业务功能和职责，给出你对于这两个服务在业务上是否存在耦合关系的认可度评分，范围是0到1，0表示完全没有耦合关系，1表示高度耦合。",
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