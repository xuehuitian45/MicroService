import json
import os
from typing import List, Dict

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit
from pydantic import ValidationError

from evaluate.model import CompareResult


def create_agent(agent_name: str, system_prompt: str, toolkit: Toolkit) -> ReActAgent:
    """根据给定信息创建智能体对象。"""
    if system_prompt is None or system_prompt.strip() == "":
        system_prompt = f"你是{agent_name}，一个专业的软件工程师。"

    agent = ReActAgent(
        name=agent_name,
        sys_prompt=system_prompt,
        model=DashScopeChatModel(
            model_name="qwen3-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
        ),
        toolkit=toolkit,
        formatter=DashScopeMultiAgentFormatter(),
    )

    async def empty_print(*args, **kwargs):
        pass

    agent.print = empty_print
    return agent


async def run_evaluate_agent(splits: List[Dict]) -> CompareResult | None:
    evaluate_agent = create_agent("evaluate_agent",
                                  "你是一个软件工程师，我会给你几个针对相同的单体系统的不同微服务划分结果，我希望你能够对比分析这几个微服务划分结果的合理性，并分别从语义一致性、语义耦合性、服务边界清晰度三个方面对这几个划分结果进行评分，并分别给出你的理由",
                                  toolkit=Toolkit())

    prompt = f"请你对以下微服务划分结果进行对比分析："
    for idx, split in enumerate(splits):
        prompt += f"\n划分结果 {idx+1}：{json.dumps(split, indent=4, ensure_ascii=False)}"

    res = await evaluate_agent(Msg(
        "user",
        prompt,
        "user",
    ),
        structured_model=CompareResult
    )

    if not hasattr(res, "metadata") or res.metadata is None:
        print("警告：模型返回结果中无有效 metadata 数据，返回空结果")
        return None
    try:
        analyze_result = CompareResult.model_validate(res.metadata)
        return analyze_result

    except ValidationError as e:
        print(f"警告：结构化数据转换失败（模型返回格式不合法）：{e}")
        print(f"模型原始返回的 metadata：{json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
        return None
