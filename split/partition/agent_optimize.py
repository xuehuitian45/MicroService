import json
import os
from typing import List, Tuple, Dict, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel

# 初始化 agent
optimize_agent = ReActAgent(
    name="Optimizer",
    sys_prompt="""你是一个有用的软件工程师，我将提供给你一个微服务划分结果，以及一份经过专家分析得到的优化意见，请优化该划分结果，确保满足以下要求：
1. 是否存在语义上不合理的划分，是否有函数应该被拆出或合并？
2. 请给出明确的 must-link 或 cannot-link 建议。
3. must-link 建议表示节点必须放在同一个服务中，cannot-link 建议表示两个节点不能放在同一个服务中。

注意：
must-link 是节点名称的列表的列表，表示这些节点必须放在同一个服务中；
cannot-link 是节点的名称的二元组的列表，确保清晰明了。
不要只被 Ping 等测试类函数影响，要综合考虑业务功能和职责。
要让不同业务的数据处理逻辑尽可能分开，减少服务之间的耦合度。
要让不同业务尽可能被放置在不同的服务中，提升服务的内聚度，降低耦合度
""",
    model=DashScopeChatModel(
        model_name="qwen3-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),
)

analyze_agent = ReActAgent(
name="Analyzer",
    sys_prompt="""你是一个有用的软件工程师，我将提供给你一个微服务划分结果，请分析该划分结果，确保满足以下要求：
1. 该划分结果是否合理，是否还需要优化？
2. 如果需要优化的话请给出明确的优化建议。

注意：
1. 服务之间的耦合度应该尽可能低，服务内部的内聚度应该尽可能高。
2. 要让不同业务的数据处理逻辑尽可能分开，减少服务之间的耦合度。
3. 要让不同业务尽可能被放置在不同的服务中，提升服务的内聚度，降低耦合度
4. 服务在语义上应该是合理的，相关功能应该被划分到同一个服务中。
5. 不要只被 Ping 等测试类函数影响，要综合考虑业务功能和职责。
""",
    model=DashScopeChatModel(
        model_name="qwen3-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),
)

class AnalyzeResult(BaseModel):
    """Agent 分析结果的数据模型"""
    needs_optimization: bool = Field(
        default=True,
        description="是否需要优化当前划分结果")
    suggestions: Optional[str] = Field(
        default=None,
        description="Agent 的分析建议说明")


class OptimizeResult(BaseModel):
    """Agent 优化结果的数据模型"""
    must_links: List[List[str]] = Field(
        default_factory=list,
        description="必须放置在同一个服务的节点对列表，每个元素为一个二元组，表示节点名称或索引")
    cannot_link: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="不能放置在同一个服务的节点对列表，每个元素为一个二元组，表示节点名称或索引")
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent 的推理过程和建议说明")

async def agent_analyze(partitions: Dict) -> Optional[AnalyzeResult]:
    """
    使用 Agent 分析微服务划分结果。

    Args:
        partitions: 分区字典，格式为 {service_id: [node_ids]}

    Returns:
        AnalyzeResult 对象，包含是否需要优化和建议说明；
        如果分析失败或返回 None，则返回 None
    """
    try:
        print("开始调用 LLM 分析微服务划分结果，等待返回...")
        res = await analyze_agent(
            Msg(
                "user",
                f"微服务划分结果：{json.dumps(partitions, ensure_ascii=False)}。"
                "请分析这个划分，告诉我是否需要优化。如果需要优化，请给出建议。",
                "user"
            ),
            structured_model=AnalyzeResult,
        )
        print("模型已完全返回，开始处理结果...")

        if not hasattr(res, "metadata") or res.metadata is None:
            print("警告：模型返回结果中无有效 metadata 数据，返回空结果")
            return AnalyzeResult(needs_optimization=True, suggestions=None)

        try:
            analyze_result = AnalyzeResult.model_validate(res.metadata)
            print(f"Agent 分析完成：needs_optimization={analyze_result.needs_optimization}")
            if analyze_result.suggestions:
                print(f"  - 建议: {analyze_result.suggestions[:100]}...")
            return analyze_result
        except ValidationError as e:
            print(f"警告：结构化数据转换失败（模型返回格式不合法）：{e}")
            print(f"模型原始返回的 metadata：{json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
            # 返回空结果而不是抛出异常，允许分析继续进行
            return AnalyzeResult(needs_optimization=True, suggestions=None)

    except Exception as e:
        print(f"错误：Agent 分析过程中发生异常：{e}")
        import traceback
        traceback.print_exc()
        return None


async def agent_optimize(partitions: Dict, advice: str) -> Optional[OptimizeResult]:
    """
    使用 Agent 优化微服务划分结果。
    
    Args:
        partitions: 分区字典，格式为 {service_id: [node_ids]}
        advice: Agent 分析得到的优化建议说明
    
    Returns:
        OptimizeResult 对象，包含 must_links 和 cannot_link 建议；
        如果优化失败或返回 None，则返回 None
    """
    try:
        print("开始调用 LLM 优化微服务划分结果，等待返回...")
        if partitions is None or len(partitions) == 0:
            res = await optimize_agent(
                Msg(
                    "user",
                    advice,
                    "user"
                ),
                structured_model=OptimizeResult
            )
        else:
            res = await optimize_agent(
                Msg(
                    "user",
                    f"微服务划分结果：{json.dumps(partitions, ensure_ascii=False)}。专家意见：{advice}。"
                    "请分析这个划分，并给出你认为的must-link和cannot-link的结果。",
                    "user"
                ),
                structured_model=OptimizeResult,
            )
        print("模型已完全返回，开始处理结果...")

        if not hasattr(res, "metadata") or res.metadata is None:
            print("警告：模型返回结果中无有效 metadata 数据，返回空结果")
            return OptimizeResult(must_links=[], cannot_link=[])

        try:
            optimize_result = OptimizeResult.model_validate(res.metadata)
            print(f"Agent 优化完成：")
            print(f"  - must_links: {len(optimize_result.must_links)} 个")
            print(f"  - cannot_link: {len(optimize_result.cannot_link)} 个")
            if optimize_result.reasoning:
                print(f"  - 推理: {optimize_result.reasoning[:100]}...")
            return optimize_result
        except ValidationError as e:
            print(f"警告：结构化数据转换失败（模型返回格式不合法）：{e}")
            print(f"模型原始返回的 metadata：{json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
            # 返回空结果而不是抛出异常，允许优化继续进行
            return OptimizeResult(must_links=[], cannot_link=[])
    
    except Exception as e:
        print(f"错误：Agent 优化过程中发生异常：{e}")
        import traceback
        traceback.print_exc()
        return None
