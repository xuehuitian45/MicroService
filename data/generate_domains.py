import os
from typing import List
import json

import asyncio
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from pydantic import BaseModel, Field, ValidationError

cohesion_agent = ReActAgent(
    name="Evaluator",
    sys_prompt="""你是一个有用的软件工程师，我将要提供给你一个节点的信息，我希望你给我返回这个类的领域对象。注意，领域对象应该是一个名词短语，描述这个类在软件系统中的职责和作用。请确保返回的领域对象与类的功能和用途密切相关，并且能够准确反映类在系统中的角色。
例如：
类名: UserManager
领域对象: 用户管理
类名: OrderProcessor
领域对象: 订单处理
""",
    model=DashScopeChatModel(
        model_name="qwen3-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),
)

def empty_print(*args, **kwargs):
    pass

cohesion_agent.print = empty_print

class DomainResult(BaseModel):
    domains: str = Field(
        description="领域对象列表，描述该节点在软件系统中的职责和作用",)

async def domain_generator(node_name) -> str:
    try:
        res = await cohesion_agent(
            Msg(
                "user",
                f"节点名称：{node_name}\n请基于上述节点名称，提取并返回该节点的领域对象，",
                "user"
            ),
            structured_model=DomainResult,
        )

        if not hasattr(res, "metadata") or res.metadata is None:
            print("警告：模型返回结果中无有效 metadata 数据，返回空结果")
            return ""

        try:
            domain_result_instance = DomainResult.model_validate(res.metadata)
            return domain_result_instance.domains
        except ValidationError as e:
            print(f"警告：结构化数据转换失败（模型返回格式不合法）：{e}")
            print(f"模型原始返回的 metadata：{json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
            return ""

    except Exception as e:
        print(f"错误：Agent 分析过程中发生异常：{e}")
        import traceback
        traceback.print_exc()
        return ""

def main():
    data_path = "data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for node in data:
        description = node.get("description", "")
        domains = asyncio.run(domain_generator(description))
        node["domains"] = domains

    with open("data_with_domains.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
