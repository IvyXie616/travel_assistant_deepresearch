"""Agent 工厂与 AgentBundle 聚合。

设计要点：
- build_agent 用 LCEL (prompt | llm) 绑定 system prompt，返回 RunnableSequence
- 不绑定 tools（节点编排模式下，工具由节点代码显式调用）
- temperature 差异化配置（规划要稳定，写作要灵活）
- user_profile 占位符用 partial 预填充（Phase 5 后由节点注入实际画像）
"""

from dataclasses import dataclass

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.mult_agents import prompts_new
from app.mult_agents.config import AppConfig
from langchain_core.language_models import BaseChatModel
from .schemas.travel_plan import TravelPlan

@dataclass
class AgentBundle:
    """聚合所有 Agent，便于依赖注入。

    每个 agent 是 LCEL 链 (prompt | llm)，实现 Runnable 协议：
    - 调用方式统一：agent.invoke({"input": "..."})
    - 兼容 Phase 2/3 的 bind_agent + functools.partial 模式
    - 兼容 Phase 7 的 with_structured_output / stream / batch
    """
    planner: Runnable
    weather: Runnable
    transport: Runnable
    hotel: Runnable
    research: Runnable
    budget: Runnable
    reflection: Runnable
    writer: Runnable
    writer_structured: Runnable

def build_agent(
    model: str,
    api_key: str,
    prompt_key: str,
    temperature: float = 0.7,
) -> Runnable:
    """
    创建单个 Agent（LCEL 链：prompt | llm）。

    Args:
        model: 模型名（如 "qwen-turbo"）
        api_key: DashScope API Key
        prompt_key: PROMPTS 字典的 key（如 "planner"）
        temperature: 温度参数

    Returns:
        RunnableSequence：invoke 时只需传 input 字符串，
        system prompt 和 user_profile 已预填充。

    v2 修订：不传 tools 参数（节点编排模式下，工具由节点代码调用）
    """
    llm = ChatTongyi(
        model=model,
        temperature=temperature,
        dashscope_api_key=api_key,
    )

    # 构造 ChatPromptTemplate，绑定 system prompt
    # partial 预填充 user_profile（Phase 5 后由节点注入实际画像）；
    # 节点编排模式下不绑定 tools，LLM 只负责基于工具输出做结构化总结
    system_prompt = prompts_new.PROMPTS[prompt_key]
    agent_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ]).partial(user_profile="（暂无用户画像）")
    # LCEL 链：invoke 时传 {"input": "..."}，
    # {user_profile} 已由 partial 预填充（Phase 5 后由节点注入真实画像）
    return agent_template | llm

def build_agent_structured(
    model: str,
    api_key: str,
    prompt_key: str,
    temperature: float = 0.7,
) -> Runnable:
    llm = ChatTongyi(
        model=model,
        temperature=temperature,
        dashscope_api_key=api_key,
    )

    # 构造 ChatPromptTemplate，绑定 system prompt
    # partial 预填充 user_profile（Phase 5 后由节点注入实际画像）；
    # 节点编排模式下不绑定 tools，LLM 只负责基于工具输出做结构化总结
    system_prompt = prompts_new.PROMPTS[prompt_key]
    agent_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ]).partial(user_profile="（暂无用户画像）")
    # LCEL 链：invoke 时传 {"input": "..."}，
    # {user_profile} 已由 partial 预填充（Phase 5 后由节点注入真实画像）
    return agent_template | llm.with_structured_output(TravelPlan)

def build_agents(config: AppConfig) -> AgentBundle:
    """
    根据配置创建所有 Agent 并打包为 AgentBundle。

    设计要点：
    - 不同 Agent 用不同 temperature（规划要稳定，写作要灵活）
    - 不绑定工具（节点编排模式）
    """
    api = config.api_key
    model = config.model
    return AgentBundle(
        planner=build_agent(model, api, "planner", temperature=0.3),
        weather=build_agent(model, api, "weather", temperature=0.5),
        transport=build_agent(model, api, "transport", temperature=0.5),
        hotel=build_agent(model, api, "hotel", temperature=0.5),
        research=build_agent(model, api, "research", temperature=0.5),
        budget=build_agent(model, api, "budget", temperature=0.3),
        reflection=build_agent(model, api, "reflection", temperature=0.3),
        writer=build_agent(model, api, "writer", temperature=0.7),
        writer_structured=build_agent_structured(model, api, "writer", temperature=0.7),
    )
