"""工作流编排模块：定义 LangGraph 节点、条件路由与整体执行路径。
工作流拓扑：
    START → intent →(路由)→ direct_answer → END
                      ↘ planner →(路由)→ write(追问) → END
                                  ↘ [weather, transport, hotel, research](并行)
                                      → geo_integrator → budget → reflection →(路由)
                                                                          ↘ planner(重规划)
                                                                          ↘ write → END
"""
"""工作流编排模块：定义 LangGraph 节点、条件路由与整体执行路径。

Phase 3：用真实节点替换占位（planner/weather/transport/hotel/geo_integrator/budget/reflection/write）。
intent/direct_answer/research 暂为占位（Phase 4+ 实现）。
"""
import functools
from langgraph.graph import StateGraph, START, END
from .travelState import TravelState
from .nodes import (
    bind_agent,
    planner_node,
    weather_node,
    transport_node,
    hotel_node,
    geo_integrator_node,
    budget_node,
    reflection_node,
    write_node,
    research_node
)

def placeholder_node(state, agent=None, agent_name=""):
    """占位节点：Phase 4+ 替换为真实实现。主要是intent节点"""
    print(f"  [{agent_name}] placeholder executed")
    return {}

# 条件路由函数
def route_after_intent(state:TravelState)->str:
    """intent 之后：返回 direct_answer（闲聊）/ planner（旅行规划）。

    Phase 2 骨架：默认走 planner（intent 判断逻辑 Phase 3 实现）。
    """
    # TODO(Phase 3): 根据 state["intent"] 判断用户意图
    return "planner"

def route_after_planner(state:TravelState):
    """planner 之后：返回 write（追问）/ 并行节点列表。⭐ 关键路由

    - needs_clarification=True：信息不足，跳过并行，直接 write 追问
    - needs_clarification=False：信息完整，fan-out 到 4 个并行节点
    """
    if state["needs_clarification"]:
        return "write"
    else:
        return ["weather", "transport", "hotel", "research"]
    
def route_after_reflection(state:TravelState):
    """reflection 之后：返回 planner（重规划）/ write。

    - needs_replan=True 且未超迭代上限：回到 planner 重新规划
    - 否则：进入 write 生成最终计划
    """
    if state["needs_replan"] and state["iteration"]<state["max_iterations"]:
        return "planner"
    else:
        return "write"
    
def build_workflow(agents, checkpointer=None):
    """构建并编译旅行规划工作流。

    Args:
        agents: AgentBundle 实例（含 8 个 LCEL 链 agent）
        checkpointer: LangGraph 检查点器（支持多轮对话/断点续跑），
                      Phase 2 骨架可为 None

    Returns:
        编译后的 LangGraph 可执行图
    """
    workflow = StateGraph(TravelState)
    workflow.add_node("intent", bind_agent(placeholder_node, None, "intent"))
    workflow.add_node("direct_answer", bind_agent(placeholder_node, None, "direct_answer"))
    # 8 个业务节点：bind_agent 注入对应 agent
    workflow.add_node("planner", bind_agent(planner_node, agents.planner, "planner"))
    workflow.add_node("weather", bind_agent(weather_node, agents.weather, "weather"))
    workflow.add_node("transport", bind_agent(transport_node, agents.transport, "transport"))
    workflow.add_node("hotel", bind_agent(hotel_node, agents.hotel, "hotel"))
    workflow.add_node("research", bind_agent(research_node, agents.research, "research"))

    # geo_integrator：纯计算节点，不用 bind_agent（无 LLM）
    workflow.add_node("geo_integrator", geo_integrator_node)

    workflow.add_node("budget", bind_agent(budget_node, agents.budget, "budget"))
    workflow.add_node("reflection", bind_agent(reflection_node, agents.reflection, "reflection"))
    workflow.add_node("write", functools.partial(
        write_node,
        structured_agent=agents.writer_structured,
        fallback_agent=agents.writer,
        agent_name="writer"
    ))

    # 定义边
    workflow.add_edge(START, "intent")
    # intent →(条件)→ direct_answer → END / planner
    workflow.add_conditional_edges(
        "intent",
        route_after_intent,
        {"direct_answer": "direct_answer", "planner": "planner"},
    )
    workflow.add_edge("direct_answer", END)
    # planner →(条件)→ write(追问) → END / [weather, transport, hotel, research](并行)
    # 不提供 path_map，route_after_planner 返回节点名 list 实现 fan-out
    workflow.add_conditional_edges("planner", route_after_planner)
    # [weather, transport, hotel, research] → geo_integrator (fan-in)
    workflow.add_edge("weather", "geo_integrator")
    workflow.add_edge("transport", "geo_integrator")
    workflow.add_edge("hotel", "geo_integrator")
    workflow.add_edge("research", "geo_integrator")
    # geo_integrator → budget → reflection
    workflow.add_edge("geo_integrator", "budget")
    workflow.add_edge("budget", "reflection")
    # reflection →(条件)→ planner(重规划) / write
    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "write":"write",
            "planner":"planner"
        }
    )
    workflow.add_edge("write", END)
    app = workflow.compile(checkpointer=checkpointer)
    return app