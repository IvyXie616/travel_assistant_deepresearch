"""节点执行模块：实现 planner、weather、transport、hotel、geo_integrator 等节点逻辑。

节点编排模式（核心）：
    state → 构造 prompt → LCEL 链 invoke → 解析 JSON → 写 state

Phase 3 Step 3：辅助函数 + planner_node
后续 Step 4-5 补充 weather/transport/hotel/geo_integrator 节点。
"""
import ast
import json
import logging
import re
from functools import partial
from langchain_core.messages import AIMessage
from .travelState import TravelState

import os
import sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.append(root)
logger = logging.getLogger("travel_agents")
from weather_server import get_weather_data
from route_server import search_routes_data
from hotel_server import search_hotels_data
from .tools import geo_distance_tool, search_knowledge_base_records

from typing import Optional
from .memory.manager import MemoryManager
# 模块级全局 MemoryManager（main.py 启动时注入）
_MEMORY_MANAGER: Optional[MemoryManager] = None

def init_memory_manager(mgr: MemoryManager):
    """初始化全局 MemoryManager。main.py 启动时调用一次。"""
    global _MEMORY_MANAGER
    _MEMORY_MANAGER = mgr

# 辅助函数
def bind_agent(node_func, agent, agent_name: str):
    """将 agent 注入节点函数（依赖注入）。

    用 functools.partial 把 agent 和 agent_name 固定到节点函数参数上，
    LangGraph 调用节点时只需传 state，agent 已预先绑定。
    """
    return partial(node_func, agent=agent, agent_name=agent_name)

def with_memory_context(state:TravelState, user_prompt: str) -> str:
    """拼接跨会话记忆到 prompt（无记忆时原样返回）。"""
    # 1. 优先用全局 MemoryManager 构建个性化上下文
    if _MEMORY_MANAGER is not None:
        user_id = state.get("user_id", "default")
        thread_id = state.get("thread_id", "default")
        query = state.get("query", "")
        try:
            context = _MEMORY_MANAGER.build_personalized_prompt_context(
                user_id, thread_id, query
            )
            if context:
                return f"{user_prompt}\n\n[记忆上下文]\n{context}"
        except Exception as e:
            logger.warning("记忆上下文构建失败: %s", e)

    # 2. 回退：用 state 里的 memory_context（兼容旧逻辑）
    memory_context = state.get("memory_context", "").strip()
    if not memory_context:
        return user_prompt
    return f"{user_prompt}\n\n[跨会话记忆]\n{memory_context}"

def extract_json_block(text: str) -> str:
    """从 LLM 输出中提取 JSON 字符串（处理 json 包裹和首尾杂字符）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned

def load_json(json_str, fallback):
    """解析 JSON，失败时返回 fallback。"""
    cleaned = extract_json_block(json_str)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        # LLM 可能输出 Python 风格的 True/False/None，用 ast.literal_eval 兜底
        return ast.literal_eval(cleaned)
    except Exception:
        return fallback
    
def invoke_json_agent(state:TravelState, prompt: str, agent, agent_name: str, node: str, fallback):
    """调用 LCEL 链 agent，返回 (解析后的dict, content字符串, messages列表)。

    ⭐ LCEL 链适配（ReAct agent）：
    - 调用：agent.invoke({"input": ...}) 而非 {"messages": [human]}
    - 返回：result.content（AIMessage）而非 result["messages"][-1].content
    """
    memo_prompt = with_memory_context(state, prompt)
    result = agent.invoke({"input": memo_prompt})
    content = result.content
    logger.info("[%s] LLM调用: 是 | 输出长度: %d", node, len(content))
    return load_json(content, fallback), content, [AIMessage(content=content, name=agent_name)]

# planner 节点
def planner_node(state:TravelState, agent, agent_name: str):
    """planner 节点：解析用户需求，输出结构化旅行计划 JSON。

    调用 planner LCEL 链，解析返回的 JSON，写入 state 的：
    plan / sub_tasks / origin / destination / travel_dates /
    needs_clarification / clarification_question
    """
    prompt = state.get("query", "")
    data, content, msgs = invoke_json_agent(state, prompt, agent, agent_name, "planner", {})
    result = {
        "plan": content,
        "sub_tasks": data.get("sub_tasks", []),
        "origin": data.get("origin", ""),
        "destination": data.get("destination", ""),
        "travel_dates": data.get("travel_dates", []),
        "needs_clarification": data.get("needs_clarification", False),
        "clarification_question": data.get("clarification_question", ""),
        "messages": msgs
    }
    # 持久化用户 query（planner 是第一个真实节点）
    if _MEMORY_MANAGER is not None:
        try:
            _MEMORY_MANAGER.persist_turn(
                user_id=state.get("user_id", "default"),
                thread_id=state.get("thread_id", "default"),
                query=state.get("query", ""),
                answer=content[:500],
            )
        except Exception as e:
            logger.warning("persist_turn 失败: %s", e)
    return result

"""
1. 参数提取    ← 从 state 取 planner 写入的 origin/destination/travel_dates
2. 工具调用    ← 直接调 server 核心函数（_get_weather_data 等），拿到结构化数据
3. LLM 总结    ← 构造 prompt（含工具输出）→ LCEL 链 invoke → 写 state
"""
def weather_node(state:TravelState, agent, agent_name: str):
    """weather 节点：查询目的地天气，LLM 总结出行建议。

    三段式：destination + travel_dates → get_weather_data → LLM 总结
    """
    destination = state.get("destination", "")
    travel_dates = state.get("travel_dates", [])
    if not destination or not travel_dates:
        return {"weather_info": "无天气数据（缺少目的地或日期）", "messages": []}
    
    weather_list = [get_weather_data(destination, date) for date in travel_dates]
    prompt = f"以下是{destination}在{travel_dates}这几天的天气信息：\n {weather_list}"

    data, content, msgs = invoke_json_agent(state, prompt, agent, agent_name, "weather", {})
    return {
        "weather_info": content,
        "messages": msgs
    }

def transport_node(state, agent, agent_name: str):
    """transport 节点：查询城际交通，LLM 总结推荐。

    三段式：origin + destination → _search_routes_data → LLM 总结
    """
    # 1. 参数提取
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    dates = state.get("travel_dates", [])
    date = dates[0] if dates else ""

    if not origin or not destination:
        return {"transport_info": "无交通数据（缺少出发地或目的地）", "messages": []}

    # 2. 工具调用
    routes = search_routes_data(origin, destination, date)

    # 3. LLM 总结
    prompt = (
        f"以下是{origin}到{destination}在{date}的交通路线：\n"
        f"{json.dumps(routes, ensure_ascii=False, indent=2)}\n"
    )
    data, content, msgs = invoke_json_agent(
        state, prompt, agent, agent_name, "transport", {}
    )
    return {"transport_info": content, "messages": msgs}

def hotel_node(state, agent, agent_name: str):
    """hotel 节点：查询目的地酒店，LLM 总结推荐。

    三段式：destination → _search_hotels_data → LLM 总结
    """
    # 1. 参数提取
    destination = state.get("destination", "")
    dates = state.get("travel_dates", [])
    date = dates[0] if dates else ""

    if not destination:
        return {"hotel_info": "无酒店数据（缺少目的地）", "messages": []}

    # 2. 工具调用
    hotels = search_hotels_data(destination, date)

    # 3. LLM 总结
    prompt = (
        f"以下是{destination}的酒店列表：\n"
        f"{json.dumps(hotels, ensure_ascii=False, indent=2)}\n"
    )
    data, content, msgs = invoke_json_agent(
        state, prompt, agent, agent_name, "hotel", {}
    )
    return {"hotel_info": content, "messages": msgs}

# ============================================================
# geo_integrator 节点（纯计算，无 LLM）
# ============================================================

def geo_integrator_node(state:TravelState):
    """地理整合节点：计算车站/酒店之间的距离矩阵（纯计算，无 LLM）。

    从 transport/hotel 的工具数据提取经纬度，
    调用 geo_distance_tool 计算两两距离，返回 geo_matrix。
    Phase 4 补充：车站→景点距离。
    """
    # 1. 参数提取
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    dates = state.get("travel_dates", [])
    date = dates[0] if dates else ""

    # 2. 获取原始数据（mock 纯内存，重新调用开销可忽略）
    routes = search_routes_data(origin, destination, date)
    hotels = search_hotels_data(destination, date)

    geo_matrix = {}

    # 3. 提取所有不同的目的地车站（同名车站自动去重）
    stations = {}
    for route in routes:
        station_name = route.get("dest_station", "")
        station_latlng = route.get("dest_latlng")
        if station_name and station_latlng:
            stations[station_name] = station_latlng

    # 4. 计算每个车站 → 每个酒店 的距离矩阵（全组合）
    for station_name, station_latlng in stations.items():
        for hotel in hotels:
            hotel_latlng = hotel.get("latlng", [0, 0])
            hotel_name = hotel.get("name", "")
            distance = geo_distance_tool.func(station_latlng, hotel_latlng)
            geo_matrix[f"{station_name}→{hotel_name}"] = distance

    return {"geo_matrix": geo_matrix}

# ============================================================
# budget / reflection / write 节点
# ============================================================
def budget_node(state:TravelState, agent, agent_name: str):
    """budget 节点：综合所有信息估算旅行预算。"""
    prompt = (
        f"请根据以下信息估算旅行预算：\n"
        f"目的地：{state.get('destination', '')}\n"
        f"出行日期：{state.get('travel_dates', [])}\n"
        f"天气信息：{state.get('weather_info', '')}\n"
        f"交通信息：{state.get('transport_info', '')}\n"
        f"酒店信息：{state.get('hotel_info', '')}\n"
        f"地理距离矩阵：{json.dumps(state.get('geo_matrix', {}), ensure_ascii=False)}\n"
    )
    data, content, msgs = invoke_json_agent(
        state, prompt, agent, agent_name, "budget", {}
    )
    return {"budget_info": content, "messages": msgs}

def reflection_node(state:TravelState, agent, agent_name: str):
    """reflection 节点：反思计划完整性，判断是否需要重规划。"""
    prompt = (
        f"请反思以下旅行计划的完整性：\n"
        f"用户需求：{state.get('query', '')}\n"
        f"计划：{state.get('plan', '')}\n"
        f"天气信息：{state.get('weather_info', '')}\n"
        f"交通信息：{state.get('transport_info', '')}\n"
        f"酒店信息：{state.get('hotel_info', '')}\n"
        f"预算信息：{state.get('budget_info', '')}\n"
        f"当前迭代：{state.get('iteration', 0)}/{state.get('max_iterations', 3)}\n"
    )
    data, content, msgs = invoke_json_agent(
        state, prompt, agent, agent_name, "reflection", {}
    )
    return {
        "reflection": content,
        "needs_replan": data.get("needs_replan", False),
        "risk_warnings": data.get("risk_warnings", []),
        "iteration": state.get("iteration", 0) + 1,
        "messages": msgs,
    }

def write_node(state:TravelState, agent, agent_name: str):
    """write 节点：生成最终旅行计划。"""
    prompt = (
        f"请根据以下信息生成完整的旅行计划：\n"
        f"用户需求：{state.get('query', '')}\n"
        f"出行日期：{state.get('travel_dates', [])}\n"
        f"出发地：{state.get('origin', '')}\n"
        f"目的地：{state.get('destination', '')}\n"
        f"天气信息：{state.get('weather_info', '')}\n"
        f"交通信息：{state.get('transport_info', '')}\n"
        f"酒店信息：{state.get('hotel_info', '')}\n"
        f"预算信息：{state.get('budget_info', '')}\n"
        f"风险提示：{state.get('risk_warnings', [])}\n"
    )
    data, content, msgs = invoke_json_agent(
        state, prompt, agent, agent_name, "write", {}
    )
    return {"final": content, "draft": content, "messages": msgs}

def research_node(state:TravelState, agent, agent_name:str):
    """research节点，调用RAG知识库搜索信息"""
    dest = state.get("destination", "")
    query = state.get("query", "")
    if not dest:
        return {"research_info": "无数据", "messages": []}
    
    rag_query = f"{dest}旅游，景点推荐，美食推荐"
    records = search_knowledge_base_records(rag_query, 5)
    if not records:
        return {"research_info": "未找到信息", "messages": []}
    
    lines = []
    for i, r in enumerate(records, 1):
        lines.append(f"{i}.{r["content"]}")
    l = '\n'.join(lines)
    prompt = (
        f"旅游地点：{dest}\n"
        f"用户要求：{query}\n\n"
        f"旅游信息：\n{l}"
    )
    data, content, msgs = invoke_json_agent(
        state, prompt, agent, agent_name, "research", {}
    )
    return {"research_info":content, "messages":msgs}
