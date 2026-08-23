from typing_extensions import Annotated, List, TypedDict
from operator import add
from langchain_core.messages import BaseMessage

class TravelState(TypedDict):
    # 组1: 会话标识与记忆
    query: str
    user_id: str
    thread_id: str
    memory_context: str
    messages: Annotated[List[BaseMessage], add]   # 使用Reducer：追加而非覆盖

    # 组2: planner 产出
    plan: str
    sub_tasks: list[str]
    travel_dates: list[str]       # 枚举每一天
    origin: str                   # 出发地点
    destination: str              # 目的地
    needs_clarification: bool     # 是否需要用户补充信息
    clarification_question: str   # 向用户提出的追加问题

    # 组3: 四个并行节点产出
    weather_info: str
    transport_info: str
    hotel_info: str               
    research_info: str

    # 组4: 地理整合
    geo_matrix: dict              # 各个地点间的距离

    # 组5: 预算与反思
    budget_info: str
    reflection: str
    needs_replan: bool
    risk_warnings: list[str]

    # 组6: 迭代控制与输出
    iteration: int
    max_iterations: int
    final: str
    draft: str

def create_initial_state(query, user_id, thread_id, memory_context, max_iterations=3):
    return {
        "query":query,
        "user_id":user_id,
        "thread_id":thread_id,
        "memory_context":memory_context,
        "messages":[],

        "plan":"",
        "sub_tasks":[],
        "travel_dates":[],
        "origin":"",
        "destination":"",
        "needs_clarification":False,
        "clarification_question":"",

        "weather_info": "",
        "transport_info": "",
        "hotel_info": "",            
        "research_info": "",

        "geo_matrix":{},

        "budget_info": "",
        "reflection": "",
        "needs_replan": False,
        "risk_warnings": [],

        "iteration": 0,
        "max_iterations":max_iterations,
        "final":"",
        "draft":""
    }