"""内部工具模块：定义旅行规划所需的计算工具与 RAG 检索工具。"""
import os

# 必须在 import chroma 之前设置，彻底禁用 Chroma 遥测
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import math
from typing import Optional

from langchain_core.tools import tool

from .rag.core import RAGSystem, RAGConfig


# ============================================================
# 全局 RAG 系统实例（延迟初始化）
# ============================================================

_RAG_SYSTEM: Optional[RAGSystem] = None


def init_rag_system(api_key: str, config: Optional[RAGConfig] = None):
    """初始化全局 RAG 系统。在 main.py 启动时调用一次。"""
    global _RAG_SYSTEM
    if _RAG_SYSTEM is None:
        try:
            _RAG_SYSTEM = RAGSystem(api_key, config)
        except Exception as e:
            print(f"RAG 系统初始化失败: {e}")


def search_knowledge_base_records(query: str, limit: int = 5) -> list[dict]:
    """检索本地知识库，返回结构化记录列表（供节点编排用）。"""
    if _RAG_SYSTEM is None:
        return []
    try:
        return _RAG_SYSTEM.search_records(query, k=limit)
    except Exception:
        return []


# ============================================================
# geo_distance_tool（haversine 距离计算）
# ============================================================

@tool
def geo_distance_tool(origin: list, destination: list) -> float:
    """计算两个经纬度坐标之间的球面距离（单位：km）。

    使用 haversine 公式，适用于短中距离的城市内/城际路径估算。
    后续可替换为高德路线规划 API（返回真实路网距离）。

    Args:
        origin: 起点经纬度，格式 [纬度, 经度]，如 [39.9042, 116.4074]（北京）
        destination: 终点经纬度，格式 [纬度, 经度]，如 [31.2304, 121.4737]（上海）

    Returns:
        两点间的球面距离（km），保留两位小数
    """
    R = 6371.0  # 地球平均半径（km）

    lat1, lon1 = origin
    lat2, lon2 = destination

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    distance = 2 * R * math.asin(math.sqrt(a))

    return round(distance, 2)


# ============================================================
# search_knowledge_base（RAG 向量检索）
# ============================================================

@tool
def search_knowledge_base(query: str, k:int=5) -> str:
    """查询本地旅游知识库（向量数据库）。

    当需要查询旅游攻略、景点介绍、美食推荐等旅游知识时使用此工具。
    输入应该是具体的查询问题，如"上海外滩游玩攻略"。
    """
    if _RAG_SYSTEM is None:
        return "错误：RAG 系统未初始化。"
    return _RAG_SYSTEM.search(query, k)
