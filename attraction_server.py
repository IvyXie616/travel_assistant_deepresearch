"""景点 MCP Server：提供景点查询（mock 数据，含经纬度）。

双层设计：
- _search_attractions_data：核心逻辑（同步，返回 list[dict]），nodes.py 直接调用
- search_attractions：MCP 工具（async，返回 JSON），供 MCP client 使用
"""
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AttractionServer")

# mock 景点数据（城市）→ 景点列表
ATTRACTIONS_MOCK = {
    "上海": [
        {
            "name": "外滩",
            "address": "黄浦区中山东一路",
            "latlng": [31.2397, 121.4906],
            "open_hours": "全天开放",
            "ticket_price": 0,
        },
        {
            "name": "东方明珠",
            "address": "浦东新区世纪大道1号",
            "latlng": [31.2397, 121.4998],
            "open_hours": "09:00-21:00",
            "ticket_price": 120,
        },
        {
            "name": "豫园",
            "address": "黄浦区安仁街218号",
            "latlng": [31.2277, 121.4922],
            "open_hours": "08:30-17:00",
            "ticket_price": 40,
        },
        {
            "name": "南京路步行街",
            "address": "黄浦区南京东路",
            "latlng": [31.2360, 121.4760],
            "open_hours": "全天开放",
            "ticket_price": 0,
        },
    ],
    "北京": [
        {
            "name": "故宫",
            "address": "东城区景山前街4号",
            "latlng": [39.9163, 116.3972],
            "open_hours": "08:30-17:00",
            "ticket_price": 60,
        },
        {
            "name": "天安门广场",
            "address": "东城区东长安街",
            "latlng": [39.9087, 116.3975],
            "open_hours": "全天开放",
            "ticket_price": 0,
        },
    ],
}


def search_attractions_data(city: str, keywords: str = "") -> list:
    """核心逻辑：返回景点列表（含经纬度）。nodes.py 直接调用。

    Args:
        city: 城市名称
        keywords: 关键词（可选，用于筛选，如"博物馆"）
    """
    attractions = ATTRACTIONS_MOCK.get(city, [])
    if keywords:
        attractions = [a for a in attractions if keywords in a["name"] or keywords in a["address"]]
    return attractions


@mcp.tool()
async def search_attractions(city: str, keywords: str = "") -> str:
    """查询指定城市的景点。

    Args:
        city: 城市名称（中文）
        keywords: 关键词（可选，如"博物馆"、"公园"）
    """
    print(f"[MCP attraction] search_attractions(city={city}, keywords={keywords})")
    return json.dumps(search_attractions_data(city, keywords), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")