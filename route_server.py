"""路线 MCP Server：提供城际交通查询（mock 数据，含车站经纬度）。

双层设计：
- _search_routes_data：核心逻辑（同步，返回 list[dict]），nodes.py 直接调用
- search_routes：MCP 工具（async，返回 JSON），供 MCP client 使用
"""
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("RouteServer")

# 车站 → 经纬度映射（城市, 车站名）→ [lat, lng]
STATION_LATLNG = {
    ("北京", "北京南站"): [39.8650, 116.3786],
    ("北京", "北京西站"): [39.8942, 116.3224],
    ("上海", "上海虹桥站"): [31.1932, 121.3194],
    ("上海", "上海站"): [31.2548, 121.4568],
}

# mock 路线数据（出发, 目的）→ 班次列表
ROUTES_MOCK = {
    ("北京", "上海"): [
        {
            "method": "高铁",
            "train_no": "G1",
            "origin_station": "北京南站",
            "dest_station": "上海虹桥站",
            "depart_time": "09:00",
            "arrive_time": "13:28",
            "price": 553,
        },
        {
            "method": "高铁",
            "train_no": "G3",
            "origin_station": "北京南站",
            "dest_station": "上海虹桥站",
            "depart_time": "11:00",
            "arrive_time": "15:38",
            "price": 553,
        },
    ],
}


def search_routes_data(origin: str, destination: str, date: str) -> list:
    """核心逻辑：返回路线列表（含车站经纬度）。nodes.py 直接调用。"""
    routes = ROUTES_MOCK.get((origin, destination), [])
    result = []
    for route in routes:
        item = route
        item["origin_latlng"] = STATION_LATLNG.get((origin, route["origin_station"]), [0, 0])
        item["dest_latlng"] = STATION_LATLNG.get((destination, route["dest_station"]), [0, 0])
        result.append(item)
    return result


@mcp.tool()
async def search_routes(origin: str, destination: str, date: str) -> str:
    """查询两城市之间的交通路线。

    Args:
        origin: 出发城市（中文）
        destination: 目的地城市（中文）
        date: 出发日期（YYYY-MM-DD）
    """
    print(f"[MCP route] search_routes({origin}→{destination}, date={date})")
    return json.dumps(search_routes_data(origin, destination, date), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")