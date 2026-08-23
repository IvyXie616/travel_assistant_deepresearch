"""酒店 MCP Server：提供酒店查询（mock 数据，含经纬度）。

双层设计：
- _search_hotels_data：核心逻辑（同步，返回 list[dict]），nodes.py 直接调用
- search_hotels：MCP 工具（async，返回 JSON），供 MCP client 使用
"""
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("HotelServer")

# mock 酒店数据（城市）→ 酒店列表
HOTELS_MOCK = {
    "上海": [
        {
            "name": "如家快捷酒店（虹桥店）",
            "address": "闵行区虹桥路",
            "price": 200,
            "latlng": [31.1950, 121.3200],
            "distance_to_station": 0.3,
        },
        {
            "name": "全季酒店（虹桥店）",
            "address": "闵行区申虹路",
            "price": 350,
            "latlng": [31.1980, 121.3220],
            "distance_to_station": 0.5,
        },
        {
            "name": "汉庭酒店（虹桥店）",
            "address": "闵行区绍虹路",
            "price": 180,
            "latlng": [31.1920, 121.3180],
            "distance_to_station": 0.4,
        },
    ],
    "北京": [
        {
            "name": "如家快捷酒店（北京南站店）",
            "address": "丰台区永外大街",
            "price": 220,
            "latlng": [39.8660, 116.3800],
            "distance_to_station": 0.2,
        },
    ],
}


def search_hotels_data(city: str, date: str, nearby_latlng: list = None) -> list:
    """核心逻辑：返回酒店列表（含经纬度）。nodes.py 直接调用。

    Args:
        city: 城市名称
        date: 入住日期
        nearby_latlng: 附近地标经纬度 [lat, lng]（可选，后续按距离排序用）
    """
    return HOTELS_MOCK.get(city, [])


@mcp.tool()
async def search_hotels(city: str, date: str, nearby_latlng: list = None) -> str:
    """查询指定城市的酒店（可指定附近地标）。

    Args:
        city: 城市名称（中文）
        date: 入住日期（YYYY-MM-DD）
        nearby_latlng: 附近地标经纬度 [lat, lng]（可选）
    """
    print(f"[MCP hotel] search_hotels(city={city}, date={date})")
    return json.dumps(search_hotels_data(city, date, nearby_latlng), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")