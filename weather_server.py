"""天气 MCP Server：提供城市天气查询（mock 数据，含经纬度）。

双层设计：
- _get_weather_data：核心逻辑（同步，返回 dict），nodes.py 直接调用
- get_weather：MCP 工具（async，返回 JSON），供 MCP client 使用
"""
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")

# 城市 → 经纬度映射
CITY_LATLNG = {
    "北京": [39.9042, 116.4074],
    "上海": [31.2304, 121.4737],
}

# mock 天气数据（city, date）→ 天气信息
WEATHER_MOCK = {
    ("北京", "2026-08-01"): {"temp": 28, "condition": "晴", "suggestion": "适宜出行，注意防晒"},
    ("北京", "2026-08-02"): {"temp": 30, "condition": "多云", "suggestion": "适宜出行"},
    ("北京", "2026-08-03"): {"temp": 27, "condition": "雷阵雨", "suggestion": "携带雨具"},
    ("上海", "2026-08-01"): {"temp": 30, "condition": "多云", "suggestion": "注意防晒，体感较热"},
    ("上海", "2026-08-02"): {"temp": 31, "condition": "晴", "suggestion": "适宜出行"},
    ("上海", "2026-08-03"): {"temp": 29, "condition": "阵雨", "suggestion": "携带雨具"},
}


def get_weather_data(city: str, date: str) -> dict:
    """核心逻辑：返回天气结构化数据（含经纬度）。nodes.py 直接调用。"""
    latlng = CITY_LATLNG.get(city, [0, 0])
    weather = WEATHER_MOCK.get((city, date), {"temp": 25, "condition": "晴", "suggestion": "适宜出行"})
    return {
        "city": city,
        "date": date,
        "temp": weather["temp"],
        "condition": weather["condition"],
        "suggestion": weather["suggestion"],
        "latlng": latlng,
    }


@mcp.tool()
async def get_weather(city: str, date: str) -> str:
    """查询指定城市指定日期的天气信息。

    Args:
        city: 城市名称（中文，如"北京"、"上海"）
        date: 日期（YYYY-MM-DD）
    """
    print(f"[MCP weather] get_weather(city={city}, date={date})")
    return json.dumps(get_weather_data(city, date), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")