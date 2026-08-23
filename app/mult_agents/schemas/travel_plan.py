"""旅行计划 Pydantic Schema 定义（v2 增强版）。

v2 增强：
- GeoPoint：统一地理坐标点，所有地点都有明确经纬度
- HotelInfo / AttractionInfo：v2 新增，含空间距离信息
- 解决用户问题"计划缺乏空间连贯性"，每个地点都有明确坐标和距离
- TransportInfo 含 origin_geo / dest_geo，支持空间连贯性检查
"""
from typing import List
from pydantic import BaseModel, Field

class GeoPoint(BaseModel):
    """地理坐标点。"""
    name: str = Field(description="地点名称，如'上海虹桥站'")
    address: str = Field(description="详细地址")
    latlng: str = Field(description="经纬度，格式'lat,lng'，如'31.1932,121.3194'")

class ItineraryItem(BaseModel):
    """按天的行程安排。"""
    day: int = Field(description="第几天，如1表示第1天")
    morning: str = Field(description="上午行程")
    afternoon: str = Field(description="下午行程")
    evening: str = Field(description="傍晚行程")
    accommodation: str = Field(description="具体酒店名，如'汉庭酒店(虹桥店)'")

class WeatherInfo(BaseModel):
    """天气信息。"""
    city: str = Field(description="城市名称")
    date: str = Field(description="日期，格式YYYY-MM-DD")
    temperature: str = Field(description="温度，如'30度'")
    condition: str = Field(description="天气状况，如'晴'、'多云'、'阵雨'")
    suggestion: str = Field(description="出行建议")

class TransportInfo(BaseModel):
    """交通信息。"""
    origin: str = Field(description="具体出发站，如'北京南站'")
    destination: str = Field(description="具体到达站，如'上海虹桥站'")
    method: str = Field(description="具体车次/航班，如'G1次高铁'")
    depart_time: str = Field(description="出发时间，如'09:00'")
    arrive_time: str = Field(description="到达时间，如'13:28'")
    cost: float = Field(description="具体票价")
    origin_geo: GeoPoint = Field(description="出发站地理坐标")
    dest_geo: GeoPoint = Field(description="到达站地理坐标")

class HotelInfo(BaseModel):
    """酒店信息（v2 新增）。"""
    name: str = Field(description="具体酒店名")
    address: str = Field(description="详细地址")
    price: float = Field(description="每晚价格")
    geo: GeoPoint = Field(description="酒店地理坐标")
    distance_to_station: str = Field(description="距下车点距离，如'1.2km，步行15分钟'")

class AttractionInfo(BaseModel):
    """景点信息（v2 新增）。"""
    name: str = Field(description="景点名称")
    address: str = Field(description="详细地址")
    geo: GeoPoint = Field(description="景点地理坐标")
    open_hours: str = Field(description="开放时间，如'08:30-17:00'")
    ticket_price: float = Field(description="门票价格，0表示免费")
    distance_to_hotel: str = Field(description="距酒店距离，如'2.3km，地铁15分钟'")

class BudgetItem(BaseModel):
    """预算明细项。"""
    category: str = Field(description="类别：transport/accommodation/attraction/meal/other")
    estimated_cost: float = Field(description="估算费用")
    currency: str = Field(default="CNY", description="货币，默认CNY")

class RiskWarning(BaseModel):
    """风险提示。"""
    type: str = Field(description="风险类型：budget/weather/schedule/spatial")
    description: str = Field(description="风险描述")
    severity: str = Field(description="严重程度：low/medium/high")
    suggestion: str = Field(description="改进建议")

class TravelPlan(BaseModel):
    """完整旅行计划（v2 增强版）。"""
    summary: str = Field(description="计划摘要")
    itinerary: List[ItineraryItem] = Field(description="按天的行程安排")
    weather: List[WeatherInfo] = Field(description="每日天气信息")
    transport: List[TransportInfo] = Field(description="交通方案")
    hotels: List[HotelInfo] = Field(description="酒店信息")
    attractions: List[AttractionInfo] = Field(description="景点信息")
    budget: List[BudgetItem] = Field(description="预算明细")
    total_budget: float = Field(description="总预算")
    geo_summary: str = Field(description="空间连贯性总结，如'酒店距车站0.8km，步行10分钟'")
    risk_warnings: List[RiskWarning] = Field(description="风险提示")
    tips: List[str] = Field(description="其他建议，如'8月3日有雨，建议携带雨具'")