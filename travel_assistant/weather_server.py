from mcp.server.fastmcp import FastMCP
import requests
import os
from dotenv import load_dotenv

os.environ["OPENWEATHER_API_KEY"] = "6937d586398e7612fb71b3bbab1a9418"
load_dotenv()  # 如果使用 .env 文件；若用 Colab 密钥则无需此句
API_KEY = os.getenv("OPENWEATHER_API_KEY")

mcp = FastMCP("WeatherServer")

@mcp.tool()
async def get_weather(city:str)->str:
    """
    查询指定城市的天气信息
    Args:
        city: 城市的英文名称，如"Beijing","Shanghai","Shenzhen","Tianjin"
    """
    print(f"调用天气工具,Args为{city}")
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)

        data = response.json()

        if data.get("cod") != 200:
            return f"查询失败：{data.get('message')}"

        temp = data["main"]["temp"]
        weather = data["weather"][0]["description"]

        return f"{city} 当前温度 {temp}°C，天气：{weather}"
    
    except Exception as e:
        print(f"天气查询出错：{e}")

if __name__ == "__main__":
    mcp.run(transport="stdio")