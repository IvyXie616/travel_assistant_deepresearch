from mcp.server.fastmcp import FastMCP
import requests
import os
from dotenv import load_dotenv

mcp = FastMCP("RouteServer")

@mcp.tool()
async def get_route(origin: str, destination: str) -> str:
    """
    查询两个城市之间的路线

    Args:
        origin: 出发城市
        destination: 目的地城市
    """
    return f"{origin} 到 {destination} 推荐高铁出行，约4小时，票价约300元"

if __name__ == "__main__":
    mcp.run(transport="stdio")