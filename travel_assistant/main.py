from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

import os,json

from llm import get_llm, LLM
from planner import plan_task
from memory import get_preference

async def main():
    with open("servers_config.json","r",encoding="utf-8") as f:
        config = json.load(f)
    client = MultiServerMCPClient(config["mcpServers"])

    tools = await client.get_tools()
    print(tools)

    agent = create_agent(
        model=LLM,
        tools=tools,
        system_prompt="""
        你是一个工具调用代理。

        严格规则：
        1. 必须调用工具获取信息
        2. 禁止自己编造数据
        3. 所有结论必须基于工具返回
        4. 如果没有足够的信息或任务无法完成，你必须统一回答"任务实现失败，请提供更多信息"
        """
    )
    
    user_input = "我住在深圳，五一想去广西旅游，请帮我规划一下"
    memory = get_preference("user1")
    print("用户偏好：", memory)
    # 任务拆解
    tasks = plan_task(user_input + f"\n用户偏好：{memory}")
    print("任务拆解：")
    for t in tasks:
        print(t)

    results = []
    
    # 2️⃣ 逐任务执行
    for task in tasks:
        try:
            result = await agent.ainvoke({
                "messages": [
                    {"role": "user", "content": task}
                ]
            })
            output = result["messages"][-1].content

            if not output or "失败" in output or "更多信息" in output:
                output = f"任务失败或无结果：{task}"
        except Exception as e:
            output = f"任务执行异常：{task}，错误：{str(e)}"

        print(output)
        results.append({
            "task":task,
            "result":output
        })
    
    # 最终汇总
    final_prompt = f"""
请基于以下信息生成旅行计划，并严格输出JSON格式：

字段要求：
- itinerary（行程安排）
- weather（天气情况）
- transport（交通建议）
- budget（预算估算）

数据：
{results}

只输出JSON，不要解释
"""

    final_result = LLM.invoke(final_prompt)

    print("\n最终结构化结果：")
    print(final_result.content)

asyncio.run(main())