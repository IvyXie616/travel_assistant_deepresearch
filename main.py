"""travel_assistant 主入口：LangGraph 工作流 + Memory 个性化。"""
import asyncio
import logging

from app.mult_agents.config import AppConfig
from app.mult_agents.agents import build_agents
from app.mult_agents.travelState import create_initial_state
from app.mult_agents.graph import build_workflow
from app.mult_agents.memory.manager import MemoryManager
from app.mult_agents.nodes import init_memory_manager
from app.mult_agents.tools import init_rag_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


async def main():
    # 1. 加载配置
    config = AppConfig.from_env()

    # 2. 初始化 RAG（research_node 依赖）
    init_rag_system(config.api_key)

    # 3. 初始化 MemoryManager（nodes.py 记忆注入依赖）
    memory_mgr = MemoryManager(api_key=config.api_key)
    init_memory_manager(memory_mgr)

    # 4. 构建 agents + 工作流
    agents = build_agents(config)
    app = build_workflow(agents)

    # 5. 模拟用户对话
    user_id = "user1"
    thread_id = "thread1"
    query = "我想2026-08-01从北京去上海旅游3天，我喜欢经济型酒店"

    # 6. 构建初始状态
    initial_state = create_initial_state(
        query=query,
        user_id=user_id,
        thread_id=thread_id,
        memory_context="",  # MemoryManager 会动态注入
        max_iterations=3,
    )

    # 7. 执行工作流
    print("=== 开始执行旅行规划工作流 ===")
    result = app.invoke(initial_state)

    # 8. 输出结果
    print("\n=== 旅行计划 ===")
    print(result.get("final", "（无输出）")[:500])

    print("\n=== 记忆状态 ===")
    profile = memory_mgr.get_user_profile(user_id)
    print(f"用户画像: {profile}")


if __name__ == "__main__":
    asyncio.run(main())