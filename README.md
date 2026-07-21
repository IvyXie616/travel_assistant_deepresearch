# travel_assistant_deepresearch
	项目背景：为解决生活中常见的痛点，即准备出行却不知如何安排的问题，开发面向个人旅行者或家庭的AI 出行助手。
	技术栈：LangChain + LangGraph + ChromaDB + DashScope/Qwen + FastAPI + MCP + SQLite + Pydantic
	功能实现：1. 设计了基于 LangGraph的多 Agent 协作工作流，包含 Planner、Research、Weather、Transport、Budget、Reflection 六个专业化智能体，通过条件路由和反思循环实现任务的自主规划与迭代优化。  2. 构建了完整的 RAG 检索系统，支持旅行攻略文档的解析、分块、嵌入，Research Agent 可结合语义检索与工具调用获取精准信息。  3.实现了短期记忆+长期记忆的记忆模块，支持基于记忆的用户个性化推荐。  4. 通过 MCP 协议封装天气、路线、酒店等外部服务为标准化 Tool，结合 LangChain @tool 装饰器实现内部工具的混合接入，统一工具调用接口。  5. 实现了 Reflection Agent 自我检查机制，自动检测超预算、行程冲突、天气风险等问题并进行迭代优化。  
