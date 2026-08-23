import os
from app.mult_agents.memory.manager import MemoryManager
from langchain_core.messages import HumanMessage, AIMessage

# 清空旧数据库
db_path = "./data/memory_test.db"
if os.path.exists(db_path):
    os.remove(db_path)

from app.mult_agents.config import AppConfig
config = AppConfig.from_env()
mgr = MemoryManager(api_key=config.api_key, db_path=db_path)

# ① 保存画像
mgr.save_user_profile("u1", {"budget_level": "经济型"})
mgr.save_user_profile("u1", {"accommodation_preference": "经济型酒店"})  # merge
profile = mgr.get_user_profile("u1")
print(f"画像: {profile}")

# ② 对话持久化
mgr.persist_turn("u1", "t1", "我想去上海旅游3天", "好的，我帮您规划上海3天行程")
mgr.persist_turn("u1", "t1", "我喜欢经济型酒店", "明白了，已记住您的偏好")

# ③ 获取短期记忆
msgs = mgr.get_short_term_messages("t1")
print(f"短期记忆: {len(msgs)} 条")

# ④ 获取任务历史
tasks = mgr.get_task_history("u1", task_type="conversation")
print(f"任务历史: {len(tasks)} 条")

# ⑤ ⭐ 构建个性化 prompt 上下文
context_text = mgr.build_personalized_prompt_context("u1", "t1", "上海旅游")
print(f"\n=== 个性化上下文 ===\n{context_text[:300]}")

# ⑥ 清除
counts = mgr.clear_user_memory("u1")
print(f"\n清除: {counts}")

print("\n✅ manager.py 验证通过")