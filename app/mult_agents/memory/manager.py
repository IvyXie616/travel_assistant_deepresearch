import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .base import MemoryEntry, MemoryType
from .long_term import EpisodicMemoryStore, SemanticMemoryStore
from .short_term import ShortTermMemory
from .utils import format_memories_for_prompt, merge_user_profile

logger = logging.getLogger("memory_manager")

class MemoryManager:
    def __init__(self, 
                api_key: str, 
                db_path: Optional[str] = None, 
                short_term_ttl: int = 3600*24,
                max_threads = 20):
        self.short_term = ShortTermMemory(ttl_seconds=short_term_ttl, max_threads=max_threads)
        self.semantic = SemanticMemoryStore(db_path, api_key)
        self.episodic = EpisodicMemoryStore(db_path, api_key)

    def add_short_term_message(
            self,
            thread_id: str,
            message: BaseMessage,
            user_id: str = "default",
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.short_term.add_message(thread_id, message, metadata)

    def get_short_term_messages(
            self,
            thread_id: str,
            include_summary: bool = True,
            last_n: Optional[int] = None
    )->List[BaseMessage]:
        return self.short_term.get_messages(thread_id, include_summary, last_n)
    
    def clear_short_term(self, thread_id: str)->bool:
        return self.short_term.clear_thread(thread_id)
    
    def save_user_profile(self, user_id: str, profile: Dict[str, Any], merge: bool = True)->str:
        return self.semantic.save_profile(user_id, profile, merge)
    
    def get_user_profile(self, user_id: str)->Optional[Dict[str, Any]]:
        return self.semantic.get_profile(user_id)
    
    def save_task(
            self,
            user_id: str,
            task_type: str,
            task_data: Dict[str, Any],
            outcome: Optional[str] = None
    )->str:
        return self.episodic.save_task_record(user_id, task_type, task_data, outcome)

    def get_task_history(
            self,
            user_id: str,
            task_type: Optional[str] = None,
            limit:int = 10
    )->List[MemoryEntry]:
        return self.episodic.get_task_history(user_id, task_type, limit)
    
    def search_semantic(
            self,
            user_id: str,
            query: str,
            namespace: Optional[str] = None,
            limit: int = 5
    )->List[MemoryEntry]:
        return self.semantic.search(query, user_id, namespace, limit)
    
    def search_similar_tasks(
            self,
            user_id: str,
            query: str,
            limit: int = 5
    )->List[MemoryEntry]:
        return self.episodic.get_similar_tasks(user_id, query, limit)
    
    def get_context_for_agent(
            self,
            user_id: str,
            thread_id: str,
            query: Optional[str] = None,
            max_memories: int = 10
    )->Dict[str, Any]:
        context = {}
        # 1.用户画像
        context["user_profile"] = self.get_user_profile(user_id)
        # 2. 最近对话（短期记忆，最近5条）
        context["recent_messages"] = self.get_short_term_messages(thread_id, last_n=5)

        # 3. 相关记忆（长期记忆搜索，仅 query 存在时）
        if query:
            semantic_results = self.search_semantic(user_id, query, limit=max_memories // 2)
            episodic_results = self.search_similar_tasks(user_id, query, limit=max_memories // 2)
            combined = semantic_results + episodic_results
            combined.sort(key=lambda x:x.created_at, reverse=True) #从新到旧
            context["relevant_memories"] = combined[:max_memories]
        else:
            context["relevant_memories"] = []

        # 4. 最近任务
        context["recent_tasks"] = self.get_task_history(user_id, limit=3)
        return context
    
    def build_personalized_prompt_context(
            self,
            user_id: str,
            thread_id: str,
            query: str,
            max_memories: int = 8
    )->str:
        # 1. 获取上下文
        context = self.get_context_for_agent(user_id,thread_id,query,max_memories)

        # 2. 格式化各部分
        sections = []
        # 2a. 用户画像
        profile = context.get("user_profile")
        if profile:
            profile_text = json.dumps(profile, ensure_ascii=False, indent=2)
            sections.append(f"[用户画像]\n{profile_text}")
        # 2b. 最近对话（最近8条，每条截断120字符）
        recent_msgs = context.get("recent_messages")
        recent_lines = []
        for msg in recent_msgs[-8:]:
            if isinstance(msg, AIMessage):
                role = "AI"
            elif isinstance(msg, HumanMessage):
                role = "用户"
            text = str(msg.content).strip()[:120]
            if text:
                recent_lines.append(f"- {role}:{text}")
        if recent_lines:
            sections.append("[最近对话]\n" + "\n".join(recent_lines))

        # 2c. 相关记忆
        memories = context.get("relevant_memories", [])
        if memories:
            memory_text = format_memories_for_prompt(memories, max_length=1000)
            sections.append("[相关记忆]\n" +memory_text)
        
        # 2d. 最近任务（每个截断80字符）
        tasks = context.get("recent_tasks", [])
        if tasks:
            task_lines = []
            for t in tasks:
                content = t.content if isinstance(t.content, dict) else {"text": str(t.content)}
                task_lines.append(f"- {content.get('task_type', 'task')}: {str(content.get('data', ''))[:80]}")
            sections.append("[最近任务]\n" + "\n".join(task_lines))

        # 3. 拼接所有部分
        injected = '\n\n'.join(sections).strip()
        
        logger.info(
            "[memory] prompt注入 | user=%s thread=%s profile=%s memories=%d tasks=%d injected_chars=%d",
            user_id, thread_id, bool(profile), len(memories), len(tasks), len(injected)
        )
        return injected
    
    def persist_turn(
            self,
            user_id: str,
            thread_id: str,
            query:str, #用户输入
            answer:str #ai回答
    ):
        # 1. 存入短期记忆
        self.add_short_term_message(thread_id, HumanMessage(content=query), user_id)
        self.add_short_term_message(thread_id, AIMessage(content=answer), user_id)

        # 2. 检测是否需要提取长期记忆（关键词触发）
        remember_markers = [
            "记住", "我喜欢", "我偏好", "我不喜欢",
            "我叫", "我是", "我希望",
            "remember", "i like", "i prefer"]
        
        should_extract = any(marker in query.lower() for marker in remember_markers)
        if should_extract:
            # 简化版：直接把 query 作为偏好存入画像
            self.save_user_profile(user_id, {"raw_preference": query}, merge=True)
            logger.info("[memory] 检测到偏好，已存入画像 | user=%s query=%s", user_id, query[:50])

        # 3. 保存任务记录（每次对话都存）
        self.save_task(
            user_id=user_id,
            task_type="conversation",
            task_data={"query": query},
            outcome=answer[:500]  # 截断，避免过长
        )
        logger.info(
            "[memory] 对话持久化 | user=%s thread=%s remember=%s",
            user_id, thread_id, should_extract)
        
    def clear_user_memory(self, user_id:str)->Dict[str, int]:
        counts = {}
        counts["semantic"] = self.semantic.clear(user_id=user_id)
        counts["episodic"] = self.episodic.clear(user_id=user_id)
        threads = self.short_term.list_active_threads()
        short_count = 0
        for thread_id in threads:
            if self.short_term.clear_thread(thread_id):
                short_count += 1
        counts["short_term"] = short_count
        return counts