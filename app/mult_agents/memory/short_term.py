import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from .base import BaseMemory, MemoryEntry, MemoryType
from langchain_community.chat_models import ChatTongyi
from app.mult_agents.config import AppConfig

class ConversationBuffer:
    """管理单个对话线程的消息历史，消息过多时自动压缩（保留最近 N 条，旧消息生成摘要）。"""
    def __init__(self, max_lines=20, max_tokens=4000, summary_threshold=10):
        self.max_messages = max_lines       # 最多保留 20 条消息
        self.max_tokens = max_tokens           # token 上限（简化估算）
        self.summary_threshold = summary_threshold  # 超过此数量触发压缩
        self.messages: List[BaseMessage] = []  # 消息列表
        self.summary: Optional[str] = None     # 历史摘要
        self.token_count: int = 0              # 当前 token 数
        config = AppConfig.from_env()
        self.compress_llm = ChatTongyi(
            model="qwen-mini",
            temperature=0.2,
            dashscope_api_key=config.api_key
        )

    def add_message(self, msg:BaseMessage):
        """追加一条消息到 self.messages ，
        调用 _update_token_count() 更新计数，
        如果消息数超过 max_messages 则调用 _compress_messages() 压缩"""
        self.messages.append(msg)
        self.update_token_count()
        if len(self.messages)>self.max_messages:
            self.compress_messages()

    def add_messages(self, msgs:List[BaseMessage]):
        """遍历列表逐条调用 add_message"""
        for msg in msgs:
            self.add_message(msg)

    def get_messages(self, include_summary:bool=False, last_n:int=None):
        """返回消息列表。如果 include_summary=True 且 self.summary 存在，
        在列表开头插入一条 SystemMessage(content=f"历史对话摘要：{self.summary}") 。
        如果 last_n 不为 None，只返回最近 N 条（对 self.messages 切片 [-last_n:] ）"""
        result = self.messages.copy()
        if last_n and last_n>0:
            result = result[-last_n:0]
        if include_summary and self.summary:
            result.insert(0, SystemMessage(content=f"历史对话摘要：{self.summary}"))
        return result

    def clear(self):
        """重置""" 
        self.messages: List[BaseMessage] = []
        self.summary = None
        self.token_count = 0

    def update_token_count(self):
        """遍历所有消息，累加 len(str(msg.content)) ，
        除以 2 得到估算 token 数，存入 self.token_count"""
        L = sum([len(str(msg.content)) for msg in self.messages])
        self.token_count = L//2

    def compress_messages(self):
        if len(self.messages)<=self.max_messages:
            return
        msgs:List[str] = []
        for m in self.messages[:self.summary_threshold]:
            msg_content = str(m.content)
            if isinstance(m, AIMessage):
                msg_content = 'AI：'+msg_content
                msgs.append(msg_content)
            else:
                msg_content = '用户：'+msg_content
                msgs.append(msg_content)
        msg_prompt = '\n'.join(msgs)
        prompt = f"""
        你是一个对话总结专家，你将收到用户与AI的长对话，请你提取关键信息，输出包含所有关键信息的简洁版对话。
        严格遵循以下规定：
            1.你的输出中，每条格式为 \"用户: ...\" 或 \"AI: ...\" ，每条用户或AI的话各占一行。
            2.你输出的简洁版对话中，\"用户: \"或 \"AI: \"之后的对话内容不应超过100字。总对话不应超过8条。
            3.你的输出应与以下示例的格式严格相同
              示例：
                  用户：我想去上海
                  AI：你想去上海哪里
                  用户：我想去美食街
                  AI：我推荐你...

        待简化的长对话：
        {msg_prompt}
        """
        summ = self.compress_llm.invoke(prompt).content
        if self.summary:
            self.summary = f"{self.summary}\n\n[以下为更新的对话]\n{summ}"
        else: self.summary = summ

class ShortTermMemory(BaseMemory):
    def __init__(self, ttl_seconds: int = 3600*24, max_threads: int = 100):
        """输入 ： ttl_seconds: int = 3600*24 （线程过期秒数）
           max_threads: int = 100 （最大线程数）"""
        super().__init__(MemoryType.SHORT_TERM)
        self.ttl_seconds = ttl_seconds
        self.max_threads = max_threads

        self.storage: Dict[str, Dict[str, Any]] = {} #存储所有对话线程
        # 结构： {thread_id: {"buffer": ConversationBuffer, "metadata": {}, "created_at": datetime, "last_access": datetime}}

    def cleanup_expired(self):
        tobe_del = []
        extra_thread = max(0, len(self.storage)-self.max_threads)
        self.storage = dict(sorted(self.storage.items(), key=lambda item: item[1]["last_access"]))
        for id, content in self.storage.items():
            if datetime.now()-content["last_access"] > timedelta(seconds=self.ttl_seconds):
                tobe_del.append(id)
                if extra_thread > 0: extra_thread -= 1
        
        if extra_thread > 0:
            tobe_del.extend(self.storage.keys()[:extra_thread])

        tobe_del = list(set(tobe_del))
        for id in tobe_del:
            del self.storage[id]

    def get_or_create_buffer(self,thread_id:str)->ConversationBuffer:
        self.cleanup_expired()
        if thread_id not in self.storage.keys():
            new_buffer = ConversationBuffer()
            new_thread = {
                "buffer": new_buffer, 
                "metadata": {}, 
                "created_at": datetime.now(), 
                "last_access": datetime.now()
                }
            self.storage[thread_id] = new_thread
        else:
            self.storage[thread_id]["last_access"] = datetime.now()
        return self.storage[thread_id]["buffer"]
    
    def add_message(self, 
                    thread_id: str, 
                    message: BaseMessage, 
                    metadata: Optional[Dict[str, Any]] = None):
        buffer = self.get_or_create_buffer(thread_id)
        buffer.add_message(message)
        if metadata:
            self.storage[thread_id]["metadata"].update(metadata)

    def get_messages(self, thread_id: str,
                    include_summary: bool = True,
                    last_n: Optional[int] = None)->List[BaseMessage]:
        if thread_id not in self.storage.keys():
            return []
        else:
            self.storage[thread_id]["last_access"] = datetime.now()
            msgs = self.storage[thread_id]["buffer"].get_messages(include_summary, last_n)
            return msgs
        
    def get_thread_metadata(self, thread_id: str):
        if thread_id not in self.storage.keys():
            return {}
        else:
            self.storage[thread_id]["last_access"] = datetime.now()
            return self.storage[thread_id]["metadata"].copy()
        
    def update_thread_metadata(self, thread_id: str, metadata: Dict[str, Any]):
        buffer = self.get_or_create_buffer(thread_id)
        self.storage[thread_id]["metadata"].update(metadata)

    def clear_thread(self, thread_id:str):
        if thread_id in self.storage.keys():
            del self.storage[thread_id]
            return True
        else:
            return False
        
    def list_active_threads(self):
        self.cleanup_expired()
        return list(self.storage.keys())
    
    def get(self, memory_id):
        return None
    
    def delete(self, memory_id):
        return False
    
    def clear(self, user_id: Optional[str] = None, namespace: Optional[str] = None):
        if namespace:
            if self.clear_thread(namespace):
                return 1
            else: return 0

        else:
            count = len(self.storage)
            self.storage.clear()
            return count
    
    def list_namespaces(self, user_id: Optional[str] = None):
        return self.list_active_threads()

    def save(self, entry: MemoryEntry) -> str:
        """保存记忆条目，返回记忆 ID。

        将 entry.content 转为消息对象，存入对应线程的 buffer。
        """
        thread_id = entry.thread_id or "default"
        buffer = self.get_or_create_buffer(thread_id)

        # content → BaseMessage 转换
        if isinstance(entry.content, str):
            message = HumanMessage(content=entry.content)
        elif isinstance(entry.content, dict):
            content_str = entry.content.get("content", "")
            role = entry.content.get("role", "human")
            if role == "ai":
                message = AIMessage(content=content_str)
            else:
                message = HumanMessage(content=content_str)
        else:
            message = HumanMessage(content=str(entry.content))

        buffer.add_message(message)

        # 更新线程 metadata（update 而非覆盖）
        if entry.metadata:
            self.storage[thread_id]["metadata"].update(entry.metadata)

        return entry.id

    def search(self, query: str, user_id: Optional[str] = None,
               namespace: Optional[str] = None, limit: int = 5,
               **kwargs) -> List[MemoryEntry]:
        """搜索记忆，返回最近消息作为 MemoryEntry 列表。

        短期记忆无向量索引，简单返回最近 limit 条消息。
        """
        thread_id = namespace or "default"
        messages = self.get_messages(thread_id, include_summary=False)

        # 取最后 limit 条
        if limit > 0:
            recent = messages[-limit:]
        else:
            recent = messages

        results: List[MemoryEntry] = []
        for msg in recent:
            entry = MemoryEntry(
                content=msg.content,
                memory_type=MemoryType.SHORT_TERM,
                thread_id=thread_id,
                user_id=user_id,
                metadata={"role": "ai" if isinstance(msg, AIMessage) else "human"},
            )
            results.append(entry)

        return results
