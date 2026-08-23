import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .base import MemoryEntry, MemoryType

def extract_memory_from_messages(
    messages: List[BaseMessage],
    user_id: str,
    thread_id: Optional[str] = None
) -> List[MemoryEntry]:
    """从对话消息列表中提取值得记忆的信息，转为 MemoryEntry 列表。"""
    entry_list:List[MemoryEntry] = []
    for msg in messages:
        if isinstance(msg.content, str):
            content = msg.content.strip() if msg.content.strip() else None
        elif isinstance(msg.content, dict):
            content = json.dumps(msg.content)
        
        if not content:continue

        if isinstance(msg, HumanMessage):
            entry = MemoryEntry(
                content=content,
                memory_type=MemoryType.SEMANTIC,
                user_id=user_id,
                thread_id=thread_id
            )
        elif isinstance(msg, AIMessage):
            entry = MemoryEntry(
                content=content,
                memory_type=MemoryType.EPISODIC,
                user_id=user_id,
                thread_id=thread_id
            )
        entry_list.append(entry)
    return entry_list

def format_memories_for_prompt(
    memories: List[MemoryEntry],
    max_length: int = 500
) -> str:
    """把记忆列表格式化为 prompt 可用的字符串。"""
    sem: List[str] = []
    epi: List[str] = []
    for memo in memories:
        content_str = memo.content if isinstance(memo.content, str) else json.dumps(memo.content, ensure_ascii=False)
        if memo.memory_type == MemoryType.SEMANTIC:
            sem.append(f"{len(sem)+1}. "+content_str[:100])
        elif memo.memory_type == MemoryType.EPISODIC:
            epi.append(f"{len(epi)+1}. "+content_str[:100])

    sem_str = "[用户画像]\n" + '\n'.join(sem)
    sem_str = sem_str[:max_length//2]
    epi_str = "[历史任务]\n" + '\n'.join(epi)
    epi_str = epi_str[:max_length//2]
    return sem_str+'\n'+epi_str

def merge_user_profile(
    existing: Optional[Dict[str, Any]],
    new_data: Dict[str, Any]
) -> Dict[str, Any]:
    """并用户画像（新数据覆盖旧数据的同名字段，保留旧数据的独有字段）"""
    if existing is None:
        return new_data.copy()
    merged = existing.copy()
    merged.update(new_data)  # 浅合并：同名字段覆盖
    return merged