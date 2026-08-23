from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

class MemoryType(Enum):
    """记忆类型枚举"""
    SHORT_TERM = "short_term"      # 短期记忆 - 当前对话上下文（内存）
    SEMANTIC = "semantic"          # 语义记忆 - 用户画像/偏好（SQLite + Chroma）
    EPISODIC = "episodic"          # 情景记忆 - 历史任务记录（SQLite）

@dataclass
class MemoryEntry:
    content: Union[str, Dict[str, Any]]       # 记忆内容（文本或结构化dict）
    memory_type: MemoryType                   # 记忆类型
    user_id: Optional[str] = None              # 用户标识
    thread_id: Optional[str] = None            # 线程标识（短期记忆用）
    namespace: Optional[str] = None            # 命名空间（长期记忆用，如"profile"、"history"）
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now) # 创建时间
    expires_at: Optional[datetime] = None      # 过期时间（短期记忆用）
    access_count: int = 0                      # 访问次数
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self)->Dict[str, Any]:
        return {
            "content":self.content,
            "memory_type":self.memory_type.value,
            "user_id":self.user_id,
            "thread_id":self.thread_id,
            "namespace":self.namespace,
            "metadata":self.metadata,
            "created_at":self.created_at.isoformat(), # datetime → str
            "expires_at":self.expires_at.isoformat() if self.expires_at else None,
            "access_count":self.access_count,
            "id":self.id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """从字典创建（SQLite 反序列化用）。"""
        return cls(
            id=data.get("id", str(uuid4())),
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),  # str → Enum
            user_id=data.get("user_id"),
            thread_id=data.get("thread_id"),
            namespace=data.get("namespace"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            access_count=data.get("access_count", 0),
        )

class BaseMemory(ABC):
    """记忆存储抽象基类。
    
    所有记忆存储后端（内存、SQLite）都需要实现此接口。
    """
    
    def __init__(self, memory_type: MemoryType):
        self.memory_type = memory_type
    
    @abstractmethod
    def save(self, entry: MemoryEntry) -> str:
        """保存记忆条目，返回记忆 ID。"""
        pass
    
    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """获取指定 ID 的记忆，不存在返回 None。"""
        pass
    
    @abstractmethod
    def search(self, query: str, user_id: Optional[str] = None, 
               namespace: Optional[str] = None, limit: int = 5, **kwargs) -> List[MemoryEntry]:
        """搜索记忆，返回记忆条目列表。"""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """删除指定记忆，返回是否删除成功。"""
        pass
    
    @abstractmethod
    def clear(self, user_id: Optional[str] = None, namespace: Optional[str] = None) -> int:
        """清除记忆，返回清除的数量。"""
        pass
    
    @abstractmethod
    def list_namespaces(self, user_id: Optional[str] = None) -> List[str]:
        """列出所有命名空间。"""
        pass