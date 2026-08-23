import json
import logging
import math
import re
import sqlite3
from abc import ABC
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jieba
from langchain_community.embeddings import DashScopeEmbeddings

from .base import BaseMemory, MemoryEntry, MemoryType

logger = logging.getLogger("travel_agents.memory")


class BaseLongTermMemory(BaseMemory, ABC):
    """长期记忆抽象基类：DashScope Embeddings + BM25 混合搜索。

    向量嵌入：DashScope text-embedding-v1（1536 维，中文语义搜索）
    关键词匹配：BM25 算法（jieba 中文分词 + 正则英文提取）
    """

    def __init__(self, memory_type: MemoryType, api_key: str = ""):
        super().__init__(memory_type)
        self.api_key = api_key
        self.embeddings: Optional[DashScopeEmbeddings] = None
        if api_key:
            try:
                self.embeddings = DashScopeEmbeddings(
                    model="text-embedding-v1",
                    dashscope_api_key=api_key,
                )
            except Exception as e:
                logger.warning("DashScope Embeddings 初始化失败，向量搜索不可用: %s", e)

    def generate_embedding(self, text: str) -> List[float]:
        """生成文本的向量嵌入（DashScope text-embedding-v1，1536 维）。

        api_key 为空或调用失败时返回空列表，搜索降级为纯 BM25。
        """
        if self.embeddings is None:
            return []
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.warning("嵌入生成失败: %s", e)
            return []

    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度 = 点积 / (范数1 × 范数2)。"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def tokenize(self, text: str) -> List[str]:
        """中英文混合分词（参考 rank_bm25 + jieba 方案）。

        英文：正则 \\b\\w+\\b 提取完整单词（如 "hotel"、"g1"）
        中文：jieba 分词（如 "经济型" → ["经济", "型"]）
        """
        if not text:
            return []
        text_lower = text.lower()
        # 提取英文单词（含数字，如 "g1"、"553"）
        words = re.findall(r"\b\w+\b", text_lower)
        # jieba 中文分词
        chinese = list(jieba.cut(text))
        return words + chinese

    def bm25_score(
        self,
        query: str,
        document: str,
        avgdl: float = 100.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """计算 BM25 关键词匹配分数。

        BM25 公式：score = Σ IDF(term) * (tf * (k1+1)) / (tf + k1*(1-b+b*dl/avgdl))

        Args:
            query: 查询文本
            document: 文档文本
            avgdl: 文档集合平均长度（简化为固定值 100）
            k1: 词频饱和参数（1.2-2.0，默认 1.5）
            b: 长度归一化参数（0-1，默认 0.75）

        Returns:
            BM25 分数（≥0，越高越相关）
        """
        query_terms = self.tokenize(query)
        doc_terms = self.tokenize(document)
        doc_len = len(doc_terms)

        if doc_len == 0 or not query_terms:
            return 0.0

        doc_counter = Counter(doc_terms)
        score = 0.0

        for term in query_terms:
            tf = doc_counter.get(term, 0)
            if tf == 0:
                continue
            # 简化 IDF：由于只比较少量文档，用固定值 1.0
            # 生产环境应从全库统计 IDF
            idf = 1.0
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))

        return score
    
class SQLiteLongTermMemory(BaseLongTermMemory):
    """基于 SQLite 的长期记忆实现。"""
    def __init__(self, memory_type: MemoryType, db_path: Optional[str] = None, api_key: str = ""):
        super().__init__(memory_type, api_key)  # 传 api_key 给父类
        if db_path is None:
            db_path = str(Path(__file__).resolve().parents[2] / "data" / "memory.db")
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_tables()

    def ensure_db_directory(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # ⭐ 让 row 支持字典访问 row["id"]
        return conn

    def init_tables(self) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories(
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    user_id TEXT,
                    namespace TEXT,
                    metadata TEXT,
                    embedding TEXT,
                    created_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            # 3 个索引加速查询
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user "
                "ON memories(user_id, memory_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_namespace "
                "ON memories(namespace)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created "
                "ON memories(created_at)"
            )
            conn.commit()

    def save(self, entry: MemoryEntry) -> str:
        """保存记忆条目，返回 entry.id。

        str 内容生成 embedding（向量搜索）；
        dict 内容不生成 embedding（靠精确匹配 + BM25）。
        """
        # 1. content 序列化 + embedding 生成
        if isinstance(entry.content, str):
            content = entry.content
            embedding = self.generate_embedding(entry.content)
        elif isinstance(entry.content, dict):
            content = json.dumps(entry.content, ensure_ascii=False)
            embedding = None  # dict 内容不生成向量
        else:
            content = str(entry.content)
            embedding = None

        # 2. metadata 序列化
        metadata = json.dumps(entry.metadata, ensure_ascii=False)

        # 3. embedding 序列化（list → JSON 字符串；None 保持 None）
        embedding_str = json.dumps(embedding) if embedding else None

        # 4. created_at 序列化
        created_at = entry.created_at.isoformat()

        # 5. INSERT OR REPLACE（同 id 覆盖）
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, memory_type, user_id, namespace, metadata,
                embedding, created_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, content, entry.memory_type.value,
                entry.user_id, entry.namespace, metadata,
                embedding_str, created_at, entry.access_count,
            ))
            conn.commit()
        return entry.id

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """按 ID 查询单条记忆。"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()
        if row is None:
            return None
        return self.row_to_entry(row)

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 5,
        **kwargs
    ) -> List[MemoryEntry]:
        """混合搜索：DashScope 向量相似度 + BM25 关键词匹配。

        DashScope 不可用时（api_key 为空或网络故障），降级为纯 BM25。
        """
        # 1. 生成查询向量（可能为空列表，降级为纯 BM25）
        query_embedding = self.generate_embedding(query)

        # 2. 构建 SQL 条件
        conditions = ["memory_type = ?"]
        params: List[Any] = [self.memory_type.value]
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        where_clause = " AND ".join(conditions)

        # 3. 查询所有候选行
        with self.get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {where_clause}",
                params
            ).fetchall()

        # 4. 计算每行的综合分数
        scored = []
        for row in rows:
            entry = self.row_to_entry(row)

            # 向量分数（DashScope 可用且有 embedding 时）
            vec_score = 0.0
            if query_embedding and row["embedding"]:
                try:
                    doc_embedding = json.loads(row["embedding"])
                    vec_score = self.calculate_similarity(query_embedding, doc_embedding)
                except (json.JSONDecodeError, TypeError):
                    pass

            # BM25 分数（归一化到 [0,1]）
            bm25 = self.bm25_score(query, str(entry.content))
            bm25_norm = min(bm25, 1.0)

            # 综合分数：向量 0.6 + BM25 0.4；DashScope 不可用时纯 BM25
            if query_embedding:
                total = vec_score * 0.6 + bm25_norm * 0.4
            else:
                total = bm25_norm

            scored.append((entry, total))

        # 5. 按分数降序，取 top-limit
        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:limit]]

    def delete(self, memory_id: str) -> bool:
        """按 ID 删除单条记忆。"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear(self, user_id: Optional[str] = None, namespace: Optional[str] = None) -> int:
        """批量清除记忆，返回清除数量。"""
        conditions = ["memory_type = ?"]
        params: List[Any] = [self.memory_type.value]
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        where_clause = " AND ".join(conditions)

        with self.get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE {where_clause}",
                params
            )
            conn.commit()
            return cursor.rowcount

    def list_namespaces(self, user_id: Optional[str] = None) -> List[str]:
        """列出所有命名空间（去重）。"""
        query = "SELECT DISTINCT namespace FROM memories WHERE memory_type = ?"
        params: List[Any] = [self.memory_type.value]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [row["namespace"] for row in rows if row["namespace"]]

    def row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """将数据库行转换为 MemoryEntry（不恢复 embedding）。"""
        # content 反序列化：尝试解析为 dict，失败则保持 str
        content = row["content"]
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        # metadata 反序列化
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        return MemoryEntry(
            id=row["id"],
            content=content,
            memory_type=MemoryType(row["memory_type"]),
            user_id=row["user_id"],
            namespace=row["namespace"],
            metadata=metadata,
            created_at=datetime.fromisoformat(row["created_at"]),
            access_count=row["access_count"],
        )

class SemanticMemoryStore(SQLiteLongTermMemory):
    """语义记忆：存储用户画像和事实知识，支持渐进式画像更新。"""
    def __init__(self, db_path: Optional[str] = None, api_key: str = ""):
        super().__init__(MemoryType.SEMANTIC, db_path, api_key)

    def save_profile(self,user_id: str,profile_data: Dict[str, Any],merge: bool = True) -> str:
        """画像渐进更新，融合新画像数据和现有数据"""
        if merge:
            existing = self.get_profile(user_id)
            if existing:
                existing.update(profile_data)
                profile_data = existing
        
        entry = MemoryEntry(
            content = profile_data,
            memory_type = MemoryType.SEMANTIC,
            user_id = user_id,
            namespace="user_profile",
            metadata={"type": "profile", "version": "1.0"},
            id=f"profile_{user_id}",
        )
        return self.save(entry)
    
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        entry = self.get(f"profile_{user_id}")
        if entry and isinstance(entry.content, dict):
            return entry.content
        if entry:
            # content 是 str，尝试 json.loads
            try:
                return json.loads(entry.content)
            except (json.JSONDecodeError, TypeError):
                return None
        return None
    
    def save_fact(self, user_id: str, fact: str, category: Optional[str] = None) -> str:
        entry = MemoryEntry(
            content=fact,                           # str 类型，父类会生成 embedding
            memory_type=MemoryType.SEMANTIC,
            user_id=user_id,
            namespace=f"facts/{category}" if category else "facts",
            metadata={"category": category},
        )
        return self.save(entry)
    
class EpisodicMemoryStore(SQLiteLongTermMemory):
    """情景记忆：存储任务执行记录，支持历史查询和相似任务搜索。"""
    def __init__(self, db_path: Optional[str] = None, api_key: str = ""):
        super().__init__(MemoryType.EPISODIC, db_path, api_key)

    def save_task_record(self,user_id: str,task_type: str,task_data: Dict[str, Any],outcome: Optional[str] = None)-> str:
        content ={
            "task_type":task_type,
            "data":task_data,
            "outcome":outcome,
            "timestamp": datetime.now().isoformat(),
        }
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.EPISODIC,
            user_id=user_id,
            namespace=f"tasks/{task_type}",         # 如 "tasks/travel_plan"
            metadata={
                "task_type": task_type,
                "has_outcome": outcome is not None,
            },
        )
        return self.save(entry)
    
    def get_similar_tasks(self, user_id:str, task_des:str, limit:int=5)-> List[MemoryEntry]:
        return self.search(query=task_des,user_id=user_id,limit=limit)
    
    def get_task_history(self,user_id: str,task_type: Optional[str] = None,limit: int = 10) -> List[MemoryEntry]:
        # 1. 构造 namespace（可选）
        namespace = f"tasks/{task_type}" if task_type else None

        # 2. 构建 SQL（按时间倒序）
        with self.get_connection() as conn:
            query = "SELECT * FROM memories WHERE memory_type = ? AND user_id = ?"
            params: List[Any] = [self.memory_type.value, user_id]

            if namespace:
                query += " AND namespace = ?"
                params.append(namespace)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()

        return [self.row_to_entry(row) for row in rows]