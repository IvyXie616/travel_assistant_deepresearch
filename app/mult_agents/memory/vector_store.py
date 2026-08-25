"""Chroma 向量存储层：长期记忆的语义索引（Phase 7.5 Step 1）。

职责边界：
- 只负责"存文本 + 按语义找"（ANN 检索 + metadata 过滤）
- 结构化查询（精确 get / 时间排序 / 事务）走 SQLite，不在这里做
- 管理 2 个 collection，与 RAG 的 travel_knowledge 隔离：
    - episodic_summaries：情景记忆（任务摘要）
    - semantic_facts：语义记忆（用户偏好/事实）
- doc_id 与 SQLite memory id 对齐（双写一致性与删除同步的基础）
- 索引层可降级：Chroma 不可用时系统照常运行（检索降级为 BM25）
"""
import os

# ⭐ 必须在 import chroma 之前设置，彻底禁用 Chroma 遥测（与 tools.py 一致）
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger("travel_agents.memory.vector")

# 支持的 collection 名（与计划文件 Phase 7.5 对应）
EPISODIC_COLLECTION = "episodic_summaries"
SEMANTIC_COLLECTION = "semantic_facts"


class ChromaMemoryStore:
    """长期记忆的 Chroma 语义索引层。

    设计要点：
    - 嵌入模型与 RAG 系统同款（text-embedding-v1），persist_dir 共用 ./data/chroma，
      靠 collection 名隔离数据（一个 Chroma 实例多 collection 是官方推荐用法）
    - 降级设计：__init__ 失败只降级为空字典，不抛异常——
      主数据在 SQLite，索引层缺失时检索走 BM25 fallback
    """

    def __init__(self, api_key: str, persist_dir: str = "./data/chroma"):
        """初始化 embeddings 与两个 collection 的 Chroma 实例。

        Args:
            api_key: DashScope API Key（用于 text-embedding-v1 嵌入）
            persist_dir: Chroma 持久化目录，与 RAG 共用（默认 ./data/chroma）

        初始化策略：
        1. embeddings 构造是轻量的（不验证 key），无需 try/except；
           真正的失败发生在 add/search 时调用嵌入接口，由各方法自行降级
        2. Chroma 实例创建可能因目录权限/版本兼容失败——逐个 try/except，
           失败的 collection 不进字典，后续操作按"collection 不存在"短路
        """
        self.api_key = api_key
        self.persist_dir = persist_dir

        # 1. 嵌入模型：与 RAGSystem 同款，保证记忆与知识库在同一向量空间
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=api_key,
        )

        # 2. 字典管理两个 collection 的 Chroma 实例（方法内按名取用，避免 if/else 分叉）
        #    ⭐ 降级关键：任一 collection 初始化失败只 warning，不阻断启动
        self._stores: Dict[str, Chroma] = {}
        for name in (EPISODIC_COLLECTION, SEMANTIC_COLLECTION):
            try:
                self._stores[name] = Chroma(
                    collection_name=name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_dir,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning(
                    "Chroma collection '%s' 初始化失败，该 collection 语义检索降级: %s",
                    name, e,
                )

        ready = list(self._stores.keys()) if self._stores else ["（无，已全部降级）"]
        logger.info(
            "ChromaMemoryStore 初始化 | dir=%s | 就绪 collections=%s",
            persist_dir, ready,
        )

    # ------------------------------------------------------------
    # 以下 4 个方法由你实现（Step 1 剩余任务）
    # 提示：每个方法开头先做降级短路——
    #   store = self._stores.get(collection)
    #   if store is None: 记 warning 并 return（add/delete）或 return []（search）
    # ------------------------------------------------------------

    def add(self, collection: str, doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """写入一条文档。doc_id = SQLite 的 memory id（对齐基础）。

        TODO: store.add_documents([Document(page_content=text, metadata=metadata)], ids=[doc_id])
        """
        store = self._stores.get(collection, None)
        if not store:
            logger.warning("Chroma collection '%s' 不存在", collection)
            return
        
        doc = Document(page_content=text, metadata=metadata)
        store.add_documents(documents=[doc], ids=[doc_id])

    def delete(self, collection: str, doc_id: str) -> None:
        """按 ID 删除单条（Chroma 的 delete 接受 ids 列表）。

        TODO: store._collection.delete(ids=[doc_id])
        """
        store = self._stores.get(collection, None)
        if not store:
            logger.warning("Chroma collection '%s' 不存在", collection)
            return
        store._collection.delete(ids=[doc_id])

    def delete_where(self, collection: str, where: Dict[str, Any]) -> None:
        """按 metadata 条件批量删除（如 {"user_id": "u1"}）。

        TODO: store._collection.delete(where=where)
        """
        store = self._stores.get(collection, None)
        if not store:
            logger.warning("Chroma collection '%s' 不存在", collection)
            return
        store._collection.delete(where=where)

    def search_with_distance(self, collection: str, query: str,
               where: Optional[Dict[str, Any]] = None, k: int = 5) -> List[Dict[str, Any]]:
        """ANN 检索：similarity_search_with_score(query, k, filter=where)。
        按metadata过滤检索结果

        ⚠️ 两个易错点：
        1. langchain_chroma 的过滤参数名是 filter（不是 where）；
           但 _collection.delete 的条件参数又叫 where——两套 API 命名不一致
        2. 返回的 score 是距离不是相似度，越小越相似
           （Step 4 的阈值过滤方向：distance > 1.2 丢弃）

        TODO: 返回 [{"text": ..., "metadata": ..., "distance": score}]
        """
        store = self._stores.get(collection, None)
        if not store:
            logger.warning("Chroma collection '%s' 不存在", collection)
            return []
        
        retrieves = store.similarity_search_with_score(query=query, k=k, filter=where)
        results:List[Dict[str,Any]] = []
        for doc, score in retrieves:
            record = {
                "text":str(doc.page_content).strip(),
                "metadata":doc.metadata,
                "distance":score
            }
            results.append(record)
        return results
    
    def search_with_similarity(self, collection: str, query: str,
               where: Optional[Dict[str, Any]] = None, k: int = 5) -> List[Dict[str, Any]]:
        """ANN 检索：similarity_search_with_relevance_scores(query, k, filter=where)。
        按metadata过滤检索结果

        ⚠️ 两个易错点：
        1. langchain_chroma 的过滤参数名是 filter（不是 where）；
           但 _collection.delete 的条件参数又叫 where——两套 API 命名不一致
        2. 返回的 score 是距离是相似度

        TODO: 返回 [{"text": ..., "metadata": ..., "distance": score}]
        """
        store = self._stores.get(collection, None)
        if not store:
            logger.warning("Chroma collection '%s' 不存在", collection)
            return []
        
        retrieves = store.similarity_search_with_relevance_scores(query=query, k=k, where=where)
        results:List[Dict[str,Any]] = []
        for doc, score in retrieves:
            record = {
                "text":str(doc.page_content).strip(),
                "metadata":doc.metadata,
                "revelance":score
            }
            results.append(record)
        return results