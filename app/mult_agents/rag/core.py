import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma as _ChromaVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RAGConfig:
    """RAG 系统配置。"""
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "travel_knowledge"          # ⭐ travel_assistant 专用
    embedding_model: str = "text-embedding-v1"
    chunk_size: int = 500
    chunk_overlap: int = 50

class RAGSystem:
    """RAG 系统：文档入库 + 向量检索。

    全流程：
        入库：text → RecursiveCharacterTextSplitter 分块 → DashScopeEmbeddings 嵌入 → Chroma 存储
        检索：query → DashScopeEmbeddings 嵌入 → Chroma 相似度搜索 → 返回 top-k
    """
    def __init__(self, api_key: str, config: Optional[RAGConfig] = None):
        if config:
            self.config = config
        else: self.config = RAGConfig()
        self.api_key = api_key
        
        # 嵌入模型，将文本转换为向量
        self.embeddings = DashScopeEmbeddings(
            model = self.config.embedding_model,
            dashscope_api_key= self.api_key
            )
        # 文本分块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.config.chunk_size, 
            chunk_overlap = self.config.chunk_overlap,
            separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            )
        # 向量数据库
        self.vectorstore = _ChromaVectorStore(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.config.chroma_persist_dir,
        )

        logger.info(
            "RAG 初始化 | collection=%s | dir=%s",
            self.config.collection_name,
            self.config.chroma_persist_dir,
        )

    def search_records(self, query:str, k=5):
        """向量检索，返回结构化记录列表（供程序用）。"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        #docs = self.vectorstore.similarity_search(query, k=k)

        records = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            source = str(metadata.get("source", ""))
            title = Path(source).name if source else f"本地知识片段-{i}"
            records.append(
                {
                    "source_id": f"LOC-{i}",
                    "doc_id": source,
                    "title": title,
                    "content": doc.page_content,
                    "source_type": "local",
                    "metadata": metadata,
                }
            )
        return records
    
    def search(self, query:str, k=3):
        """向量检索，返回格式化字符串（供 LLM prompt 用）。"""
        try:
            records = self.search_records(query, k=k)
            if not records:
                return "未找到相关信息。"
            lines = ["检索到的相关信息："]
            for idx, record in enumerate(records, 1):
                lines.append(f"{idx}.{record["content"]}")
            return "\n".join(lines)
        except Exception as exc:
            logger.error("检索失败: %s", exc)
            return f"检索过程中发生错误: {str(exc)}"
        
    def add_docs(self, documents: list[Document]) -> int:
        """批量添加文档到向量库。"""
        self.vectorstore.add_documents(documents)
        return len(documents)
    
    def ingest_text(self, text: str, source: str) -> int:
        """文本入库：分块 → 嵌入 → 存储。返回入库块数。"""
        #docs = self.text_splitter.create_documents([text], metadatas=[{"source": source}])
        chunks = self.text_splitter.split_text(text)
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"source":source}
            )
            docs.append(doc)
        return self.add_docs(docs)
    
    def ingest_paths(self, paths: Iterable[Path]) -> int:
        """文件入库：读取文件 → 分块 → 嵌入 → 存储。返回总入库块数。"""
        total = 0
        for path in paths:
            text = path.read_text(encoding="utf-8")
            total += self.ingest_text(text, source=str(path))
        return total
