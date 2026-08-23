"""RAG 文档入库 CLI 脚本。

用法：
    cd travel_assistant
    python -m app.mult_agents.rag.ingest
"""
import logging
import sys
from pathlib import Path
# 将 travel_assistant 项目根加入 sys.path（修正：parents[3] 而非 parents[4]）
project_root = Path(__file__).resolve().parents[3]  # travel_assistant/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.mult_agents.rag.core import RAGSystem, RAGConfig
from app.mult_agents.config import AppConfig

# 知识库目录（相对于 travel_assistant/）
INPUT_PATH = Path("./data/knowledge")

def collect_paths(input_path: Path) -> list[Path]:
    """递归收集 .txt/.md/.markdown 文件。"""
    if input_path.is_file():
        return [input_path]
    paths = []
    for pat in ("*.txt", "*.md", "*.markdown"):
        paths.extend(sorted(input_path.rglob(pat)))
    return paths

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # 1. 加载配置
    config = AppConfig.from_env()

    # 2. 初始化 RAG 系统
    rag_cfg = RAGConfig(
        collection_name="travel_knowledge",
        embedding_model="text-embedding-v1",
        chunk_size=500,
        chunk_overlap=50,
    )
    rag = RAGSystem(api_key=config.api_key, config=rag_cfg)

    # 3. 扫描知识库目录
    input_path = INPUT_PATH.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"知识库目录不存在: {input_path}")

    paths = collect_paths(input_path)
    if not paths:
        raise ValueError(f"未找到可入库文件: {input_path}")

    print(f"找到 {len(paths)} 个文件:")
    for p in paths:
        print(f"  - {p.name}")

    # 4. 入库
    total_chunks = rag.ingest_paths(paths)
    print(f"\n入库完成 | 文件数={len(paths)} | chunk数={total_chunks} | collection=travel_knowledge")


if __name__ == "__main__":
    main()