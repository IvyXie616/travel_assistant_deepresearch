"""
配置模块：统一管理应用配置。

设计理念：
    三层配置优先级：环境变量 > config.json > 代码默认值
    - 环境变量：敏感信息（API Key）和部署环境差异
    - config.json：非敏感的业务配置（迭代次数、模型选择等）
    - 默认值：兜底，确保永远有值可用

使用方式：
    config = AppConfig.from_file()                    # 从默认 config.json 加载
    config = AppConfig.from_file("custom.json")       # 从指定路径加载
    config = AppConfig.from_env()                     # 仅从环境变量加载
    config = config.with_overrides(user_id="user123") # 运行时覆盖部分字段
"""

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# Path(__file__) 是当前文件 config.py 的路径
# resolve().parents[2] 向上回溯 2 层：config.py -> mult_agents -> app -> travel_assistant
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class AppConfig:
    """
    应用全局配置（不可变）。

    使用 frozen=True 的原因：
        1. 配置加载后不应被运行时修改，防止意外篡改
        2. 可哈希，可作为字典 key 或缓存键
        3. 字段明确，所有配置项一目了然

    字段说明：
        api_key: DashScope API Key（通义千问）
        model: LLM 模型名称
        max_iterations: Reflection 反思循环的最大次数（防止无限循环）
        enable_memory: 是否启用记忆系统
        enable_rag: 是否启用 RAG 检索
        memory_db_path: SQLite 数据库路径（长期记忆持久化）
        chroma_persist_dir: ChromaDB 持久化目录（向量存储）
        chroma_collection: ChromaDB collection 名称（记忆系统用）
        rag_collection: RAG 知识库 collection 名称（与记忆隔离）
        mcp_config_path: MCP Server 配置文件路径
        thread_id: 会话线程标识（短期记忆隔离）
        user_id: 用户标识（长期记忆隔离）
    """

    api_key: str
    model: str
    max_iterations: int
    enable_memory: bool
    enable_rag: bool
    memory_db_path: str
    chroma_persist_dir: str
    chroma_collection: str
    rag_collection: str
    mcp_config_path: str
    thread_id: str
    user_id: str

    def with_overrides(self, **kwargs) -> "AppConfig":
        """
        创建配置副本并覆盖指定字段。

        为何用 dataclasses.replace 而非直接修改？
            因为 frozen=True 不可直接赋值，replace 会创建新实例。
            这保证了原配置不变，新配置是独立副本。

        Args:
            **kwargs: 要覆盖的字段，只传需要改的，None 值会被忽略

        Returns:
            新的 AppConfig 实例
        """
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **cleaned)

    @staticmethod
    def _default_config_path() -> Path:
        """默认配置文件路径：项目根目录下的 config.json"""
        return _PROJECT_ROOT / "config.json"

    @staticmethod
    def _resolve_str(data: dict, field: str, env_key: str, default: str = "") -> str:
        """
        按优先级解析字符串配置：环境变量 > 配置文件 > 默认值。

        Args:
            data: config.json 解析出的字典
            field: config.json 中的字段名
            env_key: 环境变量名
            default: 兜底默认值

        Returns:
            解析后的字符串值
        """
        # 1. 优先读环境变量
        env_value = os.getenv(env_key)
        if env_value is not None and str(env_value).strip() != "":
            return str(env_value).strip()
        # 2. 其次读配置文件
        file_value = data.get(field)
        if file_value is not None and str(file_value).strip() != "":
            return str(file_value).strip()
        # 3. 兜底默认值
        return default

    @staticmethod
    def _resolve_int(data: dict, field: str, env_key: str, default: int) -> int:
        """解析整数配置（复用 _resolve_str 后转为 int）"""
        value = AppConfig._resolve_str(data, field, env_key, str(default))
        return int(value)

    @staticmethod
    def _resolve_bool(data: dict, field: str, env_key: str, default: bool) -> bool:
        """解析布尔配置（字符串 "true"/"false" 转为 bool）"""
        value = AppConfig._resolve_str(data, field, env_key, "true" if default else "false")
        return value.lower() == "true"

    @staticmethod
    def from_file(path: str | Path | None = None) -> "AppConfig":
        """
        从 config.json 加载配置（环境变量优先覆盖）。

        Args:
            path: 配置文件路径，None 则使用默认路径

        Returns:
            AppConfig 实例

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 缺少必需的 api_key
        """
        config_path = Path(path) if path else AppConfig._default_config_path()
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("配置文件格式错误：根元素必须是 JSON 对象")

        # 解析各字段
        api_key = AppConfig._resolve_str(data, "api_key", "DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError(
                f"缺少 DASHSCOPE_API_KEY 配置。请在 {config_path} 中填写 api_key，"
                "或设置环境变量 DASHSCOPE_API_KEY，或在 .env 文件中配置。"
            )

        return AppConfig(
            api_key=api_key,
            model=AppConfig._resolve_str(data, "model", "MODEL", "qwen-turbo"),
            max_iterations=AppConfig._resolve_int(data, "max_iterations", "MAX_ITERATIONS", 3),
            enable_memory=AppConfig._resolve_bool(data, "enable_memory", "ENABLE_MEMORY", True),
            enable_rag=AppConfig._resolve_bool(data, "enable_rag", "ENABLE_RAG", True),
            memory_db_path=AppConfig._resolve_str(
                data, "memory_db_path", "MEMORY_DB_PATH", "./app/data/memory.db"
            ),
            chroma_persist_dir=AppConfig._resolve_str(
                data, "chroma_persist_dir", "CHROMA_PERSIST_DIR", "./app/data/chroma"
            ),
            chroma_collection=AppConfig._resolve_str(
                data, "chroma_collection", "CHROMA_COLLECTION", "travel_memory"
            ),
            rag_collection=AppConfig._resolve_str(
                data, "rag_collection", "RAG_COLLECTION", "travel_knowledge"
            ),
            mcp_config_path=AppConfig._resolve_str(
                data, "mcp_config_path", "MCP_CONFIG_PATH", "./servers_config.json"
            ),
            thread_id=AppConfig._resolve_str(data, "thread_id", "THREAD_ID", "default"),
            user_id=AppConfig._resolve_str(data, "user_id", "USER_ID", "default_user"),
        )

    @staticmethod
    def from_env() -> "AppConfig":
        """
        仅从环境变量加载配置（不读取 config.json）。

        适用场景：Docker/K8s 部署，所有配置通过环境变量注入。

        Returns:
            AppConfig 实例

        Raises:
            ValueError: 缺少必需的 DASHSCOPE_API_KEY
        """
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY 环境变量")

        return AppConfig(
            api_key=api_key,
            model=os.getenv("MODEL", "qwen-turbo"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
            enable_memory=os.getenv("ENABLE_MEMORY", "true").lower() == "true",
            enable_rag=os.getenv("ENABLE_RAG", "true").lower() == "true",
            memory_db_path=os.getenv("MEMORY_DB_PATH", "./app/data/memory.db"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./app/data/chroma"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "travel_memory"),
            rag_collection=os.getenv("RAG_COLLECTION", "travel_knowledge"),
            mcp_config_path=os.getenv("MCP_CONFIG_PATH", "./servers_config.json"),
            thread_id=os.getenv("THREAD_ID", "default"),
            user_id=os.getenv("USER_ID", "default_user"),
        )
