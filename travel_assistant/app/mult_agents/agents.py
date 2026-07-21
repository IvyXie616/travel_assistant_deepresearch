from app.mult_agents.config import AppConfig
from langchain_community.chat_models import ChatTongyi

def build_agent(
    model: str,
    api_key: str,
    prompt_key: str,
    temperature: float = 0.7,
) -> ChatTongyi:
    """
    创建单个 Agent（LLM 实例）。
    
    v2 修订：不传 tools 参数（节点编排模式下，工具由节点代码调用）
    """