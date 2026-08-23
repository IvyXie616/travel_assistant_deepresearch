from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-a987248ddab546d4ad77467f089a2ad8"
load_dotenv()  # 如果使用 .env 文件；若用 Colab 密钥则无需此句

def get_llm(MODEL):
    api = os.getenv("DASHSCOPE_API_KEY")
    print("DASHSCOPE_API_KEY 是否存在:", bool(api))
    if not api:
        print("环境变量中没有API-key，请你写入")
        return

    llm = ChatTongyi(
        model=MODEL,
        temperature=0,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    return llm

LLM = get_llm("qwen-turbo")