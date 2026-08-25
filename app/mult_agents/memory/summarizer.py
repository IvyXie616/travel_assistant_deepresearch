"""对话摘要生成器（Phase 7.5 Step 2）。

两种模式：
- 规则版：确定性模板拼接，用于 dict content 的可检索文本生成
- LLM 版：完整对话压缩为 ≤200 字第三人称摘要，用于高质量语义检索
"""
import logging
from typing import Any, Dict, List, Optional
from langchain_community.chat_models import ChatTongyi

logger = logging.getLogger("travel_agents.memory.summarizer")

class EpisodeSummarizer:
    """摘要生成器。"""
    
    def __init__(self, api_key: str = "", model: str = "qwen-turbo"):
        """初始化 LLM（仅 LLM 版需要）。
        
        Args:
            api_key: DashScope API Key（为空时 LLM 版不可用，仅规则版可用）
            model: LLM 模型名（默认 qwen-turbo，成本低）
        """
        # TODO: 如果 api_key 非空，初始化 ChatTongyi（temperature=0.3）
        #       如果 api_key 为空，self.llm = None（降级为仅规则版可用）
        if api_key:
            self.llm = ChatTongyi(
                model=model,
                temperature=0.3,
                dashscope_api_key=api_key,)
        else:
            self.llm = None
            logger.warning("api_key为空，LLM摘要不可用，已降级为仅规则版可用")
    
    def summarize_by_rule(self, task_type: str, task_data: Dict[str, Any], 
                          outcome: Optional[str] = None) -> str:
        """规则版摘要：从 dict 提取关键字段，拼接为 ≤200 字文本。
        
        Args:
            task_type: 任务类型（如 "conversation"、"planning"）
            task_data: 任务数据（可能包含 query、destination、origin、travel_dates 等）
            outcome: 任务结果（可选）
        
        Returns:
            摘要文本（≤200 字，确定性）
        """
        parts = []
        
        # 1. 任务类型（必选）
        parts.append(f"任务类型：{task_type}")
        
        # 2. 用户需求（最重要，优先提取）
        query = task_data.get("query", "")
        if query:
            parts.append(f"用户需求：{query[:100]}")  # 截断到 100 字
        
        # 3. 目的地/出发地（次级过滤）
        destination = task_data.get("destination")
        origin = task_data.get("origin")
        if destination and origin:
            parts.append(f"行程：从{origin}到{destination}")
        elif destination:
            parts.append(f"目的地：{destination}")
        elif origin:
            parts.append(f"出发地：{origin}")
        
        # 4. 日期范围（补充信息）
        travel_dates = task_data.get("travel_dates")
        if travel_dates:
            if isinstance(travel_dates, list) and len(travel_dates) > 0:
                if len(travel_dates) == 1:
                    parts.append(f"日期：{travel_dates[0]}")
                else:
                    parts.append(f"日期：{travel_dates[0]}至{travel_dates[-1]}")
            elif isinstance(travel_dates, str):
                parts.append(f"日期：{travel_dates}")
        
        # 5. 任务结果（可选，截断到 50 字）
        if outcome:
            parts.append(f"结果：{outcome[:50]}")
        
        # 6. 拼接并控制总长度
        summary = "；".join(parts)
        
        # 最终截断保护：超过 200 字则截断
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        return summary
    
    def summarize_by_llm(self, messages: List[Dict[str, str]]) -> str:
        """LLM 版摘要：压缩完整对话为 ≤300 字第三人称摘要。
        
        Args:
            messages: [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}, ...]
        
        Returns:
            摘要文本（≤300 字，第三人称）
        
        实现提示：
        1. 如果 self.llm is None，抛出 ValueError("LLM 不可用，api_key 为空")
        2. 构造 prompt：
           - system: "请将以下旅行规划对话压缩为结构化摘要..."
           - human: 拼接对话（每条截断到 200 字）
        3. 调用 LLM，返回 content.strip()
        4. 截断保护：如果 len(summary) > 300，截断到 300 + "..."
        """
        if self.llm is None:
            raise ValueError("LLM 不可用，api_key 为空")
            
        else:
            msgs = [f"{msg['role']}：{msg['content'][:200]}" for msg in messages]
            msgs = '\n'.join(msgs)
            prompt = (
                "你是一个对话总结大师。请将以下旅行规划对话压缩为结构化摘要。\n\n"
                "输出格式要求：\n"
                "1. 使用字段标签格式，字段间用中文分号\"；\"分隔\n"
                "2. 必须包含的字段：对话主题、用户需求、AI执行结果\n"
                "3. 可选字段：出发地、目的地、旅行日期（若对话中未提及则不输出该字段）\n"
                "4. 总字数不超过 300 字\n"
                "5. 不要输出\"未提及\"或\"无\"等字样，缺失的字段直接省略\n\n"
                "示例输出：\n"
                "对话主题：旅行规划；用户需求：想去成都看自然风光，5天行程；AI执行结果：规划了北京→成都5天行程；行程：从北京到成都\n\n"
                f"对话内容：\n{msgs}"
            )
                
            response = self.llm.invoke(prompt)
            summ = str(response.content).strip()
            if len(summ) > 300:
                summ = summ[:300]+"..."
            return summ
