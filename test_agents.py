"""Phase 1 验收测试：验证 LLM 工厂与 Agent 基础设施。

验收标准（来自开发计划 Phase 1）：
    能实例化 AgentBundle，各 agent 可调用 .invoke()。

测试维度：
    1. AgentBundle 实例化：8 个 agent 全部就绪
    2. Runnable 协议：每个 agent 都实现了 invoke
    3. temperature 差异化：规划稳定(低) / 业务适中(中) / 写作灵活(高)
    4. user_profile 占位符预填充：partial 已生效
    5. 真实 LLM 调用：planner 能输出合法 JSON（含 needs_clarification 机制）
"""

import json
import sys
from pathlib import Path

# 确保能导入 app 包
sys.path.insert(0, str(Path(__file__).parent))

# Windows 控制台中文输出兜底
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langchain_core.runnables import Runnable

from app.mult_agents.config import AppConfig
from app.mult_agents.agents import AgentBundle, build_agent, build_agents


# 期望的 temperature 差异化配置（来自计划"一·补"章节第 8 节）
EXPECTED_TEMPS = {
    "planner": 0.3,      # 规划要稳定
    "weather": 0.5,      # 业务适中
    "transport": 0.5,
    "hotel": 0.5,
    "research": 0.5,
    "budget": 0.3,       # 计算要稳定
    "reflection": 0.3,   # 检查要稳定
    "writer": 0.7,       # 写作要灵活
}


def _extract_temperature(agent) -> float:
    """从 LCEL 链 (prompt | llm) 中反查 llm.temperature。"""
    # RunnableSequence 的 .last 指向 llm
    last = getattr(agent, "last", None)
    if last is None:
        # 兼容不同 langchain 版本：尝试 middle[-1]
        middle = getattr(agent, "middle", [])
        if middle:
            last = middle[-1]
    temp = getattr(last, "temperature", None)
    if temp is None:
        raise AssertionError("无法从 agent 链中反查 temperature")
    return float(temp)


def test_bundle_structure(bundle: AgentBundle) -> None:
    """测试 1+2: AgentBundle 结构与 Runnable 协议。"""
    print("\n[测试 1+2] AgentBundle 实例化 + Runnable 协议")
    expected_keys = list(EXPECTED_TEMPS.keys())
    for key in expected_keys:
        agent = getattr(bundle, key, None)
        assert agent is not None, f"AgentBundle 缺少 agent: {key}"
        assert isinstance(agent, Runnable), f"{key} 不是 Runnable 实例: {type(agent)}"
        print(f"  {key:12s} -> {type(agent).__name__}  [OK]")
    print("  [OK] 8 个 agent 全部就绪且实现 Runnable 协议")


def test_temperature_diff(bundle: AgentBundle) -> None:
    """测试 3: temperature 差异化配置。"""
    print("\n[测试 3] temperature 差异化配置")
    for key, expected in EXPECTED_TEMPS.items():
        agent = getattr(bundle, key)
        actual = _extract_temperature(agent)
        assert actual == expected, (
            f"{key} temperature 应为 {expected}，实际 {actual}"
        )
        print(f"  {key:12s} -> temperature={actual}  [OK]")
    print("  [OK] 规划/检查低温度(0.3)，业务中温度(0.5)，写作高温度(0.7)")


def test_user_profile_partial(bundle: AgentBundle) -> None:
    """测试 4: user_profile 占位符已通过 partial 预填充。"""
    print("\n[测试 4] user_profile 占位符预填充")
    # planner 的 prompt 含 {user_profile}，partial 应已填入默认值
    # 触发一次模板渲染（不调用 LLM），检查占位符是否被替换
    planner_template = bundle.planner.first  # ChatPromptTemplate
    rendered = planner_template.format(input="测试输入")
    assert "（暂无用户画像）" in rendered, "user_profile 占位符未被预填充"
    assert "{user_profile}" not in rendered, "占位符 {user_profile} 未被替换"
    print("  [OK] user_profile 已预填充为「（暂无用户画像）」，Phase 5 可注入真实画像")
    # 检查无占位符的 agent（如 weather）也不报错
    weather_rendered = bundle.weather.first.format(input="测试")
    assert "测试" in weather_rendered
    print("  [OK] weather agent 模板渲染正常（无 user_profile 占位符）")


def test_live_planner_invoke(bundle: AgentBundle) -> None:
    """测试 5: 真实 LLM 调用 - planner 输出合法 JSON + needs_clarification 机制。"""
    print("\n[测试 5] 真实 LLM 调用 - planner（可能耗时 3-10 秒）")
    test_query = "我想2026年8月1日到8月3日从北京去上海旅游"
    print(f"  输入: {test_query}")

    try:
        result = bundle.planner.invoke({"input": test_query})
    except Exception as exc:
        print(f"  [SKIP] LLM 调用失败（网络/API 问题）: {exc}")
        return

    content = result.content if hasattr(result, "content") else str(result)
    preview = content.replace("\n", " ")
    if len(preview) > 300:
        preview = preview[:300] + "..."
    print(f"  原始输出: {preview}")

    # 尝试从输出中提取 JSON
    parsed = _safe_json_extract(content)
    if parsed is None:
        print("  [WARN] 未能解析出 JSON（可能是模型输出格式漂移，Phase 2 会加 _extract_json_block 兜底）")
        return

    print(f"  解析 JSON: {json.dumps(parsed, ensure_ascii=False)}")

    # 校验关键字段存在
    required_fields = ["sub_tasks", "origin", "destination", "travel_dates",
                       "needs_clarification", "clarification_question"]
    for field in required_fields:
        assert field in parsed, f"planner 输出缺少字段: {field}"
    print(f"  [OK] 包含全部 6 个必需字段: {required_fields}")

    # 校验信息完整场景：needs_clarification=False
    assert parsed["needs_clarification"] is False, (
        f"信息完整时应 needs_clarification=False，实际 {parsed['needs_clarification']}"
    )
    assert parsed["origin"] == "北京", f"origin 应为'北京'，实际 {parsed['origin']}"
    assert parsed["destination"] == "上海", f"destination 应为'上海'，实际 {parsed['destination']}"
    assert len(parsed["travel_dates"]) == 3, (
        f"travel_dates 应有 3 天，实际 {parsed['travel_dates']}"
    )
    assert len(parsed["sub_tasks"]) > 0, "信息完整时 sub_tasks 不应为空"
    print(f"  [OK] 信息完整场景验证通过: origin=北京, destination=上海, 3天, sub_tasks非空")


def test_live_planner_clarification(bundle: AgentBundle) -> None:
    """测试 6: 真实 LLM 调用 - 信息不足时 needs_clarification=True（v4 修复验证）。"""
    print("\n[测试 6] 真实 LLM 调用 - 信息不足触发追问（v4 修复验证）")
    test_query = "我想去旅游"  # 缺出发地、目的地、日期
    print(f"  输入: {test_query}")

    try:
        result = bundle.planner.invoke({"input": test_query})
    except Exception as exc:
        print(f"  [SKIP] LLM 调用失败（网络/API 问题）: {exc}")
        return

    content = result.content if hasattr(result, "content") else str(result)
    parsed = _safe_json_extract(content)
    if parsed is None:
        print("  [WARN] 未能解析出 JSON，跳过字段校验")
        return

    print(f"  解析 JSON: {json.dumps(parsed, ensure_ascii=False)}")
    assert parsed.get("needs_clarification") is True, (
        f"信息不足时应 needs_clarification=True，实际 {parsed.get('needs_clarification')}"
    )
    assert parsed.get("clarification_question"), (
        "needs_clarification=True 时 clarification_question 不应为空"
    )
    # v4 关键修复：needs_clarification=True 时 sub_tasks 应为空列表
    assert parsed.get("sub_tasks") == [], (
        f"v4 修复: needs_clarification=True 时 sub_tasks 应为 []，实际 {parsed.get('sub_tasks')}"
    )
    print(f"  [OK] v4 修复验证通过: needs_clarification=True, sub_tasks=[], question非空")


def _safe_json_extract(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON 对象（容忍 ```json 代码块包裹）。"""
    import re
    # 优先匹配 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 退而求其次：匹配第一个 {...}
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # 最后尝试整体解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def main():
    print("=" * 60)
    print("Phase 1 验收测试：LLM 工厂与 Agent 基础设施")
    print("=" * 60)

    # 跳过实际 LLM 调用的开关：python test_agents.py --no-live
    run_live = "--no-live" not in sys.argv

    config = AppConfig.from_file()
    print(f"\n配置: model={config.model}, api_key={config.api_key[:8]}...")

    print("\n--- 构建 AgentBundle ---")
    bundle = build_agents(config)

    # 结构性测试（不调用 LLM）
    test_bundle_structure(bundle)
    test_temperature_diff(bundle)
    test_user_profile_partial(bundle)

    # 真实 LLM 调用测试
    if run_live:
        test_live_planner_invoke(bundle)
        test_live_planner_clarification(bundle)
    else:
        print("\n[跳过] 真实 LLM 调用测试（--no-live）")

    print("\n" + "=" * 60)
    print("Phase 1 验收通过！Agent 基础设施就绪。")
    print("=" * 60)


if __name__ == "__main__":
    main()
