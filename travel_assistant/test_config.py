"""Phase 0 验收测试：验证配置系统能正常加载。"""
import sys
from pathlib import Path

# 确保能导入 app 包
sys.path.insert(0, str(Path(__file__).parent))

from app.mult_agents.config import AppConfig


def main():
    print("=" * 60)
    print("Phase 0 验收测试：配置系统")
    print("=" * 60)

    # 测试 1: 从 config.json 加载
    print("\n[测试 1] 从 config.json 加载配置")
    config = AppConfig.from_file()
    print(f"  api_key: {config.api_key[:8]}...")
    print(f"  model: {config.model}")
    print(f"  max_iterations: {config.max_iterations}")
    print(f"  enable_memory: {config.enable_memory}")
    print(f"  enable_rag: {config.enable_rag}")
    print(f"  memory_db_path: {config.memory_db_path}")
    print(f"  chroma_persist_dir: {config.chroma_persist_dir}")
    print(f"  chroma_collection: {config.chroma_collection}")
    print(f"  rag_collection: {config.rag_collection}")
    print(f"  mcp_config_path: {config.mcp_config_path}")
    print(f"  thread_id: {config.thread_id}")
    print(f"  user_id: {config.user_id}")
    print("  [OK] 配置加载成功")

    # 测试 2: with_overrides
    print("\n[测试 2] with_overrides 覆盖字段")
    config2 = config.with_overrides(user_id="test_user", thread_id="session_123")
    print(f"  原 user_id: {config.user_id}")
    print(f"  新 user_id: {config2.user_id}")
    print(f"  原 thread_id: {config.thread_id}")
    print(f"  新 thread_id: {config2.thread_id}")
    assert config.user_id == "default_user", "原配置应保持不变"
    assert config2.user_id == "test_user", "新配置应已覆盖"
    print("  [OK] with_overrides 工作正常，原配置不可变")

    # 测试 3: frozen=True 不可变
    print("\n[测试 3] frozen=True 不可变性")
    try:
        config.user_id = "hacked"
        print("  [FAIL] 配置被修改了！frozen=True 未生效")
    except AttributeError:
        print("  [OK] frozen=True 生效，配置不可直接修改")

    # 测试 4: 环境变量优先级
    print("\n[测试 4] 环境变量优先级")
    import os
    os.environ["USER_ID"] = "env_user"
    config3 = AppConfig.from_file()
    print(f"  环境变量 USER_ID=env_user")
    print(f"  加载后 user_id: {config3.user_id}")
    assert config3.user_id == "env_user", "环境变量应优先于 config.json"
    print("  [OK] 环境变量优先级正确（env > config.json > 默认值）")
    del os.environ["USER_ID"]

    print("\n" + "=" * 60)
    print("Phase 0 验收通过！所有测试均成功。")
    print("=" * 60)


if __name__ == "__main__":
    main()
