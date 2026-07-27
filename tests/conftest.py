"""pytest 全局配置与共享 fixture。

本文件位于 tests/ 根目录，pytest 启动时会自动加载。

职责：
1. 把项目根加入 sys.path，使 `import engine` / `from scripts.*` 可解析
2. 把 jingni-datafeed scripts 目录加入 sys.path（JingniClient 等模块依赖）
3. 把 tests/fixtures/ 目录加入 sys.path，使各子目录测试可 `import synthetic_data` 等
4. 在每个测试前后保存/恢复 sys.modules['scripts'] 缓存，防止子技能切换 scripts 包时污染
5. mock 掉重量级第三方依赖（sklearn 子模块 / talib / pandas_ta），使测试无需真实安装
6. 显式加载 factor-engine/engine.py 为 `factor_engine_engine` 模块，避免与主调度器 engine.py 同名冲突

目录结构约定（按子 Skill 边界组织）：
    tests/
    ├── conftest.py            ← 本文件（全局）
    ├── pytest.ini             ← 标记配置
    ├── fixtures/              ← 共享测试数据与构造器
    │   ├── synthetic_data.py
    │   ├── mock_datafeed.py
    │   └── sample_contexts.py
    ├── master/                ← 主调度器测试
    ├── data_engine/
    ├── factor_engine/
    ├── strategy_model_engine/
    ├── backtest_engine/
    ├── portfolio_risk_engine/
    ├── execution_monitor_engine/
    ├── reports_engine/
    └── integration/          ← 跨 skill 全链路
"""
import os
import sys
import importlib.util
from unittest import mock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# jingni-datafeed scripts 目录（供 JingniClient 等模块 import）
DATAFEED_SCRIPTS = os.path.join(ROOT, "skills", "jingni-datafeed", "scripts")
if DATAFEED_SCRIPTS not in sys.path:
    sys.path.insert(0, DATAFEED_SCRIPTS)

# tests/fixtures/ 目录加入 sys.path，使各子目录测试可 `import synthetic_data` 等
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
if FIXTURES_DIR not in sys.path:
    sys.path.insert(0, FIXTURES_DIR)

# 主调度器的 scripts 目录，用于在每个测试前重置 sys.modules['scripts']
MASTER_SCRIPTS_DIR = os.path.join(ROOT, "scripts")


def _reset_master_scripts() -> None:
    """把主调度器的 scripts 包重新注册为 sys.modules['scripts']。

    MasterEngine.execute_stage 在切换子技能时会覆盖该槽位，这里在测试
    入口恢复为主 scripts 包，确保 `from scripts.context import Context`
    总能解析到 jingni-trader/scripts/context.py。
    """
    # 清掉所有 scripts.* 缓存
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)
    # 用主 scripts 目录重新加载 scripts 包
    init_py = os.path.join(MASTER_SCRIPTS_DIR, "__init__.py")
    if os.path.exists(init_py):
        spec = importlib.util.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[MASTER_SCRIPTS_DIR],
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)


@pytest.fixture(autouse=True)
def _isolate_scripts_module():
    """每个测试前后保存/恢复 sys.modules 中 scripts 相关缓存。

    防止某个测试调用的 MasterEngine.run_pipeline 污染后续测试。
    """
    saved = {
        k: sys.modules.get(k)
        for k in list(sys.modules.keys())
        if k == "scripts" or k.startswith("scripts.")
    }
    # 测试开始前先重置为主 scripts 包
    _reset_master_scripts()
    try:
        yield
    finally:
        # 测试结束后恢复到测试前的状态
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v

# Mock 重量级第三方依赖，使 factor-engine.engine 可在无 sklearn/talib 环境下 import
# strategy-model-engine 顶层有 `from sklearn.ensemble import ...`，需拆成多个子模块
for _mod_name in (
    "sklearn", "sklearn.linear_model", "sklearn.ensemble",
    "sklearn.model_selection", "talib", "pandas_ta",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = mock.MagicMock()

# factor-engine 目录
FACTOR_ENGINE_DIR = os.path.join(ROOT, "skills", "factor-engine")

# 显式加载 factor-engine/engine.py 为独立模块（避免与主调度器 engine.py 同名冲突）
# 关键：factor-engine 和主调度器都有 `scripts/config.py`，但内容不同。
# 加载 factor_engine_engine 时临时让 factor-engine 的 scripts 包接管，
# 加载完后恢复原始缓存，使后续 `import engine`（主调度器）的
# `from scripts.config import WORK_DIR` 能正确解析到 jingni-trader/scripts/config.py
if "factor_engine_engine" not in sys.modules:
    # 保存可能已存在的 scripts 相关缓存
    _saved = {k: sys.modules.get(k) for k in (
        "scripts", "scripts.config", "scripts.context", "scripts.archive"
    )}

    # 临时把 FACTOR_ENGINE_DIR 放到 sys.path 最前面
    sys.path.insert(0, FACTOR_ENGINE_DIR)

    _fe_engine_path = os.path.join(FACTOR_ENGINE_DIR, "engine.py")
    _spec = importlib.util.spec_from_file_location("factor_engine_engine", _fe_engine_path)
    _fe_mod = importlib.util.module_from_spec(_spec)
    sys.modules["factor_engine_engine"] = _fe_mod
    _spec.loader.exec_module(_fe_mod)

    # 加载完后：恢复原始 scripts 缓存（如果原来有就恢复，没有就清除）
    for _name, _orig in _saved.items():
        if _orig is not None:
            sys.modules[_name] = _orig
        else:
            sys.modules.pop(_name, None)

    # 从 sys.path 彻底移除 FACTOR_ENGINE_DIR
    # （expression/factors 等内部模块已加载到 sys.modules，不再需要 sys.path 查找）
    while FACTOR_ENGINE_DIR in sys.path:
        sys.path.remove(FACTOR_ENGINE_DIR)
