"""pytest 统一路径处理与依赖 mock。

将项目根目录加入 sys.path，使 tests/ 下的测试脚本能直接
`import engine`（主调度器）或通过相对路径解析 `scripts` 包。

同时 mock 掉 factor-engine 依赖的重量级第三方库（sklearn/talib/pandas_ta）。

factor-engine/engine.py 与主调度器 engine.py 同名且都依赖 `scripts.config`，
为避免 import 冲突，用 importlib 显式加载 factor-engine/engine.py 为
`factor_engine_engine` 模块。加载时临时让 factor-engine 的 scripts 包接管，
加载完后恢复原始 scripts 缓存，使后续主调度器的 import 不受影响。

关键隔离：MasterEngine.execute_stage 会在运行期把 sys.modules['scripts']
切换为各子技能的 scripts 包（如 factor-engine/reports-engine），导致
后续 `from scripts.context import Context` 解析到错误的包。下面通过 autouse
fixture 在每个测试前后保存/恢复 scripts 相关缓存，杜绝测试间污染。
"""
import os
import sys
import copy
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
for _mod_name in ("sklearn", "sklearn.linear_model", "talib", "pandas_ta"):
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
