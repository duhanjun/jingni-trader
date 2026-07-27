"""7 个子 Skill 模块可加载性与 run() 接口测试。

来源：原 test_system_smoke.py::test_subskill_engine_loadable（参数化 7 用例）。

覆盖：
- 7 个子 Skill 的 engine.py 都可被独立加载
- 每个都暴露可调用的 run() 函数
- mock 掉 sklearn/talib 等重量级第三方依赖
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SUBSKILL_ENGINES = [
    ("data-engine", "skills/data-engine/engine.py"),
    ("factor-engine", "skills/factor-engine/engine.py"),
    ("strategy-model-engine", "skills/strategy-model-engine/engine.py"),
    ("backtest-engine", "skills/backtest-engine/engine.py"),
    ("portfolio-risk-engine", "skills/portfolio-risk-engine/engine.py"),
    ("execution-monitor-engine", "skills/execution-monitor-engine/engine.py"),
    ("reports-engine", "skills/reports-engine/engine.py"),
]


@pytest.mark.parametrize("name,rel_path", SUBSKILL_ENGINES)
def test_subskill_engine_loadable(name, rel_path):
    """每个子 Skill 的 engine.py 都可被独立加载，且暴露 run() 接口"""
    engine_path = os.path.join(ROOT, rel_path)
    assert os.path.exists(engine_path), f"{name}: engine.py 不存在: {engine_path}"

    # 为每个子 Skill 模拟切换 scripts 包（参考 engine.py._register_subskill_scripts）
    scripts_dir = os.path.join(os.path.dirname(engine_path), "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")

    # 先清掉旧 scripts 缓存
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    try:
        if os.path.exists(init_py):
            spec = ilu.spec_from_file_location(
                "scripts", init_py,
                submodule_search_locations=[scripts_dir],
            )
            pkg = ilu.module_from_spec(spec)
            sys.modules["scripts"] = pkg
            spec.loader.exec_module(pkg)

        # mock 掉重量级第三方库（部分子 Skill 在 import 时尝试 import）
        # sklearn 需要拆成多个子模块（strategy-model-engine 顶层 from sklearn.ensemble import ...）
        for _m in (
            "sklearn", "sklearn.linear_model", "sklearn.ensemble",
            "sklearn.model_selection", "talib", "pandas_ta",
        ):
            if _m not in sys.modules:
                sys.modules[_m] = mock.MagicMock()

        mod_name = f"_test_subskill_{name.replace('-', '_')}"
        spec = ilu.spec_from_file_location(mod_name, engine_path)
        mod = ilu.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

        # 验证 run() 接口存在
        assert hasattr(mod, "run"), f"{name}: engine.py 缺少 run() 函数"
        assert callable(mod.run), f"{name}: run 不可调用"
    finally:
        # 清理：把本测试注入的临时模块移除
        sys.modules.pop(mod_name, None)
        # 恢复 scripts 缓存
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
