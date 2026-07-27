"""execution-monitor-engine L1 契约测试。

验证 execution-monitor-engine.run(ctx) 的接口契约：
- 上游 PORTFOLIO 产物缺失 → success=False + error 含"目标权重文件不存在"
- result dict 含必需字段（success/artifact_path/metadata/error）

注：成功路径涉及实盘下单 API，不在此处测试（避免真实下单风险）。
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXECUTION_ENGINE_DIR = os.path.join(ROOT, "skills", "execution-monitor-engine")
EXECUTION_ENGINE_PATH = os.path.join(EXECUTION_ENGINE_DIR, "engine.py")


def _load_execution_engine_module():
    """显式加载 execution-monitor-engine/engine.py 为独立模块。"""
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(EXECUTION_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    for _m in ("xtquant", "gm", "talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    try:
        spec = ilu.spec_from_file_location("execution_monitor_engine_engine", EXECUTION_ENGINE_PATH)
        mod = ilu.module_from_spec(spec)
        sys.modules["execution_monitor_engine_engine"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


def _make_ctx(stock_pool=None):
    """构造 execution-monitor-engine 标准输入 Context"""
    from scripts.context import Context
    return Context(
        task_id="test_execution",
        stock_pool=stock_pool or ["000001.SZ", "600000.SH"],
        start_date="2024-01-01",
        end_date="2024-06-30",
    )


@pytest.mark.skill_execution_monitor_engine
@pytest.mark.contract
class TestExecutionMonitorEngineRunContract:
    """验证 execution-monitor-engine.run(ctx) 接口契约。"""

    def test_returns_failure_when_portfolio_artifact_missing(self, monkeypatch, tmp_path):
        """PORTFOLIO 产物未注册 → success=False + error 含"目标权重文件不存在" """
        monkeypatch.setenv("QUANT_EXECUTION_DIR", str(tmp_path))

        execution_mod = _load_execution_engine_module()
        ctx = _make_ctx()
        # 不更新 PORTFOLIO 产物

        result = execution_mod.run(ctx)

        assert result["success"] is False
        assert "目标权重文件不存在" in result["error"]

    def test_returns_failure_when_portfolio_path_not_exists(self, monkeypatch, tmp_path):
        """PORTFOLIO 路径注册但文件不存在 → success=False"""
        monkeypatch.setenv("QUANT_EXECUTION_DIR", str(tmp_path))

        execution_mod = _load_execution_engine_module()
        ctx = _make_ctx()
        ctx.update_artifact("PORTFOLIO", "/nonexistent/portfolio.json")

        result = execution_mod.run(ctx)

        assert result["success"] is False

    def test_result_has_required_fields(self, monkeypatch, tmp_path):
        """result dict 必含 success/artifact_path/metadata/error 四个字段"""
        monkeypatch.setenv("QUANT_EXECUTION_DIR", str(tmp_path))

        execution_mod = _load_execution_engine_module()
        ctx = _make_ctx()
        ctx.update_artifact("PORTFOLIO", "/nonexistent/portfolio.json")

        result = execution_mod.run(ctx)

        for field in ("success", "artifact_path", "metadata", "error"):
            assert field in result, f"result 缺少必需字段: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
