"""reports-engine L1 契约测试。

验证 reports-engine.run(ctx) 的接口契约：
- 无任何上游产物 → success=True（生成空报告）或 success=False（不抛异常）
- result dict 含必需字段（success/artifact_path/metadata/error）

reports-engine 的设计是"尽力而为"——即使没有上游产物，也应生成一份基础报告，
不抛异常。本测试验证这一关键契约。
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_ENGINE_DIR = os.path.join(ROOT, "skills", "reports-engine")
REPORTS_ENGINE_PATH = os.path.join(REPORTS_ENGINE_DIR, "engine.py")

# 预加载主项目的 Context 类，避免后续 scripts 包切换导致找不到
_CONTEXT_MODULE = None


def _get_context_class():
    """获取 Context 类（处理 scripts 包切换问题）"""
    global _CONTEXT_MODULE
    if _CONTEXT_MODULE is not None:
        return _CONTEXT_MODULE

    # 直接加载 context.py 避免 scripts 包名冲突
    context_path = os.path.join(ROOT, "scripts", "context.py")
    if os.path.exists(context_path):
        spec = ilu.spec_from_file_location("jingni_context", context_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["jingni_context"] = mod
        spec.loader.exec_module(mod)
        _CONTEXT_MODULE = mod.Context
        return mod.Context

    raise ImportError("无法加载 Context 类")


def _load_reports_engine_module():
    """显式加载 reports-engine/engine.py 为独立模块。"""
    # 清理旧的 scripts 包，为 reports-engine 的 scripts 包腾出空间
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(REPORTS_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    for _m in ("talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    spec = ilu.spec_from_file_location("reports_engine_engine", REPORTS_ENGINE_PATH)
    mod = ilu.module_from_spec(spec)
    sys.modules["reports_engine_engine"] = mod
    spec.loader.exec_module(mod)
    # 不清理 scripts 包：reports-engine 的 run() 函数依赖 scripts.template_engine 等子模块
    return mod


def _make_ctx(stock_pool=None):
    """构造 reports-engine 标准输入 Context"""
    Context = _get_context_class()
    return Context(
        task_id="test_report",
        stock_pool=stock_pool or ["000001.SZ", "600000.SH"],
        start_date="2024-01-01",
        end_date="2024-06-30",
    )


@pytest.mark.skill_reports_engine
@pytest.mark.contract
class TestReportsEngineRunContract:
    """验证 reports-engine.run(ctx) 接口契约。"""

    def test_does_not_raise_with_empty_context(self, monkeypatch, tmp_path):
        """无任何上游产物 → 不抛异常（生成空报告或返回失败，但不应 crash）"""
        monkeypatch.setenv("QUANT_REPORT_DIR", str(tmp_path))

        reports_mod = _load_reports_engine_module()
        ctx = _make_ctx()
        # 不更新任何 artifacts

        # 关键契约：不应抛异常
        result = reports_mod.run(ctx)

        assert isinstance(result, dict)
        assert "success" in result

    def test_result_has_required_fields(self, monkeypatch, tmp_path):
        """result dict 必含 success/artifact_path/metadata/error 四个字段"""
        monkeypatch.setenv("QUANT_REPORT_DIR", str(tmp_path))

        reports_mod = _load_reports_engine_module()
        ctx = _make_ctx()

        result = reports_mod.run(ctx)

        for field in ("success", "artifact_path", "metadata", "error"):
            assert field in result, f"result 缺少必需字段: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
