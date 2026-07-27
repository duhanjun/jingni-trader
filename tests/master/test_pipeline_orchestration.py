"""MasterEngine.run() 与 __init__ 调度逻辑测试。

来源：合并原 test_system_smoke.py::TestRunApiFailure（1 用例）
与原 test_system_smoke.py::TestMasterEngineInit（1 用例），共 2 用例。

覆盖：
- run() API 在无 ctx/user_input 时的失败路径
- MasterEngine.__init__ 在干净环境下不抛异常（含 skill_sync 调用）
"""
from __future__ import annotations

import pytest


class TestRunApiFailure:
    def test_run_without_ctx_and_input(self):
        """run() 不传 ctx 也不传 user_input → 失败"""
        import engine
        result = engine.run()
        assert result["success"] is False
        assert "user_input" in result["error"] or "ctx" in result["error"]


class TestMasterEngineInit:
    def test_init_does_not_raise(self, monkeypatch, tmp_path):
        """MasterEngine.__init__ 在干净环境下不应抛异常"""
        monkeypatch.setenv("QUANT_WORK_DIR", str(tmp_path))
        import engine
        # 多次实例化不应累积状态
        m1 = engine.MasterEngine()
        m2 = engine.MasterEngine()
        assert m1 is not m2
        assert m1.ctx is None
        assert m2.ctx is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
