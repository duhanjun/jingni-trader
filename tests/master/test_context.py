"""Context 模块核心 API 测试。

来源：原 test_system_smoke.py::TestContextApi（2 用例）。

覆盖：
- Context to_json / from_json 往返保持字段一致
- get_artifact 未注册返回 None
"""
from __future__ import annotations

import pytest


class TestContextApi:
    def test_context_roundtrip_json(self):
        """Context to_json / from_json 往返保持字段一致"""
        from scripts.context import Context
        ctx = Context(
            task_id="t1",
            user_intent="测试",
            target_stages=["DATA", "REPORT"],
            stock_pool=["000001.SZ"],
        )
        ctx.update_artifact("DATA", "/tmp/x.parquet")
        ctx.add_error("test error")

        s = ctx.to_json()
        ctx2 = Context.from_json(s)
        assert ctx2.task_id == "t1"
        assert ctx2.target_stages == ["DATA", "REPORT"]
        assert ctx2.stock_pool == ["000001.SZ"]
        assert ctx2.artifacts == {"DATA": "/tmp/x.parquet"}
        assert ctx2.errors == ["test error"]

    def test_context_get_artifact_missing(self):
        """get_artifact 未注册 → None"""
        from scripts.context import Context
        ctx = Context()
        assert ctx.get_artifact("NOT_EXIST") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
