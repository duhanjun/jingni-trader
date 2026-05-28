"""
数据引擎测试
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import DataEngine, run


class TestDataEngine:
    """测试 DataEngine 类"""

    def test_init(self):
        """测试引擎初始化"""
        engine = DataEngine()
        assert engine is not None
        assert engine.provider is not None

    def test_run_basic(self):
        """测试基本运行"""
        from scripts.context import Context
        ctx = Context(
            task_id="test_001",
            user_intent="测试",
            current_stage="IDLE"
        )
        ctx.stock_pool = []
        ctx.start_date = "2024-01-01"
        ctx.end_date = "2024-01-10"

        result = run(ctx)
        assert "success" in result
        assert "artifact_path" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
