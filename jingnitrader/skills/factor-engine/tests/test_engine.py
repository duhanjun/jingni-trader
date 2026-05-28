"""
因子引擎测试
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import FactorEngine, run


class TestFactorEngine:
    """测试 FactorEngine 类"""

    def test_init(self):
        """测试引擎初始化"""
        engine = FactorEngine()
        assert engine is not None
        assert engine.calculator is not None

    def test_compute_a_share_factors_empty(self):
        """测试空数据"""
        import pandas as pd
        engine = FactorEngine()
        result = engine.compute_a_share_factors(pd.DataFrame())
        assert result.empty


class TestRunFunction:
    """测试 run() 函数"""

    def test_run_basic(self):
        """测试基本运行"""
        from scripts.context import Context
        ctx = Context(
            task_id="test_001",
            user_intent="测试",
            current_stage="IDLE"
        )
        ctx.update_artifact("DATA", "./workspace/data/cleaned_data.parquet")

        result = run(ctx)
        assert "success" in result
        assert "artifact_path" in result

    def test_run_no_data(self):
        """测试无数据情况"""
        from scripts.context import Context
        ctx = Context(
            task_id="test_001",
            user_intent="测试",
            current_stage="IDLE"
        )

        result = run(ctx)
        assert result["success"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
