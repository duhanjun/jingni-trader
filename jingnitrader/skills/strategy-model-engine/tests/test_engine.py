"""
模型引擎测试
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ModelEngine, run


class TestModelEngine:
    """测试 ModelEngine 类"""

    def test_init(self):
        """测试引擎初始化"""
        engine = ModelEngine()
        assert engine is not None

    def test_purged_group_ts_split(self):
        """测试交叉验证分割"""
        import pandas as pd
        engine = ModelEngine()
        dates = pd.Series(pd.date_range('2020-01-01', periods=100, freq='D'))
        splits = engine.purged_group_ts_split(dates, n_splits=3)
        assert len(splits) > 0

    def test_generate_rule_based_signal(self):
        """测试规则型信号生成"""
        import pandas as pd
        engine = ModelEngine()
        df = pd.DataFrame({
            'code': ['000001.SZ'] * 10,
            'date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'alpha_score': range(10)
        })
        result = engine.generate_rule_based_signal(df, 'single_factor')
        assert 'signal' in result.columns


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
        ctx.update_artifact("FACTOR", "./workspace/factors/factor_data.parquet")

        result = run(ctx)
        assert "success" in result
        assert "artifact_path" in result

    def test_run_no_factor(self):
        """测试无因子数据情况"""
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
