"""
报告引擎测试
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ReportGenerator, run


class TestReportGenerator:
    """测试 ReportGenerator 类"""

    def test_init(self):
        """测试报告生成器初始化"""
        generator = ReportGenerator()
        assert generator is not None
        assert generator.charts == []
        assert generator.metrics == {}

    def test_calc_performance_metrics_empty(self):
        """测试空数据"""
        import pandas as pd
        generator = ReportGenerator()
        result = generator.calc_performance_metrics(pd.DataFrame())
        assert result == {}

    def test_make_equity_chart_empty(self):
        """测试空数据图表"""
        import pandas as pd
        generator = ReportGenerator()
        result = generator.make_equity_chart(pd.DataFrame())
        assert result == ""


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
