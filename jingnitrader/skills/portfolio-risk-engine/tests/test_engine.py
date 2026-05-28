"""
组合优化引擎测试
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import AShareConstraints, RiskManager, run


class TestAShareConstraints:
    """测试 AShareConstraints 类"""

    def test_init(self):
        """测试约束初始化"""
        constraints = AShareConstraints()
        assert constraints is not None


class TestRiskManager:
    """测试 RiskManager 类"""

    def test_init(self):
        """测试风险管理器初始化"""
        rm = RiskManager()
        assert rm is not None

    def test_check_portfolio_stop(self):
        """测试组合止损检查"""
        import pandas as pd
        rm = RiskManager()
        rm.reset_daily(1000000)
        result = rm.check_portfolio_stop(990000)
        assert "triggered" in result
        assert "daily_return" in result


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
        ctx.update_artifact("DATA", "./quant_workspace/data/cleaned_data.parquet")

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
