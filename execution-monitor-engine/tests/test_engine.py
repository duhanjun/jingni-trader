"""
执行监控引擎测试
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Account, CircuitBreaker, PaperExecutor, run


class TestAccount:
    """测试 Account 类"""

    def test_init(self):
        """测试账户初始化"""
        account = Account(nav=1000000, available_cash=1000000)
        assert account.nav == 1000000
        assert account.available_cash == 1000000
        assert account.positions == {}

    def test_apply_buy(self):
        """测试买入"""
        account = Account(nav=1000000, available_cash=1000000)
        account.apply_buy("000001.SZ", 10.0, 100, 2.5)
        assert "000001.SZ" in account.positions
        assert account.positions["000001.SZ"]["volume"] == 100


class TestCircuitBreaker:
    """测试 CircuitBreaker 类"""

    def test_init(self):
        """测试断路器初始化"""
        cb = CircuitBreaker()
        assert cb.last_order_times == []

    def test_check_frequency(self):
        """测试频率检查"""
        cb = CircuitBreaker()
        result = cb._check_frequency()
        assert result == True


class TestPaperExecutor:
    """测试 PaperExecutor 类"""

    def test_init(self):
        """测试执行器初始化"""
        executor = PaperExecutor()
        assert executor.account is not None
        assert executor.circuit_breaker is not None
        assert executor.audit is not None

    def test_send_order(self):
        """测试发送订单"""
        executor = PaperExecutor()
        result = executor.send_order("000001.SZ", "buy", 100, 10.0)
        assert result["success"] == True


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
        ctx.update_artifact("PORTFOLIO", "./quant_workspace/portfolio/portfolio_weights.json")

        result = run(ctx)
        assert "success" in result
        assert "artifact_path" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
