import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import BacktestEngine
from config import get_config


class TestBacktestEngine(unittest.TestCase):
    """回测引擎测试类"""

    def setUp(self):
        """测试前准备"""
        self.config = get_config()
        self.engine = BacktestEngine(self.config)

    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.config)
        self.assertIsNotNone(self.engine._adapter)

    def test_run(self):
        """测试运行回测"""
        def dummy_strategy(context, data):
            pass

        results = self.engine.run(
            strategy=dummy_strategy,
            start_date="2020-01-01",
            end_date="2020-12-31",
            initial_capital=1000000,
        )

        self.assertIsNotNone(results)
        self.assertIn("trades_df", results)
        self.assertIn("performance_metrics", results)
        self.assertIn("equity_curve", results)
        self.assertIn("benchmark_curve", results)

    def test_performance_metrics(self):
        """测试绩效指标"""
        def dummy_strategy(context, data):
            pass

        self.engine.run(
            strategy=dummy_strategy,
            start_date="2020-01-01",
            end_date="2020-12-31",
            initial_capital=1000000,
        )

        metrics = self.engine.get_performance_metrics()

        self.assertIsNotNone(metrics)
        self.assertIn("total_return", metrics)
        self.assertIn("annual_return", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("profit_loss_ratio", metrics)

    def test_get_trades(self):
        """测试获取交易记录"""
        def dummy_strategy(context, data):
            pass

        self.engine.run(
            strategy=dummy_strategy,
            start_date="2020-01-01",
            end_date="2020-12-31",
            initial_capital=1000000,
        )

        trades = self.engine.get_trades()
        self.assertIsInstance(trades, pd.DataFrame)

    def test_get_equity_curve(self):
        """测试获取资金曲线"""
        def dummy_strategy(context, data):
            pass

        self.engine.run(
            strategy=dummy_strategy,
            start_date="2020-01-01",
            end_date="2020-12-31",
            initial_capital=1000000,
        )

        equity = self.engine.get_equity_curve()
        self.assertIsInstance(equity, pd.Series)


class TestTradingRules(unittest.TestCase):
    """交易规则测试类"""

    def setUp(self):
        """测试前准备"""
        from rules import AShareTradingRules

        self.config = get_config()
        self.rules = AShareTradingRules(self.config)

    def test_fee_calculation(self):
        """测试费用计算"""
        # 买入
        buy_fees = self.rules.calculate_fee(10000, "buy")
        self.assertIn("commission", buy_fees)
        self.assertIn("stamp_duty", buy_fees)
        self.assertIn("transfer_fee", buy_fees)
        self.assertEqual(buy_fees["stamp_duty"], 0)  # 买入无印花税

        # 卖出
        sell_fees = self.rules.calculate_fee(10000, "sell")
        self.assertGreater(sell_fees["stamp_duty"], 0)  # 卖出有印花税

    def test_limit_up_down_check(self):
        """测试涨跌停检查"""
        pre_close = 10.0

        # 正常价格
        can_trade, reason, price = self.rules.check_limit_up_down(
            "000001.SZ", 10.5, pre_close
        )
        self.assertTrue(can_trade)

        # 超涨停
        can_trade, reason, price = self.rules.check_limit_up_down(
            "000001.SZ", 11.5, pre_close
        )
        self.assertFalse(can_trade)

        # 超跌停
        can_trade, reason, price = self.rules.check_limit_up_down(
            "000001.SZ", 8.5, pre_close
        )
        self.assertFalse(can_trade)


if __name__ == "__main__":
    unittest.main()
