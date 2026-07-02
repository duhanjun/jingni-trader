"""intent_parser 单元测试"""
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import numpy as np
import pandas as pd

from quant_opt.intent_parser import IntentParser


class TestIntentParser(unittest.TestCase):

    def setUp(self):
        self.parser = IntentParser(today=datetime(2026, 6, 19))

    def test_basic_reversal(self):
        intent = self.parser.parse("帮我用近3年A股数据做一个20日反转因子选股回测")
        self.assertIn("BACKTEST", intent.target_stages)
        self.assertIn("DATA", intent.target_stages)
        self.assertEqual(intent.strategy_name, "reversal")
        self.assertEqual(intent.strategy_params.get("lookback"), 20)
        self.assertNotEqual(intent.start_date, "")
        self.assertNotEqual(intent.end_date, "")
        self.assertGreater(intent.confidence, 0.5)

    def test_csi500_pool(self):
        intent = self.parser.parse("回测一下中证500过去1年月线MACD策略")
        self.assertIn("000905.SH", intent.stock_pool)
        self.assertEqual(intent.strategy_name, "macd")

    def test_hs300_momentum(self):
        intent = self.parser.parse("测试沪深300 5日动量选股，每月调仓")
        self.assertIn("000300.SH", intent.stock_pool)
        self.assertEqual(intent.strategy_name, "momentum")
        self.assertEqual(intent.strategy_params.get("lookback"), 5)
        self.assertEqual(intent.strategy_params.get("rebalance_freq"), "monthly")

    def test_explicit_date_range(self):
        intent = self.parser.parse("用2022-01-01到2024-06-30的A股数据做双均线回测")
        self.assertEqual(intent.start_date, "2022-01-01")
        # 解析后应落在 2024-06
        self.assertTrue(intent.end_date.startswith("2024-06"))

    def test_max_drawdown_constraint(self):
        intent = self.parser.parse("做回测，最大回撤控制在15%以内")
        self.assertIn("max_drawdown", intent.risk_constraints)
        self.assertAlmostEqual(intent.risk_constraints["max_drawdown"], 0.15, places=2)

    def test_target_annual_return(self):
        intent = self.parser.parse("回测目标年化≥20%")
        self.assertIn("target_annual_return", intent.risk_constraints)

    def test_ma_strategy(self):
        intent = self.parser.parse("用MA20做均线策略")
        self.assertEqual(intent.strategy_name, "ma_cross")
        self.assertEqual(intent.strategy_params.get("slow_ma"), 20)

    def test_rsi_strategy(self):
        intent = self.parser.parse("RSI策略回测")
        self.assertEqual(intent.strategy_name, "rsi")
        self.assertEqual(intent.strategy_params.get("period"), 14)

    def test_rsi_custom_period(self):
        intent = self.parser.parse("用RSI6做回测")
        self.assertEqual(intent.strategy_params.get("period"), 6)

    def test_alla_pool(self):
        intent = self.parser.parse("全市场回测")
        self.assertEqual(intent.stock_pool, [])  # 全 A 表示空 pool
        # start_date 没指定 -> 缺失
        self.assertIn("date_range", intent.missing_fields)

    def test_default_fallback(self):
        intent = self.parser.parse("hello world")
        # 应该有默认 stages
        self.assertGreater(len(intent.target_stages), 0)
        self.assertLess(intent.confidence, 0.6)

    def test_stages_ordered(self):
        intent = self.parser.parse("做因子和回测，下载数据")
        order = ["DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"]
        prev_idx = -1
        for s in intent.target_stages:
            idx = order.index(s) if s in order else 99
            self.assertGreaterEqual(idx, prev_idx)
            prev_idx = idx


if __name__ == "__main__":
    unittest.main()
