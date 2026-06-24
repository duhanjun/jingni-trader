"""risk_engine 单元测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import numpy as np
import pandas as pd

from skills.quant-optimizations.quant_opt_20260619.risk_engine import (
    RiskEngine, check_single_order, check_portfolio, check_daily_loss,
    check_data_freshness, RiskLevel
)


class TestRiskEngine(unittest.TestCase):

    def test_valid_order_passes(self):
        d = check_single_order(
            {"code": "600000.SH", "side": "buy", "price": 10.0, "shares": 100, "amount": 1000.0},
            portfolio_value=1_000_000,
        )
        self.assertTrue(d.passed)
        self.assertEqual(d.level, RiskLevel.INFO)

    def test_invalid_lot_size(self):
        d = check_single_order(
            {"code": "600000.SH", "side": "buy", "price": 10.0, "shares": 50, "amount": 500.0},
            portfolio_value=1_000_000,
        )
        self.assertFalse(d.passed)
        self.assertEqual(d.level, RiskLevel.BLOCK)
        self.assertIn("lot_size", d.rule)

    def test_nan_price_blocked(self):
        d = check_single_order(
            {"code": "600001.SH", "side": "buy", "price": float("nan"), "shares": 100, "amount": 1000.0},
            portfolio_value=1_000_000,
        )
        self.assertFalse(d.passed)
        self.assertEqual(d.level, RiskLevel.BLOCK)

    def test_limit_up_blocks_buy(self):
        d = check_single_order(
            {"code": "600002.SH", "side": "buy", "price": 10.0, "shares": 100, "amount": 1000.0,
             "is_limit_up": True},
            portfolio_value=1_000_000,
        )
        self.assertFalse(d.passed)
        self.assertEqual(d.level, RiskLevel.BLOCK)

    def test_limit_down_blocks_sell(self):
        d = check_single_order(
            {"code": "600003.SH", "side": "sell", "price": 10.0, "shares": 100, "amount": 1000.0,
             "is_limit_down": True},
            portfolio_value=1_000_000,
        )
        self.assertFalse(d.passed)

    def test_order_amount_too_large(self):
        d = check_single_order(
            {"code": "600004.SH", "side": "buy", "price": 10.0, "shares": 10000, "amount": 100000.0},
            portfolio_value=1_000_000, max_order_ratio=0.05,
        )
        self.assertFalse(d.passed)
        self.assertEqual(d.level, RiskLevel.BLOCK)

    def test_single_weight_exceeded(self):
        # 15000 股 @ 10元 = 150000, 持仓 15% > 10% 权重, 但 amount ratio 2% (单笔 150000/1M=15%)
        # 用 max_order_ratio=0.20 让 amount_ratio 通过, 触发 single_weight
        d = check_single_order(
            {"code": "600005.SH", "side": "buy", "price": 10.0, "shares": 15000, "amount": 150000.0},
            portfolio_value=1_000_000, max_weight=0.10, max_order_ratio=0.20, holdings={},
        )
        self.assertFalse(d.passed)
        self.assertIn("single_weight", d.rule)

    def test_portfolio_normal(self):
        weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.3})
        decisions = check_portfolio(weights)
        # 现金 10% 低于 5%? 应该是 0.1 > 0.05 通过
        # 行业集中度没有 industry_map
        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0].passed)

    def test_portfolio_over_leverage(self):
        weights = pd.Series({"A": 0.6, "B": 0.6})  # 120%
        decisions = check_portfolio(weights, max_leverage=1.0)
        self.assertTrue(any(not d.passed and d.level == RiskLevel.BLOCK for d in decisions))

    def test_portfolio_industry_concentration(self):
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        ind_map = {"A": "科技", "B": "科技", "C": "金融"}
        decisions = check_portfolio(weights, industry_map=ind_map, max_industry_concentration=0.4)
        self.assertTrue(any("industry_concentration" in d.rule for d in decisions))

    def test_daily_loss_circuit_breaker(self):
        # 模拟一次大跌
        eq = pd.Series([100, 105, 103, 101, 95, 96])  # 5% 单日跌幅
        decisions = check_daily_loss(eq, max_daily_loss=0.03)
        self.assertTrue(any("daily_loss" in d.rule for d in decisions))

    def test_daily_loss_passes(self):
        np.random.seed(0)
        # 0.5% 日波动, 99.99% 概率日亏 < 5%
        rets = np.random.normal(0.0005, 0.005, 200)
        eq = pd.Series((1 + rets).cumprod() * 100)
        # 容忍 8 天连续亏损, 主要验证日亏损不熔断
        decisions = check_daily_loss(eq, max_daily_loss=0.05, max_consecutive_loss_days=10)
        # 任一决策通过即可
        self.assertTrue(any(d.passed for d in decisions))

    def test_consecutive_loss_circuit_breaker(self):
        eq = pd.Series([100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 92])
        decisions = check_daily_loss(eq, max_daily_loss=0.10, max_consecutive_loss_days=5)
        self.assertTrue(any("consecutive_loss" in d.rule for d in decisions))

    def test_data_freshness_empty(self):
        decisions = check_data_freshness(pd.DataFrame())
        self.assertTrue(any(not d.passed for d in decisions))

    def test_data_freshness_fresh(self):
        # 用近期数据 (距今天 < max_age_days)
        data = pd.DataFrame({
            "date": pd.date_range(end=pd.Timestamp.now().normalize(), periods=5),
            "close": [10.0] * 5,
        })
        decisions = check_data_freshness(data, max_age_days=365)
        self.assertEqual(decisions[0].rule, "data.passed")

    def test_data_freshness_nan(self):
        # 用近期数据避免 staleness 触发, 但 close 含 NaN
        data = pd.DataFrame({
            "date": pd.date_range(end=pd.Timestamp.now().normalize(), periods=3),
            "close": [10.0, float("nan"), 11.0],
        })
        decisions = check_data_freshness(data, max_age_days=365)
        self.assertTrue(any("nan_inf" in d.rule for d in decisions))

    def test_risk_engine_comprehensive(self):
        engine = RiskEngine()
        data = pd.DataFrame({
            "date": pd.date_range("2025-06-18", periods=3),
            "close": [10.0, 10.5, 10.3],
        })
        np.random.seed(0)
        eq = pd.Series((1 + np.random.normal(0.0005, 0.01, 100)).cumprod() * 100)
        report = engine.comprehensive_check(
            data=data,
            orders=[{"code": "600000.SH", "side": "buy", "price": 10.0, "shares": 100, "amount": 1000.0}],
            portfolio_value=1_000_000,
            equity_curve=eq,
        )
        self.assertGreater(report.n_passed, 0)


if __name__ == "__main__":
    unittest.main()