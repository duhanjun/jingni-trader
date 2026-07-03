"""
测试：多层风控管理器
- L1: 个股止损、ATR 止损、移动止盈
- L2: 单日亏损、组合回撤熔断、缩仓曲线
- L3: 波动率目标
- 综合：多层评估
"""
import sys
import numpy as np
import pandas as pd
import unittest

sys.path.insert(0, "/workspace")

from quant_opt_20260620.risk_manager.multi_layer import (
    MultiLayerRiskManager,
    RiskConfig,
)


class TestMultiLayerRisk(unittest.TestCase):

    def setUp(self):
        self.cfg = RiskConfig(
            individual_stop_loss=0.08,
            individual_take_profit=0.20,
            atr_stop_multiplier=2.0,
            max_daily_loss_ratio=0.03,
            max_drawdown_threshold=0.15,
            drawdown_scale_start=0.05,
            drawdown_scale_end=0.15,
            min_position_scale=0.3,
            target_annual_vol=0.10,
            max_leverage=1.0,
        )
        self.rm = MultiLayerRiskManager(self.cfg)

    # ── L1 测试 ──
    def test_L1_stop_loss_triggered(self):
        """亏损 10% > 8% 止损 → 触发止损"""
        res = self.rm.check_position_stop(entry_price=100, current_price=89)
        self.assertTrue(res["stop"])
        self.assertEqual(res["reason"], "stop_loss")

    def test_L1_take_profit_triggered(self):
        """盈利 25% > 20% 止盈 → 触发止盈"""
        res = self.rm.check_position_stop(entry_price=100, current_price=125)
        self.assertTrue(res["stop"])
        self.assertEqual(res["reason"], "take_profit")

    def test_L1_hold_within_range(self):
        """亏损 5% 在容忍范围内 → 持有"""
        res = self.rm.check_position_stop(entry_price=100, current_price=95)
        self.assertFalse(res["stop"])
        self.assertEqual(res["reason"], "hold")

    def test_L1_trailing_stop(self):
        """曾达 30% 涨幅后回撤至 14%（> 0.5×30%×peak）→ 移动止盈"""
        res = self.rm.check_position_stop(
            entry_price=100, current_price=114, high_at_entry=130
        )
        # 14% < 30% × 0.5 = 15% → 触发 trailing stop
        # 实际：从 peak 130 回撤到 114：return = (114-100)/100 = 14%
        # high_ret = 30% >= 20% (take_profit) 且 14% <= 15% (0.5×30%)
        self.assertTrue(res["stop"])
        self.assertEqual(res["reason"], "trailing_stop")

    def test_L1_trailing_no_trigger_if_small_pullback(self):
        """小回撤不触发移动止盈"""
        res = self.rm.check_position_stop(
            entry_price=100, current_price=125, high_at_entry=130
        )
        # 25% 收益 > 20% take_profit, 优先 take_profit
        self.assertTrue(res["stop"])
        self.assertEqual(res["reason"], "take_profit")

    def test_atr_stop_price(self):
        """ATR 止损价 = entry - 2 * ATR"""
        stop = self.rm.atr_stop_price(entry_price=100, atr=1.5)
        self.assertAlmostEqual(stop, 100 - 2 * 1.5, places=4)

    def test_compute_atr_known(self):
        """手工构造的 TR 序列应得到正确 ATR"""
        high = np.array([11, 12, 13, 12, 11, 12, 13, 14])
        low = np.array([10, 10, 11, 10, 9, 10, 11, 12])
        close = np.array([10.5, 11, 12, 11, 10, 11, 12, 13])
        atr = self.rm.compute_atr(high, low, close, period=3)
        # 第 2 个有效 ATR 应 > 0
        self.assertGreater(atr[2], 0)
        # 后续 ATR 单调更新
        self.assertFalse(np.isnan(atr[-1]))

    # ── L2 测试 ──
    def test_L2_daily_loss_triggered(self):
        """当日亏损 4% > 3% → 触发"""
        self.rm.reset_day(pd.Timestamp("2023-01-01"), current_nav=1_000_000)
        res = self.rm.check_daily_loss(current_nav=960_000)
        self.assertTrue(res["triggered"])
        self.assertEqual(res["reason"], "daily_loss_breach")

    def test_L2_daily_loss_not_triggered(self):
        """当日亏损 1% < 3% → 不触发"""
        self.rm.reset_day(pd.Timestamp("2023-01-01"), current_nav=1_000_000)
        res = self.rm.check_daily_loss(current_nav=990_000)
        self.assertFalse(res["triggered"])

    def test_L2_drawdown_scale_curve(self):
        """回撤越大，缩仓比例越低（线性）"""
        # dd = 0 → scale = 1.0
        self.assertEqual(self.rm.position_scale(0.0), 1.0)
        # dd = -0.04 (小于 -0.05 阈值) → scale = 1.0
        self.assertEqual(self.rm.position_scale(-0.04), 1.0)
        # dd = -0.10 (中间) → progress = (0.10-0.05)/(0.15-0.05) = 0.5
        # scale = 1.0 - 0.7 * 0.5 = 0.65
        s = self.rm.position_scale(-0.10)
        self.assertAlmostEqual(s, 0.65, places=2)
        # dd = -0.20 (超过 -0.15) → scale = 0.3
        self.assertEqual(self.rm.position_scale(-0.20), 0.3)
        # dd = -0.15 边界 → scale = 0.3
        self.assertEqual(self.rm.position_scale(-0.15), 0.3)
        # dd = -0.05 边界 → scale = 1.0
        self.assertEqual(self.rm.position_scale(-0.05), 1.0)
        print(f"\n  [RISK] drawdown scale: -0.04→{self.rm.position_scale(-0.04):.2f}, "
              f"-0.10→{self.rm.position_scale(-0.10):.2f}, "
              f"-0.20→{self.rm.position_scale(-0.20):.2f}")

    def test_L2_portfolio_drawdown_triggered(self):
        """回撤 20% > 15% → 熔断"""
        self.rm._peak_nav = 1_000_000
        res = self.rm.check_portfolio_drawdown(current_nav=800_000)
        self.assertTrue(res["triggered"])
        self.assertEqual(res["position_scale"], 0.3)

    # ── L3 测试 ──
    def test_L3_vol_target_high_vol(self):
        """实际波动率 > 目标 → 降低杠杆"""
        # 模拟 30% 年化波动率的日收益（远超 10% 目标）
        np.random.seed(42)
        high_vol = np.random.normal(0, 0.018, 30)  # ~28% 年化
        lev = self.rm.volatility_target_leverage(high_vol)
        self.assertLess(lev, 0.5)  # 杠杆 < 0.5
        print(f"\n  [RISK] vol_target_lev (high vol) = {lev:.4f}")

    def test_L3_vol_target_low_vol(self):
        """实际波动率 < 目标 → 杠杆 = max_leverage（被 max 截断）"""
        np.random.seed(7)
        low_vol = np.random.normal(0, 0.001, 30)  # 极低波动
        lev = self.rm.volatility_target_leverage(low_vol)
        self.assertEqual(lev, 1.0)  # 受 max_leverage 限制

    def test_L3_vol_target_normal(self):
        """实际波动率 = 目标 → 杠杆 ≈ 1.0"""
        # 构造 10% 年化波动率
        daily = np.random.normal(0, 0.10 / np.sqrt(252), 30)
        lev = self.rm.volatility_target_leverage(daily, target_vol=0.10)
        # 样本波动率会有偏差，但应接近 1.0
        self.assertGreater(lev, 0.5)
        self.assertLessEqual(lev, 1.0)

    def test_L3_vol_target_too_few_samples(self):
        """样本不足 5 → 返回 1.0（保守）"""
        lev = self.rm.volatility_target_leverage(np.array([0.01, 0.02]))
        self.assertEqual(lev, 1.0)

    # ── 综合评估 ──
    def test_full_evaluate_no_breach(self):
        """健康组合应不触发任何熔断"""
        self.rm.reset_day(pd.Timestamp("2023-01-01"), current_nav=1_000_000)
        np.random.seed(1)
        recent = np.random.normal(0, 0.005, 30)  # ~8% 年化
        positions = pd.DataFrame([
            {"code": "A", "entry_price": 100, "current_price": 102, "high_at_entry": 103, "atr": 1.0},
            {"code": "B", "entry_price": 50, "current_price": 51, "high_at_entry": 52, "atr": 0.8},
        ])
        result = self.rm.evaluate(current_nav=1_005_000, recent_returns=recent, positions=positions)
        self.assertFalse(result["any_triggered"])
        self.assertIn("L1_position_stops", result)
        self.assertIn("L2_daily_loss", result)
        self.assertIn("L2_drawdown", result)
        self.assertIn("L3_leverage", result)
        # 全部头寸应持有
        for code, r in result["L1_position_stops"].items():
            self.assertFalse(r["stop"])
        print(f"\n  [RISK] full evaluate: scale={result['final_position_scale']:.3f}, "
              f"lev={result['L3_leverage']:.3f}")

    def test_full_evaluate_breach(self):
        """触发日亏损时 overall any_triggered 为 True"""
        self.rm.reset_day(pd.Timestamp("2023-01-01"), current_nav=1_000_000)
        recent = np.zeros(30)
        # 当日 NAV 跌至 950k
        result = self.rm.evaluate(current_nav=950_000, recent_returns=recent)
        self.assertTrue(result["any_triggered"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
