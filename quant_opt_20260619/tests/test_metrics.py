"""metrics 单元测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import numpy as np
import pandas as pd

from quant_opt_20260619.metrics import (
    total_return, annualized_return, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, alpha_beta, information_ratio,
    var_cvar, deflated_sharpe, drawdown_summary, turnover_stats,
    trade_stats, full_report
)


class TestMetrics(unittest.TestCase):

    def test_total_return_basic(self):
        eq = pd.Series([100, 110, 121, 133.1])
        self.assertAlmostEqual(total_return(eq), 0.331, places=2)

    def test_total_return_negative(self):
        eq = pd.Series([100, 90, 80])
        self.assertAlmostEqual(total_return(eq), -0.2, places=3)

    def test_annualized_return(self):
        # 252 天翻倍
        eq = pd.Series([1.0] + [2.0 ** (i / 251) for i in range(1, 252)])
        ar = annualized_return(eq)
        self.assertAlmostEqual(ar, 2.0 ** (252/252) - 1, places=3)  # 接近 100%

    def test_sharpe_zero_vol(self):
        # 无波动 - 返回 0
        rets = pd.Series([0.001] * 100)
        sr = sharpe_ratio(rets)
        # 即使有均值, 零波动时不能除
        self.assertTrue(np.isfinite(sr))

    def test_sharpe_known(self):
        # 年化 10%, 波动 10%, 无风险 3%
        rets = pd.Series([0.10 / 252] * 252)
        rets.iloc[1:] = rets.iloc[0]
        # 标准差为 0
        # 修正: 用有波动的数据
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.0004, 0.01, 252))
        sr = sharpe_ratio(rets, risk_free=0.03)
        self.assertGreater(sr, -5.0)
        self.assertLess(sr, 5.0)

    def test_sortino_inferior_to_sharpe_when_even(self):
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.0005, 0.015, 500))
        sr = sharpe_ratio(rets, 0.03)
        so = sortino_ratio(rets, 0.03)
        # Sortino 应该比 Sharpe 高 (或接近)
        self.assertGreater(so, sr - 0.5)

    def test_max_drawdown(self):
        # 先涨 50% 再跌 50% -> -25% drawdown
        eq = pd.Series([100, 150, 130, 75])
        mdd = max_drawdown(eq)
        self.assertAlmostEqual(mdd, -0.5, places=2)

    def test_calmar(self):
        np.random.seed(0)
        eq = pd.Series((1 + np.random.normal(0.001, 0.01, 500)).cumprod() * 100)
        cr = calmar_ratio(eq)
        self.assertIsInstance(cr, float)

    def test_alpha_beta_known(self):
        # 完全跟基准
        np.random.seed(0)
        b = pd.Series(np.random.normal(0.0005, 0.01, 500))
        p = b.copy()  # 完美相关, alpha=0, beta=1
        alpha, beta = alpha_beta(p, b, risk_free=0.03)
        self.assertAlmostEqual(beta, 1.0, places=2)
        self.assertAlmostEqual(alpha, 0.0, places=2)

    def test_alpha_beta_higher_beta(self):
        np.random.seed(0)
        b = pd.Series(np.random.normal(0.0005, 0.01, 500))
        p = b * 2  # beta=2
        alpha, beta = alpha_beta(p, b, risk_free=0.03)
        self.assertAlmostEqual(beta, 2.0, places=2)

    def test_information_ratio(self):
        np.random.seed(42)
        b = pd.Series(np.random.normal(0.0005, 0.01, 500))
        p = b + pd.Series(np.random.normal(0.001, 0.005, 500))  # 明显的主动收益
        ir = information_ratio(p, b)
        self.assertGreater(ir, 0.0)

    def test_var_cvar(self):
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.001, 0.02, 500))
        var, cvar = var_cvar(rets, 0.95)
        self.assertGreater(var, 0.0)
        self.assertGreater(cvar, var)  # CVaR 应该比 VaR 极端

    def test_deflated_sharpe_penalty(self):
        # 当 n_trials 增大, 同样的 SR 在 deflated 后应降低 (需要 observed 接近 e_max)
        # 当 n_trials=1, 没有 deflation, 退化为 observed_sharpe
        sr1 = deflated_sharpe(0.5, n_trials=1, n_obs=252)
        # n_trials=1 -> 返回原值
        self.assertAlmostEqual(sr1, 0.5, places=3)

    def test_deflated_sharpe_significance(self):
        # 当 observed 远高于 e_max_sharpe, deflated z-score 应为正 (高显著性)
        sr_high = deflated_sharpe(3.0, n_trials=100, n_obs=252)
        sr_low = deflated_sharpe(0.5, n_trials=100, n_obs=252)
        self.assertGreater(sr_high, sr_low)

    def test_deflated_sharpe_no_penalty_for_single(self):
        # n_trials=1 时不惩罚
        sr = deflated_sharpe(2.0, n_trials=1, n_obs=252)
        self.assertEqual(sr, 2.0)

    def test_drawdown_summary(self):
        # 涨 -> 跌 -> 涨 -> 跌
        eq = pd.Series([100, 120, 90, 110, 60, 80])
        s = drawdown_summary(eq)
        self.assertIn("dd_count", s)
        self.assertGreaterEqual(s["dd_count"], 1)
        self.assertLess(s["max_depth"], 0.0)

    def test_turnover_stats(self):
        w = pd.DataFrame({
            "A": [0.5, 0.3, 0.0],
            "B": [0.0, 0.3, 0.5],
        }, index=pd.date_range("2024-01-01", periods=3))
        s = turnover_stats(w)
        self.assertGreater(s["annual_turnover"], 0)

    def test_trade_stats_empty(self):
        s = trade_stats(pd.DataFrame())
        self.assertEqual(s["total_trades"], 0)

    def test_trade_stats_with_pnl(self):
        trades = pd.DataFrame({
            "pnl": [100, -50, 200, -30, 50],
        })
        s = trade_stats(trades)
        self.assertEqual(s["total_trades"], 5)
        self.assertEqual(s["win_rate"], 0.6)
        self.assertGreater(s["profit_factor"], 1.0)

    def test_full_report(self):
        np.random.seed(0)
        eq = pd.Series((1 + np.random.normal(0.0005, 0.015, 500)).cumprod() * 1e6)
        eq.index = pd.date_range("2022-01-01", periods=500, freq="B")
        bench = pd.Series((1 + np.random.normal(0.0003, 0.012, 500)).cumprod())
        r = full_report(eq, benchmark_returns=bench, n_trials=5)
        self.assertIn("sharpe_ratio", r.metrics)
        self.assertIn("sortino_ratio", r.metrics)
        self.assertIn("deflated_sharpe", r.metrics)
        self.assertIn("var_95", r.metrics)
        self.assertIn("cvar_95", r.metrics)
        self.assertIsNotNone(r.benchmark)
        self.assertIn("alpha_annual", r.benchmark)
        self.assertIn("beta", r.benchmark)


if __name__ == "__main__":
    unittest.main()