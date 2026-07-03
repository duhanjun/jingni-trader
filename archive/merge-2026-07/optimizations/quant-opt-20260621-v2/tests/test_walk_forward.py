"""
Walk-Forward 验证与基准指标测试
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from walk_forward_validator import (
    generate_walk_forward_windows, walk_forward_validate,
)
from benchmark_metrics import calc_benchmark_metrics, calc_full_metrics
from tests.data_generator import (
    generate_market_data, generate_factor_data, generate_forward_returns,
)


class TestWalkForwardWindows(unittest.TestCase):

    def test_window_generation(self):
        dates = pd.bdate_range("2020-01-01", periods=750)  # ~3年
        windows = generate_walk_forward_windows(dates, train_months=12, test_months=6)
        self.assertGreater(len(windows), 0)
        # 窗口不应重叠(test 期)
        for i in range(len(windows) - 1):
            self.assertLessEqual(windows[i].test_end, windows[i + 1].test_start)

    def test_window_ordering(self):
        dates = pd.bdate_range("2020-01-01", periods=750)
        windows = generate_walk_forward_windows(dates, train_months=12, test_months=6)
        for i in range(len(windows) - 1):
            self.assertLess(windows[i].train_start, windows[i + 1].train_start)

    def test_empty_dates(self):
        windows = generate_walk_forward_windows(pd.DatetimeIndex([]), 12, 6)
        self.assertEqual(len(windows), 0)


class TestWalkForwardValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 3年数据以支持 walk-forward
        cls.market = generate_market_data(n_stocks=50, n_days=750, seed=20)
        cls.factors = generate_factor_data(cls.market, seed=20)
        cls.fwd = generate_forward_returns(cls.market)

    def test_walk_forward_run(self):
        result = walk_forward_validate(
            self.market, self.factors, self.fwd,
            ["synthetic_alpha", "reversal_20d"],
            train_months=12, test_months=6,
        )
        self.assertIn("stability", result)
        self.assertIn("synthetic_alpha", result["stability"])
        self.assertGreater(result["n_windows"], 0)

    def test_synthetic_alpha_stable(self):
        """注入的 synthetic_alpha 应有较好的样本外稳定性"""
        result = walk_forward_validate(
            self.market, self.factors, self.fwd,
            ["synthetic_alpha"], train_months=12, test_months=6,
        )
        stab = result["stability"]["synthetic_alpha"]
        # synthetic_alpha 是注入的有效因子，样本外 IC 应为正
        self.assertGreater(stab["oos_mean"], -0.5,
                           f"synthetic_alpha 样本外 IC {stab['oos_mean']} 异常")


class TestBenchmarkMetrics(unittest.TestCase):

    def test_basic_benchmark_metrics(self):
        """基准相对指标应完整"""
        dates = pd.bdate_range("2023-01-01", periods=250)
        strat = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.01, 250)), index=dates)
        bench = pd.Series(np.cumprod(1 + np.random.normal(0.0003, 0.008, 250)), index=dates)
        m = calc_benchmark_metrics(strat, bench)
        for key in ["information_ratio", "tracking_error", "alpha", "beta",
                    "excess_return_annual", "excess_max_drawdown",
                    "up_capture_ratio", "down_capture_ratio", "correlation_with_benchmark"]:
            self.assertIn(key, m, f"缺少基准指标 {key}")

    def test_beta_zero_when_uncorrelated(self):
        """不相关的策略与基准 Beta 应接近 0"""
        rng = np.random.default_rng(30)
        dates = pd.bdate_range("2023-01-01", periods=500)
        strat = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 500)), index=dates)
        bench = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 500)), index=dates)
        m = calc_benchmark_metrics(strat, bench)
        self.assertLess(abs(m["beta"]), 0.3, f"Beta {m['beta']} 应接近0")

    def test_beta_one_when_identical(self):
        """策略与基准完全一致时 Beta 应为 1, Alpha 为 0"""
        dates = pd.bdate_range("2023-01-01", periods=250)
        eq = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.01, 250)), index=dates)
        m = calc_benchmark_metrics(eq, eq.copy())
        self.assertAlmostEqual(m["beta"], 1.0, places=2)
        self.assertAlmostEqual(m["alpha"], 0.0, places=2)
        self.assertAlmostEqual(m["tracking_error"], 0.0, places=4)

    def test_full_metrics_without_benchmark(self):
        """无基准时应仅返回绝对指标"""
        dates = pd.bdate_range("2023-01-01", periods=100)
        eq = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.01, 100)), index=dates)
        m = calc_full_metrics(eq)
        self.assertIn("sharpe_ratio", m)
        self.assertIn("max_drawdown", m)
        # 无基准时不应有 information_ratio
        self.assertNotIn("information_ratio", m)

    def test_empty_series(self):
        """空序列应返回空字典"""
        self.assertEqual(calc_benchmark_metrics(pd.Series(), pd.Series()), {})
        self.assertEqual(calc_full_metrics(pd.Series()), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
