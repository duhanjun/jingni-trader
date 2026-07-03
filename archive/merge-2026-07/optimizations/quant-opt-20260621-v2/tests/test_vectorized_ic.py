"""
向量化 IC 分析测试

验证内容:
1. 正确性: 向量化版与逐日循环版结果一致
2. 性能: 向量化版应显著快于逐日循环版
3. 边界: 空数据、单股票、缺失值、小样本
"""
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from vectorized_ic_analysis import (
    calc_ic_vectorized, calc_ic_stats, calc_ic_decay,
    calc_quantile_ic, calc_ic_legacy, benchmark_ic,
)
from tests.data_generator import generate_market_data, generate_factor_data, generate_forward_returns


class TestICCorrectness(unittest.TestCase):
    """正确性测试: 向量化版 vs 逐日循环版"""

    @classmethod
    def setUpClass(cls):
        cls.market = generate_market_data(n_stocks=30, n_days=120, seed=1)
        cls.factors = generate_factor_data(cls.market, seed=1)
        cls.fwd = generate_forward_returns(cls.market)
        cls.factor_names = ["reversal_5d", "reversal_20d", "volatility_20d", "synthetic_alpha"]

    def test_vectorized_matches_legacy(self):
        """向量化 IC 与逐日循环 IC 均值应高度一致(差值 < 1e-6)"""
        vec_ic = calc_ic_vectorized(
            self.factors, self.fwd, self.factor_names, "ret_forward_5d", "spearman")
        merged = self.factors[["code", "date"] + self.factor_names].merge(
            self.fwd[["code", "date", "ret_forward_5d"]], on=["code", "date"], how="inner")

        for f in self.factor_names:
            legacy = calc_ic_legacy(merged, f, "ret_forward_5d", "spearman")
            self.assertIsNotNone(legacy, f"因子 {f} 逐日循环 IC 为空")
            self.assertIn(f, vec_ic, f"因子 {f} 向量化 IC 为空")
            vec_mean = float(vec_ic[f].mean())
            legacy_mean = float(legacy.mean())
            self.assertLess(abs(vec_mean - legacy_mean), 1e-6,
                            f"因子 {f}: 向量化IC={vec_mean} vs 逐日IC={legacy_mean} 差异过大")

    def test_pearson_ic(self):
        """Pearson IC 应可计算且非 NaN"""
        vec_ic = calc_ic_vectorized(
            self.factors, self.fwd, self.factor_names, "ret_forward_5d", "pearson")
        for f in self.factor_names:
            self.assertIn(f, vec_ic)
            self.assertFalse(vec_ic[f].isna().all(), f"因子 {f} Pearson IC 全为 NaN")

    def test_synthetic_alpha_has_positive_ic(self):
        """注入的 synthetic_alpha 应有正 IC(验证计算逻辑正确)"""
        vec_ic = calc_ic_vectorized(
            self.factors, self.fwd, ["synthetic_alpha"], "ret_forward_5d", "spearman")
        ic_mean = float(vec_ic["synthetic_alpha"].mean())
        self.assertGreater(ic_mean, 0, f"synthetic_alpha IC 应为正, 实际 {ic_mean}")

    def test_ic_stats(self):
        """IC 统计量应完整"""
        vec_ic = calc_ic_vectorized(
            self.factors, self.fwd, self.factor_names, "ret_forward_5d", "spearman")
        stats = calc_ic_stats(vec_ic)
        self.assertEqual(len(stats), len(self.factor_names))
        for s in stats:
            self.assertIn("ic_mean", s)
            self.assertIn("ic_ir", s)
            self.assertIn("ic_t_stat", s)
            self.assertIn("ic_count", s)
            self.assertGreater(s["ic_count"], 0)


class TestICPerformance(unittest.TestCase):
    """性能测试"""

    @classmethod
    def setUpClass(cls):
        # 较大数据集以体现性能差异
        cls.market = generate_market_data(n_stocks=200, n_days=500, seed=2)
        cls.factors = generate_factor_data(cls.market, seed=2)
        cls.fwd = generate_forward_returns(cls.market)
        cls.factor_names = ["reversal_5d", "reversal_20d", "volatility_20d",
                            "lncap", "turnover_20d", "synthetic_alpha"]

    def test_vectorized_faster_than_legacy(self):
        """向量化版应至少比逐日循环版快 3x"""
        bench = benchmark_ic(
            self.factors, self.fwd, self.factor_names, "ret_forward_5d", "spearman")
        print(f"\n[IC性能] 向量化: {bench['vectorized_time_sec']}s, "
              f"逐日循环: {bench['legacy_time_sec']}s, "
              f"加速比: {bench['speedup']}x")
        self.assertGreater(bench["speedup"], 3.0,
                           f"加速比 {bench['speedup']}x 未达 3x 预期")
        # 结果一致性
        self.assertLess(bench["max_abs_ic_mean_diff"], 1e-6,
                        "向量化与逐日循环 IC 均值差异过大")


class TestICBoundary(unittest.TestCase):
    """边界条件测试"""

    def test_empty_data(self):
        """空数据应返回空字典而非报错"""
        empty = pd.DataFrame(columns=["code", "date", "f1"])
        fwd_empty = pd.DataFrame(columns=["code", "date", "ret_forward_5d"])
        result = calc_ic_vectorized(empty, fwd_empty, ["f1"], "ret_forward_5d")
        self.assertEqual(result, {})

    def test_single_stock(self):
        """单股票(截面不足)应跳过该日期"""
        df = pd.DataFrame({
            "code": ["000001.SZ"] * 50,
            "date": pd.bdate_range("2023-01-01", periods=50),
            "f1": np.random.randn(50),
        })
        fwd = df.copy()
        fwd["ret_forward_5d"] = np.random.randn(50)
        result = calc_ic_vectorized(df, fwd, ["f1"], "ret_forward_5d", min_stocks=10)
        # 单股票无法计算截面 IC，应返回空
        self.assertEqual(result, {})

    def test_missing_factor_column(self):
        """因子列不存在应跳过而非报错"""
        market = generate_market_data(n_stocks=20, n_days=60, seed=3)
        factors = generate_factor_data(market, seed=3)
        fwd = generate_forward_returns(market)
        result = calc_ic_vectorized(factors, fwd, ["nonexistent_factor"], "ret_forward_5d")
        self.assertEqual(result, {})

    def test_all_nan_factor(self):
        """全 NaN 因子应跳过"""
        market = generate_market_data(n_stocks=20, n_days=60, seed=4)
        factors = generate_factor_data(market, seed=4)
        factors["all_nan"] = np.nan
        fwd = generate_forward_returns(market)
        result = calc_ic_vectorized(factors, fwd, ["all_nan"], "ret_forward_5d")
        self.assertNotIn("all_nan", result)

    def test_ic_decay(self):
        """IC 衰减分析应返回多持有期结果"""
        market = generate_market_data(n_stocks=30, n_days=120, seed=5)
        factors = generate_factor_data(market, seed=5)
        fwd = generate_forward_returns(market)
        decay = calc_ic_decay(factors, fwd, ["synthetic_alpha"], periods=[1, 5, 20])
        self.assertFalse(decay.empty)
        self.assertIn(1, decay.columns)
        self.assertIn(5, decay.columns)

    def test_quantile_ic(self):
        """分层 IC 应返回各分位组"""
        market = generate_market_data(n_stocks=50, n_days=120, seed=6)
        factors = generate_factor_data(market, seed=6)
        fwd = generate_forward_returns(market)
        qi = calc_quantile_ic(factors, fwd, "synthetic_alpha", "ret_forward_5d", n_quantiles=5)
        self.assertLessEqual(len(qi), 5)
        self.assertGreater(len(qi), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
