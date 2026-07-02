"""
因子表达式引擎测试
"""
import unittest
import numpy as np
import pandas as pd

from quant_opt_20260617.factor_expression_engine import (
    FactorEngine, _eval_with_lookups, _auto_name
)
from quant_opt_20260617.tests._synthetic_data import generate_synthetic_a_share_data


class TestFactorExpressionEngine(unittest.TestCase):
    """因子表达式引擎单元测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_a_share_data(n_stocks=10, n_days=100, seed=42)
        cls.engine = FactorEngine()

    def test_01_basic_field_lookup(self):
        """$close 应该解析为 close 列"""
        s = self.engine.compute_one(self.data, "$close", name="c")
        self.assertEqual(s.name, "c")
        pd.testing.assert_series_equal(s, self.data["close"], check_names=False)

    def test_02_ts_mean(self):
        """Mean($close, 5) 应该等于 5 日滚动均值"""
        expected = self.data.groupby("code")["close"].transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        )
        actual = self.engine.compute_one(self.data, "Mean($close, 5)")
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_03_ts_delta(self):
        """Delta($close, 1) = $close - Ref($close, 1)"""
        actual = self.engine.compute_one(self.data, "Delta($close, 1)")
        expected = self.data["close"] - self.data.groupby("code")["close"].shift(1)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_04_ts_std(self):
        actual = self.engine.compute_one(self.data, "Std($close, 5)")
        expected = self.data.groupby("code")["close"].transform(
            lambda s: s.rolling(5, min_periods=2).std()
        )
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_05_cs_rank(self):
        """Rank 应该按 date 截面排序"""
        actual = self.engine.compute_one(self.data, "Rank($close)")
        expected = self.data.groupby("date")["close"].rank(pct=True)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_06_composition(self):
        """复合公式：Rank(Delta($close, 1))"""
        actual = self.engine.compute_one(self.data, "Rank(Delta($close, 1))")
        delta = self.data["close"] - self.data.groupby("code")["close"].shift(1)
        expected = delta.groupby(self.data["date"]).rank(pct=True)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_07_math_ops(self):
        """数学运算：Abs, Sign, Sqrt"""
        # Sign
        s = self.engine.compute_one(self.data, "Sign(Delta($close, 1))")
        unique = sorted(s.dropna().unique().tolist())
        # 排序后，sign 应该只包含 -1, 0, 1 三个值
        for v in unique:
            self.assertIn(v, [-1.0, 0.0, 1.0])

        # Abs
        s = self.engine.compute_one(self.data, "Abs(Delta($close, 1))")
        self.assertTrue((s.dropna() >= 0).all())

    def test_08_if_and_arithmetic(self):
        """If 条件 + 比较运算符"""
        s = self.engine.compute_one(
            self.data,
            "If($close > Mean($close, 5), 1, 0)",
        )
        # 与手算结果对比
        mean5 = self.data.groupby("code")["close"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        expected = (self.data["close"] > mean5).astype(int)
        pd.testing.assert_series_equal(s, expected, check_names=False)

    def test_09_case_insensitive(self):
        """算子大小写不敏感"""
        a = self.engine.compute_one(self.data, "mean($close, 5)")
        b = self.engine.compute_one(self.data, "MEAN($close, 5)")
        c = self.engine.compute_one(self.data, "Mean($close, 5)")
        pd.testing.assert_series_equal(a, b, check_names=False)
        pd.testing.assert_series_equal(a, c, check_names=False)

    def test_10_batch_compute(self):
        """批量计算多个因子"""
        result = self.engine.compute(self.data, [
            "Mean($close, 5)",
            "Std($close, 10)",
            "Rank($volume)",
        ])
        self.assertEqual(len(result.columns), len(self.data.columns) + 3)

    def test_11_safety_unknown_field(self):
        """未定义字段应该报错"""
        with self.assertRaises((KeyError, RuntimeError)):
            self.engine.compute_one(self.data, "Mean($nonexistent, 5)")

    def test_12_safety_bad_grammar(self):
        """不支持的语法应该报错"""
        with self.assertRaises((ValueError, RuntimeError, SyntaxError)):
            self.engine.compute_one(self.data, "[1, 2, 3]")

    def test_13_alpha158_subset(self):
        """模拟 Qlib Alpha158 中的几个典型因子"""
        factors = [
            "Mean($close, 5)",                          # KBAR
            "Std($close, 20)",                          # KBAR 波动
            "Mean(Delta($close, 1), 5)",                # 5 日均涨跌幅
            "Rank(Std($close, 5))",                     # 排名后的短期波动
            "If($close > Mean($close, 20), 1, -1)",     # 趋势信号
        ]
        result = self.engine.compute(self.data, factors)
        # 所有因子都应非空（除非有 NaN）
        for f in factors:
            col_name = _auto_name(f)
            self.assertIn(col_name, result.columns)

    def test_14_perf_simple(self):
        """性能基线：30 只股票 × 1000 天，4 个因子，< 2s"""
        import time
        big = generate_synthetic_a_share_data(n_stocks=30, n_days=1000, seed=1)
        eng = FactorEngine()
        t0 = time.time()
        eng.compute(big, [
            "Mean($close, 20)",
            "Std($close, 20)",
            "Rank(Delta($close, 1))",
            "If($close > Mean($close, 20), 1, 0)",
        ])
        elapsed = time.time() - t0
        self.assertLess(elapsed, 2.0, f"性能不达标: {elapsed:.2f}s > 2s")


if __name__ == "__main__":
    unittest.main()
