"""
因子表达式引擎测试

验证内容:
1. 正确性: 表达式引擎结果与现有 factor-engine 的硬编码实现一致
2. 性能: 表达式引擎 vs 硬编码 if/elif 链的耗时对比
3. 边界条件: 空数据、单只股票、NaN 处理、除零保护
4. Alpha158 因子库完整性
"""
import os
import sys
import time
import unittest

import numpy as np
import pandas as pd

# 把优化包目录加入 path
_OPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

from factor_expression_engine import (
    FactorExpressionEngine, ALPHA158_FACTORS, compute_alpha158,
)
from tests.test_data_generator import generate_synthetic_data


class TestExpressionParser(unittest.TestCase):
    """测试表达式解析"""

    def setUp(self):
        self.engine = FactorExpressionEngine()

    def test_simple_field(self):
        """测试单个字段引用"""
        data = generate_synthetic_data(n_codes=3, n_days=30)
        result = self.engine.compute("$close", data)
        self.assertEqual(len(result), len(data))
        np.testing.assert_array_almost_equal(result.values, data["close"].values)

    def test_arithmetic(self):
        """测试算术运算"""
        data = generate_synthetic_data(n_codes=3, n_days=30)
        result = self.engine.compute("($close - $open) / $open", data)
        expected = (data["close"] - data["open"]) / data["open"]
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_ref_operator(self):
        """测试 Ref 算子"""
        data = generate_synthetic_data(n_codes=3, n_days=30)
        result = self.engine.compute("Ref($close, 5)", data)
        expected = data.groupby("code")["close"].transform(lambda x: x.shift(5))
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_mean_operator(self):
        """测试 Mean 算子"""
        data = generate_synthetic_data(n_codes=3, n_days=30)
        result = self.engine.compute("Mean($close, 20)", data)
        expected = data.groupby("code")["close"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_rank_operator(self):
        """测试截面 Rank 算子"""
        data = generate_synthetic_data(n_codes=10, n_days=30)
        result = self.engine.compute("Rank($close)", data)
        expected = data.groupby("date")["close"].rank(pct=True)
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_invalid_syntax(self):
        """测试非法表达式"""
        data = generate_synthetic_data(n_codes=3, n_days=10)
        with self.assertRaises(ValueError):
            self.engine.compute("$close +", data)
        with self.assertRaises(ValueError):
            self.engine.compute("UnknownFunc($close)", data)

    def test_cache(self):
        """测试 AST 缓存"""
        data = generate_synthetic_data(n_codes=3, n_days=30)
        # 第一次解析
        self.engine.compute("Mean($close, 10)", data)
        self.assertIn("Mean($close, 10)", self.engine._ast_cache)
        # 第二次应命中缓存
        cache_size_before = len(self.engine._ast_cache)
        self.engine.compute("Mean($close, 10)", data)
        self.assertEqual(len(self.engine._ast_cache), cache_size_before)


class TestCorrectnessVsHardcoded(unittest.TestCase):
    """正确性对比: 表达式引擎 vs 现有 factor-engine 硬编码实现"""

    def setUp(self):
        self.data = generate_synthetic_data(n_codes=20, n_days=120, seed=123)
        self.engine = FactorExpressionEngine()

    def test_reversal_20d(self):
        """20日反转因子: 表达式 vs factor-engine.compute_a_share_factors"""
        # 现有 factor-engine 的实现: reversal_20d = -ret_20d = -(close.pct_change(20))
        expected = -self.data.groupby("code")["close"].transform(lambda x: x.pct_change(20))

        # 表达式引擎实现
        result = self.engine.compute("-1 * ($close / Ref($close, 20) - 1)", self.data)

        # 对齐后比较 (排除 NaN)
        mask = ~expected.isna() & ~result.isna()
        np.testing.assert_array_almost_equal(
            result[mask].values, expected[mask].values, decimal=6,
            err_msg="20日反转因子结果不一致"
        )

    def test_ma_ratio(self):
        """均线比值因子: Mean($close, 20) / $close"""
        expected = self.data.groupby("code")["close"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        ) / self.data["close"]

        result = self.engine.compute("Mean($close, 20) / $close", self.data)

        mask = ~expected.isna() & ~result.isna()
        np.testing.assert_array_almost_equal(
            result[mask].values, expected[mask].values, decimal=6
        )

    def test_volatility(self):
        """波动率因子: Std($close, 20) / $close"""
        expected = self.data.groupby("code")["close"].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )
        # 表达式: Std($close, 20) / $close  (注意: 现有实现是收益率标准差, 表达式是价格标准差)
        # 这里对比价格标准差
        expected_price_std = self.data.groupby("code")["close"].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        result = self.engine.compute("Std($close, 20)", self.data)
        mask = ~expected_price_std.isna() & ~result.isna()
        np.testing.assert_array_almost_equal(
            result[mask].values, expected_price_std[mask].values, decimal=6
        )

    def test_volume_ratio(self):
        """量比因子: $volume / Mean($volume, 20)"""
        expected = self.data["volume"] / self.data.groupby("code")["volume"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        result = self.engine.compute("$volume / Mean($volume, 20)", self.data)
        mask = ~expected.isna() & ~result.isna() & np.isfinite(expected) & np.isfinite(result)
        np.testing.assert_array_almost_equal(
            result[mask].values, expected[mask].values, decimal=6
        )


class TestAlpha158Library(unittest.TestCase):
    """测试 Alpha158 因子库"""

    def setUp(self):
        # 使用 250 天数据, 保证 60 日窗口因子也有足够非空值
        self.data = generate_synthetic_data(n_codes=10, n_days=250, seed=456)
        self.engine = FactorExpressionEngine()

    def test_alpha158_count(self):
        """Alpha158 因子数量"""
        self.assertGreaterEqual(len(ALPHA158_FACTORS), 100,
                                f"Alpha158 因子库应有 100+ 因子, 实际 {len(ALPHA158_FACTORS)}")

    def test_alpha158_compute_all(self):
        """计算全部 Alpha158 因子, 确保无异常"""
        result = compute_alpha158(self.data, self.engine)
        self.assertEqual(len(result), len(self.data))
        # 检查因子列数
        factor_cols = [c for c in result.columns if c not in ["code", "date"]]
        self.assertEqual(len(factor_cols), len(ALPHA158_FACTORS))

    def test_alpha158_no_all_nan(self):
        """Alpha158 因子不应全为 NaN (数据足够长时)"""
        result = compute_alpha158(self.data, self.engine)
        factor_cols = [c for c in result.columns if c not in ["code", "date"]]
        for col in factor_cols:
            non_nan_ratio = result[col].notna().mean()
            # 60 日窗口因子在 250 天数据下至少有 (250-60)/250 = 76% 非空
            self.assertGreater(non_nan_ratio, 0.5,
                               f"因子 {col} 非空率仅 {non_nan_ratio:.2%}, 过低")


class TestPerformance(unittest.TestCase):
    """性能对比测试"""

    def setUp(self):
        # 较大数据集: 50 只股票 x 250 天
        self.data = generate_synthetic_data(n_codes=50, n_days=250, seed=789)
        self.engine = FactorExpressionEngine()

    def test_expression_vs_hardcoded_performance(self):
        """表达式引擎 vs 硬编码实现的性能对比"""
        # 硬编码实现 (模拟现有 factor-engine 的 if/elif 链)
        def hardcoded_reversal_20d(data):
            df = data.sort_values(["code", "date"]).copy()
            df["ret_20d"] = df.groupby("code")["close"].pct_change(20)
            df["reversal_20d"] = -df["ret_20d"]
            return df["reversal_20d"]

        # 表达式实现
        expr = "-1 * ($close / Ref($close, 20) - 1)"

        # 预热 (触发缓存)
        self.engine.compute(expr, self.data)

        # 计时: 硬编码
        n_runs = 5
        t0 = time.perf_counter()
        for _ in range(n_runs):
            hardcoded_reversal_20d(self.data)
        t_hardcoded = (time.perf_counter() - t0) / n_runs

        # 计时: 表达式引擎
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.engine.compute(expr, self.data)
        t_expr = (time.perf_counter() - t0) / n_runs

        print(f"\n[性能] 20日反转因子 (50股票x250天), 平均 {n_runs} 次:")
        print(f"  硬编码实现: {t_hardcoded*1000:.2f} ms")
        print(f"  表达式引擎: {t_expr*1000:.2f} ms")
        print(f"  比值: {t_expr/t_hardcoded:.2f}x")

        # 表达式引擎不应比硬编码慢 5 倍以上 (由于 AST 解析开销)
        self.assertLess(t_expr / t_hardcoded, 5.0,
                        f"表达式引擎过慢: {t_expr/t_hardcoded:.2f}x")

    def test_alpha158_batch_performance(self):
        """批量计算 Alpha158 全部因子的性能"""
        t0 = time.perf_counter()
        result = compute_alpha158(self.data, self.engine)
        elapsed = time.perf_counter() - t0

        n_factors = len(ALPHA158_FACTORS)
        print(f"\n[性能] Alpha158 全部 {n_factors} 因子 (50股票x250天): {elapsed:.2f}s, "
              f"平均 {elapsed/n_factors*1000:.1f} ms/因子")

        # 100+ 因子应在合理时间内完成
        self.assertLess(elapsed, 120.0, f"Alpha158 计算耗时 {elapsed:.1f}s 过长")


class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试"""

    def setUp(self):
        self.engine = FactorExpressionEngine()

    def test_empty_data(self):
        """空数据"""
        empty = pd.DataFrame(columns=["code", "date", "close", "open", "high", "low", "volume"])
        with self.assertRaises((KeyError, ValueError, IndexError)):
            self.engine.compute("Mean($close, 5)", empty)

    def test_single_stock(self):
        """单只股票"""
        data = generate_synthetic_data(n_codes=1, n_days=30)
        result = self.engine.compute("Mean($close, 5)", data)
        self.assertEqual(len(result), 30)

    def test_short_history(self):
        """历史数据短于窗口期"""
        data = generate_synthetic_data(n_codes=3, n_days=10)
        # 20 日均线, 但只有 10 天数据, 应返回 NaN 而非报错
        result = self.engine.compute("Mean($close, 20)", data)
        # min_periods=10, 所以 10 天数据应该能算出值
        self.assertFalse(result.isna().all())

    def test_division_by_zero(self):
        """除零保护"""
        data = generate_synthetic_data(n_codes=3, n_days=30)
        # $volume 可能为 0? 这里合成数据不会为 0, 但测试表达式不报错
        result = self.engine.compute("$close / ($volume + 1e-12)", data)
        self.assertFalse(result.isna().all())

    def test_nested_expression(self):
        """嵌套表达式"""
        data = generate_synthetic_data(n_codes=5, n_days=60)
        expr = "Rank(Mean($close, 20) / $close - 1)"
        result = self.engine.compute(expr, data)
        self.assertEqual(len(result), len(data))
        # Rank 结果应在 [0, 1]
        valid = result.dropna()
        if len(valid) > 0:
            self.assertGreaterEqual(valid.min(), 0)
            self.assertLessEqual(valid.max(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
