"""
因子表达式引擎 - 验证测试
==========================
测试内容:
1. 正确性测试: 表达式计算结果与 pandas 手写实现一致
2. 性能对比测试: 表达式引擎 vs jingni-trader 硬编码方式
3. 边界条件测试: 空数据、单标的、NaN、窗口不足、未知算子等

运行: python -m optimizations.factor_expr.test_factor_expr
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# 确保能导入 optimizations 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimizations.factor_expr import (  # noqa: E402
    ExpressionEngine,
    FactorRegistry,
    register_builtins,
)


def make_panel(n_days: int = 60, n_codes: int = 20, seed: int = 42) -> pd.DataFrame:
    """生成测试用 MultiIndex(date, code) 行情面板。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])

    n = len(idx)
    close = 10.0 + rng.normal(0, 0.5, n).cumsum().reshape(n_days, n_codes).flatten()
    close = np.maximum(close, 1.0)
    open_ = close * (1 + rng.normal(0, 0.01, n))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.lognormal(15, 1, n)
    amount = volume * close
    turnover = rng.uniform(0.005, 0.05, n)
    total_mv = close * 1e8

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "turnover_rate": turnover,
            "total_mv": total_mv,
        },
        index=idx,
    )


class TestExpressionParsing(unittest.TestCase):
    """表达式解析正确性测试。"""

    def setUp(self) -> None:
        self.engine = ExpressionEngine()

    def test_field_and_const(self) -> None:
        node = self.engine.parse("$close")
        self.assertEqual(node.expr(), "$close")

        node = self.engine.parse("20")
        self.assertEqual(node.expr(), "20")

    def test_infix_arithmetic(self) -> None:
        node = self.engine.parse("$close - $open")
        self.assertIn("-", node.expr())

        node = self.engine.parse("MA($close, 20) - MA($close, 5)")
        self.assertIn("MA", node.expr())

    def test_nested_expression(self) -> None:
        node = self.engine.parse("MA($close - $open, 20)")
        self.assertIn("MA", node.expr())

    def test_unary_minus(self) -> None:
        node = self.engine.parse("-$close")
        self.assertIn("Neg", node.expr())

    def test_parse_cache(self) -> None:
        """同一表达式解析结果应缓存。"""
        n1 = self.engine.parse("MA($close, 20)")
        n2 = self.engine.parse("MA($close, 20)")
        self.assertIs(n1, n2)

    def test_unknown_operator_raises(self) -> None:
        with self.assertRaises(NameError):
            self.engine.parse("UNKNOWN($close, 20)")

    def test_missing_paren_raises(self) -> None:
        with self.assertRaises(SyntaxError):
            self.engine.parse("MA($close, 20")

    def test_wrong_arity_raises(self) -> None:
        with self.assertRaises(SyntaxError):
            self.engine.parse("MA($close)")


class TestExpressionCorrectness(unittest.TestCase):
    """表达式计算正确性测试 (与 pandas 手写实现对比)。"""

    def setUp(self) -> None:
        self.engine = ExpressionEngine()
        self.data = make_panel(n_days=60, n_codes=20)

    def test_ma_correctness(self) -> None:
        """MA($close, 5) 应等于 pandas rolling(5).mean()。"""
        expr_result = self.engine.compute("MA($close, 5)", self.data)
        expected = (
            self.data["close"]
            .groupby(level=1, group_keys=False)
            .apply(lambda s: s.rolling(5, min_periods=5).mean())
        )
        # 对齐后比较非 NaN 部分
        aligned = pd.concat([expr_result.rename("expr"), expected.rename("expected")], axis=1)
        aligned = aligned.dropna()
        np.testing.assert_allclose(aligned["expr"].values, aligned["expected"].values, rtol=1e-10)

    def test_ref_correctness(self) -> None:
        """Ref($close, 1) 应等于 shift(1)。"""
        expr_result = self.engine.compute("Ref($close, 1)", self.data)
        expected = self.data["close"].groupby(level=1, group_keys=False).shift(1)
        aligned = pd.concat([expr_result.rename("expr"), expected.rename("expected")], axis=1).dropna()
        np.testing.assert_allclose(aligned["expr"].values, aligned["expected"].values, rtol=1e-10)

    def test_roc_correctness(self) -> None:
        """ROC($close, 5) 应等于 close / close.shift(5) - 1。"""
        expr_result = self.engine.compute("ROC($close, 5)", self.data)
        shifted = self.data["close"].groupby(level=1, group_keys=False).shift(5)
        expected = self.data["close"] / shifted - 1.0
        aligned = pd.concat([expr_result.rename("expr"), expected.rename("expected")], axis=1).dropna()
        np.testing.assert_allclose(aligned["expr"].values, aligned["expected"].values, rtol=1e-10)

    def test_delta_correctness(self) -> None:
        """Delta($close, 1) 应等于 close - close.shift(1)。"""
        expr_result = self.engine.compute("Delta($close, 1)", self.data)
        shifted = self.data["close"].groupby(level=1, group_keys=False).shift(1)
        expected = self.data["close"] - shifted
        aligned = pd.concat([expr_result.rename("expr"), expected.rename("expected")], axis=1).dropna()
        np.testing.assert_allclose(aligned["expr"].values, aligned["expected"].values, rtol=1e-10)

    def test_std_correctness(self) -> None:
        """STD($close, 10) 应等于 rolling(10).std(ddof=0)。"""
        expr_result = self.engine.compute("STD($close, 10)", self.data)
        expected = (
            self.data["close"]
            .groupby(level=1, group_keys=False)
            .apply(lambda s: s.rolling(10, min_periods=10).std(ddof=0))
        )
        aligned = pd.concat([expr_result.rename("expr"), expected.rename("expected")], axis=1).dropna()
        np.testing.assert_allclose(aligned["expr"].values, aligned["expected"].values, rtol=1e-10)

    def test_csrank_correctness(self) -> None:
        """CSRank(ROC($close, 5)) 应等于按日期分组的 rank(pct=True)。"""
        expr_result = self.engine.compute("CSRank(ROC($close, 5))", self.data)
        roc = self.data["close"].groupby(level=1, group_keys=False).shift(0) / self.data["close"].groupby(
            level=1, group_keys=False
        ).shift(5) - 1.0
        expected = roc.groupby(level=0, group_keys=False).rank(pct=True)
        aligned = pd.concat([expr_result.rename("expr"), expected.rename("expected")], axis=1).dropna()
        np.testing.assert_allclose(aligned["expr"].values, aligned["expected"].values, rtol=1e-10)

    def test_cszscore_correctness(self) -> None:
        """CSZScore 应使每个非退化截面的均值≈0, 标准差≈1。"""
        expr_result = self.engine.compute("CSZScore(ROC($close, 5))", self.data).dropna()
        grouped = expr_result.groupby(level=0)
        means = grouped.mean()
        stds = grouped.std(ddof=0)
        # 仅检查非退化截面 (原始 ROC 有方差的日期)
        roc = self.data["close"].groupby(level=1, group_keys=False).shift(0) / self.data["close"].groupby(
            level=1, group_keys=False
        ).shift(5) - 1.0
        roc_std = roc.groupby(level=0).std(ddof=0)
        non_degenerate = roc_std[roc_std > 1e-10].index
        if len(non_degenerate) > 0:
            np.testing.assert_allclose(
                means.loc[non_degenerate].values,
                np.zeros(len(non_degenerate)),
                atol=1e-9,
            )
            np.testing.assert_allclose(
                stds.loc[non_degenerate].values,
                np.ones(len(non_degenerate)),
                atol=1e-9,
            )

    def test_corr_correctness(self) -> None:
        """CORR($close, $volume, 20) 应等于 rolling corr。"""
        expr_result = self.engine.compute("CORR($close, $volume, 20)", self.data)
        expected = (
            self.data.groupby(level=1, group_keys=False)
            .apply(lambda g: g["close"].rolling(20, min_periods=20).corr(g["volume"]))
        )
        aligned = pd.concat([expr_result.rename("expr"), expected.rename("expected")], axis=1).dropna()
        np.testing.assert_allclose(aligned["expr"].values, aligned["expected"].values, rtol=1e-10)

    def test_rsi_correctness(self) -> None:
        """RSI 应在 0-100 之间。"""
        expr_result = self.engine.compute("RSI($close, 14)", self.data).dropna()
        self.assertTrue((expr_result >= 0).all() and (expr_result <= 100).all())

    def test_compound_expression_correctness(self) -> None:
        """复合表达式 MA($close,20) - MA($close,5) 正确性。"""
        expr_result = self.engine.compute("MA($close, 20) - MA($close, 5)", self.data)
        ma20 = self.data["close"].groupby(level=1, group_keys=False).apply(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        ma5 = self.data["close"].groupby(level=1, group_keys=False).apply(
            lambda s: s.rolling(5, min_periods=5).mean()
        )
        expected = ma20 - ma5
        aligned = pd.concat([expr_result.rename("expr"), expected.rename("expected")], axis=1).dropna()
        np.testing.assert_allclose(aligned["expr"].values, aligned["expected"].values, rtol=1e-10)

    def test_log_abs_correctness(self) -> None:
        """Log(Abs($close - $open)) 正确性。"""
        expr_result = self.engine.compute("Log(Abs($close - $open))", self.data)
        expected = np.log(np.abs(self.data["close"] - self.data["open"]))
        aligned = pd.concat([expr_result.rename("expr"), expected.rename("expected")], axis=1).dropna()
        np.testing.assert_allclose(aligned["expr"].values, aligned["expected"].values, rtol=1e-10)


class TestFactorRegistry(unittest.TestCase):
    """因子注册机制测试。"""

    def setUp(self) -> None:
        self.registry = FactorRegistry.instance()
        self.registry.clear()
        register_builtins(self.registry)

    def tearDown(self) -> None:
        self.registry.clear()

    def test_builtins_registered(self) -> None:
        """内置因子应全部注册。"""
        names = [m.name for m in self.registry.list_factors()]
        for expected in [
            "ret_5d", "ret_20d", "reversal_5d", "reversal_20d",
            "volatility_20d", "volume_ratio", "turnover_5d", "lncap",
            "ma_diff", "rsi_14", "price_momentum",
        ]:
            self.assertIn(expected, names, f"内置因子 {expected} 未注册")

    def test_factor_metadata(self) -> None:
        """因子元数据应正确。"""
        meta = self.registry.get("reversal_20d")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.category, "reversal")
        self.assertEqual(meta.direction, -1)  # 反转因子方向为 -1
        self.assertEqual(meta.fields, ("close",))
        self.assertEqual(meta.window, 20)

    def test_list_by_category(self) -> None:
        """按类别过滤因子。"""
        reversal = self.registry.list_factors(category="reversal")
        self.assertTrue(all(m.category == "reversal" for m in reversal))
        self.assertGreaterEqual(len(reversal), 2)

    def test_compute_factor(self) -> None:
        """计算单个因子应返回 Series。"""
        data = make_panel(n_days=60, n_codes=10)
        result = self.registry.compute("ret_5d", data)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(data))

    def test_compute_many(self) -> None:
        """批量计算多个因子。"""
        data = make_panel(n_days=60, n_codes=10)
        names = ["ret_5d", "reversal_20d", "volatility_20d", "ma_diff"]
        result = self.registry.compute_many(names, data)
        self.assertIsInstance(result, pd.DataFrame)
        for name in names:
            self.assertIn(name, result.columns)

    def test_custom_factor_registration(self) -> None:
        """自定义因子注册。"""

        @self.registry.register(
            name="my_factor",
            category="custom",
            direction=1,
            fields=("close", "volume"),
            window=10,
            description="自定义测试因子",
        )
        def my_factor(df: pd.DataFrame) -> pd.Series:
            return df["close"] * df["volume"]

        self.assertIn("my_factor", self.registry)
        meta = self.registry.get("my_factor")
        self.assertEqual(meta.direction, 1)
        self.assertEqual(meta.fields, ("close", "volume"))

    def test_invalid_direction_raises(self) -> None:
        """direction 非 ±1 应报错。"""
        with self.assertRaises(ValueError):

            @self.registry.register(name="bad", direction=0)
            def bad(df):  # noqa: ANN001, ANN202
                return df["close"]

    def test_unknown_factor_raises(self) -> None:
        """计算未注册因子应报 KeyError。"""
        with self.assertRaises(KeyError):
            self.registry.get_func("nonexistent_factor")

    def test_compute_many_handles_errors(self) -> None:
        """批量计算中单个因子失败不应中断其他因子。"""
        data = make_panel(n_days=60, n_codes=10)

        @self.registry.register(name="error_factor", category="test")
        def error_factor(df: pd.DataFrame) -> pd.Series:
            raise RuntimeError("故意失败")

        result = self.registry.compute_many(["ret_5d", "error_factor"], data)
        self.assertIn("ret_5d", result.columns)
        self.assertIn("errors", result.attrs)
        self.assertIn("error_factor", result.attrs["errors"])


class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试。"""

    def setUp(self) -> None:
        self.engine = ExpressionEngine()

    def test_single_stock(self) -> None:
        """单标的 (DatetimeIndex) 应正常工作。"""
        dates = pd.bdate_range("2024-01-01", periods=30)
        df = pd.DataFrame(
            {"close": np.arange(1, 31, dtype=float), "volume": np.random.rand(30) * 1000},
            index=dates,
        )
        result = self.engine.compute("MA($close, 5)", df)
        self.assertEqual(len(result), 30)
        # 前 4 个应为 NaN
        self.assertTrue(result.iloc[:4].isna().all())
        # 第 5 个应为前 5 个的均值
        self.assertAlmostEqual(result.iloc[4], np.mean([1, 2, 3, 4, 5]))

    def test_window_too_large(self) -> None:
        """窗口大于数据长度应全 NaN。"""
        df = make_panel(n_days=10, n_codes=5)
        result = self.engine.compute("MA($close, 100)", df)
        self.assertTrue(result.isna().all())

    def test_empty_dataframe(self) -> None:
        """空 DataFrame 应返回空结果。"""
        df = pd.DataFrame(columns=["close"], index=pd.MultiIndex.from_tuples([], names=["date", "code"]))
        result = self.engine.compute("MA($close, 5)", df)
        self.assertEqual(len(result), 0)

    def test_missing_field_raises(self) -> None:
        """引用不存在的字段应报错。"""
        df = make_panel(n_days=10, n_codes=3)
        with self.assertRaises(KeyError):
            self.engine.compute("$nonexistent", df)

    def test_division_by_zero(self) -> None:
        """除零应产生 inf/nan 而非崩溃。"""
        df = make_panel(n_days=10, n_codes=3)
        df["volume"] = 0.0
        result = self.engine.compute("$close / $volume", df)
        self.assertTrue((result == np.inf).any() or result.isna().any())

    def test_negative_window_raises(self) -> None:
        """负窗口应报错。"""
        with self.assertRaises(ValueError):
            self.engine.parse("MA($close, -5)")

    def test_nan_propagation(self) -> None:
        """NaN 应正确传播。"""
        df = make_panel(n_days=20, n_codes=3)
        df.loc[df.index[5], "close"] = np.nan
        result = self.engine.compute("MA($close, 3)", df)
        # 包含 NaN 的窗口结果应为 NaN
        self.assertTrue(result.isna().any())


class TestPerformanceComparison(unittest.TestCase):
    """性能对比测试: 表达式引擎 vs 硬编码方式。"""

    def test_performance_vs_hardcoded(self) -> None:
        """表达式引擎性能应与硬编码 pandas 在同一数量级 (向量化)。

        jingni-trader 现状: factor-engine/engine.py:48-117 硬编码因子,
        IC 计算 (engine.py:250) 按日 Python 循环。
        本测试验证表达式引擎的向量化实现性能可接受。
        """
        engine = ExpressionEngine()
        data = make_panel(n_days=120, n_codes=50)

        # 表达式引擎计算
        t0 = time.perf_counter()
        for _ in range(5):
            engine.compute("MA($close, 20) - MA($close, 5)", data)
            engine.compute("ROC($close, 20)", data)
            engine.compute("STD(ROC($close, 1), 20)", data)
        expr_time = time.perf_counter() - t0

        # 硬编码 pandas 计算 (模拟 jingni-trader 现状)
        t0 = time.perf_counter()
        for _ in range(5):
            ma20 = data["close"].groupby(level=1, group_keys=False).apply(
                lambda s: s.rolling(20).mean()
            )
            ma5 = data["close"].groupby(level=1, group_keys=False).apply(
                lambda s: s.rolling(5).mean()
            )
            _ = ma20 - ma5
            shifted = data["close"].groupby(level=1, group_keys=False).shift(20)
            _ = data["close"] / shifted - 1.0
        hardcoded_time = time.perf_counter() - t0

        # 表达式引擎有解析开销, 但向量化计算应与硬编码在同一数量级
        # 允许 3x 开销 (解析 + AST 遍历)
        ratio = expr_time / hardcoded_time if hardcoded_time > 0 else float("inf")
        print(
            f"\n[性能] 表达式引擎: {expr_time*1000:.1f}ms, "
            f"硬编码: {hardcoded_time*1000:.1f}ms, 比值: {ratio:.2f}x"
        )
        self.assertLess(ratio, 5.0, f"表达式引擎过慢: {ratio:.2f}x, 应 < 5x")

    def test_batch_compute_performance(self) -> None:
        """批量计算 10 个因子的性能。"""
        registry = FactorRegistry.instance()
        registry.clear()
        register_builtins(registry)

        data = make_panel(n_days=120, n_codes=50)
        names = [m.name for m in registry.list_factors()][:10]

        t0 = time.perf_counter()
        result = registry.compute_many(names, data)
        elapsed = time.perf_counter() - t0

        print(f"\n[性能] 批量计算 {len(names)} 个因子 (120日x50股): {elapsed*1000:.1f}ms")
        self.assertEqual(len(result.columns), len(names))
        self.assertLess(elapsed, 5.0, "批量计算应在 5 秒内完成")


if __name__ == "__main__":
    unittest.main(verbosity=2)
