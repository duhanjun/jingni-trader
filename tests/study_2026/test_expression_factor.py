"""
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
优化方向: 表达式驱动的因子引擎（Expression-Based Factor Engine）

Qlib 的核心创新之一是它的表达式因子系统，允许用户通过简洁的字符串表达式
（如 `$close / Ref($close, 20) - 1`）来定义因子，而不需要硬编码每个因子。

当前 jingni-trader 的 factor-engine 在 compute_a_share_factors() 中
硬编码了所有因子计算逻辑，扩展性差。

本测试验证表达式因子引擎的可行性，对比硬编码方式和表达式方式。
"""

import sys
import os
import re
import time
import json
import unittest

import numpy as np
import pandas as pd

# ---- 表达式因子引擎实现 (原型) ----

class ExpressionFactorEngine:
    """
    表达式驱动的因子计算引擎

    支持的操作符:
      - 列引用: $close, $open, $volume 等
      - 数学运算: +, -, *, /, **, abs(), log(), sqrt()
      - 时间序列: Ref(expr, N), Mean(expr, N), Std(expr, N)
      - 分组操作: GroupRank(expr)
      - 复合: Ts_Rank(expr, N)
    """

    # 支持的列别名
    COLUMN_ALIASES = {
        "close": "close", "c": "close",
        "open": "open", "o": "open",
        "high": "high", "h": "high",
        "low": "low", "l": "low",
        "volume": "volume", "v": "volume",
        "amount": "amount", "a": "amount",
        "turnover": "turnover_rate", "turn": "turnover_rate",
        "change_pct": "change_pct",
    }

    def __init__(self, data: pd.DataFrame):
        """
        参数:
            data: 包含 code, date 和 OHLCV 列的 DataFrame
        """
        self.data = data.sort_values(["code", "date"]).reset_index(drop=True)
        self._factor_cache = {}

    def evaluate(self, expression: str) -> pd.Series:
        """
        计算因子表达式

        参数:
            expression: 因子表达式，如 "$close / Ref($close, 20) - 1"

        返回:
            与输入 data 等长的 Series
        """
        if expression in self._factor_cache:
            return self._factor_cache[expression]

        result = self._parse_and_eval(expression)
        self._factor_cache[expression] = result
        return result

    def _parse_and_eval(self, expr: str) -> pd.Series:
        """递归下降解析器"""
        expr = expr.strip()

        # 处理分组排名
        m = re.match(r'^GroupRank\((.+)\)$', expr)
        if m:
            inner = self._parse_and_eval(m.group(1))
            return inner.groupby(self.data['code']).rank(pct=True)

        # 处理 Ts_Rank
        m = re.match(r'^Ts_Rank\((.+),\s*(\d+)\)$', expr)
        if m:
            inner = self._parse_and_eval(m.group(1))
            window = int(m.group(2))
            return inner.groupby(self.data['code']).transform(
                lambda x: x.rolling(window, min_periods=1).apply(
                    lambda y: (y.iloc[-1] > y[:-1]).mean() if len(y) > 1 else 0.5, raw=False
                )
            )

        # 处理 Mean(expr, N)
        m = re.match(r'^Mean\((.+),\s*(\d+)\)$', expr)
        if m:
            inner = self._parse_and_eval(m.group(1))
            window = int(m.group(2))
            return inner.groupby(self.data['code']).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        # 处理 Std(expr, N)
        m = re.match(r'^Std\((.+),\s*(\d+)\)$', expr)
        if m:
            inner = self._parse_and_eval(m.group(1))
            window = int(m.group(2))
            return inner.groupby(self.data['code']).transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

        # 处理 Ref(expr, N)
        m = re.match(r'^Ref\((.+),\s*(\d+)\)$', expr)
        if m:
            inner = self._parse_and_eval(m.group(1))
            n = int(m.group(2))
            return inner.groupby(self.data['code']).shift(n)

        # 处理 -expr (一元负号)
        # 只在非二元运算的情况下处理一元负号
        m = re.match(r'^-(.+)$', expr)
        if m:
            inner = m.group(1)
            # 检查是否只有一对匹配的括号包裹整个表达式
            if inner.startswith('(') and inner.endswith(')'):
                # 验证括号匹配
                depth = 0
                matched = True
                for i, ch in enumerate(inner):
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                    if depth == 0 and i < len(inner) - 1:
                        matched = False
                        break
                if matched:
                    return -self._parse_and_eval(inner[1:-1])
            # 单纯的列引用或函数调用
            return -self._parse_and_eval(inner)

        # 处理 log(expr)
        m = re.match(r'^log\((.+)\)$', expr)
        if m:
            inner = self._parse_and_eval(m.group(1))
            return np.log(inner.replace(0, np.nan))

        # 处理 abs(expr)
        m = re.match(r'^abs\((.+)\)$', expr)
        if m:
            inner = self._parse_and_eval(m.group(1))
            return np.abs(inner)

        # 处理 sqrt(expr)
        m = re.match(r'^sqrt\((.+)\)$', expr)
        if m:
            inner = self._parse_and_eval(m.group(1))
            return np.sqrt(np.maximum(inner, 0))

        # 处理二元运算: expr1 OP expr2
        # 从低优先级到高优先级（先匹配外层操作符）
        ops = [
            (r'^(.*)\s*\+\s*(.*)$', lambda a, b: a + b),
            (r'^(.*)\s*-\s*(.*)$', lambda a, b: a - b),
            (r'^(.*)\s*\*\s*(.*)$', lambda a, b: a * b),
            (r'^(.*)\s*/\s*(.*)$', lambda a, b: a / b.replace(0, np.nan)),
            (r'^(.*)\s*\*\*\s*(.*)$', lambda a, b: a ** b),
        ]
        for pattern, op in ops:
            m = re.match(pattern, expr)
            if m:
                left = self._parse_and_eval(m.group(1))
                right = self._parse_and_eval(m.group(2))
                return op(left, right)

        # 处理列引用: $colname 或 $colname_N (带参数)
        m = re.match(r'^\$(\w+)$', expr)
        if m:
            col_alias = m.group(1).lower()
            col_name = self.COLUMN_ALIASES.get(col_alias, col_alias)
            if col_name in self.data.columns:
                return self.data[col_name].copy()
            raise ValueError(f"列 {col_alias} 不存在于数据中，可用列: {list(self.data.columns)}")

        # 处理数值字面量
        try:
            val = float(expr)
            return pd.Series(val, index=self.data.index)
        except ValueError:
            pass

        raise ValueError(f"无法解析表达式: {expr}")

    def cache_clear(self):
        """清除缓存"""
        self._factor_cache.clear()


# ---- 硬编码因子实现（当前 jingni-trader 方式） ----

def compute_factors_hardcoded(data: pd.DataFrame) -> pd.DataFrame:
    """与 jingni-trader 的 FactorEngine.compute_a_share_factors 逻辑一致"""
    df = data.sort_values(["code", "date"]).copy()
    result = df[["code", "date"]].copy()

    result["ret_1d"] = df.groupby("code")["close"].pct_change()
    result["ret_5d"] = df.groupby("code")["close"].pct_change(5)
    result["ret_20d"] = df.groupby("code")["close"].pct_change(20)
    result["ret_60d"] = df.groupby("code")["close"].pct_change(60)

    result["reversal_5d"] = -result["ret_5d"]
    result["reversal_20d"] = -result["ret_20d"]

    volatility_col = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )
    result["volatility_20d"] = volatility_col

    volume_20d = df.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    result["volume_ratio"] = df["volume"] / volume_20d.replace(0, np.nan)

    return result


def compute_factors_expression(data: pd.DataFrame, factor_defs: dict) -> pd.DataFrame:
    """使用表达式引擎计算因子"""
    engine = ExpressionFactorEngine(data)
    result = data[["code", "date"]].copy()

    for name, expr in factor_defs.items():
        result[name] = engine.evaluate(expr)

    return result


# ---- 测试用例 ----

class TestExpressionFactorEngine(unittest.TestCase):
    """表达式因子引擎测试"""

    @classmethod
    def setUpClass(cls):
        """生成测试数据"""
        np.random.seed(42)
        codes = [f"{i:06d}.SH" for i in range(600000, 600020)]  # 20 只股票
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")

        rows = []
        for code in codes:
            n = len(dates)
            start_price = np.random.uniform(5, 50)
            returns = np.random.normal(0.0005, 0.015, n)
            prices = start_price * np.cumprod(1 + returns)
            volumes = np.random.lognormal(12, 1, n).astype(int)

            df_code = pd.DataFrame({
                "code": code,
                "date": dates,
                "open": prices * (1 + np.random.normal(0, 0.003, n)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
                "close": prices,
                "volume": volumes,
                "amount": prices * volumes,
                "turnover_rate": np.random.uniform(0.01, 0.05, n),
                "change_pct": np.concatenate([[0], np.diff(prices) / prices[:-1] * 100]),
            })
            rows.append(df_code)

        cls.test_data = pd.concat(rows, ignore_index=True).sort_values(["code", "date"])

    def test_basic_column_reference(self):
        """测试基本列引用"""
        engine = ExpressionFactorEngine(self.test_data)
        result = engine.evaluate("$close")
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            self.test_data["close"].reset_index(drop=True),
        )

    def test_simple_expression(self):
        """测试简单表达式: $close / $open - 1"""
        engine = ExpressionFactorEngine(self.test_data)
        result = engine.evaluate("$close / $open - 1")
        expected = self.test_data["close"] / self.test_data["open"] - 1
        pd.testing.assert_series_equal(
            result.round(6).reset_index(drop=True),
            expected.round(6).reset_index(drop=True),
        )

    def test_ref_operator(self):
        """测试 Ref 操作符: Ref($close, 5)"""
        engine = ExpressionFactorEngine(self.test_data)
        result = engine.evaluate("Ref($close, 5)")
        expected = self.test_data.groupby("code")["close"].shift(5)
        self.assertTrue(result.notna().sum() < len(result))  # 前5天为NaN

    def test_momentum_factor(self):
        """测试动量因子: $close / Ref($close, 20) - 1"""
        engine = ExpressionFactorEngine(self.test_data)
        result = engine.evaluate("$close / Ref($close, 20) - 1")
        # 手动计算
        shifted = self.test_data.groupby("code")["close"].shift(20)
        expected = self.test_data["close"] / shifted - 1
        # 忽略 Series name 差异（引擎返回的无名 Series）
        result_vals = result.round(6).reset_index(drop=True)
        expected_vals = expected.round(6).reset_index(drop=True)
        result_vals.name = None
        expected_vals.name = None
        pd.testing.assert_series_equal(result_vals, expected_vals)

    def test_mean_operator(self):
        """测试 Mean 操作符: Mean($volume, 20)"""
        engine = ExpressionFactorEngine(self.test_data)
        result = engine.evaluate("Mean($volume, 20)")
        expected = self.test_data.groupby("code")["volume"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        pd.testing.assert_series_equal(
            result.round(0).reset_index(drop=True),
            expected.round(0).reset_index(drop=True),
        )

    def test_std_operator(self):
        """测试 Std 操作符"""
        engine = ExpressionFactorEngine(self.test_data)
        result = engine.evaluate("Std($close, 20)")
        expected = self.test_data.groupby("code")["close"].transform(
            lambda x: x.rolling(20, min_periods=1).std()
        )
        pd.testing.assert_series_equal(
            result.round(6).reset_index(drop=True),
            expected.round(6).reset_index(drop=True),
        )

    def test_grouprank(self):
        """测试 GroupRank 操作符"""
        engine = ExpressionFactorEngine(self.test_data)
        result = engine.evaluate("GroupRank($close)")
        expected = self.test_data.groupby("code")["close"].rank(pct=True)
        pd.testing.assert_series_equal(
            result.round(6).reset_index(drop=True),
            expected.round(6).reset_index(drop=True),
        )

    def test_log_expression(self):
        """测试 log 函数"""
        engine = ExpressionFactorEngine(self.test_data)
        result = engine.evaluate("log($close)")
        expected = np.log(self.test_data["close"].replace(0, np.nan))
        pd.testing.assert_series_equal(
            result.round(6).reset_index(drop=True),
            expected.round(6).reset_index(drop=True),
        )

    def test_caching(self):
        """测试缓存机制"""
        engine = ExpressionFactorEngine(self.test_data)
        t1 = time.perf_counter()
        result1 = engine.evaluate("$close / Ref($close, 20) - 1")
        t2 = time.perf_counter()
        result2 = engine.evaluate("$close / Ref($close, 20) - 1")
        t3 = time.perf_counter()

        first_call = t2 - t1
        cached_call = t3 - t2

        pd.testing.assert_series_equal(
            result1.reset_index(drop=True),
            result2.reset_index(drop=True),
        )
        self.assertLess(cached_call, first_call * 0.5,
                        f"缓存未加速: 首次={first_call:.6f}s, 缓存={cached_call:.6f}s")


class TestExpressionVsHardcoded(unittest.TestCase):
    """对比：表达式引擎 vs 硬编码"""

    FACTOR_DEFINITIONS = {
        "ret_1d": "$close / Ref($close, 1) - 1",
        "ret_5d": "$close / Ref($close, 5) - 1",
        "ret_20d": "$close / Ref($close, 20) - 1",
        "ret_60d": "$close / Ref($close, 60) - 1",
        "reversal_5d": "-($close / Ref($close, 5) - 1)",
        "reversal_20d": "-($close / Ref($close, 20) - 1)",
        "volatility_20d": "Std($close / Ref($close, 1) - 1, 20)",
        "volume_ratio": "$volume / Mean($volume, 20)",
    }

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = [f"{i:06d}.SH" for i in range(600000, 600020)]
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")

        rows = []
        for code in codes:
            n = len(dates)
            start_price = np.random.uniform(5, 50)
            returns = np.random.normal(0.0005, 0.015, n)
            prices = start_price * np.cumprod(1 + returns)
            volumes = np.random.lognormal(12, 1, n).astype(int)

            df_code = pd.DataFrame({
                "code": code,
                "date": dates,
                "open": prices * (1 + np.random.normal(0, 0.003, n)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
                "close": prices,
                "volume": volumes,
                "amount": prices * volumes,
                "turnover_rate": np.random.uniform(0.01, 0.05, n),
            })
            rows.append(df_code)

        cls.test_data = pd.concat(rows, ignore_index=True)

    def test_correctness_comparison(self):
        """验证表达式引擎与硬编码的结果一致性"""
        hw_result = compute_factors_hardcoded(self.test_data)
        expr_result = compute_factors_expression(self.test_data, self.FACTOR_DEFINITIONS)

        for col in ["ret_1d", "ret_5d", "ret_20d", "ret_60d",
                     "reversal_5d", "reversal_20d", "volatility_20d", "volume_ratio"]:
            hw = hw_result[col].dropna()
            ex = expr_result[col].dropna()
            common_idx = hw.index.intersection(ex.index)
            diff = (hw.loc[common_idx] - ex.loc[common_idx]).abs()
            max_diff = diff.max() if len(diff) > 0 else 0
            self.assertLess(max_diff, 1e-6,
                            f"列 {col} 差异过大: max_diff={max_diff:.10f}")

    def test_performance_comparison(self):
        """性能对比"""
        n_runs = 5

        # 硬编码方式
        hw_times = []
        for _ in range(n_runs):
            t1 = time.perf_counter()
            compute_factors_hardcoded(self.test_data)
            t2 = time.perf_counter()
            hw_times.append(t2 - t1)

        # 表达式方式（含缓存清除）
        expr_times = []
        for _ in range(n_runs):
            t1 = time.perf_counter()
            compute_factors_expression(self.test_data, self.FACTOR_DEFINITIONS)
            t2 = time.perf_counter()
            expr_times.append(t2 - t1)

        avg_hw = np.mean(hw_times)
        avg_expr = np.mean(expr_times)

        print(f"\n性能对比 ({len(self.test_data)} 行, {n_runs} 次平均):")
        print(f"  硬编码方式: {avg_hw:.4f}s")
        print(f"  表达式方式: {avg_expr:.4f}s")
        print(f"  比率: {avg_expr / avg_hw:.2f}x")

    def test_extensibility_demo(self):
        """演示表达式引擎的扩展性优势"""
        # 硬编码方式: 新增因子需修改代码
        # 表达式方式: 只需添加一行配置
        new_factors = {
            **self.FACTOR_DEFINITIONS,
            # 新增两个因子，无需修改引擎代码
            "momentum_10d": "$close / Ref($close, 10) - 1",
            "turnover_5d": "Mean($turnover, 5)",
            "vol_rank_20d": "GroupRank(Std($close / Ref($close, 1) - 1, 20))",
            "volume_breakout": "$volume / Mean($volume, 20) - 1",
            "price_ma_ratio": "$close / Mean($close, 20)",
        }

        result = compute_factors_expression(self.test_data, new_factors)
        self.assertIn("momentum_10d", result.columns)
        self.assertIn("turnover_5d", result.columns)
        self.assertIn("vol_rank_20d", result.columns)
        self.assertIn("volume_breakout", result.columns)
        self.assertIn("price_ma_ratio", result.columns)

        # 验证可扩展性：因子数可以轻松从 8 扩展到 13
        self.assertEqual(len(new_factors), 13)
        print("\n✓ 表达式引擎成功计算所有 {0} 个因子".format(len(new_factors)))
        print(f"  新增因子: momentum_10d, turnover_5d, vol_rank_20d, volume_breakout, price_ma_ratio")


if __name__ == "__main__":
    unittest.main(verbosity=2)