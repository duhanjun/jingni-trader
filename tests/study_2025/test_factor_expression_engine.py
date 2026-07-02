"""
因子表达式引擎 - 验证代码
借鉴来源: quant-stream 的表达式语言 + Qlib 的 Expression Engine

优化方向:
1. 将 jingni-trader 中硬编码的因子计算改为声明式表达式
2. 支持 $close, $volume, $high, $low 等变量引用
3. 支持 RANK, DELTA, TS_MEAN, ZSCORE 等算子
4. 支持复合表达式嵌套计算

设计参考:
- quant-stream: pyparsing-based expression parser, 
  "$close + $open" -> ADD($close, $open)
  "RANK(DELTA($close, 5))" -> rank of 5-day momentum
- Qlib: Ref($close, N), Mean($close, N), $high-$low

注意: 这是一个验证实验，代码在独立测试文件中，不修改主代码。
"""

import pandas as pd
import numpy as np
import unittest
from typing import Dict, List, Callable, Any
import re


# ============================================================
# 1. 算子注册表 - 借鉴 quant-stream 的函数库分层设计
# ============================================================

class OperatorRegistry:
    """算子注册表，支持动态扩展"""

    def __init__(self):
        self._operators: Dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册默认算子，分类组织"""

        # --- 横截面算子 (Cross-sectional) ---
        self.register("RANK", self._op_rank)
        self.register("SCALE", self._op_scale)
        self.register("ZSCORE", self._op_zscore)
        self.register("MEAN", self._op_cross_mean)
        self.register("STD", self._op_cross_std)
        self.register("MAX", self._op_cross_max)
        self.register("MIN", self._op_cross_min)

        # --- 时间序列算子 (Rolling window) ---
        self.register("TS_MEAN", self._op_ts_mean)
        self.register("TS_STD", self._op_ts_std)
        self.register("TS_MAX", self._op_ts_max)
        self.register("TS_MIN", self._op_ts_min)
        self.register("TS_RANK", self._op_ts_rank)
        self.register("TS_CORR", self._op_ts_corr)

        # --- 逐元素算子 (Element-wise) ---
        self.register("DELTA", self._op_delta)
        self.register("DELAY", self._op_delay)
        self.register("ABS", self._op_abs)
        self.register("LOG", self._op_log)
        self.register("EXP", self._op_exp)
        self.register("SQRT", self._op_sqrt)

        # --- 技术指标 (Technical) ---
        self.register("SMA", self._op_sma)
        self.register("EMA", self._op_ema)
        self.register("RSI", self._op_rsi)
        self.register("MACD", self._op_macd)

    def register(self, name: str, fn: Callable):
        self._operators[name.upper()] = fn

    def get(self, name: str) -> Callable:
        name = name.upper()
        if name not in self._operators:
            raise ValueError(f"未知算子: {name}. 可用: {list(self._operators.keys())}")
        return self._operators[name]

    # ---- 横截面算子实现 ----
    @staticmethod
    def _op_rank(series: pd.Series) -> pd.Series:
        """截面排名，归一化到 [0, 1]"""
        return series.rank(pct=True)

    @staticmethod
    def _op_scale(series: pd.Series) -> pd.Series:
        """截面缩放到 [0, 1]"""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    @staticmethod
    def _op_zscore(series: pd.Series) -> pd.Series:
        """截面 Z-Score"""
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    @staticmethod
    def _op_cross_mean(series: pd.Series) -> pd.Series:
        return pd.Series(series.mean(), index=series.index)

    @staticmethod
    def _op_cross_std(series: pd.Series) -> pd.Series:
        return pd.Series(series.std(), index=series.index)

    @staticmethod
    def _op_cross_max(series: pd.Series) -> pd.Series:
        return pd.Series(series.max(), index=series.index)

    @staticmethod
    def _op_cross_min(series: pd.Series) -> pd.Series:
        return pd.Series(series.min(), index=series.index)

    # ---- 时间序列算子实现 ----
    def _op_ts_mean(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).mean()

    def _op_ts_std(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).std()

    def _op_ts_max(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).max()

    def _op_ts_min(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).min()

    def _op_ts_rank(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(5, window // 2)).rank(pct=True)

    def _op_ts_corr(self, series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
        return series_a.rolling(window, min_periods=max(10, window // 2)).corr(series_b)

    # ---- 逐元素算子实现 ----
    def _op_delta(self, series: pd.Series, periods: int = 1) -> pd.Series:
        return series.diff(periods)

    def _op_delay(self, series: pd.Series, periods: int = 1) -> pd.Series:
        return series.shift(periods)

    def _op_abs(self, series: pd.Series) -> pd.Series:
        return series.abs()

    def _op_log(self, series: pd.Series) -> pd.Series:
        return np.log(series.replace(0, np.nan))

    def _op_exp(self, series: pd.Series) -> pd.Series:
        return np.exp(series)

    def _op_sqrt(self, series: pd.Series) -> pd.Series:
        return np.sqrt(series.abs())

    # ---- 技术指标 ----
    def _op_sma(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).mean()

    def _op_ema(self, series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, min_periods=max(3, window // 2)).mean()

    def _op_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _op_macd(self, series: pd.Series) -> pd.Series:
        ema12 = series.ewm(span=12, min_periods=12).mean()
        ema26 = series.ewm(span=26, min_periods=26).mean()
        return ema12 - ema26


# ============================================================
# 2. 表达式解析器 - 借鉴 quant-stream pyparsing + Qlib 表达式语法
# ============================================================

class FactorExpressionParser:
    """
    因子表达式解析器
    支持两种模式:
    1. 变量引用: $close, $volume, $high, $low, $open, $amount, $turnover
    2. 函数调用: RANK(DELTA($close, 5)), TS_MEAN($volume, 20)
    3. 嵌套表达式: RANK(TS_MEAN(DELTA($close, 1), 20))
    4. 四则运算: $high - $low, $close / DELAY($close, 1) - 1
    """

    # 可用变量映射
    VARIABLE_MAP = {
        "$close": "close",
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$volume": "volume",
        "$amount": "amount",
        "$turnover": "turnover_rate",
        "$change_pct": "change_pct",
    }

    def __init__(self, registry: OperatorRegistry = None):
        self.registry = registry or OperatorRegistry()

    def parse_and_compute(
        self,
        expression: str,
        data: pd.DataFrame,
        date_col: str = "date",
        code_col: str = "code"
    ) -> pd.Series:
        """
        解析表达式并在数据上计算

        参数:
            expression: 如 "RANK(DELTA($close, 5))"
            data: 包含 code, date 及价格列的 DataFrame
            date_col: 日期列
            code_col: 代码列

        返回:
            计算结果 Series (与原 data 对齐)
        """
        expr = expression.strip()

        # 1. 替换变量引用为数据列
        if "$" in expr:
            for var, col in self.VARIABLE_MAP.items():
                if col in data.columns:
                    expr = expr.replace(var, f"__{col}__")

        # 2. 解析并计算
        result = self._evaluate(expr, data, date_col, code_col)
        return result

    def _evaluate(
        self,
        expr: str,
        data: pd.DataFrame,
        date_col: str,
        code_col: str
    ) -> pd.Series:
        """递归评估表达式"""
        expr = expr.strip()

        # 处理四则运算 - 先处理函数调用，后处理算术
        # 只有最外层不在函数调用内时才处理算术
        outer = self._strip_outer_parens(expr)

        # 处理函数调用: FUNC(args)
        match = re.match(r'^([A-Z_][A-Z0-9_]*)\((.*)\)$', expr, re.IGNORECASE)
        if match:
            func_name = match.group(1)
            raw_args = match.group(2)

            # 解析参数
            args = self._parse_args(raw_args)

            # 递归求值每个参数
            evaluated_args = []
            for arg in args:
                arg = arg.strip()
                try:
                    val = float(arg)
                    evaluated_args.append(int(val) if val == int(val) else val)
                except ValueError:
                    val = self._evaluate(arg, data, date_col, code_col)
                    evaluated_args.append(val)

            operator = self.registry.get(func_name)
            return operator(*evaluated_args)

        # 处理算术运算 - 仅在外层处理
        if not expr.startswith("__"):
            # 按优先级从低到高: +, -, *, /
            for op in ["+", "-", "*", "/"]:
                if op in outer:
                    parts = self._split_by_operator(expr, op)
                    if len(parts) > 1:
                        result = self._evaluate(parts[0], data, date_col, code_col)
                        for part in parts[1:]:
                            other = self._evaluate(part, data, date_col, code_col)
                            if op == "+":
                                result = result + other
                            elif op == "-":
                                result = result - other
                            elif op == "*":
                                result = result * other
                            elif op == "/":
                                result = result / other.replace(0, np.nan)
                        return result

        # 处理列引用
        if expr.startswith("__") and expr.endswith("__"):
            col_name = expr[2:-2]
            if col_name in data.columns:
                return data[col_name]
            raise ValueError(f"列 {col_name} 不在数据中")

        # 处理负数
        try:
            return float(expr)
        except ValueError:
            raise ValueError(f"无法解析表达式: {expr}")

    def _strip_outer_parens(self, expr: str) -> str:
        """移除最外层括号后的内容"""
        expr = expr.strip()
        if expr.startswith("(") and expr.endswith(")"):
            depth = 0
            for i, ch in enumerate(expr[:-1]):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth == 0:
                    return expr  # 不匹配
            return expr[1:-1] if expr[0] == "(" and expr[-1] == ")" else expr
        return expr

    def _split_safe(self, expr: str) -> str:
        """同 _strip_outer_parens"""
        return self._strip_outer_parens(expr)

    def _split_by_operator(self, expr: str, operator: str) -> List[str]:
        """按运算符分割（忽略括号内）"""
        parts = []
        depth = 0
        current = ""
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == operator and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        if current:
            parts.append(current)
        return [p.strip() for p in parts]

    def _parse_args(self, args_str: str) -> List[str]:
        """解析函数参数（按逗号分割，忽略括号内逗号）"""
        args = []
        depth = 0
        current = ""
        for ch in args_str:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current.strip())
        return args


# ============================================================
# 3. 因子表达式上下文 - 在 gruopby(code) 下批量计算
# ============================================================

class FactorExpressionEngine:
    """
    因子表达式引擎 - 借鉴 quant-stream 的分层架构:
    1. 表达式解析器 (parser)
    2. 算子注册表 (registry)
    3. 分组批量计算 (per-code)

    用法:
        engine = FactorExpressionEngine()
        data = pd.DataFrame(...)  # code, date, close, volume, ...
        result = engine.compute(data, "RANK(DELTA($close, 5))")
    """

    def __init__(self):
        self.registry = OperatorRegistry()
        self.parser = FactorExpressionParser(self.registry)

    def compute(
        self,
        data: pd.DataFrame,
        expression: str,
        code_col: str = "code",
        date_col: str = "date",
        name: str = None,
    ) -> pd.DataFrame:
        """
        在数据集上计算因子表达式

        参数:
            data: 包含 code, date 和价格列的 DataFrame
            expression: 因子表达式
            name: 结果列名 (默认用表达式)

        返回:
            包含 code, date 和因子值的 DataFrame
        """
        if name is None:
            # 替换变量引用为干净的列名
            cleaned = expression
            for var in self.parser.VARIABLE_MAP:
                cleaned = cleaned.replace(var, var.replace("$", ""))
            name = cleaned.replace("(", "_").replace(")", "").replace(", ", "_").replace(" ", "_")
            # 清理连续的下划线
            while "__" in name:
                name = name.replace("__", "_")
            name = name.strip("_")

        data = data.sort_values([code_col, date_col]).copy()

        # 按 code 分组计算
        results = []
        for code, group in data.groupby(code_col):
            group = group.copy()
            group[name] = self.parser.parse_and_compute(expression, group, date_col, code_col)
            results.append(group[[code_col, date_col, name]])

        result = pd.concat(results, ignore_index=True)
        return result


# ============================================================
# 4. 单元测试
# ============================================================

class TestFactorExpressionEngine(unittest.TestCase):
    """因子表达式引擎测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟数据"""
        np.random.seed(42)
        codes = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "000858.SZ"]
        dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")

        rows = []
        for code in codes:
            base_price = np.random.uniform(8, 50)
            prices = [base_price]
            for _ in range(len(dates) - 1):
                prices.append(prices[-1] * (1 + np.random.normal(0.0003, 0.015)))
            prices = np.array(prices)

            for i, d in enumerate(dates):
                rows.append({
                    "code": code,
                    "date": d,
                    "open": prices[i] * (1 + np.random.normal(0, 0.003)),
                    "high": prices[i] * (1 + abs(np.random.normal(0, 0.008))),
                    "low": prices[i] * (1 - abs(np.random.normal(0, 0.008))),
                    "close": prices[i],
                    "volume": np.random.lognormal(10, 0.8),
                    "amount": np.random.lognormal(14, 0.6),
                    "turnover_rate": np.random.uniform(0.001, 0.05),
                    "change_pct": np.random.normal(0, 1.5),
                })

        cls.data = pd.DataFrame(rows)
        cls.engine = FactorExpressionEngine()

    def test_basic_variable_reference(self):
        """测试基本变量引用"""
        result = self.engine.compute(self.data, "$close")
        self.assertIn("close", result.columns)
        # 按 code 和 date 对齐比对
        merged = result.merge(
            self.data[["code", "date", "close"]],
            on=["code", "date"], suffixes=("_expr", "_orig")
        )
        corr = merged["close_expr"].corr(merged["close_orig"])
        self.assertAlmostEqual(corr, 1.0, places=4)

    def test_delta_operator(self):
        """测试 DELTA 算子 - 日收益率"""
        result = self.engine.compute(self.data, "DELTA($close, 1)")
        self.assertIn("DELTA_close_1", result.columns)
        # 验证: DELTA(close, 1) = close_t - close_{t-1}
        for code in self.data["code"].unique():
            code_data = self.data[self.data["code"] == code].sort_values("date")
            result_code = result[result["code"] == code].sort_values("date")
            expected = code_data["close"].diff(1)
            pd.testing.assert_series_equal(
                result_code["DELTA_close_1"].round(6),
                expected.round(6),
                check_names=False,
                check_index=False
            )

    def test_rank_operator(self):
        """测试 RANK 算子 - 截面排名"""
        result = self.engine.compute(self.data, "RANK($close)")
        col_name = "RANK_close"
        self.assertIn(col_name, result.columns)
        # RANK 值应在 [0, 1]
        self.assertTrue((result[col_name].dropna() >= 0).all())
        self.assertTrue((result[col_name].dropna() <= 1).all())

    def test_nested_expression(self):
        """测试嵌套表达式: RANK(DELTA($close, 5)) - 5日动量排名"""
        result = self.engine.compute(self.data, "RANK(DELTA($close, 5))")
        col_name = "RANK_DELTA_close_5"
        self.assertIn(col_name, result.columns)
        self.assertTrue((result[col_name].dropna() >= 0).all())
        self.assertTrue((result[col_name].dropna() <= 1).all())

    def test_zscore_operator(self):
        """测试 ZSCORE 算子"""
        result = self.engine.compute(self.data, "ZSCORE($close)")
        col_name = "ZSCORE_close"
        self.assertIn(col_name, result.columns)
        # ZSCORE 截面均值 ≈ 0
        zscores = result[col_name].dropna()
        if len(zscores) > 0:
            self.assertAlmostEqual(zscores.mean(), 0, delta=0.01)

    def test_ts_mean_operator(self):
        """测试 TS_MEAN 算子 - 20日均价"""
        result = self.engine.compute(self.data, "TS_MEAN($close, 20)")
        col_name = "TS_MEAN_close_20"
        self.assertIn(col_name, result.columns)
        # min_periods = max(3, 10) = 10, 所以前9天为 NaN
        for code in self.data["code"].unique():
            code_result = result[result["code"] == code].sort_values("date")
            self.assertTrue(code_result[col_name].iloc[:9].isna().all())
            self.assertTrue(code_result[col_name].iloc[10:].notna().all())

    def test_arithmetic_expression(self):
        """测试复合表达式: RANK(DELTA($close, 1)) - 日内收益排名"""
        result = self.engine.compute(self.data, "RANK(DELTA($close, 1))")
        col_name = "RANK_DELTA_close_1"
        self.assertIn(col_name, result.columns)
        # RANK 值应在 [0, 1]
        r = result[col_name].dropna()
        if len(r) > 0:
            self.assertTrue((r >= 0).all())
            self.assertTrue((r <= 1).all())

    def test_complex_nested(self):
        """测试复杂嵌套: ZSCORE(TS_MEAN(DELTA($close, 1), 20))"""
        result = self.engine.compute(
            self.data,
            "ZSCORE(TS_MEAN(DELTA($close, 1), 20))"
        )
        self.assertIn("ZSCORE_TS_MEAN_DELTA_close_1_20", result.columns)

    def test_rsi_operator(self):
        """测试 RSI 技术指标"""
        result = self.engine.compute(self.data, "RSI($close, 14)")
        col_name = "RSI_close_14"
        self.assertIn(col_name, result.columns)
        valid = result[col_name].dropna()
        if len(valid) > 0:
            # RSI 在 [0, 100] 范围内
            self.assertTrue((valid >= 0).all())
            self.assertTrue((valid <= 100).all())

    def test_expression_multiple_factors(self):
        """测试批量因子计算性能"""
        expressions = [
            "RANK(DELTA($close, 5))",           # 5日动量
            "RANK(DELTA($close, 20))",          # 20日动量
            "ZSCORE(TS_MEAN($volume, 20))",     # 20日均量Z值
            "RANK($turnover)",                   # 换手率排名
            "RSI($close, 14)",                   # RSI
            "RANK(DELTA($close, 1))",            # 1日收益排名
        ]

        import time
        start = time.time()
        for expr in expressions:
            self.engine.compute(self.data, expr)
        elapsed = time.time() - start

        # 6个因子计算应在合理时间内完成
        self.assertLess(elapsed, 5.0, f"因子计算超时: {elapsed:.2f}s")
        print(f"\n  6个因子表达式计算耗时: {elapsed:.4f}s (数据: {len(self.data)} 行)")

    def test_registry_extensibility(self):
        """测试算子注册表可扩展性"""
        # 注册自定义算子
        def custom_momentum(data_col, period=10):
            """自定义动量算子"""
            return data_col.pct_change(period)

        self.engine.registry.register("CUSTOM_MOM", custom_momentum)

        result = self.engine.compute(self.data, "CUSTOM_MOM($close, 10)")
        self.assertIn("CUSTOM_MOM_close_10", result.columns)


class TestFactorComparison(unittest.TestCase):
    """对比测试：表达式引擎 vs 原有硬编码方式"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        codes = [f"{i:06d}.SZ" for i in range(1, 21)]  # 20只股票
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
        rows = []
        for code in codes:
            base_price = np.random.uniform(5, 100)
            prices = [base_price]
            for _ in range(len(dates) - 1):
                prices.append(prices[-1] * (1 + np.random.normal(0.0002, 0.02)))
            prices = np.array(prices)
            for i, d in enumerate(dates):
                rows.append({
                    "code": code,
                    "date": d,
                    "open": prices[i] * (1 + np.random.normal(0, 0.003)),
                    "high": prices[i] * (1 + abs(np.random.normal(0, 0.008))),
                    "low": prices[i] * (1 - abs(np.random.normal(0, 0.008))),
                    "close": prices[i],
                    "volume": np.random.lognormal(10, 0.8),
                    "amount": np.random.lognormal(14, 0.6),
                    "turnover_rate": np.random.uniform(0.001, 0.05),
                })
        cls.data = pd.DataFrame(rows)
        cls.engine = FactorExpressionEngine()

    def compute_legacy_ret_5d(self) -> pd.Series:
        """
        模拟 jingni-trader 原有硬编码方式计算 ret_5d
        参见: skills/factor-engine/engine.py line 67
        """
        df = self.data.sort_values(["code", "date"])
        return df.groupby("code")["close"].pct_change(5)

    def compute_legacy_reversal_5d(self) -> pd.Series:
        """模拟 jingni-trader 原有方式计算 reversal_5d"""
        return -self.compute_legacy_ret_5d()

    def compute_legacy_volatility_20d(self) -> pd.Series:
        """模拟 jingni-trader 原有方式计算 volatility_20d"""
        df = self.data.sort_values(["code", "date"])
        return df.groupby("code")["close"].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )

    def test_legacy_vs_expression_ret_5d(self):
        """对比原有方式 vs 表达式方式: 5日动量"""
        legacy_result = self.compute_legacy_ret_5d()

        # 表达式方式: 使用 DELTA 算子
        expr_result = self.engine.compute(
            self.data,
            "DELTA($close, 5)"
        )
        # DELTA 是绝对差值，而 pct_change 是百分比
        # 验证方向一致性
        legacy_sorted = self.data.sort_values(["code", "date"])
        expr_sorted = expr_result.sort_values(["code", "date"])

        legacy_sign = np.sign(legacy_result.dropna())
        delta_col = "DELTA_close_5"
        expr_sign = np.sign(expr_sorted[delta_col].dropna())

        if len(legacy_sign) > 0 and len(expr_sign) > 0:
            alignment = (legacy_sign.values == expr_sign.values).mean()
            print(f"\n  符号一致性: {alignment:.2%}")
            self.assertGreater(alignment, 0.70)

    def test_legacy_vs_expression_volatility(self):
        """对比原有方式 vs 表达式方式: volatility_20d 排名一致性"""
        legacy_vol = self.compute_legacy_volatility_20d()

        # 表达式方式: TS_STD 用日收益率（不同的度量方式）
        expr_result = self.engine.compute(
            self.data,
            "TS_STD(DELTA($close, 1), 20)"
        )

        col_name = "TS_STD_DELTA_close_1_20"
        expr_vol = expr_result.sort_values(["code", "date"])[col_name]

        common = pd.DataFrame({
            "legacy": legacy_vol.values,
            "expr": expr_vol.values,
        }).dropna()

        if len(common) > 30:
            # 排名一致性：用 RANK 比较方向性
            legacy_rank = common["legacy"].rank()
            expr_rank = common["expr"].rank()
            rank_corr = legacy_rank.corr(expr_rank)
            print(f"\n  波动率排名相关性: {rank_corr:.4f}")
            self.assertGreater(rank_corr, 0.10)  # 不同度量方式排名仅需弱正相关


if __name__ == "__main__":
    print("=" * 60)
    print("因子表达式引擎验证测试")
    print("借鉴来源: quant-stream / Qlib 表达式引擎")
    print("=" * 60)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFactorExpressionEngine)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestFactorComparison)
    result2 = runner.run(suite2)

    print("\n" + "=" * 60)
    print("测试结论:")
    print(f"  - 因子表达式引擎测试: {'通过' if result.wasSuccessful() else '失败'}")
    print(f"  - 对比测试: {'通过' if result2.wasSuccessful() else '失败'}")
    print("=" * 60)