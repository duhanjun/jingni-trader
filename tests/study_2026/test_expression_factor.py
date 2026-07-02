"""
验证测试：表达式引擎驱动的因子定义系统
====================================================
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
优化方向: factor-engine - 将硬编码因子计算改造为表达式驱动的声明式因子定义
日期: 2026-06-14

Qlib 的表达式引擎亮点：
  - 使用 DSL 语法定义因子，如 "$close", "Ref($close, 20) / $close - 1"
  - 因子被声明为函数而非数据，便于 LLM Agent 生成和自动迭代
  - 支持向量化计算，内置 30+ 操作符（Ref, Mean, Std, Max, Min, Sum, Rank 等）
  - Alpha158/Alpha360 内置因子库覆盖常见量价因子

本测试验证：
  1. 表达式解析器的正确性（支持变量引用、嵌套函数、算术运算）
  2. 向量化计算的正确性和性能
  3. 与现有硬编码因子计算的一致性对比
  4. 可扩展性：新因子只需一行表达式字符串
"""

import unittest
import sys
import os
import time
import warnings
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import re
import operator as op

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# =====================================================
# 表达式引擎核心实现（参考 Qlib expression engine）
# =====================================================

@dataclass
class ExprContext:
    """表达式求值上下文，持有数据变量和计算结果缓存"""
    data: Dict[str, pd.Series] = field(default_factory=dict)
    _cache: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        return self.data.get(name)


# 内置操作符映射
_OPS = {
    '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv,
    '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le,
    '==': op.eq, '!=': op.ne,
    '&': op.and_, '|': op.or_,
}

# 内置函数实现
def _ref(series: pd.Series, n: int) -> pd.Series:
    """前向引用（lag），Ref($close, 5) 返回 5 天前的收盘价"""
    return series.shift(n)

def _mean(series: pd.Series, n: int) -> pd.Series:
    """滚动均值"""
    return series.rolling(n, min_periods=max(1, n // 2)).mean()

def _std(series: pd.Series, n: int) -> pd.Series:
    """滚动标准差"""
    return series.rolling(n, min_periods=max(1, n // 2)).std()

def _max(series: pd.Series, n: int) -> pd.Series:
    """滚动最大值"""
    return series.rolling(n, min_periods=max(1, n // 2)).max()

def _min(series: pd.Series, n: int) -> pd.Series:
    """滚动最小值"""
    return series.rolling(n, min_periods=max(1, n // 2)).min()

def _sum(series: pd.Series, n: int) -> pd.Series:
    """滚动求和"""
    return series.rolling(n, min_periods=max(1, n // 2)).sum()

def _rank(series: pd.Series) -> pd.Series:
    """截面排名（百分比）"""
    return series.rank(pct=True)

def _ts_rank(series: pd.Series, n: int) -> pd.Series:
    """时序排名"""
    return series.rolling(n, min_periods=max(1, n // 2)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

def _delta(series: pd.Series, n: int) -> pd.Series:
    """n 周期差分"""
    return series - series.shift(n)

def _pct_change(series: pd.Series, n: int = 1) -> pd.Series:
    """n 周期收益率"""
    return series.pct_change(n)

def _correlation(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
    """滚动相关系数"""
    return s1.rolling(n, min_periods=max(1, n // 2)).corr(s2)

def _abs(series: pd.Series) -> pd.Series:
    """绝对值"""
    return series.abs()

def _log(series: pd.Series) -> pd.Series:
    """对数"""
    return np.log(series.replace(0, np.nan))

def _sign(series: pd.Series) -> pd.Series:
    """符号"""
    return np.sign(series)

# 内置函数注册表
_BUILTIN_FUNCTIONS = {
    'Ref': _ref,
    'Mean': _mean,
    'Std': _std,
    'Max': _max,
    'Min': _min,
    'Sum': _sum,
    'Rank': _rank,
    'TsRank': _ts_rank,
    'Delta': _delta,
    'PctChange': _pct_change,
    'Correlation': _correlation,
    'Abs': _abs,
    'Log': _log,
    'Sign': _sign,
}


class ExpressionParseError(Exception):
    """表达式解析错误"""
    pass


class FactorExpression:
    """
    因子表达式解析器和求值器

    支持的语法:
      - 变量引用: $close, $volume, $high, $low, $open, $amount, $turnover
      - 函数调用: Ref($close, 20), Mean($close, 5), Std($close, 20)
      - 算术运算: +, -, *, /
      - 比较运算: >, <, >=, <=, ==, !=
      - 逻辑运算: &, |
      - 常量和括号: 1, 2.5, (a + b) / c
      - 嵌套表达式: Mean($high - $low, 20) / $close

    使用示例:
      >>> expr = FactorExpression("$close / Ref($close, 20) - 1")
      >>> result = expr.evaluate({"close": close_series, ...})
    """

    # Token 模式
    TOKEN_PATTERN = re.compile(
        r'\$(\w+)|'           # 变量: $close, $volume
        r'([A-Z][a-zA-Z]*)\(|'  # 函数名: Ref(, Mean(
        r'(\d+\.?\d*)|'       # 数字: 1, 2.5, 100
        r'([+\-*/()><=&|!]=?)|' # 运算符和括号
        r'(,)|'               # 逗号
        r'(\s+)'              # 空白
    )

    def __init__(self, expression: str, name: str = ""):
        self.expression = expression.strip()
        self.name = name or self.expression
        self._tokens = None
        self._ast = None
        self._parse()

    def _parse(self):
        """解析表达式为 token 流"""
        tokens = []
        for m in self.TOKEN_PATTERN.finditer(self.expression):
            if m.group(1):   # 变量
                tokens.append(('VAR', m.group(1)))
            elif m.group(2): # 函数名
                tokens.append(('FUNC', m.group(2)))
            elif m.group(3): # 数字 - 整数解析为 int，否则 float
                num_str = m.group(3)
                if '.' in num_str:
                    tokens.append(('NUM', float(num_str)))
                else:
                    tokens.append(('NUM', int(num_str)))
            elif m.group(4): # 运算符/括号
                tokens.append(('OP', m.group(4)))
            elif m.group(5): # 逗号
                tokens.append(('COMMA', ','))
            # 跳过空白

        if not tokens:
            raise ExpressionParseError(f"表达式为空: {self.expression}")

        self._tokens = tokens
        self._pos = 0

    def _peek(self):
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _next(self):
        tok = self._peek()
        if tok:
            self._pos += 1
        return tok

    def evaluate(self, data: Dict[str, pd.Series]) -> pd.Series:
        """
        对给定数据求值表达式

        参数:
            data: 列名到 Series 的映射，如 {"close": ..., "volume": ..., "high": ...}

        返回:
            计算结果 Series
        """
        self._pos = 0
        ctx = ExprContext(data=data)
        return self._eval_expr(ctx)

    def _eval_expr(self, ctx: ExprContext):
        """递归下降解析器：expr -> term {('+'|'-') term}"""
        left = self._eval_term(ctx)
        while True:
            tok = self._peek()
            if tok and tok[0] == 'OP' and tok[1] in ('+', '-'):
                self._next()
                right = self._eval_term(ctx)
                if tok[1] == '+':
                    left = left + right
                else:
                    left = left - right
            else:
                break
        return left

    def _eval_term(self, ctx: ExprContext):
        """term -> factor {('*'|'/') factor}"""
        left = self._eval_factor(ctx)
        while True:
            tok = self._peek()
            if tok and tok[0] == 'OP' and tok[1] in ('*', '/'):
                self._next()
                right = self._eval_factor(ctx)
                if tok[1] == '*':
                    left = left * right
                else:
                    left = left / right.replace(0, np.nan)
            else:
                break
        return left

    def _eval_factor(self, ctx: ExprContext):
        """factor -> atom | func_call"""
        tok = self._peek()
        if tok is None:
            raise ExpressionParseError(f"表达式意外结束: {self.expression}")

        if tok[0] == 'OP' and tok[1] == '(':
            self._next()
            result = self._eval_expr(ctx)
            next_tok = self._peek()
            if next_tok and next_tok[0] == 'OP' and next_tok[1] == ')':
                self._next()
            return result

        if tok[0] == 'FUNC':
            return self._eval_func_call(ctx)

        return self._eval_atom(ctx)

    def _eval_atom(self, ctx: ExprContext):
        """atom -> VAR | NUM | '-' atom"""
        tok = self._next()
        if tok is None:
            raise ExpressionParseError("预期变量或数字")

        if tok[0] == 'OP' and tok[1] == '-':
            inner = self._eval_factor(ctx)
            return -inner

        if tok[0] == 'VAR':
            val = ctx.data.get(tok[1])
            if val is None:
                raise ExpressionParseError(f"未找到变量: ${tok[1]}")
            return val

        if tok[0] == 'NUM':
            # 常量直接返回数值，Python 自动广播会处理
            return tok[1]

        raise ExpressionParseError(f"未预期的 token: {tok}")

    def _eval_func_call(self, ctx: ExprContext):
        """func_call -> FUNC '(' [arg {',' arg}] ')'"""
        func_name = self._next()[1]  # 已经消耗了 FUNC token

        if func_name not in _BUILTIN_FUNCTIONS:
            raise ExpressionParseError(f"未知函数: {func_name}")

        args = []
        while True:
            tok = self._peek()
            if tok is None:
                raise ExpressionParseError(f"函数 {func_name} 参数不完整")

            if tok[0] == 'OP' and tok[1] == ')':
                self._next()
                break

            if tok[0] == 'COMMA':
                self._next()
                continue

            args.append(self._eval_expr(ctx))

        try:
            func = _BUILTIN_FUNCTIONS[func_name]
            result = func(*args)
            if isinstance(result, (np.floating, float)):
                result = pd.Series(result, index=args[0].index if args else None)
            return result
        except Exception as e:
            raise ExpressionParseError(f"函数 {func_name} 执行失败: {e}") from e

    def __repr__(self):
        return f"FactorExpression('{self.expression}')"


class FactorRegistry:
    """
    因子注册表：管理所有已注册的因子表达式

    借鉴 Qlib Alpha158/Alpha360 设计，维护一个可扩展的因子库。
    新因子只需一行表达式即可注册，无需修改任何计算逻辑。
    """

    def __init__(self):
        self._factors: Dict[str, FactorExpression] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, name: str, expression: str, category: str = "custom"):
        """注册一个因子"""
        expr = FactorExpression(expression, name)
        self._factors[name] = expr
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        return expr

    def register_batch(self, factors: Dict[str, tuple]):
        """批量注册因子，格式: {name: (expression, category)}"""
        for name, (expr_str, cat) in factors.items():
            self.register(name, expr_str, cat)

    def compute_all(self, data: Dict[str, pd.Series]) -> pd.DataFrame:
        """计算所有已注册因子"""
        result = {}
        for name, expr in self._factors.items():
            result[name] = expr.evaluate(data)
        return pd.DataFrame(result)

    def compute_selected(self, data: Dict[str, pd.Series],
                         names: List[str]) -> pd.DataFrame:
        """计算指定因子"""
        result = {}
        for name in names:
            if name in self._factors:
                result[name] = self._factors[name].evaluate(data)
        return pd.DataFrame(result)

    def list_factors(self) -> Dict[str, List[str]]:
        """列出所有因子（按类别分组）"""
        return self._categories.copy()

    def get_expression(self, name: str) -> Optional[str]:
        """获取因子表达式"""
        if name in self._factors:
            return self._factors[name].expression
        return None


# =====================================================
# 预定义因子库（借鉴 Qlib Alpha158 设计）
# =====================================================

def create_default_factor_registry() -> FactorRegistry:
    """创建包含标准因子的注册表"""
    registry = FactorRegistry()

    # 收益率因子 (price)
    registry.register_batch({
        "ret_1d":   ("$close / Ref($close, 1) - 1", "price"),
        "ret_5d":   ("$close / Ref($close, 5) - 1", "price"),
        "ret_20d":  ("$close / Ref($close, 20) - 1", "price"),
        "ret_60d":  ("$close / Ref($close, 60) - 1", "price"),
    })

    # 反转因子 (reversal)
    registry.register_batch({
        "reversal_5d":  ("-($close / Ref($close, 5) - 1)", "reversal"),
        "reversal_20d": ("-($close / Ref($close, 20) - 1)", "reversal"),
    })

    # 波动率因子 (volatility)
    registry.register_batch({
        "volatility_20d": ("Std($close / Ref($close, 1) - 1, 20)", "volatility"),
    })

    # 成交量因子 (volume)
    registry.register_batch({
        "volume_ratio":  ("$volume / Mean($volume, 20)", "volume"),
        "volume_20d":    ("Mean($volume, 20)", "volume"),
    })

    # 价格形态因子 (pattern)
    registry.register_batch({
        "high_low_range": (("($high - $low) / $close", "pattern")),
        "close_position": (("($close - $low) / ($high - $low + 1e-8)", "pattern")),
        "ma_deviation":   (("$close / Mean($close, 20) - 1", "pattern")),
        "ma_cross":       (("Mean($close, 5) - Mean($close, 20)", "pattern")),
    })

    # 流动性因子（需要 amount 和 turnover 列）
    registry.register_batch({
        "turnover_change": ("Mean($turnover, 5) / Mean($turnover, 20) - 1", "liquidity"),
    })

    return registry


# =====================================================
# 单元测试
# =====================================================

class TestExpressionEngine(unittest.TestCase):
    """测试表达式解析引擎"""

    @classmethod
    def setUpClass(cls):
        """创建模拟数据"""
        np.random.seed(42)
        n = 200
        cls.data = {
            "close": pd.Series(10 + np.cumsum(np.random.normal(0, 0.5, n)), name="close"),
            "volume": pd.Series(np.random.lognormal(10, 0.5, n), name="volume"),
            "high": pd.Series(np.zeros(n), name="high"),
            "low": pd.Series(np.zeros(n), name="low"),
            "turnover": pd.Series(np.random.uniform(0.01, 0.1, n), name="turnover"),
        }
        # 确保 high > low > close 合理
        cls.data["high"] = cls.data["close"] * (1 + np.abs(np.random.normal(0, 0.02, n)))
        cls.data["low"] = cls.data["close"] * (1 - np.abs(np.random.normal(0, 0.02, n)))

    def test_simple_variable(self):
        """测试简单变量引用"""
        expr = FactorExpression("$close")
        result = expr.evaluate(self.data)
        pd.testing.assert_series_equal(result, self.data["close"])

    def test_constant(self):
        """测试常量"""
        expr = FactorExpression("3.14")
        result = expr.evaluate(self.data)
        self.assertAlmostEqual(float(result), 3.14)

    def test_arithmetic(self):
        """测试算术运算"""
        expr = FactorExpression("$close + $volume - $close * 2")
        result = expr.evaluate(self.data)
        expected = self.data["close"] + self.data["volume"] - self.data["close"] * 2
        pd.testing.assert_series_equal(result.round(6), expected.round(6))

    def test_ref_function(self):
        """测试 Ref 函数"""
        expr = FactorExpression("Ref($close, 5)")
        result = expr.evaluate(self.data)
        expected = self.data["close"].shift(5)
        # NaN 位置比较
        self.assertTrue(result.dropna().equals(expected.dropna()))

    def test_mean_function(self):
        """测试 Mean 函数"""
        expr = FactorExpression("Mean($close, 20)")
        result = expr.evaluate(self.data)
        expected = self.data["close"].rolling(20, min_periods=10).mean()
        pd.testing.assert_series_equal(result.round(6), expected.round(6))

    def test_std_function(self):
        """测试 Std 函数"""
        expr = FactorExpression("Std($close, 10)")
        result = expr.evaluate(self.data)
        expected = self.data["close"].rolling(10, min_periods=5).std()
        pd.testing.assert_series_equal(result.round(6), expected.round(6))

    def test_nested_expression(self):
        """测试嵌套表达式"""
        # 20日收益率 = close / Ref(close, 20) - 1
        expr = FactorExpression("$close / Ref($close, 20) - 1")
        result = expr.evaluate(self.data)
        expected = self.data["close"] / self.data["close"].shift(20) - 1
        pd.testing.assert_series_equal(result.round(6), expected.round(6))

    def test_complex_expression(self):
        """测试复杂表达式"""
        # (high - low) / close * Std(close/Ref(close,1)-1, 20)
        expr = FactorExpression(
            "($high - $low) / $close * Std($close / Ref($close, 1) - 1, 20)"
        )
        result = expr.evaluate(self.data)
        ret_1d = self.data["close"] / self.data["close"].shift(1) - 1
        expected = (self.data["high"] - self.data["low"]) / self.data["close"] * \
                   ret_1d.rolling(20, min_periods=10).std()
        pd.testing.assert_series_equal(result.round(6), expected.round(6))

    def test_parentheses_grouping(self):
        """测试括号优先级"""
        expr1 = FactorExpression("$close - $volume - 1")
        expr2 = FactorExpression("$close - ($volume - 1)")
        r1 = expr1.evaluate(self.data)
        r2 = expr2.evaluate(self.data)
        # 两个表达式应该不同（括号改变了运算顺序）
        self.assertFalse(r1.equals(r2))

    def test_expression_error_handling(self):
        """测试错误处理"""
        expr1 = FactorExpression("UnknownFunc($close, 5)")
        with self.assertRaises(ExpressionParseError):
            expr1.evaluate(self.data)
        expr2 = FactorExpression("$nonexistent_variable")
        with self.assertRaises(ExpressionParseError):
            expr2.evaluate(self.data)

    def test_expression_repr(self):
        """测试表达式字符串表示"""
        expr = FactorExpression("$close / Ref($close, 20) - 1", "ret_20d")
        self.assertIn("$close", repr(expr))
        self.assertTrue("FactorExpression" in repr(expr))


class TestFactorRegistry(unittest.TestCase):
    """测试因子注册表"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 200
        cls.data = {
            "close": pd.Series(10 + np.cumsum(np.random.normal(0, 0.5, n))),
            "volume": pd.Series(np.random.lognormal(10, 0.5, n)),
            "high": cls.data if False else pd.Series(10.5 + np.cumsum(np.random.normal(0, 0.5, n))),
            "low": cls.data if False else pd.Series(9.5 + np.cumsum(np.random.normal(0, 0.5, n))),
            "turnover": pd.Series(np.random.uniform(0.01, 0.1, n)),
        }

    def setUp(self):
        np.random.seed(42)
        n = 200
        self.data = {
            "close": pd.Series(10 + np.cumsum(np.random.normal(0, 0.5, n))),
            "volume": pd.Series(np.random.lognormal(10, 0.5, n)),
            "high": pd.Series(10.5 + np.cumsum(np.random.normal(0, 0.5, n))),
            "low": pd.Series(9.5 + np.cumsum(np.random.normal(0, 0.5, n))),
            "turnover": pd.Series(np.random.uniform(0.01, 0.1, n)),
        }

    def test_registry_create_default(self):
        """测试默认因子注册表创建"""
        reg = create_default_factor_registry()
        factors = reg.list_factors()
        self.assertIn("price", factors)
        self.assertIn("volume", factors)
        self.assertIn("pattern", factors)
        self.assertTrue(len(factors["price"]) >= 4)

    def test_compute_all_factors(self):
        """测试计算所有因子"""
        reg = create_default_factor_registry()
        result = reg.compute_all(self.data)
        self.assertGreater(len(result.columns), 10)
        self.assertEqual(len(result), len(self.data["close"]))

    def test_custom_factor_registration(self):
        """测试自定义因子注册"""
        reg = create_default_factor_registry()
        reg.register("my_factor", "$close * $volume", "custom")
        result = reg.compute_selected(self.data, ["my_factor"])
        self.assertIn("my_factor", result.columns)

    def test_factor_listing(self):
        """测试因子列表"""
        reg = create_default_factor_registry()
        factors = reg.list_factors()
        total = sum(len(v) for v in factors.values())
        self.assertGreater(total, 10)

    def test_expression_retrieval(self):
        """测试表达式检索"""
        reg = create_default_factor_registry()
        expr_str = reg.get_expression("ret_20d")
        self.assertIsNotNone(expr_str)
        self.assertIn("Ref", expr_str)


class TestConsistencyWithExistingCode(unittest.TestCase):
    """
    验证表达式引擎与现有硬编码因子计算的一致性

    对照现有 factor-engine/engine.py 中的 compute_a_share_factors 方法
    """

    @classmethod
    def setUpClass(cls):
        """创建与现有代码相同格式的面板数据"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
        codes = [f"{i:06d}.SH" for i in range(600000, 600010)]

        rows = []
        for code in codes:
            start_price = np.random.uniform(8, 50)
            returns = np.random.normal(0.0002, 0.02, len(dates))
            prices = start_price * np.cumprod(1 + returns)

            df_one = pd.DataFrame({
                "date": dates,
                "code": code,
                "close": prices,
                "volume": np.random.lognormal(15, 0.5, len(dates)),
                "amount": np.random.lognormal(19, 0.5, len(dates)),
            })
            df_one["high"] = df_one["close"] * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
            df_one["low"] = df_one["close"] * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
            df_one["turnover_rate"] = np.random.uniform(0.005, 0.05, len(dates))
            rows.append(df_one)

        cls.panel_data = pd.concat(rows, ignore_index=True).sort_values(["code", "date"])

    def test_ret_20d_consistency(self):
        """验证 ret_20d 因子一致性"""
        df = self.panel_data.copy()
        # 现有方式计算
        existing = df.groupby("code")["close"].pct_change(20)
        existing.name = "ret_20d_existing"

        # 表达式引擎计算
        reg = create_default_factor_registry()
        results = []
        for code, group in df.groupby("code"):
            data_dict = {
                "close": group["close"].reset_index(drop=True),
                "volume": group["volume"].reset_index(drop=True),
                "high": group["high"].reset_index(drop=True),
                "low": group["low"].reset_index(drop=True),
                "turnover": group["turnover_rate"].reset_index(drop=True),
            }
            expr_result = reg.compute_selected(data_dict, ["ret_20d"])
            expr_result["code"] = code
            expr_result["date"] = group["date"].values
            results.append(expr_result)

        expr_df = pd.concat(results, ignore_index=True)

        # 逐一比较每只股票的因子值
        for code in expr_df["code"].unique():
            code_mask = df["code"] == code
            existing_vals = existing[code_mask].reset_index(drop=True)
            expr_vals = expr_df[expr_df["code"] == code]["ret_20d"].reset_index(drop=True)
            # 对齐 NaN 后比较
            mask = existing_vals.notna() & expr_vals.notna()
            if mask.sum() > 10:
                max_diff = (existing_vals[mask] - expr_vals[mask]).abs().max()
                self.assertLess(max_diff, 1e-10,
                    f"code={code}: ret_20d 最大差异 {max_diff} 超出容差")


class TestPerformanceComparison(unittest.TestCase):
    """性能对比：表达式引擎 vs 硬编码计算"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_codes = 50
        n_dates = 500
        cls.data = {
            "close": pd.Series(np.random.normal(0, 1, n_dates * n_codes)),
            "volume": pd.Series(np.random.lognormal(10, 0.5, n_dates * n_codes)),
            "high": pd.Series(np.random.normal(0, 1, n_dates * n_codes)),
            "low": pd.Series(np.random.normal(0, 1, n_dates * n_codes)),
            "turnover": pd.Series(np.random.uniform(0.01, 0.1, n_dates * n_codes)),
        }

    def test_computation_time(self):
        """测试计算时间"""
        reg = create_default_factor_registry()

        start = time.perf_counter()
        _ = reg.compute_all(self.data)
        elapsed = time.perf_counter() - start

        # 50 个标的 × 500 天，计算 15+ 个因子应在合理时间内完成
        self.assertLess(elapsed, 5.0,
            f"因子计算耗时 {elapsed:.2f}s 超过 5s 阈值")
        print(f"\n  因子计算性能: {elapsed:.3f}s (50标的×500天, 15+因子)")


if __name__ == "__main__":
    print("=" * 60)
    print("表达式引擎驱动的因子定义系统 - 验证测试")
    print("借鉴来源: Microsoft Qlib (expression engine)")
    print("=" * 60)
    unittest.main(verbosity=2)