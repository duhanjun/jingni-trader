"""
验证代码 - 表达式驱动的因子计算引擎
借鉴来源: Microsoft Qlib (表达式引擎 DSL)
         vnpy vnpy.alpha.dataset (因子表达式计算)
优化方向: 将 factor-engine 中的硬编码因子计算替换为声明式表达式系统
日期: 2026-06-13

Qlib 核心设计:
  因子通过领域特定语言(DSL)声明，如:
  - $close, $open, $high, $low, $volume (原始字段)
  - Ref($close, 1)  (前移/后移)
  - Mean($close, 20) (滚动均值)
  - Std($close, 20)  (滚动标准差)
  - $close / $open - 1 (算术表达式)
  
  这种设计的好处:
  1. 因子定义与计算逻辑解耦
  2. 用户可以无需修改引擎代码即可添加新因子
  3. 因子表达式可序列化/可版本管理
  4. 便于批量并行计算优化

验证目标:
  1. 表达式解析正确性
  2. 计算结果与现有硬编码逻辑一致性
  3. 性能对比（批量表达式 vs 逐列计算）
"""

import sys
import os
import time
import json
import unittest
from typing import Dict, List, Callable, Any, Optional, Union
from functools import lru_cache

import numpy as np
import pandas as pd


# ============================================================
# 表达式引擎实现（借鉴 Qlib 设计）
# ============================================================

class ExprContext:
    """表达式求值上下文，维护字段引用和计算注册表"""

    # 内置运算符映射
    OPERATORS = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: np.where(b != 0, a / b, np.nan),
    }

    # 内置函数注册表
    BUILTIN_FUNCTIONS: Dict[str, Callable] = {}

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._fields: Dict[str, np.ndarray] = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册原始字段为可引用变量"""
        for col in self.df.columns:
            self._fields[col] = self.df[col].values

    @classmethod
    def register_function(cls, name: str, func: Callable):
        """注册自定义函数"""
        cls.BUILTIN_FUNCTIONS[name] = func

    def get_field(self, name: str) -> np.ndarray:
        if name in self._fields:
            return self._fields[name]
        raise KeyError(f"未注册的字段: {name}")

    def set_field(self, name: str, data: np.ndarray):
        self._fields[name] = data


# ---- 内置函数定义 ----

def _ref(data: np.ndarray, period: int) -> np.ndarray:
    """前移: Ref(close, 1) = 前一日收盘价"""
    data = np.array(data, copy=False, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    if period > 0:
        result[period:] = data[:-period]
    elif period < 0:
        result[:period] = data[-period:]
    else:
        result = data.copy()
    return result


def _rolling_mean(data: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """滚动均值"""
    s = pd.Series(data)
    return s.rolling(window, min_periods=min_periods).mean().values


def _rolling_std(data: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """滚动标准差"""
    result = np.full_like(data, np.nan, dtype=float)
    s = pd.Series(data)
    result = s.rolling(window, min_periods=min_periods).std().values
    return result


def _pct_change(data: np.ndarray, period: int = 1) -> np.ndarray:
    """变化率"""
    data = np.array(data, copy=False, dtype=float)
    prev = _ref(data, period)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(prev != 0, data / prev - 1, np.nan)
    return result


def _rank_pct(data: np.ndarray) -> np.ndarray:
    """百分位排名"""
    from scipy.stats import rankdata
    valid = ~np.isnan(data)
    result = np.full_like(data, np.nan, dtype=float)
    if valid.sum() > 0:
        result[valid] = rankdata(data[valid]) / valid.sum()
    return result


def _ts_sum(data: np.ndarray, window: int) -> np.ndarray:
    """滚动求和"""
    result = np.full_like(data, np.nan, dtype=float)
    s = pd.Series(data)
    result = s.rolling(window, min_periods=1).sum().values
    return result


def _ts_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """滚动相关系数"""
    result = np.full_like(a, np.nan, dtype=float)
    s_a = pd.Series(a)
    s_b = pd.Series(b)
    result = s_a.rolling(window).corr(s_b).values
    return result


# 注册内置函数
for _name, _func in [
    ('Ref', _ref),
    ('Mean', _rolling_mean),
    ('Std', _rolling_std),
    ('PctChange', _pct_change),
    ('RankPct', _rank_pct),
    ('Sum', _ts_sum),
    ('Corr', _ts_corr),
]:
    ExprContext.register_function(_name, _func)


# ============================================================
# 因子表达式定义（声明式，可序列化）
# ============================================================

class FactorExpression:
    """单个因子表达式"""

    def __init__(self, name: str, expression: str, description: str = ""):
        self.name = name
        self.expression = expression
        self.description = description

    def to_dict(self) -> Dict:
        return {"name": self.name, "expression": self.expression, "description": self.description}


# 借鉴 Qlib Alpha158 + vnpy AlphaDataset 设计的因子库
# 注意：这是声明式定义，不包含计算逻辑
FACTOR_LIBRARY = [
    # ---- 收益类因子 ----
    FactorExpression("ret_1d", "PctChange(close, 1)", "1日收益率"),
    FactorExpression("ret_5d", "PctChange(close, 5)", "5日收益率"),
    FactorExpression("ret_20d", "PctChange(close, 20)", "20日收益率"),
    FactorExpression("ret_60d", "PctChange(close, 60)", "60日收益率"),

    # ---- 反转因子 ----
    FactorExpression("reversal_5d", "-PctChange(close, 5)", "5日反转"),
    FactorExpression("reversal_20d", "-PctChange(close, 20)", "20日反转"),

    # ---- 波动率因子 ----
    FactorExpression("volatility_20d", "Std(PctChange(close, 1), 20)", "20日波动率"),
    FactorExpression("volatility_60d", "Std(PctChange(close, 1), 60)", "60日波动率"),

    # ---- 成交量因子 ----
    FactorExpression("volume_ratio_5d", "volume / Mean(volume, 5)", "5日量比"),
    FactorExpression("volume_ratio_20d", "volume / Mean(volume, 20)", "20日量比"),
    FactorExpression("volume_trend", "Corr(close, volume, 10)", "量价相关性"),

    # ---- 均线因子 ----
    FactorExpression("ma_diff_5_20", "Mean(close, 5) / Mean(close, 20) - 1", "短长期均线偏离"),
    FactorExpression("ma_diff_10_60", "Mean(close, 10) / Mean(close, 60) - 1", "中期均线偏离"),

    # ---- 换手率因子 ----
    FactorExpression("turnover_5d", "Mean(turnover_rate, 5)", "5日平均换手率"),
    FactorExpression("turnover_20d", "Mean(turnover_rate, 20)", "20日平均换手率"),
    FactorExpression("turnover_change", "Mean(turnover_rate, 5) / Mean(turnover_rate, 20) - 1", "换手率变化"),

    # ---- 价量复合因子 ----
    FactorExpression("amount_ma_5", "Mean(amount, 5)", "5日均成交额"),
    FactorExpression("amount_ratio", "amount / Mean(amount, 20)", "成交额相对变化"),

    # ---- 波幅因子 ----
    FactorExpression("amplitude", "(high - low) / Ref(close, 1)", "日内振幅"),
    FactorExpression("amplitude_ma_5", "Mean((high - low) / Ref(close, 1), 5)", "5日均振幅"),
]


# ============================================================
# 表达式计算引擎
# ============================================================

class ExpressionFactorEngine:
    """表达式驱动的因子计算引擎

    借鉴 Qlib 的 expression engine 设计理念：
    - 因子定义与计算逻辑分离
    - 支持分组计算（按股票代码分组）
    - 批量计算优化
    """

    def __init__(self, factor_library: List[FactorExpression] = None):
        self.library = factor_library or FACTOR_LIBRARY

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算因子库中的所有因子"""
        result = df[['code', 'date']].copy()

        # 按股票代码分组，每组独立计算因子
        for code, group in df.groupby('code'):
            ctx = ExprContext(group.reset_index(drop=True))
            mask = result['code'] == code

            for factor_expr in self.library:
                try:
                    values = self._evaluate(ctx, factor_expr.expression)
                    # 确保值是可写的副本（避免 np.where 等产生的只读数组）
                    values = np.array(values, copy=True)
                    result.loc[mask, factor_expr.name] = values
                except Exception as e:
                    # 跳过无法计算的因子（如字段缺失）
                    pass

        return result

    def _find_matching_paren(self, expr: str, start: int) -> int:
        """找到与 start 位置 '(' 匹配的 ')'"""
        depth = 0
        for i in range(start, len(expr)):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _is_wrapped_parens(self, expression: str) -> bool:
        """检查表达式是否被外层括号包裹，如 (high - low)"""
        if expression.startswith('(') and expression.endswith(')'):
            match = self._find_matching_paren(expression, 0)
            return match == len(expression) - 1
        return False

    def _evaluate(self, ctx: ExprContext, expression: str) -> np.ndarray:
        """递归解析并计算表达式"""
        expression = expression.strip()

        # 处理外括号包裹
        if self._is_wrapped_parens(expression):
            return self._evaluate(ctx, expression[1:-1])

        # 处理负数前缀: -PctChange(...)或 -(high - low)
        if expression.startswith('-') and len(expression) > 1:
            rest = expression[1:].strip()
            # 检查是否是简单的负函数调用
            if self._is_wrapped_parens(rest) or '(' in rest:
                inner = self._evaluate(ctx, rest)
                return -inner

        # 处理函数调用: FuncName(arg1, arg2, ...)
        # 仅当表达式不包含顶层的二元运算符时才当作函数调用处理
        if '(' in expression:
            first_paren = expression.index('(')
            func_name = expression[:first_paren].strip()
            match = self._find_matching_paren(expression, first_paren)
            if func_name and match == len(expression) - 1 and func_name in ExprContext.BUILTIN_FUNCTIONS:
                args_str = expression[first_paren + 1:match]
                args = self._parse_args(ctx, args_str)
                return ExprContext.BUILTIN_FUNCTIONS[func_name](*args)

        # 处理二元运算符
        for op in ['+', '-', '*', '/']:
            if op == '-' and expression.startswith('-'):
                continue  # 负数前缀已处理
            # 简单分割（不处理嵌套括号精确分割）
            parts = self._split_operator(expression, op)
            if len(parts) >= 2:
                left = self._evaluate(ctx, op.join(parts[:-1]))
                right = self._evaluate(ctx, parts[-1])
                return ExprContext.OPERATORS[op](left, right)

        # 字段引用
        if expression in ctx._fields:
            return ctx._fields[expression]

        # 数字字面量
        try:
            return np.full(len(ctx.df), float(expression))
        except ValueError:
            pass

        raise ValueError(f"无法解析表达式: {expression}")

    def _split_operator(self, expr: str, op: str) -> List[str]:
        """简单按运算符分割（不处理嵌套括号深度）"""
        depth = 0
        last_split = 0
        parts = []
        for i, ch in enumerate(expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == op and depth == 0:
                parts.append(expr[last_split:i].strip())
                last_split = i + 1
        parts.append(expr[last_split:].strip())
        return parts

    def _parse_args(self, ctx: ExprContext, args_str: str) -> List[Any]:
        """解析函数参数"""
        args = []
        depth = 0
        current = ""
        for ch in args_str:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current.strip())

        resolved = []
        for arg in args:
            try:
                resolved.append(int(arg))
            except ValueError:
                try:
                    resolved.append(float(arg))
                except ValueError:
                    # 字段引用或嵌套函数
                    arg = arg.strip()
                    if arg in ctx._fields:
                        resolved.append(ctx._fields[arg])
                    elif '(' in arg:
                        resolved.append(self._evaluate(ctx, arg))
                    else:
                        resolved.append(arg)
        return resolved

    def export_library(self) -> List[Dict]:
        """导出因子库定义（可序列化）"""
        return [fe.to_dict() for fe in self.library]


# ============================================================
# 测试套件
# ============================================================

class TestExpressionParsing(unittest.TestCase):
    """表达式解析正确性测试"""

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'code': ['000001'] * n,
            'date': pd.date_range('2024-01-01', periods=n, freq='B'),
            'open': np.random.uniform(10, 20, n),
            'high': np.random.uniform(10, 22, n),
            'low': np.random.uniform(8, 20, n),
            'close': np.random.uniform(10, 20, n),
            'volume': np.random.randint(1000, 10000, n).astype(float),
            'amount': np.random.uniform(10000, 50000, n),
            'turnover_rate': np.random.uniform(0.01, 0.1, n),
        })
        # 确保 high >= low
        for i in range(len(self.df)):
            self.df.loc[i, 'high'] = max(self.df.loc[i, 'open'], self.df.loc[i, 'close']) * 1.01
            self.df.loc[i, 'low'] = min(self.df.loc[i, 'open'], self.df.loc[i, 'close']) * 0.99

        self.engine = ExpressionFactorEngine()

    def test_simple_field_reference(self):
        """测试简单字段引用"""
        ctx = ExprContext(self.df)
        result = self.engine._evaluate(ctx, "close")
        np.testing.assert_array_almost_equal(result, self.df['close'].values)

    def test_ref_function(self):
        """测试 Ref 函数"""
        ctx = ExprContext(self.df)
        result = self.engine._evaluate(ctx, "Ref(close, 1)")
        expected = np.full(len(self.df), np.nan)
        expected[1:] = self.df['close'].values[:-1]
        np.testing.assert_array_almost_equal(result[1:], expected[1:])

    def test_pct_change(self):
        """测试 PctChange 函数"""
        ctx = ExprContext(self.df)
        result = self.engine._evaluate(ctx, "PctChange(close, 1)")
        expected = self.df['close'].pct_change().values
        np.testing.assert_array_almost_equal(result[1:], expected[1:])

    def test_rolling_mean(self):
        """测试 Mean 函数"""
        ctx = ExprContext(self.df)
        result = self.engine._evaluate(ctx, "Mean(close, 5)")
        expected = self.df['close'].rolling(5, min_periods=1).mean().values
        np.testing.assert_array_almost_equal(result, expected)

    def test_arithmetic_expression(self):
        """测试算术表达式"""
        ctx = ExprContext(self.df)
        result = self.engine._evaluate(ctx, "close / Mean(close, 5) - 1")
        ma5 = self.df['close'].rolling(5, min_periods=1).mean()
        expected = (self.df['close'] / ma5 - 1).values
        np.testing.assert_array_almost_equal(result, expected)

    def test_negation(self):
        """测试负数前缀"""
        ctx = ExprContext(self.df)
        result = self.engine._evaluate(ctx, "-PctChange(close, 5)")
        expected = -self.df['close'].pct_change(5).values
        # 前5个是 NaN
        np.testing.assert_array_almost_equal(result[5:], expected[5:])

    def test_multi_stock_compute_all(self):
        """测试多股票批量因子计算"""
        df_multi = pd.concat([
            self.df.assign(code='000001'),
            self.df.assign(code='000002').assign(close=self.df['close'] * 1.5),
        ], ignore_index=True)

        result = self.engine.compute_all(df_multi)
        self.assertIn('ret_1d', result.columns)
        self.assertIn('reversal_20d', result.columns)
        self.assertIn('volatility_20d', result.columns)
        self.assertIn('ma_diff_5_20', result.columns)
        self.assertEqual(len(result), len(df_multi))

    def test_factor_library_completeness(self):
        """测试因子库中所有因子都可以计算"""
        df_multi = pd.concat([
            self.df.assign(code='000001'),
            self.df.assign(code='000002'),
        ], ignore_index=True)

        result = self.engine.compute_all(df_multi)
        computed = [c for c in result.columns if c not in ['code', 'date']]
        defined = [f.name for f in FACTOR_LIBRARY]
        missing = [f for f in defined if f not in computed]
        self.assertEqual(len(missing), 0, f"未成功计算的因子: {missing}")


class TestExpressionVsHardcoded(unittest.TestCase):
    """表达式引擎 vs 硬编码因子计算一致性测试"""

    def setUp(self):
        np.random.seed(42)
        n = 200
        self.df = pd.DataFrame({
            'code': ['000001'] * n,
            'date': pd.date_range('2024-01-01', periods=n, freq='B'),
            'open': np.random.uniform(10, 20, n),
            'high': np.random.uniform(10, 22, n),
            'low': np.random.uniform(8, 20, n),
            'close': np.random.uniform(10, 20, n),
            'volume': np.random.randint(1000, 10000, n).astype(float),
            'amount': np.random.uniform(10000, 50000, n),
            'turnover_rate': np.random.uniform(0.01, 0.1, n),
            'change_pct': np.random.uniform(-5, 5, n),
        })

        self.engine = ExpressionFactorEngine()

    def test_ret_factors_consistency(self):
        """验证收益率因子与现有硬编码逻辑一致"""
        result = self.engine.compute_all(self.df)

        # 硬编码方式
        df_hc = self.df.copy()
        df_hc['ret_1d_hc'] = df_hc['close'].pct_change()
        df_hc['ret_5d_hc'] = df_hc['close'].pct_change(5)
        df_hc['ret_20d_hc'] = df_hc['close'].pct_change(20)

        # for 非NaN值应一致
        for col in ['ret_1d', 'ret_5d', 'ret_20d']:
            mask = ~result[col].isna()
            np.testing.assert_array_almost_equal(
                result.loc[mask, col].values,
                df_hc.loc[mask, f'{col}_hc'].values,
                decimal=10, err_msg=f"因子 {col} 不一致"
            )

    def test_reversal_factors_consistency(self):
        """验证反转因子与现有硬编码逻辑一致"""
        result = self.engine.compute_all(self.df)

        df_hc = self.df.copy()
        df_hc['reversal_5d_hc'] = -df_hc['close'].pct_change(5)
        df_hc['reversal_20d_hc'] = -df_hc['close'].pct_change(20)

        for col in ['reversal_5d', 'reversal_20d']:
            mask = ~result[col].isna()
            np.testing.assert_array_almost_equal(
                result.loc[mask, col].values,
                df_hc.loc[mask, f'{col}_hc'].values,
                decimal=10, err_msg=f"因子 {col} 不一致"
            )


class TestExpressionPerformance(unittest.TestCase):
    """表达式引擎性能测试"""

    def setUp(self):
        np.random.seed(42)
        n_stocks = 20
        n_days = 252
        self.df = pd.DataFrame({
            'code': np.repeat([f'{i:06d}' for i in range(n_stocks)], n_days),
            'date': np.tile(pd.date_range('2024-01-01', periods=n_days, freq='B'), n_stocks),
            'open': np.random.uniform(10, 20, n_stocks * n_days),
            'high': np.random.uniform(10, 22, n_stocks * n_days),
            'low': np.random.uniform(8, 20, n_stocks * n_days),
            'close': np.random.uniform(10, 20, n_stocks * n_days),
            'volume': np.random.randint(1000, 10000, n_stocks * n_days).astype(float),
            'amount': np.random.uniform(10000, 50000, n_stocks * n_days),
            'turnover_rate': np.random.uniform(0.01, 0.1, n_stocks * n_days),
            'change_pct': np.random.uniform(-5, 5, n_stocks * n_days),
        })
        self.engine = ExpressionFactorEngine()

    def test_compute_speed(self):
        """测试表达式引擎计算速度"""
        times = []
        for _ in range(3):
            start = time.time()
            self.engine.compute_all(self.df)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        n_factors = len(FACTOR_LIBRARY)
        print(f"\n  数据量: {len(self.df)} 行 x {n_factors} 因子")
        print(f"  平均耗时: {avg_time:.3f}s")
        print(f"  每行耗时: {avg_time / len(self.df) * 1e6:.2f}μs")
        self.assertLess(avg_time, 30.0, "表达式引擎计算超时" if avg_time < 30 else "")


def generate_synthetic_data(n_stocks=10, n_days=252) -> pd.DataFrame:
    """生成模拟行情数据用于集成测试"""
    np.random.seed(20240613)
    symbols = [f'{i:06d}' for i in range(n_stocks)]
    rows = []
    for sym in symbols:
        start_price = np.random.uniform(8, 50)
        prices = [start_price]
        for _ in range(n_days - 1):
            ret = np.random.normal(0.0003, 0.018)
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices)
        df = pd.DataFrame({
            'code': sym,
            'date': pd.date_range('2024-01-01', periods=n_days, freq='B'),
            'open': prices * (1 + np.random.normal(0, 0.003, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.008, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.008, n_days))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, n_days).astype(float),
            'amount': np.random.lognormal(12, 0.5, n_days),
            'turnover_rate': np.random.uniform(0.005, 0.08, n_days),
            'change_pct': np.random.uniform(-3, 3, n_days),
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def run_benchmark():
    """运行完整基准测试"""
    print("\n" + "=" * 60)
    print("表达式因子引擎 - 基准测试")
    print("=" * 60)

    df = generate_synthetic_data(n_stocks=50, n_days=252)
    engine = ExpressionFactorEngine()

    print(f"\n数据规模: {len(df)} 行, {df['code'].nunique()} 只股票")
    print(f"因子数量: {len(FACTOR_LIBRARY)}")

    # 预热
    engine.compute_all(df.head(1000))

    # 计时
    start = time.time()
    result = engine.compute_all(df)
    elapsed = time.time() - start

    valid_factors = [c for c in result.columns if c not in ['code', 'date'] and not result[c].isna().all()]
    print(f"\n计算结果:")
    print(f"  成功计算的因子: {len(valid_factors)}/{len(FACTOR_LIBRARY)}")
    print(f"  耗时: {elapsed:.3f}s")
    print(f"  每行因子计算速度: {elapsed / len(df) * 1e6:.1f}μs")

    # 因子有效性检查
    print(f"\n因子覆盖率:")
    for c in valid_factors[:5]:
        coverage = 1 - result[c].isna().mean()
        print(f"  {c}: {coverage:.1%}")

    print("\n因子库定义（JSON 可序列化）:")
    library_json = engine.export_library()
    print(json.dumps(library_json[:3], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    else:
        # 运行单元测试
        unittest.main(argv=[''], verbosity=2, exit=False)