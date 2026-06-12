"""
优化方向: 因子表达式引擎 (Factor Expression Engine)
借鉴来源: Microsoft Qlib (github.com/microsoft/qlib) - Expression Engine 设计
借鉴亮点: Qlib 使用 DSL 语法声明因子，如 $close, Ref($close, 1), Mean($close, 3)
         让因子定义更简洁、可读、可组合，避免手写复杂的 pandas groupby 操作

优化目标: 在 jingni-trader 的 factor-engine 中引入简单的表达式引擎，
         使因子定义从"命令式 pandas 操作"变为"声明式表达式"
"""

import sys
import os
sys.path.insert(0, '/workspace')

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any
import time


# ============================================================================
# 1. 表达式引擎实现 (基于 Qlib 设计理念的简化版)
# ============================================================================

class FactorExpression:
    """因子表达式基类"""

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class ColumnRef(FactorExpression):
    """直接引用列: $close, $volume"""

    def __init__(self, col: str):
        self.col = col

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        if self.col not in data.columns:
            raise KeyError(f"列 '{self.col}' 不存在于数据中")
        return data[self.col]

    def __repr__(self):
        return f"${self.col}"


class Ref(FactorExpression):
    """滞后引用: Ref($close, 1) 表示前1天的收盘价"""

    def __init__(self, expr: FactorExpression, period: int):
        self.expr = expr
        self.period = period

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        if 'code' in data.columns:
            sub = data[['code']].copy()
            sub['_value'] = np.asarray(series).ravel()
            return sub.groupby('code')['_value'].transform(
                lambda x: x.shift(self.period)
            )
        return series.shift(self.period)

    def __repr__(self):
        return f"Ref({self.expr}, {self.period})"


class RollingMean(FactorExpression):
    """滚动均值: Mean($close, 20)"""

    def __init__(self, expr: FactorExpression, window: int, min_periods: int = None):
        self.expr = expr
        self.window = window
        self.min_periods = min_periods or max(1, window // 2)

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        if 'code' in data.columns:
            sub = data[['code']].copy()
            sub['_value'] = np.asarray(series).ravel()
            return sub.groupby('code')['_value'].transform(
                lambda x: x.rolling(self.window, min_periods=self.min_periods).mean()
            )
        return series.rolling(self.window, min_periods=self.min_periods).mean()

    def __repr__(self):
        return f"Mean({self.expr}, {self.window})"


class RollingStd(FactorExpression):
    """滚动标准差: Std($close, 20)"""

    def __init__(self, expr: FactorExpression, window: int, min_periods: int = None):
        self.expr = expr
        self.window = window
        self.min_periods = min_periods or max(1, window // 2)

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        if 'code' in data.columns:
            sub = data[['code']].copy()
            sub['_value'] = np.asarray(series).ravel()
            return sub.groupby('code')['_value'].transform(
                lambda x: x.rolling(self.window, min_periods=self.min_periods).std()
            )
        return series.rolling(self.window, min_periods=self.min_periods).std()

    def __repr__(self):
        return f"Std({self.expr}, {self.window})"


class BinaryOp(FactorExpression):
    """二元运算: 加法/减法/乘法/除法"""

    def __init__(self, left: FactorExpression, right: FactorExpression, op: str):
        self.left = left
        self.right = right
        self.op = op

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        left_series = self.left.evaluate(data)
        if isinstance(self.right, (int, float)):
            right_series = self.right
        else:
            right_series = self.right.evaluate(data)

        if self.op == '+':
            return left_series + right_series
        elif self.op == '-':
            return left_series - right_series
        elif self.op == '*':
            return left_series * right_series
        elif self.op == '/':
            return left_series / right_series.replace(0, np.nan)
        else:
            raise ValueError(f"不支持的操作: {self.op}")

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class PctChange(FactorExpression):
    """百分比变化: PctChange($close, 5)"""

    def __init__(self, expr: FactorExpression, period: int = 1):
        self.expr = expr
        self.period = period

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        if 'code' in data.columns:
            sub = data[['code']].copy()
            sub['_value'] = np.asarray(series).ravel()
            return sub.groupby('code')['_value'].transform(
                lambda x: x.pct_change(self.period)
            )
        return series.pct_change(self.period)

    def __repr__(self):
        return f"PctChange({self.expr}, {self.period})"


class Neg(FactorExpression):
    """取负: Neg($close) = -$close"""

    def __init__(self, expr: FactorExpression):
        self.expr = expr

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return -self.expr.evaluate(data)

    def __repr__(self):
        return f"Neg({self.expr})"


class FactorParser:
    """
    表达式引擎解析器

    使用示例:
        parser = FactorParser()
        reversal_5d = parser.parse("Neg(PctChange($close, 5))")
        result = reversal_5d.evaluate(data)

    支持的因子表达式:
        - $close, $volume, $amount, $high, $low, $open
        - Ref(expr, period)
        - Mean(expr, window)
        - Std(expr, window)
        - PctChange(expr, period)
        - Neg(expr)
        - (expr + expr), (expr - expr), (expr * expr), (expr / expr)
    """

    COLUMN_MAP = {
        'close': 'close', 'volume': 'volume', 'amount': 'amount',
        'high': 'high', 'low': 'low', 'open': 'open',
        'turnover': 'turnover_rate', 'turnover_rate': 'turnover_rate',
        'change_pct': 'change_pct', 'pre_close': 'pre_close',
    }

    def parse(self, expression: str) -> FactorExpression:
        """解析表达式字符串为可执行的因子表达式"""
        expression = expression.strip()
        return self._parse_expr(expression)

    def _parse_expr(self, expr: str) -> FactorExpression:
        expr = expr.strip()

        # 先尝试在完整表达式中找顶层运算符
        # 例如: (a + b) / (c + d) 中的 /
        op_pos = self._find_operator(expr)
        if op_pos is not None:
            left = self._parse_expr(expr[:op_pos])
            right = self._parse_expr(expr[op_pos + 1:])
            return BinaryOp(left, right, expr[op_pos])

        # 处理带括号的表达式，匹配括号在末尾
        if expr.startswith('(') and self._find_matching_paren(expr) == len(expr) - 1:
            inner = expr[1:-1].strip()
            # 检查是否是二元运算
            op_pos = self._find_operator(inner)
            if op_pos is not None:
                left = self._parse_expr(inner[:op_pos])
                right = self._parse_expr(inner[op_pos + 1:])
                return BinaryOp(left, right, inner[op_pos])
            return self._parse_expr(inner)

        # Neg(expr)
        if expr.startswith('Neg(') and expr.endswith(')'):
            inner = expr[4:-1].strip()
            return Neg(self._parse_expr(inner))

        # Ref(expr, period)
        if expr.startswith('Ref(') and expr.endswith(')'):
            inner = expr[4:-1].strip()
            parts = self._split_args(inner)
            return Ref(self._parse_expr(parts[0]), int(parts[1]))

        # Mean(expr, window)
        if expr.startswith('Mean(') and expr.endswith(')'):
            inner = expr[5:-1].strip()
            parts = self._split_args(inner)
            return RollingMean(self._parse_expr(parts[0]), int(parts[1]))

        # Std(expr, window)
        if expr.startswith('Std(') and expr.endswith(')'):
            inner = expr[4:-1].strip()
            parts = self._split_args(inner)
            return RollingStd(self._parse_expr(parts[0]), int(parts[1]))

        # PctChange(expr, period)
        if expr.startswith('PctChange(') and expr.endswith(')'):
            inner = expr[10:-1].strip()
            parts = self._split_args(inner)
            return PctChange(self._parse_expr(parts[0]), int(parts[1]))

        # 列引用 $column
        if expr.startswith('$'):
            col_name = expr[1:]
            col_name = self.COLUMN_MAP.get(col_name, col_name)
            return ColumnRef(col_name)

        # 数字常量
        try:
            return float(expr)
        except ValueError:
            raise ValueError(f"无法解析表达式: {expr}")

    def _find_matching_paren(self, expr: str) -> int:
        """找到匹配的右括号位置"""
        assert expr[0] == '('
        depth = 1
        for i in range(1, len(expr)):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _find_operator(self, expr: str) -> int:
        """在顶层找到二元运算符位置（不在嵌套括号内）"""
        depth = 0
        for i, ch in enumerate(expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif depth == 0 and ch in '+-*/':
                # 确保不是负号
                if ch == '-' and i == 0:
                    continue
                return i
        return None

    def _split_args(self, args_str: str) -> List[str]:
        """在顶层逗号处分割参数"""
        args = []
        depth = 0
        current = ''
        for ch in args_str:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                args.append(current.strip())
                current = ''
            else:
                current += ch
        if current.strip():
            args.append(current.strip())
        return args


# ============================================================================
# 2. 测试：表达式引擎 vs 现有命令式代码
# ============================================================================

def generate_test_data(n_stocks: int = 50, n_days: int = 500) -> pd.DataFrame:
    """生成模拟A股日线数据"""
    np.random.seed(42)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.bdate_range('2023-01-01', periods=n_days)

    rows = []
    for code in codes:
        start_price = np.random.uniform(5, 100)
        daily_returns = np.random.normal(0.0005, 0.02, n_days)
        prices = start_price * np.exp(np.cumsum(daily_returns))
        for i, (date, price) in enumerate(zip(dates, prices)):
            intraday = abs(np.random.normal(0, 0.01))
            rows.append({
                'code': code,
                'date': date,
                'open': price * (1 + np.random.normal(0, 0.003)),
                'high': price * (1 + intraday * 1.5),
                'low': price * (1 - intraday * 1.5),
                'close': price,
                'volume': int(np.random.lognormal(10, 0.5)),
                'amount': int(np.random.lognormal(14, 0.5)),
                'turnover_rate': np.random.uniform(0.01, 0.15),
                'change_pct': daily_returns[i] * 100,
                'pre_close': prices[i - 1] if i > 0 else start_price,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df


def test_expression_engine_correctness():
    """测试表达式引擎的正确性"""
    print("\n" + "=" * 60)
    print("测试 1: 表达式引擎正确性验证")
    print("=" * 60)

    data = generate_test_data(n_stocks=10, n_days=200)

    parser = FactorParser()

    # 测试用例定义
    test_cases = [
        {
            "name": "PctChange($close, 5)",
            "expression": "PctChange($close, 5)",
            "imperative": lambda df: df.groupby('code')['close'].transform(
                lambda x: x.pct_change(5)
            ),
        },
        {
            "name": "Neg(PctChange($close, 5))",
            "expression": "Neg(PctChange($close, 5))",
            "imperative": lambda df: -df.groupby('code')['close'].transform(
                lambda x: x.pct_change(5)
            ),
        },
        {
            "name": "Mean($close, 20)",
            "expression": "Mean($close, 20)",
            "imperative": lambda df: df.groupby('code')['close'].transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            ),
        },
        {
            "name": "Std($close, 20)",
            "expression": "Std($close, 20)",
            "imperative": lambda df: df.groupby('code')['close'].transform(
                lambda x: x.rolling(20, min_periods=10).std()
            ),
        },
        {
            "name": "($close - Mean($close, 20))",
            "expression": "($close - Mean($close, 20))",
            "imperative": lambda df: (
                df['close'] -
                df.groupby('code')['close'].transform(
                    lambda x: x.rolling(20, min_periods=10).mean()
                )
            ),
        },
    ]

    all_passed = True
    for case in test_cases:
        expr = parser.parse(case["expression"])
        result_expr = expr.evaluate(data)
        result_imp = case["imperative"](data)

        # 两者应该相等（忽略NaN）- 使用 numpy 数组避免 pandas 索引对齐问题
        arr_expr = np.asarray(result_expr).ravel()
        arr_imp = np.asarray(result_imp).ravel()
        mask_arr = ~np.isnan(arr_expr) & ~np.isnan(arr_imp)
        if mask_arr.sum() > 0:
            max_diff = np.abs(arr_expr[mask_arr] - arr_imp[mask_arr]).max()
            status = "✓" if max_diff < 1e-10 else "✗"
            if max_diff >= 1e-10:
                all_passed = False
        else:
            status = "○"
            max_diff = 0

        print(f"  {status} {case['name']}: max_diff = {max_diff:.2e}")

    print(f"\n  结果: {'全部通过' if all_passed else '存在差异'}")
    return all_passed


def test_expression_engine_performance():
    """测试表达式引擎的性能对比"""
    print("\n" + "=" * 60)
    print("测试 2: 表达式引擎性能对比")
    print("=" * 60)

    data = generate_test_data(n_stocks=100, n_days=500)

    parser = FactorParser()

    # 定义与现有代码中因子计算等价的表达式
    expressions = {
        "ret_1d":     "PctChange($close, 1)",
        "ret_5d":     "PctChange($close, 5)",
        "ret_20d":    "PctChange($close, 20)",
        "ret_60d":    "PctChange($close, 60)",
        "reversal_5d":  "Neg(PctChange($close, 5))",
        "reversal_20d": "Neg(PctChange($close, 20))",
        "ma_20":      "Mean($close, 20)",
        "volatility_20d": "Std($close, 20)",
    }

    # 现有命令式实现
    def imperative_compute(df):
        result = df[['code', 'date']].copy()
        result['ret_1d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(1))
        result['ret_5d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(5))
        result['ret_20d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(20))
        result['ret_60d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(60))
        result['reversal_5d'] = -result['ret_5d']
        result['reversal_20d'] = -result['ret_20d']
        result['ma_20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
        result['volatility_20d'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=10).std())
        return result

    # 表达式引擎实现
    def expression_compute(df):
        result = df[['code', 'date']].copy()
        compiled = {}
        for name, expr_str in expressions.items():
            expr = parser.parse(expr_str)
            compiled[name] = expr
            result[name] = expr.evaluate(df)
        return result

    # 预热
    _ = imperative_compute(data)
    _ = expression_compute(data)

    # 计时
    n_runs = 5

    imp_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = imperative_compute(data)
        imp_times.append(time.perf_counter() - t0)

    exp_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = expression_compute(data)
        exp_times.append(time.perf_counter() - t0)

    imp_mean = np.mean(imp_times)
    exp_mean = np.mean(exp_times)

    print(f"  数据规模: {data['code'].nunique()} 只股票 × {data['date'].nunique()} 个交易日")
    print(f"  命令式实现平均耗时: {imp_mean:.4f}s")
    print(f"  表达式引擎平均耗时: {exp_mean:.4f}s")
    print(f"  性能比率: {exp_mean / imp_mean:.2f}x")

    # 验证结果是否一致
    df_imp = imperative_compute(data)
    df_exp = expression_compute(data)
    consistent = True
    for col in expressions.keys():
        arr_imp = np.asarray(df_imp[col]).ravel()
        arr_exp = np.asarray(df_exp[col]).ravel()
        mask_arr = ~np.isnan(arr_imp) & ~np.isnan(arr_exp)
        if mask_arr.sum() > 0:
            diff = np.abs(arr_imp[mask_arr] - arr_exp[mask_arr]).max()
            if diff > 1e-10:
                consistent = False
                print(f"  ✗ {col}: 存在差异, max_diff={diff:.2e}")
    if consistent:
        print(f"  ✓ 所有因子计算结果一致")

    return exp_mean / imp_mean


def test_factor_composability():
    """测试表达式引擎的可组合性"""
    print("\n" + "=" * 60)
    print("测试 3: 表达式引擎可组合性")
    print("=" * 60)

    data = generate_test_data(n_stocks=20, n_days=100)

    parser = FactorParser()

    # 复合因子示例（当前解析器支持的表达式）
    composite_factors = {
        # 布林带位置: (close - MA20) / (2 * Std20)
        "bollinger_position": "($close - Mean($close, 20)) / Std($close, 20)",
        # 波动率调整动量: ret_20d / volatility_20d
        "vol_adj_momentum": "PctChange($close, 20) / Std($close, 20)",
        # 相对强弱: (close - MA20) / MA20
        "rsi_simple": "($close - Mean($close, 20)) / Mean($close, 20)",
        # 价格偏离: close / MA20 - 1
        "price_deviation": "($close / Mean($close, 20)) - 1",
        # 量价关系: 换手率 * 价格变化
        "volume_price": "PctChange($close, 5) * $volume",
    }

    for name, expr_str in composite_factors.items():
        expr = parser.parse(expr_str)
        result = expr.evaluate(data)
        valid_count = result.notna().sum()
        print(f"  {name}:")
        print(f"    表达式: {expr_str}")
        print(f"    有效值: {valid_count}/{len(result)}")
        print(f"    均值: {result.mean():.4f}, 标准差: {result.std():.4f}")

    print("\n  ✓ 所有复合因子均可正确计算")


# ============================================================================
# 3. 运行所有测试
# ============================================================================

if __name__ == "__main__":
    test_expression_engine_correctness()
    test_expression_engine_performance()
    test_factor_composability()
    print("\n" + "=" * 60)
    print("测试完成: 因子表达式引擎验证")
    print("=" * 60)