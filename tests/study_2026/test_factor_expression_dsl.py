"""
测试：因子表达式 DSL 引擎
借鉴来源：Microsoft Qlib (https://github.com/microsoft/qlib)
优化方向：factor-engine - 因子定义 DSL，替代硬编码因子计算

Qlib 的核心创新之一是 Expression Engine，通过 DSL 语法（如
$close/Ref($close, 20)-1）声明因子，引擎自动解析并向量化计算。
jingni-trader 当前使用硬编码方式逐行计算因子，可维护性和扩展性均受限。

本测试验证因子表达式 DSL 的可行性：
1. 基本运算符：加减乘除、引用、滞后
2. 聚合函数：Mean, Std, Sum, Max, Min
3. 条件表达式：If, CrossAbove
4. 与现有硬编码因子计算结果的正确性对比
5. 性能基准测试
"""

import unittest
import numpy as np
import pandas as pd
import re
import time
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
import operator


# ============================================================================
# 因子表达式 DSL 引擎实现
# ============================================================================

@dataclass
class Token:
    """表达式 Token"""
    type: str  # 'number', 'operator', 'function', 'field', 'paren'
    value: str


class FactorExpressionEngine:
    """
    因子表达式引擎

    支持的语法:
    - 字段引用: $close, $open, $high, $low, $volume, $amount
    - 算术运算: +, -, *, /, **
    - 比较运算: >, <, >=, <=, ==, !=
    - 逻辑运算: & (and), | (or)
    - 滞后函数: Ref(expr, N)
    - 差分函数: Delta(expr, N)
    - 移动窗口: Mean(expr, N), Std(expr, N), Sum(expr, N)
    - 排名函数: Rank(expr), TsRank(expr, N)
    - 条件函数: If(cond, true_val, false_val)
    - 截面函数: CSRank(expr), CSZScore(expr)
    - 其他: Abs(expr), Log(expr), Sign(expr), Max(a, b), Min(a, b)
    """

    # 字段映射
    FIELD_MAP = {
        '$open': 'open',
        '$high': 'high',
        '$low': 'low',
        '$close': 'close',
        '$volume': 'volume',
        '$amount': 'amount',
        '$vwap': 'vwap',
    }

    # 内置函数
    FUNCTIONS = {
        'Ref', 'Delta', 'Mean', 'Std', 'Sum', 'Max', 'Min',
        'Rank', 'TsRank', 'Abs', 'Log', 'Sign', 'If',
        'CSRank', 'CSZScore', 'CrossAbove', 'Correlation',
        'Covariance',
    }

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._cache: Dict[str, pd.Series] = {}

    def set_data(self, data: pd.DataFrame):
        """设置计算数据"""
        required_cols = {'code', 'date', 'open', 'high', 'low', 'close', 'volume'}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

        self._data = data.sort_values(['code', 'date']).copy()
        self._cache = {}

        # 预计算 VWAP
        if 'vwap' not in self._data.columns and 'amount' in self._data.columns:
            self._data['vwap'] = np.where(
                self._data['volume'] > 0,
                self._data['amount'] / self._data['volume'],
                self._data['close']
            )

    def evaluate(self, expression: str) -> pd.Series:
        """计算因子表达式，返回与输入数据对齐的 Series"""
        if self._data is None:
            raise ValueError("请先调用 set_data() 设置数据")

        if expression in self._cache:
            return self._cache[expression]

        result = self._eval(expression)
        self._cache[expression] = result
        return result

    def evaluate_all(self, expressions: Dict[str, str]) -> pd.DataFrame:
        """批量计算多个因子"""
        result = self._data[['code', 'date']].copy()
        for name, expr in expressions.items():
            result[name] = self.evaluate(expr)
        return result

    def _eval(self, expr: str) -> pd.Series:
        """递归计算表达式"""
        expr = expr.strip()

        if not expr:
            raise ValueError("空表达式")

        # 数字常量
        try:
            val = float(expr)
            return pd.Series(val, index=self._data.index, dtype=float)
        except (ValueError, TypeError):
            pass

        # 字段引用
        if expr in self.FIELD_MAP:
            col = self.FIELD_MAP[expr]
            return self._data[col].copy()

        # 移除最外层括号
        if expr.startswith('(') and self._find_matching_paren(expr, 0) == len(expr) - 1:
            return self._eval(expr[1:-1])

        # 一元负号: -(expr) 或 -field
        if expr.startswith('-'):
            inner = expr[1:].strip()
            if inner.startswith('('):
                # -( ... ) -> 对括号内取负
                if self._find_matching_paren(inner, 0) == len(inner) - 1:
                    return -self._eval(inner[1:-1])
            # -field 或 -func(...)
            return -self._eval(inner)

        # 查找最外层运算符（优先级最低，从左到右）
        # 按优先级：最低的 || (&) -> > < >= <= == != -> + - -> * / -> **
        for op_symbols in [['&'], ['|'], ['>', '<', '>=', '<=', '==', '!='], ['+', '-'], ['*', '/'], ['**']]:
            pos = self._find_outermost_op(expr, op_symbols)
            if pos is not None:
                left_expr = expr[:pos[0]].strip()
                right_expr = expr[pos[1]:].strip()
                op = expr[pos[0]:pos[1]]
                if not left_expr:
                    # 左边为空：一元运算符（如 -expr 已在上面处理）
                    raise ValueError(f"无法解析表达式(元运算符): {expr}")
                return self._apply_binary_op(left_expr, right_expr, op)

        # 函数调用
        func_match = re.match(r'^(\w[\w]*)\((.*)\)$', expr, re.DOTALL)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            args = self._split_args(args_str)
            return self._apply_function(func_name, args)

        raise ValueError(f"无法解析表达式: {expr}")

    def _find_matching_paren(self, expr: str, start: int) -> int:
        """找到匹配的右括号位置"""
        depth = 0
        for i in range(start, len(expr)):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _find_outermost_op(self, expr: str, op_symbols: List[str]) -> Optional[tuple]:
        """在表达式字符串中找到最外层运算符的位置（考虑括号深度）"""
        i = 0
        while i < len(expr):
            if expr[i] == '(':
                i = self._find_matching_paren(expr, i) + 1
                continue

            # 检查多字符运算符
            for op in sorted(op_symbols, key=len, reverse=True):
                if expr[i:i+len(op)] == op:
                    return (i, i + len(op))
            i += 1
        return None

    def _split_args(self, args_str: str) -> List[str]:
        """按逗号分割函数参数（考虑括号嵌套）"""
        args = []
        depth = 0
        current = []
        for ch in args_str:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            args.append(''.join(current).strip())
        return args

    def _apply_binary_op(self, left_expr: str, right_expr: str, op: str) -> pd.Series:
        """应用二元运算符"""
        left = self._eval(left_expr)
        right = self._eval(right_expr)

        op_map = {
            '+': operator.add, '-': operator.sub, '*': operator.mul,
            '/': operator.truediv, '**': operator.pow,
            '>': operator.gt, '<': operator.lt,
            '>=': operator.ge, '<=': operator.le,
            '==': operator.eq, '!=': operator.ne,
            '&': operator.and_, '|': operator.or_,
        }
        fn = op_map.get(op)
        if fn is None:
            raise ValueError(f"不支持的运算符: {op}")
        return fn(left, right)

    def _apply_function(self, func_name: str, args: List[str]) -> pd.Series:
        """应用内置函数"""
        if func_name == 'Ref':
            return self._func_ref(args)
        elif func_name == 'Delta':
            return self._func_delta(args)
        elif func_name == 'Mean':
            return self._func_rolling(args, 'mean')
        elif func_name == 'Std':
            return self._func_rolling(args, 'std')
        elif func_name == 'Sum':
            return self._func_rolling(args, 'sum')
        elif func_name == 'Max':
            if len(args) == 2 and args[1].isdigit():
                return self._func_rolling(args, 'max')
            return self._func_two_arg(args, 'max')
        elif func_name == 'Min':
            if len(args) == 2 and args[1].isdigit():
                return self._func_rolling(args, 'min')
            return self._func_two_arg(args, 'min')
        elif func_name == 'Abs':
            return self._eval(args[0]).abs()
        elif func_name == 'Log':
            x = self._eval(args[0])
            return np.log(x.replace(0, np.nan))
        elif func_name == 'Sign':
            x = self._eval(args[0])
            return np.sign(x)
        elif func_name == 'If':
            return self._func_if(args)
        elif func_name == 'Rank':
            return self._func_rank(args, rolling=False)
        elif func_name == 'TsRank':
            return self._func_rank(args, rolling=True)
        elif func_name == 'CSRank':
            return self._func_csrank(args)
        elif func_name == 'CSZScore':
            return self._func_cszscore(args)
        elif func_name == 'CrossAbove':
            return self._func_cross_above(args)
        elif func_name == 'Correlation':
            return self._func_correlation(args)
        else:
            raise ValueError(f"不支持的函数: {func_name}")

    def _func_ref(self, args: List[str]) -> pd.Series:
        """Ref(expr, N): 滞后 N 期"""
        if len(args) != 2:
            raise ValueError("Ref 需要 2 个参数: Ref(expr, N)")
        series = self._eval(args[0])
        n = int(args[1])
        return series.groupby(self._data['code']).shift(n)

    def _func_delta(self, args: List[str]) -> pd.Series:
        """Delta(expr, N): 差分 = expr - Ref(expr, N)"""
        if len(args) != 2:
            raise ValueError("Delta 需要 2 个参数: Delta(expr, N)")
        series = self._eval(args[0])
        n = int(args[1])
        return series - series.groupby(self._data['code']).shift(n)

    def _func_rolling(self, args: List[str], method: str) -> pd.Series:
        """移动窗口聚合: Mean/Std/Sum/Max/Min(expr, N)"""
        if len(args) != 2:
            raise ValueError(f"{method.capitalize()} 需要 2 个参数")
        series = self._eval(args[0])
        n = int(args[1])

        def rolling_func(x):
            if method == 'mean':
                return x.rolling(n, min_periods=3).mean()
            elif method == 'std':
                return x.rolling(n, min_periods=3).std()
            elif method == 'sum':
                return x.rolling(n, min_periods=3).sum()
            elif method == 'max':
                return x.rolling(n, min_periods=3).max()
            elif method == 'min':
                return x.rolling(n, min_periods=3).min()
            return x

        return series.groupby(self._data['code']).transform(rolling_func)

    def _func_two_arg(self, args: List[str], method: str) -> pd.Series:
        """两参数函数: Max(a, b), Min(a, b)"""
        if len(args) != 2:
            raise ValueError(f"{method} 需要 2 个参数")
        a = self._eval(args[0])
        b = self._eval(args[1])
        if method == 'max':
            return pd.concat([a, b], axis=1).max(axis=1)
        else:
            return pd.concat([a, b], axis=1).min(axis=1)

    def _func_if(self, args: List[str]) -> pd.Series:
        """If(cond, true_val, false_val)"""
        if len(args) != 3:
            raise ValueError("If 需要 3 个参数: If(cond, true_val, false_val)")
        cond = self._eval(args[0])
        true_val = self._eval(args[1])
        false_val = self._eval(args[2])
        result = np.where(cond, true_val, false_val)
        return pd.Series(result, index=self._data.index, dtype=float)

    def _func_rank(self, args: List[str], rolling: bool = False) -> pd.Series:
        """Rank(expr) 或 TsRank(expr, N)"""
        series = self._eval(args[0])
        if rolling:
            if len(args) < 2:
                raise ValueError("TsRank 需要 2 个参数: TsRank(expr, N)")
            n = int(args[1])
            return series.groupby(self._data['code']).transform(
                lambda x: x.rolling(n, min_periods=3).apply(
                    lambda y: y.rank(pct=True).iloc[-1], raw=False
                )
            )
        else:
            return series.rank(pct=True)

    def _func_csrank(self, args: List[str]) -> pd.Series:
        """CSRank(expr): 截面排名"""
        series = self._eval(args[0])
        return series.groupby(self._data['date']).rank(pct=True)

    def _func_cszscore(self, args: List[str]) -> pd.Series:
        """CSZScore(expr): 截面标准化"""
        series = self._eval(args[0])
        return series.groupby(self._data['date']).transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    def _func_cross_above(self, args: List[str]) -> pd.Series:
        """CrossAbove(a, b): a 上穿 b 返回 1"""
        if len(args) != 2:
            raise ValueError("CrossAbove 需要 2 个参数")
        a = self._eval(args[0])
        b = self._eval(args[1])
        cross = pd.Series(0, index=self._data.index)
        for code in self._data['code'].unique():
            mask = self._data['code'] == code
            a_s = a[mask]
            b_s = b[mask]
            a_prev = a_s.shift(1)
            b_prev = b_s.shift(1)
            cross.loc[mask] = ((a_s > b_s) & (a_prev <= b_prev)).astype(int)
        return cross

    def _func_correlation(self, args: List[str]) -> pd.Series:
        """Correlation(a, b, N): 滚动相关系数"""
        if len(args) != 3:
            raise ValueError("Correlation 需要 3 个参数")
        a = self._eval(args[0])
        b = self._eval(args[1])
        n = int(args[2])

        result = pd.Series(np.nan, index=self._data.index)
        for code in self._data['code'].unique():
            mask = self._data['code'] == code
            result.loc[mask] = a[mask].rolling(n, min_periods=3).corr(b[mask])
        return result


# ============================================================================
# 常用因子公式库
# ============================================================================

STANDARD_FACTORS = {
    # 收益率因子
    'ret_1d': '$close / Ref($close, 1) - 1',
    'ret_5d': '$close / Ref($close, 5) - 1',
    'ret_20d': '$close / Ref($close, 20) - 1',

    # 反转因子
    'reversal_5d': '-( $close / Ref($close, 5) - 1 )',
    'reversal_20d': '-( $close / Ref($close, 20) - 1 )',

    # 波动率因子
    'volatility_20d': 'Std($close / Ref($close, 1) - 1, 20)',

    # 量价因子
    'volume_ratio': '$volume / Mean($volume, 20)',
    'price_volume_corr': 'Correlation($close, $volume, 20)',

    # 均线偏离
    'ma_deviation_20d': '$close / Mean($close, 20) - 1',
    'ma_deviation_60d': '$close / Mean($close, 60) - 1',

    # 振幅
    'daily_range': '($high - $low) / Ref($close, 1)',

    # 换手率因子（需预先计算）
    # 'turnover_20d': 'Mean($turnover_rate, 20)',

    # 截面因子
    'cs_rank_ret_20d': 'CSRank(-($close / Ref($close, 20) - 1))',

    # 复合因子示例
    'combo_momentum_vol': 'Mean($close / Ref($close, 1) - 1, 5) / (Std($close / Ref($close, 1) - 1, 20) + 0.001)',
}


# ============================================================================
# 测试用例
# ============================================================================

class TestFactorExpressionDSL(unittest.TestCase):
    """因子表达式 DSL 引擎测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟数据"""
        np.random.seed(42)
        codes = [f'{i:06d}.SH' for i in range(1000, 1010)]
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')

        data_list = []
        for code in codes:
            n = len(dates)
            close = 10 * (1 + np.cumsum(np.random.randn(n) * 0.02))
            close = np.maximum(close, 1)
            open_p = close * (1 + np.random.randn(n) * 0.005)
            high = np.maximum(open_p, close) * (1 + np.abs(np.random.randn(n) * 0.01))
            low = np.minimum(open_p, close) * (1 - np.abs(np.random.randn(n) * 0.01))
            volume = np.maximum(np.random.lognormal(14, 0.5, n), 100)
            amount = close * volume

            df = pd.DataFrame({
                'code': code,
                'date': dates,
                'open': open_p,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'amount': amount,
            })
            data_list.append(df)

        cls.test_data = pd.concat(data_list, ignore_index=True)
        cls.engine = FactorExpressionEngine()
        cls.engine.set_data(cls.test_data)

    def test_field_reference(self):
        """测试字段引用"""
        result = self.engine.evaluate('$close')
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            self.test_data['close'].reset_index(drop=True)
        )

    def test_arithmetic(self):
        """测试基本算术运算"""
        result = self.engine.evaluate('$high - $low')
        expected = self.test_data['high'] - self.test_data['low']
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_ref(self):
        """测试 Ref 滞后函数"""
        result = self.engine.evaluate('Ref($close, 5)')
        expected = self.test_data.groupby('code')['close'].shift(5)
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_ret_calculation(self):
        """测试收益率计算"""
        result = self.engine.evaluate('$close / Ref($close, 1) - 1')
        expected = self.test_data.groupby('code')['close'].pct_change()
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            rtol=1e-10
        )

    def test_rolling_mean(self):
        """测试滚动均值"""
        result = self.engine.evaluate('Mean($close, 20)')
        expected = self.test_data.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=3).mean()
        )
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            rtol=1e-10
        )

    def test_rolling_std(self):
        """测试滚动标准差"""
        result = self.engine.evaluate('Std($close / Ref($close, 1) - 1, 20)')
        returns = self.test_data.groupby('code')['close'].pct_change()
        expected = returns.groupby(self.test_data['code']).transform(
            lambda x: x.rolling(20, min_periods=3).std()
        )
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            rtol=1e-10
        )

    def test_csrank(self):
        """测试截面排名"""
        result = self.engine.evaluate('CSRank($close)')
        expected = self.test_data.groupby('date')['close'].rank(pct=True)
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            rtol=1e-10
        )

    def test_if_expression(self):
        """测试 If 条件表达式"""
        result = self.engine.evaluate('If($close > Mean($close, 20), 1, -1)')
        ma20 = self.test_data.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=3).mean()
        )
        expected = pd.Series(
            np.where(self.test_data['close'] > ma20, 1.0, -1.0),
            index=self.test_data.index,
            dtype=float
        )
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_complex_expression(self):
        """测试复合因子"""
        result = self.engine.evaluate(
            '( $close / Mean($close, 20) - 1 ) / ( Std($close / Ref($close, 1) - 1, 20) + 0.001 )'
        )
        # 手动计算对比
        ma20 = self.test_data.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=3).mean()
        )
        close = self.test_data['close']
        deviation = close / ma20 - 1
        returns = self.test_data.groupby('code')['close'].pct_change()
        vol = returns.groupby(self.test_data['code']).transform(
            lambda x: x.rolling(20, min_periods=3).std()
        )
        expected = deviation / (vol + 0.001)

        # 只比较两个都不为 NaN 的位置
        mask = ~(result.isna() & expected.isna())
        pd.testing.assert_series_equal(
            result[mask].reset_index(drop=True),
            expected[mask].reset_index(drop=True),
            check_names=False,
            rtol=1e-5
        )

    def test_factor_with_hardcoded(self):
        """对比 DSL 因子与原始硬编码因子的一致性"""
        # 使用 DSL 计算反转因子
        dsl_reversal = self.engine.evaluate('-( $close / Ref($close, 20) - 1 )')

        # 用原始硬编码方式计算
        hardcoded = -self.test_data.groupby('code')['close'].pct_change(20)

        mask = ~(dsl_reversal.isna() & hardcoded.isna())
        pd.testing.assert_series_equal(
            dsl_reversal[mask].reset_index(drop=True),
            hardcoded[mask].reset_index(drop=True),
            check_names=False,
            rtol=1e-10
        )

    def test_batch_evaluation(self):
        """测试批量因子计算"""
        results = self.engine.evaluate_all({
            'ret_1d': '$close / Ref($close, 1) - 1',
            'ret_5d': '$close / Ref($close, 5) - 1',
            'ma_deviation': '$close / Mean($close, 20) - 1',
            'volatility': 'Std($close / Ref($close, 1) - 1, 20)',
            'volume_ratio': '$volume / Mean($volume, 20)',
        })
        self.assertEqual(len(results), len(self.test_data))
        expected_cols = ['code', 'date', 'ret_1d', 'ret_5d', 'ma_deviation', 'volatility', 'volume_ratio']
        self.assertEqual(list(results.columns), expected_cols)
        # 验证所有因子列都有有效值
        for col in ['ret_1d', 'ret_5d', 'ma_deviation', 'volatility', 'volume_ratio']:
            self.assertTrue(results[col].notna().any(), f"{col} 无有效值")


class TestFactorPerformance(unittest.TestCase):
    """因子计算性能测试"""

    @classmethod
    def setUpClass(cls):
        """生成较大规模数据"""
        np.random.seed(42)
        codes = [f'{i:06d}.SH' for i in range(1000, 1050)]  # 50 只股票
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')  # ~1200 交易日

        data_list = []
        for code in codes:
            n = len(dates)
            close = 10 * (1 + np.cumsum(np.random.randn(n) * 0.02))
            close = np.maximum(close, 1)
            volume = np.maximum(np.random.lognormal(14, 0.5, n), 100)
            amount = close * volume

            df = pd.DataFrame({
                'code': code,
                'date': dates,
                'open': close * (1 + np.random.randn(n) * 0.005),
                'high': close * (1 + np.abs(np.random.randn(n) * 0.02)),
                'low': close * (1 - np.abs(np.random.randn(n) * 0.02)),
                'close': close,
                'volume': volume,
                'amount': amount,
            })
            data_list.append(df)

        cls.test_data = pd.concat(data_list, ignore_index=True)
        cls.engine = FactorExpressionEngine()
        cls.engine.set_data(cls.test_data)

    def test_performance_comparison(self):
        """对比 DSL 与硬编码方式的性能"""
        # DSL 方式
        engine = FactorExpressionEngine()
        engine.set_data(self.test_data)

        # 预计算 5 个因子
        expressions = {
            'ret_1d': '$close / Ref($close, 1) - 1',
            'ret_5d': '$close / Ref($close, 5) - 1',
            'reversal_20d': '-( $close / Ref($close, 20) - 1 )',
            'volatility_20d': 'Std($close / Ref($close, 1) - 1, 20)',
            'volume_ratio': '$volume / Mean($volume, 20)',
        }

        start = time.perf_counter()
        result_dsl = engine.evaluate_all(expressions)
        dsl_time = time.perf_counter() - start

        # 硬编码方式
        start = time.perf_counter()
        df = self.test_data.copy()
        df['ret_1d'] = df.groupby('code')['close'].pct_change()
        df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
        df['reversal_20d'] = -df.groupby('code')['close'].pct_change(20)
        df['volatility_20d'] = df.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=3).std()
        )
        df['volume_ratio'] = df['volume'] / df.groupby('code')['volume'].transform(
            lambda x: x.rolling(20, min_periods=3).mean()
        )
        hardcoded_time = time.perf_counter() - start

        print(f"\n  DSL 方式耗时: {dsl_time:.4f}s")
        print(f"  硬编码方式耗时: {hardcoded_time:.4f}s")
        print(f"  DSL/硬编码比例: {dsl_time/hardcoded_time:.2f}x")

        # DSL 方式可能略慢（解析开销），但仍在可接受范围
        # 允许 5 倍以内的性能开销（DSL 的灵活性远大于此开销）
        self.assertLess(dsl_time / hardcoded_time, 5.0,
                        f"DSL 性能差异过大: {dsl_time/hardcoded_time:.2f}x")

    def test_caching_effect(self):
        """测试缓存对性能的提升"""
        engine = FactorExpressionEngine()
        engine.set_data(self.test_data)

        expr = '($close / Mean($close, 20) - 1) / (Std($close / Ref($close, 1) - 1, 20) + 0.001)'

        # 第一次计算（无缓存）
        start = time.perf_counter()
        _ = engine.evaluate(expr)
        first_time = time.perf_counter() - start

        # 第二次计算（有缓存）
        start = time.perf_counter()
        _ = engine.evaluate(expr)
        second_time = time.perf_counter() - start

        print(f"\n  首次计算耗时: {first_time:.4f}s")
        print(f"  缓存命中耗时: {second_time:.6f}s")
        print(f"  缓存加速比: {first_time/max(second_time, 1e-9):.0f}x")

        self.assertLess(second_time, first_time * 0.1,
                        "缓存应显著减少重复计算时间")


if __name__ == '__main__':
    unittest.main(verbosity=2)