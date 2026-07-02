"""
=============================================================================
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
优化方向: 因子表达式引擎 - 将硬编码因子计算改为声明式 DSL 表达式定义
=============================================================================

核心亮点:
  Qlib 的 Expression Engine 使用领域特定语言 (DSL) 来定义因子，例如:
    $close, Ref($close, 1), Mean($close, 20), $high - $low
  这种设计让因子定义从"如何计算"转变为"声明要计算什么"，带来了以下优势:
  1. 因子可配置化 - 无需修改代码即可添加新因子
  2. 表达式可组合 - 支持嵌套和链式组合
  3. LLM 友好 - 表达式语法易于 LLM 生成和解析
  4. 计算优化 - 可批量优化表达式的计算图

对比 jingni-trader 现状:
  当前 factor-engine 的 compute_a_share_factors() 方法是硬编码的:
    result['ret_1d'] = df.groupby('code')['close'].pct_change()
    result['reversal_20d'] = -result['ret_20d']
  这种方式每次添加新因子都需要修改代码，缺乏灵活性。

验证内容:
  1. 表达式解析器正确性测试
  2. 因子计算结果与硬编码版本对比
  3. 表达式组合和嵌套能力测试
  4. 边界条件测试（空数据、NaN 处理、单股票）
  5. 性能对比测试
"""

import os
import sys
import json
import time
import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ═══════════════════════════════════════════════════════════════════════════
# 因子表达式引擎原型实现 (借鉴 Qlib Expression Engine)
# ═══════════════════════════════════════════════════════════════════════════

class Expression(ABC):
    """表达式基类"""

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        ...

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """返回表达式依赖的原始字段"""
        ...

    @abstractmethod
    def get_lookback_window(self) -> int:
        """返回表达式需要的最大回溯窗口"""
        ...


class Field(Expression):
    """原始字段引用，如 $close, $volume"""

    def __init__(self, name: str):
        self.name = name
        self._lookback = 0

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        if self.name not in data.columns:
            raise KeyError(f"字段 {self.name} 不存在于数据中")
        return data[self.name]

    def get_required_fields(self) -> List[str]:
        return [self.name]

    def get_lookback_window(self) -> int:
        return 0

    def __str__(self):
        return f"${self.name}"


class Constant(Expression):
    """常量"""

    def __init__(self, value: float):
        self.value = value

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=data.index)

    def get_required_fields(self) -> List[str]:
        return []

    def get_lookback_window(self) -> int:
        return 0

    def __str__(self):
        return str(self.value)


class Ref(Expression):
    """前值引用: Ref($close, 1) 表示前一日收盘价"""

    def __init__(self, expr: Expression, n: int = 1):
        self.expr = expr
        self.n = n

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        return series.groupby(data['code']).shift(self.n)

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window() + self.n

    def __str__(self):
        return f"Ref({self.expr}, {self.n})"


class Mean(Expression):
    """滚动均值: Mean($close, 20)"""

    def __init__(self, expr: Expression, window: int):
        self.expr = expr
        self.window = window

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        return series.groupby(data['code']).transform(
            lambda x: x.rolling(self.window, min_periods=max(1, self.window // 2)).mean()
        )

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window() + self.window

    def __str__(self):
        return f"Mean({self.expr}, {self.window})"


class Std(Expression):
    """滚动标准差: Std($close, 20)"""

    def __init__(self, expr: Expression, window: int):
        self.expr = expr
        self.window = window

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        return series.groupby(data['code']).transform(
            lambda x: x.rolling(self.window, min_periods=max(1, self.window // 2)).std()
        )

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window() + self.window

    def __str__(self):
        return f"Std({self.expr}, {self.window})"


class Sum(Expression):
    """滚动求和: Sum($volume, 20)"""

    def __init__(self, expr: Expression, window: int):
        self.expr = expr
        self.window = window

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        return series.groupby(data['code']).transform(
            lambda x: x.rolling(self.window, min_periods=max(1, self.window // 2)).sum()
        )

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window() + self.window

    def __str__(self):
        return f"Sum({self.expr}, {self.window})"


class Max(Expression):
    """滚动最大值"""

    def __init__(self, expr: Expression, window: int):
        self.expr = expr
        self.window = window

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        return series.groupby(data['code']).transform(
            lambda x: x.rolling(self.window, min_periods=max(1, self.window // 2)).max()
        )

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window() + self.window

    def __str__(self):
        return f"Max({self.expr}, {self.window})"


class Min(Expression):
    """滚动最小值"""

    def __init__(self, expr: Expression, window: int):
        self.expr = expr
        self.window = window

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        return series.groupby(data['code']).transform(
            lambda x: x.rolling(self.window, min_periods=max(1, self.window // 2)).min()
        )

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window() + self.window

    def __str__(self):
        return f"Min({self.expr}, {self.window})"


class Rank(Expression):
    """截面排名（百分比）"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        return series.groupby(data['date']).rank(pct=True)

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window()

    def __str__(self):
        return f"Rank({self.expr})"


class Sign(Expression):
    """符号函数"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return np.sign(self.expr.evaluate(data))

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window()

    def __str__(self):
        return f"Sign({self.expr})"


class Abs(Expression):
    """绝对值"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return self.expr.evaluate(data).abs()

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window()

    def __str__(self):
        return f"Abs({self.expr})"


class Log(Expression):
    """对数"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return np.log(self.expr.evaluate(data).replace(0, np.nan))

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window()

    def __str__(self):
        return f"Log({self.expr})"


class BinaryOp(Expression):
    """二元运算基类"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def get_required_fields(self) -> List[str]:
        return list(set(self.left.get_required_fields() + self.right.get_required_fields()))

    def get_lookback_window(self) -> int:
        return max(self.left.get_lookback_window(), self.right.get_lookback_window())


class Add(BinaryOp):
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return self.left.evaluate(data) + self.right.evaluate(data)
    def __str__(self): return f"({self.left} + {self.right})"


class Sub(BinaryOp):
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return self.left.evaluate(data) - self.right.evaluate(data)
    def __str__(self): return f"({self.left} - {self.right})"


class Mul(BinaryOp):
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return self.left.evaluate(data) * self.right.evaluate(data)
    def __str__(self): return f"({self.left} * {self.right})"


class Div(BinaryOp):
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        denominator = self.right.evaluate(data).replace(0, np.nan)
        return self.left.evaluate(data) / denominator
    def __str__(self): return f"({self.left} / {self.right})"


class Gt(BinaryOp):
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return (self.left.evaluate(data) > self.right.evaluate(data)).astype(float)
    def __str__(self): return f"({self.left} > {self.right})"


class Lt(BinaryOp):
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        return (self.left.evaluate(data) < self.right.evaluate(data)).astype(float)
    def __str__(self): return f"({self.left} < {self.right})"


class Return(Expression):
    """收益率: Return($close, 5) 表示5日收益率"""

    def __init__(self, expr: Expression, period: int = 1):
        self.expr = expr
        self.period = period

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        return series.groupby(data['code']).pct_change(self.period)

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window() + self.period

    def __str__(self):
        return f"Return({self.expr}, {self.period})"


class Volatility(Expression):
    """波动率: Volatility($close, 20)"""

    def __init__(self, expr: Expression, window: int):
        self.expr = expr
        self.window = window

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        series = self.expr.evaluate(data)
        return series.groupby(data['code']).transform(
            lambda x: x.pct_change().rolling(self.window, min_periods=max(1, self.window // 2)).std()
        )

    def get_required_fields(self) -> List[str]:
        return self.expr.get_required_fields()

    def get_lookback_window(self) -> int:
        return self.expr.get_lookback_window() + self.window + 1

    def __str__(self):
        return f"Volatility({self.expr}, {self.window})"


class FactorExpressionEngine:
    """因子表达式引擎 - 管理和执行因子表达式"""

    def __init__(self):
        self._factors: Dict[str, Expression] = {}
        self._cache: Dict[str, pd.Series] = {}

    def register(self, name: str, expr: Expression):
        """注册因子表达式"""
        self._factors[name] = expr

    def register_from_dict(self, factor_defs: Dict[str, Expression]):
        """批量注册因子"""
        self._factors.update(factor_defs)

    def evaluate(self, name: str, data: pd.DataFrame) -> pd.Series:
        """计算单个因子"""
        if name not in self._factors:
            raise KeyError(f"因子 {name} 未注册")
        cache_key = f"{name}_{id(data)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = self._factors[name].evaluate(data)
        self._cache[cache_key] = result
        return result

    def evaluate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """批量计算所有因子"""
        result = data[['code', 'date']].copy()
        for name in self._factors:
            result[name] = self.evaluate(name, data)
        return result

    def get_required_fields(self) -> List[str]:
        """获取所有因子依赖的原始字段"""
        fields = set()
        for expr in self._factors.values():
            fields.update(expr.get_required_fields())
        return sorted(fields)

    def get_required_lookback(self) -> int:
        """获取所有因子需要的最大回溯窗口"""
        return max((expr.get_lookback_window() for expr in self._factors.values()), default=0)

    def clear_cache(self):
        self._cache.clear()

    def list_factors(self) -> List[str]:
        return list(self._factors.keys())


# ═══════════════════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════════════════

def generate_test_data(n_stocks: int = 5, n_days: int = 200):
    """生成模拟A股数据"""
    np.random.seed(42)
    stocks = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')

    data = []
    for code in stocks:
        close = np.cumsum(np.random.randn(n_days) * 0.02) + 10
        close = np.maximum(close, 1)
        volume = np.random.randint(1000, 100000, n_days) * 100
        amount = volume * close * np.random.uniform(0.9, 1.1, n_days)
        high = close * np.random.uniform(1.0, 1.05, n_days)
        low = close * np.random.uniform(0.95, 1.0, n_days)
        open_price = low + np.random.uniform(0, 1, n_days) * (high - low)

        for i, dt in enumerate(dates):
            data.append({
                'code': code,
                'date': dt,
                'open': open_price[i],
                'high': high[i],
                'low': low[i],
                'close': close[i],
                'volume': volume[i],
                'amount': amount[i],
            })

    return pd.DataFrame(data)


class TestFactorExpressionEngine(unittest.TestCase):
    """因子表达式引擎核心功能测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_test_data(n_stocks=10, n_days=200)
        cls.engine = FactorExpressionEngine()

    def setUp(self):
        self.engine.clear_cache()

    # ── 基础表达式测试 ──────────────────────────────────────

    def test_field_expression(self):
        """测试原始字段引用"""
        expr = Field('close')
        result = expr.evaluate(self.data)
        pd.testing.assert_series_equal(result, self.data['close'], check_names=False)

    def test_ref_expression(self):
        """测试前值引用 Ref($close, 1)"""
        expr = Ref(Field('close'), 1)
        result = expr.evaluate(self.data)
        expected = self.data.groupby('code')['close'].shift(1)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_return_expression(self):
        """测试收益率 Return($close, 5)"""
        expr = Return(Field('close'), 5)
        result = expr.evaluate(self.data)
        expected = self.data.groupby('code')['close'].pct_change(5)
        pd.testing.assert_series_equal(result, expected, check_names=False, rtol=1e-10)

    def test_mean_expression(self):
        """测试滚动均值 Mean($close, 20)"""
        expr = Mean(Field('close'), 20)
        result = expr.evaluate(self.data)
        expected = self.data.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        pd.testing.assert_series_equal(result, expected, check_names=False, rtol=1e-10)

    # ── 组合表达式测试 ──────────────────────────────────────

    def test_arithmetic_binary(self):
        """测试算术运算组合"""
        expr = Div(Sub(Field('close'), Field('open')), Add(Sub(Field('high'), Field('low')), Constant(0.001)))
        result = expr.evaluate(self.data)
        expected = (self.data['close'] - self.data['open']) / ((self.data['high'] - self.data['low']) + 0.001)
        pd.testing.assert_series_equal(result, expected, check_names=False, rtol=1e-10)

    def test_nested_expression(self):
        """测试嵌套表达式: Mean(Return($close, 1), 5)"""
        expr = Mean(Return(Field('close'), 1), 5)
        result = expr.evaluate(self.data)
        ret = self.data.groupby('code')['close'].pct_change(1)
        expected = ret.groupby(self.data['code']).transform(
            lambda x: x.rolling(5, min_periods=max(1, 5 // 2)).mean()
        )
        pd.testing.assert_series_equal(result, expected, check_names=False, rtol=1e-10)

    def test_complex_factor(self):
        """测试复杂因子: 20日均线偏离 = (close - Mean(close, 20)) / Std(close, 20)"""
        # 这是类似于 z-score 的因子
        ma20 = Mean(Field('close'), 20)
        std20 = Std(Field('close'), 20)
        expr = Div(Sub(Field('close'), ma20), std20)
        result = expr.evaluate(self.data)

        close = self.data['close']
        ma20_val = close.groupby(self.data['code']).transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        std20_val = close.groupby(self.data['code']).transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        expected = (close - ma20_val) / std20_val
        pd.testing.assert_series_equal(result, expected, check_names=False, rtol=1e-10)

    # ── 引擎集成测试 ──────────────────────────────────────

    def test_register_and_evaluate(self):
        """测试因子的注册和计算"""
        self.engine.register('ret_1d', Return(Field('close'), 1))
        self.engine.register('ma_20', Mean(Field('close'), 20))
        self.engine.register('vol_20', Volatility(Field('close'), 20))
        self.engine.register('ma_deviation', Div(
            Sub(Field('close'), Mean(Field('close'), 20)),
            Std(Field('close'), 20)
        ))

        result_df = self.engine.evaluate_all(self.data)
        self.assertIn('ret_1d', result_df.columns)
        self.assertIn('ma_20', result_df.columns)
        self.assertIn('vol_20', result_df.columns)
        self.assertIn('ma_deviation', result_df.columns)

    def test_required_fields(self):
        """测试依赖字段分析"""
        self.engine.register('ret_1d', Return(Field('close'), 1))
        fields = self.engine.get_required_fields()
        self.assertIn('close', fields)

    def test_lookback_calculation(self):
        """测试回溯窗口计算"""
        self.engine.register('ma_20', Mean(Field('close'), 20))
        lookback = self.engine.get_required_lookback()
        self.assertEqual(lookback, 20)

    # ── 与硬编码版本对比测试 ──────────────────────────────

    def test_vs_hardcoded_reversal_factor(self):
        """对比表达式引擎与硬编码版本的反转因子计算结果"""
        # 准备数据
        data = self.data.copy()

        # 表达式引擎版本
        self.engine.register('reversal_20d', Mul(Return(Field('close'), 20), Constant(-1)))
        expr_result = self.engine.evaluate('reversal_20d', data)

        # 硬编码版本
        ret_20d = data.groupby('code')['close'].pct_change(20)
        reversal_20d = -ret_20d

        common_idx = expr_result.dropna().index.intersection(reversal_20d.dropna().index)
        if len(common_idx) > 0:
            diff = (expr_result.loc[common_idx] - reversal_20d.loc[common_idx]).abs()
            self.assertLess(diff.max(), 1e-10, "表达式引擎与硬编码版本结果不一致")

    # ── 边界条件测试 ──────────────────────────────────────

    def test_empty_data(self):
        """测试空数据"""
        empty_df = pd.DataFrame(columns=['code', 'date', 'close'])
        self.engine.register('ret_1d', Return(Field('close'), 1))
        result = self.engine.evaluate('ret_1d', empty_df)
        self.assertEqual(len(result), 0)

    def test_single_stock(self):
        """测试单股票"""
        single = self.data[self.data['code'] == self.data['code'].iloc[0]].copy()
        self.engine.register('ma_20', Mean(Field('close'), 20))
        result = self.engine.evaluate('ma_20', single)
        self.assertEqual(len(result), len(single))

    def test_nan_handling(self):
        """测试 NaN 处理"""
        data = self.data.copy()
        data.loc[data.sample(20).index, 'close'] = np.nan
        self.engine.register('ma_20', Mean(Field('close'), 20))
        result = self.engine.evaluate('ma_20', data)
        self.assertFalse(result.isna().all(), "滚动均值应该能处理 NaN")

    def test_string_representation(self):
        """测试表达式字符串表示"""
        expr = Div(Sub(Field('close'), Mean(Field('close'), 20)), Std(Field('close'), 20))
        str_repr = str(expr)
        self.assertIn('$close', str_repr)
        self.assertIn('Mean', str_repr)
        self.assertIn('Std', str_repr)

    # ── 性能对比测试 ──────────────────────────────────────

    def test_performance_comparison(self):
        """对比表达式引擎与硬编码的性能"""
        # 大样本数据
        big_data = generate_test_data(n_stocks=50, n_days=500)

        # 表达式引擎
        self.engine.register('ret_20d', Return(Field('close'), 20))
        self.engine.register('reversal_20d', Mul(Return(Field('close'), 20), Constant(-1)))
        self.engine.register('vol_20d', Volatility(Field('close'), 20))
        self.engine.register('ma_20', Mean(Field('close'), 20))
        self.engine.register('ma_deviation', Div(
            Sub(Field('close'), Mean(Field('close'), 20)),
            Std(Field('close'), 20)
        ))

        start = time.time()
        expr_result = self.engine.evaluate_all(big_data)
        expr_time = time.time() - start

        # 硬编码版本
        start = time.time()
        ret_20d = big_data.groupby('code')['close'].pct_change(20)
        reversal_20d = -ret_20d
        vol_20d = big_data.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )
        ma_20 = big_data.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        close = big_data['close']
        ma_deviation = (close - ma_20) / big_data.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        hardcoded_time = time.time() - start

        print(f"\n  性能对比 ({len(big_data)} 行数据):")
        print(f"    表达式引擎: {expr_time:.4f}s (计算了 {len(self.engine._factors)} 个因子)")
        print(f"    硬编码版本: {hardcoded_time:.4f}s (计算了 5 个因子)")
        print(f"    比值: {expr_time/hardcoded_time:.2f}x")

        # 表达式引擎的灵活性优势体现在可扩展性上
        self.assertGreater(len(expr_result), 0)

    # ── 可扩展性测试 ──────────────────────────────────────

    def test_extensibility_demo(self):
        """演示表达式引擎的可扩展性：无需修改引擎代码即可添加新因子"""
        # 模拟经典 Alpha 因子: Alpha#1 from WorldQuant
        # rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
        # 简化版: (close > Mean(close, 20)) 的截面排名

        new_factor = Rank(Gt(Field('close'), Mean(Field('close'), 20)))

        self.engine.register('alpha_simple', new_factor)
        result = self.engine.evaluate('alpha_simple', self.data)

        # 验证排名结果在 [0, 1] 范围内
        self.assertTrue((result.dropna() >= 0).all())
        self.assertTrue((result.dropna() <= 1).all())


def run_tests():
    """运行所有测试并生成报告"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFactorExpressionEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("因子表达式引擎验证测试")
    print("借鉴来源: Microsoft Qlib Expression Engine")
    print("=" * 70)
    results = run_tests()
    print("\n" + "=" * 70)
    print(f"测试结果: {results['tests_run']} 个测试, "
          f"{results['failures']} 个失败, {results['errors']} 个错误")
    print(f"总体: {'通过' if results['success'] else '失败'}")
    print("=" * 70)