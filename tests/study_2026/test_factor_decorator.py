"""
验证文件: 装饰器驱动的因子注册与计算 API

借鉴来源:
  - Factor Engine (arxiv: 2602.14138) — 模块化、extensible 的
    decorator-based 因子 API，使用 @simple_factor / @advanced_factor 装饰器
    自动注册因子函数，解耦因子定义与引擎核心。
  - AKQuant Factor Expression Engine — 基于 Polars 的高性能因子计算引擎，
    支持 Alpha101 风格的表达式语法。

优化方向:
  当前 jingni-trader 的因子计算直接硬编码在 FactorEngine.compute_a_share_factors() 中
  (约 200 行内联计算)。每个新因子需要修改引擎核心代码，扩展性差。
  借鉴 Factor Engine 的装饰器模式，将因子定义与引擎解耦，提升可维护性和可扩展性。

验证目标:
  1. 正确性：装饰器注册的因子计算与原始硬编码结果一致
  2. 可扩展性：新增因子仅需定义函数并添加装饰器，无需修改引擎代码
  3. 性能对比：装饰器模式 vs 原始硬编码的性能差异

创建日期: 2026-06-11
分支: feature/quant-stream-inspired (建议)
"""

import unittest
import timeit
import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable

# ── 装饰器驱动的因子框架 ──────────────────────────────────────

class FactorRegistry:
    """因子注册表，管理所有已注册的因子"""

    def __init__(self):
        self._simple_factors: Dict[str, Callable] = {}
        self._advanced_factors: Dict[str, Callable] = {}

    def register_simple(self, name: str, func: Callable):
        """注册简单因子（直接计算，无状态）"""
        self._simple_factors[name] = func

    def register_advanced(self, name: str, func: Callable):
        """注册高级因子（可能需要额外参数或状态管理）"""
        self._advanced_factors[name] = func

    def list_factors(self) -> List[str]:
        return list(self._simple_factors.keys()) + list(self._advanced_factors.keys())

    def compute(self, data: pd.DataFrame, factor_names: List[str] = None) -> pd.DataFrame:
        """批量计算指定因子"""
        if factor_names is None:
            factor_names = self.list_factors()

        df = data.sort_values(['code', 'date']).copy()
        result = df[['code', 'date']].copy()

        for name in factor_names:
            if name in self._simple_factors:
                result[name] = self._simple_factors[name](df)
            elif name in self._advanced_factors:
                result[name] = self._advanced_factors[name](df)

        return result


# 全局注册表实例
_registry = FactorRegistry()


def simple_factor(name: str):
    """@simple_factor 装饰器：将一个函数注册为简单因子"""
    def decorator(func):
        _registry.register_simple(name, func)
        return func
    return decorator


# ── 因子定义（使用装饰器注册） ──────────────────────────────────

@simple_factor("ret_1d")
def calc_ret_1d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].pct_change()

@simple_factor("ret_5d")
def calc_ret_5d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].pct_change(5)

@simple_factor("ret_20d")
def calc_ret_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].pct_change(20)

@simple_factor("reversal_5d")
def calc_reversal_5d(df: pd.DataFrame) -> pd.Series:
    return -df.groupby('code')['close'].pct_change(5)

@simple_factor("volatility_20d")
def calc_volatility_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )

@simple_factor("turnover_20d")
def calc_turnover_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

@simple_factor("volume_ratio")
def calc_volume_ratio(df: pd.DataFrame) -> pd.Series:
    vol_5d = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    vol_20d = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    return vol_5d / vol_20d.replace(0, np.nan)


# ── 原始硬编码计算（模拟当前 engine.py 的实现） ────────────────

def original_compute_factors(data: pd.DataFrame) -> pd.DataFrame:
    """模拟当前 FactorEngine.compute_a_share_factors 的实现"""
    if data.empty:
        return data
    df = data.sort_values(['code', 'date']).copy()
    result = df[['code', 'date']].copy()

    result['ret_1d'] = df.groupby('code')['close'].pct_change()
    result['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    result['ret_20d'] = df.groupby('code')['close'].pct_change(20)
    result['reversal_5d'] = -result['ret_5d']
    result['volatility_20d'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )
    result['turnover_20d'] = df.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    vol_5d = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    vol_20d = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    result['volume_ratio'] = vol_5d / vol_20d.replace(0, np.nan)

    return result


# ── 测试数据生成 ──────────────────────────────────────────────

def generate_test_data(n_stocks: int = 10, n_days: int = 252) -> pd.DataFrame:
    """生成模拟 A 股日线数据"""
    np.random.seed(42)
    rows = []
    for code in [f"SH600{i:03d}" for i in range(n_stocks)]:
        base_price = np.random.uniform(5, 50)
        for d in range(n_days):
            price = base_price * (1 + np.random.normal(0.0005, 0.02))
            base_price = price
            rows.append({
                'code': code,
                'date': pd.Timestamp('2025-01-02') + pd.Timedelta(days=d),
                'open': price * (1 + np.random.normal(0, 0.005)),
                'close': price,
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'volume': np.random.uniform(1e5, 1e7),
                'amount': np.random.uniform(5e5, 5e8),
                'turnover_rate': np.random.uniform(0.005, 0.05),
            })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ── 测试类 ────────────────────────────────────────────────────

class TestFactorDecoratorAPI(unittest.TestCase):
    """测试装饰器驱动的因子 API"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_test_data(n_stocks=10, n_days=252)

    def test_01_registry_populated(self):
        """验证因子注册表已正确填充"""
        factors = _registry.list_factors()
        self.assertGreater(len(factors), 0, "注册表中应至少有一个因子")
        expected = {'ret_1d', 'ret_5d', 'ret_20d', 'reversal_5d',
                     'volatility_20d', 'turnover_20d', 'volume_ratio'}
        for name in expected:
            self.assertIn(name, factors, f"因子 {name} 应存在于注册表中")

    def test_02_correctness_vs_original(self):
        """验证装饰器计算结果与原始硬编码一致"""
        factors = ['ret_1d', 'ret_5d', 'ret_20d', 'reversal_5d',
                    'volatility_20d', 'turnover_20d', 'volume_ratio']

        result_decorator = _registry.compute(self.data, factors)
        result_original = original_compute_factors(self.data)

        for name in factors:
            a = result_decorator[name].fillna(-99999).values
            b = result_original[name].fillna(-99999).values
            np.testing.assert_array_almost_equal(
                a, b, decimal=6,
                err_msg=f"因子 {name} 计算结果不一致"
            )
        print(f"  [PASS] 所有 {len(factors)} 个因子计算结果完全一致")

    def test_03_extensibility_add_new_factor(self):
        """验证新增因子无需修改引擎代码"""

        # 模拟用户定义新因子：20日动量
        @simple_factor("momentum_20d")
        def calc_momentum_20d(df: pd.DataFrame) -> pd.Series:
            return df.groupby('code')['close'].pct_change(20)

        # 验证新因子已注册
        self.assertIn("momentum_20d", _registry.list_factors())

        # 计算新因子并验证合理性
        result = _registry.compute(self.data, ["momentum_20d"])
        vals = result["momentum_20d"].dropna()
        self.assertGreater(len(vals), 0, "新因子应产生有效值")
        # 动量值应在合理范围内（20天收益率一般在 -1 到 +5 之间为合理）
        extreme_ratio = (abs(vals) > 10).mean()
        self.assertLess(extreme_ratio, 0.01, "极端值比例过高")

        print(f"  [PASS] 新因子 momentum_20d 注册成功，"
              f"共 {len(vals)} 个有效值")

    def test_04_selective_factor_computation(self):
        """验证可以选择性计算部分因子"""
        factors_subset = ['ret_1d', 'volatility_20d']
        result = _registry.compute(self.data, factors_subset)
        computed_cols = set(result.columns) - {'code', 'date'}
        self.assertEqual(computed_cols, set(factors_subset),
                         "应只计算请求的因子")
        print(f"  [PASS] 选择性因子计算正确：只计算了 {factors_subset}")

    def test_05_performance_comparison(self):
        """性能对比：装饰器 vs 硬编码"""
        factors_all = ['ret_1d', 'ret_5d', 'ret_20d', 'reversal_5d',
                        'volatility_20d', 'turnover_20d', 'volume_ratio']

        n_runs = 10

        t_original = timeit.timeit(
            lambda: original_compute_factors(self.data),
            number=n_runs
        )
        t_decorator = timeit.timeit(
            lambda: _registry.compute(self.data, factors_all),
            number=n_runs
        )

        avg_original = t_original / n_runs * 1000  # ms
        avg_decorator = t_decorator / n_runs * 1000

        overhead_pct = (avg_decorator - avg_original) / avg_original * 100

        print(f"\n  性能对比 ({n_runs} 次运行均值):")
        print(f"    硬编码模式:     {avg_original:.2f} ms")
        print(f"    装饰器模式:     {avg_decorator:.2f} ms")
        print(f"    额外开销:       {overhead_pct:+.1f}%")

        # 装饰器模式的额外开销应在可接受范围内 (<30%)
        self.assertLess(overhead_pct, 30,
                        f"装饰器模式开销过大 ({overhead_pct:.1f}%)")
        print(f"  [PASS] 装饰器模式额外开销 {overhead_pct:.1f}% 在可接受范围内")

    def test_06_empty_data_handling(self):
        """边界测试：空数据"""
        empty_df = pd.DataFrame(columns=['code', 'date', 'close', 'volume', 'turnover_rate'])
        result = _registry.compute(empty_df, ['ret_1d'])
        self.assertTrue(result.empty or len(result) == 0)

    def test_07_single_stock_handling(self):
        """边界测试：单一股票"""
        single_data = self.data[self.data['code'] == 'SH600000'].copy()
        self.assertEqual(single_data['code'].nunique(), 1)
        result = _registry.compute(single_data, ['ret_1d', 'ret_5d'])
        self.assertIn('ret_1d', result.columns)
        # 单股票前几行为 NaN 是正常的（pct_change 需要前值）
        valid_vals = result['ret_1d'].dropna()
        self.assertGreater(len(valid_vals), 0)


# ── 邮件 ──────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("验证：装饰器驱动的因子注册与计算 API")
    print("借鉴来源：Factor Engine (arxiv:2602.14138) / AKQuant")
    print("=" * 60)
    unittest.main(verbosity=2)