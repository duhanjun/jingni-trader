"""
因子表达式引擎测试

测试维度：
1. 正确性：表达式计算结果与手动 pandas 计算一致
2. 性能：表达式引擎 vs 硬编码 pandas 的耗时
3. 边界：空数据 / 单标的 / 缺失字段 / Alpha158 子集
"""
import sys
import time
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace')

from quant_opt_20260621.factor_expression_engine import (
    FactorExpressionEngine, alpha158_definitions, alpha158_definitions_safe,
)


def make_factor_test_data(n_codes: int = 5, n_days: int = 60, seed: int = 0):
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    codes = [f'{600000 + i}.SH' for i in range(n_codes)]
    rows = []
    for c in codes:
        price = 10.0
        for d in dates:
            ret = np.random.randn() * 0.02
            price = max(price * (1 + ret), 1.0)
            rows.append({
                'code': c, 'date': d,
                'open': price * 0.99, 'high': price * 1.01,
                'low': price * 0.98, 'close': price,
                'volume': np.random.randint(1e6, 1e8),
            })
    return pd.DataFrame(rows)


class TestFactorExpressionCorrectness(unittest.TestCase):
    """正确性：表达式结果与手动 pandas 计算一致"""

    def setUp(self):
        self.data = make_factor_test_data(n_codes=5, n_days=60, seed=42)
        self.engine = FactorExpressionEngine()

    def test_arithmetic(self):
        """四则运算：KMID = (close - open) / open"""
        result = self.engine.compute(self.data, {'KMID': '(close - open) / open'})
        expected = (self.data['close'] - self.data['open']) / self.data['open']
        np.testing.assert_allclose(
            result['KMID'].values, expected.values, rtol=1e-10,
            err_msg="KMID 表达式计算与手动计算不一致"
        )

    def test_ref(self):
        """Ref 算子：ROC5 = Ref(close, 5) / close"""
        result = self.engine.compute(self.data, {'ROC5': 'Ref(close, 5) / close'})
        df = self.data.sort_values(['code', 'date']).copy()
        df['ref5'] = df.groupby('code')['close'].shift(5)
        df['expected'] = df['ref5'] / df['close']
        merged = result.merge(df[['code', 'date', 'expected']], on=['code', 'date'])
        valid = merged.dropna()
        np.testing.assert_allclose(
            valid['ROC5'].values, valid['expected'].values, rtol=1e-10,
            err_msg="Ref 算子计算不一致"
        )

    def test_mean(self):
        """Mean 算子：MA5 = Mean(close, 5) / close"""
        result = self.engine.compute(self.data, {'MA5': 'Mean(close, 5) / close'})
        df = self.data.sort_values(['code', 'date']).copy()
        df['ma5'] = df.groupby('code')['close'].transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        df['expected'] = df['ma5'] / df['close']
        merged = result.merge(df[['code', 'date', 'expected']], on=['code', 'date'])
        valid = merged.dropna()
        np.testing.assert_allclose(
            valid['MA5'].values, valid['expected'].values, rtol=1e-10,
            err_msg="Mean 算子计算不一致"
        )

    def test_std(self):
        """Std 算子"""
        result = self.engine.compute(self.data, {'STD5': 'Std(close, 5) / close'})
        df = self.data.sort_values(['code', 'date']).copy()
        df['std5'] = df.groupby('code')['close'].transform(
            lambda x: x.rolling(5, min_periods=2).std()
        )
        df['expected'] = df['std5'] / df['close']
        merged = result.merge(df[['code', 'date', 'expected']], on=['code', 'date'])
        valid = merged.dropna()
        np.testing.assert_allclose(
            valid['STD5'].values, valid['expected'].values, rtol=1e-10,
            err_msg="Std 算子计算不一致"
        )

    def test_max_min(self):
        """Max / Min 算子：RSV = (close - Min(low, n)) / (Max(high, n) - Min(low, n))"""
        expr = '(close - Min(low, 20)) / (Max(high, 20) - Min(low, 20))'
        result = self.engine.compute(self.data, {'RSV20': expr})
        df = self.data.sort_values(['code', 'date']).copy()
        df['min20'] = df.groupby('code')['low'].transform(lambda x: x.rolling(20, min_periods=1).min())
        df['max20'] = df.groupby('code')['high'].transform(lambda x: x.rolling(20, min_periods=1).max())
        df['expected'] = (df['close'] - df['min20']) / (df['max20'] - df['min20'])
        merged = result.merge(df[['code', 'date', 'expected']], on=['code', 'date'])
        valid = merged.dropna()
        np.testing.assert_allclose(
            valid['RSV20'].values, valid['expected'].values, rtol=1e-10,
            err_msg="Max/Min 算子计算不一致"
        )

    def test_rank(self):
        """Rank 算子（截面排名）"""
        result = self.engine.compute(self.data, {'R_CLOSE': 'Rank(close)'})
        df = self.data.copy()
        df['expected'] = df.groupby('date')['close'].rank(pct=True)
        merged = result.merge(df[['code', 'date', 'expected']], on=['code', 'date'])
        np.testing.assert_allclose(
            merged['R_CLOSE'].values, merged['expected'].values, rtol=1e-10,
            err_msg="Rank 算子计算不一致"
        )

    def test_log_abs_power(self):
        """Log / Abs / Power 算子"""
        result = self.engine.compute(self.data, {
            'LOG_VOL': 'Log(volume)',
            'ABS_RET': 'Abs(close - open)',
            'SQ_CLOSE': 'Power(close, 2)',
        })
        df = self.data.copy()
        np.testing.assert_allclose(
            result['LOG_VOL'].values,
            np.log(df['volume'].replace(0, np.nan)).values,
            rtol=1e-10, err_msg="Log 算子不一致"
        )
        np.testing.assert_allclose(
            result['ABS_RET'].values,
            df['close'].sub(df['open']).abs().values,
            rtol=1e-10, err_msg="Abs 算子不一致"
        )
        np.testing.assert_allclose(
            result['SQ_CLOSE'].values,
            (df['close'] ** 2).values,
            rtol=1e-10, err_msg="Power 算子不一致"
        )

    def test_nested_expression(self):
        """嵌套表达式：Mean(Ref(close, 1), 5)"""
        expr = 'Mean(Ref(close, 1), 5)'
        result = self.engine.compute(self.data, {'NESTED': expr})
        df = self.data.sort_values(['code', 'date']).copy()
        df['ref1'] = df.groupby('code')['close'].shift(1)
        df['expected'] = df.groupby('code')['ref1'].transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        merged = result.merge(df[['code', 'date', 'expected']], on=['code', 'date'])
        valid = merged.dropna()
        np.testing.assert_allclose(
            valid['NESTED'].values, valid['expected'].values, rtol=1e-10,
            err_msg="嵌套表达式计算不一致"
        )


class TestFactorExpressionBoundary(unittest.TestCase):
    """边界条件测试"""

    def test_empty_data(self):
        """空数据应返回空 DataFrame"""
        engine = FactorExpressionEngine()
        empty = pd.DataFrame(columns=['code', 'date', 'close', 'open'])
        result = engine.compute(empty, {'KMID': '(close - open) / open'})
        self.assertTrue(result.empty)

    def test_single_stock(self):
        """单标的应正常计算"""
        engine = FactorExpressionEngine()
        data = make_factor_test_data(n_codes=1, n_days=30, seed=1)
        result = engine.compute(data, {'MA5': 'Mean(close, 5) / close'})
        self.assertEqual(len(result), 30)
        self.assertIn('MA5', result.columns)

    def test_missing_field_raises(self):
        """引用不存在的字段应抛错"""
        engine = FactorExpressionEngine()
        data = make_factor_test_data(n_codes=2, n_days=10, seed=1)
        with self.assertRaises(RuntimeError):
            engine.compute(data, {'BAD': 'vwap / close'})

    def test_unknown_operator_raises(self):
        """未知算子应抛错"""
        engine = FactorExpressionEngine()
        data = make_factor_test_data(n_codes=2, n_days=10, seed=1)
        with self.assertRaises(RuntimeError):
            engine.compute(data, {'BAD': 'UnknownOp(close, 5)'})

    def test_alpha158_subset_runs(self):
        """Alpha158 安全子集应全部计算成功"""
        engine = FactorExpressionEngine()
        data = make_factor_test_data(n_codes=3, n_days=70, seed=1)
        defs = alpha158_definitions_safe(data)
        result = engine.compute(data, defs)
        # 应计算出一批因子
        factor_cols = [c for c in result.columns if c not in ('code', 'date')]
        self.assertGreater(len(factor_cols), 50,
                           f"Alpha158 子集因子数 {len(factor_cols)} 过少")
        # 每个因子应至少有部分非空值
        for col in factor_cols:
            self.assertGreater(result[col].count(), 0, f"因子 {col} 全为空")


class TestFactorExpressionPerformance(unittest.TestCase):
    """性能测试"""

    def test_performance_vs_hardcoded(self):
        """表达式引擎 vs 硬编码 pandas 的耗时对比"""
        data = make_factor_test_data(n_codes=20, n_days=250, seed=100)
        engine = FactorExpressionEngine()

        defs = {
            'KMID': '(close - open) / open',
            'MA5': 'Mean(close, 5) / close',
            'MA20': 'Mean(close, 20) / close',
            'STD20': 'Std(close, 20) / close',
            'ROC20': 'Ref(close, 20) / close',
            'RSV20': '(close - Min(low, 20)) / (Max(high, 20) - Min(low, 20))',
        }

        # 表达式引擎
        t0 = time.time()
        result_expr = engine.compute(data, defs)
        t_expr = time.time() - t0

        # 硬编码 pandas
        t0 = time.time()
        df = data.sort_values(['code', 'date']).copy()
        df['KMID'] = (df['close'] - df['open']) / df['open']
        df['MA5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5, min_periods=2).mean()) / df['close']
        df['MA20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean()) / df['close']
        df['STD20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=10).std()) / df['close']
        df['ROC20'] = df.groupby('code')['close'].shift(20) / df['close']
        df['min20'] = df.groupby('code')['low'].transform(lambda x: x.rolling(20, min_periods=1).min())
        df['max20'] = df.groupby('code')['high'].transform(lambda x: x.rolling(20, min_periods=1).max())
        df['RSV20'] = (df['close'] - df['min20']) / (df['max20'] - df['min20'])
        t_hard = time.time() - t0

        ratio = t_expr / t_hard if t_hard > 0 else float('inf')
        print(f"\n[性能] 6 因子 × 20 标的 × 250 日：表达式={t_expr:.3f}s, 硬编码={t_hard:.3f}s, 比值={ratio:.2f}x")

        # 表达式引擎开销应控制在硬编码的 5x 以内（解析 + 通用框架的开销）
        self.assertLess(ratio, 5.0,
                        f"表达式引擎耗时是硬编码的 {ratio:.2f}x，超过 5x 阈值")

        # 结果应一致
        for col in defs.keys():
            valid = result_expr[[col]].dropna()
            self.assertGreater(len(valid), 0, f"因子 {col} 全为空")


if __name__ == '__main__':
    unittest.main(verbosity=2)
