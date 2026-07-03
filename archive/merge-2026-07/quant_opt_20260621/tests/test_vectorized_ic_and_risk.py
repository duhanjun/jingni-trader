"""
向量化 IC 分析 + 扩展风险指标测试

测试维度：
1. IC 正确性：向量化 RankIC 与 scipy.stats.spearmanr 逐日计算一致
2. IC 性能：向量化 vs 逐日循环
3. IC 边界：空数据 / 样本不足 / 单因子
4. 风险指标正确性：已知值验证
"""
import sys
import time
import unittest

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '/workspace')

from quant_opt_20260621.vectorized_ic import vectorized_ic_analysis
from quant_opt_20260621.risk_metrics import (
    calc_var_historical, calc_cvar_historical, calc_var_parametric,
    calc_information_ratio, calc_beta, calc_alpha, calc_turnover,
    calc_profit_factor, calc_extended_metrics,
)


def make_ic_test_data(n_codes: int = 30, n_days: int = 60, seed: int = 0):
    """生成因子 + 远期收益数据"""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    codes = [f'{600000 + i}.SH' for i in range(n_codes)]
    rows = []
    for c in codes:
        for d in dates:
            rows.append({
                'code': c, 'date': d,
                'factor_a': np.random.randn(),
                'factor_b': np.random.randn(),
                'ret_forward_1d': np.random.randn() * 0.02,
                'ret_forward_5d': np.random.randn() * 0.05,
            })
    return pd.DataFrame(rows)


# ======================================================================
# 向量化 IC 测试
# ======================================================================

class TestVectorizedICCorrectness(unittest.TestCase):
    """正确性：向量化 RankIC 与 scipy 逐日计算一致"""

    def test_rankic_matches_scipy(self):
        """向量化 RankIC 应与 scipy.stats.spearmanr 逐日计算结果一致"""
        data = make_ic_test_data(n_codes=30, n_days=40, seed=42)
        fwd = data[['code', 'date', 'ret_forward_1d', 'ret_forward_5d']].copy()
        factor_df = data[['code', 'date', 'factor_a', 'factor_b']].copy()

        # 向量化
        vec_result = vectorized_ic_analysis(
            factor_df, fwd, factor_names=['factor_a', 'factor_b'],
            ic_type='spearman', min_cross_size=10,
        )

        # scipy 逐日计算 factor_a 的 1d RankIC
        ic_list = []
        for dt in sorted(data['date'].unique()):
            cross = data[data['date'] == dt].dropna(subset=['factor_a', 'ret_forward_1d'])
            if len(cross) < 10:
                continue
            ic, _ = stats.spearmanr(cross['factor_a'], cross['ret_forward_1d'])
            if not np.isnan(ic):
                ic_list.append(ic)

        scipy_mean = np.mean(ic_list)
        scipy_std = np.std(ic_list, ddof=1)

        # 找到向量化结果中 factor_a / ret_forward_1d 的条目
        vec_entry = None
        for item in vec_result.get('ret_forward_1d', []):
            if item['factor'] == 'factor_a':
                vec_entry = item
                break

        self.assertIsNotNone(vec_entry, "未找到 factor_a 的 IC 结果")
        self.assertAlmostEqual(vec_entry['ic_mean'], round(float(scipy_mean), 6), places=5,
                               msg=f"IC mean 不一致: vec={vec_entry['ic_mean']}, scipy={scipy_mean:.6f}")
        self.assertAlmostEqual(vec_entry['ic_std'], round(float(scipy_std), 6), places=5,
                               msg=f"IC std 不一致: vec={vec_entry['ic_std']}, scipy={scipy_std:.6f}")

    def test_pearson_ic(self):
        """Pearson IC 应正常运行"""
        data = make_ic_test_data(n_codes=20, n_days=30, seed=1)
        fwd = data[['code', 'date', 'ret_forward_1d']].copy()
        factor_df = data[['code', 'date', 'factor_a']].copy()
        result = vectorized_ic_analysis(
            factor_df, fwd, factor_names=['factor_a'],
            ic_type='pearson', min_cross_size=10,
        )
        self.assertIn('ret_forward_1d', result)
        self.assertEqual(len(result['ret_forward_1d']), 1)

    def test_multiple_forward_periods(self):
        """多远期周期应全部计算"""
        data = make_ic_test_data(n_codes=20, n_days=30, seed=2)
        fwd = data[['code', 'date', 'ret_forward_1d', 'ret_forward_5d']].copy()
        factor_df = data[['code', 'date', 'factor_a']].copy()
        result = vectorized_ic_analysis(factor_df, fwd, factor_names=['factor_a'])
        self.assertIn('ret_forward_1d', result)
        self.assertIn('ret_forward_5d', result)


class TestVectorizedICBoundary(unittest.TestCase):
    """边界条件"""

    def test_empty_data(self):
        empty = pd.DataFrame(columns=['code', 'date', 'factor_a'])
        result = vectorized_ic_analysis(empty, empty, ['factor_a'])
        self.assertEqual(result, {})

    def test_insufficient_cross_section(self):
        """截面样本不足时应返回空结果"""
        # 仅 2 个标的，min_cross_size=10 应全部过滤
        data = make_ic_test_data(n_codes=2, n_days=10, seed=3)
        fwd = data[['code', 'date', 'ret_forward_1d']].copy()
        factor_df = data[['code', 'date', 'factor_a']].copy()
        result = vectorized_ic_analysis(
            factor_df, fwd, ['factor_a'], min_cross_size=10
        )
        # 应无有效 IC
        if 'ret_forward_1d' in result:
            self.assertEqual(len(result['ret_forward_1d']), 0)


class TestVectorizedICPerformance(unittest.TestCase):
    """性能测试"""

    def test_performance_vs_loop(self):
        """向量化 IC vs 逐日 scipy 循环"""
        n_codes = 100
        n_days = 120
        data = make_ic_test_data(n_codes=n_codes, n_days=n_days, seed=100)
        fwd = data[['code', 'date', 'ret_forward_1d', 'ret_forward_5d']].copy()
        factor_df = data[['code', 'date', 'factor_a', 'factor_b']].copy()

        # 逐日 scipy
        t0 = time.time()
        for factor in ['factor_a', 'factor_b']:
            for fwd_col in ['ret_forward_1d', 'ret_forward_5d']:
                for dt in sorted(data['date'].unique()):
                    cross = data[data['date'] == dt].dropna(subset=[factor, fwd_col])
                    if len(cross) >= 10:
                        stats.spearmanr(cross[factor], cross[fwd_col])
        t_loop = time.time() - t0

        # 向量化
        t0 = time.time()
        vec_result = vectorized_ic_analysis(
            factor_df, fwd, ['factor_a', 'factor_b'], ic_type='spearman'
        )
        t_vec = time.time() - t0

        speedup = t_loop / t_vec if t_vec > 0 else float('inf')
        print(f"\n[性能] 2 因子 × 100 标的 × 120 日：逐日循环={t_loop:.3f}s, 向量化={t_vec:.3f}s, 加速比={speedup:.1f}x")

        self.assertGreater(speedup, 2.0,
                           f"向量化 IC 加速比 {speedup:.1f}x 未达到 2x 阈值")


# ======================================================================
# 扩展风险指标测试
# ======================================================================

class TestRiskMetricsCorrectness(unittest.TestCase):
    """风险指标正确性：已知值验证"""

    def test_var_historical_known_value(self):
        """历史 VaR 已知值验证"""
        # 1% 分位数为 -0.05，95% VaR 应为 0.05
        returns = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04])
        var = calc_var_historical(returns, confidence=0.95)
        # 第 5 百分位（1 - 0.95 = 0.05）→ np.percentile 取第 5 分位
        expected = -np.percentile(returns, 5)
        self.assertAlmostEqual(var, expected, places=6)

    def test_cvar_historical(self):
        """CVaR 应 >= VaR"""
        np.random.seed(0)
        returns = pd.Series(np.random.randn(1000) * 0.02)
        var = calc_var_historical(returns, 0.95)
        cvar = calc_cvar_historical(returns, 0.95)
        self.assertGreaterEqual(cvar, var * 0.99,
                                f"CVaR ({cvar}) 应 >= VaR ({var})")

    def test_var_parametric_normal(self):
        """参数法 VaR 在正态分布下应接近理论值"""
        np.random.seed(0)
        returns = pd.Series(np.random.randn(10000) * 0.01)
        var_p = calc_var_parametric(returns, 0.95)
        # 理论 95% VaR = 1.645 * std
        expected = 1.645 * 0.01
        self.assertAlmostEqual(var_p, expected, delta=0.002,
                               msg=f"参数法 VaR {var_p} 偏离理论值 {expected}")

    def test_beta_known_value(self):
        """Beta 已知值验证：完全相关时 Beta = std(r)/std(b)"""
        b = pd.Series(np.arange(1, 101, dtype=float))
        r = b * 2 + 1  # 完全线性相关
        beta = calc_beta(r, b)
        self.assertAlmostEqual(beta, 2.0, places=6,
                               msg=f"完全相关时 Beta 应为 2.0, 实际 {beta}")

    def test_information_ratio_zero_when_equal(self):
        """策略与基准相同时信息比率应为 0"""
        r = pd.Series(np.random.randn(100) * 0.01, index=pd.date_range('2024-01-01', periods=100))
        ir = calc_information_ratio(r, r)
        self.assertAlmostEqual(ir, 0.0, places=6)

    def test_alpha_zero_when_beta_one(self):
        """策略 = 基准时 Alpha 应接近 0"""
        r = pd.Series(np.random.randn(100) * 0.01 + 0.001, index=pd.date_range('2024-01-01', periods=100))
        alpha = calc_alpha(r, r, risk_free=0.0)
        self.assertAlmostEqual(alpha, 0.0, places=6)

    def test_extended_metrics_keys(self):
        """扩展指标应包含所有字段"""
        np.random.seed(0)
        dates = pd.date_range('2024-01-01', periods=100)
        equity = pd.Series(np.cumprod(1 + np.random.randn(100) * 0.01) * 1e6, index=dates)
        equity_curve = pd.DataFrame({
            'date': dates, 'equity': equity.values,
            'cash': 0, 'market_value': equity.values,
        })
        trades = pd.DataFrame([
            {'date': dates[10], 'code': 'A', 'action': 'buy', 'amount': 100000, 'pnl': -100000},
            {'date': dates[20], 'code': 'A', 'action': 'sell', 'amount': 105000, 'pnl': 5000},
            {'date': dates[30], 'code': 'B', 'action': 'buy', 'amount': 100000, 'pnl': -100000},
            {'date': dates[40], 'code': 'B', 'action': 'sell', 'amount': 95000, 'pnl': -5000},
        ])
        metrics = calc_extended_metrics(equity_curve, trades)
        required = {'var_95', 'cvar_95', 'var_parametric_95', 'downside_deviation',
                    'turnover', 'profit_factor', 'expectancy'}
        self.assertTrue(required.issubset(set(metrics.keys())),
                        f"缺失指标: {required - set(metrics.keys())}")
        # profit_factor = 5000 / 5000 = 1.0
        self.assertAlmostEqual(metrics['profit_factor'], 1.0, places=4)
        # expectancy = (5000 + -5000) / 2 = 0
        self.assertAlmostEqual(metrics['expectancy'], 0.0, places=4)

    def test_extended_metrics_with_benchmark(self):
        """带基准时应计算 IR / Beta / Alpha"""
        np.random.seed(0)
        dates = pd.date_range('2024-01-01', periods=100)
        equity = pd.Series(np.cumprod(1 + np.random.randn(100) * 0.01) * 1e6, index=dates)
        equity_curve = pd.DataFrame({
            'date': dates, 'equity': equity.values,
            'cash': 0, 'market_value': equity.values,
        })
        bench = pd.Series(np.random.randn(100) * 0.01, index=dates)
        trades = pd.DataFrame(columns=['date', 'code', 'action', 'amount', 'pnl'])
        metrics = calc_extended_metrics(equity_curve, trades, benchmark_returns=bench)
        self.assertIn('information_ratio', metrics)
        self.assertIn('beta', metrics)
        self.assertIn('alpha', metrics)


if __name__ == '__main__':
    unittest.main(verbosity=2)
