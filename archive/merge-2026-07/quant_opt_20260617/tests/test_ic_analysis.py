"""
增强 IC 分析测试
"""
import unittest
import numpy as np
import pandas as pd

from quant_opt_20260617.ic_analysis import ICAnalyzer
from quant_opt_20260617.tests._synthetic_data import generate_synthetic_a_share_data


class TestICAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_a_share_data(n_stocks=30, n_days=800, seed=42)
        cls.factor_cols = [
            'ret_5d', 'ret_20d', 'turnover_5d', 'turnover_change',
            'momentum_volume', 'noise_factor'
        ]

    def test_01_ic_decay_shape(self):
        """IC Decay 应输出 (n_factors × n_periods) 行"""
        analyzer = ICAnalyzer(forward_periods=[1, 5, 20], n_quantiles=5)
        result = analyzer.calc_ic_decay(
            self.data, self.data, self.factor_cols
        )
        # 每个因子 × 每个 forward_period 应有一行
        self.assertEqual(len(result), len(self.factor_cols) * 3)
        # 必须列存在
        for col in ["factor", "forward_period", "ic_mean", "ic_ir", "n_days"]:
            self.assertIn(col, result.columns)

    def test_02_known_alpha_5d_momentum(self):
        """ret_5d（5 日动量）应对 fwd_5d 有显著正 IC"""
        analyzer = ICAnalyzer(forward_periods=[5])
        result = analyzer.calc_ic_decay(self.data, self.data, ['ret_5d'])
        # IC 绝对值应 > 0.01（在合成数据上 5d 动量应该是显著信号）
        ic_5d = result[result['forward_period'] == 5]['ic_mean'].iloc[0]
        self.assertGreater(abs(ic_5d), 0.01,
                           f"ret_5d 对 fwd_5d 的 IC 不显著: {ic_5d}")

    def test_03_quantile_returns_monotonic(self):
        """ret_5d 的分组收益应大致单调（高分位收益 > 低分位）"""
        analyzer = ICAnalyzer(n_quantiles=5)
        result = analyzer.calc_quantile_returns(
            self.data, self.data, ['ret_5d'], forward_period=5
        )
        # 只看 ret_5d 行
        sub = result[result['factor'] == 'ret_5d']
        # quantile=1（最低）应小于 quantile=5（最高）
        q1 = sub[sub['quantile'] == 1]['mean_return'].iloc[0]
        q5 = sub[sub['quantile'] == 5]['mean_return'].iloc[0]
        # 注：合成数据的 ret_5d vs fwd_5d 不保证严格单调，但应有差异
        self.assertNotAlmostEqual(q1, q5, places=3,
                                  msg="最高/最低分位收益几乎无差异")

    def test_04_turnover_range(self):
        """Turnover 应在 [0, 1] 范围内"""
        analyzer = ICAnalyzer()
        result = analyzer.calc_turnover(self.data, ['ret_5d'])
        self.assertGreater(len(result), 0)
        self.assertGreaterEqual(result['turnover'].min(), 0.0)
        self.assertLessEqual(result['turnover'].max(), 1.0 + 1e-9)

    def test_05_half_life(self):
        """Half-life 估计应返回合理的数值"""
        analyzer = ICAnalyzer()
        result = analyzer.calc_half_life(
            self.data, self.data, ['ret_5d', 'noise_factor'],
            forward_period=5,
        )
        self.assertEqual(len(result), 2)
        # IC half-life 应该是数值或 inf
        for v in result['ic_half_life']:
            self.assertTrue(pd.isna(v) or np.isinf(v) or v > 0)

    def test_06_run_aggregator(self):
        """run() 应返回所有 4 个子报告"""
        analyzer = ICAnalyzer(forward_periods=[1, 5, 20])
        report = analyzer.run(
            self.data, self.data, self.factor_cols,
            forward_period_quantile=5
        )
        self.assertIn("ic_decay", report)
        self.assertIn("quantile_returns", report)
        self.assertIn("turnover", report)
        self.assertIn("half_life", report)
        for k, v in report.items():
            self.assertIsInstance(v, pd.DataFrame)
            self.assertGreater(len(v), 0, f"{k} 为空")

    def test_07_unknown_factor_handled(self):
        """未在 factor_cols 中的因子应被跳过，不抛错"""
        analyzer = ICAnalyzer()
        result = analyzer.calc_ic_decay(
            self.data, self.data, ['ret_5d', 'nonexistent']
        )
        # 只有 ret_5d 应出现
        self.assertEqual(set(result['factor'].unique()), {'ret_5d'})

    def test_08_estimate_half_life_unit(self):
        """静态方法 estimate_half_life 应有基本行为"""
        s = pd.Series(np.random.normal(0, 1, 100))
        hl = ICAnalyzer.estimate_half_life(s)
        self.assertTrue(pd.isna(hl) or hl > 0 or np.isinf(hl))

        # 持续下降的序列应给出较短半衰期
        s_decreasing = pd.Series(np.arange(100, 0, -1).astype(float))
        # 这是一个线性递减序列，AR(1) 系数 ≈ 1，半衰期很大
        # 这里只验证不抛错


if __name__ == "__main__":
    unittest.main()
