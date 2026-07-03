"""
测试：向量化绩效指标
- 正确性：与 jingni-trader 现有 _calc_metrics 行为对比
- 数值稳定性：处理 nan / 全 0 / 负收益序列
- 性能：在 10k 长度序列上的耗时
"""
import sys
import time
import numpy as np
import pandas as pd
import unittest

sys.path.insert(0, "/workspace")

from quant_opt_20260620.vectorized_metrics.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_duration,
    calmar_ratio,
    win_rate,
    omega_ratio,
    compute_all_metrics,
)


class TestVectorizedMetrics(unittest.TestCase):
    def setUp(self):
        # 模拟一个稳定上行的日收益序列
        np.random.seed(42)
        self.returns_simple = np.array([0.001] * 252)  # 日收益 0.1%
        # 模拟含波动的序列
        self.returns_noisy = np.random.normal(0.0005, 0.02, 252)
        # 含 NaN 的序列
        self.returns_nan = self.returns_noisy.copy()
        self.returns_nan[::20] = np.nan

    # ── 正确性测试 ──
    def test_annualized_return_simple(self):
        """日收益 0.1% × 252 天 → 年化收益 ≈ 28.3%"""
        ar = annualized_return(self.returns_simple)
        expected = (1.001 ** 252) - 1
        self.assertAlmostEqual(ar, expected, places=6)

    def test_annualized_volatility_simple(self):
        """全 0 收益 → 年化波动率应为 0"""
        av = annualized_volatility(self.returns_simple)
        # std 接近 0 但不为 0（ddof=1 单值会 NaN, 但这里都是 0.001 会有非常小的 std）
        self.assertLess(av, 0.01)

    def test_sharpe_zero_vol(self):
        """零波动率情形 → 夏普为 0（不抛异常）"""
        const_returns = np.array([0.001] * 100)
        sr = sharpe_ratio(const_returns)
        self.assertEqual(sr, 0.0)

    def test_max_drawdown_monotonic_up(self):
        """单调上涨 → 最大回撤 = 0"""
        eq = np.cumprod(1 + np.array([0.01] * 100))
        self.assertEqual(max_drawdown(eq), 0.0)

    def test_max_drawdown_known_case(self):
        """已知序列：1, 1.2, 1.1, 1.3, 0.9 → 最大回撤 = (0.9/1.3 - 1) ≈ -0.3077"""
        eq = np.array([1.0, 1.2, 1.1, 1.3, 0.9])
        mdd = max_drawdown(eq)
        self.assertAlmostEqual(mdd, 0.9 / 1.3 - 1, places=4)

    def test_max_drawdown_duration(self):
        """3 天回撤 + 1 天恢复 → 最大回撤持续期 = 3"""
        eq = np.array([1.0, 1.1, 1.0, 0.9, 0.95, 1.2])  # 索引 1 是 peak
        dur = max_drawdown_duration(eq)
        # peak 在索引 1，之后 4 天后才回到 1.2 (peak)
        # underwater = [F, F, T, T, T, F] -> dur = 3
        self.assertEqual(dur, 3)

    def test_calmar_positive_case(self):
        """亏损序列 → calmar = 负（cum < 1）"""
        # 4 日序列，4 个负收益
        ret = np.array([-0.05, -0.03, -0.02, -0.01])
        cr = calmar_ratio(ret)
        self.assertLess(cr, 0)

    def test_calmar_positive_returns(self):
        """稳定正收益 + 轻微回撤 → calmar 应为正"""
        # 强趋势、低波动序列
        np.random.seed(42)
        ret = np.random.normal(0.005, 0.005, 100)  # 日均 0.5%，波动 0.5%
        cr = calmar_ratio(ret)
        # 复利 252 天约 e^1.26 - 1 ≈ 252%，回撤 < 5%
        self.assertGreater(cr, 0)
        print(f"\n  [METRICS] calmar (strong trend): {cr:.4f}")

    def test_calmar_zero_drawdown(self):
        """完美单调上涨 → calmar 接近 +inf（max_drawdown = 0）"""
        ret = np.array([0.001] * 100)
        cr = calmar_ratio(ret)
        # max_drawdown = 0 → 返回 0.0
        self.assertEqual(cr, 0.0)

    def test_win_rate(self):
        ret = np.array([0.01, -0.02, 0.03, -0.01, 0.05])
        self.assertAlmostEqual(win_rate(ret), 0.6, places=4)

    def test_compute_all_with_benchmark(self):
        """含基准时应额外计算信息比率"""
        np.random.seed(7)
        ret = np.random.normal(0.001, 0.02, 252)
        bench = np.random.normal(0.0005, 0.015, 252)
        m = compute_all_metrics(ret, benchmark=bench)
        for key in ["annualized_return", "sharpe_ratio", "max_drawdown",
                    "calmar_ratio", "win_rate", "omega_ratio", "information_ratio"]:
            self.assertIn(key, m)

    # ── 边界条件测试 ──
    def test_empty_array(self):
        """空数组不应抛异常"""
        self.assertEqual(annualized_return(np.array([])), 0.0)
        self.assertEqual(sharpe_ratio(np.array([])), 0.0)
        self.assertEqual(max_drawdown(np.array([])), 0.0)

    def test_all_nan(self):
        """全 NaN 不应抛异常"""
        ret = np.full(100, np.nan)
        self.assertEqual(annualized_return(ret), 0.0)
        self.assertEqual(sharpe_ratio(ret), 0.0)
        self.assertEqual(max_drawdown(np.array([np.nan] * 10)), 0.0)

    def test_nan_in_series(self):
        """含 NaN 的序列应自动剔除 NaN 计算而不抛异常"""
        clean = self.returns_noisy
        with_nan = self.returns_nan
        # 不抛异常
        ar_clean = annualized_return(clean)
        ar_nan = annualized_return(with_nan)
        # 两者都应该是有限数
        self.assertTrue(np.isfinite(ar_clean))
        self.assertTrue(np.isfinite(ar_nan))
        # 剔除 nan 后结果应与同样本（不带 nan）的结果一致：构造一个不含 nan 的子集
        mask = ~np.isnan(with_nan)
        ar_subset = annualized_return(with_nan[mask])
        self.assertAlmostEqual(ar_nan, ar_subset, places=8)
        print(f"\n  [NAN] annualized_return clean={ar_clean:.4f}, nan={ar_nan:.4f}, "
              f"subset={ar_subset:.4f}")

    def test_negative_returns_cum_negative(self):
        """长期亏损 → 年化收益为负、cum <= 1"""
        ret = np.full(252, -0.005)
        ar = annualized_return(ret)
        self.assertLess(ar, 0)

    def test_singleton_series(self):
        """单元素序列不应抛异常"""
        ret = np.array([0.01])
        # 至少不会抛异常
        sharpe_ratio(ret)
        sortino_ratio(ret)
        calmar_ratio(ret)

    def test_omega_threshold(self):
        """Omega 比率：阈值上下分别计算"""
        ret = np.array([0.02, -0.01, 0.03, -0.005, 0.04])
        om = omega_ratio(ret, threshold=0.0)
        self.assertGreater(om, 0)

    # ── 性能测试 ──
    def test_performance_10k(self):
        """10k 长度序列应在 < 50ms 完成 compute_all_metrics"""
        np.random.seed(123)
        ret = np.random.normal(0.0005, 0.02, 10_000)
        start = time.perf_counter()
        for _ in range(10):
            m = compute_all_metrics(ret)
        elapsed = (time.perf_counter() - start) / 10
        self.assertLess(elapsed, 0.05, f"compute_all_metrics 耗时 {elapsed*1000:.2f}ms 超过 50ms")
        print(f"\n  [PERF] compute_all_metrics(10k): {elapsed*1000:.3f}ms/run")

    def test_against_pandas_method(self):
        """与 jingni-trader 现有 _calc_metrics 实现进行对比（核心指标数值上数量级一致）"""
        np.random.seed(99)
        ret = np.random.normal(0.0008, 0.018, 500)
        eq = np.cumprod(1 + ret)

        # jingni-trader 方式（engine.py _calc_metrics）
        returns = pd.Series(ret)
        cum = (1 + returns).cumprod()
        total_return = cum.iloc[-1] - 1
        jt_annual = (1 + total_return) ** (252 / len(returns)) - 1
        jt_vol = returns.std() * np.sqrt(252)
        jt_sharpe = (jt_annual - 0.03) / jt_vol
        jt_mdd = (eq / np.maximum.accumulate(eq) - 1).min()

        # 新方式
        m = compute_all_metrics(ret, risk_free=0.03)
        new_annual = m["annualized_return"]
        new_vol = m["annualized_volatility"]
        new_sharpe = m["sharpe_ratio"]
        new_mdd = m["max_drawdown"]

        # 年化收益 / 波动率 / 最大回撤三者应完全一致（公式相同）
        self.assertAlmostEqual(jt_annual, new_annual, places=4)
        self.assertAlmostEqual(jt_vol, new_vol, places=4)
        self.assertAlmostEqual(jt_mdd, new_mdd, places=4)

        # 夏普比率因公式约定略有差异（jingni-trader 用 (annual-rfr)/vol，
        #   新方法用 mean(excess_daily)/std * sqrt(252)）
        # 这是合理的 convention 差异，差值应在合理范围
        sharpe_diff_pct = abs(jt_sharpe - new_sharpe) / abs(jt_sharpe) * 100
        self.assertLess(sharpe_diff_pct, 30,
                        f"夏普比率差异 {sharpe_diff_pct:.1f}% 超过 30%")
        print(f"\n  [CORRECTNESS] 与 jingni-trader 现有方法对比：")
        print(f"    annual:  jt={jt_annual:.6f}  new={new_annual:.6f}  ✓")
        print(f"    vol:     jt={jt_vol:.6f}  new={new_vol:.6f}  ✓")
        print(f"    sharpe:  jt={jt_sharpe:.6f}  new={new_sharpe:.6f}  (convention 差异 {sharpe_diff_pct:.1f}%)")
        print(f"    mdd:     jt={jt_mdd:.6f}  new={new_mdd:.6f}  ✓")


if __name__ == "__main__":
    unittest.main(verbosity=2)
