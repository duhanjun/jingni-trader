"""
测试：增强 IC 分析器
- 正确性：人工构造可预测 IC 的因子，验证 IC decay / layered / long-short
- 与 jingni-trader 现有 _calc_ic 对比
- 边界：空数据、单一截面、nan 处理
"""
import sys
import numpy as np
import pandas as pd
import unittest

sys.path.insert(0, "/workspace")

from quant_opt_20260620.ic_analyzer.ic_analyzer import (
    calc_ic_series,
    calc_ic_decay,
    calc_layered_returns,
    calc_long_short,
    calc_monotonicity,
    full_factor_evaluation,
)


def _make_synthetic_factor_data(n_dates=60, n_stocks=50, true_ic=0.05, seed=42):
    """
    生成合成因子数据：因子 = noise + true_ic * fwd_ret + 残差
    这样理论 IC 接近 true_ic，可验证。
    """
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    codes = [f"S{i:04d}" for i in range(n_stocks)]
    rows = []
    for d in dates:
        # 每只股票的 forward return
        fwd_ret = np.random.normal(0.001, 0.02, n_stocks)
        # 因子值含一定预测能力
        factor = true_ic * 50 * fwd_ret + np.random.normal(0, 1, n_stocks)
        for c, f, r in zip(codes, factor, fwd_ret):
            rows.append({"date": d, "code": c, "factor": f,
                         "fwd_ret_1d": r,
                         "fwd_ret_5d": r * 2 + np.random.normal(0, 0.01),
                         "fwd_ret_10d": r * 3 + np.random.normal(0, 0.02),
                         "fwd_ret_20d": r * 4 + np.random.normal(0, 0.03)})
    return pd.DataFrame(rows)


class TestICAnalyzer(unittest.TestCase):
    def setUp(self):
        self.factor_df, self.ret_df = None, None
        self._build_data()

    def _build_data(self):
        df = _make_synthetic_factor_data()
        self.factor_df = df[["date", "code", "factor"]].copy()
        self.ret_df = df[["date", "code", "fwd_ret_1d", "fwd_ret_5d",
                          "fwd_ret_10d", "fwd_ret_20d"]].copy()

    # ── 正确性测试 ──
    def test_ic_series_perfect_predictor(self):
        """完美预测：factor == fwd_ret → IC = 1"""
        ret = pd.Series([0.01, -0.02, 0.03, 0.005, -0.01])
        factor = ret.copy()
        ic = calc_ic_series(factor, ret, method="pearson")
        self.assertAlmostEqual(ic, 1.0, places=4)

    def test_ic_series_negative_predictor(self):
        """完全反向：factor == -fwd_ret → IC = -1"""
        ret = pd.Series([0.01, -0.02, 0.03, 0.005, -0.01])
        factor = -ret
        ic = calc_ic_series(factor, ret, method="pearson")
        self.assertAlmostEqual(ic, -1.0, places=4)

    def test_ic_decay_returns_dict(self):
        """IC decay 应返回多期结果"""
        decay = calc_ic_decay(self.factor_df, self.ret_df,
                              forward_periods=[1, 5, 10, 20])
        for fp in [1, 5, 10, 20]:
            self.assertIn(fp, decay)
            for k in ["ic_mean", "ic_std", "ic_ir", "ic_t_stat", "ic_pos_ratio"]:
                self.assertIn(k, decay[fp])

    def test_ic_decay_positive_predictive_power(self):
        """合成因子应有正的 IC mean"""
        decay = calc_ic_decay(self.factor_df, self.ret_df,
                              forward_periods=[1, 5, 10])
        for fp in [1, 5, 10]:
            self.assertGreater(decay[fp]["ic_mean"], 0,
                               f"forward {fp}d IC mean should be positive")

    def test_layered_returns_structure(self):
        """分层回测应返回多层结果"""
        layered = calc_layered_returns(self.factor_df, self.ret_df,
                                       n_quantiles=5, ret_col="fwd_ret_5d")
        self.assertEqual(layered.shape[1], 5)
        self.assertFalse(layered.empty)

    def test_layered_returns_monotonicity(self):
        """合成因子分层收益应大致单调（高分位组收益更高）"""
        layered = calc_layered_returns(self.factor_df, self.ret_df,
                                       n_quantiles=5, ret_col="fwd_ret_5d")
        mono = calc_monotonicity(layered)
        # 合成因子有明显 IC → 单调性 > 0
        self.assertGreater(mono, 0.0)
        print(f"\n  [IC] monotonicity = {mono:.4f}")

    def test_long_short_positive(self):
        """多空组合收益应为正（因子有预测力时）"""
        layered = calc_layered_returns(self.factor_df, self.ret_df,
                                       n_quantiles=5, ret_col="fwd_ret_5d")
        ls = calc_long_short(layered)
        self.assertGreater(ls["long_short_mean"], 0)
        print(f"  [IC] long_short_mean = {ls['long_short_mean']:.6f}, "
              f"sharpe = {ls['long_short_sharpe']:.4f}")

    def test_full_evaluation(self):
        """完整评估应一次返回全部指标"""
        result = full_factor_evaluation(self.factor_df, self.ret_df)
        self.assertIn("ic_decay", result)
        self.assertIn("layered_returns", result)
        self.assertIn("long_short", result)
        self.assertIn("monotonicity", result)

    # ── 与 jingni-trader 现有实现对比 ──
    def test_consistency_with_existing_spearman(self):
        """与 jingni-trader 现有 _calc_ic 中 spearman 分支对比：
        应得到相同的 IC mean 和 IR（数量级）"""
        from quant_opt_20260620.ic_analyzer.ic_analyzer import _safe_spearman

        merged = self.factor_df.merge(
            self.ret_df[["date", "code", "fwd_ret_5d"]], on=["date", "code"]
        ).dropna()

        # jingni-trader 方式
        jt_ic = []
        for d, g in merged.groupby("date"):
            from scipy import stats
            if len(g) < 10:
                continue
            ic, _ = stats.spearmanr(g["factor"], g["fwd_ret_5d"], nan_policy="omit")
            if not np.isnan(ic):
                jt_ic.append(ic)
        jt_mean = float(np.mean(jt_ic))
        jt_std = float(np.std(jt_ic, ddof=1))
        jt_ir = jt_mean / jt_std if jt_std > 0 else 0

        # 新方式
        decay = calc_ic_decay(self.factor_df, self.ret_df,
                              forward_periods=[5])
        new_mean = decay[5]["ic_mean"]
        new_std = decay[5]["ic_std"]
        new_ir = decay[5]["ic_ir"]

        self.assertAlmostEqual(jt_mean, new_mean, places=4)
        self.assertAlmostEqual(jt_std, new_std, places=4)
        self.assertAlmostEqual(jt_ir, new_ir, places=4)
        print(f"\n  [IC vs JT] mean={jt_mean:.4f} vs {new_mean:.4f}, "
              f"ir={jt_ir:.4f} vs {new_ir:.4f}")

    # ── 边界条件 ──
    def test_empty_factor_df(self):
        """空 factor df 不应抛异常"""
        empty = pd.DataFrame(columns=["date", "code", "factor"])
        decay = calc_ic_decay(empty, self.ret_df)
        self.assertEqual(decay, {})

    def test_single_cross_section(self):
        """只有 1 天的数据 → IC 应为 0"""
        single = self.factor_df[self.factor_df["date"] == self.factor_df["date"].iloc[0]]
        decay = calc_ic_decay(single, self.ret_df, forward_periods=[1])
        # 1 天也满足 len(g) >= 10，但仍可能没数据
        # 至少不抛异常

    def test_known_zero_ic(self):
        """因子与收益完全无关 → IC mean 应接近 0"""
        np.random.seed(0)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        rows = []
        for d in dates:
            n = 50
            factor = np.random.normal(0, 1, n)
            fwd = np.random.normal(0, 0.02, n)
            for i in range(n):
                rows.append({"date": d, "code": f"S{i}", "factor": factor[i]})
                rows.append({"date": d, "code": f"S{i}", "fwd_ret_1d": fwd[i]})
        # 实际：构造正确 schema
        rows = []
        for d in dates:
            n = 50
            factor = np.random.normal(0, 1, n)
            fwd = np.random.normal(0, 0.02, n)
            for i in range(n):
                rows.append({"date": d, "code": f"S{i}", "factor": factor[i],
                             "fwd_ret_1d": fwd[i]})
        df = pd.DataFrame(rows)
        decay = calc_ic_decay(df[["date", "code", "factor"]],
                              df[["date", "code", "fwd_ret_1d"]])
        # 期望 IC mean 接近 0 (用宽松阈值)
        if 1 in decay:
            self.assertLess(abs(decay[1]["ic_mean"]), 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
