"""
Module 1 测试: 向量化 IC 分析
"""
import time
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from skills.quant-optimizations.quant_opt_20260617.core.vectorized_ic import (
    VectorizedICAnalyzer,
    ic_analysis_compatible,
    _safe_pearson,
    _safe_spearman,
)


def _make_panel(n_stocks: int = 30, n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """构造合成的 (date, code, factor, ret) 面板"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    codes = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]
    rows = []
    for d in dates:
        for c in codes:
            # 因子 = 正态噪声 + 一些公共因子
            f = rng.normal(0, 1)
            # 收益率 = 0.3 * 因子 + 噪声  => 期望 IC ≈ 0.3
            r = 0.3 * f + rng.normal(0, 1)
            rows.append((d, c, f, r))
    df = pd.DataFrame(rows, columns=["date", "code", "factor1", "ret"])
    return df


def _make_factor_ret_pair(seed: int = 42, n: int = 200) -> tuple:
    """构造单日截面数据"""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, n)
    r = 0.5 * f + rng.normal(0, 1, n)
    return f, r


class TestVectorizedIC:
    """向量化 IC 分析测试套件"""

    def test_pearson_matches_scipy(self):
        """正确性: 我们的 pearson 与 scipy.stats.pearsonr 应一致"""
        f, r = _make_factor_ret_pair(seed=1, n=500)
        ours = _safe_pearson(f, r)
        ref, _ = stats.pearsonr(f, r)
        assert abs(ours - ref) < 1e-9, f"Pearson mismatch: {ours} vs {ref}"

    def test_spearman_matches_scipy(self):
        """正确性: 我们的 spearman 与 scipy 应一致"""
        f, r = _make_factor_ret_pair(seed=2, n=500)
        ours = _safe_spearman(f, r)
        ref, _ = stats.spearmanr(f, r)
        assert abs(ours - ref) < 1e-9, f"Spearman mismatch: {ours} vs {ref}"

    def test_nan_handling(self):
        """边界条件: NaN 处理 - 仅保留双侧有效值"""
        f = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        r = np.array([0.1, 0.2, 0.3, np.nan, 0.5])
        result = _safe_pearson(f, r)
        assert np.isfinite(result), "NaN 输入下应返回有限值"
        # 双侧有效位置: 0,1,4 → 与 scipy 对齐
        ref, _ = stats.pearsonr([1.0, 2.0, 5.0], [0.1, 0.2, 0.5])
        assert abs(result - ref) < 1e-9

    def test_short_input_returns_nan(self):
        """边界条件: 输入过短返回 NaN"""
        assert np.isnan(_safe_pearson(np.array([1.0, 2.0]), np.array([0.1, 0.2])))
        assert np.isnan(_safe_spearman(np.array([1.0]), np.array([0.1])))

    def test_constant_input_returns_nan(self):
        """边界条件: 常数列不应抛异常"""
        f = np.ones(10)
        r = np.arange(10, dtype=float)
        result = _safe_pearson(f, r)
        assert np.isnan(result) or result == 0.0

    def test_ic_summary_with_known_signal(self):
        """正确性: 已知 IC≈0.3 的合成数据应能恢复"""
        panel = _make_panel(n_stocks=50, n_days=120, seed=7)
        analyzer = VectorizedICAnalyzer()
        results = analyzer.analyze(
            panel[["date", "code", "factor1"]],
            panel[["date", "code"]].assign(ret_forward_5d=panel["ret"]),
            ["factor1"],
            periods=(5,),
        )
        m = results["factor1"][5]
        # 我们构造 ret = 0.3 * f + noise，所以 IC 期望 0.3 附近
        assert 0.20 < m["ic_mean"] < 0.40, f"IC_mean 偏离预期: {m['ic_mean']}"
        assert m["ic_ir"] > 0.3
        assert m["n_days"] > 50

    def test_hac_t_stat_is_smaller_than_naive(self):
        """正确性: 当 IC 序列正自相关时，HAC t-stat 应 <= 普通 t-stat"""
        # 构造强自相关 IC 序列
        rng = np.random.default_rng(0)
        ic = np.cumsum(rng.normal(0.05, 0.3, 100))
        analyzer = VectorizedICAnalyzer()
        t_naive = ic.mean() / (ic.std() / np.sqrt(len(ic)))
        t_hac = analyzer._hac_t_stat(ic, max_lag=10)
        # HAC t 应更保守
        assert t_hac < t_naive, f"HAC t={t_hac} 应 < naive t={t_naive}"

    def test_compatible_api(self):
        """兼容性: 与原 engine.ic_analysis 返回结构兼容"""
        panel = _make_panel(n_stocks=20, n_days=60, seed=3)
        ret_df = panel[["date", "code"]].assign(
            ret_forward_1d=panel["ret"],
            ret_forward_5d=panel["ret"],
            ret_forward_20d=panel["ret"],
        )
        result = ic_analysis_compatible(
            panel[["date", "code", "factor1"]],
            ret_df,
            ["factor1"],
        )
        assert "ret_forward_5d" in result
        assert isinstance(result["ret_forward_5d"], list)
        item = result["ret_forward_5d"][0]
        # 关键字段都在
        for key in [
            "factor", "forward_period", "ic_mean", "ic_std",
            "ic_ir", "ic_positive_ratio", "ic_t_stat",
        ]:
            assert key in item, f"缺失字段 {key}"
        # 新增字段
        for key in ["ic_t_stat_hac", "rank_ic_mean", "rank_ic_ir", "n_days"]:
            assert key in item, f"新增字段缺失 {key}"

    def test_auto_select(self):
        """auto_select: 应能筛选掉无信号因子"""
        panel = _make_panel(n_stocks=30, n_days=80, seed=4)
        # 加一个噪声因子
        rng = np.random.default_rng(99)
        panel["noise"] = rng.normal(0, 1, len(panel))
        ret_df = panel[["date", "code"]].assign(
            ret_forward_1d=panel["ret"],
            ret_forward_5d=panel["ret"],
            ret_forward_20d=panel["ret"],
        )
        analyzer = VectorizedICAnalyzer()
        results = analyzer.analyze(
            panel[["date", "code", "factor1", "noise"]],
            ret_df, ["factor1", "noise"], periods=(5,),
        )
        selected = analyzer.auto_select(results, primary_period=5)
        assert "factor1" in selected
        assert "noise" not in selected, "纯噪声因子应被剔除"

    def test_performance_speedup(self):
        """性能: 向量化版本应显著快于逐日 for-loop"""
        panel = _make_panel(n_stocks=80, n_days=500, seed=11)
        f_series = panel.set_index(["date", "code"])["factor1"]
        r_series = panel.set_index(["date", "code"])["ret"]

        analyzer = VectorizedICAnalyzer()

        # 向量化版
        t0 = time.perf_counter()
        for _ in range(3):
            ic_vec = analyzer.compute_ic_series(f_series, r_series, "pearson")
        t_vec = time.perf_counter() - t0

        # for-loop 版（用 apply）
        t0 = time.perf_counter()
        for _ in range(3):
            aligned = pd.concat([f_series.rename("f"), r_series.rename("r")], axis=1).dropna()
            ic_loop = pd.Series({
                d: _safe_pearson(g["f"].values, g["r"].values)
                for d, g in aligned.groupby(level="date")
            })
        t_loop = time.perf_counter() - t0

        # 向量化版结果应一致
        assert len(ic_vec) == len(ic_loop)
        np.testing.assert_allclose(
            ic_vec.sort_index().values,
            ic_loop.sort_index().values,
            atol=1e-6,
        )
        # 速度对比
        speedup = t_loop / t_vec
        print(f"\n  [IC perf] vec={t_vec:.3f}s vs loop={t_loop:.3f}s, speedup={speedup:.2f}x")
        assert t_vec < t_loop, "向量化版应快于循环版"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])