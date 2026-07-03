"""
向量化 IC 分析测试

测试内容：
1. 正确性测试：向量化 IC 与逐日循环 IC（复刻现有 engine._calc_ic 逻辑）
   结果应在数值容差内一致
2. 性能对比测试：向量化版本应显著快于逐日循环版本
3. 边界条件测试：空数据、单日、样本不足、全 NaN
4. IC 类型测试：spearman vs pearson
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_engine_opt.vectorized_ic import (
    calc_ic_series_vectorized,
    calc_ic_stats_vectorized,
    ic_analysis_vectorized,
)


# ----------------------------------------------------------------------
# 参考实现：复刻 jingni-trader 现有 FactorEngine._calc_ic 的逐日循环逻辑
# ----------------------------------------------------------------------
def calc_ic_loop_reference(
    data: pd.DataFrame,
    factor_col: str,
    forward_col: str,
    ic_type: str = "spearman",
    min_count: int = 10,
) -> pd.Series:
    """逐日循环计算 IC（与现有 engine._calc_ic 等价）"""
    if forward_col not in data.columns:
        return pd.Series(dtype=float)
    ic_list = []
    dates = sorted(data["date"].unique())
    for dt in dates:
        cross = data[data["date"] == dt].dropna(subset=[factor_col, forward_col])
        if len(cross) < min_count:
            continue
        if ic_type == "spearman":
            ic, _ = stats.spearmanr(
                cross[factor_col], cross[forward_col], nan_policy="omit"
            )
        else:
            ic, _ = stats.pearsonr(
                cross[factor_col].fillna(0), cross[forward_col].fillna(0)
            )
        if not np.isnan(ic):
            ic_list.append({"date": dt, "ic": ic})
    if not ic_list:
        return pd.Series(dtype=float)
    ic_df = pd.DataFrame(ic_list)
    ic_df["date"] = pd.to_datetime(ic_df["date"])
    return ic_df.set_index("date")["ic"]


# ----------------------------------------------------------------------
# 测试数据生成
# ----------------------------------------------------------------------
def make_factor_data(n_dates=120, n_stocks=200, seed=42):
    """生成含因子与未来收益的测试数据"""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for dt in dates:
        # 因子与未来收益存在弱正相关（IC ≈ 0.05）
        factor = np.random.normal(0, 1, n_stocks)
        noise = np.random.normal(0, 1, n_stocks)
        forward_ret = 0.05 * factor + noise * 0.99
        for i, s in enumerate(stocks):
            rows.append({
                "date": dt,
                "code": s,
                "alpha_test": factor[i],
                "ret_forward_1d": forward_ret[i],
                "ret_forward_5d": forward_ret[i] / 5 + np.random.normal(0, 0.5),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 正确性测试
# ----------------------------------------------------------------------
class TestICCorrectness:
    def test_spearman_matches_loop(self):
        """向量化 Spearman IC 应与逐日循环结果一致"""
        data = make_factor_data(n_dates=60, n_stocks=100)
        ic_vec = calc_ic_series_vectorized(
            data, "alpha_test", "ret_forward_1d", ic_type="spearman"
        )
        ic_loop = calc_ic_loop_reference(
            data, "alpha_test", "ret_forward_1d", ic_type="spearman"
        )
        # 对齐 index 比较
        common = ic_vec.index.intersection(ic_loop.index)
        assert len(common) > 0, "无共同日期"
        diff = (ic_vec.loc[common] - ic_loop.loc[common]).abs()
        # scipy spearmanr 与向量化 rank+pearson 在数值上应有微小差异（浮点），
        # 但应在 1e-6 以内
        assert diff.max() < 1e-6, f"Spearman IC 偏差过大: max={diff.max()}"

    def test_pearson_matches_loop(self):
        """向量化 Pearson IC 应与逐日循环结果一致"""
        data = make_factor_data(n_dates=60, n_stocks=100)
        ic_vec = calc_ic_series_vectorized(
            data, "alpha_test", "ret_forward_1d", ic_type="pearson"
        )
        ic_loop = calc_ic_loop_reference(
            data, "alpha_test", "ret_forward_1d", ic_type="pearson"
        )
        common = ic_vec.index.intersection(ic_loop.index)
        assert len(common) > 0
        diff = (ic_vec.loc[common] - ic_loop.loc[common]).abs()
        # pearsonr 对 fillna(0) 敏感，向量化版本用 dropna，容差放宽
        assert diff.max() < 1e-6, f"Pearson IC 偏差过大: max={diff.max()}"

    def test_ic_stats_consistency(self):
        """IC 统计量应与 IC 序列一致"""
        data = make_factor_data(n_dates=60, n_stocks=100)
        ic_series = calc_ic_series_vectorized(
            data, "alpha_test", "ret_forward_1d", ic_type="spearman"
        )
        stats_dict = calc_ic_stats_vectorized(ic_series)
        assert abs(stats_dict["ic_mean"] - round(ic_series.mean(), 6)) < 1e-5
        assert abs(stats_dict["ic_positive_ratio"] - round((ic_series > 0).mean(), 4)) < 1e-3
        # IC_IR = mean/std
        expected_ir = ic_series.mean() / ic_series.std()
        assert abs(stats_dict["ic_ir"] - round(expected_ir, 4)) < 1e-3

    def test_ic_analysis_batch(self):
        """批量 IC 分析应返回所有因子的结果"""
        data = make_factor_data(n_dates=60, n_stocks=100)
        results = ic_analysis_vectorized(
            data, data,
            factor_names=["alpha_test"],
            forward_cols=["ret_forward_1d", "ret_forward_5d"],
        )
        assert "ret_forward_1d" in results
        assert "ret_forward_5d" in results
        assert len(results["ret_forward_1d"]) == 1
        assert results["ret_forward_1d"][0]["factor"] == "alpha_test"


# ----------------------------------------------------------------------
# 性能对比测试
# ----------------------------------------------------------------------
class TestICPerformance:
    def test_vectorized_faster_than_loop(self):
        """向量化 IC 应显著快于逐日循环（至少 2x）"""
        data = make_factor_data(n_dates=200, n_stocks=300)

        # 逐日循环
        t0 = time.perf_counter()
        ic_loop = calc_ic_loop_reference(
            data, "alpha_test", "ret_forward_1d", ic_type="spearman"
        )
        t_loop = time.perf_counter() - t0

        # 向量化
        t0 = time.perf_counter()
        ic_vec = calc_ic_series_vectorized(
            data, "alpha_test", "ret_forward_1d", ic_type="spearman"
        )
        t_vec = time.perf_counter() - t0

        speedup = t_loop / t_vec if t_vec > 0 else float("inf")
        print(f"\n[IC 性能] 逐日循环: {t_loop:.4f}s, 向量化: {t_vec:.4f}s, "
              f"加速比: {speedup:.2f}x")
        assert speedup >= 2.0, f"向量化未达 2x 加速: {speedup:.2f}x"
        assert len(ic_vec) == len(ic_loop), "结果行数不一致"


# ----------------------------------------------------------------------
# 边界条件测试
# ----------------------------------------------------------------------
class TestICBoundary:
    def test_empty_data(self):
        """空数据应返回空 Series"""
        empty = pd.DataFrame(columns=["date", "code", "f", "r"])
        ic = calc_ic_series_vectorized(empty, "f", "r")
        assert ic.empty

    def test_missing_column(self):
        """列不存在应返回空"""
        data = make_factor_data(n_dates=10, n_stocks=20)
        ic = calc_ic_series_vectorized(data, "nonexistent", "ret_forward_1d")
        assert ic.empty

    def test_insufficient_samples(self):
        """截面样本不足应跳过该日"""
        np.random.seed(1)
        # 仅 5 只股票，低于 min_count=10
        dates = pd.bdate_range("2023-01-01", periods=10)
        rows = []
        for dt in dates:
            for i in range(5):
                rows.append({
                    "date": dt, "code": f"{i:06d}.SZ",
                    "f": np.random.randn(), "r": np.random.randn(),
                })
        data = pd.DataFrame(rows)
        ic = calc_ic_series_vectorized(data, "f", "r", min_count=10)
        assert ic.empty, "样本不足的日期应被跳过"

    def test_all_nan_factor(self):
        """因子全 NaN 应返回空"""
        data = make_factor_data(n_dates=10, n_stocks=20)
        data["alpha_test"] = np.nan
        ic = calc_ic_series_vectorized(data, "alpha_test", "ret_forward_1d")
        assert ic.empty

    def test_single_date(self):
        """单日数据应能计算（若样本足够）"""
        np.random.seed(2)
        rows = []
        for i in range(50):
            rows.append({
                "date": pd.Timestamp("2023-01-02"), "code": f"{i:06d}.SZ",
                "f": np.random.randn(), "r": np.random.randn(),
            })
        data = pd.DataFrame(rows)
        ic = calc_ic_series_vectorized(data, "f", "r", min_count=10)
        assert len(ic) == 1, "单日应返回 1 个 IC 值"
        assert not np.isnan(ic.iloc[0])
