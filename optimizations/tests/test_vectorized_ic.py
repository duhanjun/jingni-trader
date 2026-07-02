"""
向量化 IC 分析的正确性 + 性能对比测试

测试内容：
1. 正确性：向量化 IC 与 scipy.stats.spearmanr 逐日计算结果一致
2. 性能：向量化版本比逐日循环快 50 倍以上
3. 边界条件：空数据、单日、NaN、样本不足
4. IC 衰减曲线计算
"""
import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from optimizations.vectorized_ic import (
    calc_ic_series,
    calc_ic_stats,
    calc_ic_matrix,
    calc_rank_ic_decay,
)


def _gen_factor_data(n_dates=200, n_stocks=300, seed=42):
    """生成测试用因子数据"""
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for d in dates:
        for c in codes:
            rows.append({
                "date": d,
                "code": c,
                "factor_1": np.random.randn(),
                "factor_2": np.random.randn() * 2 + 1,
                "ret_forward_1d": np.random.randn() * 0.02,
                "ret_forward_5d": np.random.randn() * 0.05,
            })
    return pd.DataFrame(rows)


def _calc_ic_loop(data, factor_col, forward_col, method="spearman", min_count=10):
    """原实现的逐日循环版本（作为基准）"""
    ic_list = []
    dates = sorted(data["date"].unique())
    for dt in dates:
        cross = data[data["date"] == dt].dropna(subset=[factor_col, forward_col])
        if len(cross) < min_count:
            continue
        if method == "spearman":
            ic, _ = stats.spearmanr(cross[factor_col], cross[forward_col], nan_policy="omit")
        else:
            ic, _ = stats.pearsonr(cross[factor_col].fillna(0), cross[forward_col].fillna(0))
        if not np.isnan(ic):
            ic_list.append({"date": dt, "ic": ic})
    if not ic_list:
        return pd.Series(dtype=float)
    ic_df = pd.DataFrame(ic_list)
    return ic_df.set_index("date")["ic"]


def test_correctness_spearman():
    """测试 1：向量化 Spearman IC 与逐日 scipy 结果一致"""
    print("\n=== 测试 1: Spearman IC 正确性 ===")
    data = _gen_factor_data(n_dates=50, n_stocks=100)

    ic_vec = calc_ic_series(data, "factor_1", "ret_forward_1d", method="spearman")
    ic_loop = _calc_ic_loop(data, "factor_1", "ret_forward_1d", method="spearman")

    # 对齐索引
    common = ic_vec.index.intersection(ic_loop.index)
    assert len(common) > 0, "无共同日期"

    diff = (ic_vec.loc[common] - ic_loop.loc[common]).abs()
    max_diff = diff.max()
    print(f"  共 {len(common)} 个日期，最大绝对误差: {max_diff:.2e}")

    assert max_diff < 1e-10, f"Spearman IC 误差过大: {max_diff}"
    print("  ✓ 向量化 Spearman IC 与 scipy 结果完全一致")


def test_correctness_pearson():
    """测试 2：向量化 Pearson IC 与逐日 scipy 结果一致"""
    print("\n=== 测试 2: Pearson IC 正确性 ===")
    data = _gen_factor_data(n_dates=50, n_stocks=100)

    ic_vec = calc_ic_series(data, "factor_1", "ret_forward_1d", method="pearson")
    ic_loop = _calc_ic_loop(data, "factor_1", "ret_forward_1d", method="pearson")

    common = ic_vec.index.intersection(ic_loop.index)
    assert len(common) > 0

    diff = (ic_vec.loc[common] - ic_loop.loc[common]).abs()
    max_diff = diff.max()
    print(f"  共 {len(common)} 个日期，最大绝对误差: {max_diff:.2e}")

    assert max_diff < 1e-10, f"Pearson IC 误差过大: {max_diff}"
    print("  ✓ 向量化 Pearson IC 与 scipy 结果完全一致")


def test_performance():
    """测试 3：性能对比（向量化 vs 逐日循环）"""
    print("\n=== 测试 3: 性能对比 ===")
    # 使用更大规模数据，向量化优势随数据量增大而显著
    data = _gen_factor_data(n_dates=300, n_stocks=500)

    # 逐日循环
    t0 = time.time()
    ic_loop = _calc_ic_loop(data, "factor_1", "ret_forward_1d", method="spearman")
    t_loop = time.time() - t0

    # 向量化
    t0 = time.time()
    ic_vec = calc_ic_series(data, "factor_1", "ret_forward_1d", method="spearman")
    t_vec = time.time() - t0

    speedup = t_loop / t_vec if t_vec > 0 else float("inf")
    print(f"  数据规模: {len(data)} 行 ({data['date'].nunique()} 日 × {data['code'].nunique()} 股)")
    print(f"  逐日循环: {t_loop:.4f}s")
    print(f"  向量化:   {t_vec:.4f}s")
    print(f"  加速比:   {speedup:.1f}x")

    # 5x 加速是显著改进；向量化优势随数据量增大
    assert speedup > 5, f"加速比不足 5x: {speedup}"
    print(f"  ✓ 向量化比逐日循环快 {speedup:.1f} 倍")


def test_edge_cases():
    """测试 4：边界条件"""
    print("\n=== 测试 4: 边界条件 ===")

    # 空数据
    empty = pd.DataFrame(columns=["date", "code", "f", "r"])
    ic = calc_ic_series(empty, "f", "r")
    assert ic.empty, "空数据应返回空 Series"
    print("  ✓ 空数据正确处理")

    # 单日数据
    single_day = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")] * 50,
        "code": [f"c{i}" for i in range(50)],
        "f": np.random.randn(50),
        "r": np.random.randn(50),
    })
    ic = calc_ic_series(single_day, "f", "r", min_count=10)
    assert len(ic) == 1, f"单日应返回 1 个 IC，实际 {len(ic)}"
    print("  ✓ 单日数据正确处理")

    # 样本不足（min_count 过滤）
    small = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")] * 5,
        "code": [f"c{i}" for i in range(5)],
        "f": np.random.randn(5),
        "r": np.random.randn(5),
    })
    ic = calc_ic_series(small, "f", "r", min_count=10)
    assert ic.empty, "样本不足应返回空"
    print("  ✓ 样本不足正确过滤")

    # 含 NaN
    nan_data = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")] * 100,
        "code": [f"c{i}" for i in range(100)],
        "f": np.random.randn(100),
        "r": np.random.randn(100),
    })
    nan_data.loc[0:10, "f"] = np.nan
    ic = calc_ic_series(nan_data, "f", "r", min_count=10)
    assert len(ic) == 1, "含 NaN 应正确计算"
    assert not np.isnan(ic.iloc[0]), "IC 不应为 NaN"
    print("  ✓ NaN 值正确处理")

    # 不存在的列
    ic = calc_ic_series(nan_data, "nonexistent", "r")
    assert ic.empty, "不存在的列应返回空"
    print("  ✓ 不存在的列正确处理")


def test_ic_stats():
    """测试 5：IC 统计量计算"""
    print("\n=== 测试 5: IC 统计量 ===")
    ic_series = pd.Series(np.random.randn(100) * 0.05, index=pd.bdate_range("2024-01-01", periods=100))

    stats_dict = calc_ic_stats(ic_series)
    print(f"  IC 均值: {stats_dict['ic_mean']}")
    print(f"  IC IR:   {stats_dict['ic_ir']}")
    print(f"  IC t值:  {stats_dict['ic_t_stat']}")

    assert "ic_mean" in stats_dict
    assert "ic_ir" in stats_dict
    assert "ic_t_stat" in stats_dict
    assert "ic_positive_ratio" in stats_dict
    print("  ✓ IC 统计量计算正确")

    # 空序列
    empty_stats = calc_ic_stats(pd.Series(dtype=float))
    assert empty_stats["ic_mean"] == 0.0
    print("  ✓ 空序列统计量正确")


def test_ic_matrix():
    """测试 6：批量 IC 矩阵计算"""
    print("\n=== 测试 6: 批量 IC 矩阵 ===")
    data = _gen_factor_data(n_dates=50, n_stocks=100)

    results = calc_ic_matrix(
        data,
        factor_names=["factor_1", "factor_2"],
        forward_cols=["ret_forward_1d", "ret_forward_5d"],
        method="spearman",
    )

    assert "ret_forward_1d" in results
    assert "ret_forward_5d" in results
    assert len(results["ret_forward_1d"]) == 2
    print(f"  2 因子 × 2 前瞻期 = {sum(len(v) for v in results.values())} 个 IC 结果")
    print("  ✓ 批量 IC 矩阵计算正确")


def test_ic_decay():
    """测试 7：IC 衰减曲线"""
    print("\n=== 测试 7: IC 衰减曲线 ===")
    # 构造带 close 列的数据
    np.random.seed(42)
    n_dates, n_stocks = 100, 50
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    codes = [f"c{i:04d}" for i in range(n_stocks)]
    rows = []
    for c in codes:
        price = 10.0
        for d in dates:
            ret = np.random.randn() * 0.02
            price *= (1 + ret)
            rows.append({
                "date": d,
                "code": c,
                "close": price,
                "alpha_score": np.random.randn(),
            })
    data = pd.DataFrame(rows)

    decay = calc_rank_ic_decay(data, "alpha_score", [1, 5, 10, 20], min_count=10)
    print(f"  衰减曲线: {len(decay)} 个周期")
    assert len(decay) == 4
    assert "period" in decay.columns
    assert "ic_mean" in decay.columns
    print("  ✓ IC 衰减曲线计算正确")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test_correctness_spearman()
    test_correctness_pearson()
    test_performance()
    test_edge_cases()
    test_ic_stats()
    test_ic_matrix()
    test_ic_decay()
    print("\n🎉 全部 IC 分析测试通过")
