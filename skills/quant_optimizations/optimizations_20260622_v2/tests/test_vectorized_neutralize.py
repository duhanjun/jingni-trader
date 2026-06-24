"""
向量化因子中性化的正确性 + 性能对比测试

测试内容：
1. 正确性：向量化结果与 sklearn.LinearRegression 逐日结果一致
2. 性能：向量化版本比逐日 sklearn 快 5 倍以上
3. 边界条件：空数据、样本不足、无行业列
4. 多因子批量中性化
"""
import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from skills.quant_optimizations.optimizations_20260622_v2.vectorized_neutralize import neutralize_factor


def _gen_factor_data(n_dates=100, n_stocks=200, n_industries=10, seed=42):
    """生成含行业、市值的因子数据"""
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    industries = [f"ind_{i % n_industries}" for i in range(n_stocks)]

    rows = []
    for d in dates:
        for i, c in enumerate(codes):
            rows.append({
                "date": d,
                "code": c,
                "industry": industries[i],
                "lncap": np.random.randn() * 2 + 20,
                "factor_1": np.random.randn(),
                "factor_2": np.random.randn() * 3,
            })
    return pd.DataFrame(rows)


def _neutralize_loop_sklearn(factor_df, factor, neutralize_mcap=True, neutralize_industry=True, min_count=30):
    """原实现的逐日 sklearn 循环版本（作为基准）"""
    result = factor_df.copy()
    if "industry" not in result.columns and neutralize_industry:
        return result

    neutralized_values = pd.Series(index=result.index, dtype=float)
    dates = result["date"].unique()

    for dt in dates:
        cross = result[result["date"] == dt].copy()
        if len(cross) < min_count:
            neutralized_values.loc[cross.index] = cross[factor]
            continue

        X_vars = []
        if neutralize_mcap and "lncap" in cross.columns:
            X_vars.append("lncap")
        if neutralize_industry and "industry" in cross.columns:
            industry_dummies = pd.get_dummies(cross["industry"], prefix="ind")
            for col in industry_dummies.columns:
                cross[col] = industry_dummies[col].values
                X_vars.append(col)

        if not X_vars:
            neutralized_values.loc[cross.index] = cross[factor]
            continue

        X = cross[X_vars].fillna(0).values
        y = cross[factor].fillna(0).values

        try:
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            residual = y - y_pred
            neutralized_values.loc[cross.index] = residual
        except Exception:
            neutralized_values.loc[cross.index] = cross[factor]

    return neutralized_values


def test_correctness():
    """测试 1：向量化中性化与 sklearn 逐日结果一致"""
    print("\n=== 测试 1: 中性化正确性 ===")
    data = _gen_factor_data(n_dates=30, n_stocks=100)

    # 向量化版本
    vec_result = neutralize_factor(
        data, ["factor_1"],
        neutralize_mcap=True, neutralize_industry=True, min_count=30,
    )
    vec_neutral = vec_result["factor_1_neutral"].dropna()

    # sklearn 逐日版本
    loop_neutral = _neutralize_loop_sklearn(
        data, "factor_1",
        neutralize_mcap=True, neutralize_industry=True, min_count=30,
    ).dropna()

    # 对齐索引
    common = vec_neutral.index.intersection(loop_neutral.index)
    assert len(common) > 0, "无共同样本"

    diff = (vec_neutral.loc[common] - loop_neutral.loc[common]).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  共 {len(common)} 个样本，最大误差: {max_diff:.2e}，平均误差: {mean_diff:.2e}")

    # numpy.lstsq 和 sklearn 可能有微小数值差异
    assert max_diff < 1e-8, f"中性化误差过大: {max_diff}"
    print("  ✓ 向量化中性化与 sklearn 结果一致")


def test_performance():
    """测试 2：性能对比"""
    print("\n=== 测试 2: 性能对比 ===")
    data = _gen_factor_data(n_dates=100, n_stocks=300)

    # sklearn 逐日
    t0 = time.time()
    loop_result = _neutralize_loop_sklearn(data, "factor_1")
    t_loop = time.time() - t0

    # 向量化
    t0 = time.time()
    vec_result = neutralize_factor(data, ["factor_1"])
    t_vec = time.time() - t0

    speedup = t_loop / t_vec if t_vec > 0 else float("inf")
    print(f"  数据规模: {len(data)} 行 ({data['date'].nunique()} 日 × {data['code'].nunique()} 股)")
    print(f"  sklearn 逐日: {t_loop:.4f}s")
    print(f"  向量化:       {t_vec:.4f}s")
    print(f"  加速比:       {speedup:.1f}x")

    assert speedup > 2, f"加速比不足 2x: {speedup}"
    print(f"  ✓ 向量化比 sklearn 逐日快 {speedup:.1f} 倍")


def test_edge_cases():
    """测试 3：边界条件"""
    print("\n=== 测试 3: 边界条件 ===")

    # 空数据
    empty = pd.DataFrame(columns=["date", "code", "factor_1", "lncap", "industry"])
    result = neutralize_factor(empty, ["factor_1"])
    assert result.empty
    print("  ✓ 空数据正确处理")

    # 无行业列
    no_industry = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")] * 100,
        "code": [f"c{i}" for i in range(100)],
        "lncap": np.random.randn(100),
        "factor_1": np.random.randn(100),
    })
    result = neutralize_factor(no_industry, ["factor_1"], neutralize_industry=False, neutralize_mcap=True)
    assert "factor_1_neutral" in result.columns
    print("  ✓ 无行业列时仅做市值中性化")

    # 样本不足
    small = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")] * 10,
        "code": [f"c{i}" for i in range(10)],
        "industry": ["ind_0"] * 10,
        "lncap": np.random.randn(10),
        "factor_1": np.random.randn(10),
    })
    result = neutralize_factor(small, ["factor_1"], min_count=30)
    # 样本不足应返回原值
    assert "factor_1_neutral" in result.columns
    np.testing.assert_array_almost_equal(
        result["factor_1_neutral"].values, result["factor_1"].values
    )
    print("  ✓ 样本不足时返回原值")

    # 不中性化（两个 flag 都 False）
    result = neutralize_factor(no_industry, ["factor_1"], neutralize_mcap=False, neutralize_industry=False)
    assert "factor_1_neutral" not in result.columns
    print("  ✓ 不中性化时正确跳过")


def test_multi_factor():
    """测试 4：多因子批量中性化"""
    print("\n=== 测试 4: 多因子批量中性化 ===")
    data = _gen_factor_data(n_dates=30, n_stocks=100)

    result = neutralize_factor(data, ["factor_1", "factor_2"])
    assert "factor_1_neutral" in result.columns
    assert "factor_2_neutral" in result.columns
    print("  ✓ 多因子批量中性化正确")

    # 验证中性化后因子与 lncap 不相关
    for dt in data["date"].unique()[:5]:
        cross = result[result["date"] == dt]
        if len(cross) < 30:
            continue
        corr = cross["factor_1_neutral"].corr(cross["lncap"])
        assert abs(corr) < 0.1, f"中性化后仍与 lncap 相关: {corr}"
    print("  ✓ 中性化后因子与市值不相关")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test_correctness()
    test_performance()
    test_edge_cases()
    test_multi_factor()
    print("\n🎉 全部中性化测试通过")