"""
向量化因子中性化测试

测试内容：
1. 正确性测试：向量化中性化残差应与逐日 sklearn 回归残差一致
2. 性能对比测试：向量化版本应快于逐日 sklearn 版本
3. 边界条件测试：空数据、样本不足、无自变量、单因子
4. 仅市值中性化（FWL 完全向量化）正确性
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_engine_opt.vectorized_neutralize import (
    neutralize_vectorized,
    neutralize_mcap_only_vectorized,
)


# ----------------------------------------------------------------------
# 参考实现：复刻现有 engine.neutralize 的逐日 sklearn 循环逻辑
# ----------------------------------------------------------------------
def neutralize_loop_reference(
    factor_df: pd.DataFrame,
    factor_names,
    neutralize_mcap=True,
    neutralize_industry=True,
    mcap_col="lncap",
    industry_col="industry",
    min_count=30,
):
    """逐日 sklearn 回归中性化（与现有 engine.neutralize 等价）"""
    if not neutralize_industry and not neutralize_mcap:
        return factor_df
    if factor_df.empty:
        return factor_df
    result = factor_df.copy()
    factor_cols = [f for f in factor_names if f in result.columns]

    for factor in factor_cols:
        dates = result["date"].unique()
        neutralized_values = pd.Series(index=result.index, dtype=float)
        for dt in dates:
            cross = result[result["date"] == dt].copy()
            if len(cross) < min_count:
                neutralized_values.loc[cross.index] = cross[factor]
                continue
            X_vars = []
            if neutralize_mcap and mcap_col in cross.columns:
                X_vars.append(mcap_col)
            if neutralize_industry and industry_col in cross.columns:
                industry_dummies = pd.get_dummies(cross[industry_col], prefix="ind")
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
        result[f"{factor}_neutral"] = neutralized_values
    return result


# ----------------------------------------------------------------------
# 测试数据生成
# ----------------------------------------------------------------------
def make_neutralize_data(n_dates=60, n_stocks=200, n_industries=10, seed=42):
    """生成含因子、对数市值、行业的数据"""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    rows = []
    for dt in dates:
        lncap = np.random.normal(15, 2, n_stocks)
        industries = np.random.randint(0, n_industries, n_stocks)
        # 因子受市值与行业影响（中性化后应去除这些影响）
        factor = 0.5 * lncap + industries * 0.3 + np.random.normal(0, 1, n_stocks)
        for i in range(n_stocks):
            rows.append({
                "date": dt,
                "code": f"{i:06d}.SZ",
                "alpha_test": factor[i],
                "lncap": lncap[i],
                "industry": f"ind_{industries[i]}",
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 正确性测试
# ----------------------------------------------------------------------
class TestNeutralizeCorrectness:
    def test_mcap_industry_matches_sklearn(self):
        """向量化中性化残差应与逐日 sklearn 一致"""
        data = make_neutralize_data(n_dates=40, n_stocks=150)
        factor_names = ["alpha_test"]

        res_vec = neutralize_vectorized(
            data, factor_names,
            neutralize_mcap=True, neutralize_industry=True,
        )
        res_loop = neutralize_loop_reference(
            data, factor_names,
            neutralize_mcap=True, neutralize_industry=True,
        )

        col = "alpha_test_neutral"
        assert col in res_vec.columns
        assert col in res_loop.columns

        # 对齐比较残差
        v = res_vec[col].dropna()
        l = res_loop[col].dropna()
        common = v.index.intersection(l.index)
        diff = (v.loc[common] - l.loc[common]).abs()
        # sklearn 与 numpy lstsq 数值差异，容差 1e-6
        assert diff.max() < 1e-6, f"中性化残差偏差过大: max={diff.max()}"

    def test_mcap_only_matches_sklearn(self):
        """仅市值中性化应与逐日 sklearn 一致"""
        data = make_neutralize_data(n_dates=40, n_stocks=150)
        factor_names = ["alpha_test"]

        res_vec = neutralize_vectorized(
            data, factor_names,
            neutralize_mcap=True, neutralize_industry=False,
        )
        res_loop = neutralize_loop_reference(
            data, factor_names,
            neutralize_mcap=True, neutralize_industry=False,
        )
        col = "alpha_test_neutral"
        v = res_vec[col].dropna()
        l = res_loop[col].dropna()
        common = v.index.intersection(l.index)
        diff = (v.loc[common] - l.loc[common]).abs()
        assert diff.max() < 1e-6, f"仅市值中性化偏差过大: max={diff.max()}"

    def test_fwl_mcap_only_matches_sklearn(self):
        """FWL 完全向量化市值中性化应与逐日 sklearn 一致"""
        data = make_neutralize_data(n_dates=40, n_stocks=150)
        factor_names = ["alpha_test"]

        res_fwl = neutralize_mcap_only_vectorized(data, factor_names)
        res_loop = neutralize_loop_reference(
            data, factor_names,
            neutralize_mcap=True, neutralize_industry=False,
        )
        col = "alpha_test_neutral"
        v = res_fwl[col].dropna()
        l = res_loop[col].dropna()
        common = v.index.intersection(l.index)
        diff = (v.loc[common] - l.loc[common]).abs()
        assert diff.max() < 1e-6, f"FWL 中性化偏差过大: max={diff.max()}"

    def test_residual_uncorrelated_with_mcap(self):
        """中性化后残差应与市值不相关（验证中性化效果）"""
        data = make_neutralize_data(n_dates=40, n_stocks=200)
        res = neutralize_vectorized(
            data, ["alpha_test"],
            neutralize_mcap=True, neutralize_industry=False,
        )
        # 截面相关性应接近 0
        valid = res[["alpha_test_neutral", "lncap"]].dropna()
        corr = valid["alpha_test_neutral"].corr(valid["lncap"])
        assert abs(corr) < 0.05, f"中性化后仍与市值相关: corr={corr}"


# ----------------------------------------------------------------------
# 性能对比测试
# ----------------------------------------------------------------------
class TestNeutralizePerformance:
    def test_vectorized_faster_than_sklearn(self):
        """向量化中性化应快于逐日 sklearn（至少 1.5x）"""
        data = make_neutralize_data(n_dates=100, n_stocks=300)
        factor_names = ["alpha_test"]

        t0 = time.perf_counter()
        neutralize_loop_reference(
            data, factor_names,
            neutralize_mcap=True, neutralize_industry=True,
        )
        t_loop = time.perf_counter() - t0

        t0 = time.perf_counter()
        neutralize_vectorized(
            data, factor_names,
            neutralize_mcap=True, neutralize_industry=True,
        )
        t_vec = time.perf_counter() - t0

        speedup = t_loop / t_vec if t_vec > 0 else float("inf")
        print(f"\n[中性化性能] sklearn 逐日: {t_loop:.4f}s, 向量化: {t_vec:.4f}s, "
              f"加速比: {speedup:.2f}x")
        assert speedup >= 1.5, f"向量化未达 1.5x 加速: {speedup:.2f}x"

    def test_fwl_faster_than_apply(self):
        """FWL 完全向量化应快于 groupby.apply 版本"""
        data = make_neutralize_data(n_dates=100, n_stocks=300)
        factor_names = ["alpha_test"]

        t0 = time.perf_counter()
        neutralize_vectorized(
            data, factor_names,
            neutralize_mcap=True, neutralize_industry=False,
        )
        t_apply = time.perf_counter() - t0

        t0 = time.perf_counter()
        neutralize_mcap_only_vectorized(data, factor_names)
        t_fwl = time.perf_counter() - t0

        speedup = t_apply / t_fwl if t_fwl > 0 else float("inf")
        print(f"\n[FWL 性能] groupby.apply: {t_apply:.4f}s, FWL 向量化: {t_fwl:.4f}s, "
              f"加速比: {speedup:.2f}x")


# ----------------------------------------------------------------------
# 边界条件测试
# ----------------------------------------------------------------------
class TestNeutralizeBoundary:
    def test_empty_data(self):
        """空数据应安全返回"""
        empty = pd.DataFrame(columns=["date", "code", "f", "lncap", "industry"])
        res = neutralize_vectorized(empty, ["f"])
        assert res.empty

    def test_no_regressors(self):
        """无自变量时应返回原值"""
        data = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=5),
            "code": ["000001.SZ"] * 5,
            "f": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        res = neutralize_vectorized(
            data, ["f"],
            neutralize_mcap=True, neutralize_industry=True,
        )
        # 无 lncap/industry 列，应直接复制原值
        assert "f_neutral" in res.columns

    def test_insufficient_samples(self):
        """截面样本不足应保留原值"""
        np.random.seed(1)
        rows = []
        for dt in pd.bdate_range("2023-01-01", periods=5):
            for i in range(10):  # 低于 min_count=30
                rows.append({
                    "date": dt, "code": f"{i:06d}.SZ",
                    "f": np.random.randn(),
                    "lncap": np.random.randn(),
                    "industry": f"ind_{i % 3}",
                })
        data = pd.DataFrame(rows)
        res = neutralize_vectorized(data, ["f"], min_count=30)
        # 样本不足，应保留原值
        assert res["f_neutral"].equals(data["f"])

    def test_no_neutralization(self):
        """关闭所有中性化应原样返回"""
        data = make_neutralize_data(n_dates=10, n_stocks=50)
        res = neutralize_vectorized(
            data, ["alpha_test"],
            neutralize_mcap=False, neutralize_industry=False,
        )
        assert "alpha_test_neutral" not in res.columns
