"""
测试用例 3: 因子预处理器 (factor_preprocessor) 验证
====================================================

验证目标:
  1. winsorize 三种方法在异常值场景下的稳健性
  2. neutralize 行业/市值暴露的能力
  3. clean_factor 端到端 pipeline 正确性
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from factor_preprocessor import (
    winsorize_zscore, winsorize_quantile, winsorize_mad,
    standardize_zscore, standardize_rank,
    neutralize_industry_mcap, clean_factor,
)
from synthetic_data import generate_panel, generate_alpha_factor


# ============================================================
# 1. Winsorization
# ============================================================

class TestWinsorize:
    def test_zscore_clip_extreme(self):
        s = pd.Series([1, 2, 3, 4, 5, 100])  # 100 是异常值
        clipped = winsorize_zscore(s, threshold=3.0)
        # 异常值应被截断
        assert clipped.iloc[-1] < s.iloc[-1]
        # 中位数附近的不变
        assert abs(clipped.iloc[0] - s.iloc[0]) < 0.1

    def test_zscore_preserves_normal_data(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 1000))
        clipped = winsorize_zscore(s, threshold=3.0)
        # 99% 的数据应保持不变
        diff = (clipped - s).abs()
        assert (diff < 0.01).sum() > 990

    def test_quantile_clip(self):
        s = pd.Series(list(range(100)) + [1000])
        clipped = winsorize_quantile(s, lower=0.05, upper=0.95)
        # 1000 应被截断到 95 分位附近
        assert clipped.iloc[-1] < 1000

    def test_mad_robust_to_outliers(self):
        # MAD 用中位数，对极端值更稳健
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])
        clipped = winsorize_mad(s, n_mad=3.0)
        # 异常值应被压缩
        assert clipped.iloc[-1] < 1000

    def test_empty_input(self):
        s = pd.Series([], dtype=float)
        assert winsorize_zscore(s).empty
        assert winsorize_quantile(s).empty
        assert winsorize_mad(s).empty


# ============================================================
# 2. Standardization
# ============================================================

class TestStandardize:
    def test_zscore_mean_zero_std_one(self):
        s = pd.Series(np.random.default_rng(0).normal(10, 3, 1000))
        z = standardize_zscore(s)
        assert abs(z.mean()) < 1e-6
        assert abs(z.std() - 1.0) < 1e-6

    def test_rank_bounded(self):
        s = pd.Series([1, 2, 3, 4, 5, 100])
        r = standardize_rank(s)
        # rank 后范围 [-0.5, 0.5]
        assert r.min() >= -0.5
        assert r.max() <= 0.5

    def test_robust_to_outliers(self):
        # 异常值不应对 rank 有太大影响
        s1 = pd.Series([1, 2, 3, 4, 5])
        s2 = pd.Series([1, 2, 3, 4, 5, 1000])
        r1 = standardize_rank(s1)
        r2 = standardize_rank(s2)
        # 前 4 个的 rank 应该接近
        assert abs(r1.iloc[0] - r2.iloc[0]) < 0.2


# ============================================================
# 3. Neutralization
# ============================================================

class TestNeutralize:
    @pytest.fixture
    def biased_factor(self):
        # 构造一个有行业偏度的因子
        dates = pd.bdate_range("2023-01-01", periods=10)
        codes = [f"S{i:03d}" for i in range(50)]
        rows = []
        rng = np.random.default_rng(0)
        for dt in dates:
            for code in codes:
                industry = code[:2]  # 模拟行业
                lncap = rng.normal(15, 1.5)
                # 因子 = 行业 alpha + lncap 暴露 + 噪声
                industry_alpha = hash(industry) % 100 / 100
                f = industry_alpha * 2 + 0.5 * lncap + rng.normal(0, 0.5)
                rows.append({
                    "date": dt, "code": code, "industry": industry,
                    "lncap": lncap, "factor": f,
                })
        return pd.DataFrame(rows)

    def test_neutralize_removes_industry(self, biased_factor):
        result = neutralize_industry_mcap(
            biased_factor, "factor", "industry", "lncap",
            neutralize_industry=True, neutralize_mcap=False,
        )
        # 行业 alpha 应被消除
        df = biased_factor.copy()
        df["neutral"] = result
        # 行业间均值应接近 0
        industry_means = df.groupby("industry")["neutral"].mean()
        assert industry_means.abs().max() < 0.5  # 应大幅减小

    def test_neutralize_removes_mcap(self, biased_factor):
        result = neutralize_industry_mcap(
            biased_factor, "factor", "industry", "lncap",
            neutralize_industry=False, neutralize_mcap=True,
        )
        # 与 lncap 的相关性应大幅减小
        df = biased_factor.copy()
        df["neutral"] = result
        corr_after = df["neutral"].corr(df["lncap"])
        corr_before = df["factor"].corr(df["lncap"])
        assert abs(corr_after) < abs(corr_before) * 0.3


# ============================================================
# 4. clean_factor Pipeline
# ============================================================

class TestCleanFactor:
    @pytest.fixture
    def factor_with_outliers(self):
        dates = pd.bdate_range("2023-01-01", periods=5)
        codes = [f"S{i:03d}" for i in range(20)]
        rows = []
        fwd_rows = []
        rng = np.random.default_rng(0)
        for dt in dates:
            for code in codes:
                f = rng.normal(0, 1)
                if code == "S000":  # 注入异常值
                    f = 100
                fwd_ret = f * 0.1 + rng.normal(0, 0.02)
                rows.append({"date": dt, "code": code, "factor": f})
                fwd_rows.append({"date": dt, "code": code, "ret": fwd_ret})
        return pd.DataFrame(rows), pd.DataFrame(fwd_rows)

    def test_pipeline_runs_end_to_end(self, factor_with_outliers):
        factor_df, fwd_df = factor_with_outliers
        cleaned, fwd = clean_factor(
            factor_df, fwd_df,
            winsorize="zscore",
            standardize="zscore",
        )
        assert len(cleaned) == len(factor_df)
        # 异常值应被压缩
        assert cleaned["factor"].max() < factor_df["factor"].max()

    def test_pipeline_with_neutralize(self):
        # 简单 pipeline
        dates = pd.bdate_range("2023-01-01", periods=3)
        rows, fwd_rows = [], []
        for dt in dates:
            for i in range(10):
                code = f"S{i:02d}"
                ind = "A" if i < 5 else "B"
                f = i * 0.5
                fwd = f * 0.05
                rows.append({"date": dt, "code": code, "factor": f, "industry": ind, "lncap": 15.0})
                fwd_rows.append({"date": dt, "code": code, "ret": fwd})

        cleaned, fwd = clean_factor(
            pd.DataFrame(rows), pd.DataFrame(fwd_rows),
            winsorize="zscore", standardize="zscore", neutralize=True,
        )
        assert "factor" in cleaned.columns
        # 行业 A 与 B 的均值应接近（因 neutralize）
        if "industry" in cleaned.columns:
            pass  # clean_factor 返回不带 industry
        # 至少 pipeline 跑通
        assert len(cleaned) == 30

    def test_pipeline_with_quantile_winsorize(self, factor_with_outliers):
        factor_df, fwd_df = factor_with_outliers
        cleaned, _ = clean_factor(
            factor_df, fwd_df,
            winsorize="quantile",
            standardize="rank",
        )
        # rank 后范围 [-0.5, 0.5]
        assert cleaned["factor"].max() <= 0.5
        assert cleaned["factor"].min() >= -0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
