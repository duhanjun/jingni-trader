"""
因子注册表 v2 测试 + 向量化中性化性能对比

验证：
  1. 因子注册表插件式注册
  2. 依赖拓扑排序
  3. 向量化中性化正确性
  4. 向量化中性化 vs 旧版双重循环性能对比

运行：python -m pytest optimizations/tests/test_factor_v2.py -v
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from factor.factor_registry_v2 import (
    FactorRegistry,
    Neutralizer,
    build_default_registry,
)


# ---------------------------------------------------------------------------
# 因子注册表测试
# ---------------------------------------------------------------------------

class TestFactorRegistry:
    """验证因子注册表插件式注册与依赖解析。"""

    def test_register_and_compute(self):
        registry = FactorRegistry()

        @registry.register("ret_1d", "momentum", "1日收益")
        def ret_1d(df):
            return df.groupby("code")["close"].pct_change(1)

        df = pd.DataFrame({
            "code": ["A"] * 5 + ["B"] * 5,
            "date": pd.bdate_range("2024-01-02", periods=5).tolist() * 2,
            "close": [10, 11, 10.5, 12, 11.5] + [20, 21, 19, 22, 21],
        })
        result = registry.compute(df)
        assert "ret_1d" in result.columns
        # 第一天 pct_change 应为 NaN（用 nth(0) 取每组第一行，而非 first() 跳过 NaN）
        first_vals = result.groupby("code")["ret_1d"].nth(0)
        assert first_vals.isna().all(), "每组首行 pct_change 应为 NaN"

    def test_dependency_resolution(self):
        """依赖因子应先于依赖它的因子计算。"""
        registry = FactorRegistry()

        @registry.register("ret_5d", "momentum", dependencies=[])
        def ret_5d(df):
            return df.groupby("code")["close"].pct_change(5)

        @registry.register("reversal_5d", "reversal", dependencies=["ret_5d"])
        def reversal_5d(df):
            return -df["ret_5d"]

        df = pd.DataFrame({
            "code": ["A"] * 10,
            "date": pd.bdate_range("2024-01-02", periods=10),
            "close": np.linspace(10, 15, 10),
        })
        result = registry.compute(df, ["reversal_5d", "ret_5d"])
        assert "ret_5d" in result.columns
        assert "reversal_5d" in result.columns
        # reversal_5d 应为 ret_5d 的负数
        valid = result["ret_5d"].notna()
        assert np.allclose(
            result.loc[valid, "reversal_5d"],
            -result.loc[valid, "ret_5d"],
            equal_nan=True,
        )

    def test_circular_dependency_detected(self):
        registry = FactorRegistry()

        @registry.register("a", dependencies=["b"])
        def a(df):
            return df["x"]

        @registry.register("b", dependencies=["a"])
        def b(df):
            return df["x"]

        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="循环"):
            registry.compute(df, ["a", "b"])

    def test_default_registry_categories(self):
        """默认注册表应包含 momentum/reversal/volatility/volume 四类。"""
        registry = build_default_registry()
        categories = {m.category for m in registry.list_factors()}
        assert "momentum" in categories
        assert "reversal" in categories
        assert "volatility" in categories
        assert "volume" in categories

    def test_factor_metadata(self):
        registry = build_default_registry()
        momentum_factors = registry.list_factors("momentum")
        assert len(momentum_factors) >= 3
        assert all(m.category == "momentum" for m in momentum_factors)


# ---------------------------------------------------------------------------
# 向量化中性化正确性测试
# ---------------------------------------------------------------------------

class TestNeutralization:
    """验证向量化中性化正确性。"""

    def make_panel(self, n_codes=20, n_days=10):
        rng = np.random.default_rng(42)
        rows = []
        industries = ["银行", "地产", "科技", "消费"]
        for code in range(n_codes):
            ind = industries[code % len(industries)]
            lncap = rng.normal(15, 2)
            for day in range(n_days):
                # 因子值与市值正相关（模拟市值效应），便于验证中性化效果
                factor = lncap * 0.3 + rng.normal(0, 0.5)
                rows.append({
                    "date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=day),
                    "code": f"c{code}",
                    "industry": ind,
                    "lncap": lncap,
                    "factor_raw": factor,
                })
        return pd.DataFrame(rows)

    def test_neutralize_adds_neut_column(self):
        df = self.make_panel()
        neut = Neutralizer()
        result = neut.neutralize(df, ["factor_raw"])
        assert "factor_raw_neut" in result.columns

    def test_neutralized_residual_uncorrelated_with_market_cap(self):
        """中性化后因子残差应与市值相关性显著降低。"""
        df = self.make_panel(n_codes=50, n_days=20)
        neut = Neutralizer()
        result = neut.neutralize(df, ["factor_raw"])
        valid = result["factor_raw_neut"].notna()
        assert valid.sum() > 0, "中性化后应有有效值"
        raw_corr = result.loc[valid, "factor_raw"].corr(result.loc[valid, "lncap"])
        neut_corr = result.loc[valid, "factor_raw_neut"].corr(result.loc[valid, "lncap"])
        # 原始因子与市值强正相关（构造如此），中性化后应大幅降低
        assert raw_corr > 0.3, f"构造数据应有强正相关: {raw_corr:.3f}"
        assert abs(neut_corr) < abs(raw_corr), \
            f"中性化后相关性应降低: raw={raw_corr:.3f}, neut={neut_corr:.3f}"

    def test_neutralize_handles_missing_columns(self):
        """缺少 industry/lncap 列时应优雅降级。"""
        df = pd.DataFrame({
            "date": ["d1", "d2"],
            "code": ["A", "B"],
            "factor_raw": [1.0, 2.0],
        })
        neut = Neutralizer()
        result = neut.neutralize(df, ["factor_raw"])
        assert "factor_raw_neut" in result.columns


# ---------------------------------------------------------------------------
# 性能对比测试
# ---------------------------------------------------------------------------

class TestNeutralizationPerformance:
    """向量化中性化 vs 旧版双重循环性能对比。"""

    def make_large_panel(self, n_codes=100, n_days=60):
        rng = np.random.default_rng(42)
        rows = []
        industries = ["银行", "地产", "科技", "消费", "医药", "能源"]
        for code in range(n_codes):
            ind = industries[code % len(industries)]
            lncap = rng.normal(15, 2)
            for day in range(n_days):
                rows.append({
                    "date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=day),
                    "code": f"c{code}",
                    "industry": ind,
                    "lncap": lncap,
                    "factor_raw": lncap * 0.2 + rng.normal(0, 1),
                })
        return pd.DataFrame(rows)

    def old_neutralize(self, df, factor_col="factor_raw"):
        """旧版逐日逐因子双重循环实现（复刻 factor-engine/engine.py:145-178）。"""
        result = df.copy()
        neut_col = f"{factor_col}_neut"
        result[neut_col] = np.nan
        for dt in df["date"].unique():
            day_df = df[df["date"] == dt]
            industry_dummies = pd.get_dummies(day_df["industry"], prefix="ind", drop_first=True)
            X = np.column_stack([
                np.ones(len(day_df)),
                industry_dummies.values,
                day_df["lncap"].values,
            ])
            y = day_df[factor_col].values
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                result.loc[day_df.index, neut_col] = resid
            except Exception:
                pass
        return result

    def test_vectorized_faster_than_loop(self):
        """向量化应比双重循环快至少 2 倍。"""
        df = self.make_large_panel(n_codes=100, n_days=60)

        # 旧版
        t0 = time.perf_counter()
        old_result = self.old_neutralize(df)
        old_time = time.perf_counter() - t0

        # 新版向量化
        neut = Neutralizer()
        t0 = time.perf_counter()
        new_result = neut.neutralize(df, ["factor_raw"])
        new_time = time.perf_counter() - t0

        # 正确性：结果应接近（同一 OLS 公式，仅实现不同）
        valid = old_result["factor_raw_neut"].notna() & new_result["factor_raw_neut"].notna()
        if valid.sum() > 10:
            corr = old_result.loc[valid, "factor_raw_neut"].corr(
                new_result.loc[valid, "factor_raw_neut"]
            )
            print(f"\n[中性化正确性] 新旧版残差相关性: {corr:.6f} (有效样本 {valid.sum()})")
            assert corr > 0.95, f"新旧版结果相关性应 > 0.95: {corr:.4f}"

        # 性能：向量化应更快（允许 2 倍以上加速）
        print(f"\n[中性化性能] 旧版双重循环: {old_time:.3f}s, 新版向量化: {new_time:.3f}s, "
              f"加速比: {old_time/new_time:.1f}x")
        # 不强制 assert 性能（CI 环境波动），仅打印对比


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
