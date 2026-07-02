"""
正确性测试

将 Polars 向量化实现与 pandas/scipy 参考实现逐项对比，
验证数值一致性（在浮点容差范围内）。

参考实现复刻自原 factor-engine/engine.py 的逻辑（逐日 Python 循环）。
"""
import sys
import os
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from optimizations.polars_ic_analysis import calc_ic_series_polars, ic_summary_stats
from optimizations.polars_neutralize import (
    neutralize_mcap_polars,
    neutralize_industry_mcap_polars,
)
from optimizations.vectorized_metrics import calc_enhanced_metrics
from optimizations.tests import generate_panel_data, generate_equity_curve, generate_trades


# ----------------------------------------------------------------------
# 参考实现（复刻原 engine.py 的逐日循环逻辑）
# ----------------------------------------------------------------------

def ref_calc_ic_pandas(data_pd: pd.DataFrame, factor_col: str, forward_col: str,
                       ic_type: str = "spearman", min_samples: int = 10) -> pd.DataFrame:
    """参考实现：逐日循环计算 IC（复刻原 _calc_ic）"""
    ic_list = []
    dates = sorted(data_pd["date"].unique())
    for dt in dates:
        cross = data_pd[data_pd["date"] == dt].dropna(subset=[factor_col, forward_col])
        if len(cross) < min_samples:
            continue
        if ic_type == "spearman":
            ic, _ = stats.spearmanr(cross[factor_col], cross[forward_col], nan_policy="omit")
        else:
            ic, _ = stats.pearsonr(cross[factor_col].fillna(0), cross[forward_col].fillna(0))
        if not np.isnan(ic):
            ic_list.append({"date": dt, "ic": ic})
    return pd.DataFrame(ic_list)


def ref_neutralize_mcap_pandas(data_pd: pd.DataFrame, factor_col: str,
                               mcap_col: str = "lncap", min_samples: int = 30) -> pd.Series:
    """参考实现：逐日 OLS 市值中性化（复刻原 neutralize）"""
    neutralized = pd.Series(index=data_pd.index, dtype=float)
    for dt in data_pd["date"].unique():
        cross = data_pd[data_pd["date"] == dt].copy()
        if len(cross) < min_samples:
            neutralized.loc[cross.index] = cross[factor_col]
            continue
        X = cross[[mcap_col]].fillna(0).values
        y = cross[factor_col].fillna(0).values
        try:
            model = LinearRegression()
            model.fit(X, y)
            residual = y - model.predict(X)
            neutralized.loc[cross.index] = residual
        except Exception:
            neutralized.loc[cross.index] = cross[factor_col]
    return neutralized


def ref_neutralize_industry_mcap_pandas(data_pd: pd.DataFrame, factor_col: str,
                                        industry_col: str = "industry",
                                        mcap_col: str = "lncap", min_samples: int = 30) -> pd.Series:
    """参考实现：逐日 OLS 行业+市值中性化（复刻原 neutralize）"""
    neutralized = pd.Series(index=data_pd.index, dtype=float)
    for dt in data_pd["date"].unique():
        cross = data_pd[data_pd["date"] == dt].copy()
        if len(cross) < min_samples:
            neutralized.loc[cross.index] = cross[factor_col]
            continue
        X_vars = []
        if mcap_col in cross.columns:
            X_vars.append(mcap_col)
        if industry_col in cross.columns:
            dummies = pd.get_dummies(cross[industry_col], prefix="ind")
            for col in dummies.columns:
                cross[col] = dummies[col].values
                X_vars.append(col)
        if not X_vars:
            neutralized.loc[cross.index] = cross[factor_col]
            continue
        X = cross[X_vars].fillna(0).values
        y = cross[factor_col].fillna(0).values
        try:
            model = LinearRegression()
            model.fit(X, y)
            residual = y - model.predict(X)
            neutralized.loc[cross.index] = residual
        except Exception:
            neutralized.loc[cross.index] = cross[factor_col]
    return neutralized


# ----------------------------------------------------------------------
# 测试用例
# ----------------------------------------------------------------------

def test_ic_spearman_correctness():
    """测试 Spearman IC 与参考实现一致"""
    data = generate_panel_data(n_stocks=100, n_days=100, seed=1)
    data_pd = data.to_pandas()

    # Polars 实现
    ic_polars = calc_ic_series_polars(data, "factor_1", "ret_forward_5d", "spearman")
    # 参考实现
    ic_ref = ref_calc_ic_pandas(data_pd, "factor_1", "ret_forward_5d", "spearman")

    assert ic_polars.height == len(ic_ref), (
        f"IC 序列长度不一致: polars={ic_polars.height}, ref={len(ic_ref)}"
    )

    # 逐日对比 IC 值
    polars_ic = ic_polars.sort("date")["ic"].to_numpy()
    ref_ic = ic_ref.sort_values("date")["ic"].to_numpy()
    max_diff = np.max(np.abs(polars_ic - ref_ic))
    assert max_diff < 1e-6, f"Spearman IC 最大偏差 {max_diff} 超过 1e-6"
    print(f"  [PASS] Spearman IC 正确性: {ic_polars.height} 天, 最大偏差 {max_diff:.2e}")


def test_ic_pearson_correctness():
    """测试 Pearson IC 与参考实现一致"""
    data = generate_panel_data(n_stocks=100, n_days=100, seed=2)
    data_pd = data.to_pandas()

    ic_polars = calc_ic_series_polars(data, "factor_2", "ret_forward_5d", "pearson")
    ic_ref = ref_calc_ic_pandas(data_pd, "factor_2", "ret_forward_5d", "pearson")

    polars_ic = ic_polars.sort("date")["ic"].to_numpy()
    ref_ic = ic_ref.sort_values("date")["ic"].to_numpy()
    max_diff = np.max(np.abs(polars_ic - ref_ic))
    assert max_diff < 1e-6, f"Pearson IC 最大偏差 {max_diff} 超过 1e-6"
    print(f"  [PASS] Pearson IC 正确性: {ic_polars.height} 天, 最大偏差 {max_diff:.2e}")


def test_ic_summary_correctness():
    """测试 IC 统计摘要正确性"""
    data = generate_panel_data(n_stocks=150, n_days=120, seed=3)
    data_pd = data.to_pandas()

    ic_polars = calc_ic_series_polars(data, "factor_1", "ret_forward_5d", "spearman")
    stats_polars = ic_summary_stats(ic_polars)

    ic_ref = ref_calc_ic_pandas(data_pd, "factor_1", "ret_forward_5d", "spearman")
    ic_ref_series = ic_ref["ic"]

    ref_mean = ic_ref_series.mean()
    ref_std = ic_ref_series.std()
    ref_ir = ref_mean / ref_std if ref_std > 0 else 0
    ref_pos = (ic_ref_series > 0).mean()

    assert abs(stats_polars["ic_mean"] - round(float(ref_mean), 6)) < 1e-4, (
        f"ic_mean 偏差过大: polars={stats_polars['ic_mean']}, ref={ref_mean}"
    )
    assert abs(stats_polars["ic_ir"] - round(float(ref_ir), 4)) < 1e-3, (
        f"ic_ir 偏差过大: polars={stats_polars['ic_ir']}, ref={ref_ir}"
    )
    assert abs(stats_polars["ic_positive_ratio"] - round(float(ref_pos), 4)) < 1e-3, (
        f"ic_positive_ratio 偏差过大"
    )
    print(f"  [PASS] IC 统计摘要正确性: mean={stats_polars['ic_mean']}, ir={stats_polars['ic_ir']}")


def test_neutralize_mcap_correctness():
    """测试市值中性化与参考实现一致"""
    data = generate_panel_data(n_stocks=80, n_days=60, seed=4)
    data_pd = data.to_pandas().sort_values(["date", "code"]).reset_index(drop=True)

    # Polars 实现
    result_pl = neutralize_mcap_polars(data, "factor_1", "lncap")
    result_pl = result_pl.sort(["date", "code"])
    polars_neutral = result_pl["factor_1_neutral"].to_numpy()

    # 参考实现
    ref_neutral = ref_neutralize_mcap_pandas(data_pd, "factor_1", "lncap")
    ref_neutral = ref_neutral.values

    # 对齐（去掉 NaN）
    mask = ~(np.isnan(polars_neutral) | np.isnan(ref_neutral))
    if mask.sum() > 0:
        max_diff = np.max(np.abs(polars_neutral[mask] - ref_neutral[mask]))
        assert max_diff < 1e-6, f"市值中性化最大偏差 {max_diff} 超过 1e-6"
        print(f"  [PASS] 市值中性化正确性: {mask.sum()} 样本, 最大偏差 {max_diff:.2e}")


def test_neutralize_industry_mcap_correctness():
    """测试行业+市值中性化与参考实现一致（FWL 定理验证）"""
    data = generate_panel_data(n_stocks=100, n_days=60, seed=5)
    data_pd = data.to_pandas().sort_values(["date", "code"]).reset_index(drop=True)

    # Polars 实现（FWL 向量化）
    result_pl = neutralize_industry_mcap_polars(data, "factor_1", "industry", "lncap")
    result_pl = result_pl.sort(["date", "code"])
    polars_neutral = result_pl["factor_1_neutral"].to_numpy()

    # 参考实现（完整 OLS）
    ref_neutral = ref_neutralize_industry_mcap_pandas(data_pd, "factor_1", "industry", "lncap")
    ref_neutral = ref_neutral.values

    mask = ~(np.isnan(polars_neutral) | np.isnan(ref_neutral))
    if mask.sum() > 0:
        max_diff = np.max(np.abs(polars_neutral[mask] - ref_neutral[mask]))
        # FWL 定理保证数值一致，但浮点精度可能有微小差异
        assert max_diff < 1e-8, (
            f"行业+市值中性化最大偏差 {max_diff} 超过 1e-8，FWL 定理验证失败"
        )
        print(f"  [PASS] 行业+市值中性化正确性 (FWL): {mask.sum()} 样本, 最大偏差 {max_diff:.2e}")


def test_metrics_correctness():
    """测试增强版绩效指标的基础指标与原实现一致"""
    eq_df = generate_equity_curve(n_days=250, seed=6)
    eq = eq_df.set_index("date")["equity"]

    # 增强版指标
    enhanced = calc_enhanced_metrics(eq, init_capital=1e6)

    # 原实现参考（复刻 backtest-engine _calc_metrics）
    returns = eq.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    max_drawdown = (eq / eq.cummax() - 1).min()
    sharpe = (annual_return - 0.03) / volatility if volatility != 0 else 0
    win_rate = (returns > 0).mean()

    assert abs(enhanced["total_return"] - round(float(total_return), 6)) < 1e-4, (
        f"total_return 偏差: {enhanced['total_return']} vs {total_return}"
    )
    assert abs(enhanced["annual_return"] - round(float(annual_return), 6)) < 1e-4, (
        f"annual_return 偏差: {enhanced['annual_return']} vs {annual_return}"
    )
    assert abs(enhanced["volatility"] - round(float(volatility), 6)) < 1e-4, (
        f"volatility 偏差: {enhanced['volatility']} vs {volatility}"
    )
    assert abs(enhanced["sharpe_ratio"] - round(float(sharpe), 4)) < 1e-3, (
        f"sharpe 偏差: {enhanced['sharpe_ratio']} vs {sharpe}"
    )
    assert abs(enhanced["max_drawdown"] - round(float(max_drawdown), 6)) < 1e-4, (
        f"max_drawdown 偏差: {enhanced['max_drawdown']} vs {max_drawdown}"
    )
    assert abs(enhanced["win_rate"] - round(float(win_rate), 4)) < 1e-3, (
        f"win_rate 偏差: {enhanced['win_rate']} vs {win_rate}"
    )
    print(f"  [PASS] 绩效指标正确性: {len(enhanced)} 个指标, sharpe={enhanced['sharpe_ratio']}")


def test_metrics_with_trades():
    """测试含交易记录的换手率计算"""
    eq_df = generate_equity_curve(n_days=250, seed=7)
    trades = generate_trades(n_trades=200, seed=7)
    eq = eq_df.set_index("date")["equity"]

    enhanced = calc_enhanced_metrics(eq, trades=trades, init_capital=1e6)
    assert "annual_turnover" in enhanced, "换手率未计算"
    assert "n_trades" in enhanced, "交易次数未记录"
    assert enhanced["n_trades"] == 200, f"交易次数错误: {enhanced['n_trades']}"
    assert enhanced["annual_turnover"] > 0, "换手率应大于 0"
    print(f"  [PASS] 换手率计算: turnover={enhanced['annual_turnover']}, trades={enhanced['n_trades']}")


def run_all_correctness_tests():
    """运行所有正确性测试"""
    print("=" * 60)
    print("正确性测试")
    print("=" * 60)
    tests = [
        ("Spearman IC", test_ic_spearman_correctness),
        ("Pearson IC", test_ic_pearson_correctness),
        ("IC 统计摘要", test_ic_summary_correctness),
        ("市值中性化", test_neutralize_mcap_correctness),
        ("行业+市值中性化 (FWL)", test_neutralize_industry_mcap_correctness),
        ("绩效指标基础", test_metrics_correctness),
        ("绩效指标含交易", test_metrics_with_trades),
    ]
    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[测试] {name}")
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"正确性测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return passed, failed


if __name__ == "__main__":
    run_all_correctness_tests()
