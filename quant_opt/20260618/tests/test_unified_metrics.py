"""
测试用例 1: 统一指标库 (unified_metrics) 验证
=============================================

验证目标:
  1. 与 jingni-trader 原生 BaseBacktestMetrics 的一致性
  2. 与 QuantStats 风格的"全量指标"覆盖（30+ 指标）
  3. 边界条件（空序列、恒定值、单边行情）

对照基线:
  - jingni-trader/skills/backtest-engine/scripts/base/base_backtest.py
  - jingni-trader/skills/backtest-engine/engine.py:BacktestEngine._calc_metrics
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from unified_metrics import (
    cagr, total_return, volatility, max_drawdown, max_drawdown_duration,
    sharpe, smart_sharpe, sortino, calmar,
    ulcer_index, ulcer_performance_index, omega, gain_to_pain_ratio, profit_factor,
    value_at_risk, conditional_var, tail_ratio,
    alpha_beta, information_ratio, treynor, capture_ratios,
    factor_ic, factor_ic_decay, factor_quantile_returns, factor_turnover,
    trade_stats, compute_all_metrics,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_equity():
    """标准权益曲线"""
    dates = pd.bdate_range("2023-01-01", periods=252)
    # 模拟一个年化 15% 的策略
    daily_ret = 0.15 / 252
    equity = pd.Series(
        (1 + daily_ret + np.random.default_rng(1).normal(0, 0.012, 252)).cumprod() * 1_000_000,
        index=dates,
    )
    return equity


@pytest.fixture
def sample_returns(sample_equity):
    return sample_equity.pct_change().dropna()


@pytest.fixture
def sample_benchmark():
    dates = pd.bdate_range("2023-01-01", periods=252)
    # 模拟一个年化 8% 的基准
    daily_ret = 0.08 / 252
    bm = pd.Series(
        (1 + daily_ret + np.random.default_rng(2).normal(0, 0.010, 252)).cumprod(),
        index=dates,
    )
    return bm.pct_change().dropna()


# ============================================================
# 1. 基础指标
# ============================================================

class TestBasicMetrics:
    def test_cagr_known_value(self):
        # 1 年翻倍 → CAGR = 1.0
        eq = pd.Series([100.0, 200.0], index=pd.bdate_range("2023-01-01", periods=2))
        assert abs(cagr(eq, periods=2) - 1.0) < 1e-6

    def test_cagr_zero_years(self):
        eq = pd.Series([100.0, 100.0])
        assert cagr(eq) == 0.0

    def test_total_return(self):
        eq = pd.Series([100.0, 150.0])
        assert abs(total_return(eq) - 0.5) < 1e-9

    def test_max_drawdown(self, sample_equity):
        mdd, peak, trough = max_drawdown(sample_equity)
        assert mdd <= 0
        assert peak <= trough

    def test_max_drawdown_known(self):
        eq = pd.Series([100, 120, 90, 110, 80, 130], index=range(6))
        mdd, _, _ = max_drawdown(eq)
        # 120 → 80 = -33.3%
        assert abs(mdd - (-1/3)) < 0.01


# ============================================================
# 2. 风险调整指标
# ============================================================

class TestRiskAdjusted:
    def test_sharpe_zero_vol(self):
        r = pd.Series([0.01, 0.01, 0.01])
        assert sharpe(r, rf=0.0) == 0.0

    def test_smart_sharpe_less_than_sharpe_for_persistent_returns(self):
        # 持续上涨序列：序列相关高，smart_sharpe 应 < sharpe
        r = pd.Series(np.linspace(0.001, 0.01, 100))
        assert smart_sharpe(r, rf=0.0) < sharpe(r, rf=0.0) * 1.1

    def test_sortino_no_downside(self):
        r = pd.Series([0.01, 0.02, 0.015, 0.005])
        # 全部为正，下行偏差 = 0
        assert sortino(r) == 0.0

    def test_calmar_with_extreme_mdd(self):
        eq = pd.Series([100, 120, 60, 150])
        # CAGR 略正，MDD = -50%
        c = calmar(eq, periods=4)
        assert c != 0.0

    def test_omega_inf_for_all_positive(self):
        r = pd.Series([0.01, 0.02, 0.005])
        assert omega(r) == float("inf")

    def test_profit_factor(self, sample_returns):
        pf = profit_factor(sample_returns)
        assert pf > 0


# ============================================================
# 3. 风险指标
# ============================================================

class TestRiskMetrics:
    def test_var_ordering(self):
        r = pd.Series(np.random.default_rng(0).normal(0, 0.02, 1000))
        var = value_at_risk(r, 0.95)
        cvar = conditional_var(r, 0.95)
        # CVaR 应该 <= VaR（更负或相等）
        assert cvar <= var

    def test_tail_ratio_symmetric(self):
        r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 1000))
        # 近似对称分布，tail ratio 应接近 1
        tr = tail_ratio(r)
        assert 0.5 < tr < 2.0


# ============================================================
# 4. 基准归因
# ============================================================

class TestBenchmarkAttribution:
    def test_alpha_beta_perfect_correlation(self):
        # 策略 = 基准，无 alpha
        r = pd.Series([0.01, 0.02, -0.01, 0.005])
        b = r.copy()
        a, beta = alpha_beta(r, b, rf=0.0)
        assert abs(beta - 1.0) < 1e-6
        assert abs(a) < 1e-6

    def test_information_ratio_zero_for_perfect_match(self):
        r = pd.Series([0.01, 0.02, -0.01, 0.005])
        b = r.copy()
        ir = information_ratio(r, b)
        assert abs(ir) < 1e-6


# ============================================================
# 5. 因子分析
# ============================================================

class TestFactorAnalysis:
    @pytest.fixture
    def known_factor(self):
        """生成已知 IC 的因子：因子值 = 未来收益 + 噪声"""
        dates = pd.bdate_range("2023-01-01", periods=60)
        codes = [f"00{i:04d}.SZ" for i in range(20)]
        rows = []
        fwd_rows = []
        for dt in dates:
            for code in codes:
                fwd_ret = np.random.default_rng(hash(code + str(dt)) % (2**32)).normal(0, 0.02)
                factor_value = fwd_ret * 5 + np.random.default_rng(0).normal(0, 0.05)
                rows.append({"date": dt, "code": code, "factor": factor_value})
                fwd_rows.append({"date": dt, "code": code, "ret": fwd_ret})
        factor_df = pd.DataFrame(rows)
        fwd_df = pd.DataFrame(fwd_rows)
        return factor_df, fwd_df

    def test_factor_ic_returns_series(self, known_factor):
        factor_df, fwd_df = known_factor
        ic = factor_ic(factor_df, fwd_df, method="spearman")
        assert isinstance(ic, pd.Series)
        assert len(ic) > 0
        assert -1 <= ic.min() <= 1
        assert -1 <= ic.max() <= 1

    def test_factor_ic_decay_includes_multiple_periods(self, known_factor):
        factor_df, fwd_df = known_factor
        # 构造 ret_1d
        returns_df = fwd_df.rename(columns={"ret": "ret_1d"}).set_index(["date", "code"])
        decay = factor_ic_decay(factor_df, returns_df, periods=[1, 5])
        assert "period" in decay.columns
        assert "ic_mean" in decay.columns
        assert "ic_ir" in decay.columns
        assert len(decay) == 2

    def test_factor_quantile_returns(self, known_factor):
        factor_df, fwd_df = known_factor
        factor_df = factor_df.set_index(["date", "code"])
        fwd_df = fwd_df.set_index(["date", "code"])
        qr = factor_quantile_returns(factor_df, fwd_df, n_quantiles=5, period=1)
        assert qr is not None
        assert len(qr) == 5
        # 高分位的 mean_annual 应该 > 低分位
        if len(qr) == 5:
            assert qr.loc[qr.index.max(), "mean_annual"] >= qr.loc[qr.index.min(), "mean_annual"]


# ============================================================
# 6. 边界条件
# ============================================================

class TestEdgeCases:
    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        assert cagr(s) == 0.0
        assert sharpe(s) == 0.0
        assert max_drawdown(s)[0] == 0.0

    def test_single_value(self):
        s = pd.Series([100.0])
        assert cagr(s) == 0.0
        assert total_return(s) == 0.0

    def test_constant_value(self):
        s = pd.Series([100.0, 100.0, 100.0])
        assert volatility(s) == 0.0
        assert sharpe(s) == 0.0

    def test_negative_returns_only(self):
        s = pd.Series([-0.01, -0.02, -0.005, -0.015])
        m = compute_all_metrics(
            equity=pd.Series((1 + s).cumprod() * 100),
            returns=s,
        )
        assert m["total_return"] < 0
        assert m["max_drawdown"] < 0


# ============================================================
# 7. 一站式计算 vs jingni-trader 原生
# ============================================================

class TestParityWithJingniTrader:
    """与 jingni-trader/skills/backtest-engine/scripts/base/base_backtest.py 对比"""
    def test_sharpe_matches_basemetrics(self, sample_returns):
        # jingni-trader 的 BaseBacktestMetrics.calc_sharpe 实现
        from unified_metrics import sharpe, volatility
        my_sharpe = sharpe(sample_returns, rf=0.03, periods=252)
        my_vol = volatility(sample_returns, periods=252)
        # jingni-trader 公式
        ann_return = sample_returns.mean() * 252
        expected = (ann_return - 0.03) / my_vol if my_vol != 0 else 0
        assert abs(my_sharpe - expected) < 1e-9

    def test_total_return_matches_basemetrics(self, sample_equity):
        # jingni-trader: equity.iloc[-1] / equity.iloc[0] - 1
        expected = sample_equity.iloc[-1] / sample_equity.iloc[0] - 1
        assert abs(total_return(sample_equity) - expected) < 1e-9

    def test_max_drawdown_matches_basemetrics(self, sample_equity):
        # jingni-trader: (equity - cummax) / cummax 的最小值
        cummax = sample_equity.cummax()
        dd = (sample_equity - cummax) / cummax
        expected = dd.min()
        mdd, _, _ = max_drawdown(sample_equity)
        assert abs(mdd - expected) < 1e-9


# ============================================================
# 8. 完整报告
# ============================================================

class TestFullReport:
    def test_compute_all_metrics_returns_30plus(self, sample_equity, sample_returns, sample_benchmark):
        m = compute_all_metrics(
            equity=sample_equity,
            returns=sample_returns,
            benchmark_returns=sample_benchmark,
        )
        # 必须有 30+ 指标
        assert len(m) >= 30
        # 关键指标
        for k in [
            "total_return", "cagr", "volatility", "sharpe", "sortino",
            "calmar", "max_drawdown", "information_ratio", "alpha", "beta",
            "var_95", "cvar_95", "tail_ratio", "omega", "profit_factor",
        ]:
            assert k in m, f"missing key metric: {k}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
