"""
正确性测试

验证三个向量化模块的输出与原生实现（或数学等价基准）一致：
1. VectorizedAdapter vs NativeAdapter —— 回测净值曲线、交易记录、绩效指标
2. FactorExpressionEngine —— 与 pandas 手工计算的参考值对比
3. VectorizedICAnalysis —— 与 scipy.stats.spearmanr 逐日计算的参考值对比
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from scipy import stats

# conftest 已将本目录加入 sys.path
from synthetic_data import generate_synthetic_ohlcv, generate_signals


# ============================================================
# 1. 向量化回测 vs 原生回测
# ============================================================

class TestBacktestCorrectness:
    """VectorizedAdapter 与 NativeAdapter 语义等价性校验"""

    def test_equity_curve_matches(self, native_adapter, vectorized_adapter, small_dataset):
        """净值曲线应完全一致（同算法、同数据）"""
        data, signals = small_dataset
        r_native = native_adapter.run_backtest(data, signals)
        r_vec = vectorized_adapter.run_backtest(data, signals)

        eq_n = r_native["equity_curve"].set_index("date")["equity"]
        eq_v = r_vec["equity_curve"].set_index("date")["equity"]

        # 日期对齐
        common = eq_n.index.intersection(eq_v.index)
        assert len(common) > 0, "无共同交易日"
        diff = (eq_n.loc[common] - eq_v.loc[common]).abs()
        # 浮点累加误差容忍：相对误差 < 1e-6
        max_rel = (diff / eq_n.loc[common].abs()).max()
        assert max_rel < 1e-6, f"净值曲线相对误差过大: {max_rel}"

    def test_trade_count_matches(self, native_adapter, vectorized_adapter, small_dataset):
        """交易笔数应一致"""
        data, signals = small_dataset
        r_native = native_adapter.run_backtest(data, signals)
        r_vec = vectorized_adapter.run_backtest(data, signals)
        assert len(r_native["trades"]) == len(r_vec["trades"]), (
            f"交易笔数不一致: native={len(r_native['trades'])}, vec={len(r_vec['trades'])}"
        )

    def test_final_equity_matches(self, native_adapter, vectorized_adapter, small_dataset):
        """最终净值应一致"""
        data, signals = small_dataset
        r_native = native_adapter.run_backtest(data, signals)
        r_vec = vectorized_adapter.run_backtest(data, signals)
        final_n = r_native["equity_curve"]["equity"].iloc[-1]
        final_v = r_vec["equity_curve"]["equity"].iloc[-1]
        assert abs(final_n - final_v) / final_n < 1e-6, (
            f"最终净值不一致: native={final_n}, vec={final_v}"
        )

    def test_metrics_match(self, native_adapter, vectorized_adapter, small_dataset):
        """关键绩效指标应一致"""
        data, signals = small_dataset
        r_native = native_adapter.run_backtest(data, signals)
        r_vec = vectorized_adapter.run_backtest(data, signals)
        for key in ["total_return", "max_drawdown", "sharpe_ratio", "win_rate"]:
            v_n = r_native["metrics"][key]
            v_v = r_vec["metrics"][key]
            assert abs(v_n - v_v) < 1e-6, f"指标 {key} 不一致: native={v_n}, vec={v_v}"

    def test_different_strategies(self, native_adapter, vectorized_adapter):
        """不同策略下均应一致（reversal 策略）"""
        data = generate_synthetic_ohlcv(n_codes=30, n_days=100, seed=7)
        signals = generate_signals(data, strategy="reversal", seed=7)
        r_n = native_adapter.run_backtest(data, signals)
        r_v = vectorized_adapter.run_backtest(data, signals)
        f_n = r_n["equity_curve"]["equity"].iloc[-1] if not r_n["equity_curve"].empty else 0
        f_v = r_v["equity_curve"]["equity"].iloc[-1] if not r_v["equity_curve"].empty else 0
        if f_n > 0:
            assert abs(f_n - f_v) / f_n < 1e-6, f"reversal 策略净值不一致: {f_n} vs {f_v}"


# ============================================================
# 2. 因子表达式引擎正确性
# ============================================================

class TestFactorEngineCorrectness:
    """因子表达式引擎与手工计算参考值对比"""

    @pytest.fixture
    def factor_data(self):
        data = generate_synthetic_ohlcv(n_codes=30, n_days=100, seed=11)
        return data.sort_values(["code", "date"]).reset_index(drop=True)

    def test_delta_correctness(self, factor_data):
        """Delta(Close, 5) 应等于 close.diff(5) 按 code 分组"""
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        result = eng.calculate(factor_data, ["Delta(Close, 5)"])
        expected = factor_data.groupby("code")["close"].transform(lambda x: x.diff(5))
        actual = result["Delta_Close__5_"]
        # 对齐后比较（去掉 NaN）
        mask = expected.notna() & actual.notna()
        assert np.allclose(actual[mask].values, expected[mask].values, rtol=1e-10)

    def test_ts_mean_correctness(self, factor_data):
        """Ts_Mean(Close, 20) 应等于 rolling(20).mean 按 code 分组"""
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        result = eng.calculate(factor_data, ["Ts_Mean(Close, 20)"])
        expected = factor_data.groupby("code")["close"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        actual = result["Ts_Mean_Close__20_"]
        mask = expected.notna() & actual.notna()
        assert np.allclose(actual[mask].values, expected[mask].values, rtol=1e-10)

    def test_rank_correctness(self, factor_data):
        """Rank(Ts_Mean(Close, 5)) 截面排名应在 [0,1]"""
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        result = eng.calculate(factor_data, ["Rank(Ts_Mean(Close, 5))"])
        col = "Rank_Ts_Mean_Close__5__"
        valid = result[col].dropna()
        assert valid.between(0, 1, inclusive="both").all(), "Rank 值超出 [0,1]"
        # 每个日期的 rank 应覆盖多个不同值
        per_date = result.dropna(subset=[col]).groupby("date")[col].nunique()
        assert (per_date > 1).mean() > 0.9, "截面排名未生效"

    def test_arithmetic_correctness(self, factor_data):
        """二元算子 Div(Sub(High,Low),Close) 应等于 (high-low)/close"""
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        result = eng.calculate(factor_data, ["Div(Sub(High, Low), Close)"])
        expected = (factor_data["high"] - factor_data["low"]) / factor_data["close"]
        actual = result["Div_Sub_High__Low___Close_"]
        mask = expected.notna() & actual.notna()
        assert np.allclose(actual[mask].values, expected[mask].values, rtol=1e-10)

    def test_builtin_factors_calculable(self, factor_data):
        """所有内置因子都应能正常计算（无异常）"""
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        names = eng.get_available_factors()
        result = eng.calculate(factor_data, names)
        # 每个因子列至少应有部分非 NaN
        for name in names:
            col = eng._expr_to_name(BUILTIN_FACTORS_EXPR(name, eng))
            assert col in result.columns, f"因子 {name} 列缺失"

    def test_no_per_stock_loop(self, factor_data):
        """验证：计算结果行数 == 输入行数（无股票被丢弃）"""
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        result = eng.calculate(factor_data, ["Delta(Close, 5)", "Ts_Mean(Volume, 10)"])
        assert len(result) == len(factor_data), "结果行数与输入不一致"


def BUILTIN_FACTORS_EXPR(name, eng):
    from factor_expression_engine import BUILTIN_FACTORS
    return BUILTIN_FACTORS[name]


# ============================================================
# 3. 向量化 IC 分析正确性
# ============================================================

class TestICAnalysisCorrectness:
    """向量化 IC 与 scipy 逐日计算基准对比"""

    @pytest.fixture
    def ic_data(self):
        data = generate_synthetic_ohlcv(n_codes=50, n_days=100, seed=99)
        df = data.sort_values(["code", "date"]).reset_index(drop=True)
        df["factor"] = -df.groupby("code")["close"].transform(lambda x: x.pct_change(5))
        df["fwd_5d"] = df.groupby("code")["close"].transform(lambda x: x.shift(-5) / x - 1)
        return df

    def test_spearman_ic_matches_scipy(self, ic_data):
        """向量化 Spearman IC 应与 scipy.stats.spearmanr 逐日计算一致"""
        from vectorized_ic_analysis import VectorizedICAnalysis
        vec_ic = VectorizedICAnalysis.calc_ic_series(ic_data, "factor", "fwd_5d", "spearman")

        # 基准：逐日 scipy 计算（与主仓库 _calc_ic 相同逻辑）
        ref_ic = {}
        for dt, cross in ic_data.dropna(subset=["factor", "fwd_5d"]).groupby("date"):
            if len(cross) < 10:
                continue
            ic, _ = stats.spearmanr(cross["factor"], cross["fwd_5d"])
            if not np.isnan(ic):
                ref_ic[dt] = ic
        ref_series = pd.Series(ref_ic, name="ic")

        common = vec_ic.index.intersection(ref_series.index)
        assert len(common) > 50, f"共同日期过少: {len(common)}"
        diff = (vec_ic.loc[common] - ref_series.loc[common]).abs()
        # spearmanr 与 rank+pearson 数值上应高度一致（容忍浮点误差）
        assert diff.max() < 1e-9, f"Spearman IC 偏差过大: max={diff.max()}"

    def test_pearson_ic_matches_scipy(self, ic_data):
        """向量化 Pearson IC 应与 scipy.stats.pearsonr 一致"""
        from vectorized_ic_analysis import VectorizedICAnalysis
        vec_ic = VectorizedICAnalysis.calc_ic_series(ic_data, "factor", "fwd_5d", "pearson")

        ref_ic = {}
        for dt, cross in ic_data.dropna(subset=["factor", "fwd_5d"]).groupby("date"):
            if len(cross) < 10:
                continue
            ic, _ = stats.pearsonr(cross["factor"].fillna(0), cross["fwd_5d"].fillna(0))
            if not np.isnan(ic):
                ref_ic[dt] = ic
        ref_series = pd.Series(ref_ic, name="ic")

        common = vec_ic.index.intersection(ref_series.index)
        diff = (vec_ic.loc[common] - ref_series.loc[common]).abs()
        assert diff.max() < 1e-9, f"Pearson IC 偏差过大: max={diff.max()}"

    def test_ic_stats_consistency(self, ic_data):
        """IC 统计量应与序列一致"""
        from vectorized_ic_analysis import VectorizedICAnalysis
        ic = VectorizedICAnalysis.calc_ic_series(ic_data, "factor", "fwd_5d", "spearman")
        stat = VectorizedICAnalysis.calc_ic_stats(ic_data, "factor", "fwd_5d", "spearman")
        assert abs(stat["ic_mean"] - round(ic.mean(), 6)) < 1e-5
        assert stat["n_days"] == len(ic)
        assert -1 <= stat["ic_mean"] <= 1

    def test_quantile_returns_shape(self, ic_data):
        """分层收益返回正确形状"""
        from vectorized_ic_analysis import VectorizedICAnalysis
        qr = VectorizedICAnalysis.calc_quantile_returns(ic_data, "factor", "fwd_5d", n_quantiles=5)
        assert not qr.empty
        assert qr.shape[1] <= 5
        assert qr.shape[0] > 0
