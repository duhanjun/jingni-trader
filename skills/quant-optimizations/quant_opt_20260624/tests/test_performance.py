"""
性能对比测试

对比向量化实现 vs 原生实现的执行时间，量化加速比。
"""
import time
import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_ohlcv, generate_signals


def _timed(fn, *args, **kwargs):
    """计时执行，返回 (result, elapsed_seconds)"""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ============================================================
# 1. 回测引擎性能对比
# ============================================================

class TestBacktestPerformance:
    """VectorizedAdapter vs NativeAdapter 性能"""

    @pytest.mark.parametrize("n_codes,n_days", [(50, 250), (200, 500)])
    def test_backtest_speedup(self, native_adapter, vectorized_adapter, n_codes, n_days):
        data = generate_synthetic_ohlcv(n_codes=n_codes, n_days=n_days, seed=2024)
        signals = generate_signals(data, strategy="ma_cross", seed=2024)

        # 各跑 2 次取较快值，减少抖动
        _, t_n1 = _timed(native_adapter.run_backtest, data, signals)
        _, t_n2 = _timed(native_adapter.run_backtest, data, signals)
        t_native = min(t_n1, t_n2)

        _, t_v1 = _timed(vectorized_adapter.run_backtest, data, signals)
        _, t_v2 = _timed(vectorized_adapter.run_backtest, data, signals)
        t_vec = min(t_v1, t_v2)

        speedup = t_native / t_vec if t_vec > 0 else float("inf")
        print(f"\n[回测 {n_codes}股×{n_days}日] native={t_native*1000:.1f}ms  vec={t_vec*1000:.1f}ms  加速={speedup:.2f}x")
        # 向量化应不慢于原生（至少 1.0x，通常显著更快）
        assert speedup >= 1.0, f"向量化未带来加速: {speedup:.2f}x"


# ============================================================
# 2. 因子引擎性能对比
# ============================================================

class TestFactorEnginePerformance:
    """因子表达式引擎 vs 逐股票循环基准"""

    @pytest.mark.parametrize("n_codes,n_days", [(100, 250), (300, 500)])
    def test_factor_calc_speedup(self, n_codes, n_days):
        from factor_expression_engine import FactorExpressionEngine

        data = generate_synthetic_ohlcv(n_codes=n_codes, n_days=n_days, seed=55)
        eng = FactorExpressionEngine()

        # 基准：逐股票 Python 循环计算 Delta(Close,5) + Ts_Mean(Volume,20)
        def baseline_loop():
            df = data.sort_values(["code", "date"]).copy()
            df["f1"] = np.nan
            df["f2"] = np.nan
            for code in df["code"].unique():
                mask = df["code"] == code
                df.loc[mask, "f1"] = df.loc[mask, "close"].diff(5)
                df.loc[mask, "f2"] = df.loc[mask, "volume"].rolling(20, min_periods=10).mean()
            return df

        _, t_base1 = _timed(baseline_loop)
        _, t_base2 = _timed(baseline_loop)
        t_base = min(t_base1, t_base2)

        _, t_v1 = _timed(eng.calculate, data, ["Delta(Close, 5)", "Ts_Mean(Volume, 20)"])
        _, t_v2 = _timed(eng.calculate, data, ["Delta(Close, 5)", "Ts_Mean(Volume, 20)"])
        t_vec = min(t_v1, t_v2)

        speedup = t_base / t_vec if t_vec > 0 else float("inf")
        print(f"\n[因子 {n_codes}股×{n_days}日] 逐股循环={t_base*1000:.1f}ms  表达式引擎={t_vec*1000:.1f}ms  加速={speedup:.2f}x")
        assert speedup >= 1.0, f"表达式引擎未带来加速: {speedup:.2f}x"


# ============================================================
# 3. IC 分析性能对比
# ============================================================

class TestICAnalysisPerformance:
    """向量化 IC vs 逐日 scipy 循环"""

    @pytest.mark.parametrize("n_codes,n_days", [(100, 250), (300, 500)])
    def test_ic_speedup(self, n_codes, n_days):
        from vectorized_ic_analysis import VectorizedICAnalysis
        from scipy import stats

        data = generate_synthetic_ohlcv(n_codes=n_codes, n_days=n_days, seed=77)
        df = data.sort_values(["code", "date"]).reset_index(drop=True)
        df["factor"] = -df.groupby("code")["close"].transform(lambda x: x.pct_change(5))
        df["fwd_5d"] = df.groupby("code")["close"].transform(lambda x: x.shift(-5) / x - 1)

        # 基准：逐日 scipy spearmanr（与主仓库 _calc_ic 相同）
        def baseline_ic():
            ic_list = []
            for dt, cross in df.dropna(subset=["factor", "fwd_5d"]).groupby("date"):
                if len(cross) < 10:
                    continue
                ic, _ = stats.spearmanr(cross["factor"], cross["fwd_5d"])
                if not np.isnan(ic):
                    ic_list.append(ic)
            return ic_list

        _, t_b1 = _timed(baseline_ic)
        _, t_b2 = _timed(baseline_ic)
        t_base = min(t_b1, t_b2)

        _, t_v1 = _timed(VectorizedICAnalysis.calc_ic_series, df, "factor", "fwd_5d", "spearman")
        _, t_v2 = _timed(VectorizedICAnalysis.calc_ic_series, df, "factor", "fwd_5d", "spearman")
        t_vec = min(t_v1, t_v2)

        speedup = t_base / t_vec if t_vec > 0 else float("inf")
        print(f"\n[IC {n_codes}股×{n_days}日] 逐日scipy={t_base*1000:.1f}ms  向量化={t_vec*1000:.1f}ms  加速={speedup:.2f}x")
        assert speedup >= 1.0, f"向量化 IC 未带来加速: {speedup:.2f}x"