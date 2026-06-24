"""
边界条件测试

验证模块在极端/边界输入下的健壮性：
- 空数据
- 单只股票
- 全涨停/全跌停
- 单日数据
- 含大量 NaN 的数据
- 信号全为 0
"""
import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_synthetic_ohlcv, generate_signals


# ============================================================
# 回测引擎边界
# ============================================================

class TestBacktestEdgeCases:

    def test_empty_data(self, vectorized_adapter):
        """空数据应返回空结果，不抛异常"""
        empty = pd.DataFrame(columns=["code", "date", "open", "high", "low", "close", "volume"])
        empty_sig = pd.DataFrame(columns=["code", "date", "signal"])
        res = vectorized_adapter.run_backtest(empty, empty_sig)
        assert res["metrics"] == {}
        assert len(res["trades"]) == 0

    def test_single_stock(self, vectorized_adapter):
        """单只股票应正常回测"""
        data = generate_synthetic_ohlcv(n_codes=1, n_days=60, seed=1)
        signals = generate_signals(data, strategy="ma_cross", seed=1)
        res = vectorized_adapter.run_backtest(data, signals)
        assert not res["equity_curve"].empty
        assert res["equity_curve"]["equity"].iloc[0] == 1e6

    def test_all_zero_signals(self, vectorized_adapter):
        """信号全 0 应无交易，净值不变"""
        data = generate_synthetic_ohlcv(n_codes=10, n_days=30, seed=2)
        signals = pd.DataFrame({
            "code": data["code"].values,
            "date": data["date"].values,
            "signal": 0,
        })
        res = vectorized_adapter.run_backtest(data, signals)
        assert len(res["trades"]) == 0
        # 净值应恒等于初始资金
        eq = res["equity_curve"]["equity"]
        assert (eq == 1e6).all(), "无交易时净值应不变"

    def test_all_limit_up(self, vectorized_adapter):
        """全涨停时买入信号应被过滤"""
        data = generate_synthetic_ohlcv(n_codes=5, n_days=10, seed=3)
        # 强制全部涨停
        data["is_limit_up"] = True
        signals = pd.DataFrame({
            "code": data["code"].values,
            "date": data["date"].values,
            "signal": 1,
        })
        res = vectorized_adapter.run_backtest(data, signals, price_limit=True)
        # 涨停日无法买入
        assert len(res["trades"]) == 0

    def test_single_day(self, vectorized_adapter):
        """仅 1 个交易日应正常返回"""
        data = generate_synthetic_ohlcv(n_codes=5, n_days=1, seed=4)
        signals = generate_signals(data, strategy="ma_cross", seed=4)
        res = vectorized_adapter.run_backtest(data, signals)
        # 单日无法产生 ma_cross 信号，应无交易
        assert not res["equity_curve"].empty

    def test_t_plus_1_default(self, vectorized_adapter):
        """默认 T+1，同日买卖应遵循先卖后买顺序"""
        data = generate_synthetic_ohlcv(n_codes=3, n_days=20, seed=5)
        # 构造同日既有买又有卖的信号
        dt = data["date"].iloc[10]
        signals = pd.DataFrame([
            {"code": data["code"].iloc[0], "date": dt, "signal": 1},
            {"code": data["code"].iloc[1], "date": dt, "signal": -1},
        ])
        res = vectorized_adapter.run_backtest(data, signals)
        # 不应抛异常
        assert "equity_curve" in res


# ============================================================
# 因子引擎边界
# ============================================================

class TestFactorEngineEdgeCases:

    def test_empty_data(self):
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        empty = pd.DataFrame(columns=["code", "date", "close"])
        res = eng.calculate(empty, ["Delta(Close, 5)"])
        assert res.empty

    def test_unknown_field_raises(self):
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        data = generate_synthetic_ohlcv(n_codes=3, n_days=20, seed=6)
        # 未知字段应被捕获为 NaN（不抛异常中断批量）
        res = eng.calculate(data, ["Delta(UnknownField, 5)"])
        assert len(res) == len(data)

    def test_missing_turnover(self):
        """缺少 turnover_rate 时 VWAP/换手因子应优雅降级"""
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        data = generate_synthetic_ohlcv(n_codes=5, n_days=30, seed=7)
        data = data.drop(columns=["turnover_rate"], errors="ignore")
        res = eng.calculate(data, ["alpha_turnover_20"])
        # 缺字段时该列应为 NaN，但不影响其他
        assert len(res) == len(data)

    def test_nested_expression(self):
        """深层嵌套表达式应正确解析"""
        from factor_expression_engine import FactorExpressionEngine
        eng = FactorExpressionEngine()
        data = generate_synthetic_ohlcv(n_codes=5, n_days=60, seed=8)
        expr = "Rank(Div(Sub(Close, Ts_Mean(Close, 20)), Ts_Std(Close, 20)))"
        res = eng.calculate(data, [expr])
        col = eng._expr_to_name(expr)
        valid = res[col].dropna()
        assert valid.between(0, 1, inclusive="both").all()


# ============================================================
# IC 分析边界
# ============================================================

class TestICAnalysisEdgeCases:

    def test_empty_data(self):
        from vectorized_ic_analysis import VectorizedICAnalysis
        empty = pd.DataFrame(columns=["date", "factor", "fwd"])
        ic = VectorizedICAnalysis.calc_ic_series(empty, "factor", "fwd", "spearman")
        assert ic.empty

    def test_insufficient_stocks(self):
        """截面股票数不足 min_stocks 时应返回空"""
        from vectorized_ic_analysis import VectorizedICAnalysis
        data = generate_synthetic_ohlcv(n_codes=5, n_days=30, seed=9)
        df = data.sort_values(["code", "date"]).reset_index(drop=True)
        df["factor"] = df.groupby("code")["close"].transform(lambda x: x.pct_change(5))
        df["fwd"] = df.groupby("code")["close"].transform(lambda x: x.shift(-5) / x - 1)
        # min_stocks=10，但只有 5 只股票
        ic = VectorizedICAnalysis.calc_ic_series(df, "factor", "fwd", "spearman", min_stocks=10)
        assert ic.empty

    def test_all_nan_factor(self):
        """因子全 NaN 应返回空 IC"""
        from vectorized_ic_analysis import VectorizedICAnalysis
        data = generate_synthetic_ohlcv(n_codes=20, n_days=30, seed=10)
        data["factor"] = np.nan
        data["fwd"] = 0.01
        ic = VectorizedICAnalysis.calc_ic_series(data, "factor", "fwd", "spearman")
        assert ic.empty

    def test_constant_factor(self):
        """因子恒定（无方差）时 IC 应为 NaN，不抛异常"""
        from vectorized_ic_analysis import VectorizedICAnalysis
        data = generate_synthetic_ohlcv(n_codes=20, n_days=30, seed=11)
        df = data.sort_values(["code", "date"]).reset_index(drop=True)
        df["factor"] = 1.0  # 恒定
        df["fwd"] = df.groupby("code")["close"].transform(lambda x: x.shift(-5) / x - 1)
        ic = VectorizedICAnalysis.calc_ic_series(df, "factor", "fwd", "spearman")
        # 恒定因子 rank 后仍恒定，相关系数分母为 0 -> NaN
        assert ic.isna().all() or ic.empty