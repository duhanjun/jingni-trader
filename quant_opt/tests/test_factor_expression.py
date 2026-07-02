"""
Module 3 测试: 因子表达式引擎
"""
import numpy as np
import pandas as pd
import pytest

from quant_opt.core.factor_expression import (
    FactorExpressionEngine,
    OPERATORS,
    ALPHA101_DEMO,
    VAR_REGEX,
)


def _make_data(n_stocks: int = 30, n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    codes = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]
    rows = []
    for c in codes:
        start = rng.uniform(10, 50)
        ret = rng.normal(0.0005, 0.02, n_days)
        price = start * (1 + ret).cumprod()
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "code": c,
                "open": price[i],
                "high": price[i] * 1.01,
                "low": price[i] * 0.99,
                "close": price[i],
                "volume": int(rng.lognormal(15, 0.3)),
                "amount": float(price[i] * rng.lognormal(15, 0.3)),
            })
    return pd.DataFrame(rows)


class TestFactorExpression:
    """因子表达式引擎测试套件"""

    def test_variable_reference(self):
        """正确性: 简单变量引用"""
        data = _make_data(n_stocks=10, n_days=50, seed=1)
        eng = FactorExpressionEngine(data)
        s = eng.compute("$close")
        assert s.name is None or len(s) == len(data)

    def test_ts_mean_matches_pandas_rolling(self):
        """正确性: Ts_Mean 应与 pandas rolling().mean() 一致"""
        data = _make_data(n_stocks=5, n_days=80, seed=2)
        eng = FactorExpressionEngine(data)
        ours = eng.compute("Ts_Mean($close, 10)")

        # pandas 参考实现（按 (code, date) 排序的 close）
        data_sorted = data.sort_values(["code", "date"]).reset_index(drop=True)
        ref = data_sorted.groupby("code")["close"].transform(
            lambda x: x.rolling(10, min_periods=2).mean()
        )

        # ours 是带 (date, code) MultiIndex 的 Series
        # 重塑成与 ref 完全一致的 (code, date) 顺序
        ours_flat = ours.reset_index().sort_values(["code", "date"]).reset_index(drop=True)
        assert len(ours_flat) == len(ref), \
            f"长度不一致 {len(ours_flat)} vs {len(ref)}"
        np.testing.assert_allclose(
            ours_flat.iloc[:, -1].values,  # 最后一列是计算结果
            ref.values,
            atol=1e-6,
        )

    def test_rank_is_cross_sectional(self):
        """正确性: Rank 应是截面归一化"""
        data = _make_data(n_stocks=20, n_days=30, seed=3)
        eng = FactorExpressionEngine(data)
        rank = eng.compute("Rank($close)")
        # 截面秩归一化后,每日应在 [0,1] 内
        by_date = rank.reset_index(level="code", drop=True)
        daily_groups = by_date.groupby(level="date")
        for d, vals in daily_groups:
            valid = vals.dropna()
            if len(valid) > 1:
                assert valid.min() >= 0 - 1e-9
                assert valid.max() <= 1 + 1e-9

    def test_nested_expression(self):
        """正确性: 嵌套算子应能求值"""
        data = _make_data(n_stocks=10, n_days=50, seed=4)
        eng = FactorExpressionEngine(data)
        # Rank(Return($close, 5))  =  5 日反转因子的截面秩
        result = eng.compute("Rank(Return($close, 5))")
        assert len(result) == len(data)
        # 截面 rank 应在 [0,1]
        by_date = result.reset_index(level="code", drop=True).groupby(level="date")
        for d, vals in list(by_date)[:5]:
            valid = vals.dropna()
            if len(valid) > 0:
                assert valid.min() >= 0 - 1e-9
                assert valid.max() <= 1 + 1e-9

    def test_arithmetic_in_expression(self):
        """正确性: 算术表达式"""
        data = _make_data(n_stocks=8, n_days=40, seed=5)
        eng = FactorExpressionEngine(data)
        # 均线偏离
        result = eng.compute("Ts_Mean($close, 5) - Ts_Mean($close, 20)")
        assert result.notna().sum() > 0

    def test_batch_compute(self):
        """正确性: 批量计算"""
        data = _make_data(n_stocks=10, n_days=60, seed=6)
        eng = FactorExpressionEngine(data)
        exprs = [
            "Rank(Return($close, 20))",
            "Sign($close - Delay($close, 1))",
        ]
        result = eng.compute_batch(exprs)
        assert "date" in result.columns
        assert "code" in result.columns
        # 至少 2 个因子列
        factor_cols = [c for c in result.columns if c not in ("date", "code")]
        assert len(factor_cols) == 2
        for c in factor_cols:
            assert result[c].notna().sum() > 0

    def test_alpha101_demo_runs(self):
        """正确性: Alpha101 示例表达式都能成功求值"""
        data = _make_data(n_stocks=20, n_days=100, seed=7)
        eng = FactorExpressionEngine(data)
        for expr in ALPHA101_DEMO:
            result = eng.compute(expr)
            assert len(result) == len(data), f"长度异常: {expr}"
            # 大部分应有有效值
            valid = result.notna().sum() / len(result)
            assert valid > 0.3, f"{expr} 有效值比例仅 {valid:.2%}"

    def test_unsafe_expression_rejected(self):
        """安全性: 含 eval/import 的表达式应被拒绝"""
        data = _make_data(n_stocks=5, n_days=20, seed=8)
        eng = FactorExpressionEngine(data)
        with pytest.raises(ValueError):
            eng.compute("__import__('os').system('rm -rf /')")
        with pytest.raises(ValueError):
            eng.compute("eval('1+1')")

    def test_unknown_operator_rejected(self):
        """边界: 未知算子应抛错而不是静默忽略"""
        data = _make_data(n_stocks=5, n_days=20, seed=9)
        eng = FactorExpressionEngine(data)
        with pytest.raises((ValueError, RuntimeError), match="未知算子"):
            eng.compute("NonexistentOp($close, 5)")

    def test_unknown_variable_rejected(self):
        """边界: 未知变量应抛错"""
        data = _make_data(n_stocks=5, n_days=20, seed=10)
        eng = FactorExpressionEngine(data)
        with pytest.raises((KeyError, RuntimeError), match="未知变量"):
            eng.compute("Rank($nonexistent)")

    def test_decay_linear_window(self):
        """正确性: Decay_Linear 加权平均窗口长度"""
        data = _make_data(n_stocks=5, n_days=60, seed=11)
        eng = FactorExpressionEngine(data)
        result = eng.compute("Decay_Linear($volume, 10)")
        # 前 9 个应为 NaN（窗口不足）
        result_reset = result.reset_index(drop=True)
        # 检查每个 code 的前 9 个
        for c in data["code"].unique()[:2]:
            mask = data.sort_values(["date", "code"]).reset_index(drop=True)["code"] == c
            vals = result_reset[mask].values
            assert np.isnan(vals[:9]).all(), f"Decay_Linear 前 9 个应为 NaN, 但 {vals[:9]}"
            # 第 10 个起应有效
            assert np.isfinite(vals[10:]).all()

    def test_extensibility(self):
        """可扩展性: 用户可注册新算子"""
        from quant_opt.core.factor_expression import OPERATORS
        data = _make_data(n_stocks=5, n_days=30, seed=12)
        eng = FactorExpressionEngine(data)

        # 注册一个自定义算子
        def _neg(s: pd.Series) -> pd.Series:
            return -s

        OPERATORS["Neg"] = _neg
        try:
            result = eng.compute("Neg($close)")
            assert (result.values == -data.sort_values(["date", "code"])["close"].values).all()
        finally:
            del OPERATORS["Neg"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
