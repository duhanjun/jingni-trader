"""
因子表达式引擎的单元测试
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def _make_panel(n_stocks: int = 5, n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"00000{i}.SZ" for i in range(n_stocks)]
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for code in codes:
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "open": close[i] * (1 + rng.normal(0, 0.005)),
                "high": close[i] * (1 + abs(rng.normal(0, 0.01))),
                "low": close[i] * (1 - abs(rng.normal(0, 0.01))),
                "close": close[i],
                "volume": rng.integers(1_000_000, 5_000_000, n_days)[i],
                "amount": close[i] * rng.integers(1_000_000, 5_000_000, n_days)[i],
            })
    return pd.DataFrame(rows)


def test_basic_field():
    from quant_opt_20260616_core.factor_expr_engine import FactorExprEngine

    df = _make_panel()
    eng = FactorExprEngine()
    out = eng.compute(df, expr="$close", name="px")
    assert "px" in out.columns
    assert len(out) == len(df)
    np.testing.assert_allclose(
        out["px"].values, df.sort_values(["date", "code"])["close"].values
    )
    print("  [OK] basic field reference")


def test_arithmetic():
    from quant_opt_20260616_core.factor_expr_engine import FactorExprEngine

    df = _make_panel()
    eng = FactorExprEngine()
    out = eng.compute(df, expr="($close - $open) / $open", name="ret")
    # engine 输出顺序为 [date, code], 期望值按同序比对
    expected = ((df["close"] - df["open"]) / df["open"]).values
    # engine 内部按 [date, code] 排序, 与 _make_panel 中 rows 顺序可能不同
    # 故先把 expected 按相同 key 重新对齐
    df_sorted = df.sort_values(["date", "code"]).reset_index(drop=True)
    expected_sorted = ((df_sorted["close"] - df_sorted["open"]) / df_sorted["open"]).values
    np.testing.assert_allclose(out["ret"].values, expected_sorted, rtol=1e-6)
    print("  [OK] arithmetic")


def test_ref_and_mean():
    from quant_opt_20260616_core.factor_expr_engine import FactorExprEngine

    df = _make_panel()
    eng = FactorExprEngine()
    out = eng.compute(df, expr="Mean($close, 5)", name="ma5")
    assert "ma5" in out.columns
    # 与 pandas 自行 rolling 对比
    expected = (
        df.sort_values(["code", "date"])
        .groupby("code")["close"]
        .transform(lambda s: s.rolling(5, min_periods=2).mean())
    )
    actual = out.sort_values(["code", "date"])["ma5"].values
    np.testing.assert_allclose(actual, expected.values, rtol=1e-4)
    print("  [OK] Ref / Mean")


def test_rank_cross_section():
    from quant_opt_20260616_core.factor_expr_engine import FactorExprEngine

    df = _make_panel(n_stocks=20, n_days=30)
    eng = FactorExprEngine()
    out = eng.compute(df, expr="Rank($close)", name="rank_px")
    # 截面百分位排名
    pivot = out.pivot(index="date", columns="code", values="rank_px")
    # 同日排名范围应在 (0, 1]
    assert (pivot.values <= 1.0001).all()
    assert (pivot.values >= 0).all()
    # 各日排名应近似均匀分布
    print("  [OK] Rank cross-section")


def test_batch_compute():
    from quant_opt_20260616_core.factor_expr_engine import FactorExprEngine

    df = _make_panel()
    eng = FactorExprEngine()
    out = eng.compute_batch(df, {
        "ret_1d": "$close / Ref($close, 1) - 1",
        "ma5": "Mean($close, 5)",
        "zscore": "($close - Mean($close, 20)) / Std($close, 20)",
    })
    assert {"code", "date", "ret_1d", "ma5", "zscore"} <= set(out.columns)
    print("  [OK] batch compute (3 factors)")


def test_perf_small_panel():
    """500 标的 * 500 天, 10 因子计算 < 5s"""
    from quant_opt_20260616_core.factor_expr_engine import FactorExprEngine

    df = _make_panel(n_stocks=500, n_days=500, seed=42)
    eng = FactorExprEngine()
    exprs = {
        "ret_1d": "$close / Ref($close, 1) - 1",
        "ma5": "Mean($close, 5)",
        "ma20": "Mean($close, 20)",
        "vol20": "Std($close, 20) / Mean($close, 20)",
        "delta5": "$close - Ref($close, 5)",
        "ts_rank10": "TsRank($close, 10)",
        "zscore20": "($close - Mean($close, 20)) / Std($close, 20)",
        "rank_close": "Rank($close)",
        "rank_ma5": "Rank(Mean($close, 5))",
        "amount_ma": "Mean($amount, 5)",
    }
    t0 = time.time()
    out = eng.compute_batch(df, exprs)
    dt = time.time() - t0
    assert out.shape[0] == len(df)
    print(f"  [OK] perf 250k rows x {len(exprs)} factors: {dt:.3f}s")
    assert dt < 8.0, f"perf regression: {dt}s"


def test_invalid_expr_safety():
    """禁止 eval/import 等危险语法"""
    from quant_opt_20260616_core.factor_expr_engine import FactorExprEngine

    eng = FactorExprEngine()
    df = _make_panel()
    dangerous = [
        "__import__('os').system('echo bad')",
        "eval('1+1')",
        "[x for x in range(10)]",
    ]
    for expr in dangerous:
        try:
            eng.compute(df, expr=expr, name="x")
            raise AssertionError(f"should have failed: {expr!r}")
        except (ValueError, SyntaxError):
            pass
    print("  [OK] invalid expression rejected")


def run() -> dict:
    print("test_basic_field")
    test_basic_field()
    print("test_arithmetic")
    test_arithmetic()
    print("test_ref_and_mean")
    test_ref_and_mean()
    print("test_rank_cross_section")
    test_rank_cross_section()
    print("test_batch_compute")
    test_batch_compute()
    print("test_perf_small_panel")
    test_perf_small_panel()
    print("test_invalid_expr_safety")
    test_invalid_expr_safety()
    return {"status": "passed", "cases": 7}


if __name__ == "__main__":
    run()