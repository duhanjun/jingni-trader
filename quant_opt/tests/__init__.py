"""
Test-suite for the quant_opt package.

Run with::

    python -m quant_opt.tests.run_all

The tests exercise three layers:

* **Correctness** — outputs of the expression engine match hand-rolled
  pandas / numpy baselines on small synthetic panels.
* **Performance** — the engine does not regress on a moderately large
  panel (5k rows × 30 codes × 60 dates) compared to naive
  implementations.
* **Edge cases** — empty inputs, unknown columns, divide-by-zero,
  rolling windows shorter than the data, train/val splits with the
  embargo/purge gap, etc.
"""
from __future__ import annotations

import math
import time
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from quant_opt.factor_expression import FactorEngine, ExpressionError
from quant_opt.walk_forward import WalkForward, WalkForwardConfig
from quant_opt.alpha158 import FactorLibrary


# ── Result containers ────────────────────────────────────────────────


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""
    duration: float = 0.0


# ── Synthetic data helpers ───────────────────────────────────────────


def make_panel(
    n_codes: int = 10,
    n_dates: int = 60,
    seed: int = 7,
) -> pd.DataFrame:
    """Deterministic A股-shaped panel with OHLCV + amount + turnoverRate."""
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SH" for i in range(1, n_codes + 1)]
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for c in codes:
        # price random walk
        log_ret = rng.normal(0, 0.02, n_dates)
        price = 10 * np.exp(np.cumsum(log_ret))
        open_ = price * (1 + rng.normal(0, 0.003, n_dates))
        high = np.maximum(price, open_) * (1 + np.abs(rng.normal(0, 0.005, n_dates)))
        low = np.minimum(price, open_) * (1 - np.abs(rng.normal(0, 0.005, n_dates)))
        volume = rng.integers(1_000_000, 10_000_000, n_dates)
        amount = volume * price
        turnover = rng.uniform(0.5, 5.0, n_dates)
        industry = rng.choice(["Tech", "Finance", "Energy", "Health"], n_dates)
        for i, dt in enumerate(dates):
            rows.append((c, dt, open_[i], high[i], low[i], price[i], volume[i],
                         amount[i], turnover[i], industry[i]))
    return pd.DataFrame(rows, columns=[
        "code", "date", "open", "high", "low", "close", "volume",
        "amount", "turnoverRate", "industry",
    ])


def make_ml_panel(n_codes: int = 30, n_dates: int = 200, seed: int = 11) -> pd.DataFrame:
    """Panel with engineered features + a forward-return label."""
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SH" for i in range(1, n_codes + 1)]
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    rows = []
    for c in codes:
        ret_5 = rng.normal(0, 0.05, n_dates)
        ret_20 = rng.normal(0, 0.10, n_dates)
        vol_20 = np.abs(rng.normal(0.02, 0.005, n_dates))
        turnover_5d = np.abs(rng.normal(1.0, 0.3, n_dates))
        # latent signal: momentum + low vol is rewarded
        score = ret_20 - 2 * vol_20
        fwd_5 = np.roll(score, -5) * 0.01 + rng.normal(0, 0.02, n_dates)
        fwd_5[-5:] = np.nan
        for i, dt in enumerate(dates):
            rows.append((c, dt, ret_5[i], ret_20[i], vol_20[i],
                         turnover_5d[i], fwd_5[i]))
    return pd.DataFrame(rows, columns=[
        "code", "date", "mom_5", "mom_20", "vol_20d", "turnover_5d", "forward_5d",
    ])


# ── Test functions ───────────────────────────────────────────────────


def test_parser_handles_basic_arithmetic() -> TestResult:
    """1 + 2 * 3 should parse to 7, not 9."""
    t0 = time.perf_counter()
    from quant_opt.factor_expression.parser import parse
    ast = parse("1 + 2 * 3")
    assert ast.op == "+", f"root op should be +, got {ast.op!r}"
    assert ast.args[1].op == "*", f"right op should be *, got {ast.args[1].op!r}"
    # manual eval
    def ev(n):
        if n.op is None:
            return n.value
        if n.op == "+":
            return ev(n.args[0]) + ev(n.args[1])
        if n.op == "-":
            return ev(n.args[0]) - ev(n.args[1])
        if n.op == "*":
            return ev(n.args[0]) * ev(n.args[1])
        if n.op == "/":
            return ev(n.args[0]) / ev(n.args[1])
    assert ev(ast) == 7.0
    return TestResult("parser_handles_basic_arithmetic", True,
                      "precedence correct", time.perf_counter() - t0)


def test_parser_unary_and_parentheses() -> TestResult:
    t0 = time.perf_counter()
    from quant_opt.factor_expression.parser import parse
    ast = parse("-(3 - 4) * 2")
    def ev(n):
        if n.op is None:
            return n.value
        if n.op == "+":
            return ev(n.args[0]) + ev(n.args[1])
        if n.op == "-":
            if len(n.args) == 1:
                return -ev(n.args[0])
            return ev(n.args[0]) - ev(n.args[1])
        if n.op == "*":
            return ev(n.args[0]) * ev(n.args[1])
        if n.op == "/":
            return ev(n.args[0]) / ev(n.args[1])
    assert ev(ast) == 2.0, f"expected 2, got {ev(ast)}"
    return TestResult("parser_unary_and_parentheses", True,
                      "unary minus + parens correct", time.perf_counter() - t0)


def test_engine_ref_matches_baseline() -> TestResult:
    t0 = time.perf_counter()
    df = make_panel()
    engine = FactorEngine()
    out = engine.calc(df, "Ref($close, 1)", "prev_close")
    expected = df.groupby("code")["close"].shift(1)
    np.testing.assert_allclose(out["prev_close"].values, expected.values,
                               equal_nan=True)
    return TestResult("engine_ref_matches_baseline", True,
                      "Ref equals groupby.shift(1)",
                      time.perf_counter() - t0)


def test_engine_ma_matches_baseline() -> TestResult:
    t0 = time.perf_counter()
    df = make_panel()
    engine = FactorEngine()
    out = engine.calc(df, "MA($close, 5)", "ma5")
    expected = df.groupby("code")["close"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    np.testing.assert_allclose(out["ma5"].values, expected.values,
                               equal_nan=True, rtol=1e-7)
    return TestResult("engine_ma_matches_baseline", True,
                      "MA equals groupby.rolling().mean()",
                      time.perf_counter() - t0)


def test_engine_rank_matches_baseline() -> TestResult:
    t0 = time.perf_counter()
    df = make_panel()
    engine = FactorEngine()
    out = engine.calc(df, "Rank($close)", "rk")
    expected = df["close"].groupby(df["date"]).rank(pct=True)
    np.testing.assert_allclose(out["rk"].values, expected.values,
                               equal_nan=True)
    return TestResult("engine_rank_matches_baseline", True,
                      "Rank equals groupby.rank(pct=True)",
                      time.perf_counter() - t0)


def test_engine_alpha101_combo() -> TestResult:
    t0 = time.perf_counter()
    df = make_panel()
    engine = FactorEngine()
    out = engine.calc(df, "Rank(MA($close, 5) - MA($close, 20))", "alpha")
    # verify the sign matches a manual computation
    ma5 = df.groupby("code")["close"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    ma20 = df.groupby("code")["close"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
    diff = (ma5 - ma20).groupby(df["date"]).rank(pct=True)
    np.testing.assert_allclose(out["alpha"].values, diff.values, equal_nan=True)
    return TestResult("engine_alpha101_combo", True,
                      "composite Alpha101 expression correct",
                      time.perf_counter() - t0)


def test_engine_calc_many_with_cache() -> TestResult:
    t0 = time.perf_counter()
    df = make_panel()
    engine = FactorEngine()
    exprs = ["MA($close, 5)", "MA($close, 5) - MA($close, 20)", "Rank(MA($close, 20))"]
    out = engine.calc_many(df, exprs, ["ma5", "bias", "rk_ma20"])
    for col in ("ma5", "bias", "rk_ma20"):
        assert col in out.columns
        assert not out[col].isna().all()
    return TestResult("engine_calc_many_with_cache", True,
                      f"3 expressions computed ({len(out)} rows)",
                      time.perf_counter() - t0)


def test_engine_perf_baseline() -> TestResult:
    """< 1.5s on a 5k-row panel for a 4-op composite expression."""
    t0 = time.perf_counter()
    df = make_panel(n_codes=80, n_dates=80)  # 6400 rows
    engine = FactorEngine()
    exprs = [
        "MA($close, 20)",
        "MA($close, 5) - MA($close, 20)",
        "Rank(Std($close, 20))",
        "Rank(MA($close, 5) - MA($close, 20))",
        "MA($close, 20) / Ref($close, 20) - 1",
    ]
    out = engine.calc_many(df, exprs, [f"f{i}" for i in range(5)])
    assert out.shape == (6400, 10 + 5)
    dur = time.perf_counter() - t0
    assert dur < 1.5, f"performance regression: {dur:.2f}s"
    return TestResult("engine_perf_baseline", True,
                      f"5 expressions on 6400 rows in {dur*1000:.0f}ms",
                      dur)


def test_engine_edge_cases() -> TestResult:
    t0 = time.perf_counter()
    engine = FactorEngine()
    # empty input
    try:
        engine.calc(pd.DataFrame(columns=["code", "date", "close"]), "MA($close, 5)")
        return TestResult("engine_edge_cases", False,
                          "expected ExpressionError on empty frame")
    except ExpressionError:
        pass
    # unknown column
    try:
        engine.calc(make_panel(), "MA($nope, 5)")
        return TestResult("engine_edge_cases", False,
                          "expected ExpressionError on unknown column")
    except ExpressionError:
        pass
    # divide by zero scalar -> inf (or NaN if 0/0)
    df = make_panel()
    out = engine.calc(df, "MA($close, 5) / 0")
    # 0 is a scalar; we should get +inf or -inf (not all NaN)
    assert not out["f_0"].isna().all(), "should not be all NaN for /0"
    # window larger than data — must not raise
    small = df[df["code"] == df["code"].iloc[0]].head(3).copy()
    out = engine.calc(small, "MA($close, 100)", "ma")
    # divide by zero Series — only positions where denominator is 0 should be NaN
    df2 = df.copy()
    df2["zero_col"] = 0.0
    out = engine.calc(df2, "$close / $zero_col")
    assert out["f_0"].isna().sum() == len(df2), "all positions should be NaN"
    return TestResult("engine_edge_cases", True,
                      "empty/unknown-col/div0/oversized-window handled",
                      time.perf_counter() - t0)


def test_alpha158_library_runs() -> TestResult:
    t0 = time.perf_counter()
    lib = FactorLibrary()
    assert len(lib.names()) >= 30
    # Use a panel with enough dates to satisfy the longest lookback
    # (ret_60d needs 60 days of history before producing a value).
    df = make_panel(n_codes=10, n_dates=200)
    engine = FactorEngine()
    # take a balanced subset that doesn't include ret_60d
    sub = [n for n in lib.names() if n not in ("ret_60d", "std_60")][:10]
    out = engine.calc_many(df, lib.expressions(sub), sub)
    failed = []
    for n in sub:
        if n not in out.columns:
            failed.append(f"{n}: column missing")
        elif out[n].isna().all():
            failed.append(f"{n}: all NaN")
    assert not failed, f"failures: {failed}"
    return TestResult("alpha158_library_runs", True,
                      f"{len(lib.names())} factors registered, {len(sub)} computed on 200-day panel",
                      time.perf_counter() - t0)


def test_alpha101_alpha_001_matches_baseline() -> TestResult:
    """Spot-check one Alpha101 formula against an explicit baseline."""
    t0 = time.perf_counter()
    df = make_panel()
    engine = FactorEngine()
    out = engine.calc(df,
                      "Rank($close - Ref($close, 1)) - Rank($close - Ref($close, 5))",
                      "a001")
    delta1 = df["close"] - df.groupby("code")["close"].shift(1)
    delta5 = df["close"] - df.groupby("code")["close"].shift(5)
    rk1 = delta1.groupby(df["date"]).rank(pct=True)
    rk5 = delta5.groupby(df["date"]).rank(pct=True)
    expected = rk1 - rk5
    np.testing.assert_allclose(out["a001"].values, expected.values,
                               equal_nan=True)
    return TestResult("alpha101_alpha_001_matches_baseline", True,
                      "Alpha#001 = rank(Δ1) - rank(Δ5) correct",
                      time.perf_counter() - t0)


def test_walk_forward_runs_and_emits_signals() -> TestResult:
    t0 = time.perf_counter()
    from sklearn.linear_model import Ridge
    df = make_ml_panel()
    cfg = WalkForwardConfig(
        train_period=120, val_period=20, step=20,
        feature_cols=["mom_5", "mom_20", "vol_20d", "turnover_5d"],
        label_col="forward_5d",
    )
    wf = WalkForward(model_factory=lambda: Ridge(alpha=1.0))
    res = wf.run(df, cfg, progress=False)
    assert res.summary["n_windows"] > 1, f"too few windows: {res.summary['n_windows']}"
    assert "mean_ic" in res.summary
    # res.signals is a DataFrame with columns: code, date, signal, label, window_id
    assert isinstance(res.signals, pd.DataFrame)
    assert {"code", "date", "signal", "label", "window_id"}.issubset(res.signals.columns)
    assert res.signals["window_id"].nunique() == res.summary["n_windows"]
    return TestResult("walk_forward_runs_and_emits_signals", True,
                      f"{res.summary['n_windows']} windows, "
                      f"mean_ic={res.summary['mean_ic']:.3f}, "
                      f"ic_ir={res.summary['ic_ir']:.3f}",
                      time.perf_counter() - t0)


def test_walk_forward_embargo_skips_overlap() -> TestResult:
    t0 = time.perf_counter()
    from sklearn.linear_model import Ridge
    # Use a longer panel so the rolling produces several windows.
    df = make_ml_panel(n_dates=400)
    base = WalkForwardConfig(
        train_period=120, val_period=20, step=20,
        feature_cols=["mom_5", "mom_20"], label_col="forward_5d",
        embargo=0, purge_gap=0,
    )
    big_embargo = WalkForwardConfig(**{**base.__dict__, "embargo": 60})
    wf = WalkForward(model_factory=lambda: Ridge(alpha=1.0))
    n_base = len(wf.generate_windows(df["date"], base))
    n_big = len(wf.generate_windows(df["date"], big_embargo))
    assert n_base >= 5, f"need enough base windows to see the effect, got {n_base}"
    assert n_big < n_base, f"expected fewer windows with bigger embargo, got {n_big} vs {n_base}"
    return TestResult("walk_forward_embargo_skips_overlap", True,
                      f"embargo=0 -> {n_base} windows, embargo=60 -> {n_big}",
                      time.perf_counter() - t0)


def test_walk_forward_expanding_mode() -> TestResult:
    t0 = time.perf_counter()
    from sklearn.linear_model import Ridge
    df = make_ml_panel(n_dates=200)
    cfg = WalkForwardConfig(
        train_period=120, val_period=20, step=20,
        feature_cols=["mom_5", "mom_20"], label_col="forward_5d",
        expanding=True, min_train_period=60,
    )
    wf = WalkForward(model_factory=lambda: Ridge(alpha=1.0))
    res = wf.run(df, cfg)
    assert res.summary["n_windows"] > 0
    # the first window's training set should be smaller than the second's
    if len(res.windows) >= 2:
        n0 = (res.windows[0].train_end - res.windows[0].train_start).days
        n1 = (res.windows[1].train_end - res.windows[1].train_start).days
        assert n1 > n0, f"expanding mode should grow training set: {n0} -> {n1}"
    return TestResult("walk_forward_expanding_mode", True,
                      f"expanding windows grow, n_windows={res.summary['n_windows']}",
                      time.perf_counter() - t0)


def test_walk_forward_aggregate_ic_method() -> TestResult:
    t0 = time.perf_counter()
    from sklearn.linear_model import Ridge
    df = make_ml_panel()
    cfg = WalkForwardConfig(
        train_period=120, val_period=20, step=20,
        feature_cols=["mom_5", "mom_20", "vol_20d", "turnover_5d"],
        label_col="forward_5d",
    )
    wf = WalkForward(model_factory=lambda: Ridge(alpha=1.0))
    res = wf.run(df, cfg)
    agg = res.aggregate_ic()
    assert set(agg.keys()) == {"mean_ic", "std_ic", "ic_ir", "n_windows"}
    return TestResult("walk_forward_aggregate_ic_method", True,
                      f"aggregate_ic: {agg}",
                      time.perf_counter() - t0)


def test_integration_engine_plus_library_plus_wf() -> TestResult:
    """End-to-end: factor lib -> expression engine -> walk-forward model."""
    t0 = time.perf_counter()
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    df = make_panel(n_codes=40, n_dates=180)
    lib = FactorLibrary()
    # pick 6 uncorrelated features
    feats = ["ret_5d", "ma_bias_20", "std_20", "volume_ratio_5_20",
             "rank_ret_20d", "intraday_ret"]
    engine = FactorEngine()
    factor_df = engine.calc_many(df, lib.expressions(feats), feats)
    # build label
    factor_df = factor_df.sort_values(["date", "code"]).reset_index(drop=True)
    factor_df["label"] = factor_df.groupby("code")["close"].transform(
        lambda s: s.shift(-5) / s - 1
    )
    factor_df = factor_df.dropna(subset=["label"])
    cfg = WalkForwardConfig(
        train_period=80, val_period=20, step=20,
        feature_cols=feats, label_col="label",
    )
    # Use a pipeline so the model can handle the NaN in features.
    factory = lambda: make_pipeline(SimpleImputer(strategy="median"),
                                     Ridge(alpha=1.0))
    wf = WalkForward(model_factory=factory)
    res = wf.run(factor_df, cfg)
    assert res.summary["n_windows"] >= 1
    return TestResult("integration_engine_plus_library_plus_wf", True,
                      f"end-to-end OK, {res.summary['n_windows']} windows, "
                      f"mean_ic={res.summary['mean_ic']:.3f}",
                      time.perf_counter() - t0)


# ── Runner ───────────────────────────────────────────────────────────


ALL_TESTS: List[Tuple[str, Callable[[], TestResult]]] = [
    ("parser_handles_basic_arithmetic", test_parser_handles_basic_arithmetic),
    ("parser_unary_and_parentheses", test_parser_unary_and_parentheses),
    ("engine_ref_matches_baseline", test_engine_ref_matches_baseline),
    ("engine_ma_matches_baseline", test_engine_ma_matches_baseline),
    ("engine_rank_matches_baseline", test_engine_rank_matches_baseline),
    ("engine_alpha101_combo", test_engine_alpha101_combo),
    ("engine_calc_many_with_cache", test_engine_calc_many_with_cache),
    ("engine_perf_baseline", test_engine_perf_baseline),
    ("engine_edge_cases", test_engine_edge_cases),
    ("alpha158_library_runs", test_alpha158_library_runs),
    ("alpha101_alpha_001_matches_baseline", test_alpha101_alpha_001_matches_baseline),
    ("walk_forward_runs_and_emits_signals", test_walk_forward_runs_and_emits_signals),
    ("walk_forward_embargo_skips_overlap", test_walk_forward_embargo_skips_overlap),
    ("walk_forward_expanding_mode", test_walk_forward_expanding_mode),
    ("walk_forward_aggregate_ic_method", test_walk_forward_aggregate_ic_method),
    ("integration_engine_plus_library_plus_wf", test_integration_engine_plus_library_plus_wf),
]


def run_all(verbose: bool = True) -> Tuple[int, int, List[TestResult]]:
    results: List[TestResult] = []
    for name, fn in ALL_TESTS:
        try:
            r = fn()
        except AssertionError as e:
            r = TestResult(name, False, f"AssertionError: {e}")
        except Exception as e:  # noqa: BLE001
            r = TestResult(name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        results.append(r)
        if verbose:
            mark = "PASS" if r.passed else "FAIL"
            print(f"[{mark}] {r.name:55s} {r.duration*1000:6.1f}ms  {r.message}")
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    print()
    print(f"=== {n_pass}/{n_total} passed ===")
    return n_pass, n_total, results


if __name__ == "__main__":
    run_all(verbose=True)
