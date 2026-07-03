"""
Test suite for the quant_opt module.

The tests are written as plain functions returning a list of
(name, passed, message) tuples so that we can run them in a
dependency-light environment and aggregate results.

Each test prints PASS / FAIL with a short message.
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

# allow running from anywhere
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from quant_opt import (  # noqa: E402
    evaluate_formula,
    parse_formula,
    list_operators,
    compute_ic_series,
    summarize_ic,
    batch_ic,
    rank_ic_decay,
    TimeSeriesCV,
    PurgedKFold,
    leakage_check,
    get_catalog,
    get_formula,
)


# ---------------------------------------------------------------------------
# Tiny test runner
# ---------------------------------------------------------------------------

TestResult = Tuple[str, bool, str]
TestFn = Callable[[], None]


def _make_synthetic_data(n_dates: int = 60, n_codes: int = 20,
                          seed: int = 42) -> pd.DataFrame:
    """Build a synthetic OHLCV panel for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    codes = [f"T{i:04d}.SH" for i in range(n_codes)]
    rows = []
    for code in codes:
        base = 10 + rng.normal() * 2
        drift = rng.normal() * 0.001
        vol = 0.02
        rets = rng.normal(drift, vol, size=n_dates)
        prices = base * np.exp(np.cumsum(rets))
        for i, d in enumerate(dates):
            close = prices[i]
            high = close * (1 + abs(rng.normal(0, 0.005)))
            low = close * (1 - abs(rng.normal(0, 0.005)))
            open_ = close * (1 + rng.normal(0, 0.003))
            volume = abs(rng.normal(1e6, 2e5))
            amount = volume * close
            rows.append({
                "code": code, "date": d,
                "open": open_, "high": high, "low": low,
                "close": close, "volume": volume, "amount": amount,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: Expression Engine
# ---------------------------------------------------------------------------

def test_tokenizer_basic():
    """Function names are tokenized as NAME; '$close' as FIELD; digits as NUM;
    punctuation as OP. The order of token types matches our grammar."""
    from quant_opt.expression_engine.expr_engine import tokenize
    out = tokenize("Ref($close, 5)")
    expected = [
        ("NAME",  "Ref"),
        ("OP",    "("),
        ("FIELD", "$close"),
        ("OP",    ","),
        ("NUM",   "5"),
        ("OP",    ")"),
    ]
    assert out == expected, f"expected {expected}, got {out}"


def test_parser_simple_binary():
    ast = parse_formula("$close / Ref($close, 5) - 1")
    # root: -
    from quant_opt.expression_engine.expr_engine import BinaryOp, FieldRef, FuncCall, Number
    assert isinstance(ast, BinaryOp) and ast.name == "-"
    assert isinstance(ast.left, BinaryOp) and ast.left.name == "/"


def test_parser_unary_negation():
    ast = parse_formula("-($close / Ref($close, 5) - 1)")
    from quant_opt.expression_engine.expr_engine import UnaryOp
    assert isinstance(ast, UnaryOp) and ast.name == "-"


def test_parser_nested_calls():
    ast = parse_formula("Mean(Mean($close, 5), 10)")
    from quant_opt.expression_engine.expr_engine import FuncCall
    assert isinstance(ast, FuncCall) and ast.name == "Mean"
    assert isinstance(ast.args[0], FuncCall) and ast.args[0].name == "Mean"


def test_parser_invalid_should_raise():
    raised = False
    try:
        parse_formula("Mean($close, 5")  # missing closing paren
    except Exception:
        raised = True
    assert raised, "parser should raise on unclosed parenthesis"


def test_evaluator_momentum():
    df = _make_synthetic_data()
    out = evaluate_formula("Ref($close, 5) / $close - 1", df)
    assert isinstance(out, pd.Series)
    assert out.shape[0] == df.shape[0]
    # not all NaN
    assert out.notna().sum() > 0


def test_evaluator_ma_ratio():
    df = _make_synthetic_data()
    out = evaluate_formula("Mean($close, 5) / Mean($close, 20) - 1", df)
    # all results should be finite (ignoring NaN)
    finite = np.isfinite(out.dropna())
    assert bool(finite.all())


def test_evaluator_cross_sectional_rank():
    df = _make_synthetic_data()
    out = evaluate_formula("Rank($close)", df)
    # per-date ranks should be in (0, 1]
    grouped = out.groupby(level="date")
    for dt, vals in grouped:
        v = vals.dropna()
        assert v.between(0, 1).all(), f"rank values out of bounds on {dt}"


def test_evaluator_log_return():
    df = _make_synthetic_data()
    out = evaluate_formula("Log($close / Ref($close, 1))", df)
    # for any given code, log returns should be continuous after first row
    sample_code = df["code"].iloc[0]
    series = out.xs(sample_code)
    assert series.iloc[1:].notna().all()


def test_evaluator_signed_power():
    df = _make_synthetic_data()
    out = evaluate_formula("SignedPower($close / Ref($close, 5) - 1, 0.5)", df)
    assert out.notna().sum() > 0


def test_operator_registry_known_ops():
    ops = set(list_operators())
    for must_have in ("Ref", "Mean", "Std", "Rank", "Zscore", "Scale", "Log"):
        assert must_have in ops, f"missing operator {must_have}"


def test_alpha_catalog_unique():
    catalog = get_catalog()
    names = [n for n, _ in catalog]
    assert len(names) == len(set(names)), "duplicate names in catalog"
    for n, f in catalog:
        # every formula should parse
        parse_formula(f)


def test_alpha_catalog_all_evaluable():
    df = _make_synthetic_data(n_dates=120, n_codes=20)  # need ≥60 for REV_60
    catalog = get_catalog()
    for name, formula in catalog:
        out = evaluate_formula(formula, df)
        # a few non-nan values are enough
        assert out.notna().sum() > 5, f"{name} produced too few values"


# ---------------------------------------------------------------------------
# Tests: Vectorized IC
# ---------------------------------------------------------------------------

def test_ic_summary_monotone():
    """IC of a noisy factor should be near zero; of a strongly predictive
    factor should be larger in magnitude."""
    rng = np.random.default_rng(0)
    n = 1000
    target = pd.Series(rng.normal(size=n))
    noise = pd.Series(rng.normal(size=n))
    signal = target * 2 + noise * 0.1
    factor = pd.Series(signal, name="f")
    target_s = pd.Series(target, name="t")
    dates = pd.date_range("2024-01-01", periods=10).repeat(100)
    codes = np.tile(np.arange(100), 10)
    idx = pd.MultiIndex.from_arrays([codes, dates], names=["code", "date"])
    factor.index = idx
    target_s.index = idx

    ic_series = compute_ic_series(factor, target_s, method="spearman")
    s = summarize_ic(ic_series)
    assert s["n_periods"] == 10
    # |IC| should be much larger than zero
    assert abs(s["ic_mean"]) > 0.3, f"IC should be strong, got {s}"


def test_ic_pearson_equals_spearman_for_linear():
    """For a perfectly linear relationship, pearson and spearman should match."""
    rng = np.random.default_rng(1)
    n = 500
    target = pd.Series(rng.normal(size=n))
    factor = pd.Series(target * 1.5 + 0.01, name="f")
    dates = pd.date_range("2024-01-01", periods=10).repeat(50)
    codes = np.tile(np.arange(50), 10)
    idx = pd.MultiIndex.from_arrays([codes, dates], names=["code", "date"])
    factor.index = idx
    target.index = idx

    p = compute_ic_series(factor, target, method="pearson").mean()
    s = compute_ic_series(factor, target, method="spearman").mean()
    assert abs(p - s) < 0.05, f"pearson={p} spearman={s}"


def test_batch_ic_returns_dict():
    rng = np.random.default_rng(2)
    n = 200
    dates = pd.date_range("2024-01-01", periods=10).repeat(20)
    codes = np.tile(np.arange(20), 10)
    idx = pd.MultiIndex.from_arrays([codes, dates], names=["code", "date"])
    target = pd.Series(rng.normal(size=n), index=idx)
    f1 = target * 0.5
    f2 = pd.Series(rng.normal(size=n), index=idx)
    df = pd.DataFrame({"good": f1, "noise": f2}, index=idx)
    out = batch_ic(df, target, method="spearman")
    assert "good" in out and "noise" in out
    # the 'good' factor should have higher |IC|
    assert abs(out["good"]["ic_mean"]) > abs(out["noise"]["ic_mean"])


def test_rank_ic_decay_shape():
    rng = np.random.default_rng(3)
    n = 200
    dates = pd.date_range("2024-01-01", periods=10).repeat(20)
    codes = np.tile(np.arange(20), 10)
    idx = pd.MultiIndex.from_arrays([codes, dates], names=["code", "date"])
    target = pd.Series(rng.normal(size=n), index=idx)
    factor = pd.Series(target * 0.5, index=idx)
    decay = rank_ic_decay(factor, target, max_lag=5)
    assert len(decay) == 5
    assert decay.index[0] == 1 and decay.index[-1] == 5


# ---------------------------------------------------------------------------
# Tests: Time-series CV + Purged K-Fold + Leakage check
# ---------------------------------------------------------------------------

def test_ts_cv_basic():
    dates = pd.bdate_range("2024-01-01", periods=100)
    cv = TimeSeriesCV(train_size=30, valid_size=10, test_size=10, step=10)
    splits = list(cv.split(dates))
    assert len(splits) > 0
    for s in splits:
        assert len(s.train) == 30
        assert len(s.valid) == 10
        assert len(s.test) == 10
        # ensure ordering
        assert s.train_period[1] < s.valid_period[0] < s.test_period[0]


def test_ts_cv_purge_gap():
    dates = pd.bdate_range("2024-01-01", periods=100)
    cv = TimeSeriesCV(train_size=20, valid_size=5, test_size=5,
                      step=5, purge_gap=3)
    for s in cv.split(dates):
        # gap between train end and valid start must be >= 3 days
        gap = (s.valid_period[0] - s.train_period[1]).days
        assert gap >= 3


def test_purged_kfold_no_overlap():
    pkf = PurgedKFold(n_splits=5, purge_gap=2)
    for fold in pkf.split(50):
        assert set(fold.train_idx).isdisjoint(set(fold.test_idx))


def test_purged_kfold_purge_respects_gap():
    pkf = PurgedKFold(n_splits=5, purge_gap=3)
    n = 50
    for fold in pkf.split(n):
        # any train index within 3 positions of a test index should be excluded
        for t in fold.test_idx:
            for j in (t - 3, t - 2, t - 1, t + 1, t + 2, t + 3):
                if 0 <= j < n:
                    assert j not in set(fold.train_idx), \
                        f"purge gap violated: train idx {j} too close to test {t}"


def test_leakage_check_clean():
    df = _make_synthetic_data(n_dates=30, n_codes=10)
    rep = leakage_check(df)
    assert rep.is_clean, f"clean data should pass, got {rep}"


def test_leakage_check_dups():
    df = _make_synthetic_data(n_dates=30, n_codes=10)
    # duplicate a row
    dup_row = df.iloc[[0]]
    df = pd.concat([df, dup_row], ignore_index=True)
    rep = leakage_check(df)
    assert not rep.is_clean
    assert any("Duplicate" in i for i in rep.issues)


def test_leakage_check_future_dated():
    df = _make_synthetic_data(n_dates=30, n_codes=10)
    df.loc[0, "date"] = pd.Timestamp("2099-01-01")
    rep = leakage_check(df)
    assert not rep.is_clean
    assert any("Future-dated" in i for i in rep.issues)


def test_leakage_check_perfect_corr():
    df = _make_synthetic_data(n_dates=30, n_codes=10)
    df["ret_forward_1d"] = df["close"]  # garbage but correlated
    rep = leakage_check(df, feature_cols=["close"])
    assert not rep.is_clean


# ---------------------------------------------------------------------------
# Test aggregator
# ---------------------------------------------------------------------------

ALL_TESTS: List[Tuple[str, TestFn]] = [
    ("tokenizer_basic", test_tokenizer_basic),
    ("parser_simple_binary", test_parser_simple_binary),
    ("parser_unary_negation", test_parser_unary_negation),
    ("parser_nested_calls", test_parser_nested_calls),
    ("parser_invalid_should_raise", test_parser_invalid_should_raise),
    ("evaluator_momentum", test_evaluator_momentum),
    ("evaluator_ma_ratio", test_evaluator_ma_ratio),
    ("evaluator_cross_sectional_rank", test_evaluator_cross_sectional_rank),
    ("evaluator_log_return", test_evaluator_log_return),
    ("evaluator_signed_power", test_evaluator_signed_power),
    ("operator_registry_known_ops", test_operator_registry_known_ops),
    ("alpha_catalog_unique", test_alpha_catalog_unique),
    ("alpha_catalog_all_evaluable", test_alpha_catalog_all_evaluable),
    ("ic_summary_monotone", test_ic_summary_monotone),
    ("ic_pearson_equals_spearman_for_linear", test_ic_pearson_equals_spearman_for_linear),
    ("batch_ic_returns_dict", test_batch_ic_returns_dict),
    ("rank_ic_decay_shape", test_rank_ic_decay_shape),
    ("ts_cv_basic", test_ts_cv_basic),
    ("ts_cv_purge_gap", test_ts_cv_purge_gap),
    ("purged_kfold_no_overlap", test_purged_kfold_no_overlap),
    ("purged_kfold_purge_respects_gap", test_purged_kfold_purge_respects_gap),
    ("leakage_check_clean", test_leakage_check_clean),
    ("leakage_check_dups", test_leakage_check_dups),
    ("leakage_check_future_dated", test_leakage_check_future_dated),
    ("leakage_check_perfect_corr", test_leakage_check_perfect_corr),
]


def run_all() -> Tuple[int, int, List[TestResult]]:
    results: List[TestResult] = []
    for name, fn in ALL_TESTS:
        t0 = time.time()
        try:
            fn()
            dt = (time.time() - t0) * 1000
            results.append((name, True, f"ok ({dt:.1f} ms)"))
        except AssertionError as e:
            results.append((name, False, f"assert: {e}"))
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=1)
            results.append((name, False, f"{type(e).__name__}: {e}\n{tb}"))
    passed = sum(1 for _, ok, _ in results if ok)
    return passed, len(results), results


if __name__ == "__main__":
    p, total, results = run_all()
    for name, ok, msg in results:
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] {name:<42s} {msg}")
    print(f"\n{p}/{total} tests passed")
    sys.exit(0 if p == total else 1)
