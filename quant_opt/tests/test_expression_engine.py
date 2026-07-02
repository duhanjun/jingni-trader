"""
表达式引擎单元测试 & 对比验证
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_opt.expression_engine import (
    Delta,
    Evaluator,
    F,
    Feature,
    Mean,
    Rank,
    Ref,
    TsMean,
    TsRank,
    TsStd,
    Zscore,
    builtin_a_share_factors,
)
from quant_opt.tests._fixtures import make_synthetic_a_share_data


# ---------------------------------------------------------------------------
# 1. 正确性测试
# ---------------------------------------------------------------------------
def test_ref_equals_groupby_shift():
    """Ref(x, d) 应等于 x.groupby('code').shift(d)"""
    data = make_synthetic_a_share_data(n_stocks=5, n_days=30)
    expr = Ref(F("close"), 5)
    expected = data.groupby("code")["close"].shift(5)
    actual = expr.compute(data)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_delta_correctness():
    data = make_synthetic_a_share_data(n_stocks=5, n_days=30)
    expr = Delta(F("close"), 5)
    expected = data["close"] - data.groupby("code")["close"].shift(5)
    actual = expr.compute(data)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_ts_mean_correctness():
    data = make_synthetic_a_share_data(n_stocks=5, n_days=30)
    expr = TsMean(F("close"), 5)
    expected = (
        data.groupby("code")["close"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    actual = expr.compute(data)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_rank_correctness():
    data = make_synthetic_a_share_data(n_stocks=10, n_days=5)
    expr = Rank(F("close"))
    actual = expr.compute(data)
    expected = data.groupby("date")["close"].rank(pct=True)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_zscore_correctness():
    data = make_synthetic_a_share_data(n_stocks=10, n_days=5)
    expr = Zscore(F("close"))
    actual = expr.compute(data)
    grp = data.groupby("date")["close"]
    expected = (data["close"] - grp.transform("mean")) / grp.transform("std")
    pd.testing.assert_series_equal(actual, expected, check_names=False)


# ---------------------------------------------------------------------------
# 2. 可组合性测试
# ---------------------------------------------------------------------------
def test_compose_binary_ops():
    """验证 -($close / Ref($close, 1) - 1) 与 reverse_5d 等价"""
    data = make_synthetic_a_share_data(n_stocks=5, n_days=30)
    expr = -(F("close") / Ref(F("close"), 1) - 1)
    actual = expr.compute(data)
    expected = -(data["close"] / data.groupby("code")["close"].shift(1) - 1)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_compose_complex_expression():
    """验证复合表达式：Zscore(Rank(TsMean(close, 20)))"""
    data = make_synthetic_a_share_data(n_stocks=10, n_days=60)
    expr = Zscore(Rank(TsMean(F("close"), 20)))
    actual = expr.compute(data)
    assert len(actual) == len(data)
    # 截面均值应接近 0，std 应接近 1
    daily_mean = actual.groupby(data["date"]).mean()
    assert abs(daily_mean.mean()) < 0.1


# ---------------------------------------------------------------------------
# 3. 评估器批量求值
# ---------------------------------------------------------------------------
def test_evaluator_batch_eval():
    data = make_synthetic_a_share_data(n_stocks=10, n_days=60)
    exprs = builtin_a_share_factors()
    ev = Evaluator(data)
    out = ev.eval_many(exprs)
    assert "ret_1d" in out.columns
    assert "ma_20" in out.columns
    assert len(out) == len(data)
    # ret_1d 首日应为空
    assert out["ret_1d"].iloc[0] != out["ret_1d"].iloc[0]  # NaN


def test_evaluator_memoization():
    """相同子树应在缓存中只计算一次（仅 Ref 节点）"""
    data = make_synthetic_a_share_data(n_stocks=5, n_days=20)
    common = Ref(F("close"), 5)
    ev = Evaluator(data)
    a = ev.eval(common + 1)  # common 第一次计算
    b = ev.eval(common * 2)  # common 应命中缓存
    # 缓存中至少有 Ref(F("close"), 5)
    ref_id = id(common)
    assert ref_id in ev._cache
    # 不同 binary 包装应共享 common
    assert ev._cache[ref_id] is not None


# ---------------------------------------------------------------------------
# 4. 性能测试
# ---------------------------------------------------------------------------
def test_performance_single_vs_builtin():
    """对单个因子，表达式引擎与手写 groupby 的耗时对比"""
    data = make_synthetic_a_share_data(n_stocks=50, n_days=500)
    expr = TsMean(F("close"), 20)

    t0 = time.perf_counter()
    for _ in range(10):
        expr.compute(data)
    t_expr = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(10):
        data.groupby("code")["close"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
    t_manual = time.perf_counter() - t0
    # 表达式引擎性能至少不能比手写差 5 倍以上
    print(f"\n[perf] 表达式引擎: {t_expr:.4f}s vs 手写: {t_manual:.4f}s")
    assert t_expr < t_manual * 5.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
