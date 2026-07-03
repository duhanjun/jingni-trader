"""
向量化因子计算器测试
"""
import numpy as np
import pandas as pd
import pytest

from validation.synth_data import make_synthetic_panel
from validation.vectorized_factor import (
    FACTOR_REGISTRY,
    LoopFactorCalculator,
    VectorizedFactorCalculator,
    benchmark,
)


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(n_stocks=10, n_days=200, seed=42)


def test_supported_factors_match():
    loop = LoopFactorCalculator()
    vec = VectorizedFactorCalculator()
    assert set(loop.get_available_factors()) == set(vec.get_available_factors())
    assert "rsi_14" in FACTOR_REGISTRY
    assert "ma_20" in FACTOR_REGISTRY
    assert "momentum_20d" in FACTOR_REGISTRY


def test_vectorized_matches_loop(panel):
    loop = LoopFactorCalculator()
    vec = VectorizedFactorCalculator()
    factor_list = ["ma_5", "ma_20", "rsi_14", "momentum_20d", "zscore_20"]
    out_loop = loop.calculate(panel, factor_list).sort_values(["code", "date"]).reset_index(drop=True)
    out_vec = vec.calculate(panel, factor_list).sort_values(["code", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        out_loop, out_vec, check_dtype=False, atol=1e-8, rtol=1e-6
    )


def test_empty_input_returns_empty():
    vec = VectorizedFactorCalculator()
    out = vec.calculate(pd.DataFrame(columns=["code", "date", "close"]), ["ma_5"])
    assert out.empty


def test_unknown_factor_raises(panel):
    vec = VectorizedFactorCalculator()
    with pytest.raises(ValueError):
        vec.calculate(panel, ["unknown_factor"])


def test_factor_info_direction():
    vec = VectorizedFactorCalculator()
    info = vec.get_factor_info("momentum_20d")
    assert info["direction"] == -1  # A 股反转因子


def test_benchmark_runs():
    panel = make_synthetic_panel(n_stocks=20, n_days=200, seed=1)
    factors = ["ma_5", "ma_20", "rsi_14", "std_20", "momentum_20d", "zscore_20"]
    res = benchmark(panel, factors, n_repeat=1)
    assert res["n_stocks"] == 20
    assert res["n_rows"] == 20 * 200
    assert res["loop_seconds"] > 0
    assert res["vec_seconds"] > 0
    assert res["speedup"] > 0
