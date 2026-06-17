"""
因子注册表测试
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_opt.factor_registry import (
    REGISTRY,
    FactorDirection,
    FactorRegistry,
    FactorSpec,
    register_factor,
)
from quant_opt.tests._fixtures import make_synthetic_a_share_data


def test_builtin_factors_registered():
    assert "ret_1d" in REGISTRY
    assert "volatility_20d" in REGISTRY
    assert "ma_deviation_20" in REGISTRY


def test_register_decorator_basic():
    @register_factor(
        "test_my_factor",
        description="测试因子",
        direction=FactorDirection.POSITIVE,
        category="test",
        tags=["demo"],
    )
    def my_factor(data):
        return data["close"] / data["open"] - 1

    assert "test_my_factor" in REGISTRY
    spec = REGISTRY.get("test_my_factor")
    assert spec.direction == FactorDirection.POSITIVE
    assert spec.category == "test"
    assert "demo" in spec.tags


def test_register_duplicate_raises():
    with pytest.raises(ValueError):
        @register_factor("ret_1d")
        def _dummy(data):
            return data["close"]


def test_filter_by_category():
    mom_factors = REGISTRY.list_by_category("momentum")
    assert "ret_1d" in mom_factors
    assert "ret_20d" in mom_factors


def test_filter_by_tag():
    classic = REGISTRY.list_by_tag("classic")
    assert "reversal_5d" in classic
    assert "reversal_20d" in classic


def test_compute_single():
    data = make_synthetic_a_share_data(n_stocks=10, n_days=50)
    s = REGISTRY.compute("ret_1d", data)
    assert len(s) == len(data)
    # 第一只股票首日应为空
    first_stock = data["code"].iloc[0]
    first_idx = data[data["code"] == first_stock].index[0]
    assert np.isnan(s.iloc[first_idx])


def test_compute_many_with_shared_evaluator():
    """compute_many 在共享 Evaluator 时应能复用子表达式缓存"""
    data = make_synthetic_a_share_data(n_stocks=10, n_days=50)
    out = REGISTRY.compute_many(
        ["ret_1d", "ret_5d", "reversal_5d", "expr_reversal_20d"],
        data,
        shared_evaluator=True,
    )
    assert "ret_1d" in out.columns
    assert "expr_reversal_20d" in out.columns
    # reversal_5d = -ret_5d
    np.testing.assert_array_equal(
        out["reversal_5d"].fillna(0).values,
        (-out["ret_5d"]).fillna(0).values,
    )


def test_override_params():
    data = make_synthetic_a_share_data(n_stocks=10, n_days=100)
    s_default = REGISTRY.compute("ma_deviation_20", data)
    s_window5 = REGISTRY.compute("ma_deviation_20", data, window=5)
    # 两种窗口结果应不同
    assert not np.allclose(
        s_default.fillna(0).values,
        s_window5.fillna(0).values,
    )


def test_summary():
    df = REGISTRY.summary()
    assert "name" in df.columns
    assert len(df) == len(REGISTRY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
