"""
参数化扫描测试
"""
import numpy as np
import pandas as pd
import pytest

from validation.parameterized import parameterized, sweep, chunked


def test_sweep_basic_grid():
    def f(x, y):
        return x ** 2 + y

    res = sweep(f, {"x": [1, 2, 3], "y": [10, 20]})
    assert len(res.grid) == 6
    assert res.values[0] == 1 + 10
    assert res.values[-1] == 9 + 20
    # best(): 寻找最大
    best = res.best(higher_is_better=True)
    assert best["params"]["x"] == 3
    assert best["params"]["y"] == 20


def test_sweep_common_kwargs():
    def f(x, y, scale=1.0):
        return scale * (x + y)

    res = sweep(f, {"x": [1, 2]}, common_kwargs={"y": 5, "scale": 2.0})
    assert res.values[0] == 12.0
    assert res.values[-1] == 14.0


def test_sweep_chunk_size():
    def f(x):
        return x

    res = sweep(f, {"x": list(range(20))}, chunk_size=5, n_jobs=1)
    assert len(res.grid) == 20
    assert res.values[-1] == 19


def test_sweep_empty_param_raises():
    def f(x):
        return x

    with pytest.raises(ValueError):
        sweep(f, {})


def test_parameterized_decorator():
    @parameterized({"window": [5, 10, 20]})
    def rolling_mean(series, window):
        return float(pd.Series(series).rolling(window).mean().iloc[-1])

    series = list(range(50))
    res = rolling_mean(series=series)
    assert len(res.grid) == 3
    assert res.values[0] == pytest.approx(np.mean(series[-5:]))
    assert res.values[-1] == pytest.approx(np.mean(series[-20:]))


def test_parameterized_with_common():
    @parameterized({"x": [1, 2, 3]})
    def double_with_offset(x, offset):
        return 2 * x + offset

    res = double_with_offset(offset=5)
    assert len(res.grid) == 3
    # 验证 sweep 输出可用
    assert res.values[0] == 2 * 1 + 5
    assert res.values[-1] == 2 * 3 + 5


def test_chunked_decorator_attribute():
    @chunked(chunk_len=10)
    def f(x):
        return x

    assert hasattr(f, "_vbt_chunk_len")
    assert f._vbt_chunk_len == 10


def test_sweep_result_to_dataframe():
    def f(x):
        return x

    res = sweep(f, {"x": [1, 2, 3]})
    df = res.to_dataframe("v")
    assert len(df) == 3
    assert "x" in df.columns
    assert "v_0" in df.columns


def test_sweep_result_best_min():
    def f(x):
        return (x - 3) ** 2  # 最小在 x=3

    res = sweep(f, {"x": [0, 1, 2, 3, 4, 5]})
    best = res.best(higher_is_better=False)
    assert best["params"]["x"] == 3
    assert best["value"] == 0.0
