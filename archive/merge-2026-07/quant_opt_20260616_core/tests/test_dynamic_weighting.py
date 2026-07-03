"""
动态因子加权模块的测试
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_ic_history(n_days: int = 200, factors=("A", "B", "C"), seed: int = 0) -> pd.DataFrame:
    """构造 IC 序列: A 一直有效, B 中途失效, C 一直无效 (均值 0)"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    # A 一直有显著 IC
    a = rng.normal(0.05, 0.04, n_days)
    # B 前半段有效, 后半段失效
    b = np.concatenate([
        rng.normal(0.04, 0.04, n_days // 2),
        rng.normal(0.0, 0.05, n_days - n_days // 2),
    ])
    # C 一直无效 (均值 0, 噪声大)
    c = rng.normal(0.0, 0.05, n_days)
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)


def test_icir_decay_basic():
    from quant_opt_20260616_core.dynamic_weighting import icir_decay_weights
    ic = _make_ic_history()
    w = icir_decay_weights(ic, halflife=60, min_periods=20)
    assert set(w.keys()) == {"A", "B", "C"}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # A 应该权重最大
    assert w["A"] > w["B"] >= w["C"]
    print(f"  [OK] icir_decay_basic: w={w}")


def test_icir_decay_recent_bias():
    """最近表现好的因子应得到更高权重"""
    from quant_opt_20260616_core.dynamic_weighting import icir_decay_weights
    ic = _make_ic_history()
    w_long = icir_decay_weights(ic, halflife=2000, min_periods=20)
    w_short = icir_decay_weights(ic, halflife=10, min_periods=20)
    # 短半衰期: B 失活后的近期 IC 不进入加权 -> B 权重应低于长半衰期
    assert w_short["B"] < w_long["B"], (w_short, w_long)
    print(f"  [OK] recent bias: w_short={w_short}, w_long={w_long}")


def test_softmax_ic_weights():
    from quant_opt_20260616_core.dynamic_weighting import softmax_ic_weights
    ic = _make_ic_history()
    w = softmax_ic_weights(ic, lookback=200, temperature=0.01)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # A 的 IC 均值最大, 权重应最大
    assert w["A"] > w["B"]
    print(f"  [OK] softmax: w={w}")


def test_floor_floor_rebalance():
    from quant_opt_20260616_core.dynamic_weighting import icir_decay_weights
    ic = _make_ic_history()
    w_floor = icir_decay_weights(ic, halflife=60, min_periods=20, floor=0.1)
    w_no = icir_decay_weights(ic, halflife=60, min_periods=20, floor=0.0)
    # 总和仍为 1
    assert abs(sum(w_floor.values()) - 1.0) < 1e-6
    # 地板后, 原来 < 0.1 的因子被"拉起"到 0.1 再归一化
    # 所以小因子的相对占比会显著提升
    # 用一个比较极端的例子: 让 B 在不加地板时占比 0.01
    small_ic = pd.DataFrame(
        {"big": np.full(100, 0.05), "tiny": np.full(100, 0.001)},
        index=pd.bdate_range("2024-01-01", periods=100),
    )
    w_s = icir_decay_weights(small_ic, halflife=60, min_periods=20, floor=0.2)
    assert w_s["tiny"] >= 0.15, w_s  # 至少 0.2 * 1/(1+0.2) = 0.167 左右
    print(f"  [OK] floor: w={w_floor}; small case w_s={w_s}")


def test_dynamic_weighting_class():
    from quant_opt_20260616_core.dynamic_weighting import DynamicFactorWeighting
    ic = _make_ic_history()
    w1 = DynamicFactorWeighting(method="icir_decay").compute(ic)
    w2 = DynamicFactorWeighting(method="softmax_ic", lookback=120).compute(ic)
    assert w1 != w2
    print("  [OK] DynamicFactorWeighting class")


def test_empty_history():
    from quant_opt_20260616_core.dynamic_weighting import icir_decay_weights, softmax_ic_weights
    empty = pd.DataFrame()
    assert icir_decay_weights(empty) == {}
    assert softmax_ic_weights(empty) == {}
    print("  [OK] empty history")


def test_min_periods_filter():
    """样本数不足的因子应被剔除"""
    from quant_opt_20260616_core.dynamic_weighting import icir_decay_weights
    dates = pd.bdate_range("2024-01-01", periods=10)
    ic = pd.DataFrame({"A": np.zeros(10), "B": np.zeros(10)}, index=dates)
    w = icir_decay_weights(ic, min_periods=20)
    assert w == {}
    print("  [OK] min_periods filter")


def run() -> dict:
    test_icir_decay_basic()
    test_icir_decay_recent_bias()
    test_softmax_ic_weights()
    test_floor_floor_rebalance()
    test_dynamic_weighting_class()
    test_empty_history()
    test_min_periods_filter()
    return {"status": "passed", "cases": 7}


if __name__ == "__main__":
    run()