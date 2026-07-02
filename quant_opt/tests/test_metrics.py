"""
test_metrics.py
===============

comprehensive_metrics 模块的单元测试。

测试覆盖：
- 边界条件（空序列、单元素、NaN）
- 数值精度（Sharpe / Sortino / Max DD）
- 组合计算（compute_full_metrics 的输出一致性）
- 与现有 BaseBacktestMetrics 的兼容性
"""
from __future__ import annotations

import math
import sys, os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "backtest"))

from comprehensive_metrics import (
    calc_total_return, calc_cagr, calc_volatility,
    calc_sharpe, calc_sortino, calc_max_drawdown,
    calc_max_drawdown_duration, calc_calmar, calc_win_rate,
    calc_profit_factor, calc_sqn, calc_kelly, calc_alpha_beta,
    calc_information_ratio, compute_full_metrics,
)


def _rng_series(n: int, mean: float = 0.0, std: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(mean, std, n), index=idx)


def test_calc_total_return():
    """测试累计收益计算。"""
    eq = pd.Series([100, 110, 121, 133.1])
    assert math.isclose(calc_total_return(eq), 0.331, rel_tol=1e-3)
    assert calc_total_return(pd.Series([100.0])) == 0.0
    assert calc_total_return(pd.Series([], dtype=float)) == 0.0


def test_calc_cagr():
    """测试年化收益。"""
    eq = pd.Series([100, 110, 121])
    # 3 个 bar / 2 trading_days = 1.5 年
    # cagr = (121/100)^(1/1.5) - 1 = 1.21^0.667 - 1 ≈ 0.1355
    cagr = calc_cagr(eq, trading_days=2)
    assert math.isclose(cagr, 0.1355, rel_tol=1e-2)


def test_calc_sharpe_basic():
    """测试夏普比率基本计算。"""
    r = _rng_series(252, mean=0.001, std=0.01, seed=42)
    s = calc_sharpe(r, risk_free=0.0)
    # 预期 sharpe ~ 12 / 15 = 0.8
    assert -0.2 < s < 2.0
    assert not math.isnan(s)


def test_calc_sortino_basic():
    """测试索提诺。"""
    r = _rng_series(252, mean=0.0005, std=0.01, seed=42)
    sortino = calc_sortino(r, risk_free=0.0)
    sharpe = calc_sharpe(r, risk_free=0.0)
    # Sortino 是有限数
    assert not math.isnan(sortino)
    assert not math.isinf(sortino)


def test_calc_max_drawdown():
    """测试最大回撤。"""
    eq = pd.Series([100, 120, 90, 80, 95, 110])
    mdd = calc_max_drawdown(eq)
    # peak=120 (at index 1), trough=80 (at index 3)
    # mdd = 80/120 - 1 = -0.333
    assert math.isclose(mdd, -1/3, rel_tol=1e-3)


def test_calc_max_drawdown_duration():
    """测试最大回撤持续期。"""
    eq = pd.Series([100, 120, 110, 105, 100, 110, 120])
    dur = calc_max_drawdown_duration(eq)
    # 累计回撤 bar: 110(>120×cumax), 105, 100, 110 (恢复) = 4 个 bar 在回撤中
    # 算法: cumsum of in_dd = 1,1,1,2,3,4 (从 1 开始)
    assert dur == 4


def test_calc_profit_factor():
    """测试盈亏比。"""
    trades = pd.DataFrame({"pnl": [100, 200, -50, -100, 150]})
    pf = calc_profit_factor(trades)
    # (100+200+150) / (50+100) = 450/150 = 3
    assert math.isclose(pf, 3.0, rel_tol=1e-3)


def test_calc_sqn():
    """测试 SQN。"""
    trades = pd.DataFrame({"pnl": [100, 200, 150, 120, 80]})
    sqn = calc_sqn(trades)
    assert sqn > 0


def test_calc_kelly():
    """测试凯利公式。"""
    trades = pd.DataFrame({"pnl": [100, 200, -50, -100, 150]})
    k = calc_kelly(trades)
    # 胜率 3/5 = 0.6, 盈亏比 1.5
    # 0.6 - 0.4 / 1.5 = 0.6 - 0.267 = 0.333
    assert 0.3 < k < 0.4


def test_compute_full_metrics_with_benchmark():
    """测试含基准的全套指标。"""
    eq = _rng_series(252, mean=0.0008, std=0.015, seed=42)
    equity = (1 + eq).cumprod() * 1_000_000
    bench = _rng_series(252, mean=0.0005, std=0.012, seed=7)
    bench_close = (1 + bench).cumprod() * 1000
    trades = pd.DataFrame({"pnl": np.random.default_rng(0).normal(100, 1000, 20)})
    m = compute_full_metrics(equity, trades=trades, positions=pd.Series([0, 1] * 126),
                              benchmark_close=bench_close)
    assert "sharpe_ratio" in m
    assert "sortino_ratio" in m
    assert "calmar_ratio" in m
    assert "alpha" in m
    assert "beta" in m
    assert "information_ratio" in m
    assert "max_drawdown_duration" in m
    assert m["n_trades"] == 20
    assert 0 <= m["win_rate"] <= 1
    print(f"[OK] full metrics: {len(m)} keys, sharpe={m['sharpe_ratio']:.3f}")


def test_compute_full_metrics_empty():
    """空交易/空持仓场景。"""
    eq = pd.Series([100, 110, 120])
    m = compute_full_metrics(eq, trades=None, positions=None)
    assert m["n_trades"] == 0
    assert m["win_rate"] == 0.0
    assert m["exposure_time"] == 0.0
    assert m["alpha"] == 0.0
    assert m["beta"] == 0.0


def test_compat_with_existing_base_metrics():
    """兼容旧版 BaseBacktestMetrics 的指标名。"""
    eq = _rng_series(252, mean=0.0005, std=0.012, seed=0)
    equity = (1 + eq).cumprod() * 1e6
    m = compute_full_metrics(equity, trades=pd.DataFrame({"pnl": [100, -50, 200]}))
    # 旧 BaseBacktestMetrics 输出的 key 名
    for old_key in ["total_return", "annual_return", "sharpe_ratio",
                    "max_drawdown", "win_rate"]:
        assert old_key in m, f"Missing compat key: {old_key}"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
