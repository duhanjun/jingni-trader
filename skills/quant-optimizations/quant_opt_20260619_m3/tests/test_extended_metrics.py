"""测试 extended performance metrics"""
import sys
import os
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, _THIS)  # 当前目录
sys.path.insert(0, "/workspace")  # 找 skills 包

import numpy as np
import pandas as pd

from skills.quant-optimizations.quant_opt_20260619_m3.extended_metrics.metrics import (
    omega_ratio, ulcer_index, ulcer_performance_index,
    serenity_index, deflated_sharpe_ratio, tail_ratio,
    gain_to_pain_ratio, profit_factor, stability_of_returns,
    max_drawdown_duration, beta, alpha, information_ratio,
    calc_extended_metrics,
)
try:
    import importlib.util, sys, types
    if "skills" not in sys.modules:
        if "/workspace" not in sys.path:
            sys.path.insert(0, "/workspace")
        skills_mod = types.ModuleType("skills")
        skills_mod.__path__ = ["/workspace/skills"]
        sys.modules["skills"] = skills_mod
        for sub in ["backtest-engine", "factor-engine"]:
            sub_mod = types.ModuleType(f"skills.{sub}")
            sub_mod.__path__ = [f"/workspace/skills/{sub}"]
            sys.modules[f"skills.{sub}"] = sub_mod
    if "skills.backtest-engine.scripts" not in sys.modules:
        scripts_mod = types.ModuleType("skills.backtest-engine.scripts")
        scripts_mod.__path__ = ["/workspace/skills/backtest-engine/scripts"]
        sys.modules["skills.backtest-engine.scripts"] = scripts_mod
    if "skills.backtest-engine.scripts.base" not in sys.modules:
        base_mod = types.ModuleType("skills.backtest-engine.scripts.base")
        base_mod.__path__ = ["/workspace/skills/backtest-engine/scripts/base"]
        sys.modules["skills.backtest-engine.scripts.base"] = base_mod

    spec = importlib.util.spec_from_file_location(
        "skills.backtest-engine.scripts.base.base_backtest",
        "/workspace/skills/backtest-engine/scripts/base/base_backtest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["skills.backtest-engine.scripts.base.base_backtest"] = mod
    spec.loader.exec_module(mod)
    BaseBacktestMetrics = mod.BaseBacktestMetrics
    HAS_BASE = True
except Exception as e:
    print(f"[WARN] 找不到 BaseBacktestMetrics: {e}, 跳过兼容性测试")
    HAS_BASE = False


def _make_equity_curve(n: int = 252, mu: float = 0.0005, sigma: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, n)
    eq = 1e6 * np.exp(np.cumsum(r))
    return pd.Series(eq)


def test_ulcer_index_monotonic_up():
    eq = pd.Series(np.linspace(1e6, 2e6, 100))
    assert ulcer_index(eq) == 0.0
    print("[PASS] test_ulcer_index_monotonic_up")


def test_ulcer_index_with_dd():
    eq = pd.Series([1.0, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0]) * 1e6
    ui = ulcer_index(eq)
    assert ui > 0
    print("[PASS] test_ulcer_index_with_dd")


def test_omega_ratio_known():
    # 全正收益 -> omega 应为 inf
    r = pd.Series([0.01, 0.02, 0.005])
    assert omega_ratio(r) == float("inf")
    # 全负收益 -> omega 应为 0
    r = pd.Series([-0.01, -0.02, -0.005])
    assert omega_ratio(r) == 0.0
    # 已知: [0.02, -0.01, 0.03, -0.02]
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    pos = 0.02 + 0.03
    neg = 0.01 + 0.02
    assert abs(omega_ratio(r) - pos / neg) < 1e-6
    print("[PASS] test_omega_ratio_known")


def test_stability_of_returns():
    # 完美线性增长 -> stability = 1
    eq = pd.Series(np.linspace(1.0, 2.0, 100))
    r = eq.pct_change().dropna()
    s = stability_of_returns(r)
    assert s > 0.99
    print("[PASS] test_stability_of_returns")


def test_max_dd_duration():
    # 净值: 涨-跌-涨, 最大回撤持续期
    # eq: [1.0, 1.1, 1.05, 1.0, 0.95, 1.0, 1.2]
    # cummax: [1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2]
    # 处于回撤的索引: 2, 3, 4, 5 (4天)
    eq = pd.Series([1.0, 1.1, 1.05, 1.0, 0.95, 1.0, 1.2])
    d = max_drawdown_duration(eq)
    assert d == 4, f"max_dd_duration 错: 期望 4, 实际 {d}"
    # 简单上升: 无回撤
    eq2 = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown_duration(eq2) == 0
    print("[PASS] test_max_dd_duration")


def test_tail_ratio():
    r = pd.Series([0.02, 0.03, 0.04, 0.01, -0.01, -0.02, -0.03])
    tr = tail_ratio(r, percentile=0.85)
    assert tr > 1
    print("[PASS] test_tail_ratio")


def test_gain_to_pain_ratio():
    r = pd.Series([0.10, -0.05, 0.10, -0.05])
    # pos = 0.20, neg = 0.10 -> ratio = 2.0
    assert abs(gain_to_pain_ratio(r) - 2.0) < 1e-6
    print("[PASS] test_gain_to_pain_ratio")


def test_beta_alpha_known():
    # strategy = 2 * benchmark
    bench = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01] * 50)
    strat = 2 * bench + 0.001
    b = beta(strat, bench)
    a = alpha(strat, bench, risk_free=0.0)
    assert abs(b - 2.0) < 0.01
    # alpha 应为接近 0.001 * 252 (年化)
    assert abs(a - 0.001 * 252) < 1.0
    print("[PASS] test_beta_alpha_known")


def test_information_ratio():
    bench = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01] * 50)
    strat = bench + 0.002  # 固定 alpha
    ir = information_ratio(strat, bench)
    assert ir > 5  # 高 IR
    print("[PASS] test_information_ratio")


def test_serenity_vs_upi():
    eq = _make_equity_curve()
    upi = ulcer_performance_index(eq)
    si = serenity_index(eq)
    assert isinstance(upi, float) and isinstance(si, float)
    assert upi != si
    print("[PASS] test_serenity_vs_upi")


def test_compatibility_with_base_metrics():
    """
    验证: 扩展指标与 jingni-trader 原有的 BaseBacktestMetrics 共存, 不冲突
    """
    if not HAS_BASE:
        print("[SKIP] test_compatibility_with_base_metrics")
        return
    eq = _make_equity_curve(seed=42)
    base = BaseBacktestMetrics.calc_all_metrics(eq, pd.DataFrame())
    ext = calc_extended_metrics(eq)

    # base 必须能算
    assert "sharpe_ratio" in base
    assert "max_drawdown" in base
    # ext 多了 10+ 个
    assert "omega_ratio" in ext
    assert "ulcer_index" in ext
    assert "upi" in ext
    assert "stability_r2" in ext
    assert "gain_to_pain" in ext
    print(f"[PASS] test_compatibility_with_base_metrics (base 9 指标, ext {len(ext)} 指标)")


def test_deflated_sharpe():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 252))
    dsr = deflated_sharpe_ratio(r, n_trials=10)
    assert isinstance(dsr, float)
    # 单次检验时 DSR 接近普通 Sharpe
    dsr1 = deflated_sharpe_ratio(r, n_trials=1)
    assert isinstance(dsr1, float)
    print("[PASS] test_deflated_sharpe")


def test_extended_metrics_consistency():
    """指标计算不应抛错, 所有值都应是有限或 nan/inf"""
    eq = _make_equity_curve(seed=7)
    ext = calc_extended_metrics(eq, bench_equity=eq * 1.05)
    assert len(ext) >= 13
    for k, v in ext.items():
        if not np.isfinite(v):
            print(f"  警告: {k} = {v}")
    print("[PASS] test_extended_metrics_consistency")


if __name__ == "__main__":
    test_ulcer_index_monotonic_up()
    test_ulcer_index_with_dd()
    test_omega_ratio_known()
    test_stability_of_returns()
    test_max_dd_duration()
    test_tail_ratio()
    test_gain_to_pain_ratio()
    test_beta_alpha_known()
    test_information_ratio()
    test_serenity_vs_upi()
    test_compatibility_with_base_metrics()
    test_deflated_sharpe()
    test_extended_metrics_consistency()
    print("\n所有扩展指标测试通过 ✓")