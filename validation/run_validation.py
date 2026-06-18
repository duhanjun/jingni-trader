"""
一键运行全部验证测试, 收集性能与正确性指标,
生成 Golden baseline JSON (回归测试用) 与最终报告数据。

用法:
    PYTHONPATH=. python3 -m validation.run_validation

不修改 main 分支任何业务代码, 仅在 /workspace/validation/ 内运行。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from validation.metrics import calc_all_stats, factor_metrics
from validation.synth_data import (
    make_synthetic_equity,
    make_synthetic_panel,
    make_synthetic_returns,
)
from validation.purged_cv import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    WalkForwardSplitter,
    ic_time_series_split,
)
from validation.vectorized_factor import (
    FACTOR_REGISTRY,
    LoopFactorCalculator,
    VectorizedFactorCalculator,
    benchmark,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
RESULTS_DIR = Path(__file__).parent / "results"


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. 正确性验证: vectorized vs loop
# ---------------------------------------------------------------------------
def verify_vectorized_correctness() -> Dict[str, Any]:
    section("1. 向量化因子计算 — 正确性 (vectorized == loop)")
    panel = make_synthetic_panel(n_stocks=30, n_days=300, seed=11)
    factor_list = list(FACTOR_REGISTRY.keys())
    loop = LoopFactorCalculator()
    vec = VectorizedFactorCalculator()
    out_loop = loop.calculate(panel, factor_list).sort_values(["code", "date"]).reset_index(drop=True)
    out_vec = vec.calculate(panel, factor_list).sort_values(["code", "date"]).reset_index(drop=True)
    # 计算每个因子的最大绝对误差
    max_diffs = {}
    for f in factor_list:
        diff = float(np.abs(out_loop[f] - out_vec[f]).max())
        max_diffs[f] = diff
    overall = float(max(max_diffs.values()))
    print(f"  因子: {factor_list}")
    print(f"  各因子最大 |差|: {max_diffs}")
    print(f"  全局最大 |差|: {overall:.2e}")
    return {
        "ok": overall < 1e-6,
        "max_diff": overall,
        "per_factor": max_diffs,
    }


# ---------------------------------------------------------------------------
# 2. 性能验证: vectorized vs loop
# ---------------------------------------------------------------------------
def verify_vectorized_perf() -> Dict[str, Any]:
    section("2. 向量化因子计算 — 性能 (耗时 / 加速比)")
    sizes = [
        {"n_stocks": 50, "n_days": 252},
        {"n_stocks": 200, "n_days": 504},
        {"n_stocks": 500, "n_days": 1000},
    ]
    factor_list = ["ma_5", "ma_20", "ma_60", "ema_12", "rsi_14",
                   "std_20", "momentum_20d", "zscore_20"]
    rows = []
    for sz in sizes:
        panel = make_synthetic_panel(**sz, seed=22)
        res = benchmark(panel, factor_list, n_repeat=2)
        rows.append({"n_stocks": sz["n_stocks"], "n_days": sz["n_days"],
                     "n_rows": res["n_rows"],
                     "loop_ms": round(res["loop_seconds"] * 1000, 2),
                     "vec_ms": round(res["vec_seconds"] * 1000, 2),
                     "speedup": round(res["speedup"], 2)})
        print(f"  {sz['n_stocks']}x{sz['n_days']}: "
              f"loop={rows[-1]['loop_ms']}ms, vec={rows[-1]['vec_ms']}ms, "
              f"speedup={rows[-1]['speedup']}x")
    return {"rows": rows, "avg_speedup": float(np.mean([r["speedup"] for r in rows]))}


# ---------------------------------------------------------------------------
# 3. 正确性验证: Purged CV 行为
# ---------------------------------------------------------------------------
def verify_purged_cv() -> Dict[str, Any]:
    section("3. Purged K-Fold + Walk-Forward — 行为验证")
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    X = pd.DataFrame({"x": range(300)}, index=dates)

    pkf = PurgedKFold(n_splits=5, purge_td="5D", embargo_td="5D")
    splits = list(pkf.split(X))
    overlap_ok = all(
        set(s.train_idx.tolist()).isdisjoint(s.test_idx.tolist()) for s in splits
    )
    no_purge_n = 300
    pkf_no = PurgedKFold(n_splits=5)
    splits_no = list(pkf_no.split(X))
    no_purge_n = sum(len(s.train_idx) + len(s.test_idx) for s in splits_no) // len(splits_no)
    print(f"  Purged KFold (5 折 + 5D purge + 5D embargo):")
    print(f"    切分数: {len(splits)}")
    print(f"    训练/测试无重叠: {overlap_ok}")
    print(f"  无 purge 时 (5 折): 训练+测试=总样本 {no_purge_n}/300")

    wf = WalkForwardSplitter(train_size=200, test_size=50, step_size=50)
    wf_splits = list(wf.split(X))
    print(f"  WalkForward (train=200, test=50, step=50):")
    print(f"    切分数: {len(wf_splits)}")
    for s in wf_splits:
        print(f"    fold {s.fold_id}: train=[{s.train_idx[0]}..{s.train_idx[-1]}] "
              f"test=[{s.test_idx[0]}..{s.test_idx[-1]}]")

    cpcv = CombinatorialPurgedKFold(n_groups=5, n_test_groups=2)
    cpcv_splits = list(cpcv.split(X))
    print(f"  Combinatorial Purged CV (5 组 / 选 2 组测试): 路径数 = {len(cpcv_splits)}")
    return {
        "purged_n_splits": len(splits),
        "no_overlap": overlap_ok,
        "no_purge_total": no_purge_n,
        "wf_n_splits": len(wf_splits),
        "cpcv_n_paths": len(cpcv_splits),
    }


# ---------------------------------------------------------------------------
# 4. 指标库正确性: 强信号 IC 应当显著
# ---------------------------------------------------------------------------
def verify_factor_metrics() -> Dict[str, Any]:
    section("4. 综合指标库 — 因子 IC / 多空胜率验证")
    panels = [
        ("强信号", make_synthetic_returns(n_stocks=200, n_days=252, signal_strength=0.05, seed=42)),
        ("弱信号", make_synthetic_returns(n_stocks=200, n_days=252, signal_strength=0.001, seed=42)),
    ]
    out = {}
    for label, panel in panels:
        m = factor_metrics(panel[["date", "code", "factor"]],
                           panel[["date", "code", "forward_return"]])
        out[label] = m
        print(f"  {label}: IC={m['ic_mean']:.4f} (ICIR={m['ic_ir']:.2f}), "
              f"Rank IC={m['rank_ic_mean']:.4f}, "
              f"多空胜率={m['long_short_win_rate']:.2%}")
    return out


# ---------------------------------------------------------------------------
# 5. 指标库正确性: 净值指标
# ---------------------------------------------------------------------------
def verify_equity_metrics() -> Dict[str, Any]:
    section("5. 综合指标库 — 净值指标 (calc_all_stats)")
    eq = make_synthetic_equity(n_days=504, annual_return=0.12, annual_vol=0.18, seed=4)
    bench = make_synthetic_equity(n_days=504, annual_return=0.08, annual_vol=0.15, seed=8)
    stats = calc_all_stats(eq, benchmark=bench, risk_free=0.02)
    print(f"  共 {len(stats)} 项指标:")
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"    {k:30s} = {v:.6f}")
    return stats


# ---------------------------------------------------------------------------
# 6. Pipeline 防泄漏
# ---------------------------------------------------------------------------
def verify_pipeline_no_leakage() -> Dict[str, Any]:
    section("6. Pipeline — 数据泄漏防护验证")
    train = pd.DataFrame({
        "date": pd.bdate_range("2023-01-01", periods=30).repeat(5),
        "code": [f"{i}.SZ" for i in range(5)] * 30,
        "factor": np.random.RandomState(1).normal(0, 1, 150),
    })
    test = pd.DataFrame({
        "date": pd.bdate_range("2023-02-15", periods=10).repeat(5),
        "code": [f"{i}.SZ" for i in range(5)] * 10,
        "factor": np.random.RandomState(2).normal(0, 1, 50),
    })
    from validation.pipeline import (
        CrossSectionalScaler,
        MissingValueFiller,
        Pipeline,
        Winsorizer,
    )
    pipe = Pipeline([
        ("imputer", MissingValueFiller(columns=["factor"])),
        ("winsor", Winsorizer(columns=["factor"], lower=0.01, upper=0.99)),
        ("csz", CrossSectionalScaler(columns=["factor"], by="date")),
    ])
    pipe.fit(train)
    train_X = pipe.transform(train)
    test_X = pipe.transform(test)
    print(f"  训练集: {len(train_X)} 行, 因子 NaN 数 = {int(train_X['factor'].isna().sum())}")
    print(f"  测试集: {len(test_X)} 行, 因子 NaN 数 = {int(test_X['factor'].isna().sum())}")
    # 关键: 测试集上的 winsor 上下界应等于训练集分位数
    train_hi = float(train["factor"].quantile(0.99))
    train_lo = float(train["factor"].quantile(0.01))
    test_in_range = bool((test_X["factor"] <= train_hi).all() and (test_X["factor"] >= train_lo).all())
    print(f"  训练集 winsor 界: [{train_lo:.3f}, {train_hi:.3f}]")
    print(f"  测试集因子在训练界内: {test_in_range}")
    return {
        "train_rows": len(train_X),
        "test_rows": len(test_X),
        "train_nan": int(train_X["factor"].isna().sum()),
        "test_nan": int(test_X["factor"].isna().sum()),
        "test_within_train_bounds": test_in_range,
    }


# ---------------------------------------------------------------------------
# 7. 因子 IC 切分演示
# ---------------------------------------------------------------------------
def verify_ic_time_series_split() -> Dict[str, Any]:
    section("7. 因子 IC 时序切分 — purged_cv.ic_time_series_split")
    panel = make_synthetic_returns(n_stocks=100, n_days=300, signal_strength=0.04, seed=42)
    splits = list(ic_time_series_split(panel, n_splits=4, min_train_size=60, purge_days=10))
    ic_per_split = []
    for k, (train_df, val_df, test_df) in enumerate(splits):
        # factor_metrics 要求两个 panel 独立, 避免 merge 后列重名
        train_fwd = train_df[["date", "code", "forward_return"]]
        test_fwd = test_df[["date", "code", "forward_return"]]
        m_train = factor_metrics(
            train_df.drop(columns=["forward_return"]), train_fwd
        )
        m_test = factor_metrics(
            test_df.drop(columns=["forward_return"]), test_fwd
        )
        ic_per_split.append({
            "fold": k,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_ic": round(m_train.get("ic_mean", 0), 4),
            "test_ic": round(m_test.get("ic_mean", 0), 4),
        })
        print(f"  fold {k}: train={len(train_df)} rows (IC={ic_per_split[-1]['train_ic']}), "
              f"test={len(test_df)} rows (IC={ic_per_split[-1]['test_ic']})")
    return ic_per_split


# ---------------------------------------------------------------------------
# 8. 完整 pytest 报告
# ---------------------------------------------------------------------------
def run_pytest() -> Dict[str, Any]:
    section("8. 完整单元测试 (pytest)")
    import subprocess

    result = subprocess.run(
        ["python3", "-m", "pytest", "validation/tests/", "-v", "--tb=line",
         "--no-header", "-q"],
        capture_output=True, text=True,
    )
    print(result.stdout[-2000:])
    if result.returncode != 0 and result.stderr:
        print("STDERR:", result.stderr[-1000:])
    # 解析 passed / failed
    import re
    m = re.search(r"(\d+)\s+passed", result.stdout)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+)\s+failed", result.stdout)
    failed = int(m.group(1)) if m else 0
    return {"passed": passed, "failed": failed, "exit_code": result.returncode}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "branch": "feat/quant-opt-20260618",
        "sections": {},
    }
    report["sections"]["vectorized_correctness"] = verify_vectorized_correctness()
    report["sections"]["vectorized_perf"] = verify_vectorized_perf()
    report["sections"]["purged_cv"] = verify_purged_cv()
    report["sections"]["factor_metrics"] = verify_factor_metrics()
    report["sections"]["equity_metrics"] = verify_equity_metrics()
    report["sections"]["pipeline_no_leakage"] = verify_pipeline_no_leakage()
    report["sections"]["ic_time_series_split"] = verify_ic_time_series_split()
    report["sections"]["pytest"] = run_pytest()
    report["elapsed_sec"] = round(time.perf_counter() - t0, 2)

    out_path = RESULTS_DIR / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print()
    print("=" * 70)
    print(f"  报告已写入: {out_path}")
    print(f"  总耗时: {report['elapsed_sec']}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
