"""
生成 Golden baseline: 用固定 seed 的合成数据, 把关键指标序列化保存。
将来回归测试可以对照此 baseline 验证计算结果稳定。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.metrics import calc_all_stats, factor_metrics
from validation.purged_cv import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    WalkForwardSplitter,
    ic_time_series_split,
)
from validation.synth_data import (
    make_synthetic_equity,
    make_synthetic_panel,
    make_synthetic_returns,
)
from validation.vectorized_factor import (
    LoopFactorCalculator,
    VectorizedFactorCalculator,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def round_dict(d: dict, decimals: int = 6) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            if not np.isfinite(v):
                out[k] = None
            else:
                out[k] = round(v, decimals)
        elif isinstance(v, (np.floating,)):
            out[k] = round(float(v), decimals)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, dict):
            out[k] = round_dict(v, decimals)
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = v
    return out


def make_golden() -> dict:
    out: dict = {}

    # 1. 向量化因子: 数值一致性 (与 loop 实现完全一致, 最大绝对差=0)
    panel = make_synthetic_panel(n_stocks=20, n_days=200, seed=11)
    factor_list = ["ma_5", "ma_20", "ma_60", "ema_12", "rsi_14", "momentum_20d", "zscore_20"]
    loop = LoopFactorCalculator()
    vec = VectorizedFactorCalculator()
    out_loop = loop.calculate(panel, factor_list).sort_values(["code", "date"]).reset_index(drop=True)
    out_vec = vec.calculate(panel, factor_list).sort_values(["code", "date"]).reset_index(drop=True)
    out["vectorized_factors"] = {
        f"max_abs_diff_{f}": float(np.abs(out_loop[f] - out_vec[f]).max()) for f in factor_list
    }

    # 2. 净值指标
    eq = make_synthetic_equity(n_days=504, annual_return=0.12, annual_vol=0.18, seed=4)
    bench = make_synthetic_equity(n_days=504, annual_return=0.08, annual_vol=0.15, seed=8)
    stats = calc_all_stats(eq, benchmark=bench, risk_free=0.02)
    out["equity_stats"] = round_dict({k: v for k, v in stats.items()
                                       if isinstance(v, (int, float, np.floating))})

    # 3. 因子 IC
    panel = make_synthetic_returns(n_stocks=200, n_days=252, signal_strength=0.05, seed=42)
    m = factor_metrics(panel[["date", "code", "factor"]],
                       panel[["date", "code", "forward_return"]])
    out["factor_metrics_strong_signal"] = round_dict(m)

    panel = make_synthetic_returns(n_stocks=200, n_days=252, signal_strength=0.001, seed=42)
    m = factor_metrics(panel[["date", "code", "factor"]],
                       panel[["date", "code", "forward_return"]])
    out["factor_metrics_weak_signal"] = round_dict(m)

    # 4. Purged CV 行为
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    X = pd.DataFrame({"x": range(300)}, index=dates)
    pkf = PurgedKFold(n_splits=5, purge_td="5D", embargo_td="5D")
    splits = list(pkf.split(X))
    out["purged_kfold"] = {
        "n_splits": len(splits),
        "train_size_per_fold": [int(len(s.train_idx)) for s in splits],
        "test_size_per_fold": [int(len(s.test_idx)) for s in splits],
    }
    wf = WalkForwardSplitter(train_size=200, test_size=50, step_size=50)
    splits = list(wf.split(X))
    out["walk_forward"] = {
        "n_splits": len(splits),
        "fold_train_start": [int(s.train_idx[0]) for s in splits],
        "fold_test_start": [int(s.test_idx[0]) for s in splits],
    }
    cpcv = CombinatorialPurgedKFold(n_groups=5, n_test_groups=2)
    splits = list(cpcv.split(X))
    out["combinatorial_purged_kfold"] = {
        "n_paths": len(splits),
        "test_size_per_path": [int(len(s.test_idx)) for s in splits],
    }

    # 5. IC time series split
    panel = make_synthetic_returns(n_stocks=100, n_days=300, signal_strength=0.04, seed=42)
    splits = list(ic_time_series_split(panel, n_splits=4, min_train_size=60, purge_days=10))
    out["ic_time_series_split"] = {
        "n_splits": len(splits),
        "train_sizes": [int(len(t[0])) for t in splits],
        "test_sizes": [int(len(t[2])) for t in splits],
    }

    return out


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden = make_golden()
    out_path = GOLDEN_DIR / "golden_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    print(f"Golden baseline 写入: {out_path}")
    print(f"包含 {len(golden)} 个模块的 baseline 数据")
    for k in golden:
        print(f"  - {k}")
