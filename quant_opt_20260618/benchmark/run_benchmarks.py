"""
端到端基准测试与对比报告

目的：
  把"借鉴自外部项目"的 4 个优化点统一跑一次基准，对比
  jingni-trader 现有实现的不足与新 PoC 的优势。

包含 4 个对比维度：
  1. 因子计算：
     - 老：硬编码 5+ 个因子在 FactorEngine.compute_a_share_factors
     - 新：公式 DSL（"Rank(Ts_Mean($close, 5))"）
  2. PIT 数据合并：老实现 vs 新 pit_merge
  3. 滚动训练：老 purged_group_ts_split vs 新 walk_forward
  4. 缓存：当前 if-exists 跳过 vs 新 fingerprint cache

运行：
  PYTHONPATH=. python quant_opt_20260618/benchmark/run_benchmarks.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# 允许脚本独立运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_engine import calc_factor, calc_factors
from pit_join import PITConfig, pit_merge, detect_lookahead
from walk_forward import WalkForwardConfig, walk_forward_train_predict, aggregate_wf_metrics
from cache import fingerprint, DiskCache
from tests.fixtures import make_synthetic_ashare_data, make_financial_data


RESULTS: List[Dict[str, Any]] = []


def _record(category: str, name: str, payload: Dict[str, Any]) -> None:
    payload = {"category": category, "name": name, **payload}
    RESULTS.append(payload)
    print(f"[{category}] {name}: {json.dumps(payload, ensure_ascii=False, default=str)}")


# ─────────────────────────────────────────────────────────────
# 1. 因子计算：硬编码 vs 公式 DSL
# ─────────────────────────────────────────────────────────────

def benchmark_factor_engine(n_stocks: int = 50, n_days: int = 1200) -> None:
    df = make_synthetic_ashare_data(n_stocks=n_stocks, n_days=n_days, seed=11)

    # 老实现：模拟 jingni-trader 中 FactorEngine.compute_a_share_factors 的 5 个核心因子
    def legacy_compute(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values(["code", "date"]).copy()
        out = d[["code", "date"]].copy()
        out["ret_1d"] = d.groupby("code")["close"].pct_change()
        out["ret_5d"] = d.groupby("code")["close"].pct_change(5)
        out["ret_20d"] = d.groupby("code")["close"].pct_change(20)
        out["volatility_20d"] = d.groupby("code")["close"].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )
        out["turnover_20d"] = d.groupby("code")["turnover_rate"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        return out

    # 新实现：4 行公式搞定同样的因子 + 额外的 rank 化
    formulas = {
        "ret_1d":         "Delta($close, 1) / Delay($close, 1)",
        "ret_5d":         "Delta($close, 5) / Delay($close, 5)",
        "ret_20d":        "Delta($close, 20) / Delay($close, 20)",
        "volatility_20d": "Ts_Std(Delta($close, 1) / Delay($close, 1), 20)",
        "turnover_20d":   "Ts_Mean($turnover_rate, 20)",
        "mom_rank":       "Rank(Delta($close, 5) / Delay($close, 5))",
    }

    t0 = time.perf_counter()
    legacy = legacy_compute(df)
    t_legacy = time.perf_counter() - t0

    t0 = time.perf_counter()
    new = calc_factors(df, formulas)
    t_new = time.perf_counter() - t0

    # 一致性抽样：ret_5d 在两个实现上应当数值一致
    legacy_ret5 = legacy.sort_values(["code", "date"])["ret_5d"].reset_index(drop=True)
    new_ret5 = new.sort_values(["code", "date"])["ret_5d"].reset_index(drop=True)
    diff = float((legacy_ret5.fillna(-1) - new_ret5.fillna(-1)).abs().max())

    _record("factor_engine", "硬编码 vs 公式 DSL", {
        "n_stocks": n_stocks,
        "n_days": n_days,
        "n_rows": len(df),
        "legacy_seconds": round(t_legacy, 4),
        "new_seconds": round(t_new, 4),
        "speedup_x": round(t_legacy / max(t_new, 1e-9), 2),
        "max_ret5_diff": diff,
        "legacy_factors": 5,
        "new_factors": len(formulas),
        "new_extras": ["mom_rank"],  # 仅新实现支持
        "lines_of_code_legacy": 11,
        "lines_of_code_new": len(formulas),
    })


# ─────────────────────────────────────────────────────────────
# 2. PIT 数据合并：直接 merge vs pit_merge
# ─────────────────────────────────────────────────────────────

def benchmark_pit_merge(n_stocks: int = 20, n_periods: int = 12) -> None:
    financial = make_financial_data(n_stocks=n_stocks, n_periods=n_periods, seed=23)
    # 让 left 覆盖 600 个交易日
    bdays = pd.bdate_range("2022-01-01", periods=600)
    left = pd.DataFrame([
        {"code": c, "date": d}
        for c in financial["code"].unique()
        for d in bdays
    ])

    cfg = PITConfig(asof_col="date", announce_col="announce_date", by="code")

    # 老实现：直接按 period_end join（会引入未来函数）
    t0 = time.perf_counter()
    bad_right = financial.rename(columns={"period_end": "_period_end"})
    bad_merge = left.merge(bad_right, left_on="code", right_on="code", how="left")
    t_legacy = time.perf_counter() - t0

    # 新实现：PIT 合并
    t0 = time.perf_counter()
    pit_merged = pit_merge(left, financial, cfg, value_cols=["pe_ttm", "roe", "revenue_growth"])
    t_new = time.perf_counter() - t0

    # 报告检测出的未来函数
    report = detect_lookahead(bad_merge, pit_merged, value_cols=["pe_ttm", "roe", "revenue_growth"])

    _record("pit_merge", "直接 merge vs PIT merge", {
        "left_rows": len(left),
        "right_rows": len(financial),
        "legacy_seconds": round(t_legacy, 4),
        "new_seconds": round(t_new, 4),
        "lookahead_eliminated": report["total_lookahead_eliminated"],
        "lookahead_ratio_pct": round(
            100.0 * report["total_lookahead_eliminated"] / max(1, len(bad_merge)), 4
        ),
    })


# ─────────────────────────────────────────────────────────────
# 3. 滚动训练：purged_group_ts_split vs walk_forward
# ─────────────────────────────────────────────────────────────

def _make_panel(n_stocks: int = 20, n_days: int = 1500, seed: int = 1) -> pd.DataFrame:
    df = make_synthetic_ashare_data(n_stocks=n_stocks, n_days=n_days, seed=seed)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["ret_1d"] = df.groupby("code")["close"].pct_change().shift(-1)
    df["feat_ma5"] = df.groupby("code")["close"].transform(lambda s: s.rolling(5).mean())
    df["feat_ma20"] = df.groupby("code")["close"].transform(lambda s: s.rolling(20).mean())
    df = df.dropna()
    return df


def _fit_predict_linear(X_train, y_train, X_val, y_val, X_test):
    Xt = np.hstack([np.ones((len(X_train), 1)), X_train.values])
    Xv = np.hstack([np.ones((len(X_test), 1)), X_test.values])
    coef, *_ = np.linalg.lstsq(Xt, y_train.values, rcond=None)
    return Xv @ coef


def benchmark_walk_forward() -> None:
    df = _make_panel(n_stocks=20, n_days=1500, seed=3)
    X = df[["feat_ma5", "feat_ma20"]]
    y = df["ret_1d"]
    dates = df["date"]

    cfg = WalkForwardConfig(
        train_window_days=400,
        val_window_days=120,
        test_window_days=120,
        step_days=120,
        purge_days=5,
        embargo_days=5,
    )
    t0 = time.perf_counter()
    oos, results = walk_forward_train_predict(X, y, dates, _fit_predict_linear, cfg)
    t_new = time.perf_counter() - t0

    summary = aggregate_wf_metrics(results)
    _record("walk_forward", "滚动训练 OOS 预测", {
        "n_rows": len(df),
        "n_folds": summary.get("n_folds", 0),
        "seconds": round(t_new, 4),
        "oos_rows": int(oos.shape[0]),
        "oos_coverage_pct": round(100 * float(oos["pred"].notna().mean()), 2),
        "mean_ic": summary.get("mean_ic"),
        "icir": summary.get("icir"),
        "positive_ic_ratio": summary.get("positive_ic_ratio"),
    })


# ─────────────────────────────────────────────────────────────
# 4. 内容指纹缓存
# ─────────────────────────────────────────────────────────────

def benchmark_fingerprint_cache(tmp_dir: str) -> None:
    cache = DiskCache(root=tmp_dir)
    df = make_synthetic_ashare_data(n_stocks=10, n_days=300, seed=5)
    fp = fingerprint(df)

    # 模拟昂贵计算
    def expensive():
        time.sleep(0.1)
        return {"rows": len(df), "fp": fp}

    t0 = time.perf_counter()
    cache.get_or_compute("panel", fp, expensive)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    cache.get_or_compute("panel", fp, expensive)
    t_second = time.perf_counter() - t0

    # 同一份"参数略改"后指纹变化，触发重算
    fp_v2 = fingerprint(df.assign(close=df["close"] * 1.001))
    t0 = time.perf_counter()
    cache.get_or_compute("panel", fp_v2, expensive)
    t_third = time.perf_counter() - t0

    _record("cache", "fingerprint 缓存", {
        "first_compute_seconds": round(t_first, 4),
        "cache_hit_seconds": round(t_second, 4),
        "speedup_x": round(t_first / max(t_second, 1e-9), 1),
        "fingerprint_changed_invalidate_seconds": round(t_third, 4),
        "stats": {
            "hits": cache.stats.hits,
            "misses": cache.stats.misses,
            "hit_rate": round(cache.stats.hit_rate(), 2),
        },
    })


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 80)
    print(" jingni-trader 优化方向 PoC 基准报告")
    print(f" 运行时间: {datetime.now().isoformat()}")
    print("=" * 80)

    benchmark_factor_engine()
    benchmark_pit_merge()
    benchmark_walk_forward()
    benchmark_fingerprint_cache(tmp_dir=os.path.join(os.path.dirname(__file__), "_tmp_cache"))
    try:
        import shutil
        shutil.rmtree(os.path.join(os.path.dirname(__file__), "_tmp_cache"), ignore_errors=True)
    except Exception:
        pass

    # 保存结果 JSON
    out = {
        "generated_at": datetime.now().isoformat(),
        "results": RESULTS,
    }
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print()
    print(f"基准结果已保存至: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
