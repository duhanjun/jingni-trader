"""
向量化回测引擎 - 性能对比测试
============================

对比项目：
1. jingni-trader 原生事件循环回测（native_adapter.py）
2. 本次新增的向量化回测引擎（vectorized_engine.py）

测试规模：
- 50/200/500 只股票
- 252 / 504 个交易日（约 1 / 2 年）
- 等权 Top-K 选股，周五调仓
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 加入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 兼容 hyphen 命名的子包
import importlib as _importlib

_native_mod = _importlib.import_module("skills.backtest-engine.scripts.adapters.native_adapter")
NativeAdapter = _native_mod.NativeAdapter

from research_20260617.vectorized_bt.vectorized_engine import (  # noqa: E402
    VectorizedBacktestEngine,
    VectorizedBacktestConfig,
)


def make_synthetic_data(n_stocks: int, n_days: int, seed: int = 42) -> pd.DataFrame:
    """生成 A 股风格的合成日频数据（用于纯性能/正确性对比，不含涨跌停）"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    codes = [f"{i:06d}.SH" for i in range(600000, 600000 + n_stocks)]

    rows = []
    for code in codes:
        close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, size=n_days)))
        for i, d in enumerate(dates):
            open_p = close[i] * (1 + rng.normal(0, 0.005))
            high = max(open_p, close[i]) * (1 + abs(rng.normal(0, 0.003)))
            low = min(open_p, close[i]) * (1 - abs(rng.normal(0, 0.003)))
            vol = abs(rng.normal(1e6, 3e5))
            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "code": code,
                    "open": open_p,
                    "high": high,
                    "low": low,
                    "close": close[i],
                    "volume": vol,
                    "amount": vol * close[i],
                }
            )
    return pd.DataFrame(rows)


def make_signals(data: pd.DataFrame, n_stocks: int, top_k: int = 20) -> pd.DataFrame:
    """构造信号：在每个调仓日(每5个交易日)标记 top_k 只股票为 +1，其余为 -1。

    这种二值信号可以让 native 与 vectorized 引擎在「选股 top-k」语义下
    行为一致（native 买所有 signal>0 的票；vectorized 在 top-k 区间内
    等权持仓），便于横向比较回测结果与性能。
    """
    df = data.sort_values(["code", "date"]).copy()
    df["ret_20"] = df.groupby("code")["close"].pct_change(20)
    rebal_dates = pd.to_datetime(df["date"]).drop_duplicates().sort_values()[::5]

    sig_rows = []
    for d in rebal_dates:
        d_str = d.strftime("%Y-%m-%d")
        sub = df[df["date"] == d_str][["code", "ret_20"]].dropna()
        if sub.empty:
            continue
        top_codes = sub.nlargest(top_k, "ret_20")["code"].tolist()
        for code in sub["code"]:
            sig_rows.append(
                {
                    "date": d_str,
                    "code": code,
                    "signal": 1.0 if code in top_codes else -1.0,
                }
            )
    return pd.DataFrame(sig_rows)


def benchmark(n_stocks: int, n_days: int) -> dict:
    print(f"\n=== n_stocks={n_stocks} n_days={n_days} ===")
    data = make_synthetic_data(n_stocks, n_days)
    signals = make_signals(data, n_stocks)
    print(f"  data rows={len(data):,} signals rows={len(signals):,}")

    # --- 原生事件循环 ---
    native = NativeAdapter()
    t0 = time.perf_counter()
    native_res = native.run_backtest(data, signals)
    native_t = time.perf_counter() - t0
    native_metrics = native_res.get("metrics", {})

    # --- 向量化 ---
    vec = VectorizedBacktestEngine(
        VectorizedBacktestConfig(rebalance_freq="W-FRI", top_k=max(20, n_stocks // 20))
    )
    t0 = time.perf_counter()
    vec_res = vec.run_backtest(data, signals)
    vec_t = time.perf_counter() - t0
    vec_metrics = vec_res.get("metrics", {})

    # 校正：先打印再算 speedup
    print(f"  native : {native_t:8.3f}s  sharpe={native_metrics.get('sharpe_ratio', 0):.3f}  ret={native_metrics.get('total_return', 0):.3f}  trades={len(native_res['trades'])}")
    print(f"  vector : {vec_t:8.3f}s  sharpe={vec_metrics.get('sharpe_ratio', 0):.3f}  ret={vec_metrics.get('total_return', 0):.3f}  trades={len(vec_res['trades'])}")
    speedup = native_t / vec_t if vec_t > 0 else float("inf")
    print(f"  speedup: {speedup:6.1f}x")

    return {
        "n_stocks": n_stocks,
        "n_days": n_days,
        "native_time_s": native_t,
        "vector_time_s": vec_t,
        "speedup": speedup,
        "native_sharpe": native_metrics.get("sharpe_ratio", 0.0),
        "vector_sharpe": vec_metrics.get("sharpe_ratio", 0.0),
        "native_total_return": native_metrics.get("total_return", 0.0),
        "vector_total_return": vec_metrics.get("total_return", 0.0),
        "native_max_drawdown": native_metrics.get("max_drawdown", 0.0),
        "vector_max_drawdown": vec_metrics.get("max_drawdown", 0.0),
        "native_trades": len(native_res["trades"]),
        "vector_trades": len(vec_res["trades"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[50, 200, 500],
        help="测试的股票数量",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=504,
        help="测试的交易日数量",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "research_20260617" / "vectorized_bt" / "benchmark_result.json"),
    )
    args = parser.parse_args()

    results = []
    for n in args.scales:
        results.append(benchmark(n, args.days))

    out = {
        "scenarios": results,
        "summary": {
            "avg_speedup": float(np.mean([r["speedup"] for r in results])),
            "min_speedup": float(np.min([r["speedup"] for r in results])),
            "max_speedup": float(np.max([r["speedup"] for r in results])),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
