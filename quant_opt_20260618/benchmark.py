"""
Benchmark + integration test
=============================

Runs a full A-share-style workflow:
  1. Synthesize a 250-day × 2000-stock panel.
  2. Evaluate every factor in ``ALPHA158_LITE`` using the DSL.
  3. Run the vectorized IC on each factor in one batch.
  4. Validate the top factors with the bootstrap rule significance test.
  5. Print a small summary table.

Usage::

    python -m quant_opt_20260618.benchmark
"""
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

from quant_opt_20260618 import (
    ic_series_pearson,
    ic_summary,
    validate_factor,
)
from quant_opt_20260618.expression_dsl import (
    ALPHA158_LITE,
    evaluate,
    list_operators,
)


def build_panel(n_dates: int = 250, n_stocks: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic A-share-style panel with 5 latent signals."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
    codes = [f"{i:06d}.SH" for i in range(n_stocks)]

    # Five latent factors -> drive five observable factors
    latents = rng.standard_normal((n_dates, 5))
    fwd = rng.standard_normal((n_dates, n_stocks)) * 0.01
    fwd += 0.5 * latents[:, 0:1]               # momentum
    fwd += -0.3 * latents[:, 1:2]              # reversal
    fwd += 0.2 * latents[:, 2:3]               # vol
    fwd += 0.4 * latents[:, 3:4]               # amount chg
    fwd += -0.25 * latents[:, 4:5]             # mean reversion

    rows = []
    for ci, c in enumerate(codes):
        noise = rng.standard_normal(n_dates) * 0.01
        close = 100 * np.exp(np.cumsum(rng.standard_normal(n_dates) * 0.02))
        op = close * (1 + rng.standard_normal(n_dates) * 0.005)
        hi = close * (1 + np.abs(rng.standard_normal(n_dates)) * 0.01)
        lo = close * (1 - np.abs(rng.standard_normal(n_dates)) * 0.01)
        vol = rng.integers(1_000_000, 10_000_000, n_dates).astype(float)
        amt = vol * close
        tr = rng.random(n_dates) * 5
        f1 = latents[:, 0] + noise * 0.1
        f2 = latents[:, 1] + noise * 0.1
        f3 = latents[:, 2] + noise * 0.1
        f4 = latents[:, 3] + noise * 0.1
        f5 = latents[:, 4] + noise * 0.1
        for i in range(n_dates):
            rows.append({
                "code": c, "date": dates[i],
                "open": op[i], "high": hi[i], "low": lo[i],
                "close": close[i], "volume": vol[i], "amount": amt[i],
                "turnover_rate": tr[i], "change_pct": (close[i] - op[i]) / op[i],
                "f_mom": f1[i], "f_rev": f2[i], "f_vol": f3[i],
                "f_amt": f4[i], "f_mr": f5[i],
                "ret_fwd_5d": fwd[i, ci],
            })
    return pd.DataFrame(rows)


def run_benchmark():
    print("== Building panel …", flush=True)
    t0 = time.perf_counter()
    df = build_panel()
    print(f"   {len(df):,} rows × {df['code'].nunique():,} stocks × "
          f"{df['date'].nunique():,} dates "
          f"in {time.perf_counter() - t0:.1f}s\n", flush=True)

    # 1) DSL: 评估 ALPHA158_LITE 中的全部因子
    print(f"== Evaluating {len(ALPHA158_LITE)} factors via DSL …", flush=True)
    t0 = time.perf_counter()
    factor_results: Dict[str, pd.Series] = {}
    for name, expr in ALPHA158_LITE.items():
        t1 = time.perf_counter()
        factor_results[name] = evaluate(expr, df)
        print(f"   {name:20s}  {(time.perf_counter() - t1) * 1000:6.1f} ms", flush=True)
    print(f"   total {time.perf_counter() - t0:.1f}s\n", flush=True)

    # 把所有因子拼回主表
    print("== Vectorized IC pass …", flush=True)
    t0 = time.perf_counter()
    fwd = df["ret_fwd_5d"]
    rows = []
    for name, s in factor_results.items():
        ic = ic_series_pearson(s, fwd, df["date"], min_obs=30)
        s_dict = ic_summary(ic)
        rows.append({"factor": name, **s_dict})
    summary = pd.DataFrame(rows).sort_values("ic_ir", key=lambda x: x.abs(),
                                             ascending=False)
    print(summary.to_string(index=False))
    print(f"   IC pass in {time.perf_counter() - t0:.2f}s\n", flush=True)

    # 2) Bootstrap 规则显著性：只跑 top 5
    print("== Bootstrap rule significance (top 5 by |ICIR|) …", flush=True)
    t0 = time.perf_counter()
    top5 = summary.head(5)["factor"].tolist()
    for name in top5:
        s = factor_results[name]
        t1 = time.perf_counter()
        v = validate_factor(s, fwd, df["date"], n_bootstrap=200)
        dt = time.perf_counter() - t1
        print(f"   {name:20s}  {v.decision:7s}  IR={v.ic_ir:+.3f}  "
              f"p={v.bootstrap_p:.3f}  n={v.n_obs}  ({dt*1000:.0f} ms)", flush=True)
    print(f"   bootstrap in {time.perf_counter() - t0:.2f}s\n", flush=True)

    # 3) 列出可用算子（仅展示）
    print("== Available operators ==")
    print("  " + ", ".join(list_operators()))

    print("\n[OK] benchmark complete", flush=True)
    return summary


if __name__ == "__main__":
    run_benchmark()
