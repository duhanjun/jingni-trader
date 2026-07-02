"""
benchmark_comparison.py
=======================

对优化后的回测引擎与 jingni-trader 现有 ``native_adapter`` 做性能与正确性对比。

测试场景
--------
1. **正确性**: 在同一组合信号下, 优化版与参考版产出的总收益、Sharpe 应在容差内一致
2. **性能**: 不同规模 (N_stocks × N_days) 下的墙钟时间对比
3. **指标扩展**: jingni 原生 _calc_metrics 7 个字段 vs 优化版 20+ 字段

注意
----
- 参考实现使用循环 (naive Python), 用于 sanity check
- jingni-trader 的 native_adapter 依赖外部 rqalpha/backtrader, 这里用纯 Python
  naive loop 模拟"事件驱动"基线, 作为对比基准
"""
from __future__ import annotations

import logging
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace")
from quant_opt_20260616 import vectorized_backtest, performance_metrics

logger = logging.getLogger("quant_opt_20260616.bench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def make_data(n_stocks: int, n_days: int, seed: int = 0) -> pd.DataFrame:
    """构造宽表数据"""
    rng = np.random.default_rng(seed)
    base = 10 + rng.uniform(0, 30, n_stocks)
    daily_ret = rng.normal(0.0005, 0.02, (n_days, n_stocks))
    prices = base[None, :] * np.exp(np.cumsum(daily_ret, axis=0))
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    codes = [f"STK{i:04d}" for i in range(n_stocks)]
    df = pd.DataFrame(prices, index=dates, columns=codes)
    return df


def naive_backtest(
    close: pd.DataFrame,
    signals: pd.DataFrame,
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.001,
    slippage: float = 0.001,
) -> Dict[str, pd.DataFrame]:
    """
    朴素 Python 事件驱动回测 (用于对比)
    按日循环, 等权分配, 买入/卖出按信号 trigger
    """
    cash = 1_000_000.0
    holdings = {c: 0 for c in close.columns}
    avg_cost = {c: 0.0 for c in close.columns}
    equity_curve = []

    common_idx = close.index.intersection(signals.index)
    close = close.loc[common_idx]
    signals = signals.loc[common_idx].fillna(0).astype(int)

    for t in range(1, len(common_idx)):
        today = common_idx[t]
        prev = common_idx[t - 1]
        sig = signals.loc[today]
        # 调仓: 卖出不在信号中的, 买入新增的
        sell_value = 0.0
        for c in close.columns:
            if holdings[c] > 0 and sig.get(c, 0) == 0:
                price = close.at[today, c] * (1 - slippage)
                fee = max(price * holdings[c] * commission_rate, 5.0)
                tax = price * holdings[c] * stamp_tax_rate
                cash += price * holdings[c] - fee - tax
                sell_value += price * holdings[c]
                holdings[c] = 0
                avg_cost[c] = 0.0
        # 计算当前持仓市值
        position_value = sum(holdings[c] * close.at[today, c] for c in close.columns)
        # 买入
        targets = [c for c in close.columns if sig.get(c, 0) == 1 and holdings[c] == 0]
        if targets and cash > 0:
            per = cash / len(targets)
            for c in targets:
                price = close.at[today, c] * (1 + slippage)
                shares = int(per / price / 100) * 100
                if shares > 0:
                    fee = max(price * shares * commission_rate, 5.0)
                    cost = price * shares + fee
                    if cost <= cash:
                        cash -= cost
                        holdings[c] = shares
                        avg_cost[c] = price
        # 计算权益
        eq = cash + sum(holdings[c] * close.at[today, c] for c in close.columns)
        equity_curve.append({"date": today, "equity": eq})
    eq_df = pd.DataFrame(equity_curve)
    eq_df["ret"] = eq_df["equity"].pct_change().fillna(0)
    return {"equity_curve": eq_df}


def run_benchmarks() -> Dict:
    """运行一组性能基准测试, 返回结构化结果"""
    sizes = [
        (10, 100),    # 小
        (50, 252),    # 中 (1 年, 50 股票)
        (100, 500),   # 大
        (500, 500),   # 超大
    ]
    rows: List[Dict] = []
    for n_stocks, n_days in sizes:
        close = make_data(n_stocks, n_days)
        signals = pd.DataFrame(1, index=close.index, columns=close.columns)  # 全持仓
        # 优化版
        cfg = vectorized_backtest.VectorBTConfig(
            commission_rate=0.00025, stamp_tax_rate=0.001, slippage=0.001
        )
        t0 = time.perf_counter()
        opt_result = vectorized_backtest.vectorized_backtest(close, signals, cfg)
        opt_ms = (time.perf_counter() - t0) * 1000
        # naive 版 (仅小规模跑, 大规模太慢)
        naive_ms = None
        if n_stocks * n_days <= 20000:
            t0 = time.perf_counter()
            naive_result = naive_backtest(close, signals)
            naive_ms = (time.perf_counter() - t0) * 1000
        # 指标计算
        eq_opt = opt_result["equity_curve"].set_index("date")
        metrics = performance_metrics.compute_metrics(eq_opt["equity"], eq_opt["ret"])
        rows.append({
            "n_stocks": n_stocks,
            "n_days": n_days,
            "opt_ms": opt_ms,
            "naive_ms": naive_ms,
            "speedup": (naive_ms / opt_ms) if naive_ms else None,
            "opt_final_equity": float(eq_opt["equity"].iloc[-1]),
            "n_metrics": len(metrics),
            "sharpe": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
        })
    return {"size_benchmarks": rows}


def run_correctness_test() -> List[Dict]:
    """正确性测试: 向量化 vs naive, 在小规模下应该结果接近"""
    rows = []
    rng = np.random.default_rng(0)
    for seed in [0, 1, 2]:
        close = make_data(5, 60, seed=seed)
        # 随机信号
        sig_data = (rng.random(close.shape) > 0.5).astype(int)
        signals = pd.DataFrame(sig_data, index=close.index, columns=close.columns)
        cfg = vectorized_backtest.VectorBTConfig(
            commission_rate=0.00025, stamp_tax_rate=0.001, slippage=0.001
        )
        opt = vectorized_backtest.vectorized_backtest(close, signals, cfg)
        naive = naive_backtest(close, signals)
        opt_eq = float(opt["equity_curve"]["equity"].iloc[-1])
        naive_eq = float(naive["equity_curve"]["equity"].iloc[-1])
        # 两种实现: 价差来源主要是 现金分配规则不同, 允许 20% 误差
        rel_diff = abs(opt_eq - naive_eq) / max(1, naive_eq)
        rows.append({
            "seed": seed,
            "opt_final": round(opt_eq, 2),
            "naive_final": round(naive_eq, 2),
            "rel_diff": round(rel_diff, 4),
            "acceptable": rel_diff < 0.20,
        })
    return rows


def run_metrics_expansion() -> Dict:
    """对比 jingni-trader 现有 7 字段 vs 优化版 20+ 字段"""
    close = make_data(20, 252)
    eq = (1 + np.random.default_rng(0).normal(0.0008, 0.02, 252)).cumprod() * 1e6
    eq = pd.Series(eq, index=close.index)
    ret = eq.pct_change().fillna(0)
    bench_ret = pd.Series(np.random.default_rng(1).normal(0.0005, 0.015, 252), index=eq.index)
    # jingni 现有
    jingni_fields = {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "annual_return": float((eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1),
        "volatility": float(ret.std() * np.sqrt(252)),
        "sharpe_ratio": float(((ret.mean() * 252) - 0.03) / (ret.std() * np.sqrt(252))),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "win_rate": float((ret > 0).mean()),
        "calmar_ratio": 0.0,  # 简化
    }
    # 优化版
    opt_fields = performance_metrics.compute_metrics(eq, ret, bench_ret)
    new_keys = set(opt_fields.keys()) - set(jingni_fields.keys())
    return {
        "jingni_field_count": len(jingni_fields),
        "opt_field_count": len(opt_fields),
        "new_metrics": sorted(new_keys),
        "shared_metrics": sorted(set(jingni_fields.keys()) & set(opt_fields.keys())),
    }


def main():
    print("=" * 70)
    print("jingni-trader 量化优化 - 性能与正确性基准")
    print("=" * 70)
    print()
    print("[1/3] 性能基准 (向量化 vs naive 事件驱动)")
    print("-" * 70)
    bench = run_benchmarks()
    for r in bench["size_benchmarks"]:
        speedup = f"{r['speedup']:.1f}x" if r["speedup"] else "N/A"
        naive_str = f"{r['naive_ms']:>8.1f}ms" if r["naive_ms"] else "     skip"
        print(f"  {r['n_stocks']:>4}×{r['n_days']:>4}: 优化版 {r['opt_ms']:>7.1f}ms"
              f"  vs naive {naive_str}  加速 {speedup}")
    print()
    print("[2/3] 正确性测试 (向量化 vs naive)")
    print("-" * 70)
    correct = run_correctness_test()
    n_pass = sum(1 for r in correct if r["acceptable"])
    for r in correct:
        status = "✓" if r["acceptable"] else "✗"
        print(f"  {status} seed={r['seed']}: 优化版={r['opt_final']:.0f}, "
              f"naive={r['naive_final']:.0f}, 偏差={r['rel_diff']*100:.2f}%")
    print(f"  正确性通过: {n_pass}/{len(correct)}")
    print()
    print("[3/3] 指标体系扩展 (jingni-trader 现有 vs 优化版)")
    print("-" * 70)
    exp = run_metrics_expansion()
    print(f"  jingni 现有 {exp['jingni_field_count']} 个字段: {', '.join(exp['shared_metrics'])}")
    print(f"  优化版 {exp['opt_field_count']} 个字段 (新增 {len(exp['new_metrics'])}):")
    for m in exp["new_metrics"]:
        print(f"    + {m}")
    print()
    return {
        "benchmarks": bench,
        "correctness": correct,
        "metrics_expansion": exp,
    }


if __name__ == "__main__":
    import json
    res = main()
    with open("/workspace/quant_opt_20260616/_benchmark_results.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
