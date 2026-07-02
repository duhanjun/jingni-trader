"""
性能与精度对比基准测试
======================

对比 jingni-trader 的 ``native_adapter`` 与 ``VectorizedBacktest``：

1. 性能对比（runtime）
2. 精度对比（metrics 差异）

运行：
    cd /workspace
    python -m skills.quant_opt.benchmark
"""

import sys
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from skills.quant_opt.vectorized_backtest import VectorizedBacktest, compare_results
# 用 package 形式 import native_adapter
import importlib
_pkg = importlib.import_module("skills.backtest-engine.scripts.adapters.native_adapter")
NativeAdapter = _pkg.NativeAdapter


def make_synth_data(n_codes=20, n_days=120, seed=42):
    """生成合成 A 股数据"""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    for code_i in range(n_codes):
        code = f"{code_i:06d}.SZ"
        base = 10.0 + code_i * 3
        ret = rng.normal(0.001, 0.02, n_days)
        prices = base * np.cumprod(1 + ret)
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "open": prices[i] * 0.99,
                "high": prices[i] * 1.01,
                "low": prices[i] * 0.99,
                "close": prices[i],
                "volume": int(rng.integers(1_000_000, 5_000_000)),
                "amount": float(prices[i] * rng.integers(1_000_000, 5_000_000)),
            })
    return pd.DataFrame(rows)


def make_topk_signals(data, top_k=5):
    rows = []
    for dt, g in data.groupby("date"):
        ranked = g.sort_values("close", ascending=False).head(top_k)
        for _, r in ranked.iterrows():
            rows.append({"code": r["code"], "date": dt, "signal": 1})
    return pd.DataFrame(rows)


def run_benchmark(n_codes=20, n_days=120, top_k=5, init_capital=1_000_000.0):
    """运行性能/精度对比基准"""
    print(f"\n{'='*60}")
    print(f"  基准测试: {n_codes} 只股票 × {n_days} 个交易日, top-{top_k}")
    print(f"{'='*60}\n")

    data = make_synth_data(n_codes=n_codes, n_days=n_days)
    signals = make_topk_signals(data, top_k=top_k)

    # 1. native_adapter
    print("[1/3] 运行 native_adapter...")
    native = NativeAdapter()
    t0 = time.time()
    native_result = native.run_backtest(
        data=data, signals=signals, init_capital=init_capital,
    )
    native_time = time.time() - t0
    print(f"      耗时: {native_time:.3f}s")
    print(f"      total_return: {native_result['metrics'].get('total_return', 0):.4%}")
    print(f"      sharpe_ratio: {native_result['metrics'].get('sharpe_ratio', 0):.4f}")
    print(f"      max_drawdown: {native_result['metrics'].get('max_drawdown', 0):.4%}")

    # 2. VectorizedBacktest
    print("\n[2/3] 运行 VectorizedBacktest...")
    vectorized = VectorizedBacktest(init_capital=init_capital)
    t0 = time.time()
    v_result = vectorized.run(data, signals)
    v_time = time.time() - t0
    print(f"      耗时: {v_time:.3f}s")
    print(f"      total_return: {v_result.metrics.get('total_return', 0):.4%}")
    print(f"      sharpe_ratio: {v_result.metrics.get('sharpe_ratio', 0):.4f}")
    print(f"      max_drawdown: {v_result.metrics.get('max_drawdown', 0):.4%}")

    # 3. 对比
    print("\n[3/3] 精度与性能对比:")
    report = compare_results(native_result, v_result)

    print(f"      native 耗时: {native_time:.3f}s")
    print(f"      vectorized 耗时: {v_time:.3f}s")
    if v_time > 0:
        print(f"      加速比: {native_time / v_time:.2f}x")

    print(f"\n      精度差异 (abs / rel / pass):")
    all_passed = True
    for k, v in report.items():
        status = "✓" if v["passed"] else "✗"
        if not v["passed"]:
            all_passed = False
        print(
            f"        {status} {k:18s}  native={v['native']:+.4f}  "
            f"vectorized={v['vectorized']:+.4f}  abs={v['abs_error']:.4e}  "
            f"rel={v['rel_error']:.2%}"
        )

    return {
        "config": {
            "n_codes": n_codes,
            "n_days": n_days,
            "top_k": top_k,
        },
        "native": {
            "runtime_s": native_time,
            "metrics": native_result["metrics"],
        },
        "vectorized": {
            "runtime_s": v_time,
            "metrics": v_result.metrics,
        },
        "speedup": native_time / v_time if v_time > 0 else 0.0,
        "precision_report": report,
        "all_precision_passed": all_passed,
    }


def main():
    """主函数：运行多组对比并输出报告"""
    results = []
    configs = [
        {"n_codes": 10, "n_days": 60, "top_k": 3},
        {"n_codes": 50, "n_days": 240, "top_k": 10},
        {"n_codes": 200, "n_days": 480, "top_k": 20},
        {"n_codes": 500, "n_days": 1000, "top_k": 50},
    ]
    for cfg in configs:
        result = run_benchmark(**cfg)
        results.append(result)

    # 输出 JSON 报告
    report_path = HERE / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n\n报告已保存至: {report_path}")

    # 输出汇总
    print(f"\n{'='*60}")
    print("  汇总")
    print(f"{'='*60}\n")
    print(f"{'配置 (n_codes×n_days)':<25} {'native(s)':<12} {'vectorized(s)':<15} {'speedup':<10} {'precision':<12}")
    print("-" * 75)
    for r in results:
        cfg_str = f"{r['config']['n_codes']}×{r['config']['n_days']}"
        print(
            f"{cfg_str:<25} {r['native']['runtime_s']:<12.3f} "
            f"{r['vectorized']['runtime_s']:<15.3f} {r['speedup']:<10.2f} "
            f"{'PASS' if r['all_precision_passed'] else 'FAIL':<12}"
        )


if __name__ == "__main__":
    main()
