"""
性能对比基准测试：向量化回测 vs 原生回测

对比 jingni-trader 现有 native_adapter.py 与优化后的 vectorized_engine.py
在不同数据规模下的性能差异
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimizations.vectorized_backtest import VectorizedBacktestEngine
from optimizations.enhanced_metrics import EnhancedMetrics
from optimizations.expression_factors import Alpha158FactorLibrary, VectorizedICAnalysis


def generate_data(n_codes, n_days, seed=42):
    """生成合成数据"""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rows = []
    for code in codes:
        price = np.random.uniform(10, 50)
        for dt in dates:
            ret = np.random.normal(0.0005, 0.02)
            price = max(price * (1 + ret), 1.0)
            open_p = price * (1 + np.random.normal(0, 0.005))
            high = max(price, open_p) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(price, open_p) * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.lognormal(15, 0.5))
            amount = price * volume
            pre_close = price / (1 + ret)
            change_pct = (price - pre_close) / pre_close * 100
            rows.append({
                "date": dt, "code": code,
                "open": round(open_p, 4), "high": round(high, 4),
                "low": round(low, 4), "close": round(price, 4),
                "volume": volume, "amount": round(amount, 2),
                "pre_close": round(pre_close, 4),
                "change_pct": round(change_pct, 4),
                "is_st": False,
                "is_limit_up": change_pct >= 9.9,
                "is_limit_down": change_pct <= -9.9,
            })
    return pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)


def generate_signals(data):
    """生成动量信号"""
    signals = []
    for code, group in data.groupby("code"):
        group = group.sort_values("date").copy()
        group["ma20"] = group["close"].rolling(20, min_periods=20).mean()
        group["signal"] = 0
        group.loc[group["close"] > group["ma20"], "signal"] = 1
        group.loc[group["close"] < group["ma20"], "signal"] = -1
        for _, row in group.iterrows():
            if row["signal"] != 0:
                signals.append({
                    "date": row["date"], "code": row["code"],
                    "signal": int(row["signal"]),
                })
    return pd.DataFrame(signals)


def run_benchmark():
    """运行性能基准测试"""
    print("=" * 70)
    print("性能对比基准测试：向量化回测引擎")
    print("=" * 70)

    results = []

    # 不同数据规模
    scenarios = [
        ("小规模", 10, 100),
        ("中规模", 30, 250),
        ("大规模", 50, 500),
    ]

    for name, n_codes, n_days in scenarios:
        print(f"\n--- 场景: {name} ({n_codes}只股票 × {n_days}天) ---")
        data = generate_data(n_codes, n_days)
        signals = generate_signals(data)
        n_rows = len(data)
        n_signals = len(signals)
        print(f"  数据行数: {n_rows}, 信号数: {n_signals}")

        # 向量化回测
        engine = VectorizedBacktestEngine()
        t0 = time.perf_counter()
        result = engine.run(data, signals)
        elapsed = time.perf_counter() - t0
        metrics = result["metrics"]
        print(f"  向量化回测耗时: {elapsed:.4f}s")
        print(f"    总收益: {metrics.get('total_return', 0):.4f}")
        print(f"    夏普: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"    最大回撤: {metrics.get('max_drawdown', 0):.4f}")
        print(f"    成交笔数: {metrics.get('total_trades', 0)}")

        results.append({
            "scenario": name,
            "n_codes": n_codes,
            "n_days": n_days,
            "n_rows": n_rows,
            "n_signals": n_signals,
            "vectorized_elapsed_s": round(elapsed, 4),
            "total_return": round(metrics.get("total_return", 0), 6),
            "sharpe": round(metrics.get("sharpe_ratio", 0), 4),
            "max_drawdown": round(metrics.get("max_drawdown", 0), 6),
            "total_trades": metrics.get("total_trades", 0),
        })

    # 因子计算性能
    print(f"\n--- Alpha158 因子计算性能 ---")
    data = generate_data(20, 250)
    lib = Alpha158FactorLibrary()
    print(f"  因子总数: {len(lib.list_factors())}")
    t0 = time.perf_counter()
    factors = lib.compute_all(data)
    factor_elapsed = time.perf_counter() - t0
    factor_cols = [c for c in factors.columns if c not in ("code", "date")]
    print(f"  计算耗时: {factor_elapsed:.4f}s")
    print(f"  成功因子数: {len(factor_cols)}")
    print(f"  输出形状: {factors.shape}")

    results.append({
        "scenario": "Alpha158因子计算",
        "n_codes": 20,
        "n_days": 250,
        "n_rows": len(data),
        "factor_elapsed_s": round(factor_elapsed, 4),
        "n_factors_total": len(lib.list_factors()),
        "n_factors_computed": len(factor_cols),
    })

    # 增强指标计算
    print(f"\n--- 增强指标计算 ---")
    eq_series = result["equity_curve"].set_index("date")["equity"]
    mc = EnhancedMetrics()
    t0 = time.perf_counter()
    all_metrics = mc.calc_all(eq_series, result["trades"])
    metrics_elapsed = time.perf_counter() - t0
    print(f"  指标总数: {len(all_metrics)}")
    print(f"  计算耗时: {metrics_elapsed:.4f}s")
    print(f"  指标列表:")
    for k, v in sorted(all_metrics.items()):
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")

    return results, all_metrics


if __name__ == "__main__":
    results, metrics = run_benchmark()

    # 保存结果
    output = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "benchmark_results": results,
        "enhanced_metrics_sample": {k: float(v) if isinstance(v, (int, float)) else v
                                     for k, v in metrics.items()},
    }
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "reports", "benchmark_results.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n基准测试结果已保存至: {output_path}")
