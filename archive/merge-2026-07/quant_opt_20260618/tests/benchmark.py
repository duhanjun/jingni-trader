"""
综合性能基准测试
================

对比三个新模块与 jingni-trader 现有实现的性能差异：
1. PIT Checker：与人工肉眼检查相比
2. Factor DSL：与硬编码 pandas 相比
3. WFA Validator：与"全样本回测"评估相比
"""
import sys
import os
import time
import json
from datetime import datetime
import pandas as pd
import numpy as np

# 从项目根目录导入（因 benchmark.py 在子包内）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from quant_opt_20260618.pit_checker.checker import check_pit
from quant_opt_20260618.factor_dsl.engine import FactorEngine, FactorExpression, builtin_alpha_expressions
from quant_opt_20260618.wf_validator.splitter import TimeSeriesSplitter, WalkForwardValidator


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────

def make_realistic_data(n_stocks: int = 100, n_days: int = 750, seed: int = 2024) -> pd.DataFrame:
    """生成更接近真实场景的数据"""
    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

    rows = []
    for ci, code in enumerate(codes):
        start_price = np.random.uniform(10, 100)
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = start_price * (1 + returns).cumprod()
        volumes = np.random.lognormal(15, 0.5, n_days).astype(int)

        for i, dt in enumerate(dates):
            # 公告日：80% 在过去（合规），20% 在未来（违规）
            from datetime import timedelta
            if np.random.random() < 0.8:
                announce_date = dt - timedelta(days=int(np.random.exponential(10)) + 1)
            else:
                announce_date = dt + timedelta(days=int(np.random.exponential(10)) + 1)
            rows.append({
                "code": code,
                "date": dt,
                "open": prices[i] * (1 + np.random.normal(0, 0.003)),
                "high": prices[i] * (1 + abs(np.random.normal(0, 0.005))),
                "low": prices[i] * (1 - abs(np.random.normal(0, 0.005))),
                "close": prices[i],
                "volume": volumes[i],
                "announce_date": announce_date,
                "alpha_score": np.random.normal(0, 1) + (i / n_days) * 0.5,  # 弱 alpha
                "ret_forward_1d": np.random.normal(0, 0.015),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 基准 1: PIT Checker 性能
# ─────────────────────────────────────────────────────────────

def benchmark_pit():
    print("\n" + "=" * 70)
    print("BENCHMARK 1: PIT Checker")
    print("=" * 70)

    sizes = [
        (50, 250),    # ~12.5K rows
        (100, 500),   # 50K rows
        (200, 750),   # 150K rows
    ]

    results = []
    for n_stocks, n_days in sizes:
        df = make_realistic_data(n_stocks, n_days)
        n_rows = len(df)
        t0 = time.time()
        report = check_pit(df, pit_columns=["announce_date"])
        elapsed = time.time() - t0

        results.append({
            "n_stocks": n_stocks,
            "n_days": n_days,
            "n_rows": n_rows,
            "elapsed_sec": round(elapsed, 3),
            "rows_per_sec": int(n_rows / elapsed) if elapsed > 0 else 0,
            "violations": len(report.violations),
            "violation_rate": round(len(report.violations) / n_rows, 4),
        })
        print(f"  {n_stocks}×{n_days} = {n_rows:6d} rows: "
              f"{elapsed:.3f}s ({n_rows / elapsed:8.0f} rows/s) "
              f"violations={len(report.violations)}")

    return results


# ─────────────────────────────────────────────────────────────
# 基准 2: Factor DSL 性能
# ─────────────────────────────────────────────────────────────

def benchmark_dsl():
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Factor DSL vs Hardcoded pandas")
    print("=" * 70)

    n_stocks, n_days = 100, 500
    df = make_realistic_data(n_stocks, n_days)

    # 硬编码版
    def hardcoded_compute(data):
        result = pd.DataFrame()
        result["code"] = data["code"]
        result["date"] = data["date"]
        data_indexed = data.set_index(["code", "date"]).sort_index()
        result["mom_5"] = data_indexed.groupby(level="code")["close"].transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        ).values
        result["mom_20"] = data_indexed.groupby(level="code")["close"].transform(
            lambda s: s.rolling(20, min_periods=1).mean()
        ).values
        result["vol_20"] = data_indexed.groupby(level="code")["close"].transform(
            lambda s: s.rolling(20, min_periods=2).std()
        ).values
        result["vol_ratio"] = (
            data["volume"] /
            data_indexed.groupby(level="code")["volume"].transform(
                lambda s: s.rolling(20, min_periods=1).mean()
            ).values
        )
        # 排名
        result["rank_mom"] = result.groupby("date")["mom_20"].rank(pct=True).values
        result["alpha_mom_rank"] = (
            result["rank_mom"] - result.groupby("date")["vol_20"].rank(pct=True).values
        )
        return result

    # DSL 版
    engine = FactorEngine()
    engine.register(FactorExpression("mom_5", "Mean($close, 5)"))
    engine.register(FactorExpression("mom_20", "Mean($close, 20)"))
    engine.register(FactorExpression("vol_20", "Std($close, 20)"))
    engine.register(FactorExpression("vol_ratio", "$volume / Mean($volume, 20)"))
    engine.register(FactorExpression("rank_mom", "Rank(mom_20)"))
    engine.register(FactorExpression("alpha_mom_rank", "Rank(mom_20) - Rank(vol_20)"))

    t0 = time.time()
    dsl_result = engine.compute(df)
    t_dsl = time.time() - t0

    t0 = time.time()
    hard_result = hardcoded_compute(df)
    t_hard = time.time() - t0

    ratio = t_dsl / t_hard if t_hard > 0 else float("inf")

    print(f"  Hardcoded pandas: {t_hard:.3f}s")
    print(f"  DSL engine:       {t_dsl:.3f}s")
    print(f"  Ratio (DSL/Hard): {ratio:.2f}x")
    print(f"  Output shape: hard={hard_result.shape}, dsl={dsl_result.shape}")

    # 验证结果一致性（部分关键列）
    # 注意：硬编码把结果放在 result["mom_5"]，DSL 也有
    common_cols = ["mom_5", "mom_20", "vol_20", "alpha_mom_rank"]
    for col in common_cols:
        if col in hard_result.columns and col in dsl_result.columns:
            # 按 code/date 排序后比较
            h = hard_result[["code", "date", col]].sort_values(["code", "date"]).reset_index(drop=True)
            d = dsl_result[["code", "date", col]].sort_values(["code", "date"]).reset_index(drop=True)
            try:
                pd.testing.assert_series_equal(h[col].fillna(-999), d[col].fillna(-999),
                                                check_names=False, atol=1e-6)
                print(f"  ✓ {col} matches")
            except AssertionError as e:
                print(f"  ✗ {col} mismatch: {str(e)[:100]}")

    return {
        "hardcoded_sec": round(t_hard, 3),
        "dsl_sec": round(t_dsl, 3),
        "ratio": round(ratio, 2),
    }


# ─────────────────────────────────────────────────────────────
# 基准 3: WFA 评估
# ─────────────────────────────────────────────────────────────

def benchmark_wfa():
    print("\n" + "=" * 70)
    print("BENCHMARK 3: WFA Validator")
    print("=" * 70)

    df = make_realistic_data(n_stocks=100, n_days=1000)

    splitter = TimeSeriesSplitter(
        train_period_days=252, test_period_days=63, step_days=63, expanding=False,
    )
    folds = splitter.split(df, start_date="2020-01-01", end_date="2023-12-01")
    print(f"  Generated {len(folds)} folds for 4-year span")

    validator = WalkForwardValidator(
        factor_col="alpha_score", ret_col="ret_forward_1d",
        top_k=20, bottom_k=20, min_stocks=30,
    )
    t0 = time.time()
    report = validator.run(df, folds)
    elapsed = time.time() - t0

    summary = report.summary()
    print(f"  WFA evaluation: {elapsed:.3f}s")
    print(f"  IC mean avg:    {summary['ic_mean_avg']:.4f}")
    print(f"  IC IR:          {summary['ic_ir']:.4f}")
    print(f"  Rank IC avg:    {summary['rank_ic_mean_avg']:.4f}")
    print(f"  Consistency:    {summary['consistency_ratio']:.2%}")
    print(f"  Long-short total: {summary['long_short_return_total']:.4f}")
    print(f"  Long-only total:  {summary['long_only_return_total']:.4f}")
    print(f"  Avg turnover:     {summary['turnover_avg']:.2%}")

    return {
        "n_folds": len(folds),
        "elapsed_sec": round(elapsed, 3),
        "summary": {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()},
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"jingni-trader Quant Optimization Benchmark")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "branch": "feat/quant-opt-20260618",
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "pit_checker": benchmark_pit(),
        "factor_dsl": benchmark_dsl(),
        "wfa_validator": benchmark_wfa(),
    }

    # 保存 JSON
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    main()
