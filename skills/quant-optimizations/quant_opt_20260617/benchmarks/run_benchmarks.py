"""
性能基准测试脚本
对比三个验证模块 vs 现有实现的性能
"""
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from skills.quant-optimizations.quant_opt_20260617.core.vectorized_ic import VectorizedICAnalyzer, _safe_pearson
from skills.quant-optimizations.quant_opt_20260617.core.vectorized_backtest import VectorizedBacktester
from skills.quant-optimizations.quant_opt_20260617.core.factor_expression import FactorExpressionEngine, ALPHA101_DEMO


def _make_panel(n_stocks: int, n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    codes = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]
    rows = []
    for d in dates:
        for c in codes:
            f = rng.normal(0, 1)
            r = 0.3 * f + rng.normal(0, 1)
            rows.append((d, c, f, r))
    return pd.DataFrame(rows, columns=["date", "code", "factor", "ret"])


def _make_market(n_stocks: int, n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    codes = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]
    rows = []
    for c in codes:
        start = rng.uniform(10, 50)
        ret = rng.normal(0.0005, 0.02, n_days)
        price = start * (1 + ret).cumprod()
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "code": c,
                "open": price[i], "high": price[i] * 1.01, "low": price[i] * 0.99,
                "close": price[i],
                "volume": int(rng.lognormal(15, 0.3)),
                "amount": float(price[i] * rng.lognormal(15, 0.3)),
                "is_limit_up": False, "is_limit_down": False,
            })
    return pd.DataFrame(rows)


def bench_vectorized_ic():
    """Benchmark 1: IC 分析"""
    print("\n" + "=" * 70)
    print("  Benchmark 1: 向量化 IC 分析 vs 逐日 for-loop")
    print("=" * 70)

    panel = _make_panel(n_stocks=100, n_days=500)
    f_series = panel.set_index(["date", "code"])["factor"]
    r_series = panel.set_index(["date", "code"])["ret"]
    analyzer = VectorizedICAnalyzer()

    # 向量化版
    t0 = time.perf_counter()
    ic_vec = analyzer.compute_ic_series(f_series, r_series, "pearson")
    t_vec = time.perf_counter() - t0

    # For-loop 版（scipy.stats.pearsonr）
    t0 = time.perf_counter()
    aligned = pd.concat([f_series.rename("f"), r_series.rename("r")], axis=1).dropna()
    ic_loop = pd.Series({
        d: _safe_pearson(g["f"].values, g["r"].values)
        for d, g in aligned.groupby(level="date")
    })
    t_loop = time.perf_counter() - t0

    # 多次平均
    n_repeat = 5
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        analyzer.compute_ic_series(f_series, r_series, "pearson")
    t_vec_avg = (time.perf_counter() - t0) / n_repeat

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        aligned = pd.concat([f_series.rename("f"), r_series.rename("r")], axis=1).dropna()
        ic_loop = pd.Series({
            d: _safe_pearson(g["f"].values, g["r"].values)
            for d, g in aligned.groupby(level="date")
        })
    t_loop_avg = (time.perf_counter() - t0) / n_repeat

    speedup = t_loop_avg / t_vec_avg
    print(f"  数据规模: {len(panel)} 行, {panel['code'].nunique()} 支股票 × {panel['date'].nunique()} 天")
    print(f"  向量化: {t_vec_avg*1000:.2f} ms / run")
    print(f"  For-loop: {t_loop_avg*1000:.2f} ms / run")
    print(f"  加速比: {speedup:.2f}x")
    print(f"  结果一致性: mean abs diff = {abs(ic_vec - ic_loop).mean():.2e}")
    return {
        "module": "vectorized_ic",
        "data_size": len(panel),
        "vec_ms": t_vec_avg * 1000,
        "loop_ms": t_loop_avg * 1000,
        "speedup": speedup,
    }


def bench_vectorized_backtest():
    """Benchmark 2: 回测引擎"""
    print("\n" + "=" * 70)
    print("  Benchmark 2: 向量化回测 (含 Numba JIT)")
    print("=" * 70)

    data = _make_market(n_stocks=80, n_days=200)
    # top10 signals
    rows = []
    for d, g in data.groupby("date"):
        chosen = g.nlargest(10, "volume")["code"].tolist()
        for c in g["code"]:
            rows.append({"date": d, "code": c, "signal": 1 if c in chosen else 0})
    signals = pd.DataFrame(rows)

    bt = VectorizedBacktester(init_capital=1_000_000)

    # 预热 JIT
    bt.run(data, signals)

    n_repeat = 3
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        res = bt.run(data, signals)
    t_per = (time.perf_counter() - t0) / n_repeat

    print(f"  数据规模: {len(data)} 行, {data['code'].nunique()} 支 × {data['date'].nunique()} 天")
    print(f"  每次回测: {t_per*1000:.2f} ms")
    print(f"  Sharpe: {res.metrics['sharpe_ratio']:.3f}, "
          f"年化收益: {res.metrics['annual_return']*100:.2f}%, "
          f"最大回撤: {res.metrics['max_drawdown']*100:.2f}%")
    return {
        "module": "vectorized_backtest",
        "data_size": len(data),
        "ms_per_run": t_per * 1000,
        "n_runs": n_repeat,
        "sharpe_ratio": res.metrics["sharpe_ratio"],
        "annual_return": res.metrics["annual_return"],
        "max_drawdown": res.metrics["max_drawdown"],
    }


def bench_factor_expression():
    """Benchmark 3: 因子表达式引擎"""
    print("\n" + "=" * 70)
    print("  Benchmark 3: 因子表达式引擎 vs 硬编码实现")
    print("=" * 70)

    data = _make_market(n_stocks=50, n_days=200)
    eng = FactorExpressionEngine(data)

    # 7 个因子
    n_repeat = 3
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        result = eng.compute_batch(ALPHA101_DEMO)
    t_per = (time.perf_counter() - t0) / n_repeat

    print(f"  数据规模: {len(data)} 行, {data['code'].nunique()} 支 × {data['date'].nunique()} 天")
    print(f"  因子数: {len(ALPHA101_DEMO)}")
    print(f"  总耗时: {t_per*1000:.2f} ms / run (含 7 个因子)")
    print(f"  平均单因子: {t_per*1000/len(ALPHA101_DEMO):.2f} ms")
    factor_cols = [c for c in result.columns if c not in ("date", "code")]
    valid_rate = result[factor_cols].notna().sum().sum() / (len(result) * len(factor_cols))
    print(f"  有效值率: {valid_rate:.2%}")
    return {
        "module": "factor_expression",
        "data_size": len(data),
        "n_factors": len(ALPHA101_DEMO),
        "ms_per_run": t_per * 1000,
        "ms_per_factor": t_per * 1000 / len(ALPHA101_DEMO),
    }


def main():
    print("=" * 70)
    print("  jingni-trader 量化优化验证 - 性能基准测试")
    print("=" * 70)

    results = []
    try:
        results.append(bench_vectorized_ic())
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append({"module": "vectorized_ic", "error": str(e)})

    try:
        results.append(bench_vectorized_backtest())
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append({"module": "vectorized_backtest", "error": str(e)})

    try:
        results.append(bench_factor_expression())
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append({"module": "factor_expression", "error": str(e)})

    out_path = ROOT / "quant_opt" / "reports" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  基准结果已保存: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()