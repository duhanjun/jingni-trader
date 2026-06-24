"""
一键运行所有测试 + 性能对比 + 总结输出

执行：python -m quant_opt_experiments.run_all
"""
import sys
import time
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# 让 import 找到本包
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_opt_experiments.tests.fixtures import make_synthetic_panel
from quant_opt_experiments.factor_expression_engine import FactorEngine, register_alpha158_pv
from quant_opt_experiments.vectorized_backtest import (
    vectorized_backtest_multi, ma_cross_signals, rank_topk_signals, calc_metrics,
)
from quant_opt_experiments.walk_forward_eval import (
    analyze_all_factors, walk_forward,
)


REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_factor_engine(panel):
    """对比 表达式引擎 vs 手动 groupby 计算耗时"""
    engine = FactorEngine(panel)
    factors = ["MA5", "MA20", "MA60", "ROC5", "ROC20", "STD5", "STD20"]
    for f in factors:
        engine.register(f, {
            "MA5": "Mean($close, 5)",
            "MA20": "Mean($close, 20)",
            "MA60": "Mean($close, 60)",
            "ROC5": "$close / Ref($close, 5) - 1",
            "ROC20": "$close / Ref($close, 20) - 1",
            "STD5": "Std($close, 5) / $close",
            "STD20": "Std($close, 20) / $close",
        }[f])

    t0 = time.time()
    for f in factors:
        engine.compute(f)
    t_engine = time.time() - t0

    # 手动 groupby
    t0 = time.time()
    panel_sorted = panel.sort_values(["code", "date"]).copy()
    for code, sub in panel_sorted.groupby("code"):
        sub["ma5"] = sub["close"].rolling(5).mean()
        sub["ma20"] = sub["close"].rolling(20).mean()
        sub["ma60"] = sub["close"].rolling(60).mean()
        sub["roc5"] = sub["close"] / sub["close"].shift(5) - 1
        sub["roc20"] = sub["close"] / sub["close"].shift(20) - 1
    t_loop = time.time() - t0
    return {
        "engine_ms": t_engine * 1000,
        "loop_ms": t_loop * 1000,
        "speedup": t_loop / max(t_engine, 1e-6),
    }


def benchmark_backtest(panel, close_pivot):
    """矢量化 vs native_adapter"""
    import sys, os
    # 用 CWD 而不是 __file__ 来定位项目根目录（避免 module 运行时的路径问题）
    cwd = Path(os.getcwd())
    # 向上找包含 skills/ 的目录
    project_root = cwd
    while project_root.parent != project_root:
        if (project_root / "skills" / "backtest-engine").exists():
            break
        project_root = project_root.parent
    bt_path = project_root / "skills" / "backtest-engine"

    if str(bt_path) not in sys.path:
        sys.path.insert(0, str(bt_path))
    NativeAdapter = None
    try:
        from scripts.adapters.native_adapter import NativeAdapter  # type: ignore
    except Exception:
        # fallback: 用 importlib 但需要构造正确的 parent module context
        import types
        import importlib.util
        scripts_pkg = types.ModuleType("scripts")
        scripts_pkg.__path__ = [str(bt_path / "scripts")]
        sys.modules["scripts"] = scripts_pkg
        adapters_sub = types.ModuleType("scripts.adapters")
        adapters_sub.__path__ = [str(bt_path / "scripts" / "adapters")]
        sys.modules["scripts.adapters"] = adapters_sub
        base_pkg = types.ModuleType("scripts.adapters.base")
        sys.modules["scripts.adapters.base"] = base_pkg
        # 加载 base
        base_file = bt_path / "scripts" / "adapters" / "base" / "base_backtest_engine.py"
        if base_file.exists():
            spec = importlib.util.spec_from_file_location(
                "scripts.adapters.base.base_backtest_engine",
                base_file,
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["scripts.adapters.base.base_backtest_engine"] = mod
                spec.loader.exec_module(mod)
                base_pkg.BaseBacktestEngine = mod.BaseBacktestEngine
        # 加载 native_adapter
        na_file = bt_path / "scripts" / "adapters" / "native_adapter.py"
        spec = importlib.util.spec_from_file_location(
            "scripts.adapters.native_adapter",
            na_file,
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["scripts.adapters.native_adapter"] = mod
            spec.loader.exec_module(mod)
            NativeAdapter = mod.NativeAdapter

    if NativeAdapter is None:
        return {"error": "native_adapter 加载失败"}

    try:
        close = close_pivot
        factor = -close.pct_change(5)
        entries, exits = rank_topk_signals(factor, top_pct=0.34, hold_days=20)

        # 矢量化
        t0 = time.time()
        res_v = vectorized_backtest_multi(
            close, entries, exits, alloc_per_asset=0.15, init_cash=1_000_000,
        )
        t_v = time.time() - t0

        # native
        data_long = panel[panel["date"].isin(close.index)].copy()
        sig_rows = []
        for dt in close.index:
            for code in close.columns:
                sig_rows.append({
                    "date": dt, "code": code,
                    "signal": 1 if entries.loc[dt, code] else (-1 if exits.loc[dt, code] else 0)
                })
        sig_df = pd.DataFrame(sig_rows)
        adapter = NativeAdapter()
        t0 = time.time()
        res_n = adapter.run_backtest(data=data_long, signals=sig_df, init_capital=1_000_000)
        t_n = time.time() - t0
        n_metrics = {}
        if isinstance(res_n, dict):
            n_metrics = res_n.get("metrics", {})
        elif hasattr(res_n, "metrics"):
            n_metrics = res_n.metrics
        return {
            "vectorized_ms": t_v * 1000,
            "native_ms": t_n * 1000,
            "speedup": t_n / max(t_v, 1e-6),
            "vectorized_metrics": res_v.metrics,
            "native_metrics": n_metrics,
        }
    except Exception as e:
        return {"error": f"benchmark_backtest 异常: {e}"}


def run_walk_forward_eval(panel, close_pivot):
    engine = FactorEngine(panel)
    register_alpha158_pv(engine)
    factor_long = engine.compute_all()
    factor_cols = [c for c in factor_long.columns if c not in ("date", "code")]

    df_ic = analyze_all_factors(factor_long, close_pivot, factor_cols, forward_periods=(1, 5, 20))
    df_ic_sorted = df_ic.sort_values(["forward", "ic_ir"], ascending=[True, False])

    # walk forward for top factor
    top_factor = df_ic_sorted.iloc[0]["factor"]
    folds = walk_forward(factor_long, close_pivot, top_factor,
                         train_months=4, test_months=2, forward=5)
    return {
        "ic_table": df_ic,
        "top_factor": top_factor,
        "wf_folds": folds,
    }


def main():
    print("=" * 70)
    print("jingni-trader 量化优化实验 · 验证报告")
    print("=" * 70)

    panel = make_synthetic_panel(n_stocks=6, n_days=500, seed=42)
    close = panel.pivot(index="date", columns="code", values="close").sort_index()
    print(f"\n📊 合成数据: {panel['code'].nunique()} 只股票 × {panel['date'].nunique()} 个交易日")
    print(f"   范围: {panel['date'].min().date()} ~ {panel['date'].max().date()}")

    # ---- 1) 因子表达式引擎 ----
    print("\n" + "=" * 70)
    print("1️⃣  因子表达式引擎 (Qlib 风格) 性能测试")
    print("=" * 70)
    perf1 = benchmark_factor_engine(panel)
    print(f"   表达式引擎: {perf1['engine_ms']:.1f} ms")
    print(f"   手动 groupby: {perf1['loop_ms']:.1f} ms")
    print(f"   加速比: {perf1['speedup']:.1f}x")

    # ---- 2) 矢量化回测 ----
    print("\n" + "=" * 70)
    print("2️⃣  矢量化回测引擎 (vectorbt 风格) 性能测试")
    print("=" * 70)
    perf2 = benchmark_backtest(panel, close)
    if "error" in perf2:
        print(f"   ⚠️ 跳过 native 对比: {perf2['error']}")
        # 仍然跑矢量化版本单独
        from quant_opt_experiments.vectorized_backtest import vectorized_backtest_multi
        from quant_opt_experiments.tests.fixtures import make_synthetic_panel as _ms
        import time as _t
        factor = -close.pct_change(5)
        from quant_opt_experiments.vectorized_backtest import rank_topk_signals
        entries, exits = rank_topk_signals(factor, top_pct=0.34, hold_days=20)
        t0 = _t.time()
        res_v = vectorized_backtest_multi(close, entries, exits, alloc_per_asset=0.15, init_cash=1_000_000)
        t_v = _t.time() - t0
        perf2 = {
            "vectorized_ms": t_v * 1000,
            "native_ms": float("nan"),
            "speedup": float("nan"),
            "vectorized_metrics": res_v.metrics,
            "native_metrics": {},
            "note": "native_adapter 加载失败，仅跑矢量化版本",
        }
    print(f"   矢量化版本: {perf2['vectorized_ms']:.1f} ms")
    if not np.isnan(perf2.get('native_ms', float('nan'))):
        print(f"   native_adapter: {perf2['native_ms']:.1f} ms")
        print(f"   加速比: {perf2['speedup']:.1f}x")
    print(f"   矢量化指标: {perf2['vectorized_metrics']}")
    print(f"   native 指标: {perf2['native_metrics']}")

    # ---- 3) IC + walk-forward ----
    print("\n" + "=" * 70)
    print("3️⃣  因子 IC 稳定性 + Walk-Forward 评估")
    print("=" * 70)
    wf = run_walk_forward_eval(panel, close)
    print(f"   评估因子数: {wf['ic_table']['factor'].nunique()}")
    print(f"   Top 5 因子 (按 5d forward IR):")
    top5 = wf["ic_table"][wf["ic_table"]["forward"] == 5].sort_values("ic_ir", ascending=False).head(5)
    print(top5[["factor", "ic_mean", "ic_ir", "ic_pos_ratio"]].to_string(index=False))
    print(f"\n   Walk-Forward on '{wf['top_factor']}': {len(wf['wf_folds'])} 折")
    for f in wf["wf_folds"][:6]:
        print(f"     train[{f.train_start}~{f.train_end}] IC={f.train_ic:+.3f} | "
              f"test[{f.test_start}~{f.test_end}] IC={f.test_ic:+.3f} | "
              f"Sharpe train={f.train_sharpe:+.2f} test={f.test_sharpe:+.2f}")

    # ---- 写报告 ----
    report_path = REPORT_DIR / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_shape": {
                "stocks": panel["code"].nunique(),
                "days": panel["date"].nunique(),
            },
            "expr_engine_benchmark": perf1,
            "vectorized_bt_benchmark": perf2,
            "ic_top_factors": top5.to_dict(orient="records"),
            "wf_summary": {
                "factor": wf["top_factor"],
                "n_folds": len(wf["wf_folds"]),
                "folds": [f.__dict__ for f in wf["wf_folds"]],
            },
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 验证结果已保存到: {report_path}")

    # IC 表保存为 CSV
    wf["ic_table"].to_csv(REPORT_DIR / "factor_ic_table.csv", index=False, encoding="utf-8-sig")
    print(f"✅ IC 表已保存到: {REPORT_DIR / 'factor_ic_table.csv'}")


if __name__ == "__main__":
    main()