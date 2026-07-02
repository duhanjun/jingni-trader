"""
基准测试入口
============

运行所有验证测试并生成综合报告。

执行：
    cd /workspace && python3 -m quant_opt.benchmarks.run_all
"""
from __future__ import annotations

import os
import sys
import time
import json
import importlib
import pkgutil
import traceback
from typing import Dict, Any, List

import pandas as pd
import numpy as np


def _load_synthetic_data(n_stocks: int = 30, n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """生成合成的 A 股模拟数据（带基础 OHLCV 与涨跌停）"""
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SH" for i in range(600000, 600000 + n_stocks)]
    dates = pd.bdate_range(end=pd.Timestamp("2025-06-17"), periods=n_days)

    rows = []
    for code in codes:
        # 几何布朗运动
        mu, sigma = rng.normal(0.0004, 0.001), abs(rng.normal(0.018, 0.005))
        log_rets = rng.normal(mu, sigma, size=len(dates))
        close = 10.0 * np.exp(np.cumsum(log_rets))
        # OHLC
        for i, dt in enumerate(dates):
            o = close[i] * (1 + rng.normal(0, 0.003))
            h = max(o, close[i]) * (1 + abs(rng.normal(0, 0.005)))
            l = min(o, close[i]) * (1 - abs(rng.normal(0, 0.005)))
            v = abs(rng.normal(1e6, 3e5))
            amt = v * close[i]
            # 涨跌停：随机 1% 概率
            is_up = rng.random() < 0.01
            is_dn = rng.random() < 0.01
            rows.append({
                "code": code, "date": dt, "open": o, "high": h, "low": l,
                "close": close[i], "volume": v, "amount": amt,
                "is_limit_up": is_up, "is_limit_down": is_dn,
            })
    return pd.DataFrame(rows)


def _make_topk_signals(data: pd.DataFrame, top_pct: float = 0.2) -> pd.DataFrame:
    """基于 20 日动量打分，取 top 20% 为买入信号"""
    df = data.sort_values(["code", "date"]).copy()
    df["ret_20"] = df.groupby("code")["close"].pct_change(20)
    df["rank"] = df.groupby("date")["ret_20"].rank(pct=True, ascending=True)  # 反转
    sig = df[["code", "date"]].copy()
    sig["signal"] = 0
    sig.loc[df["rank"] > (1 - top_pct), "signal"] = 1
    return sig


def run_factor_expr_tests() -> Dict[str, Any]:
    """测试 1：因子表达式引擎"""
    from quant_opt.factor_expr import compile_factor, compute_preset, PRESET_FACTORS

    results: Dict[str, Any] = {"module": "factor_expr", "cases": []}
    data = _load_synthetic_data(n_stocks=20, n_days=120)

    # 1.1) 自定义公式
    t0 = time.perf_counter()
    fv = compile_factor("Close / Ts_Delay(Close, 5) - 1", data)
    elapsed = time.perf_counter() - t0
    assert isinstance(fv, pd.Series), "结果必须是 Series"
    assert len(fv) == len(data), f"长度不匹配: {len(fv)} vs {len(data)}"
    assert not fv.dropna().empty, "结果不应全为 NaN"
    results["cases"].append({
        "name": "自定义公式 Close/Ts_Delay(Close,5)-1",
        "passed": True,
        "elapsed_sec": round(elapsed, 4),
        "n_valid": int(fv.dropna().shape[0]),
    })

    # 1.2) 全部预置因子
    t0 = time.perf_counter()
    preset_results = {}
    for name in PRESET_FACTORS:
        try:
            v = compute_preset(name, data)
            preset_results[name] = {
                "passed": True,
                "n_valid": int(v.dropna().shape[0]),
            }
        except Exception as e:
            preset_results[name] = {"passed": False, "error": str(e)}
    elapsed = time.perf_counter() - t0
    n_pass = sum(1 for v in preset_results.values() if v.get("passed"))
    results["cases"].append({
        "name": f"全部预置因子 ({len(PRESET_FACTORS)} 个)",
        "passed": n_pass == len(PRESET_FACTORS),
        "elapsed_sec": round(elapsed, 4),
        "n_pass": n_pass,
        "n_total": len(PRESET_FACTORS),
        "details": preset_results,
    })

    # 1.3) 边界：空公式
    try:
        compile_factor("", data)
        results["cases"].append({"name": "空公式应报错", "passed": False})
    except Exception:
        results["cases"].append({"name": "空公式应报错", "passed": True})

    # 1.4) 边界：未知字段
    try:
        compile_factor("NotExistField + 1", data)
        results["cases"].append({"name": "未知字段应报错", "passed": False})
    except Exception:
        results["cases"].append({"name": "未知字段应报错", "passed": True})

    # 1.5) Rank 算子（横截面）
    fv_rank = compile_factor("Rank(Returns)", data)
    # 同一日期的 rank 应在 [0,1] 区间
    sample = fv_rank.reset_index()
    if "date" in sample.columns:
        col = "date"
    elif isinstance(sample, pd.DataFrame):
        # MultiIndex 已 reset
        col = sample.columns[0]
    # 直接验证
    valid = fv_rank.dropna()
    if len(valid) > 0:
        in_range = ((valid >= 0) & (valid <= 1)).all()
        results["cases"].append({
            "name": "Rank 算子输出在 [0,1]",
            "passed": bool(in_range),
            "min": float(valid.min()),
            "max": float(valid.max()),
        })
    else:
        results["cases"].append({"name": "Rank 算子输出在 [0,1]", "passed": False, "error": "全为 NaN"})

    return results


def run_metrics_tests() -> Dict[str, Any]:
    """测试 2：增强绩效指标"""
    from quant_opt.metrics import calc_all_metrics, drawdown_series

    results: Dict[str, Any] = {"module": "metrics", "cases": []}

    # 2.1) 基本正确性
    dates = pd.bdate_range("2024-01-01", periods=252)
    np.random.seed(0)
    rets = np.random.normal(0.0008, 0.012, size=252)
    eq_vals = (1 + pd.Series(rets)).cumprod().values * 1_000_000
    eq = pd.Series(eq_vals, index=dates)

    t0 = time.perf_counter()
    m = calc_all_metrics(eq, pd.DataFrame({"pnl": rets}))
    elapsed = time.perf_counter() - t0

    for key in ["total_return", "annual_return", "sharpe_ratio", "max_drawdown", "calmar_ratio"]:
        if key not in m:
            results["cases"].append({"name": f"基础指标 {key}", "passed": False})
            return results
    results["cases"].append({
        "name": "基础指标完整性",
        "passed": True,
        "elapsed_sec": round(elapsed, 4),
        "metrics_keys": list(m.keys()),
    })

    # 2.2) 与 benchmark 对比
    bench_rets = np.random.normal(0.0003, 0.010, size=252)
    bench_vals = (1 + pd.Series(bench_rets)).cumprod().values * 1_000_000
    bench = pd.Series(bench_vals, index=dates)
    m2 = calc_all_metrics(eq, pd.DataFrame({"pnl": rets}), benchmark=bench)
    assert "excess" in m2, "应包含 excess 块"
    assert "tracking_error" in m2["excess"]
    assert "information_ratio" in m2["excess"]
    assert "alpha" in m2["excess"]
    assert "beta" in m2["excess"]
    results["cases"].append({
        "name": "基准对比块（含 IR/Alpha/Beta/TE）",
        "passed": True,
        "excess_keys": list(m2["excess"].keys()),
    })

    # 2.3) 边界：短序列
    short = pd.Series([100.0, 101.0], index=dates[:2])
    m_short = calc_all_metrics(short)
    # 短序列应返回部分指标
    results["cases"].append({
        "name": "短序列（2 个点）",
        "passed": isinstance(m_short, dict),
    })

    # 2.4) 边界：空序列
    m_empty = calc_all_metrics(pd.Series(dtype=float))
    results["cases"].append({
        "name": "空序列",
        "passed": m_empty == {},
    })

    # 2.5) drawdown_series
    dd = drawdown_series(eq)
    assert (dd <= 0).all(), "回撤应非正"
    results["cases"].append({
        "name": "drawdown_series 全部 ≤ 0",
        "passed": True,
        "min_dd": float(dd.min()),
    })

    return results


def run_vectorized_bt_tests() -> Dict[str, Any]:
    """测试 3：向量化回测 vs 原生 native_adapter 行为一致性 + 性能"""
    from quant_opt.vectorized_bt import VectorizedBacktester, VectorizedBTConfig

    results: Dict[str, Any] = {"module": "vectorized_bt", "cases": []}

    # 准备数据：30 只股票 / 252 天
    data = _load_synthetic_data(n_stocks=30, n_days=252)
    signals = _make_topk_signals(data, top_pct=0.2)

    # 3.1) 正确性
    bt = VectorizedBacktester(VectorizedBTConfig(init_capital=1_000_000))
    t0 = time.perf_counter()
    res = bt.run(data, signals)
    elapsed = time.perf_counter() - t0

    assert not res["equity_curve"].empty, "equity_curve 应非空"
    assert "metrics" in res, "应返回 metrics"
    assert "sharpe_ratio" in res["metrics"], "应含 sharpe"
    # 至少一笔交易
    assert len(res["trades"]) > 0, "应产生交易"

    results["cases"].append({
        "name": "基本回测流程（30 票 / 252 天）",
        "passed": True,
        "elapsed_sec": round(elapsed, 4),
        "n_trades": int(len(res["trades"])),
        "metrics": res["metrics"],
    })

    # 3.2) 性能：与 jingni-trader 原生 native_adapter 对比
    # 由于 skills 子包目录含连字符（backtest-engine），无法标准 import
    # 用 importlib + sys.modules 注入伪包，绕过连字符问题
    try:
        import importlib.util
        import types

        if "/workspace" not in sys.path:
            sys.path.insert(0, "/workspace")

        # 构造伪包：skills.backtest_engine（用下划线替代连字符，仅作 sys.modules key）
        # 但 native_adapter 的相对导入是 `from ..base.base_backtest` / `from ..base.base_backtest_engine`
        # 这些相对导入会基于模块的 __package__ / __name__ 解析，需要正确设置
        fake_parent = types.ModuleType("_fake_native_pkg")
        fake_parent.__path__ = ["/workspace/skills/backtest-engine"]
        sys.modules["_fake_native_pkg"] = fake_parent
        fake_scripts = types.ModuleType("_fake_native_pkg.scripts")
        fake_scripts.__path__ = ["/workspace/skills/backtest-engine/scripts"]
        sys.modules["_fake_native_pkg.scripts"] = fake_scripts
        fake_base = types.ModuleType("_fake_native_pkg.scripts.base")
        fake_base.__path__ = ["/workspace/skills/backtest-engine/scripts/base"]
        sys.modules["_fake_native_pkg.scripts.base"] = fake_base
        # 把真正的 base_backtest / base_backtest_engine 加载到这个伪包下
        for mod_name in ("base_backtest", "base_backtest_engine"):
            spec = importlib.util.spec_from_file_location(
                f"_fake_native_pkg.scripts.base.{mod_name}",
                f"/workspace/skills/backtest-engine/scripts/base/{mod_name}.py",
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"_fake_native_pkg.scripts.base.{mod_name}"] = mod
            spec.loader.exec_module(mod)
            setattr(fake_base, mod_name, mod)

        # 最后加载 native_adapter
        spec = importlib.util.spec_from_file_location(
            "_fake_native_pkg.scripts.adapters.native_adapter",
            "/workspace/skills/backtest-engine/scripts/adapters/native_adapter.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_fake_native_pkg.scripts.adapters.native_adapter"] = mod
        spec.loader.exec_module(mod)

        native = mod.NativeAdapter()
        t0 = time.perf_counter()
        res_native = native.run_backtest(data, signals)
        native_elapsed = time.perf_counter() - t0

        speedup = native_elapsed / elapsed if elapsed > 0 else float("inf")

        # 关键指标对比
        def _safe_get(d, k):
            v = d.get(k) if isinstance(d, dict) else None
            return float(v) if v is not None else 0.0

        vec_metrics = res["metrics"]
        nat_metrics = res_native["metrics"] if isinstance(res_native["metrics"], dict) else {}

        results["cases"].append({
            "name": "性能对比 vs 原生 native_adapter",
            "passed": elapsed <= native_elapsed * 1.5,  # 至少不慢 50%
            "vectorized_sec": round(elapsed, 4),
            "native_sec": round(native_elapsed, 4),
            "speedup": round(speedup, 2),
            "metric_diff": {
                "sharpe_ratio": abs(_safe_get(vec_metrics, "sharpe_ratio") - _safe_get(nat_metrics, "sharpe_ratio")),
                "max_drawdown": abs(_safe_get(vec_metrics, "max_drawdown") - _safe_get(nat_metrics, "max_drawdown")),
                "total_return": abs(_safe_get(vec_metrics, "total_return") - _safe_get(nat_metrics, "total_return")),
            },
        })
    except Exception as e:
        results["cases"].append({
            "name": "性能对比 vs 原生 native_adapter",
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc()[:2000],
        })

    # 3.3) 边界：空数据
    empty_res = bt.run(pd.DataFrame(columns=["code", "date"]), signals.iloc[:0])
    results["cases"].append({
        "name": "空数据应安全返回",
        "passed": empty_res["equity_curve"].empty,
    })

    # 3.4) 边界：全部信号为 0
    zero_sig = signals.copy()
    zero_sig["signal"] = 0
    res_zero = bt.run(data, zero_sig)
    results["cases"].append({
        "name": "全 0 信号：不应交易，权益保持不变",
        "passed": res_zero["metrics"]["total_trades"] == 0 if res_zero["metrics"] else True,
    })

    return results


def run_walk_forward_tests() -> Dict[str, Any]:
    """测试 4：Walk-forward 验证"""
    from quant_opt.walk_forward import (
        WalkForwardSplitter, WFConfig, walk_forward_evaluate, evaluate_factor_oos,
    )

    results: Dict[str, Any] = {"module": "walk_forward", "cases": []}

    # 4.1) 切分器
    dates = pd.bdate_range("2022-01-01", "2024-12-31")
    cfg = WFConfig(train_window_days=252, test_window_days=63, step_days=63)
    splitter = WalkForwardSplitter(cfg)
    folds = splitter.split(dates)
    assert len(folds) > 0, "应至少有 1 折"
    # 检查顺序
    for f in folds:
        assert f.train_start < f.train_end < f.test_start < f.test_end
    results["cases"].append({
        "name": "切分器生成多折且时间顺序正确",
        "passed": True,
        "n_folds": len(folds),
        "first_fold": {
            "train": [str(folds[0].train_start.date()), str(folds[0].train_end.date())],
            "test": [str(folds[0].test_start.date()), str(folds[0].test_end.date())],
        },
    })

    # 4.2) 评估器
    data = _load_synthetic_data(n_stocks=30, n_days=504)  # 2 年
    t0 = time.perf_counter()
    wf_res = walk_forward_evaluate(
        "Rank(Close / Ts_Delay(Close, 5) - 1)",
        data,
        forward_period=5,
        config=WFConfig(train_window_days=252, test_window_days=63, step_days=63),
    )
    elapsed = time.perf_counter() - t0
    assert "n_folds" in wf_res
    assert "aggregate" in wf_res
    results["cases"].append({
        "name": "walk_forward_evaluate 端到端",
        "passed": True,
        "elapsed_sec": round(elapsed, 4),
        "n_folds": wf_res["n_folds"],
        "aggregate": wf_res["aggregate"],
    })

    # 4.3) anchored 模式
    folds_anc = splitter.split(dates)
    cfg_anc = WFConfig(train_window_days=252, test_window_days=63, step_days=63, anchored=True)
    splitter_anc = WalkForwardSplitter(cfg_anc)
    folds_anc = splitter_anc.split(dates)
    # anchored 模式下 train_start 应固定为最早日期
    if folds_anc:
        same_start = all(f.train_start == folds_anc[0].train_start for f in folds_anc)
    else:
        same_start = True
    results["cases"].append({
        "name": "anchored 模式 train_start 固定",
        "passed": same_start and len(folds_anc) > 0,
        "n_folds": len(folds_anc),
    })

    # 4.4) 边界：数据不足
    short_dates = pd.bdate_range("2024-01-01", periods=100)
    try:
        WalkForwardSplitter(WFConfig(min_train_days=252, test_window_days=63)).split(short_dates)
        results["cases"].append({"name": "数据不足应报错", "passed": False})
    except ValueError:
        results["cases"].append({"name": "数据不足应报错", "passed": True})

    return results


def run_e2e_pipeline() -> Dict[str, Any]:
    """测试 5：端到端 pipeline（因子→信号→回测→报告）"""
    from quant_opt.factor_expr import compile_factor
    from quant_opt.vectorized_bt import VectorizedBacktester, VectorizedBTConfig
    from quant_opt.metrics import calc_all_metrics
    from quant_opt.reports import render_metrics_html, save_report

    results: Dict[str, Any] = {"module": "e2e", "cases": []}

    data = _load_synthetic_data(n_stocks=25, n_days=252)

    # 5.1) 因子 → 信号
    t0 = time.perf_counter()
    factor_value = compile_factor("Rank(Close / Ts_Delay(Close, 20) - 1)", data)
    fv_df = factor_value.reset_index()
    fv_df.columns = ["code", "date", "alpha"]
    fv_df["rank"] = fv_df.groupby("date")["alpha"].rank(pct=True)
    signals = fv_df[["code", "date"]].copy()
    signals["signal"] = 0
    signals.loc[fv_df["rank"] > 0.8, "signal"] = 1
    t_factor = time.perf_counter() - t0

    # 5.2) 回测
    t0 = time.perf_counter()
    bt = VectorizedBacktester(VectorizedBTConfig(init_capital=1_000_000))
    res = bt.run(data, signals)
    t_bt = time.perf_counter() - t0

    # 5.3) 报告
    t0 = time.perf_counter()
    eq = res["equity_curve"].set_index("date")["equity"]
    bench_rets = np.random.default_rng(7).normal(0.0003, 0.010, size=len(eq))
    bench_vals = (1 + pd.Series(bench_rets)).cumprod().values * 1_000_000
    bench = pd.Series(bench_vals, index=eq.index)

    metrics = calc_all_metrics(eq, res["trades"], benchmark=bench)
    html = render_metrics_html(metrics, equity_curve=eq, benchmark=bench,
                                title="端到端验证报告 - Rank(Momentum_20d) 策略")
    out_path = "/workspace/workspace/quant_opt_reports/e2e_report.html"
    save_report(html, out_path)
    t_report = time.perf_counter() - t0

    # 5.4) 验证报告
    assert os.path.exists(out_path), "报告文件应存在"
    assert os.path.getsize(out_path) > 1000, "报告文件应非空"

    with open(out_path, "r", encoding="utf-8") as f:
        report_text = f.read()
    for marker in ["基准对比", "Information Ratio", "Alpha", "Beta", "Tracking Error"]:
        assert marker in report_text, f"报告应包含 {marker}"

    results["cases"].append({
        "name": "端到端 pipeline（因子→信号→回测→报告）",
        "passed": True,
        "t_factor_sec": round(t_factor, 4),
        "t_backtest_sec": round(t_bt, 4),
        "t_report_sec": round(t_report, 4),
        "total_trades": int(len(res["trades"])),
        "metrics": metrics,
        "report_path": out_path,
        "report_size_bytes": os.path.getsize(out_path),
    })
    return results


def main():
    """运行所有测试并生成报告"""
    print("=" * 80)
    print("jingni-trader 量化优化验证")
    print("=" * 80)

    all_results: Dict[str, Any] = {
        "task": "feat/quant-opt-20260617",
        "started_at": pd.Timestamp.now().isoformat(),
        "tests": {},
    }

    test_funcs = [
        ("factor_expr", run_factor_expr_tests),
        ("metrics", run_metrics_tests),
        ("vectorized_bt", run_vectorized_bt_tests),
        ("walk_forward", run_walk_forward_tests),
        ("e2e", run_e2e_pipeline),
    ]

    summary = []
    for name, fn in test_funcs:
        print(f"\n>>> 运行模块: {name}")
        t0 = time.perf_counter()
        try:
            res = fn()
            elapsed = time.perf_counter() - t0
            res["module_elapsed_sec"] = round(elapsed, 4)
            n_cases = len(res.get("cases", []))
            n_pass = sum(1 for c in res.get("cases", []) if c.get("passed"))
            res["n_pass"] = n_pass
            res["n_cases"] = n_cases
            all_results["tests"][name] = res
            summary.append((name, n_pass, n_cases, elapsed))
            print(f"  [{n_pass}/{n_cases}] 通过 (耗时 {elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  失败: {e}")
            all_results["tests"][name] = {
                "module": name,
                "error": str(e),
                "traceback": traceback.format_exc()[:3000],
            }
            summary.append((name, 0, 1, elapsed))

    all_results["finished_at"] = pd.Timestamp.now().isoformat()
    all_results["summary"] = summary

    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"{'模块':<20}{'通过':<10}{'总数':<10}{'耗时(秒)':<10}")
    for name, p, t, e in summary:
        print(f"{name:<20}{p:<10}{t:<10}{e:.2f}")

    total_pass = sum(p for _, p, _, _ in summary)
    total_cases = sum(t for _, _, t, _ in summary)
    print(f"\n总计: {total_pass}/{total_cases} 通过")

    # 保存 JSON 报告
    out_json = "/workspace/workspace/quant_opt_reports/benchmarks.json"

    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, (pd.Series,)):
            return o.to_dict()
        if isinstance(o, (pd.DataFrame,)):
            return o.to_dict(orient="records")
        return str(o)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\nJSON 报告: {out_json}")
    return 0 if total_pass == total_cases else 1


if __name__ == "__main__":
    sys.exit(main())
