"""
统一测试入口 —— 一次性验证 4 个优化模块。

测试流程:
  T1  Expression Engine  : 注册 Qlib 风格因子,与硬编码版本对比
  T2  Look-Ahead Detector: 扫描 jingni-trader 主代码,报告前视偏差
  T3  Vectorized IC Engine: 与 factor-engine 的 _calc_ic 对比性能
  T4  Walk-Forward Validator: 在合成数据上跑滚动训练

所有测试都生成 assert + 性能数据,失败时立即打印堆栈。
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # /workspace

from quant_opt.expression_engine.expression import (
    ExpressionEngine, ExpressionParser, register_factor,
    ALPHA158_TECHNICAL_SUBSET, compute_alpha158_subset,
)
from quant_opt.look_ahead_detector.detector import LookAheadDetector
from quant_opt.ic_optimizer.ic_engine import VectorizedICEngine
from quant_opt.walk_forward.validator import WalkForwardValidator


# ===========================================================================
# 数据生成工具
# ===========================================================================
def make_synthetic_data(
    n_stocks: int = 50,
    n_days: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成带"真实"统计结构的 A 股日线数据。
    - 收盘价 = 几何布朗运动 + 微弱动量
    - 成交量 = 价格变化率的反向 + 随机
    """
    rng = np.random.default_rng(seed)
    base_dates = pd.bdate_range("2022-01-01", periods=n_days)
    rows = []
    for i in range(n_stocks):
        mu = rng.normal(0.0003, 0.001)
        sigma = rng.uniform(0.015, 0.03)
        ret = rng.normal(mu, sigma, n_days)
        # 注入动量
        ret = ret + 0.05 * np.concatenate([[0], ret[:-1]])
        ret = ret + 0.03 * np.concatenate([[0, 0], ret[:-2]])
        close = 10 * np.exp(np.cumsum(ret))
        # 开高低
        open_ = close * (1 + rng.normal(0, 0.005, n_days))
        high  = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.008, n_days)))
        low   = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.008, n_days)))
        volume = rng.integers(1_000_000, 50_000_000, n_days).astype(float)
        amount = volume * close
        for j in range(n_days):
            rows.append({
                "date": base_dates[j],
                "code": f"{i:06d}.SZ",
                "open": open_[j], "high": high[j], "low": low[j],
                "close": close[j], "volume": volume[j], "amount": amount[j],
                "turnover_rate": volume[j] / 1e8,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# T1 — 表达式引擎
# ===========================================================================
def test_expression_engine(data: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("[T1] 表达式引擎 (借鉴 Qlib) — 一行代码注册 23 个 Alpha158 因子")
    print("=" * 72)

    # 1) 解析能力测试
    exprs = [
        "Mean($close, 5)",
        "Ref($close, 1) / $close - 1",
        "Rank(($close - Ref($close, 1)) / Ref($close, 1))",
        "($close - Mean($close, 20)) / Std($close, 20)",
        "Slope($close, 20)",
    ]
    parse_results = []
    for e in exprs:
        op = ExpressionParser.parse(e)
        parse_results.append({"expr": e, "type": type(op).__name__})

    # 2) 性能: 23 个因子
    t0 = time.perf_counter()
    result_df = compute_alpha158_subset(data)
    elapsed = time.perf_counter() - t0

    assert not result_df.empty, "Alpha158 因子结果为空"
    assert result_df.shape[1] == 2 + len(ALPHA158_TECHNICAL_SUBSET), \
        f"列数 {result_df.shape[1]} != 期望 {2 + len(ALPHA158_TECHNICAL_SUBSET)}"
    assert result_df["date"].is_monotonic_increasing
    # 检查没有全为 NaN
    non_na = result_df.iloc[:, 2:].notna().sum().sum()
    assert non_na > 0, "所有因子全为 NaN"

    print(f"  ✓ 成功解析 {len(exprs)} 个表达式,示例类型: {[r['type'] for r in parse_results]}")
    print(f"  ✓ Alpha158 子集 ({len(ALPHA158_TECHNICAL_SUBSET)} 因子) 计算耗时: {elapsed:.3f}s")
    print(f"    结果形状: {result_df.shape}, 非空数: {non_na}")

    # 3) 与"硬编码"结果对比: 重算 MA20,必须与原版 groupby().rolling(20).mean() 一致
    legacy_indexed = data.set_index(['date', 'code']).sort_index()
    legacy_ma20 = legacy_indexed['close'].groupby(level='code').rolling(20, min_periods=10).mean()
    legacy_ma20 = legacy_ma20.droplevel(0).sort_index()
    legacy_ma20.index.set_names(['date', 'code'], inplace=True)

    new_ma20 = register_factor(data, "Mean($close, 20)", name="MA20")
    new_ma20 = new_ma20.sort_index()
    # 确保两者索引名一致
    new_ma20.index.set_names(['date', 'code'], inplace=True)
    common = new_ma20.index.intersection(legacy_ma20.index)
    diff = (new_ma20.loc[common] - legacy_ma20.loc[common]).abs().max()
    print(f"  ✓ 与硬编码 groupby().rolling(20).mean() 最大差异: {diff:.2e}")
    assert diff < 1e-8, f"差异过大: {diff}"

    return {
        "n_parsed": len(exprs),
        "n_factors": len(ALPHA158_TECHNICAL_SUBSET),
        "elapsed_s": round(elapsed, 3),
        "n_rows": len(result_df),
        "max_diff_vs_legacy": float(diff),
        "status": "PASS",
    }


# ===========================================================================
# T2 — Look-Ahead Detector
# ===========================================================================
def test_look_ahead_detector() -> dict:
    print("\n" + "=" * 72)
    print("[T2] Look-Ahead Detector — 扫描 jingni-trader 主代码,标注前视偏差")
    print("=" * 72)

    project_root = Path("/data/user/skills/jingni-trader")
    # 扫描关键的 engine 文件
    targets = [
        "skills/factor-engine/engine.py",
        "skills/strategy-model-engine/engine.py",
        "skills/backtest-engine/engine.py",
    ]
    files = [str(project_root / t) for t in targets]

    detector = LookAheadDetector()
    report = detector.scan_all(source_files=files)

    print(f"  扫描文件: {len(files)}")
    for fp in files:
        n = sum(1 for i in report.issues if i.location.startswith(fp))
        print(f"    {fp}: {n} 个问题")

    n_critical = sum(1 for i in report.issues if i.severity == "critical")
    n_warning  = sum(1 for i in report.issues if i.severity == "warning")
    print(f"  严重问题 (critical): {n_critical}")
    print(f"  警告 (warning):      {n_warning}")

    # 展示前 5 个问题
    print("\n  Top-5 问题:")
    for i in report.issues[:5]:
        print(f"    • [{i.severity}] {i.location}")
        print(f"      模式: {i.pattern}")
        print(f"      说明: {i.description}")
        print(f"      建议: {i.fix}")
        print()

    n_total = n_critical + n_warning
    return {
        "files_scanned": len(files),
        "n_critical": n_critical,
        "n_warning":  n_warning,
        "status": "PASS" if n_total > 0 else "WARN",
    }


# ===========================================================================
# T3 — Vectorized IC Engine
# ===========================================================================
def test_vectorized_ic_engine(data: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("[T3] Vectorized IC Engine — 与 jingni-trader 旧实现对比")
    print("=" * 72)

    # 准备因子: 用 ROC20, MA20, STD20, BETA20 4 个简单因子做 IC 测试
    factors = compute_alpha158_subset(data)
    factor_cols = ["ROC20", "MA20", "STD20", "BETA20"]

    # --- 新引擎 ---
    new_engine = VectorizedICEngine(ic_type="spearman")
    t0 = time.perf_counter()
    new_results = new_engine.compute_ic_series(
        factor_df=factors,
        price_df=data,
        factor_cols=factor_cols,
        forward_periods=[1, 5, 20],
    )
    new_elapsed = time.perf_counter() - t0

    # --- 旧实现 (复制 factor-engine/engine.py 的 _calc_ic 逻辑) ---
    from scipy import stats as _stats
    def legacy_calc_ic(factor_df, forward_col, factor_col):
        ic_list = []
        dates = sorted(factor_df['date'].unique())
        for dt in dates:
            cross = factor_df[factor_df['date'] == dt].dropna(subset=[factor_col, forward_col])
            if len(cross) < 10:
                continue
            ic, _ = _stats.spearmanr(cross[factor_col], cross[forward_col], nan_policy='omit')
            if not np.isnan(ic):
                ic_list.append(ic)
        return np.array(ic_list)

    # 准备 forward_returns
    fr = data[['date', 'code', 'close']].copy()
    for p in [1, 5, 20]:
        fr[f'ret_forward_{p}d'] = fr.groupby('code')['close'].transform(
            lambda x: x.shift(-p) / x - 1
        )
    merged = factors.merge(fr, on=['date', 'code'], how='inner')

    t0 = time.perf_counter()
    legacy_ic = {}
    for factor in factor_cols:
        legacy_ic[factor] = {}
        for p in [1, 5, 20]:
            arr = legacy_calc_ic(merged, f'ret_forward_{p}d', factor)
            legacy_ic[factor][p] = {
                "mean": float(arr.mean()),
                "std":  float(arr.std(ddof=1)),
                "ir":   float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else 0.0,
                "n":    int(len(arr)),
            }
    legacy_elapsed = time.perf_counter() - t0

    # --- 对比 ---
    print(f"  数据规模: {len(data):,} 行 × {data['code'].nunique()} 只股票 × {data['date'].nunique()} 天")
    print(f"  因子数: {len(factor_cols)}, forward 窗口: [1, 5, 20]")
    print()
    print(f"  旧实现 _calc_ic (Python for 循环):")
    print(f"    耗时: {legacy_elapsed:.3f}s")
    print(f"  新实现 VectorizedICEngine (groupby + rank + corrwith):")
    print(f"    耗时: {new_elapsed:.3f}s")
    speedup = legacy_elapsed / new_elapsed if new_elapsed > 0 else float('inf')
    print(f"  ⏱  加速比: {speedup:.1f}x")

    print()
    print("  正确性对比 (新 vs 旧, IC_mean):")
    max_diff = 0.0
    for factor in factor_cols:
        for p in [1, 5, 20]:
            new_ic = new_results[p][factor].ic_mean
            old_ic = legacy_ic[factor][p]["mean"]
            d = abs(new_ic - old_ic)
            max_diff = max(max_diff, d)
            print(f"    {factor:8s} fwd={p}d | new={new_ic:+.4f} old={old_ic:+.4f} diff={d:.2e}")
    print(f"  ✓ 最大差异: {max_diff:.2e}")
    # 由于新实现先 groupby rank 再 corr,旧实现逐天 spearmanr,理论上数值应该一致到 1e-4 量级
    assert max_diff < 1e-3, f"差异过大: {max_diff}"

    return {
        "n_factors": len(factor_cols),
        "n_periods": 3,
        "n_rows": int(len(data)),
        "legacy_elapsed_s": round(legacy_elapsed, 4),
        "new_elapsed_s":    round(new_elapsed, 4),
        "speedup":          round(speedup, 2),
        "max_ic_diff":      float(max_diff),
        "status": "PASS",
    }


# ===========================================================================
# T4 — Walk-Forward Validator
# ===========================================================================
def test_walk_forward_validator(data: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("[T4] Walk-Forward Validator — 滚动训练 + 评估 (Qlib TrainerRM 风格)")
    print("=" * 72)

    # 用 Alpha158 子集做特征, 5 日 forward return 做标签
    factors = compute_alpha158_subset(data)
    feature_cols = [c for c in factors.columns if c not in ("date", "code")]

    fr = data[['date', 'code', 'close']].copy()
    fr['fwd_5d'] = fr.groupby('code')['close'].transform(lambda x: x.shift(-5) / x - 1)

    merged = factors[feature_cols + ["date", "code"]].merge(
        fr[['date', 'code', 'fwd_5d']], on=['date', 'code'], how='inner'
    ).dropna()

    X = merged[feature_cols]
    y = merged['fwd_5d']
    dates = merged['date']

    # 用 LightGBM 作为基模型
    try:
        import lightgbm as lgb
        def model_factory():
            return lgb.LGBMRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.05,
                n_jobs=1, random_state=42, verbosity=-1,
            )
        backend = "lightgbm"
    except ImportError:
        from sklearn.linear_model import Ridge
        def model_factory():
            return Ridge(alpha=1.0, random_state=42)
        backend = "sklearn.Ridge"

    validator = WalkForwardValidator(
        train_window_months=6,
        test_window_months=1,
        step_months=1,
        purge_days=1,
        min_train_samples=200,
    )
    t0 = time.perf_counter()
    result = validator.run(X, y, dates, model_factory)
    elapsed = time.perf_counter() - t0

    print(f"  模型后端: {backend}")
    print(f"  训练窗口: {validator.train_window_months} 月, 测试窗口: {validator.test_window_months} 月")
    print(f"  生成 fold 数: {len(result.folds)}")
    print(f"  总耗时: {elapsed:.3f}s")
    print()
    print("  关键指标:")
    for k, v in result.summary.items():
        if isinstance(v, float):
            print(f"    {k:24s} = {v:+.4f}")
        else:
            print(f"    {k:24s} = {v}")

    print()
    print("  各 fold 详细:")
    print(f"    {'fold':>4s} {'train':>6s} {'test':>6s}  {'IC':>8s} {'RankIC':>8s}")
    for m in result.fold_metrics:
        print(f"    {m['fold_id']:>4d} {m['n_train']:>6d} {m['n_test']:>6d}  "
              f"{m['ic_pearson']:>+8.4f} {m['ic_spearman']:>+8.4f}")

    assert "overall_ic_pearson" in result.summary, "缺少 overall_ic_pearson"
    assert result.summary["n_folds"] >= 2, f"fold 数 {result.summary['n_folds']} 过少"

    return {
        "backend": backend,
        "n_folds": int(result.summary["n_folds"]),
        "overall_ic":  round(result.summary["overall_ic_pearson"], 4),
        "overall_rank_ic": round(result.summary["overall_ic_spearman"], 4),
        "ic_ir":       round(result.summary["ic_ir"], 4),
        "ic_win_rate": round(result.summary["ic_win_rate"], 4),
        "elapsed_s":   round(elapsed, 3),
        "status": "PASS",
    }


# ===========================================================================
# 主入口
# ===========================================================================
def main():
    print("开始生成合成 A 股数据 (50 只股票 × 500 个交易日 = 25,000 行)...")
    t0 = time.perf_counter()
    data = make_synthetic_data(n_stocks=50, n_days=500)
    print(f"  数据生成完成: {len(data):,} 行, 耗时 {time.perf_counter()-t0:.2f}s")

    results = {}
    try:
        results["T1_expression_engine"] = test_expression_engine(data)
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T1_expression_engine"] = {"status": "FAIL", "error": str(e)}

    try:
        results["T2_look_ahead_detector"] = test_look_ahead_detector()
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T2_look_ahead_detector"] = {"status": "FAIL", "error": str(e)}

    try:
        results["T3_vectorized_ic"] = test_vectorized_ic_engine(data)
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T3_vectorized_ic"] = {"status": "FAIL", "error": str(e)}

    try:
        results["T4_walk_forward"] = test_walk_forward_validator(data)
    except Exception as e:
        import traceback; traceback.print_exc()
        results["T4_walk_forward"] = {"status": "FAIL", "error": str(e)}

    # 汇总
    print("\n" + "=" * 72)
    print("验证总览")
    print("=" * 72)
    n_pass = sum(1 for r in results.values() if r.get("status") == "PASS")
    n_total = len(results)
    for name, r in results.items():
        status = r.get("status", "UNKNOWN")
        print(f"  [{status:4s}] {name}")

    summary = {
        "n_pass": n_pass,
        "n_total": n_total,
        "results": results,
    }
    out_path = Path(__file__).resolve().parents[1] / "reports" / "test_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {out_path}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
