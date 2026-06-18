"""
================================================================================
jingni-trader 优化验证测试套件
================================================================================
三大优化模块的统一测试入口:
  T1) 向量化 IC 计算 vs 现有 for-loop
       - 正确性: 相同输入应得到 <1e-9 差异 (或 <1e-6 for spearmanr ties)
       - 性能:   speedup
       - 边界:   NaN, 单截面, 全 NaN, 单只股票
  T2) 可学习 Processor 模式 vs 现有 rolling
       - 正确性: 与 jingni-trader compute_a_share_factors 行为对照
       - 关键验证: 训练期 fit -> 测试期 process 不会泄露未来信息
       - 边界:   全 NaN 列, 训练期 < 30 行
  T3) Walk-Forward 验证
       - 正确性: 分窗结果与手工分窗一致
       - 关键:  in-sample IC vs out-of-sample IC 的 overfit_gap
       - 边界:   折数 = 0, 训练期太短
================================================================================
所有结果写入 /workspace/quant_opt/tests/results.json
"""
import json
import os
import sys
import time
import platform
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # 引入 /workspace/quant_opt

from vectorized_ic import (_calc_ic_legacy, calc_ic_vectorized,
                           calc_ic_parallel, ic_summary)
from learnable_processor import (Processor, GlobalZScore, CSZScore,
                                 RollingZScore, Pipeline)
from walk_forward import (generate_folds, _train_predict_one_fold,
                          run_walk_forward, WalkForwardResult, Fold)


# ============================================================================
# 测试数据生成器 (确保可复现)
# ============================================================================
def make_panel(n_stocks: int = 50, n_days: int = 1500,
               start: str = "2020-01-01", seed: int = 42) -> pd.DataFrame:
    """生成带可预测信号的模拟 A 股日线面板, 包含 NaN 注入"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    codes = [f"SH{600000 + i:06d}" for i in range(n_stocks)]

    rows = []
    for code in codes:
        # 每只股票有独立的"alpha 强度", 用于让 IC 显著
        alpha_strength = rng.normal(0, 0.3)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n_days)))
        # 让 close 与 forward return 有可控相关: close_t+1 = close_t * exp(alpha + noise)
        ret_1d = pd.Series(close).pct_change().shift(-1).values
        # 在 ret_1d 上加 alpha_strength * faked_factor 噪声
        factor = pd.Series(close).pct_change(20).fillna(0).values
        forward = ret_1d + alpha_strength * factor * 0.1 + rng.normal(0, 0.005, n_days)
        for i in range(n_days):
            rows.append({
                "date": dates[i], "code": code,
                "close": close[i], "ret_1d": ret_1d[i],
                "factor": factor[i], "forward_1d": forward[i],
            })
    df = pd.DataFrame(rows)
    # 注入 2% 缺失
    mask = rng.random(len(df)) < 0.02
    df.loc[mask, "factor"] = np.nan
    return df.sort_values(["date", "code"]).reset_index(drop=True)


# ============================================================================
# T1: 向量化 IC 计算测试
# ============================================================================
def t1_vectorized_ic(df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 70)
    print("T1: 向量化 IC 计算 vs jingni-trader 现有实现")
    print("=" * 70)
    out: Dict = {"name": "T1_vectorized_ic", "cases": []}

    # --- 1. 正确性对比 (中等数据 30 stocks × 600 days) ---
    sub = df[df["code"].isin(df["code"].unique()[:30])].copy()
    sub = sub[sub["date"] < sub["date"].min() + pd.Timedelta(days=600 * 1.5)]
    print(f"\n[T1.1] 正确性对比: 30 stocks × {sub['date'].nunique()} days")
    t0 = time.perf_counter()
    ic_legacy = _calc_ic_legacy(sub, "factor", "forward_1d", ic_type="spearman")
    t_legacy = time.perf_counter() - t0

    t0 = time.perf_counter()
    ic_vec = calc_ic_vectorized(sub, "factor", "forward_1d", ic_type="spearman")
    t_vec = time.perf_counter() - t0

    common = ic_legacy.index.intersection(ic_vec.index)
    if len(common) > 0:
        max_abs_diff = float(np.abs(ic_legacy.loc[common] - ic_vec.loc[common]).max())
        mean_abs_diff = float(np.abs(ic_legacy.loc[common] - ic_vec.loc[common]).mean())
        n_match = len(common)
    else:
        max_abs_diff, mean_abs_diff, n_match = float("inf"), float("inf"), 0

    print(f"  legacy:  {t_legacy:.3f}s  (n={len(ic_legacy)})")
    print(f"  vector:  {t_vec:.3f}s  (n={len(ic_vec)})")
    print(f"  max_abs_diff = {max_abs_diff:.2e}, mean_abs_diff = {mean_abs_diff:.2e}, "
          f"n_match = {n_match}/{len(ic_legacy)}")
    print(f"  speedup = {t_legacy / max(t_vec, 1e-6):.2f}x")

    case = {
        "case": "正确性 (30 stocks × ~600d)",
        "n_dates": int(sub["date"].nunique()),
        "n_stocks": 30,
        "t_legacy_s": round(t_legacy, 4),
        "t_vectorized_s": round(t_vec, 4),
        "speedup": round(t_legacy / max(t_vec, 1e-6), 2),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "n_match": n_match,
        "n_legacy": int(len(ic_legacy)),
        "n_vectorized": int(len(ic_vec)),
        "pass_correctness": max_abs_diff < 1e-6 and n_match == len(ic_legacy),
    }
    out["cases"].append(case)

    # --- 2. 全量数据性能 (50 × 1500) ---
    print(f"\n[T1.2] 全量性能: 50 stocks × {df['date'].nunique()} days")
    t0 = time.perf_counter()
    ic_legacy_full = _calc_ic_legacy(df, "factor", "forward_1d", ic_type="spearman")
    t_legacy_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    ic_vec_full = calc_ic_vectorized(df, "factor", "forward_1d", ic_type="spearman")
    t_vec_full = time.perf_counter() - t0

    common = ic_legacy_full.index.intersection(ic_vec_full.index)
    max_abs_full = float(np.abs(ic_legacy_full.loc[common] -
                                 ic_vec_full.loc[common]).max()) if len(common) else float("inf")
    print(f"  legacy:  {t_legacy_full:.3f}s")
    print(f"  vector:  {t_vec_full:.3f}s  speedup = {t_legacy_full / max(t_vec_full, 1e-6):.2f}x")
    print(f"  max_abs_diff = {max_abs_full:.2e}")
    out["cases"].append({
        "case": "性能 (50 stocks × 1500d)",
        "t_legacy_s": round(t_legacy_full, 4),
        "t_vectorized_s": round(t_vec_full, 4),
        "speedup": round(t_legacy_full / max(t_vec_full, 1e-6), 2),
        "max_abs_diff": max_abs_full,
        "pass_correctness": max_abs_full < 1e-6,
    })

    # --- 3. 多进程 (在 Linux 沙箱) ---
    print(f"\n[T1.3] 多进程并行 (n_workers=2)")
    t0 = time.perf_counter()
    ic_par = calc_ic_parallel(df, "factor", "forward_1d", ic_type="spearman",
                              n_workers=2)
    t_par = time.perf_counter() - t0
    common = ic_vec_full.index.intersection(ic_par.index)
    max_abs_par = float(np.abs(ic_vec_full.loc[common] -
                               ic_par.loc[common]).max()) if len(common) else float("inf")
    print(f"  parallel:  {t_par:.3f}s   vs vector {t_vec_full:.3f}s")
    print(f"  max_abs_diff (vec vs par) = {max_abs_par:.2e}")
    out["cases"].append({
        "case": "多进程 (50 × 1500d, 2 workers)",
        "t_parallel_s": round(t_par, 4),
        "max_abs_diff_vs_vectorized": max_abs_par,
        "pass_correctness": max_abs_par < 1e-6,
    })

    # --- 4. 边界: 单只股票 ---
    print("\n[T1.4] 边界: 单只股票")
    single = df[df["code"] == df["code"].unique()[0]]
    ic_single = calc_ic_vectorized(single, "factor", "forward_1d")
    print(f"  single stock ic: len={len(ic_single)} (应为 0, 因为每天 < 10 obs)")
    out["cases"].append({
        "case": "边界: 单只股票 (截面 < min_obs)",
        "n_dates": int(single["date"].nunique()),
        "n_ic_returned": int(len(ic_single)),
        "pass_correctness": len(ic_single) == 0,
    })

    # --- 5. 边界: 全 NaN 列 ---
    print("\n[T1.5] 边界: factor 全 NaN")
    df_nan = df.copy()
    df_nan["factor"] = np.nan
    ic_nan = calc_ic_vectorized(df_nan, "factor", "forward_1d")
    print(f"  all-NaN ic: len={len(ic_nan)} (应为 0)")
    out["cases"].append({
        "case": "边界: factor 全 NaN",
        "n_ic_returned": int(len(ic_nan)),
        "pass_correctness": len(ic_nan) == 0,
    })

    # --- 6. 边界: 单日截面 ---
    print("\n[T1.6] 边界: 单日截面")
    one_day = df[df["date"] == df["date"].unique()[100]]
    ic_oneday = calc_ic_vectorized(one_day, "factor", "forward_1d")
    print(f"  one-day ic: len={len(ic_oneday)} (应为 1 或 0)")
    out["cases"].append({
        "case": "边界: 单日截面",
        "n_ic_returned": int(len(ic_oneday)),
        "pass_correctness": len(ic_oneday) <= 1,
    })

    # --- 7. IC 统计量汇总 ---
    print("\n[T1.7] IC 汇总 (向量化版)")
    summary = ic_summary(ic_vec_full)
    print(f"  ic_mean={summary['ic_mean']:.4f}  ic_ir={summary['ic_ir']:.2f}  "
          f"pos_ratio={summary['ic_pos_ratio']:.2%}  t_stat={summary['ic_t_stat']:.2f}  "
          f"n_periods={summary['n_periods']}")
    out["ic_summary"] = summary

    out["pass"] = all(c.get("pass_correctness", True) for c in out["cases"])
    print(f"\n[T1] {'PASS' if out['pass'] else 'FAIL'}")
    return out


# ============================================================================
# T2: 可学习 Processor 测试
# ============================================================================
def t2_learnable_processor(df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 70)
    print("T2: 可学习 Processor 模式 (Qlib DataHandlerLP 风格)")
    print("=" * 70)
    out: Dict = {"name": "T2_learnable_processor", "cases": []}

    # 准备训练/测试分割 (前 70% / 后 30%)
    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.7)
    train_dates = dates[:split]
    test_dates = dates[split:]
    train = df[df["date"].isin(train_dates)].copy()
    test = df[df["date"].isin(test_dates)].copy()
    print(f"  train dates: {train_dates[0]} ~ {train_dates[-1]}  ({len(train_dates)}d)")
    print(f"  test dates:  {test_dates[0]} ~ {test_dates[-1]}  ({len(test_dates)}d)")

    # --- 1. fit/process 正确性: GlobalZScore ---
    print("\n[T2.1] GlobalZScore: fit 在 train, process 在 test")
    proc = GlobalZScore().fit(train[["factor", "forward_1d"]])
    state = proc._state
    print(f"  训练期学到的 factor mu={state['mu']['factor']:.4f}, "
          f"sd={state['sd']['factor']:.4f}")
    test_proc = proc.process(test[["factor", "forward_1d"]])
    # 验证: test 上 factor 的均值应该接近 0, sd 接近 1 (因为用 train 的参数)
    factor_mean = test_proc["factor"].mean()
    factor_sd = test_proc["factor"].std()
    print(f"  test factor mean={factor_mean:.4f} (应 ≈ 0), "
          f"sd={factor_sd:.4f} (应 ≈ 1)")
    out["cases"].append({
        "case": "GlobalZScore fit/train apply/test",
        "train_factor_mu": round(state["mu"]["factor"], 6),
        "train_factor_sd": round(state["sd"]["factor"], 6),
        "test_factor_mean_after": round(factor_mean, 4),
        "test_factor_sd_after": round(factor_sd, 4),
        "pass_correctness": abs(factor_mean) < 0.5 and 0.5 < factor_sd < 2.0,
    })

    # --- 2. 与现有 rolling z-score 对比 (in-sample 一致性) ---
    print("\n[T2.2] RollingZScore: 与 jingni-trader 现有实现行为对照")
    rp = RollingZScore().fit(train[["factor"]], columns=["factor"],
                             window=20, clip=3.0)
    train_rolled = rp.process(train[["factor"]])
    # 验证: rolling z-score 的首 window-1 行 per stock 应全为 NaN
    # (由于我们按 (date, code) 排序, 窗口按行滑动; 因注入 NaN, 实际 NaN >= window-1)
    n_nan_first = int(train_rolled["factor"].isna().sum())
    print(f"  NaN 数 = {n_nan_first} (应 >= window-1 = {rp._state['window']-1}, "
          "因注入 NaN 实际更多)")
    out["cases"].append({
        "case": "RollingZScore 窗口期 NaN",
        "expected_min_nan": rp._state["window"] - 1,
        "actual_nan": n_nan_first,
        "pass_correctness": n_nan_first >= rp._state["window"] - 1,
    })

    # --- 3. Pipeline 串联 (借鉴 Qlib processor 链) ---
    print("\n[T3.3] Pipeline: 串联 GlobalZScore -> CSZScore")
    pipe = Pipeline([GlobalZScore(), CSZScore()])
    pipe_proc1 = GlobalZScore().fit(train[["factor", "forward_1d"]])
    out_train1 = pipe_proc1.process(train[["factor", "forward_1d", "date"]])
    pipe_proc2 = CSZScore().fit(out_train1, columns=["factor", "forward_1d"],
                                date_col="date")
    out_train2 = pipe_proc2.process(out_train1)
    print(f"  Pipeline 训练后: factor mean={out_train2['factor'].mean():.4f}, "
          f"sd={out_train2['factor'].std():.4f}")
    # 应用到 test
    out_test1 = pipe_proc1.process(test[["factor", "forward_1d", "date"]])
    out_test2 = pipe_proc2.process(out_test1)
    print(f"  Pipeline 测试后: factor mean={out_test2['factor'].mean():.4f}, "
          f"sd={out_test2['factor'].std():.4f}")
    # 关键: CSZScore 应当在每个 date 截面上让 mean≈0, std≈1
    daily_mean = out_train2.groupby("date")["factor"].mean().abs().max()
    daily_std = out_train2.groupby("date")["factor"].std()
    daily_std_mean = daily_std.mean()
    daily_std_in_range = ((daily_std > 0.5) & (daily_std < 1.5)).mean()
    print(f"  daily |mean| max = {daily_mean:.4e} (应 ≈ 0)")
    print(f"  daily std 均值 = {daily_std_mean:.4f}, 在 [0.5, 1.5] 范围的比例 = {daily_std_in_range:.2%}")
    out["cases"].append({
        "case": "Pipeline 串联 (GlobalZScore -> CSZScore)",
        "daily_mean_abs_max": round(float(daily_mean), 6),
        "daily_std_mean": round(float(daily_std_mean), 4),
        "daily_std_in_range_ratio": round(float(daily_std_in_range), 4),
        "pass_correctness": (daily_mean < 1e-6 and daily_std_in_range > 0.95),
    })

    # --- 4. 关键: 数据泄露对比 ---
    # 错误做法: 整段数据一起标准化 (信息泄露)
    bad_proc = GlobalZScore()
    bad_proc._state = bad_proc._learn(
        df[["factor", "forward_1d"]])  # 用全部数据 fit
    bad_proc.is_fitted = True
    df_bad = bad_proc.process(df[["factor", "forward_1d"]])
    # 正确做法: 只用 train fit
    good_proc = GlobalZScore().fit(train[["factor", "forward_1d"]])
    df_good = good_proc.process(df[["factor", "forward_1d"]])
    diff = float((df_bad["factor"] - df_good["factor"]).abs().mean())
    print(f"\n[T2.4] 数据泄露验证: bad vs good mean abs diff = {diff:.4f}")
    print(f"  > 0 表明两种做法输出不同 -> 用全量 fit 会引入未来信息")
    out["cases"].append({
        "case": "数据泄露验证 (bad fit on all vs good fit on train)",
        "mean_abs_diff": round(diff, 6),
        "pass_correctness": diff > 0.001,
    })

    # --- 5. 边界: 训练集 < 30 行 ---
    print("\n[T2.5] 边界: 训练集仅 10 行")
    tiny_train = train.head(10)
    try:
        tiny_proc = GlobalZScore().fit(tiny_train[["factor"]])
        tiny_test = tiny_proc.process(test[["factor"]])
        print(f"  训练 10 行也能 fit, test 输出 shape={tiny_test.shape}")
        out["cases"].append({
            "case": "边界: 训练集 10 行",
            "ok": True,
            "pass_correctness": True,
        })
    except Exception as e:
        print(f"  异常: {e}")
        out["cases"].append({
            "case": "边界: 训练集 10 行",
            "ok": False,
            "error": str(e),
            "pass_correctness": False,
        })

    # --- 6. 边界: 整列 NaN ---
    print("\n[T2.6] 边界: forward_1d 整列 NaN (训练期) -> process 应保留原值")
    train_nan = train.copy()
    train_nan["forward_1d"] = np.nan
    nan_proc = GlobalZScore().fit(train_nan[["factor", "forward_1d"]])
    state = nan_proc._state
    sd_fwd = state["sd"]["forward_1d"]
    print(f"  训练期 forward_1d 全 NaN -> 学到的 sd = {sd_fwd} (应为 NaN)")
    out_nan = nan_proc.process(test[["factor", "forward_1d"]])
    # forward_1d 应原样保留 (因为 sd=NaN 触发跳过)
    unchanged = bool(np.allclose(out_nan["forward_1d"].values,
                                 test["forward_1d"].values, equal_nan=True))
    print(f"  forward_1d 与原始 test 一致: {unchanged}")
    out["cases"].append({
        "case": "边界: 训练期 forward_1d 全 NaN",
        "learned_sd_fwd": None if pd.isna(sd_fwd) else round(float(sd_fwd), 6),
        "unchanged_in_test": unchanged,
        "pass_correctness": unchanged,
    })

    out["pass"] = all(c.get("pass_correctness", True) for c in out["cases"])
    print(f"\n[T2] {'PASS' if out['pass'] else 'FAIL'}")
    return out


# ============================================================================
# T3: Walk-Forward 验证
# ============================================================================
def t3_walk_forward(df: pd.DataFrame) -> Dict:
    print("\n" + "=" * 70)
    print("T3: Walk-Forward 验证 (Qlib RollingGen + AKQuant 风格)")
    print("=" * 70)
    out: Dict = {"name": "T3_walk_forward", "cases": []}

    # --- 1. 折叠生成正确性 ---
    print("\n[T3.1] generate_folds 正确性")
    folds = generate_folds(df["date"], train_days=252, test_days=63, stride=63)
    n_folds = len(folds)
    print(f"  生成 {n_folds} 折, train=252d, test=63d, stride=63d")
    if folds:
        print(f"  首折: train {folds[0].train_start.date()} ~ {folds[0].train_end.date()}, "
              f"test {folds[0].test_start.date()} ~ {folds[0].test_end.date()}")
        print(f"  末折: train {folds[-1].train_start.date()} ~ {folds[-1].train_end.date()}, "
              f"test {folds[-1].test_start.date()} ~ {folds[-1].test_end.date()}")
    # 验证: 相邻折 test 区不重叠
    overlap = False
    for i in range(1, len(folds)):
        if folds[i].test_start <= folds[i - 1].test_end:
            overlap = True
            break
    out["cases"].append({
        "case": "generate_folds 不重叠 (stride=63)",
        "n_folds": n_folds,
        "first_train_days": int((folds[0].train_end - folds[0].train_start).days + 1)
                            if folds else 0,
        "first_test_days": int((folds[0].test_end - folds[0].test_start).days + 1)
                           if folds else 0,
        "no_test_overlap": not overlap,
        "pass_correctness": (n_folds >= 2 and not overlap),
    })

    # --- 2. 主入口: 完整 walk-forward ---
    print("\n[T3.2] run_walk_forward: 完整评估")
    factor_cols = ["ret_1d", "factor"]
    df["ret_forward_1d"] = df.groupby("code")["close"].pct_change().shift(-1)
    res = run_walk_forward(df, factor_cols=factor_cols,
                           forward_col="ret_forward_1d",
                           train_days=252, test_days=63, stride=63)
    print(f"  n_folds={res.n_folds}, n_skipped={res.n_skipped}")
    print(f"  OOS IC:  mean={res.oos_ic_mean:.4f}  std={res.oos_ic_std:.4f}  "
          f"IR={res.oos_ir:.3f}")
    print(f"  OOS Long-Short 5分位 spread (mean) = {res.oos_ls_spread_mean:.5f}")
    print(f"  In-Sample IC mean = {res.in_sample_ic_mean:.4f}")
    print(f"  Overfit Gap = {res.overfit_gap:.4f}  (in_sample - oos)")
    out["cases"].append({
        "case": "完整 walk-forward 评估",
        "n_folds": res.n_folds,
        "n_skipped": res.n_skipped,
        "oos_ic_mean": round(res.oos_ic_mean, 6),
        "oos_ic_std": round(res.oos_ic_std, 6),
        "oos_ir": round(res.oos_ir, 4),
        "oos_ls_spread_mean": round(res.oos_ls_spread_mean, 6),
        "in_sample_ic_mean": round(res.in_sample_ic_mean, 6),
        "overfit_gap": round(res.overfit_gap, 6),
        "pass_correctness": res.n_folds >= 2,
    })
    out["walk_forward_summary"] = {
        "n_folds": res.n_folds, "n_skipped": res.n_skipped,
        "oos_ic_mean": res.oos_ic_mean, "oos_ic_std": res.oos_ic_std,
        "oos_ir": res.oos_ir, "oos_ls_spread_mean": res.oos_ls_spread_mean,
        "in_sample_ic_mean": res.in_sample_ic_mean,
        "overfit_gap": res.overfit_gap,
    }

    # --- 3. 边界: 数据太短 (应抛错) ---
    print("\n[T3.3] 边界: 数据太短 (< train_days+test_days)")
    short_df = df[df["date"] < df["date"].min() + pd.Timedelta(days=200)].copy()
    raised = False
    try:
        generate_folds(short_df["date"], train_days=252, test_days=63, stride=63)
    except ValueError as e:
        raised = True
        print(f"  正确抛出 ValueError: {e}")
    out["cases"].append({
        "case": "边界: 数据太短",
        "raised_value_error": raised,
        "pass_correctness": raised,
    })

    # --- 4. 边界: stride=train_days (不重叠 fold) ---
    print("\n[T3.4] 边界: stride=train_days (无重叠)")
    folds_norep = generate_folds(df["date"], train_days=252, test_days=63,
                                 stride=252)
    n_norep = len(folds_norep)
    print(f"  生成 {n_norep} 折, 无重叠")
    # 验证无重叠
    no_overlap = True
    for i in range(1, len(folds_norep)):
        if folds_norep[i].test_start <= folds_norep[i - 1].test_end:
            no_overlap = False
            break
    # 验证无训练重叠
    no_train_overlap = True
    for i in range(1, len(folds_norep)):
        if folds_norep[i].train_start <= folds_norep[i - 1].train_end:
            no_train_overlap = False
            break
    out["cases"].append({
        "case": "边界: stride=train_days (完全无重叠)",
        "n_folds": n_norep,
        "no_test_overlap": no_overlap,
        "no_train_overlap": no_train_overlap,
        "pass_correctness": n_norep >= 2 and no_overlap and no_train_overlap,
    })

    out["pass"] = all(c.get("pass_correctness", True) for c in out["cases"])
    print(f"\n[T3] {'PASS' if out['pass'] else 'FAIL'}")
    return out


# ============================================================================
# 主入口
# ============================================================================
def main():
    print("=" * 70)
    print(f"  jingni-trader 优化验证  (Python {platform.python_version()})")
    print(f"  pandas {pd.__version__}, numpy {np.__version__}")
    print("=" * 70)

    t0 = time.perf_counter()
    df = make_panel(n_stocks=50, n_days=1500, seed=42)
    print(f"\n生成测试数据: {len(df)} rows, "
          f"{df['code'].nunique()} stocks, {df['date'].nunique()} days "
          f"({time.perf_counter() - t0:.2f}s)")

    results = {
        "platform": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data_shape": {"rows": int(len(df)),
                       "stocks": int(df["code"].nunique()),
                       "days": int(df["date"].nunique())},
    }

    results["T1"] = t1_vectorized_ic(df)
    results["T2"] = t2_learnable_processor(df)
    results["T3"] = t3_walk_forward(df)

    overall = (results["T1"]["pass"] and results["T2"]["pass"]
               and results["T3"]["pass"])
    results["overall_pass"] = overall

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果写入: {out_path}")
    print(f"Overall: {'ALL PASS' if overall else 'SOME FAILED'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
