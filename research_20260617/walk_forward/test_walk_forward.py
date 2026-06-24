"""
Walk-Forward 滚动验证 - 测试用例
================================

测试场景：
1. 划分器在长时段数据上生成合理数量的段
2. anchored=True 时 train 起点固定；stepped 模式下起点滑动
3. 验证函数：每段都执行 train/valid/test 三步并产出 metrics
4. 边界条件：数据不足时应优雅处理
5. purged 间隔正确生效
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research_20260617.walk_forward.walk_forward import (
    WalkForwardConfig,
    WalkForwardSplitter,
    aggregate_metrics,
    run_walk_forward,
)


def make_long_data(n_days: int = 1500) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "code": ["000001.SZ"] * n_days,
            "close": 10 + np.cumsum(np.random.RandomState(42).normal(0, 0.02, n_days)),
        }
    )


def test_basic_splits():
    """基础划分测试"""
    print("\n=== Test 1: Basic walk-forward splits ===")
    dates = pd.bdate_range("2020-01-01", periods=1500)
    cfg = WalkForwardConfig(
        train_window_months=12,
        valid_window_months=6,
        test_window_months=6,
        purge_gap_days=5,
        step_months=6,
    )
    splitter = WalkForwardSplitter(cfg)
    segments = splitter.split(dates)
    print(f"生成 {len(segments)} 段")
    for s in segments[:3]:
        print(f"  segment {s.segment_id}: train {s.train_start}->{s.train_end} | "
              f"valid {s.valid_start}->{s.valid_end} | test {s.test_start}->{s.test_end}")

    assertions = [
        (f"生成 >=3 段（实际 {len(segments)}）", len(segments) >= 3),
        ("每段 train 起点 < valid 起点 < test 起点",
            all(pd.to_datetime(s.train_start) < pd.to_datetime(s.valid_start) < pd.to_datetime(s.test_start)
                for s in segments)),
        ("purge gap 生效：valid_start 与 train_end 差 >= 5 天",
            all((pd.to_datetime(s.valid_start) - pd.to_datetime(s.train_end)).days >= 5
                for s in segments)),
    ]
    return assertions, segments


def test_anchored_vs_rolling():
    """anchored vs 滚动模式对比"""
    print("\n=== Test 2: anchored vs rolling ===")
    dates = pd.bdate_range("2020-01-01", periods=1500)

    cfg_anchor = WalkForwardConfig(
        train_window_months=12, valid_window_months=6, test_window_months=6,
        purge_gap_days=5, step_months=6, anchored=True,
    )
    seg_anchor = WalkForwardSplitter(cfg_anchor).split(dates)

    cfg_roll = WalkForwardConfig(
        train_window_months=12, valid_window_months=6, test_window_months=6,
        purge_gap_days=5, step_months=6, anchored=False,
    )
    seg_roll = WalkForwardSplitter(cfg_roll).split(dates)

    if len(seg_anchor) >= 2 and len(seg_roll) >= 2:
        anchor_same_start = seg_anchor[0].train_start == seg_anchor[1].train_start
        roll_diff_start = seg_roll[0].train_start != seg_roll[1].train_start
        print(f"  anchored: 段0 train_start = {seg_anchor[0].train_start}, 段1 = {seg_anchor[1].train_start}")
        print(f"  rolling : 段0 train_start = {seg_roll[0].train_start}, 段1 = {seg_roll[1].train_start}")
        assertions = [
            ("anchored: 所有段 train_start 相同", anchor_same_start),
            ("rolling: 相邻段 train_start 不同", roll_diff_start),
        ]
    else:
        assertions = [("数据足够生成多段", False)]
    return assertions, {"anchor": seg_anchor, "roll": seg_roll}


def test_aggregate_metrics():
    """指标聚合测试"""
    print("\n=== Test 3: aggregate_metrics ===")
    per_seg = [
        {"sharpe_ratio": 1.0, "total_return": 0.10, "max_drawdown": -0.05},
        {"sharpe_ratio": 1.5, "total_return": 0.15, "max_drawdown": -0.07},
        {"sharpe_ratio": 0.5, "total_return": 0.05, "max_drawdown": -0.10},
        {"sharpe_ratio": -0.5, "total_return": -0.05, "max_drawdown": -0.15},
    ]
    agg = aggregate_metrics(per_seg)
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    assertions = [
        ("avg_sharpe = 0.625", abs(agg["avg_sharpe"] - 0.625) < 1e-6),
        ("consistency = 0.75 (3 段正收益)", abs(agg["consistency"] - 0.75) < 1e-6),
        ("worst_drawdown = -0.15", abs(agg["worst_drawdown"] - (-0.15)) < 1e-6),
    ]
    return assertions, agg


def test_end_to_end_run():
    """端到端：使用一个简单的"哑策略"走完 walk-forward 流程"""
    print("\n=== Test 4: end-to-end walk-forward run ===")

    data = make_long_data(n_days=1500)

    def train_fn(train_df: pd.DataFrame) -> Dict:
        # 哑"训练"：返回 stats
        return {"mean_close": float(train_df["close"].mean())}

    def valid_fn(valid_df: pd.DataFrame, train_info: Dict) -> Dict:
        return {**train_info, "valid_mean": float(valid_df["close"].mean())}

    def test_fn(test_df: pd.DataFrame, params: Dict, train_info: Dict) -> Dict:
        # 构造一个"策略净值" = (1 + 测试段收益累计) 用于产出 metrics
        test_df = test_df.copy()
        test_df["ret"] = test_df["close"].pct_change().fillna(0.0)
        equity = (1.0 + test_df["ret"]).cumprod()
        import importlib
        _bb_mod = importlib.import_module("skills.backtest-engine.scripts.base.base_backtest")
        metrics = _bb_mod.BaseBacktestMetrics.calc_all_metrics(equity, pd.DataFrame())
        return {
            "metrics": metrics,
            "equity_curve": pd.DataFrame({"date": test_df["date"], "equity": equity.values}),
        }

    cfg = WalkForwardConfig(
        train_window_months=12, valid_window_months=6, test_window_months=6,
        purge_gap_days=5, step_months=6, anchored=False,
    )
    splitter = WalkForwardSplitter(cfg)
    result = run_walk_forward(data, splitter, train_fn, valid_fn, test_fn)
    print(f"  segments executed: {len(result.segments)}")
    print(f"  aggregated metrics: {result.aggregated_metrics}")
    assertions = [
        (f"段数 >= 2（实际 {len(result.segments)}）", len(result.segments) >= 2),
        ("aggregated_metrics 包含 avg_sharpe", "avg_sharpe" in result.aggregated_metrics),
        ("out_of_sample_equity 非空", result.out_of_sample_equity is not None and not result.out_of_sample_equity.empty),
    ]
    return assertions, result


def test_short_data():
    """数据不足时的边界处理"""
    print("\n=== Test 5: short data edge case ===")
    dates = pd.bdate_range("2024-01-01", periods=120)  # 6 个月
    cfg = WalkForwardConfig(
        train_window_months=36, valid_window_months=12, test_window_months=12,
        purge_gap_days=5, step_months=12, min_train_months=12,
    )
    splitter = WalkForwardSplitter(cfg)
    segments = splitter.split(dates)
    print(f"  短数据生成 {len(segments)} 段（应为 0）")
    assertions = [
        (f"短数据生成 0 段（实际 {len(segments)}）", len(segments) == 0),
    ]
    return assertions, segments


def main() -> int:
    all_results = {}
    all_assertions = []

    a1, _ = test_basic_splits()
    all_results["basic_splits"] = [{"name": n, "passed": bool(p)} for n, p in a1]
    all_assertions.extend(a1)

    a2, _ = test_anchored_vs_rolling()
    all_results["anchored_vs_rolling"] = [{"name": n, "passed": bool(p)} for n, p in a2]
    all_assertions.extend(a2)

    a3, _ = test_aggregate_metrics()
    all_results["aggregate_metrics"] = [{"name": n, "passed": bool(p)} for n, p in a3]
    all_assertions.extend(a3)

    a4, _ = test_end_to_end_run()
    all_results["end_to_end"] = [{"name": n, "passed": bool(p)} for n, p in a4]
    all_assertions.extend(a4)

    a5, _ = test_short_data()
    all_results["short_data"] = [{"name": n, "passed": bool(p)} for n, p in a5]
    all_assertions.extend(a5)

    total = len(all_assertions)
    passed = sum(1 for _, p in all_assertions if p)
    print(f"\n{'='*50}")
    print(f"PASSED: {passed}/{total}")
    for n, p in all_assertions:
        print(f"  [{'OK' if p else 'FAIL'}] {n}")

    out = {
        "summary": {"total": int(total), "passed": int(passed), "failed": int(total - passed), "all_passed": bool(passed == total)},
        "results": {k: v for k, v in all_results.items()},
    }
    out_path = Path(__file__).parent / "test_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"详细结果已写入 {out_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())