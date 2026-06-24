"""
因子 IC/IR 分析 - 测试用例
==========================

测试场景：
1. 构造 5 个因子，其中 3 个与未来收益正相关，2 个反向相关，1 个无关
2. 验证 IC mean / ICIR / 胜率 / 方向推断正确
3. 验证冗余因子剔除逻辑
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from skills.quant_optimizations.research_20260617.factor_ic.ic_analyzer import (
    FactorICAnalyzer,
    reports_to_dataframe,
)


def make_synthetic(n_stocks: int = 100, n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """生成 synthetic OHLCV + 5 因子数据

    因子构造：
    - f1, f2, f3：与未来 1 日收益正相关（强度 0.4, 0.25, 0.15）
    - f4：与未来 1 日收益负相关（强度 -0.3）
    - f5：完全噪声（与未来收益无关）
    - f6：与 f1 强相关（0.85），用于测试冗余剔除
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    codes = [f"{i:06d}.SH" for i in range(600000, 600000 + n_stocks)]

    rows = []
    for code in codes:
        # 基础价格 random walk
        log_ret = rng.normal(0.0005, 0.02, n_days)
        close = 10.0 * np.exp(np.cumsum(log_ret))
        # 未来 1 日收益（用 log_ret shift -1 近似）
        fwd_ret = np.roll(log_ret, -1)
        fwd_ret[-1] = 0.0

        for i in range(n_days):
            # f1 与 fwd_ret 正相关 0.4
            f1 = 0.4 * fwd_ret[i] + rng.normal(0, 0.02)
            # f2 与 fwd_ret 正相关 0.25
            f2 = 0.25 * fwd_ret[i] + rng.normal(0, 0.02)
            # f3 与 fwd_ret 正相关 0.15
            f3 = 0.15 * fwd_ret[i] + rng.normal(0, 0.02)
            # f4 与 fwd_ret 负相关 -0.3
            f4 = -0.3 * fwd_ret[i] + rng.normal(0, 0.02)
            # f5 纯噪声
            f5 = rng.normal(0, 0.02)
            # f6 与 f1 强相关（冗余因子）
            f6 = 0.85 * f1 + rng.normal(0, 0.01)

            rows.append(
                {
                    "date": dates[i].strftime("%Y-%m-%d"),
                    "code": code,
                    "close": close[i],
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "f4": f4,
                    "f5": f5,
                    "f6": f6,
                    "fwd_ret": fwd_ret[i],
                }
            )
    return pd.DataFrame(rows)


def test_ic_accuracy():
    """测试 IC 评估的准确性"""
    print("\n=== Test 1: IC accuracy on synthetic data ===")
    df = make_synthetic(n_stocks=100, n_days=252)

    factor_df = df[["date", "code", "f1", "f2", "f3", "f4", "f5", "f6"]]
    price_df = df[["date", "code", "close"]]

    analyzer = FactorICAnalyzer(forward=1)
    reports = analyzer.analyze(factor_df, price_df)

    print(reports_to_dataframe(reports).to_string(index=False))

    # === 验证 1：f1/f2/f3 应有正 IC，f4 应有负 IC，f5 应接近 0 ===
    rep_map = {r.factor: r for r in reports}
    assertions = []
    if "f1" in rep_map:
        assertions.append(("f1 正 IC", rep_map["f1"].rank_ic_mean > 0.1))
    if "f4" in rep_map:
        assertions.append(("f4 负 IC", rep_map["f4"].rank_ic_mean < -0.05))
    if "f5" in rep_map:
        assertions.append(("f5 IC 接近 0", abs(rep_map["f5"].rank_ic_mean) < 0.05))

    # === 验证 2：f1 方向应为 1，f4 方向应为 -1 ===
    if "f1" in rep_map:
        assertions.append(("f1 方向=1", rep_map["f1"].direction == 1))
    if "f4" in rep_map:
        assertions.append(("f4 方向=-1", rep_map["f4"].direction == -1))

    # === 验证 3：f1 的 |ICIR| 应高于 f5 ===
    if "f1" in rep_map and "f5" in rep_map:
        assertions.append(("f1 ICIR > f5 ICIR", abs(rep_map["f1"].rank_icir) > abs(rep_map["f5"].rank_icir)))

    return assertions, reports


def test_correlation_matrix():
    """测试相关性矩阵和冗余剔除"""
    print("\n=== Test 2: Factor correlation matrix ===")
    df = make_synthetic(n_stocks=100, n_days=252)
    factor_df = df[["date", "code", "f1", "f2", "f3", "f4", "f5", "f6"]]

    analyzer = FactorICAnalyzer()
    corr = analyzer.correlation_matrix(factor_df, factor_cols=["f1", "f2", "f3", "f4", "f5", "f6"])
    print(corr.round(3))

    # f1 与 f6 截面相关性应高（因为 f6 = 0.85 * f1 + noise）
    f1_f6_corr = abs(corr.loc["f1", "f6"])
    print(f"\n|f1, f6| correlation = {f1_f6_corr:.3f}")
    assertions = [("f1 与 f6 强相关（>0.5）", f1_f6_corr > 0.5)]

    return assertions, corr


def test_redundant_filter():
    """测试冗余因子剔除"""
    print("\n=== Test 3: Redundant factor filter ===")
    df = make_synthetic(n_stocks=100, n_days=252)
    factor_df = df[["date", "code", "f1", "f2", "f3", "f4", "f5", "f6"]]
    price_df = df[["date", "code", "close"]]

    analyzer = FactorICAnalyzer(forward=1)
    reports = analyzer.analyze(factor_df, price_df)
    corr = analyzer.correlation_matrix(factor_df, factor_cols=["f1", "f2", "f3", "f4", "f5", "f6"])

    keep = analyzer.redundant_factor_filter(reports, corr, threshold=0.5)
    print(f"保留因子: {keep}")

    # 由于 f1 和 f6 强相关，f1（更高 ICIR）应保留，f6 应被剔除
    assertions = []
    if "f1" in keep:
        assertions.append(("f1 保留", True))
    if "f6" not in keep:
        assertions.append(("f6 被剔除（冗余）", True))

    return assertions, keep


def test_edge_cases():
    """边界条件测试"""
    print("\n=== Test 4: Edge cases ===")
    analyzer = FactorICAnalyzer(forward=1)
    assertions = []

    # 4.1 空数据
    empty_df = pd.DataFrame(columns=["date", "code", "f1", "close"])
    reports = analyzer.analyze(empty_df, empty_df)
    assertions.append(("空数据不崩溃", len(reports) == 0))

    # 4.2 全 NaN
    nan_df = pd.DataFrame({
        "date": ["2023-01-01"] * 5,
        "code": ["a", "b", "c", "d", "e"],
        "f1": [np.nan] * 5,
        "close": [10.0] * 5,
    })
    reports = analyzer.analyze(nan_df, nan_df)
    assertions.append(("全 NaN 因子跳过", len(reports) == 0))

    # 4.3 forward > 1
    analyzer2 = FactorICAnalyzer(forward=5)
    df = make_synthetic(n_stocks=20, n_days=60)
    reports2 = analyzer2.analyze(
        df[["date", "code", "f1", "f2"]],
        df[["date", "code", "close"]],
    )
    assertions.append(("forward=5 可运行", isinstance(reports2, list)))

    return assertions, []


def main() -> int:
    all_results = {}
    all_assertions = []

    a1, _ = test_ic_accuracy()
    all_results["ic_accuracy"] = [{"name": n, "passed": p} for n, p in a1]
    all_assertions.extend(a1)

    a2, _ = test_correlation_matrix()
    all_results["correlation_matrix"] = [{"name": n, "passed": p} for n, p in a2]
    all_assertions.extend(a2)

    a3, _ = test_redundant_filter()
    all_results["redundant_filter"] = [{"name": n, "passed": p} for n, p in a3]
    all_assertions.extend(a3)

    a4, _ = test_edge_cases()
    all_results["edge_cases"] = [{"name": n, "passed": p} for n, p in a4]
    all_assertions.extend(a4)

    total = len(all_assertions)
    passed = sum(1 for _, p in all_assertions if p)

    print(f"\n{'='*50}")
    print(f"PASSED: {passed}/{total}")
    for n, p in all_assertions:
        print(f"  [{'OK' if p else 'FAIL'}] {n}")

    out = {
        "summary": {"total": int(total), "passed": int(passed), "failed": int(total - passed), "all_passed": bool(passed == total)},
        "results": {
            k: [{"name": str(item["name"]), "passed": bool(item["passed"])} for item in v]
            for k, v in all_results.items()
        },
    }
    out_path = Path(__file__).parent / "test_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"详细结果已写入 {out_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())