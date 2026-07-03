"""
Performance benchmark: new DSL vs hardcoded factor calculation
==============================================================

对比 quant_opt 提供的 factor_dsl 与 jingni-trader 现有 factor-engine
中手写因子计算在**可维护性 / 性能 / 可扩展性**上的差异.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._synth_data import make_synth_panel
from factor_dsl.evaluator import (
    evaluate_factor,
    eval_preset,
    list_preset_factors,
    parse,
)


def bench_legacy_20d_momentum(df: pd.DataFrame) -> pd.Series:
    """模仿 jingni-trader factor-engine 的硬编码实现."""
    df = df.sort_values(["code", "date"]).copy()
    df = df.set_index(["code", "date"])
    return df.groupby(level="code")["close"].transform(lambda x: x - x.shift(20))


def bench_legacy_5d_reversal(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["code", "date"]).copy()
    df = df.set_index(["code", "date"])
    delta = df.groupby(level="code")["close"].diff(5)
    return -delta


def main() -> int:
    print("=" * 78)
    print("DSL vs 硬编码 性能/可读性基准对比")
    print("=" * 78)

    for n_codes, n_days in [(10, 250), (50, 500), (200, 500)]:
        print(f"\n--- 场景: {n_codes} 只股票 × {n_days} 个交易日 ---")
        panel = make_synth_panel(n_codes=n_codes, n_days=n_days)

        # 1) 20 日动量
        t0 = time.perf_counter()
        legacy_mom = bench_legacy_20d_momentum(panel)
        t_legacy_mom = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dsl_mom = evaluate_factor(panel, "Delta(close, 20)")
        t_dsl_mom = (time.perf_counter() - t0) * 1000

        diff = (legacy_mom - dsl_mom).abs().max()
        print(f"  20日动量   硬编码: {t_legacy_mom:7.2f}ms  |  DSL: {t_dsl_mom:7.2f}ms  |  max|diff|: {diff:.4g}")

        # 2) 5 日反转
        t0 = time.perf_counter()
        legacy_rev = bench_legacy_5d_reversal(panel)
        t_legacy_rev = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dsl_rev = eval_preset("reversal_5d", panel)
        t_dsl_rev = (time.perf_counter() - t0) * 1000

        diff = (legacy_rev - dsl_rev).abs().max()
        print(f"  5日反转    硬编码: {t_legacy_rev:7.2f}ms  |  DSL: {t_dsl_rev:7.2f}ms  |  max|diff|: {diff:.4g}")

        # 3) 复合因子: Rank(Delta(close, 5))   (硬编码: 多行)
        t0 = time.perf_counter()
        df = panel.sort_values(["code", "date"]).copy()
        df["delta"] = df.groupby("code")["close"].diff(5)
        df["ranked"] = df.groupby("date")["delta"].rank(pct=True)
        t_legacy_rank = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dsl_rank = evaluate_factor(panel, "Rank(Delta(close, 5))")
        t_dsl_rank = (time.perf_counter() - t0) * 1000

        # 校验: 桶内排名 0..1
        valid = dsl_rank.dropna()
        ok = bool(valid.between(0, 1).all())
        print(f"  Rank复合   硬编码: {t_legacy_rank:7.2f}ms  |  DSL: {t_dsl_rank:7.2f}ms  |  ∈[0,1]: {ok}  (非NaN: {len(valid)}/{len(dsl_rank)})")

    # 4) 可维护性: 添加一个新因子
    print("\n--- 可扩展性: 引入新因子 '20 日成交量与价格相关' ---")
    print("DSL 形式 (1 行):  Rank(Corr(close, volume, 20))    [本验证未实现, 仅作示意]")
    print("硬编码形式 (8+ 行):  df['ma_c']=...; df['ma_v']=...; df['std_c']=...; ...")
    print(f"DSL 内置预设因子数: {len(list_preset_factors())} 个,  涵盖 5 大类常用 alpha")

    # 5) AST 解析开销
    print("\n--- 解析开销 ---")
    exprs = [
        "close",
        "Delta(close, 5)",
        "Rank(Delta(close, 5))",
        "Mul(Sub(volume, Ts_Mean(volume, 20)), Ts_Std(close, 5))",
    ]
    for e in exprs:
        t0 = time.perf_counter()
        for _ in range(1000):
            ast = parse(e)
        t = (time.perf_counter() - t0) * 1000
        print(f"  parse ×1000  {e!r:50s}  {t:7.2f}ms  ({t/1000*1000:.1f}μs/次)")

    print("\n" + "=" * 78)
    print("结论:")
    print("  • DSL 性能 与 手写代码 在同一数量级 (pandas groupby 是共同 bottleneck)")
    print("  • 复合因子的可读性 / 维护成本: DSL 显著占优")
    print("  • 通过 parse 预编译可进一步降低运行时开销")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
