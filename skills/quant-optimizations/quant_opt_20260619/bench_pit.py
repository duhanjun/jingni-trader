"""
PIT 验证器 vs jingni-trader 原 data-engine 行为对比

目的：演示 PIT 校验如何在 jingni-trader 现有架构中识别出 look-ahead 风险
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import numpy as np
import time

from skills.quant-optimizations.quant_opt_20260619.pit import (
    PITSpec, PITValidator, make_synthetic_pit, PITDataFrame
)


def main():
    print("=" * 70)
    print("PIT 验证器演示 / 风险审计")
    print("=" * 70)

    df = make_synthetic_pit(n_stocks=5, n_periods=12)
    print(f"\n合成 PIT 数据集: {len(df)} 行, "
          f"{df['code'].nunique()} 只股票, {df['period'].nunique()} 期")

    spec = PITSpec(feature_name="roe", period_col="period", asof_col="asof", freq="q")
    v = PITValidator(df, spec)

    eval_ts_list = [
        pd.Timestamp("2023-04-01"),
        pd.Timestamp("2023-09-01"),
        pd.Timestamp("2024-04-01"),
    ]
    report = v.audit_pipeline(eval_ts_list)
    print(f"\n审计 {report['n_eval_points']} 个回测时点")
    print(f"按严重度统计: {report['by_severity']}")
    print(f"问题总数: {report['total_issues']}")
    if report["details"]:
        print("详细时点:")
        for d in report["details"]:
            print(f"  {d['as_of'].date()}: "
                  f"look-ahead={d['n_lookahead']}, "
                  f"version_conflict={d['n_version_conflict']}")

    print("\n" + "=" * 70)
    print("【对比 jingni-trader 现有行为】")
    print("=" * 70)
    print("""
jingni-trader 现有 data-engine (skills/data-engine/scripts/adapters/*.py)：
  - 直接从 Tushare/Baostock 拉取日线和财务数据
  - 落 parquet 文件，data-engine 不感知 "announce_date" 字段
  - factor-engine 在每个 date 计算因子时，merge 财务数据用的是原始 report_date
  - 这意味着：Q1 财报 4 月底发布，但若历史库中存的是 Q1 实际发生日期 (3月底)
    因子在 4 月初就用了 Q1 数据 → look-ahead

引入 PITValidator 后：
  1. data-engine 落数据时，必须保留 announce_date (asof) 字段
  2. factor-engine 在 factor_df 上跑 PITValidator.audit_pipeline
  3. 任何发现 look-ahead 的因子 _neutral 列会被标记，不参与 alpha_score 融合
  4. 给出 PIT 审计报告，写入 IC_report.json 的 metadata.pit_audit 字段
""")

    print("=" * 70)
    print("PIT 校验性能基线")
    print("=" * 70)
    big = make_synthetic_pit(n_stocks=200, n_periods=40)
    print(f"大数据集: {len(big)} 行, {big['code'].nunique()} 股票")
    spec = PITSpec(feature_name="big", period_col="period", asof_col="asof", freq="q")
    pit = PITDataFrame(big, spec)

    t0 = time.time()
    for _ in range(50):
        pit.filter_asof(pd.Timestamp("2024-06-01"))
    elapsed = time.time() - t0
    print(f"50次 filter_asof: {elapsed:.3f}s ({elapsed*20:.1f}ms/次)")


if __name__ == "__main__":
    main()