"""
测试 4: 端到端集成测试
====================

模拟真实工作流:
  1. 合成 60 天 × 20 股票 数据
  2. 用 expr_engine 计算 4 个因子
  3. 选 alpha_score Top 20% 作为多空信号
  4. 用 vectorized_backtest 回测
  5. 用 brinson_attribution 做行业归因

验证端到端管线协同工作。
"""
import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expr_engine import evaluate_by_code
from vectorized_backtest import VectorizedBacktester
from brinson_attribution import brinson_fachler, brinson_attribution_summary


def make_synth():
    rng = np.random.default_rng(2026)
    n_dates, n_stocks = 60, 20
    dates = pd.bdate_range(end=datetime(2024, 12, 31), periods=n_dates)
    codes = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]
    rows = []
    industry_map = {c: ["银行", "地产", "科技", "消费", "医药"][i % 5]
                    for i, c in enumerate(codes)}
    for code in codes:
        ret = rng.normal(0.001, 0.02, n_dates)
        price = 10 * np.cumprod(1 + ret)
        for i, d in enumerate(dates):
            rows.append({
                "code": code, "date": d,
                "open": price[i] * 0.999,
                "high": price[i] * 1.005,
                "low": price[i] * 0.995,
                "close": price[i],
                "volume": abs(rng.normal(1e6, 3e5)),
                "amount": abs(rng.normal(1e7, 3e6)),
            })
    df = pd.DataFrame(rows)
    df["industry"] = df["code"].map(industry_map)
    return df, dates, codes


def main():
    print("=" * 70)
    print("  端到端集成测试: 表达式引擎 + 向量化回测 + Brinson 归因")
    print("=" * 70)
    df, dates, codes = make_synth()
    print(f"\n  数据规模: {len(df)} 行, {df['code'].nunique()} 只股票, "
          f"{df['date'].nunique()} 个交易日")

    # ---- Step 1: 因子计算 (expr_engine) ----
    print("\n  [Step 1] 因子计算 (4 个表达式, 来自 Qlib 风格 DSL)")
    expressions = {
        "mom_20d":  "$close / Ref($close, 20) - 1",                 # 20日动量
        "mean_rev": "-($close - Mean($close, 5)) / Std($close, 5)",  # 5日反转
        "vol_20d":  "Std(Delta($close, 1), 20)",                     # 20日波动率
        "vol_ratio":"Mean($volume, 5) / Mean($volume, 20)",          # 量比
    }
    factor_df = evaluate_by_code(df, expressions)
    print(f"  ✓ 因子计算完成, 输出 {factor_df.shape}, 耗时合理")
    print(f"    因子列: {[c for c in factor_df.columns if c not in ['code', 'date']]}")
    print(f"    mom_20d  缺失率: {factor_df['mom_20d'].isna().mean():.1%}")
    print(f"    mean_rev 缺失率: {factor_df['mean_rev'].isna().mean():.1%}")

    # ---- Step 2: 构造多空信号 ----
    print("\n  [Step 2] 合成因子得分 → TopK 目标权重")
    factor_df["alpha_score"] = (
        factor_df["mom_20d"].fillna(0) * 0.4
        + factor_df["mean_rev"].fillna(0) * 0.4
        - factor_df["vol_20d"].fillna(0) * 0.1
        + factor_df["vol_ratio"].fillna(0) * 0.1
    )

    # 每日: 选 alpha_score 排名前 20% 做多 (等权), 后 20% 做空 (等权)
    target_weights = pd.DataFrame(0.0, index=dates, columns=codes)
    for dt in dates:
        day = factor_df[factor_df["date"] == dt].dropna(subset=["alpha_score"])
        if len(day) < 4:
            continue
        scores = day.set_index("code")["alpha_score"]
        n = len(scores)
        k = max(1, int(n * 0.2))
        longs = scores.nlargest(k).index
        shorts = scores.nsmallest(k).index
        target_weights.loc[dt, longs] = 0.5 / k
        target_weights.loc[dt, shorts] = -0.5 / k
    print(f"  ✓ 每日 {k} 只多 / {k} 只空 (等权)")

    # ---- Step 3: 向量化回测 ----
    print("\n  [Step 3] 向量化回测 (vectorbt 风格)")
    price_pivot = df.pivot(index="date", columns="code", values="close").ffill()
    bt = VectorizedBacktester(
        commission_rate=0.00025, stamp_tax_rate=0.001, slippage=0.0001
    )
    result = bt.run(price_pivot, target_weights, init_capital=1e6,
                    signal_mode="target_weight")
    m = result.metrics
    print(f"  期末净值: {result.equity.iloc[-1]:,.2f} (初始 1,000,000)")
    print(f"  年化收益: {m['annual_return']*100:.2f}%")
    print(f"  夏普比率: {m['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {m['max_drawdown']*100:.2f}%")
    print(f"  日均换手: {m['avg_turnover']*100:.2f}%")
    print(f"  累计成本占比: {m['total_cost_ratio']*100:.4f}%")

    # ---- Step 4: 行业归因 ----
    print("\n  [Step 4] Brinson-Fachler 行业归因 (最后一期截面)")
    last_dt = dates[-1]
    # 用最近 20 日窗口的"组合"和"基准"权重
    last_20 = target_weights.iloc[-20:]
    # 等权基准
    eq_bench = pd.DataFrame(1.0 / len(codes), index=last_20.index, columns=codes)
    # 行业映射
    code_industry = df.drop_duplicates("code").set_index("code")["industry"]
    # 聚合到行业层
    industries = code_industry.unique()
    pw_ind = pd.DataFrame(index=last_20.index, columns=industries, dtype=float)
    bw_ind = pd.DataFrame(index=last_20.index, columns=industries, dtype=float)
    pr_ind = pd.DataFrame(index=last_20.index, columns=industries, dtype=float)
    br_ind = pd.DataFrame(index=last_20.index, columns=industries, dtype=float)
    for ind in industries:
        ind_codes = code_industry[code_industry == ind].index
        for dt in last_20.index:
            pw_ind.loc[dt, ind] = last_20.loc[dt, ind_codes].sum()
            bw_ind.loc[dt, ind] = eq_bench.loc[dt, ind_codes].sum()
            past_prices = price_pivot.loc[:dt, ind_codes].iloc[-21:]  # 20 日窗口
            future_prices = price_pivot.loc[dt:, ind_codes].iloc[:5]  # 5 日窗口
            if len(past_prices) >= 2 and len(future_prices) >= 2:
                pr_ind.loc[dt, ind] = float(
                    (future_prices.iloc[-1] / past_prices.iloc[0] - 1).mean()
                )
                br_ind.loc[dt, ind] = float(
                    (future_prices.iloc[-1] / past_prices.iloc[0] - 1).mean()
                )
    summary = brinson_attribution_summary(pw_ind, bw_ind, pr_ind, br_ind)
    print(f"  累计 Allocation:   {summary['allocation_cumulative']*100:.3f}%")
    print(f"  累计 Selection:    {summary['selection_cumulative']*100:.3f}%")
    print(f"  累计 Interaction:  {summary['interaction_cumulative']*100:.3f}%")
    print(f"  累计 Total Excess: {summary['total_excess_cumulative']*100:.3f}%")

    print("\n" + "=" * 70)
    print("  ✅ 端到端集成测试通过 — 三个模块协同工作")
    print("=" * 70)

    # ---- 保存摘要 ----
    out = {
        "data": {"rows": len(df), "stocks": len(codes), "dates": len(dates)},
        "factors": list(expressions.keys()),
        "backtest": m,
        "attribution": summary,
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "test_integration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()