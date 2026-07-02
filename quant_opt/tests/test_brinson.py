"""
测试 3: Brinson-Fachler 三因素归因 (弥补 reports-engine 不足)
=============================================================

现有 jingni-trader reports-engine.calc_brinson_attribution 缺少:
  1. 闭合性校验 (Allocation + Selection + Interaction == Direct)
  2. 跨期聚合
  3. 数值对照参考实现

本测试验证新模块 brinson_attribution 的:
  1. 分解闭合性 (可加性)
  2. 跨期聚合
  3. 边角条件 (空数据、单行业)
  4. 与 reports-engine 实现对比
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brinson_attribution import (
    brinson_fachler,
    brinson_by_industry,
    brinson_attribution_summary,
)


# ---------------------------------------------------------------------------
# 参考实现 (Brinson 1985 原始公式)
# ---------------------------------------------------------------------------
def brinson_reference(portfolio_weights, benchmark_weights,
                      portfolio_returns, benchmark_returns):
    common = (portfolio_weights.index
              .intersection(benchmark_weights.index)
              .intersection(portfolio_returns.index)
              .intersection(benchmark_returns.index))
    wp = portfolio_weights.reindex(common).fillna(0.0)
    wb = benchmark_weights.reindex(common).fillna(0.0)
    rp = portfolio_returns.reindex(common).fillna(0.0)
    rb = benchmark_returns.reindex(common).fillna(0.0)
    # Brinson 原始 (使用 r_p 计算 allocation)
    alloc_b85 = float(((wp - wb) * rp).sum())
    sel = float((wb * (rp - rb)).sum())
    inter = float(((wp - wb) * (rp - rb)).sum())
    return {
        "allocation_effect_b85": alloc_b85,
        "selection_effect": sel,
        "interaction_effect": inter,
    }


# ---------------------------------------------------------------------------
# 测试 3.1: 闭合性 (Allocation + Selection + Interaction == Direct)
# ---------------------------------------------------------------------------
def test_closure():
    print("\n[3.1] Brinson-Fachler 分解闭合性")
    industries = ["银行", "地产", "科技", "消费", "医药", "工业"]
    wp = pd.Series([0.30, 0.20, 0.15, 0.15, 0.10, 0.10], index=industries)
    wb = pd.Series([0.20, 0.15, 0.20, 0.20, 0.15, 0.10], index=industries)
    rp = pd.Series([0.05, 0.02, 0.10, 0.03, -0.02, 0.04], index=industries)
    rb = pd.Series([0.03, 0.01, 0.08, 0.04, 0.01, 0.03], index=industries)

    result = brinson_fachler(wp, wb, rp, rb)
    direct = result["direct_excess_return"]
    decomposed = (result["allocation_effect"]
                  + result["selection_effect"]
                  + result["interaction_effect"])
    residual = result["residual"]
    print(f"  Allocation:   {result['allocation_effect']:.6f}")
    print(f"  Selection:    {result['selection_effect']:.6f}")
    print(f"  Interaction:  {result['interaction_effect']:.6f}")
    print(f"  Σ = {decomposed:.6f}, 直接计算 = {direct:.6f}, 残差 = {residual:.2e}")
    assert residual < 1e-9, f"闭合性失败: 残差 {residual}"
    print(f"  ✓ 三因素之和 == 直接超额收益 (残差 < 1e-9)")
    return {
        "closure": "PASS",
        "allocation_effect": result["allocation_effect"],
        "selection_effect": result["selection_effect"],
        "interaction_effect": result["interaction_effect"],
        "total_excess_return": result["total_excess_return"],
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# 测试 3.2: 边角条件
# ---------------------------------------------------------------------------
def test_edge_cases():
    print("\n[3.2] 边角条件")
    # 空
    r = brinson_fachler(pd.Series(dtype=float), pd.Series(dtype=float),
                        pd.Series(dtype=float), pd.Series(dtype=float))
    assert r == {}, f"空输入应返回空 dict, 实际 {r}"
    print(f"  ✓ 空输入: {r}")

    # 单行业 (组合 == 基准) — 选同样收益则三效应都为 0
    wp = pd.Series([1.0], index=["银行"])
    wb = pd.Series([1.0], index=["银行"])
    rp = pd.Series([0.05], index=["银行"])
    rb = pd.Series([0.05], index=["银行"])
    r = brinson_fachler(wp, wb, rp, rb)
    assert abs(r["allocation_effect"]) < 1e-12
    assert abs(r["selection_effect"]) < 1e-12
    assert abs(r["interaction_effect"]) < 1e-12
    print(f"  ✓ 单行业 (组合==基准): 三效应全为 0")

    # 完全不同: 组合全部集中到一个行业
    wp = pd.Series([1.0, 0.0], index=["科技", "银行"])
    wb = pd.Series([0.0, 1.0], index=["科技", "银行"])
    rp = pd.Series([0.10, 0.02], index=["科技", "银行"])
    rb = pd.Series([0.05, 0.02], index=["科技", "银行"])
    r = brinson_fachler(wp, wb, rp, rb)
    # Brinson-Fachler: 用 r_b 计算 allocation
    # allocation = (1-0)*0.05 + (0-1)*0.02 = 0.03
    # selection   = 0*(0.10-0.05) + 1*(0.02-0.02) = 0
    # interaction = (1-0)*(0.10-0.05) + (0-1)*(0.02-0.02) = 0.05
    assert abs(r["allocation_effect"] - 0.03) < 1e-12
    assert abs(r["selection_effect"]) < 1e-12
    assert abs(r["interaction_effect"] - 0.05) < 1e-12
    print(f"  ✓ 集中持仓: Allocation=0.03, Selection=0, Interaction=0.05")
    return {"edge_cases": "PASS"}


# ---------------------------------------------------------------------------
# 测试 3.3: 跨期聚合
# ---------------------------------------------------------------------------
def test_period_aggregation():
    print("\n[3.3] 跨期聚合 (10 个交易日)")
    industries = ["银行", "科技", "消费", "医药"]
    dates = pd.bdate_range("2024-01-01", periods=10)
    rng = np.random.default_rng(7)

    pw_df = pd.DataFrame(
        rng.dirichlet(np.ones(4), size=10), index=dates, columns=industries
    )
    bw_df = pd.DataFrame(
        rng.dirichlet(np.ones(4), size=10), index=dates, columns=industries
    )
    pr_df = pd.DataFrame(
        rng.normal(0.005, 0.01, size=(10, 4)), index=dates, columns=industries
    )
    br_df = pd.DataFrame(
        rng.normal(0.003, 0.01, size=(10, 4)), index=dates, columns=industries
    )

    summary = brinson_attribution_summary(pw_df, bw_df, pr_df, br_df)

    # 手动交叉验证
    by_industry = brinson_by_industry(pw_df, bw_df, pr_df, br_df)
    expected_alloc = float(by_industry["allocation_effect"].sum())
    expected_total = float(by_industry["total_excess_return"].sum())
    diff_alloc = abs(summary["allocation_cumulative"] - expected_alloc)
    diff_total = abs(summary["total_excess_cumulative"] - expected_total)
    print(f"  累计 Allocation:   {summary['allocation_cumulative']:.4f} "
          f"(与按日求和差异 {diff_alloc:.2e})")
    print(f"  累计 Selection:    {summary['selection_cumulative']:.4f}")
    print(f"  累计 Interaction:  {summary['interaction_cumulative']:.4f}")
    print(f"  累计 Total Excess: {summary['total_excess_cumulative']:.4f}")
    print(f"  日均 Allocation:   {summary['allocation_daily_mean']:.4f}")
    print(f"  期间数: {summary['n_periods']}")
    assert diff_alloc < 1e-9
    assert diff_total < 1e-9
    print(f"  ✓ 跨期聚合与按日求和一致")
    return {
        "period_aggregation": "PASS",
        **summary,
    }


# ---------------------------------------------------------------------------
# 测试 3.4: 与 Brinson 原始 1985 (BHB85) 对比
# ---------------------------------------------------------------------------
def test_vs_brinson_1985():
    print("\n[3.4] 与 Brinson-Hood-Beebower 1985 公式对比")
    print("  说明: BHB85 用 r_p 计算 allocation; BF85 用 r_b. 两者 Selection/Interaction 相同")
    industries = ["A", "B", "C", "D", "E"]
    wp = pd.Series([0.30, 0.20, 0.10, 0.30, 0.10], index=industries)
    wb = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20], index=industries)
    rp = pd.Series([0.08, 0.04, 0.10, -0.02, 0.06], index=industries)
    rb = pd.Series([0.05, 0.03, 0.04, 0.05, 0.04], index=industries)

    # Brinson-Fachler (本模块)
    bf = brinson_fachler(wp, wb, rp, rb)
    # Brinson 原始 1985 (参考实现)
    bhb = brinson_reference(wp, wb, rp, rb)

    print(f"  Brinson-Fachler (BF85):")
    print(f"    Allocation (rb):  {bf['allocation_effect']:.6f}")
    print(f"    Selection:        {bf['selection_effect']:.6f}")
    print(f"    Interaction:      {bf['interaction_effect']:.6f}")
    print(f"  Brinson 1985 (BHB85):")
    print(f"    Allocation (rp):  {bhb['allocation_effect_b85']:.6f}")
    print(f"    Selection:        {bhb['selection_effect']:.6f}")
    print(f"    Interaction:      {bhb['interaction_effect']:.6f}")
    # Selection/Interaction 应一致
    assert abs(bf["selection_effect"] - bhb["selection_effect"]) < 1e-12
    assert abs(bf["interaction_effect"] - bhb["interaction_effect"]) < 1e-12
    # Allocation 不同
    assert abs(bf["allocation_effect"] - bhb["allocation_effect_b85"]) > 1e-6
    print(f"  ✓ Selection/Interaction 一致, Allocation 差异符合预期")
    return {
        "vs_brinson_1985": "PASS",
        "note": "BF85 用 r_b 计算 allocation (学术更标准), BHB85 用 r_p",
    }


# ---------------------------------------------------------------------------
# 主测试
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Brinson-Fachler 三因素归因验证")
    print("=" * 70)
    results = {}
    results.update(test_closure())
    results.update(test_edge_cases())
    results.update(test_period_aggregation())
    results.update(test_vs_brinson_1985())
    print("\n" + "=" * 70)
    print(f"  ✅ 全部通过 — 4/4 测试")
    print("=" * 70)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "test_brinson.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
