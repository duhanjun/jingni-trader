"""
向量化 IC 分析验证测试

验证内容：
1. 正确性：向量化 IC 与 scipy 逐日计算结果一致
2. 性能对比：向量化 vs factor-engine 逐日循环
3. 多因子批量计算
4. 边界条件：样本不足、空数据
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats

from quant_opt.vectorized_ic import VectorizedICAnalyzer, calc_ic_vectorized
from quant_opt.tests._test_utils import generate_synthetic_data, generate_factor_data


def test_correctness_vs_scipy():
    """正确性：向量化 IC 与 scipy spearmanr 逐日计算对比"""
    print("\n=== 测试1: 正确性（向量化 vs scipy 逐日）===")
    data = generate_synthetic_data(n_codes=50, n_days=120)
    factor_df, fwd = generate_factor_data(data, n_factors=3)

    analyzer = VectorizedICAnalyzer()

    for factor in ["factor_0", "factor_1"]:
        for fwd_col in ["ret_forward_1d", "ret_forward_5d"]:
            # 向量化
            vec_ic = analyzer.calc_ic_series(
                factor_df, fwd, factor, fwd_col, ic_type="spearman"
            )

            # scipy 逐日（模拟 factor-engine._calc_ic）
            merged = factor_df[["code", "date", factor]].merge(
                fwd[["code", "date", fwd_col]], on=["code", "date"], how="inner"
            ).dropna(subset=[factor, fwd_col])

            scipy_ic = {}
            for dt in sorted(merged["date"].unique()):
                cross = merged[merged["date"] == dt]
                if len(cross) < 10:
                    continue
                ic, _ = stats.spearmanr(cross[factor], cross[fwd_col])
                if not np.isnan(ic):
                    scipy_ic[dt] = ic
            scipy_series = pd.Series(scipy_ic)

            # 对齐比较
            aligned = pd.concat([vec_ic.rename("vec"), scipy_series.rename("scipy")], axis=1).dropna()
            if len(aligned) == 0:
                continue
            diff = (aligned["vec"] - aligned["scipy"]).abs().max()
            assert diff < 1e-10, f"{factor}/{fwd_col} IC 误差过大: {diff}"
            print(f"  ✓ {factor}/{fwd_col}: 向量化与 scipy 一致，"
                  f"最大误差 {diff:.2e}, IC均值 {vec_ic.mean():.4f}")


def test_pearson_ic():
    """测试 Pearson IC"""
    print("\n=== 测试2: Pearson IC ===")
    data = generate_synthetic_data(n_codes=40, n_days=80)
    factor_df, fwd = generate_factor_data(data, n_factors=2)

    analyzer = VectorizedICAnalyzer()
    pearson_ic = analyzer.calc_ic_series(
        factor_df, fwd, "factor_0", "ret_forward_5d", ic_type="pearson"
    )
    assert not pearson_ic.empty
    print(f"  ✓ Pearson IC 计算: {len(pearson_ic)} 日, 均值 {pearson_ic.mean():.4f}")


def test_multi_factor_summary():
    """测试多因子 IC 汇总"""
    print("\n=== 测试3: 多因子 IC 汇总 ===")
    data = generate_synthetic_data(n_codes=50, n_days=150)
    factor_df, fwd = generate_factor_data(data, n_factors=5)

    analyzer = VectorizedICAnalyzer()
    summary = analyzer.calc_ic_summary(
        factor_df, fwd,
        factor_names=[f"factor_{i}" for i in range(5)],
        forward_cols=["ret_forward_1d", "ret_forward_5d", "ret_forward_20d"],
    )

    for fwd_col, ic_list in summary.items():
        print(f"  {fwd_col}:")
        for item in ic_list:
            print(f"    {item['factor']}: IC={item['ic_mean']:.4f}, "
                  f"IR={item['ic_ir']:.4f}, t={item['ic_t_stat']:.2f}")

    # factor_0 是有效因子，应有更高的 |IC|
    fwd5 = summary.get("ret_forward_5d", [])
    if fwd5:
        ic_map = {item["factor"]: abs(item["ic_mean"]) for item in fwd5}
        if "factor_0" in ic_map:
            f0_ic = ic_map["factor_0"]
            others = [v for k, v in ic_map.items() if k != "factor_0"]
            if others:
                print(f"  ✓ factor_0 |IC|={f0_ic:.4f} vs 其他均值 {np.mean(others):.4f}")


def test_performance_vs_loop():
    """性能对比：向量化 vs 逐日循环"""
    print("\n=== 测试4: 性能对比（向量化 vs 逐日循环）===")
    data = generate_synthetic_data(n_codes=100, n_days=250)
    factor_df, fwd = generate_factor_data(data, n_factors=5)

    analyzer = VectorizedICAnalyzer()

    # 向量化
    t0 = time.time()
    vec_summary = analyzer.calc_ic_summary(
        factor_df, fwd,
        factor_names=[f"factor_{i}" for i in range(5)],
        forward_cols=["ret_forward_1d", "ret_forward_5d", "ret_forward_20d"],
    )
    t1 = time.time()
    vec_time = t1 - t0

    # 逐日循环（模拟 factor-engine._calc_ic）
    t2 = time.time()
    merged = factor_df.merge(
        fwd[["code", "date", "ret_forward_1d", "ret_forward_5d", "ret_forward_20d"]],
        on=["code", "date"], how="inner"
    )
    loop_summary = {}
    for fwd_col in ["ret_forward_1d", "ret_forward_5d", "ret_forward_20d"]:
        ic_list = []
        for factor in [f"factor_{i}" for i in range(5)]:
            ic_series = []
            for dt in sorted(merged["date"].unique()):
                cross = merged[merged["date"] == dt].dropna(subset=[factor, fwd_col])
                if len(cross) < 10:
                    continue
                ic, _ = stats.spearmanr(cross[factor], cross[fwd_col])
                if not np.isnan(ic):
                    ic_series.append(ic)
            if ic_series:
                ic_arr = np.array(ic_series)
                ic_list.append({
                    "factor": factor,
                    "ic_mean": ic_arr.mean(),
                    "ic_ir": ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
                })
        loop_summary[fwd_col] = ic_list
    t3 = time.time()
    loop_time = t3 - t2

    speedup = loop_time / vec_time if vec_time > 0 else float("inf")
    print(f"  向量化（5因子×3周期）: {vec_time*1000:.1f}ms")
    print(f"  逐日循环（5因子×3周期）: {loop_time*1000:.1f}ms")
    print(f"  加速比: {speedup:.1f}x")

    # 验证结果一致
    for fwd_col in ["ret_forward_1d", "ret_forward_5d", "ret_forward_20d"]:
        vec_list = vec_summary.get(fwd_col, [])
        loop_list = loop_summary.get(fwd_col, [])
        for v, l in zip(vec_list, loop_list):
            diff = abs(v["ic_mean"] - l["ic_mean"])
            assert diff < 1e-6, f"{fwd_col}/{v['factor']} IC均值不一致: {diff}"
    print(f"  ✓ 结果一致（IC均值误差 < 1e-6）")


def test_edge_cases():
    """测试边界条件"""
    print("\n=== 测试5: 边界条件 ===")
    analyzer = VectorizedICAnalyzer()

    # 空数据
    result = analyzer.calc_ic_summary(
        pd.DataFrame(columns=["code", "date", "f"]),
        pd.DataFrame(columns=["code", "date", "ret_forward_5d"]),
    )
    assert result == {}
    print("  ✓ 空数据返回空字典")

    # 样本不足（每日 < 10 只）
    data = generate_synthetic_data(n_codes=5, n_days=30)
    factor_df, fwd = generate_factor_data(data, n_factors=1)
    ic = analyzer.calc_ic_series(factor_df, fwd, "factor_0", "ret_forward_5d")
    # 每日仅 5 只 < 10，应被过滤
    print(f"  ✓ 样本不足时 IC 序列长度: {len(ic)}（应过滤 <10 的日期）")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("向量化 IC 分析验证测试")
    print("=" * 60)

    tests = [
        test_correctness_vs_scipy,
        test_pearson_ic,
        test_multi_factor_summary,
        test_performance_vs_loop,
        test_edge_cases,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ✗ 失败: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
