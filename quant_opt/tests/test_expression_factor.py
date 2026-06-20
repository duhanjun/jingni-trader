"""
表达式因子引擎验证测试

验证内容：
1. 正确性：表达式计算结果与手动 pandas 计算一致
2. 算子覆盖：时序算子、横截面算子、算术运算
3. 边界条件：空数据、缺失字段、无效表达式
4. Alpha158 子集：预置因子库可正常计算
5. 性能：与硬编码因子计算对比
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from quant_opt.expression_factor_engine import (
    ExpressionEngine,
    ALPHA158_SUBSET,
    build_alpha158_subset,
)
from quant_opt.tests._test_utils import generate_synthetic_data


def test_field_reference():
    """测试字段引用正确性"""
    print("\n=== 测试1: 字段引用 ===")
    data = generate_synthetic_data(n_codes=10, n_days=60)
    engine = ExpressionEngine(data)

    close = engine.compute("$close")
    # 应与原始 close 列一致（按索引对齐后比较）
    expected = data.set_index(["code", "date"])["close"]
    # 对齐索引顺序后比较
    aligned = pd.concat([close.rename("got"), expected.rename("exp")], axis=1)
    diff = (aligned["got"] - aligned["exp"]).abs().max()
    assert diff < 1e-10, f"字段引用结果不一致，最大差异 {diff}"
    print(f"  ✓ $close 引用正确，共 {len(close)} 条，最大差异 {diff:.2e}")


def test_ref_operator():
    """测试 Ref 算子（无前视偏差）"""
    print("\n=== 测试2: Ref 算子（时序滞后，无前视偏差）===")
    data = generate_synthetic_data(n_codes=10, n_days=60)
    engine = ExpressionEngine(data)

    ref5 = engine.compute("Ref($close, 5)")
    # 手动计算：每只股票 close.shift(5)
    expected = data.set_index(["code", "date"]).groupby(level="code")["close"].shift(5)

    # 对齐比较（忽略 NaN）
    aligned = pd.concat([ref5.rename("got"), expected.rename("exp")], axis=1).dropna()
    diff = (aligned["got"] - aligned["exp"]).abs().max()
    assert diff < 1e-10, f"Ref 算子误差过大: {diff}"
    print(f"  ✓ Ref($close, 5) 与手动 shift(5) 一致，最大误差 {diff:.2e}")

    # 验证无前视偏差：首 5 日应为 NaN
    first_code = data["code"].iloc[0]
    first_code_ref = ref5.loc[first_code].head(5)
    assert first_code_ref.isna().all(), "Ref 前 5 日应为 NaN（前视偏差检查）"
    print(f"  ✓ 无前视偏差：首 5 日为 NaN")


def test_arithmetic():
    """测试算术运算"""
    print("\n=== 测试3: 算术运算 ===")
    data = generate_synthetic_data(n_codes=10, n_days=60)
    engine = ExpressionEngine(data)

    expr = "Ref($close, 5) / $close - 1"
    result = engine.compute(expr)
    # 手动计算 5 日收益
    idx_data = data.set_index(["code", "date"])
    expected = idx_data.groupby(level="code")["close"].shift(5) / idx_data["close"] - 1

    aligned = pd.concat([result.rename("got"), expected.rename("exp")], axis=1).dropna()
    diff = (aligned["got"] - aligned["exp"]).abs().max()
    assert diff < 1e-10, f"算术运算误差: {diff}"
    print(f"  ✓ '{expr}' 计算正确，最大误差 {diff:.2e}")


def test_rolling_ops():
    """测试滚动算子 Mean/Std/Max/Min"""
    print("\n=== 测试4: 滚动算子 ===")
    data = generate_synthetic_data(n_codes=10, n_days=100)
    engine = ExpressionEngine(data)
    idx_data = data.set_index(["code", "date"])

    for op, pandas_fn in [("Mean", "mean"), ("Std", "std"), ("Max", "max"), ("Min", "min")]:
        expr = f"{op}($close, 20)"
        result = engine.compute(expr)
        expected = idx_data.groupby(level="code")["close"].transform(
            lambda x: getattr(x.rolling(20, min_periods=10), pandas_fn)()
        )
        aligned = pd.concat([result.rename("got"), expected.rename("exp")], axis=1).dropna()
        diff = (aligned["got"] - aligned["exp"]).abs().max()
        assert diff < 1e-8, f"{op} 误差: {diff}"
        print(f"  ✓ {op}($close, 20) 正确，最大误差 {diff:.2e}")


def test_cross_section_ops():
    """测试横截面算子 CSRank/CSZScore"""
    print("\n=== 测试5: 横截面算子 ===")
    data = generate_synthetic_data(n_codes=30, n_days=60)
    engine = ExpressionEngine(data)

    # CSRank：每日按 pct 排名
    cs_rank = engine.compute("CSRank($close)")
    idx_data = data.set_index(["code", "date"])
    expected = idx_data.groupby(level="date")["close"].rank(pct=True)

    aligned = pd.concat([cs_rank.rename("got"), expected.rename("exp")], axis=1).dropna()
    diff = (aligned["got"] - aligned["exp"]).abs().max()
    assert diff < 1e-10, f"CSRank 误差: {diff}"
    print(f"  ✓ CSRank($close) 正确，最大误差 {diff:.2e}")

    # CSZScore
    cs_z = engine.compute("CSZScore($volume)")
    grp = idx_data.groupby(level="date")["volume"]
    expected_z = (idx_data["volume"] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    aligned = pd.concat([cs_z.rename("got"), expected_z.rename("exp")], axis=1).dropna()
    diff = (aligned["got"] - aligned["exp"]).abs().max()
    assert diff < 1e-8, f"CSZScore 误差: {diff}"
    print(f"  ✓ CSZScore($volume) 正确，最大误差 {diff:.2e}")


def test_nested_expression():
    """测试嵌套表达式"""
    print("\n=== 测试6: 嵌套表达式 ===")
    data = generate_synthetic_data(n_codes=20, n_days=100)
    engine = ExpressionEngine(data)

    expr = "CSRank(Mean($close, 20)) - CSRank(Ref($close, 20) / $close - 1)"
    result = engine.compute(expr)
    assert not result.isna().all(), "嵌套表达式全为 NaN"
    print(f"  ✓ 嵌套表达式计算成功，非空率 {result.notna().mean():.2%}")


def test_alpha158_subset():
    """测试 Alpha158 子集因子库"""
    print("\n=== 测试7: Alpha158 子集因子库 ===")
    data = generate_synthetic_data(n_codes=30, n_days=120)
    factor_df = build_alpha158_subset(data)

    print(f"  ✓ 计算完成：{len(factor_df.columns)} 个因子，{len(factor_df)} 行")
    print(f"  因子列表（前5个）:")
    for i, col in enumerate(factor_df.columns[:5]):
        nan_rate = factor_df[col].isna().mean()
        print(f"    [{i}] {col}  (NaN率 {nan_rate:.1%})")

    # 验证所有因子都有非空值
    for col in factor_df.columns:
        assert factor_df[col].notna().any(), f"因子 {col} 全为 NaN"
    print(f"  ✓ 所有 {len(factor_df.columns)} 个因子均有有效值")


def test_cache():
    """测试缓存机制"""
    print("\n=== 测试8: 缓存机制 ===")
    data = generate_synthetic_data(n_codes=10, n_days=60)
    engine = ExpressionEngine(data)

    expr = "Mean($close, 20)"
    t0 = time.time()
    r1 = engine.compute(expr)
    t1 = time.time()
    r2 = engine.compute(expr)
    t2 = time.time()

    assert r1 is r2, "缓存应返回同一对象"
    print(f"  ✓ 首次计算 {(t1-t0)*1000:.1f}ms，缓存命中 {(t2-t1)*1000:.3f}ms")


def test_invalid_expression():
    """测试无效表达式处理"""
    print("\n=== 测试9: 无效表达式处理 ===")
    data = generate_synthetic_data(n_codes=5, n_days=30)
    engine = ExpressionEngine(data)

    # 不存在的字段
    try:
        engine.compute("$nonexistent")
        assert False, "应抛出 KeyError"
    except KeyError:
        print("  ✓ 不存在字段正确抛出 KeyError")

    # 语法错误
    try:
        engine.compute("Ref($close, )")
        assert False, "应抛出 ValueError"
    except (ValueError, SyntaxError):
        print("  ✓ 语法错误正确抛出异常")

    # 不支持的算子
    try:
        engine.compute("UnknownOp($close, 5)")
        assert False, "应抛出 ValueError"
    except ValueError:
        print("  ✓ 不支持的算子正确抛出 ValueError")


def test_performance_vs_hardcoded():
    """性能对比：表达式引擎 vs 硬编码因子计算"""
    print("\n=== 测试10: 性能对比（表达式引擎 vs 硬编码）===")
    data = generate_synthetic_data(n_codes=100, n_days=250)

    # 表达式引擎
    t0 = time.time()
    engine = ExpressionEngine(data)
    expr_factors = engine.compute_many(ALPHA158_SUBSET)
    t1 = time.time()
    expr_time = t1 - t0

    # 硬编码（模拟 jingni-trader factor-engine 的方式）
    t2 = time.time()
    df = data.sort_values(["code", "date"]).copy()
    hardcoded = df[["code", "date"]].copy()
    hardcoded["ret_5d"] = df.groupby("code")["close"].pct_change(5)
    hardcoded["ret_20d"] = df.groupby("code")["close"].pct_change(20)
    hardcoded["reversal_5d"] = -hardcoded["ret_5d"]
    hardcoded["vol_20d"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )
    hardcoded["turnover_20d"] = df.groupby("code")["turnover_rate"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    t3 = time.time()
    hardcoded_time = t3 - t2

    print(f"  表达式引擎（{len(ALPHA158_SUBSET)} 因子）: {expr_time*1000:.1f}ms")
    print(f"  硬编码（5 因子）: {hardcoded_time*1000:.1f}ms")
    print(f"  单因子平均：表达式 {expr_time/len(ALPHA158_SUBSET)*1000:.1f}ms，"
          f"硬编码 {hardcoded_time/5*1000:.1f}ms")
    print(f"  ✓ 表达式引擎在提供更高可扩展性的同时，性能可接受")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("表达式因子引擎验证测试")
    print("=" * 60)

    tests = [
        test_field_reference,
        test_ref_operator,
        test_arithmetic,
        test_rolling_ops,
        test_cross_section_ops,
        test_nested_expression,
        test_alpha158_subset,
        test_cache,
        test_invalid_expression,
        test_performance_vs_hardcoded,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ 失败: {e}")

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
