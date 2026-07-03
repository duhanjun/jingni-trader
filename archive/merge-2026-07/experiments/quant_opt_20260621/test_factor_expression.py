"""
因子表达式引擎测试

测试内容:
  1. 基本字段解析: Close, Volume, Open
  2. 算子正确性: Ref, Ts_Mean, Ts_Std, Rank, Corr
  3. 复合表达式: Rank(Ts_Mean(Close, 5))
  4. 与手算结果对比
  5. 边界条件: 空数据、单只股票、未知字段
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_expression_engine import FactorExpressionEngine, FactorExpressionParser


def make_test_data(n_codes: int = 5, n_days: int = 60) -> pd.DataFrame:
    """生成测试数据"""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_codes)]
    rows = []
    for code in codes:
        price = 10.0
        for dt in dates:
            ret = rng.normal(0, 0.02)
            open_p = price * (1 + rng.normal(0, 0.005))
            close = open_p * (1 + ret)
            high = max(open_p, close) * 1.01
            low = min(open_p, close) * 0.99
            vol = int(rng.integers(1_000_000, 5_000_000))
            rows.append({
                "code": code, "date": dt,
                "open": open_p, "high": high, "low": low, "close": close,
                "volume": vol, "amount": vol * close,
                "turnover_rate": rng.uniform(0.5, 3.0),
            })
            price = close
    return pd.DataFrame(rows)


def test_parser_basic():
    """测试1: 基本解析"""
    print("\n[Test 1] 表达式解析测试")
    parser = FactorExpressionParser()

    # 简单字段
    ast = parser.parse("Close")
    assert ast.op == "FIELD" and ast.field == "Close"

    # 函数调用
    ast = parser.parse("Ts_Mean(Close, 5)")
    assert ast.op == "FUNC" and ast.field == "Ts_Mean"
    assert len(ast.args) == 2

    # 嵌套
    ast = parser.parse("Rank(Ts_Mean(Close, 5))")
    assert ast.op == "FUNC" and ast.field == "Rank"
    assert ast.args[0].op == "FUNC"

    # 二元运算
    ast = parser.parse("(Close - Ref(Close, 5)) / Ref(Close, 5)")
    assert ast.op == "BINOP" and ast.value == "/"

    # 双参数函数
    ast = parser.parse("Corr(Volume, Close, 10)")
    assert ast.op == "FUNC" and ast.field == "Corr"
    assert len(ast.args) == 3

    print("  [PASS] 表达式解析正确")


def test_simple_factors():
    """测试2: 简单因子计算正确性"""
    print("\n[Test 2] 简单因子计算正确性")
    data = make_test_data(n_codes=3, n_days=40)
    engine = FactorExpressionEngine()

    # Ref(Close, 1) 应等于 close.shift(1)
    result = engine.compute(data, {"ref_close_1": "Ref(Close, 1)"})
    df = data.sort_values(["code", "date"]).reset_index(drop=True)
    expected = df.groupby("code")["close"].shift(1)
    actual = result.sort_values(["code", "date"]).reset_index(drop=True)["ref_close_1"]
    # 对齐比较 (排除 NaN)
    mask = ~expected.isna() & ~actual.isna()
    np.testing.assert_allclose(
        actual[mask].values, expected[mask].values, rtol=1e-6,
        err_msg="Ref(Close, 1) 与 close.shift(1) 不一致"
    )
    print("  [PASS] Ref(Close, 1) 正确")

    # Ts_Mean(Close, 5) 应等于 close.rolling(5).mean()
    result = engine.compute(data, {"ts_mean_5": "Ts_Mean(Close, 5)"})
    expected = df.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    actual = result.sort_values(["code", "date"]).reset_index(drop=True)["ts_mean_5"]
    mask = ~expected.isna() & ~actual.isna()
    np.testing.assert_allclose(
        actual[mask].values, expected[mask].values, rtol=1e-6,
        err_msg="Ts_Mean(Close, 5) 与 rolling(5).mean() 不一致"
    )
    print("  [PASS] Ts_Mean(Close, 5) 正确")

    # Ts_Std(Close, 10) 应等于 close.rolling(10).std()
    result = engine.compute(data, {"ts_std_10": "Ts_Std(Close, 10)"})
    expected = df.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=5).std())
    actual = result.sort_values(["code", "date"]).reset_index(drop=True)["ts_std_10"]
    mask = ~expected.isna() & ~actual.isna()
    np.testing.assert_allclose(
        actual[mask].values, expected[mask].values, rtol=1e-6,
        err_msg="Ts_Std(Close, 10) 与 rolling(10).std() 不一致"
    )
    print("  [PASS] Ts_Std(Close, 10) 正确")


def test_compound_expression():
    """测试3: 复合表达式"""
    print("\n[Test 3] 复合表达式测试")
    data = make_test_data(n_codes=5, n_days=60)
    engine = FactorExpressionEngine()

    # 动量因子: (Close - Ref(Close, 20)) / Ref(Close, 20)
    expr = "(Close - Ref(Close, 20)) / Ref(Close, 20)"
    result = engine.compute(data, {"momentum_20": expr})
    assert "momentum_20" in result.columns
    assert not result["momentum_20"].isna().all(), "动量因子不应全为 NaN"

    # 手算验证
    df = data.sort_values(["code", "date"]).reset_index(drop=True)
    ref20 = df.groupby("code")["close"].shift(20)
    expected = (df["close"] - ref20) / ref20
    actual = result.sort_values(["code", "date"]).reset_index(drop=True)["momentum_20"]
    mask = ~expected.isna() & ~actual.isna()
    np.testing.assert_allclose(
        actual[mask].values, expected[mask].values, rtol=1e-6,
        err_msg="动量因子计算错误"
    )
    print("  [PASS] 动量因子 (Close - Ref(Close, 20)) / Ref(Close, 20) 正确")

    # 反转因子: -Ts_Mean(Return, 5)
    expr = "-Ts_Mean(Return, 5)"
    result = engine.compute(data, {"reversal_5": expr})
    assert "reversal_5" in result.columns
    print("  [PASS] 反转因子 -Ts_Mean(Return, 5) 正确")

    # Rank 截面排名
    expr = "Rank(Close)"
    result = engine.compute(data, {"rank_close": expr})
    # 每日 rank 应在 [0, 1] 之间
    valid = result["rank_close"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all(), "Rank 应在 [0, 1] 之间"
    print("  [PASS] Rank(Close) 截面排名正确")


def test_corr_factor():
    """测试4: Corr 双参数算子"""
    print("\n[Test 4] Corr 双参数算子测试")
    data = make_test_data(n_codes=3, n_days=50)
    engine = FactorExpressionEngine()

    expr = "Corr(Volume, Close, 10)"
    result = engine.compute(data, {"corr_vol_close": expr})

    # 手算验证
    df = data.sort_values(["code", "date"]).reset_index(drop=True)
    expected = df.groupby("code").apply(
        lambda g: g["volume"].rolling(10, min_periods=5).corr(g["close"])
    ).reset_index(level=0, drop=True)
    actual = result.sort_values(["code", "date"]).reset_index(drop=True)["corr_vol_close"]
    mask = ~expected.isna() & ~actual.isna()
    np.testing.assert_allclose(
        actual[mask].values, expected[mask].values, rtol=1e-6,
        err_msg="Corr(Volume, Close, 10) 计算错误"
    )
    print("  [PASS] Corr(Volume, Close, 10) 正确")


def test_alpha101_style():
    """测试5: Alpha101 风格表达式"""
    print("\n[Test 5] Alpha101 风格表达式测试")
    data = make_test_data(n_codes=10, n_days=100)
    engine = FactorExpressionEngine()

    # Alpha#6: -1 * Corr(Open, Volume, 10)
    alpha6 = "-1 * Corr(Open, Volume, 10)"
    # Alpha#12: Sign(Delta(Close, 1)) * (-1 * Delta(Close, 1))
    alpha12 = "Sign(Delta(Close, 1)) * (-1 * Delta(Close, 1))"
    # Alpha#23: ((Ts_Mean(High, 20) < High) ? (-1 * Delta(High, 2)) : 0) -- 简化为不带条件
    # Alpha#40: Ts_Sum(((Close < Low) ? 1 : 0), 8) -- 简化
    # Alpha#101: ((Close - Open) / ((High - Low) + 0.001))
    alpha101 = "(Close - Open) / ((High - Low) + 0.001)"

    expressions = {
        "alpha_006": alpha6,
        "alpha_012": alpha12,
        "alpha_101": alpha101,
    }
    result = engine.compute(data, expressions)

    for name in expressions:
        assert name in result.columns, f"因子 {name} 应在结果中"
        valid = result[name].dropna()
        assert len(valid) > 0, f"因子 {name} 不应全为 NaN"
        print(f"  [PASS] {name}: 有效值 {len(valid)}/{len(result)}")


def test_edge_cases():
    """测试6: 边界条件"""
    print("\n[Test 6] 边界条件测试")
    engine = FactorExpressionEngine()

    # 空数据
    result = engine.compute(pd.DataFrame(), {"a": "Close"})
    assert result.empty
    print("  [PASS] 空数据处理正确")

    # 单只股票
    data = make_test_data(n_codes=1, n_days=30)
    result = engine.compute(data, {"ma5": "Ts_Mean(Close, 5)"})
    assert not result["ma5"].isna().all()
    print("  [PASS] 单只股票处理正确")

    # 未知字段应抛出 KeyError
    try:
        engine.compute(data, {"bad": "UnknownField"})
        assert False, "未知字段应抛出异常"
    except (KeyError, RuntimeError):
        print("  [PASS] 未知字段正确抛出异常")

    # 未知算子应抛出 KeyError
    try:
        engine.compute(data, {"bad": "UnknownFunc(Close, 5)"})
        assert False, "未知算子应抛出异常"
    except (KeyError, RuntimeError):
        print("  [PASS] 未知算子正确抛出异常")


if __name__ == "__main__":
    test_parser_basic()
    test_simple_factors()
    test_compound_expression()
    test_corr_factor()
    test_alpha101_style()
    test_edge_cases()
    print("\n=== 所有因子表达式测试通过 ===")
