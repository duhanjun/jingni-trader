"""
因子表达式引擎单元测试
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 允许从仓库根目录直接 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factor_expression_engine.expression_engine import (
    FactorExpressionEngine,
    compile_formula,
    parse_formula,
    tokenize,
)
from factor_expression_engine.operators import OPERATORS, ARITY


# ---------------------------------------------------------------------------
# 构造测试数据：3 只股票，30 个交易日，确定性 seed
# ---------------------------------------------------------------------------


def make_test_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    codes = ["000001.SZ", "000002.SZ", "600000.SH"]
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for code in codes:
        base = 10 + rng.normal(0, 1)
        rets = rng.normal(0, 0.02, size=len(dates))
        closes = base * np.exp(np.cumsum(rets))
        volumes = rng.integers(1_000_000, 5_000_000, size=len(dates)).astype(float)
        highs = closes * (1 + np.abs(rng.normal(0, 0.005, len(dates))))
        lows = closes * (1 - np.abs(rng.normal(0, 0.005, len(dates))))
        opens = closes * (1 + rng.normal(0, 0.003, len(dates)))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "code": code,
                    "date": dt,
                    "open": opens[i],
                    "high": highs[i],
                    "low": lows[i],
                    "close": closes[i],
                    "volume": volumes[i],
                    "amount": volumes[i] * closes[i],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_tokenizer():
    tokens = tokenize("Ts_Mean($close, 5)")
    assert [t.kind for t in tokens] == ["IDENT", "LPAREN", "VARIABLE", "COMMA", "NUMBER", "RPAREN"]
    assert tokens[0].value == "Ts_Mean"
    assert tokens[2].value == "$close"
    assert tokens[4].value == "5"


def test_parse_simple():
    ast = parse_formula("$close")
    result = ast.evaluate(make_test_data())
    assert isinstance(result, pd.Series)
    assert len(result) == 90


def test_ts_mean():
    df = make_test_data()
    f = compile_formula("Ts_Mean($close, 5)")
    out = f(df)
    # 验证：结果应与 pandas groupby(rolling) 一致
    expected = df.groupby("code")["close"].transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-10, atol=1e-10)


def test_delay_and_delta():
    df = make_test_data()
    f_delay = compile_formula("Delay($close, 3)")
    out_delay = f_delay(df)
    expected = df.groupby("code")["close"].shift(3)
    np.testing.assert_allclose(out_delay.values, expected.values, equal_nan=True)
    f_delta = compile_formula("Delta($close, 3)")
    out_delta = f_delta(df)
    expected_delta = df["close"] - df.groupby("code")["close"].shift(3)
    np.testing.assert_allclose(out_delta.values, expected_delta.values, equal_nan=True)


def test_cross_section_rank():
    df = make_test_data()
    f = compile_formula("Rank($close)")
    out = f(df)
    # 横截面秩 -> 每只股票在每天的排名百分位
    by_date = df.assign(_x=df["close"]).groupby("date")["_x"].rank(pct=True)
    np.testing.assert_allclose(out.values, by_date.values, rtol=1e-10, atol=1e-10)


def test_composite_formula():
    """复合公式：动量 + 量价共振 (Alpha101 风格)。"""
    df = make_test_data()
    formula = "Add(Mul(Ts_Mean($close, 5), Sign(Delta($close, 1))), Rank($volume))"
    f = compile_formula(formula)
    out = f(df)
    assert isinstance(out, pd.Series)
    assert len(out) == 90
    assert out.notna().sum() > 0


def test_factor_engine_batch():
    df = make_test_data()
    eng = FactorExpressionEngine()
    eng.register("mom_5", "$close / Delay($close, 5) - 1")
    eng.register("rev_20", "Sub(0, Delta($close, 20))")
    eng.register("cs_rank_close", "Rank($close)")
    result = eng.compute_all(df)
    assert set(result.columns) >= {"code", "date", "mom_5", "rev_20", "cs_rank_close"}
    assert len(result) == 90


def test_evaluate_formula_without_register():
    df = make_test_data()
    eng = FactorExpressionEngine()
    s = eng.evaluate_formula("Abs(Sub($close, Ts_Mean($close, 5)))", df)
    assert isinstance(s, pd.Series)
    assert len(s) == 90


def test_register_custom_operator():
    """验证自定义算子扩展能力。"""
    from factor_expression_engine.operators import register_operator

    def neg_op(df, x):
        return -x

    register_operator("Neg", neg_op, arity=1)
    f = compile_formula("Neg($close)")
    out = f(make_test_data())
    df = make_test_data()
    np.testing.assert_allclose(out.values, (-df["close"]).values)


def test_error_handling():
    import pytest

    with pytest.raises(SyntaxError):
        parse_formula("Ts_Mean($close,")
    with pytest.raises(NameError):
        compile_formula("UnknownOp($close)")(make_test_data())
    with pytest.raises(TypeError):
        compile_formula("Ts_Mean($close, 5, 6)")(make_test_data())


def test_if_operator():
    df = make_test_data()
    f = compile_formula("If(Greater($close, Ts_Mean($close, 5)), 1, 0)")
    out = f(df)
    expected = (df["close"] > df.groupby("code")["close"].transform(
        lambda s: s.rolling(5, min_periods=2).mean())).astype(float)
    np.testing.assert_allclose(out.values, expected.values, equal_nan=True)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
