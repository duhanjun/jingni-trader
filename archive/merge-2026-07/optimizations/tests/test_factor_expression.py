"""
OPTIMIZATION 2 验证：因子表达式引擎
====================================
测试内容：
(a) 正确性：
    - "MA(Close, 5)" vs groupby('code')['close'].rolling(5).mean()
    - "DELTA(Close, 1)" vs close.diff(1) per code
    - "RANK(Close)" vs groupby('date')['close'].rank(pct=True)
    - "RANK(-MA(Close,5))" 复合表达式可运行且语义正确
(b) "Open + Close" 算术
(c) 错误处理：未知函数 / 未知字段 / 非法节点 抛 ValueError

运行：python tests/test_factor_expression.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_generator import generate_test_data
from factor_expression_engine import FactorExpressionEngine


def _max_rel_diff(a: pd.Series, b: pd.Series) -> float:
    """忽略 NaN 后的最大相对差"""
    a = a.astype(float).reset_index(drop=True)
    b = b.astype(float).reset_index(drop=True)
    mask = ~(a.isna() | b.isna())
    if not mask.any():
        return 0.0
    aa, bb = a[mask].to_numpy(), b[mask].to_numpy()
    denom = np.maximum(np.abs(aa), 1e-12)
    return float(np.max(np.abs(aa - bb) / denom))


def test_ma():
    print("\n=== [2a-1] MA(Close, 5) ===")
    data, _ = generate_test_data(n_stocks=20, n_days=120, seed=11)
    eng = FactorExpressionEngine(data)
    got = eng.evaluate("MA(Close, 5)")
    expect = data.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    d = _max_rel_diff(got, expect)
    print(f"  最大相对差: {d:.2e}")
    assert d < 1e-9, f"MA(Close,5) 不一致, rel diff={d}"
    print("  [PASS] MA(Close, 5) 与手动 rolling 一致")


def test_delta():
    print("\n=== [2a-2] DELTA(Close, 1) ===")
    data, _ = generate_test_data(n_stocks=20, n_days=120, seed=12)
    eng = FactorExpressionEngine(data)
    got = eng.evaluate("DELTA(Close, 1)")
    expect = data.groupby("code")["close"].transform(lambda x: x.diff(1))
    d = _max_rel_diff(got, expect)
    print(f"  最大相对差: {d:.2e}")
    assert d < 1e-9, f"DELTA(Close,1) 不一致, rel diff={d}"
    print("  [PASS] DELTA(Close, 1) 与 close.diff(1) 一致")


def test_rank():
    print("\n=== [2a-3] RANK(Close) 横截面排名 ===")
    data, _ = generate_test_data(n_stocks=30, n_days=80, seed=13)
    eng = FactorExpressionEngine(data)
    got = eng.evaluate("RANK(Close)")
    expect = data.groupby("date")["close"].rank(pct=True)
    d = _max_rel_diff(got, expect)
    print(f"  最大相对差: {d:.2e}")
    assert d < 1e-9, f"RANK(Close) 不一致, rel diff={d}"
    print("  [PASS] RANK(Close) 与 groupby(date).rank(pct=True) 一致")


def test_composite_rank_neg_ma():
    print("\n=== [2a-4] RANK(-MA(Close, 5)) 复合表达式 ===")
    data, _ = generate_test_data(n_stocks=30, n_days=80, seed=14)
    eng = FactorExpressionEngine(data)
    got = eng.evaluate("RANK(-MA(Close, 5))")
    # 手动构造期望：先 -MA5 再按 date 横截面 pct rank
    ma5 = data.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    neg_ma5 = -ma5
    expect = neg_ma5.groupby(data["date"]).rank(pct=True)
    d = _max_rel_diff(got, expect)
    print(f"  最大相对差: {d:.2e}")
    assert d < 1e-9, f"RANK(-MA(Close,5)) 不一致, rel diff={d}"
    print("  [PASS] RANK(-MA(Close, 5)) 复合语义正确")


def test_arithmetic():
    print("\n=== [2b] Open + Close 算术 ===")
    data, _ = generate_test_data(n_stocks=10, n_days=60, seed=15)
    eng = FactorExpressionEngine(data)
    got = eng.evaluate("Open + Close")
    expect = data["open"] + data["close"]
    d = _max_rel_diff(got, expect)
    print(f"  最大相对差: {d:.2e}")
    assert d < 1e-12, f"Open+Close 不一致, rel diff={d}"
    # 再测一个稍复杂： (High - Low) / Close
    got2 = eng.evaluate("(High - Low) / Close")
    expect2 = (data["high"] - data["low"]) / data["close"]
    d2 = _max_rel_diff(got2, expect2)
    print(f"  (High-Low)/Close 最大相对差: {d2:.2e}")
    assert d2 < 1e-12
    print("  [PASS] 算术表达式正确")


def test_more_functions():
    print("\n=== [2a-5] 其它算子 (STD/SUM/REF/TS_MAX/TS_MIN/ABS/LOG/CORR/COV) ===")
    data, _ = generate_test_data(n_stocks=8, n_days=60, seed=16)
    eng = FactorExpressionEngine(data)

    # STD
    got = eng.evaluate("STD(Close, 10)")
    exp = data.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=1).std())
    assert _max_rel_diff(got, exp) < 1e-9
    # SUM
    got = eng.evaluate("SUM(Volume, 5)")
    exp = data.groupby("code")["volume"].transform(lambda x: x.rolling(5, min_periods=1).sum())
    assert _max_rel_diff(got, exp) < 1e-9
    # REF
    got = eng.evaluate("REF(Close, 2)")
    exp = data.groupby("code")["close"].shift(2)
    assert _max_rel_diff(got, exp) < 1e-9
    # TS_MAX / TS_MIN
    got = eng.evaluate("TS_MAX(High, 10)")
    exp = data.groupby("code")["high"].transform(lambda x: x.rolling(10, min_periods=1).max())
    assert _max_rel_diff(got, exp) < 1e-9
    got = eng.evaluate("TS_MIN(Low, 10)")
    exp = data.groupby("code")["low"].transform(lambda x: x.rolling(10, min_periods=1).min())
    assert _max_rel_diff(got, exp) < 1e-9
    # ABS / LOG
    got = eng.evaluate("ABS(Low - High)")
    exp = (data["low"] - data["high"]).abs()
    assert _max_rel_diff(got, exp) < 1e-9
    got = eng.evaluate("LOG(Amount)")
    exp = np.log(data["amount"])
    assert _max_rel_diff(got, exp) < 1e-9
    # CORR / COV (双序列)
    got = eng.evaluate("CORR(Close, Volume, 10)")
    # 手动按 code 滚动 corr
    exp = pd.Series(index=data.index, dtype=float)
    for c, idx in data.groupby("code", sort=False).indices.items():
        sub = data.iloc[idx]
        exp.iloc[idx] = sub["close"].rolling(10, min_periods=1).corr(sub["volume"]).values
    assert _max_rel_diff(got, exp) < 1e-9
    got = eng.evaluate("COV(Close, Volume, 10)")
    exp = pd.Series(index=data.index, dtype=float)
    for c, idx in data.groupby("code", sort=False).indices.items():
        sub = data.iloc[idx]
        exp.iloc[idx] = sub["close"].rolling(10, min_periods=1).cov(sub["volume"]).values
    assert _max_rel_diff(got, exp) < 1e-9
    print("  [PASS] STD/SUM/REF/TS_MAX/TS_MIN/ABS/LOG/CORR/COV 全部一致")


def test_error_handling():
    print("\n=== [2c] 错误处理 ===")
    data, _ = generate_test_data(n_stocks=5, n_days=30, seed=17)
    eng = FactorExpressionEngine(data)

    # 未知函数
    try:
        eng.evaluate("FOO(Close, 5)")
        raise AssertionError("未知函数应抛 ValueError")
    except ValueError as e:
        print(f"  未知函数正确抛出: {e}")

    # 未知字段
    try:
        eng.evaluate("Unknown + Close")
        raise AssertionError("未知字段应抛 ValueError")
    except ValueError as e:
        print(f"  未知字段正确抛出: {e}")

    # 非法节点（属性访问，不在白名单）
    try:
        eng.evaluate("Close.__class__")
        raise AssertionError("非法节点应抛 ValueError")
    except ValueError as e:
        print(f"  非法节点正确抛出: {e}")

    # 参数个数错误
    try:
        eng.evaluate("MA(Close)")
        raise AssertionError("参数个数错误应抛 ValueError")
    except ValueError as e:
        print(f"  参数个数错误正确抛出: {e}")

    print("  [PASS] 错误处理全部正确")


def run_all():
    test_ma()
    test_delta()
    test_rank()
    test_composite_rank_neg_ma()
    test_more_functions()
    test_arithmetic()
    test_error_handling()
    print("\n=== 全部因子表达式测试通过 ===")


if __name__ == "__main__":
    run_all()
