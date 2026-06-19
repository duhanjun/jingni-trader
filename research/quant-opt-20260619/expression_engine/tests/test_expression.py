"""表达式引擎单元测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from expression_engine import ExpressionEngine
from expression_engine.engine import build_pivot_panel


def make_panel(n_dates=60, n_codes=8, seed=42):
    """构造测试宽表"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    codes = [f"S{i:04d}" for i in range(n_codes)]
    long = []
    for c in codes:
        # 用带漂移的随机游走构造 close
        rets = rng.normal(0.001, 0.02, n_dates)
        close = 10 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.01, n_dates)))
        low = close * (1 - np.abs(rng.normal(0, 0.01, n_dates)))
        open_ = close * (1 + rng.normal(0, 0.005, n_dates))
        volume = rng.integers(1_000_000, 10_000_000, n_dates)
        amount = close * volume
        for i, d in enumerate(dates):
            long.append({
                "code": c, "date": d, "open": open_[i], "high": high[i],
                "low": low[i], "close": close[i], "volume": volume[i],
                "amount": amount[i],
            })
    df = pd.DataFrame(long)
    return build_pivot_panel(df)


def test_tokenize_basic():
    """基础 tokenizer 测试"""
    from expression_engine.parser import tokenize
    toks = tokenize("Ref($close, 5) / $close - 1")
    types = [t[0] for t in toks]
    assert "FEATURE" in types
    assert "NUM" in types
    assert "OP" in types
    print("test_tokenize_basic PASS")


def test_parse_ref():
    """Ref 解析测试"""
    eng = ExpressionEngine()
    ast = eng.parse("Ref($close, 5)")
    assert ast.__class__.__name__ == "Ref"
    assert ast.window == 5
    print("test_parse_ref PASS")


def test_parse_arithmetic():
    """算术运算解析"""
    eng = ExpressionEngine()
    ast = eng.parse("($close + $open) / 2")
    # 最外层应为 Div
    assert ast.__class__.__name__ == "Div"
    print("test_parse_arithmetic PASS")


def test_parse_corr():
    """三参数函数解析"""
    eng = ExpressionEngine()
    ast = eng.parse("Corr($close, $volume, 20)")
    assert ast.__class__.__name__ == "Corr"
    assert ast.window == 20
    print("test_parse_corr PASS")


def test_parse_unary_minus():
    """一元负号解析"""
    eng = ExpressionEngine()
    ast = eng.parse("-$close")
    # 应被解析为 Mul(Constant(-1), Feature)
    assert ast.__class__.__name__ == "Mul"
    print("test_parse_unary_minus PASS")


def test_evaluate_ref():
    """Ref 求值测试"""
    panel = make_panel()
    eng = ExpressionEngine(panel)
    result = eng.evaluate("Ref($close, 5)")
    # 第 5 行起所有列应非空，第 0~4 行空
    assert result.iloc[5:].notna().any().any()
    assert result.iloc[:5].isna().all().all()
    print("test_evaluate_ref PASS")


def test_evaluate_momentum():
    """5 日动量因子 = close[t]/close[t-5] - 1"""
    panel = make_panel()
    eng = ExpressionEngine(panel)
    expr = "$close / Ref($close, 5) - 1"
    result = eng.evaluate(expr)
    # 手动校验：t=10, code=S0000
    code = panel["close"].columns[0]
    expected = panel["close"][code].iloc[10] / panel["close"][code].iloc[5] - 1
    actual = result[code].iloc[10]
    assert abs(expected - actual) < 1e-9, f"mismatch: {expected} vs {actual}"
    print("test_evaluate_momentum PASS")


def test_evaluate_ma_ratio():
    """20 日均价/最新价"""
    panel = make_panel()
    eng = ExpressionEngine(panel)
    result = eng.evaluate("Mean($close, 20) / $close")
    code = panel["close"].columns[0]
    # 位置 25 时，20 日窗口 = [6..25]，min_periods=10 已满足
    expected = panel["close"][code].iloc[6:26].mean() / panel["close"][code].iloc[25]
    actual = result[code].iloc[25]
    assert abs(expected - actual) < 1e-6, f"mismatch: {expected} vs {actual}"
    print("test_evaluate_ma_ratio PASS")


def test_evaluate_zscore():
    """标准化偏离"""
    panel = make_panel()
    eng = ExpressionEngine(panel)
    expr = "($close - Mean($close, 20)) / Std($close, 20)"
    result = eng.evaluate(expr)
    # 验证 Std 不为零
    assert result.notna().any().any()
    print("test_evaluate_zscore PASS")


def test_evaluate_rank():
    """截面排名"""
    panel = make_panel()
    eng = ExpressionEngine(panel)
    result = eng.evaluate("Rank($volume)")
    # 截面排名 pct=True -> 所有值应在 [0,1]
    valid = result.dropna(how="all")
    assert (valid.values >= 0).all() and (valid.values <= 1).all()
    print("test_evaluate_rank PASS")


def test_evaluate_corr():
    """滚动相关系数"""
    panel = make_panel()
    eng = ExpressionEngine(panel)
    result = eng.evaluate("Corr($close, $volume, 20)")
    assert result.notna().any().any()
    print("test_evaluate_corr PASS")


def test_evaluate_log_abs():
    """Log + Abs 组合"""
    panel = make_panel()
    eng = ExpressionEngine(panel)
    result = eng.evaluate("Log($close)")
    code = panel["close"].columns[0]
    expected = np.log(panel["close"][code].iloc[10])
    actual = result[code].iloc[10]
    assert abs(expected - actual) < 1e-9
    print("test_evaluate_log_abs PASS")


def test_evaluate_if():
    """If 条件运算"""
    panel = make_panel()
    eng = ExpressionEngine(panel)
    expr = "If($close > $open, $volume, 0)"
    result = eng.evaluate(expr)
    code = panel["close"].columns[0]
    for i in range(panel.shape[0]):
        if panel["close"][code].iloc[i] > panel["open"][code].iloc[i]:
            assert result[code].iloc[i] == panel["volume"][code].iloc[i]
        else:
            assert result[code].iloc[i] == 0
    print("test_evaluate_if PASS")


def test_cache():
    """缓存命中测试"""
    panel = make_panel()
    eng = ExpressionEngine(panel, enable_cache=True)
    eng.evaluate("$close / Ref($close, 5)")
    eng.evaluate("$close / Ref($close, 5)")  # 第二次应命中
    s = eng.stats()
    assert s["cache_hit"] == 1
    assert s["cache_miss"] == 1
    print("test_cache PASS")


def test_batch_evaluate():
    """批量评估"""
    panel = make_panel()
    eng = ExpressionEngine(panel)
    df = eng.evaluate_many([
        "Ref($close, 5) / $close - 1",
        "Mean($volume, 10)",
        "Rank($close)",
    ])
    assert df.shape[1] == 3 * panel["close"].shape[1]
    print("test_batch_evaluate PASS")


if __name__ == "__main__":
    test_tokenize_basic()
    test_parse_ref()
    test_parse_arithmetic()
    test_parse_corr()
    test_parse_unary_minus()
    test_evaluate_ref()
    test_evaluate_momentum()
    test_evaluate_ma_ratio()
    test_evaluate_zscore()
    test_evaluate_rank()
    test_evaluate_corr()
    test_evaluate_log_abs()
    test_evaluate_if()
    test_cache()
    test_batch_evaluate()
    print("\nAll expression_engine tests PASSED")
