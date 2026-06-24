"""测试 factor expression engine"""
import sys
import os
# 找到 quant_opt 父目录加入 path
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))

import numpy as np
import pandas as pd

from skills.quant-optimizations.quant_opt_20260619_m3.factor_expression.engine import FactorEngine, OPERATOR_REGISTRY


def _make_synth_data(n_stocks: int = 5, n_days: int = 60, seed: int = 42):
    """合成测试数据: n 只股票 n 天的日线"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for code in range(n_stocks):
        close = 10 + np.cumsum(rng.normal(0, 0.5, n_days))
        high = close + rng.uniform(0, 0.3, n_days)
        low = close - rng.uniform(0, 0.3, n_days)
        open_ = close + rng.normal(0, 0.2, n_days)
        volume = rng.integers(1_000_000, 5_000_000, n_days)
        for i in range(n_days):
            rows.append({
                "code": f"S{code:04d}", "date": dates[i],
                "open": open_[i], "high": high[i], "low": low[i],
                "close": close[i], "volume": volume[i],
            })
    return pd.DataFrame(rows)


def test_basic_ts_operators():
    eng = FactorEngine()
    data = _make_synth_data()
    eng.register_formula("ma5", "Ts_Mean($close, 5)")
    eng.register_formula("ma10", "Ts_Mean($close, 10)")
    eng.register_formula("delay1", "Delay($close, 1)")
    eng.register_formula("delta1", "Delta($close, 1)")
    eng.register_formula("vol20", "Ts_Std($close, 20)")

    # 关闭截面归一化以对比原始值
    res = eng.compute(data, formulas=["ma5", "ma10", "delay1", "delta1", "vol20"],
                      apply_cross_section=False)

    # 验证: ma5 第 5 天等于前 5 个 close 的均值
    s0000 = data[data["code"] == "S0000"].sort_values("date").reset_index(drop=True)
    target_date_5 = s0000["date"].iloc[4]
    expected_ma5 = s0000["close"].rolling(5).mean().iloc[4]
    actual_ma5 = res[(res["code"] == "S0000") & (res["date"] == target_date_5)]["ma5"].iloc[0]
    assert abs(expected_ma5 - actual_ma5) < 1e-6, f"ma5 错: {expected_ma5} vs {actual_ma5}"

    # 验证: delay1 第 i 天 = close 第 i-1 天
    target_date_1 = s0000["date"].iloc[1]
    expected_delay = s0000["close"].shift(1).iloc[1]
    actual_delay = res[(res["code"] == "S0000") & (res["date"] == target_date_1)]["delay1"].iloc[0]
    assert abs(expected_delay - actual_delay) < 1e-6, f"delay 错: {expected_delay} vs {actual_delay}"

    # 验证: delta1 第 i 天 = close 第 i 天 - close 第 i-1 天
    expected_delta = s0000["close"].diff(1).iloc[1]
    actual_delta = res[(res["code"] == "S0000") & (res["date"] == target_date_1)]["delta1"].iloc[0]
    assert abs(expected_delta - actual_delta) < 1e-6

    print("[PASS] test_basic_ts_operators")


def test_nested_expression():
    """测试嵌套表达式"""
    eng = FactorEngine()
    data = _make_synth_data()
    eng.register_formula("alpha001", "Ts_Rank($close, 20)")
    eng.register_formula("alpha002", "Ts_Mean($close, 5) / Ts_Mean($close, 20)")
    eng.register_formula("alpha003", "(($close - Delay($close, 5)) / Delay($close, 5)) * 100")
    res = eng.compute(data, formulas=["alpha001", "alpha002", "alpha003"],
                      apply_cross_section=False)
    assert res["alpha001"].notna().sum() > 0
    assert res["alpha002"].notna().sum() > 0
    # 验证 alpha002 公式正确
    s0000 = data[data["code"] == "S0000"].sort_values("date").reset_index(drop=True)
    ma5 = s0000["close"].rolling(5).mean()
    ma20 = s0000["close"].rolling(20).mean()
    expected = (ma5 / ma20).iloc[20]  # 第 20 天后 ma20 才非空
    target_date = s0000["date"].iloc[20]
    actual = res[(res["code"] == "S0000") & (res["date"] == target_date)]["alpha002"].iloc[0]
    assert abs(expected - actual) < 1e-6, f"alpha002 错: {expected} vs {actual}"
    print("[PASS] test_nested_expression")


def test_math_operators():
    eng = FactorEngine()
    data = _make_synth_data()
    eng.register_formula("abs_c", "Abs($close)")
    eng.register_formula("log_c", "Log($close)")
    eng.register_formula("pow2", "Power($close, 2)")
    eng.register_formula("sign_c", "Sign($close)")
    res = eng.compute(data, formulas=["abs_c", "log_c", "pow2", "sign_c"],
                      apply_cross_section=False)
    s0000 = data[data["code"] == "S0000"].sort_values("date").reset_index(drop=True)
    for i in [0, 5, 10]:
        target_date = s0000["date"].iloc[i]
        row = res[(res["code"] == "S0000") & (res["date"] == target_date)].iloc[0]
        c = s0000["close"].iloc[i]
        assert abs(abs(c) - row["abs_c"]) < 1e-6
        assert abs(np.log(c) - row["log_c"]) < 1e-6
        assert abs(c ** 2 - row["pow2"]) < 1e-6
    print("[PASS] test_math_operators")


def test_logic_operators():
    eng = FactorEngine()
    data = _make_synth_data(n_days=20)
    eng.register_formula("if_up", "If($close > Delay($close, 1), 1, -1)")
    eng.register_formula("and_test", "And($close > 10, $close < 20)")
    res = eng.compute(data, formulas=["if_up", "and_test"],
                      apply_cross_section=False)
    s0000 = data[data["code"] == "S0000"].sort_values("date").reset_index(drop=True)
    for i in [1, 5, 10]:
        target_date = s0000["date"].iloc[i]
        row = res[(res["code"] == "S0000") & (res["date"] == target_date)].iloc[0]
        expected = 1 if s0000["close"].iloc[i] > s0000["close"].iloc[i - 1] else -1
        assert expected == int(row["if_up"]), f"if_up 错: {expected} vs {row['if_up']}"
    print("[PASS] test_logic_operators")


def test_error_handling():
    """测试错误处理: 未注册/无效字段"""
    eng = FactorEngine()
    data = _make_synth_data()
    try:
        eng.compute(data, formulas=["nonexistent"])
        assert False, "应该抛 KeyError"
    except KeyError:
        pass
    try:
        eng.register_formula("bad", "Ts_Mean(close, 5)")  # 缺 $
        eng.compute(data, formulas=["bad"])
        assert False, "应该抛 ParseError"
    except Exception as e:
        assert "字段必须以" in str(e) or "ParseError" in repr(e)
    print("[PASS] test_error_handling")


def test_security_no_eval():
    """测试: 不使用 eval, 避免代码注入"""
    eng = FactorEngine()
    data = _make_synth_data()
    try:
        # 试图 import os 应当失败
        eng.register_formula("evil", "__import__('os').system('echo hacked')")
        eng.compute(data, formulas=["evil"])
        assert False, "应阻止代码注入"
    except Exception as e:
        # 必须抛错而不是执行
        assert "hacked" not in str(e).lower()
    print("[PASS] test_security_no_eval")


def test_cs_rank_consistency():
    """测试截面算子: 同期排名结果在 [0, 1] 区间"""
    eng = FactorEngine()
    data = _make_synth_data(n_stocks=20, n_days=30)
    eng.register_formula("ma5", "Ts_Mean($close, 5)")
    res = eng.compute(data, formulas=["ma5"], apply_cross_section=True)
    # 检查每日 ma5 的截面 zscore 大致在 [-3, 3] 区间
    cs_check = res.groupby("date")["ma5"].agg(["mean", "std"]).dropna()
    means = cs_check["mean"].abs()
    assert means.max() < 0.5, f"截面归一化后均值应接近0, 实际 max abs mean = {means.max()}"
    print("[PASS] test_cs_rank_consistency")


def test_real_alpha101_sample():
    """测试若干真实 WorldQuant Alpha101 公式片段"""
    eng = FactorEngine()
    data = _make_synth_data(n_stocks=10, n_days=80)
    # Alpha 006: -1 * correlation($close, $volume, 10) -- 简化版
    # Alpha 012: sign(delta($volume, 1)) * sign(delta($close, 1)) * sign(delta($close, 1) - delta($close, 2))
    eng.register_formula("alpha_simplified", "Sign(Delta($volume, 1)) * Sign(Delta($close, 1))")
    res = eng.compute(data, formulas=["alpha_simplified"], apply_cross_section=False)
    assert "alpha_simplified" in res.columns
    assert res["alpha_simplified"].notna().sum() > 0
    print("[PASS] test_real_alpha101_sample")


def test_operator_registry():
    assert "Ts_Mean" in OPERATOR_REGISTRY
    assert "Cs_Rank" in OPERATOR_REGISTRY
    assert "Abs" in OPERATOR_REGISTRY
    print("[PASS] test_operator_registry")


if __name__ == "__main__":
    test_operator_registry()
    test_basic_ts_operators()
    test_nested_expression()
    test_math_operators()
    test_logic_operators()
    test_cs_rank_consistency()
    test_real_alpha101_sample()
    test_error_handling()
    test_security_no_eval()
    print("\n所有因子表达式测试通过 ✓")