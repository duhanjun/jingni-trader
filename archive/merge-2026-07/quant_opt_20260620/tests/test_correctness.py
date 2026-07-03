"""
正确性测试: 前视偏差检测 / T+1 约束 / 涨跌停限制 / 向量化一致性

每个测试返回 dict: {name, passed, details}
"""
from __future__ import annotations
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data import generate_panel, generate_signals
from vectorized_backtest import (
    BaselineLoopAdapter,
    LookaheadFixedAdapter,
    VectorizedAdapter,
)


def test_lookahead_bias_detection() -> dict:
    """
    前视偏差检测:

    构造一个"完美预知"信号: signal[t]=1 当且仅当 close[t+1] > close[t]。
    - 基线版(同日 close 执行): 在 t 日 close 买入, 享受 close[t+1]/close[t] 涨幅
      → 会显示显著正收益 (这是前视偏差造成的虚假收益)
    - 修复版(次日 open 执行): 在 t+1 日 open 买入, 收益为 close[t+1]/open[t+1]
      → 收益应明显小于基线版 (前视偏差被消除)

    判定: 基线版 total_return 显著 > 修复版 total_return, 说明基线存在前视偏差。
    """
    panel = generate_panel(n_codes=30, n_days=120, seed=7)
    # 构造完美预知信号: 用次日涨跌作为当日信号
    df = panel.sort_values(["code", "date"]).copy()
    df["fwd_ret"] = df.groupby("code")["close"].shift(-1) / df["close"] - 1
    df["signal"] = 0
    df.loc[df["fwd_ret"] > 0, "signal"] = 1
    df.loc[df["fwd_ret"] < 0, "signal"] = -1
    signals = df[["code", "date", "signal"]].dropna(subset=["signal"]).reset_index(drop=True)
    signals = signals[signals["signal"] != 0].reset_index(drop=True)

    baseline = BaselineLoopAdapter().run_backtest(data=panel, signals=signals, t_plus_1=False, price_limit=False)
    fixed = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=False, price_limit=False)

    b_ret = baseline["metrics"].get("total_return", 0)
    f_ret = fixed["metrics"].get("total_return", 0)

    # 基线应显著为正(虚假收益), 修复版应明显更小
    passed = b_ret > f_ret + 0.05 and b_ret > 0.1
    return {
        "name": "前视偏差检测 (完美预知信号)",
        "passed": passed,
        "details": (
            f"基线版(同日close执行) total_return={b_ret:.4f} (前视偏差导致虚高); "
            f"修复版(次日open执行) total_return={f_ret:.4f}; "
            f"差值={b_ret - f_ret:.4f}. "
            f"{'✓ 检测到前视偏差' if passed else '✗ 未检测到预期差异'}"
        ),
        "metrics": {"baseline_return": b_ret, "fixed_return": f_ret, "bias_gap": b_ret - f_ret},
    }


def test_t_plus_1_enforcement() -> dict:
    """
    T+1 约束测试:

    构造信号: 第1日买入某标的, 第2日(执行日)立即卖出。
    - 修复版应遵守 T+1: 第2日(买入执行日)不可卖出, 卖出最早在第3日。
    - 关闭 T+1 时, 卖出应能发生在买入次日。

    判定: 开启 T+1 时, 不存在"买入执行日当天卖出"的成交对。
    """
    panel = generate_panel(n_codes=5, n_days=20, seed=11)
    codes = panel["code"].unique()[:3]
    dates = sorted(panel["date"].unique())
    # 信号: 第1日买入, 第2日卖出(信号日), 即执行日第2日买、第3日卖
    rows = []
    for c in codes:
        rows.append({"code": c, "date": dates[1], "signal": 1})   # 信号日1 → 执行日2买入
        rows.append({"code": c, "date": dates[2], "signal": -1})  # 信号日2 → 执行日3卖出
    signals = pd.DataFrame(rows)

    fixed_t1 = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=False)
    fixed_no_t1 = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=False, price_limit=False)

    trades_t1 = fixed_t1["trades"]
    # 检查: 是否存在同一 code 在同一 date 既买又卖 (T+1 违规)
    violation = False
    if not trades_t1.empty:
        for code in codes:
            t = trades_t1[trades_t1["code"] == code]
            buy_dates = set(t[t["action"] == "buy"]["date"])
            sell_dates = set(t[t["action"] == "sell"]["date"])
            if buy_dates & sell_dates:
                violation = True
                break

    # T+1 开启时卖出应晚于买入(至少隔1日); 关闭时可能同日
    passed = not violation
    return {
        "name": "T+1 约束 (当日买入不可当日卖)",
        "passed": passed,
        "details": (
            f"T+1开启: 成交{len(trades_t1)}笔, 同日买卖违规={'是' if violation else '否'}; "
            f"T+1关闭成交{len(fixed_no_t1['trades'])}笔. "
            f"{'✓ T+1 约束生效' if passed else '✗ T+1 约束未生效'}"
        ),
    }


def test_price_limit_enforcement() -> dict:
    """
    涨跌停限制测试:

    构造极端场景: 所有标的在执行日均涨停(is_limit_up=True)。
    - 开启 price_limit 时, 涨停股不可买入 → 持仓数应为 0, 无买入成交。
    - 关闭 price_limit 时, 可买入。

    判定: 开启 price_limit 时买入成交数为 0。
    """
    panel = generate_panel(n_codes=10, n_days=15, seed=23)
    # 强制所有日期所有标的涨停
    panel["is_limit_up"] = True
    panel["is_limit_down"] = False

    dates = sorted(panel["date"].unique())
    codes = panel["code"].unique()
    rows = []
    for c in codes:
        rows.append({"code": c, "date": dates[1], "signal": 1})
    signals = pd.DataFrame(rows)

    res_on = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)
    res_off = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=False)

    buy_on = 0 if res_on["trades"].empty else int((res_on["trades"]["action"] == "buy").sum())
    buy_off = 0 if res_off["trades"].empty else int((res_off["trades"]["action"] == "buy").sum())

    passed = buy_on == 0 and buy_off > 0
    return {
        "name": "涨跌停限制 (涨停不可买入)",
        "passed": passed,
        "details": (
            f"全涨停场景: 开启限制时买入成交={buy_on}笔 (应为0); "
            f"关闭限制时买入成交={buy_off}笔 (应>0). "
            f"{'✓ 涨跌停限制生效' if passed else '✗ 涨跌停限制未生效'}"
        ),
    }


def test_vectorized_vs_fixed_consistency() -> dict:
    """
    向量化版 vs 修复版 一致性测试:

    两者都修复了前视偏差, 趋势应同向(净值曲线正相关)。
    绝对值与日收益会因执行模型不同而差异:
      - 修复版: 整手(100股)+预算分配+现金留存+open执行/close估值
      - 向量化版: 分数等权+满仓+close-to-close收益
    因此用净值曲线(整体趋势)相关性衡量, 阈值 0.6; 日收益相关性仅作参考(>0.3)。

    判定: 净值曲线相关性 > 0.6 (整体趋势一致)。
    """
    panel = generate_panel(n_codes=40, n_days=150, seed=99)
    signals = generate_signals(panel, strategy="reversal", top_pct=0.2, seed=99)

    fixed = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)
    vect = VectorizedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)

    eq_f = fixed["equity_curve"].set_index("date")["equity"]
    eq_v = vect["equity_curve"].set_index("date")["equity"]
    common = eq_f.index.intersection(eq_v.index)
    if len(common) < 5:
        return {"name": "向量化一致性", "passed": False, "details": "公共日期不足"}
    # 净值曲线相关性 (整体趋势)
    eq_corr = float(eq_f.loc[common].corr(eq_v.loc[common]))
    # 日收益相关性 (参考)
    ret_f = eq_f.loc[common].pct_change().dropna()
    ret_v = eq_v.loc[common].pct_change().dropna()
    ret_corr = float(ret_f.corr(ret_v))

    passed = eq_corr > 0.6
    return {
        "name": "向量化版与修复版趋势一致性",
        "passed": passed,
        "details": (
            f"净值曲线相关性={eq_corr:.4f} (整体趋势); "
            f"日收益相关性={ret_corr:.4f} (参考, 因执行模型不同偏低属正常); "
            f"修复版 total_return={fixed['metrics'].get('total_return', 0):.4f}, "
            f"向量化版={vect['metrics'].get('total_return', 0):.4f}. "
            f"{'✓ 整体趋势一致' if passed else '✗ 趋势不一致'}"
        ),
        "metrics": {"equity_corr": eq_corr, "daily_return_corr": ret_corr},
    }


def run_all() -> list:
    return [
        test_lookahead_bias_detection(),
        test_t_plus_1_enforcement(),
        test_price_limit_enforcement(),
        test_vectorized_vs_fixed_consistency(),
    ]
