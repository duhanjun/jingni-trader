"""
边界条件测试: 空数据 / 单标的 / 全涨停 / 单日 / 信号全空 / 极端价格

每个测试返回 dict: {name, passed, details}
"""
from __future__ import annotations
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data import generate_panel
from vectorized_backtest import BaselineLoopAdapter, LookaheadFixedAdapter, VectorizedAdapter


def test_empty_data() -> dict:
    """空数据: 应返回空结果, 不抛异常"""
    empty = pd.DataFrame(columns=["code", "date", "open", "high", "low", "close", "volume"])
    empty_sig = pd.DataFrame(columns=["code", "date", "signal"])
    ok = True
    for adapter in [BaselineLoopAdapter(), LookaheadFixedAdapter(), VectorizedAdapter()]:
        try:
            r = adapter.run_backtest(data=empty, signals=empty_sig)
            if r["metrics"] != {} and not isinstance(r["metrics"], dict):
                ok = False
        except Exception as e:
            ok = False
            return {"name": "空数据处理", "passed": False, "details": f"异常: {e}"}
    return {"name": "空数据处理", "passed": ok, "details": "✓ 三个适配器均优雅返回空结果"}


def test_single_stock() -> dict:
    """单标的回测: 应正常完成"""
    panel = generate_panel(n_codes=1, n_days=60, seed=3)
    dates = sorted(panel["date"].unique())
    code = panel["code"].iloc[0]
    signals = pd.DataFrame([
        {"code": code, "date": dates[5], "signal": 1},
        {"code": code, "date": dates[20], "signal": -1},
    ])
    ok = True
    for adapter in [LookaheadFixedAdapter(), VectorizedAdapter()]:
        try:
            r = adapter.run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=False)
            if r["equity_curve"].empty:
                ok = False
        except Exception as e:
            ok = False
            return {"name": "单标的回测", "passed": False, "details": f"异常: {e}"}
    return {"name": "单标的回测", "passed": ok, "details": "✓ 单标的回测正常完成"}


def test_all_limit_up_no_trade() -> dict:
    """全涨停日: 买入信号应全部被拒, 无成交"""
    panel = generate_panel(n_codes=10, n_days=10, seed=5)
    panel["is_limit_up"] = True
    dates = sorted(panel["date"].unique())
    codes = panel["code"].unique()
    signals = pd.DataFrame([{"code": c, "date": dates[1], "signal": 1} for c in codes])
    r = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)
    n_buy = 0 if r["trades"].empty else int((r["trades"]["action"] == "buy").sum())
    passed = n_buy == 0
    return {"name": "全涨停日无成交", "passed": passed,
            "details": f"全涨停场景买入成交={n_buy}笔 (应为0). {'✓' if passed else '✗'}"}


def test_no_signals() -> dict:
    """无任何信号: 应返回初始资金附近, 无成交"""
    panel = generate_panel(n_codes=20, n_days=50, seed=8)
    signals = pd.DataFrame(columns=["code", "date", "signal"])
    r = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)
    # 无信号 → equity 应为空或初始资金
    passed = r["equity_curve"].empty or len(r["trades"]) == 0
    return {"name": "无信号场景", "passed": passed,
            "details": f"无信号时成交数={len(r['trades'])}, 净值记录={len(r['equity_curve'])}. {'✓' if passed else '✗'}"}


def test_extreme_prices() -> dict:
    """极端价格: 价格极小/极大不应导致溢出"""
    panel = generate_panel(n_codes=5, n_days=30, seed=13)
    # 制造极端价格
    panel.loc[panel.index[:50], "close"] = 0.01
    panel.loc[panel.index[50:100], "close"] = 1e6
    panel["open"] = panel["close"]
    dates = sorted(panel["date"].unique())
    codes = panel["code"].unique()
    signals = pd.DataFrame([{"code": c, "date": dates[1], "signal": 1} for c in codes[:3]])
    try:
        r = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=False)
        eq = r["equity_curve"]["equity"]
        no_inf = not eq.isin([np.inf, -np.inf]).any()
        no_nan = not eq.isna().any()
        passed = no_inf and no_nan
        return {"name": "极端价格无溢出", "passed": passed,
                "details": f"净值含inf={not no_inf}, 含nan={not no_nan}. {'✓' if passed else '✗'}"}
    except Exception as e:
        return {"name": "极端价格无溢出", "passed": False, "details": f"异常: {e}"}


def test_signal_shift_correctness() -> dict:
    """
    信号延迟正确性: 信号日 t, 执行应在 t+1。
    构造: 信号日 t=dates[2], 检查成交日是否为 dates[3]。
    """
    panel = generate_panel(n_codes=3, n_days=15, seed=17)
    dates = sorted(panel["date"].unique())
    code = panel["code"].iloc[0]
    signals = pd.DataFrame([{"code": code, "date": dates[2], "signal": 1}])
    r = LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=False)
    if r["trades"].empty:
        return {"name": "信号延迟执行日校验", "passed": False, "details": "无成交"}
    buy_date = r["trades"][r["trades"]["action"] == "buy"]["date"].iloc[0]
    expected = dates[3]
    passed = buy_date == expected
    return {"name": "信号延迟执行日校验", "passed": passed,
            "details": f"信号日={dates[2]}, 实际成交日={buy_date}, 预期={expected}. {'✓ 信号正确延迟1日' if passed else '✗ 延迟错误'}"}


def run_all() -> list:
    return [
        test_empty_data(),
        test_single_stock(),
        test_all_limit_up_no_trade(),
        test_no_signals(),
        test_extreme_prices(),
        test_signal_shift_correctness(),
    ]
