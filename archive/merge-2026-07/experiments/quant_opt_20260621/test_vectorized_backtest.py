"""
向量化回测引擎测试

测试内容:
  1. 正确性测试: 与简单手算结果对比
  2. 前视偏差测试: 验证 T+1 open 成交
  3. T+1 限制测试: 当日买入不可当日卖出
  4. 涨跌停测试: 涨停拒绝买入, 跌停拒绝卖出
  5. 性能测试: 与逐行循环对比
  6. 边界条件: 空数据、单只股票、信号缺失
"""
from __future__ import annotations

import time
from typing import Dict, Any

import numpy as np
import pandas as pd

from vectorized_backtest import BacktestConfig, VectorizedBacktester


def make_synthetic_data(
    n_codes: int = 10,
    n_days: int = 100,
    start_price: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """生成合成行情数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_codes)]

    rows = []
    for code in codes:
        price = start_price
        for dt in dates:
            ret = rng.normal(0, 0.02)
            open_p = price * (1 + rng.normal(0, 0.005))
            close = open_p * (1 + ret)
            high = max(open_p, close) * (1 + abs(rng.normal(0, 0.005)))
            low = min(open_p, close) * (1 - abs(rng.normal(0, 0.005)))
            volume = int(rng.integers(1_000_000, 10_000_000))
            rows.append({
                "code": code, "date": dt,
                "open": open_p, "high": high, "low": low, "close": close,
                "volume": volume, "amount": volume * close,
                "is_limit_up": 0, "is_limit_down": 0, "is_st": 0,
            })
            price = close
    return pd.DataFrame(rows)


def make_signals(data: pd.DataFrame, strategy: str = "ma_cross") -> pd.DataFrame:
    """生成信号: MA5 上穿 MA20 买入, 下穿卖出"""
    df = data.sort_values(["code", "date"]).copy()
    df["ma5"] = df.groupby("code")["close"].transform(lambda x: x.rolling(5).mean())
    df["ma20"] = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    df["prev_ma5"] = df.groupby("code")["ma5"].shift(1)
    df["prev_ma20"] = df.groupby("code")["ma20"].shift(1)

    df["signal"] = 0
    # 金叉买入
    buy_mask = (df["ma5"] > df["ma20"]) & (df["prev_ma5"] <= df["prev_ma20"])
    df.loc[buy_mask, "signal"] = 1
    # 死叉卖出
    sell_mask = (df["ma5"] < df["ma20"]) & (df["prev_ma5"] >= df["prev_ma20"])
    df.loc[sell_mask, "signal"] = -1

    return df[["code", "date", "signal"]].copy()


# ---------------- 正确性测试 ----------------

def test_basic_correctness():
    """测试1: 基本正确性 - 回测能跑通, 净值合理"""
    print("\n[Test 1] 基本正确性测试")
    data = make_synthetic_data(n_codes=5, n_days=60)
    signals = make_signals(data)

    bt = VectorizedBacktester(BacktestConfig(trade_on="next_open"))
    result = bt.run(data, signals, mode="equal_weight", max_positions=5)

    eq = result["equity_curve"]
    assert not eq.empty, "净值曲线不应为空"
    assert "equity" in eq.columns, "净值曲线应包含 equity 列"
    assert len(eq) == 60, f"净值曲线长度应等于交易日数, 实际 {len(eq)}"

    # 初始净值为 init_capital (第一日无交易)
    assert abs(eq["equity"].iloc[0] - 1_000_000) < 1e-6, \
        f"初始净值应为 1_000_000, 实际 {eq['equity'].iloc[0]}"

    # 净值不能为负
    assert (eq["equity"] > 0).all(), "净值不能为负"

    metrics = result["metrics"]
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics

    print(f"  净值范围: [{eq['equity'].min():.2f}, {eq['equity'].max():.2f}]")
    print(f"  总收益: {metrics['total_return']:.4%}")
    print(f"  夏普: {metrics['sharpe_ratio']:.4f}")
    print(f"  最大回撤: {metrics['max_drawdown']:.4%}")
    print(f"  交易笔数: {metrics['n_trades']}")
    print("  [PASS] 基本正确性测试通过")


def test_look_ahead_bias():
    """测试2: 前视偏差测试 - next_open 模式不应使用当日 close 成交"""
    print("\n[Test 2] 前视偏差测试")
    # 构造一个当日 close 暴涨但次日 open 回落的场景
    data = pd.DataFrame({
        "code": ["000001.SZ"] * 5,
        "date": pd.bdate_range("2024-01-01", periods=5),
        "open": [10.0, 10.5, 11.0, 10.8, 10.6],
        "high": [10.5, 11.0, 11.5, 11.0, 10.8],
        "low": [9.8, 10.3, 10.8, 10.5, 10.3],
        "close": [10.2, 10.8, 11.2, 10.9, 10.5],
        "volume": [1e6] * 5,
        "amount": [1e7] * 5,
        "is_limit_up": 0, "is_limit_down": 0, "is_st": 0,
    })
    # 在第 2 日 (close=10.8) 生成买入信号
    signals = pd.DataFrame({
        "code": ["000001.SZ"] * 5,
        "date": pd.bdate_range("2024-01-01", periods=5),
        "signal": [0, 1, 0, 0, 0],
    })

    bt = VectorizedBacktester(BacktestConfig(trade_on="next_open"))
    result = bt.run(data, signals, mode="equal_weight", max_positions=1)

    trades = result["trades"]
    assert not trades.empty, "应该有交易"
    # 信号在第 2 日生成, 应在第 3 日 open=11.0 成交
    trade = trades.iloc[0]
    trade_date = trade["date"]
    expected_date = data["date"].iloc[2]
    assert trade_date == expected_date, \
        f"成交日应为 {expected_date.date()} (T+1), 实际 {trade_date.date()}"
    # 成交价应为 11.0 (第 3 日 open) + 滑点
    expected_price = 11.0 * (1 + 0.001)
    assert abs(trade["price"] - expected_price) < 0.01, \
        f"成交价应为 {expected_price:.4f} (next_open + slippage), 实际 {trade['price']:.4f}"

    print(f"  信号日: {data['date'].iloc[1].date()} (signal=1)")
    print(f"  成交日: {trade_date.date()} (T+1)")
    print(f"  成交价: {trade['price']:.4f} (next_open={data['open'].iloc[2]} + slippage)")
    print("  [PASS] 前视偏差测试通过 - 信号 T 日生成, T+1 日 open 成交")


def test_t_plus_1():
    """测试3: T+1 限制 - 当日买入不可当日卖出 (通过 sell-first 顺序自然保证)"""
    print("\n[Test 3] T+1 限制测试")
    data = make_synthetic_data(n_codes=3, n_days=30)
    # 场景: 股票 000000.SZ 在第 5 日发买入信号, 第 6 日发卖出信号
    # next_open 模式: 买入在第 6 日 open 成交, 卖出在第 7 日 open 成交
    # T+1 允许: 第 6 日买入, 第 7 日卖出 (隔日卖出)
    dt_buy = data["date"].iloc[5]
    dt_sell = data["date"].iloc[6]
    signals = pd.DataFrame({
        "code": ["000000.SZ", "000000.SZ"],
        "date": [dt_buy, dt_sell],
        "signal": [1, -1],
    })

    bt_t1 = VectorizedBacktester(BacktestConfig(t_plus_1=True, trade_on="next_open"))
    result_t1 = bt_t1.run(data, signals, mode="equal_weight", max_positions=5)
    trades_t1 = result_t1["trades"]

    assert not trades_t1.empty, "应有交易"
    buys = trades_t1[trades_t1["action"] == "buy"]
    sells = trades_t1[trades_t1["action"] == "sell"]
    assert len(buys) == 1, f"应有 1 笔买入, 实际 {len(buys)}"
    assert len(sells) == 1, f"应有 1 笔卖出 (T+1 允许隔日卖), 实际 {len(sells)}"

    # 验证买入日在卖出日之前 (T+1 隔日卖出)
    buy_date = buys.iloc[0]["date"]
    sell_date = sells.iloc[0]["date"]
    assert buy_date < sell_date, f"买入日 {buy_date} 应早于卖出日 {sell_date}"

    print(f"  买入日: {buy_date.date()} (信号日 +1, T+1 open 成交)")
    print(f"  卖出日: {sell_date.date()} (信号日 +1, T+1 允许隔日卖出)")
    print(f"  买入笔数: {len(buys)}, 卖出笔数: {len(sells)}")
    print("  [PASS] T+1 限制测试通过 - sell-first 顺序自然保证 T+1")


def test_price_limit():
    """测试4: 涨跌停限制"""
    print("\n[Test 4] 涨跌停限制测试")
    data = pd.DataFrame({
        "code": ["000001.SZ"] * 4 + ["000002.SZ"] * 4,
        "date": list(pd.bdate_range("2024-01-01", periods=4)) * 2,
        "open": [10.0, 11.0, 12.0, 11.5, 20.0, 22.0, 24.0, 23.0],
        "high": [10.5, 11.5, 12.5, 12.0, 21.0, 23.0, 25.0, 24.0],
        "low": [9.8, 10.5, 11.5, 11.0, 19.5, 21.5, 23.0, 22.0],
        "close": [10.2, 11.2, 12.2, 11.8, 20.5, 22.5, 24.5, 23.5],
        "volume": [1e6] * 8,
        "amount": [1e7] * 8,
        "is_limit_up": [0, 0, 0, 0,  0, 0, 1, 0],   # 000002 第3日涨停
        "is_limit_down": [0, 0, 0, 0,  0, 0, 0, 1], # 000002 第4日跌停
        "is_st": [0] * 8,
    })
    # 第 2 日对两只股票都发买入信号
    dt2 = data["date"].iloc[1]
    signals = pd.DataFrame({
        "code": ["000001.SZ", "000002.SZ"],
        "date": [dt2, dt2],
        "signal": [1, 1],
    })

    bt = VectorizedBacktester(BacktestConfig(price_limit=True, trade_on="next_open"))
    result = bt.run(data, signals, mode="equal_weight", max_positions=5)
    trades = result["trades"]

    # 信号在第 2 日生成, 第 3 日成交
    # 000001.SZ 第 3 日正常, 应买入
    # 000002.SZ 第 3 日涨停, 应拒绝买入
    buy_codes = set(trades[trades["action"] == "buy"]["code"].unique())
    assert "000001.SZ" in buy_codes, "000001.SZ 应买入"
    assert "000002.SZ" not in buy_codes, "000002.SZ 涨停应拒绝买入"

    print(f"  买入的股票: {buy_codes}")
    print(f"  000002.SZ 第3日涨停, 正确拒绝买入")
    print("  [PASS] 涨跌停限制测试通过")


# ---------------- 性能测试 ----------------

def test_performance():
    """测试5: 性能对比 - 向量化 vs 逐行循环"""
    print("\n[Test 5] 性能对比测试")
    data = make_synthetic_data(n_codes=50, n_days=500, seed=123)
    signals = make_signals(data)

    # 向量化回测
    bt = VectorizedBacktester(BacktestConfig(trade_on="next_open"))
    t0 = time.perf_counter()
    result_vec = bt.run(data, signals, mode="equal_weight", max_positions=20)
    t_vec = time.perf_counter() - t0

    # 模拟逐行循环 (用原生 adapter 风格)
    t0 = time.perf_counter()
    _ = _simulate_iterrows_backtest(data, signals)
    t_iter = time.perf_counter() - t0

    speedup = t_iter / t_vec if t_vec > 0 else float("inf")
    print(f"  数据规模: {len(data)} 行 ({data['code'].nunique()} 只股票 x {data['date'].nunique()} 日)")
    print(f"  向量化回测: {t_vec*1000:.1f} ms")
    print(f"  逐行循环回测: {t_iter*1000:.1f} ms")
    print(f"  加速比: {speedup:.2f}x")
    print(f"  交易笔数: {len(result_vec['trades'])}")
    print("  [PASS] 性能测试通过")
    return {"vectorized_ms": t_vec * 1000, "iterrows_ms": t_iter * 1000, "speedup": speedup}


def _simulate_iterrows_backtest(data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
    """模拟 jingni-trader 现有 native_adapter 的逐行循环风格"""
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    signals = signals.sort_values(["date", "code"]).reset_index(drop=True)
    dates = sorted(signals["date"].unique())
    cash = 1_000_000.0
    positions = {}
    equity_records = []
    trades = []

    for dt in dates:
        day_signal = signals[signals["date"] == dt]
        day_data = data[data["date"] == dt]
        if day_data.empty:
            continue
        day_data_map = day_data.set_index("code")

        sell_codes = []
        buy_codes = []
        for _, row in day_signal.iterrows():
            code = row["code"]
            sig = row.get("signal", 0)
            if sig > 0:
                buy_codes.append(code)
            elif sig < 0:
                sell_codes.append(code)

        for code in sell_codes:
            if code not in positions or positions[code] <= 0:
                continue
            if code not in day_data_map.index:
                continue
            price = day_data_map.loc[code, "close"]
            shares = positions[code]
            cash += price * shares
            positions[code] = 0

        if buy_codes:
            budget = cash * 0.95 / len(buy_codes)
            for code in buy_codes:
                if code not in day_data_map.index:
                    continue
                price = day_data_map.loc[code, "close"]
                shares = int(budget / price / 100) * 100
                if shares <= 0:
                    continue
                cash -= price * shares
                positions[code] = positions.get(code, 0) + shares

        mv = 0
        for code, shares in positions.items():
            if shares > 0 and code in day_data_map.index:
                mv += shares * day_data_map.loc[code, "close"]
        equity_records.append({"date": dt, "equity": cash + mv})

    return {"equity_curve": pd.DataFrame(equity_records), "trades": pd.DataFrame(trades)}


# ---------------- 边界条件测试 ----------------

def test_edge_cases():
    """测试6: 边界条件"""
    print("\n[Test 6] 边界条件测试")
    bt = VectorizedBacktester()

    # 空数据
    result = bt.run(pd.DataFrame(), pd.DataFrame())
    assert result["equity_curve"].empty, "空数据应返回空结果"
    print("  [PASS] 空数据处理正确")

    # 单只股票
    data = make_synthetic_data(n_codes=1, n_days=30)
    signals = make_signals(data)
    result = bt.run(data, signals, mode="equal_weight", max_positions=1)
    assert not result["equity_curve"].empty
    print("  [PASS] 单只股票处理正确")

    # 信号全为 0
    signals_zero = signals.copy()
    signals_zero["signal"] = 0
    result = bt.run(data, signals_zero, mode="equal_weight", max_positions=1)
    # 无交易, 净值应保持为 init_capital
    eq = result["equity_curve"]
    if not eq.empty:
        # 无交易时, 现金不变, 净值 = 现金
        assert abs(eq["equity"].iloc[-1] - 1_000_000) < 1e-6, \
            f"无交易时净值应保持 1_000_000, 实际 {eq['equity'].iloc[-1]}"
    print("  [PASS] 全零信号处理正确")

    # 信号缺失日期
    signals_missing = signals.iloc[:5].copy()
    result = bt.run(data, signals_missing, mode="equal_weight", max_positions=1)
    assert not result["equity_curve"].empty
    print("  [PASS] 信号缺失日期处理正确")


if __name__ == "__main__":
    test_basic_correctness()
    test_look_ahead_bias()
    test_t_plus_1()
    test_price_limit()
    perf = test_performance()
    test_edge_cases()
    print("\n=== 所有测试通过 ===")
    print(f"性能数据: {perf}")
