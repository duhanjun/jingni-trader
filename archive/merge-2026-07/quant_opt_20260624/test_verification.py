"""
验证测试套件 (feat/quant-opt-20260624)

测试内容:
  1. 正确性测试: 优化回测 vs 参考实现、因子 DSL vs 手工计算、T+1 标签
  2. 性能对比测试: 预分组 vs 重复过滤、向量化因子 vs 逐股循环、缓存命中
  3. 边界条件测试: 空数据、单只股票、停牌、涨跌停、T+1 约束
  4. 风险修复测试: HRP 修复、CVaR 修复

运行: python3 quant_opt_20260624/test_verification.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# 确保能导入优化模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimized_backtest import (
    OptimizedBacktestEngine, Position, PriceLimitValidator,
    MaxPositionValidator, OrderRequest,
)
from factor_expression_dsl import (
    FactorExpressionEngine, ExpressionParser, t_plus_1_label, naive_label,
)
from risk_fixes import RiskFixes


# ---------------------------------------------------------------------------
# 测试数据生成
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n_stocks: int = 50,
    n_days: int = 250,
    start_price: float = 20.0,
    seed: int = 42,
    include_benchmark: bool = True,
) -> pd.DataFrame:
    """生成 A 股日线模拟数据(含基准)"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=n_days)
    rows = []
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    if include_benchmark:
        codes.append("000300.SH")  # 基准

    for code in codes:
        is_bench = code == "000300.SH"
        price = start_price * (1.0 if is_bench else rng.uniform(0.5, 2.0))
        drift = 0.0003 if is_bench else rng.uniform(-0.0008, 0.0012)
        vol = 0.008 if is_bench else rng.uniform(0.012, 0.025)

        rets = rng.normal(drift, vol, n_days)
        rets[0] = 0
        prices = price * np.cumprod(1 + rets)

        for i in range(n_days):
            close = round(float(prices[i]), 4)
            open_ = round(close * (1 + rng.normal(0, 0.003)), 4)
            high = round(max(open_, close) * (1 + abs(rng.normal(0, 0.004))), 4)
            low = round(min(open_, close) * (1 - abs(rng.normal(0, 0.004))), 4)
            volume = int(rng.lognormal(12, 0.4))
            change_pct = 0.0 if i == 0 else (close - prev_close) / prev_close * 100
            rows.append({
                'date': dates[i], 'code': code,
                'open': open_, 'high': high, 'low': low, 'close': close,
                'volume': volume, 'amount': volume * close,
                'change_pct': change_pct,
                'is_st': False,
                'is_limit_up': change_pct >= 9.9,
                'is_limit_down': change_pct <= -9.9,
                'turnover_rate': round(float(rng.uniform(0.5, 5.0)), 4),
            })
            prev_close = close

    df = pd.DataFrame(rows)
    return df.sort_values(['date', 'code']).reset_index(drop=True)


def make_signals(data: pd.DataFrame, top_pct: float = 0.2) -> pd.DataFrame:
    """基于动量生成买卖信号(每月调仓)"""
    df = data[data['code'] != '000300.SH'].copy()
    df = df.sort_values(['code', 'date'])
    df['mom_20'] = df.groupby('code')['close'].transform(lambda x: x.shift(20) / x.shift(25) - 1)

    # 每月第一个交易日调仓
    df['ym'] = df['date'].dt.to_period('M')
    monthly_first = df.groupby(['code', 'ym']).head(1).copy()
    monthly_first['rank'] = monthly_first.groupby('date')['mom_20'].rank(pct=True)

    signals = monthly_first[['code', 'date']].copy()
    signals['signal'] = 0
    signals.loc[monthly_first['rank'] >= (1 - top_pct), 'signal'] = 1
    signals.loc[monthly_first['rank'] <= top_pct, 'signal'] = -1
    return signals.sort_values(['date', 'code']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 参考回测实现(独立,用于正确性校验)
# ---------------------------------------------------------------------------

def reference_backtest(
    data: pd.DataFrame, signals: pd.DataFrame, init_capital: float = 1e6,
    commission_rate: float = 0.00025, stamp_tax_rate: float = 0.001,
    slippage: float = 0.001,
) -> Dict[str, Any]:
    """独立参考回测(显式 T+1 + 成本基准),用于校验优化版"""
    data = data.sort_values(['date', 'code']).reset_index(drop=True)
    signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

    cash = init_capital
    positions: Dict[str, Dict] = {}  # code -> {shares, today_bought, cost_basis}
    equity_records = []
    trades = []

    dates = sorted(signals['date'].unique())
    for dt in dates:
        day_signal = signals[signals['date'] == dt]
        day_data = data[data['date'] == dt]
        if day_data.empty:
            continue
        day_map = day_data.set_index('code')

        # 卖出
        for _, row in day_signal.iterrows():
            if row.get('signal', 0) >= 0:
                continue
            code = row['code']
            pos = positions.get(code)
            if not pos or pos['shares'] <= 0:
                continue
            closable = pos['shares'] - pos['today_bought']
            if closable <= 0:
                continue
            if code not in day_map.index:
                continue
            prow = day_map.loc[code]
            if prow.get('is_limit_down', False):
                continue
            price = float(prow['close']) * (1 - slippage)
            shares = closable
            amt = price * shares
            comm = max(amt * commission_rate, 5)
            tax = amt * stamp_tax_rate
            cash += amt - comm - tax
            pnl = amt - comm - tax - pos['cost_basis'] * shares
            trades.append({'date': dt, 'code': code, 'action': 'sell',
                           'price': price, 'shares': shares, 'amount': amt,
                           'pnl': pnl})
            pos['shares'] -= shares

        # 买入
        buy_codes = [r['code'] for _, r in day_signal.iterrows() if r.get('signal', 0) > 0]
        buy_codes = [c for c in buy_codes if c in day_map.index
                     and not day_map.loc[c].get('is_limit_up', False)]
        if buy_codes:
            budget = cash * 0.95 / len(buy_codes)
            for code in buy_codes:
                prow = day_map.loc[code]
                price = float(prow['close']) * (1 + slippage)
                shares = int(budget / price / 100) * 100
                if shares <= 0:
                    continue
                amt = price * shares
                comm = max(amt * commission_rate, 5)
                cost = amt + comm
                if cost > cash:
                    shares = int((cash * 0.98) / price / 100) * 100
                    if shares <= 0:
                        continue
                    amt = price * shares
                    comm = max(amt * commission_rate, 5)
                    cost = amt + comm
                cash -= cost
                pos = positions.setdefault(code, {'shares': 0, 'today_bought': 0, 'cost_basis': 0.0})
                total_cost = pos['cost_basis'] * pos['shares'] + price * shares
                pos['shares'] += shares
                pos['today_bought'] += shares
                pos['cost_basis'] = total_cost / pos['shares'] if pos['shares'] > 0 else 0
                trades.append({'date': dt, 'code': code, 'action': 'buy',
                               'price': price, 'shares': shares, 'amount': amt, 'pnl': -comm})

        # 估值
        mv = 0
        for code, pos in positions.items():
            if pos['shares'] <= 0:
                continue
            if code in day_map.index:
                mv += pos['shares'] * float(day_map.loc[code, 'close'])
        equity_records.append({'date': dt, 'equity': cash + mv})

        # T+1 结算
        for pos in positions.values():
            pos['today_bought'] = 0

    eq = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trades)
    return {'equity_curve': eq, 'trades': trades_df}


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestRunner:
    def __init__(self):
        self.results: List[Dict] = []
        self.passed = 0
        self.failed = 0

    def assert_true(self, name: str, cond: bool, detail: str = ""):
        status = "PASS" if cond else "FAIL"
        if cond:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append({"name": name, "status": status, "detail": detail})
        print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))

    def assert_close(self, name: str, a: float, b: float, tol: float = 1e-6, detail: str = ""):
        cond = abs(a - b) <= tol
        d = f"{a} vs {b} (tol={tol})" + (f" {detail}" if detail else "")
        self.assert_true(name, cond, d)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"测试汇总: {self.passed}/{total} 通过, {self.failed} 失败")
        print(f"{'='*60}")
        return self.failed == 0


def test_backtest_correctness(t: TestRunner):
    """测试1: 回测正确性 — 优化版 vs 独立参考实现"""
    print("\n=== 测试1: 回测正确性 (优化版 vs 参考实现) ===")
    data = make_synthetic_data(n_stocks=30, n_days=120)
    signals = make_signals(data)

    opt = OptimizedBacktestEngine()
    opt_result = opt.run_backtest(data, signals, init_capital=1e6)
    ref_result = reference_backtest(data, signals, init_capital=1e6)

    # 终值应接近(允许微小差异来自浮点)
    opt_final = float(opt_result['equity_curve']['equity'].iloc[-1])
    ref_final = float(ref_result['equity_curve']['equity'].iloc[-1])
    t.assert_true("回测终值与参考实现一致",
                  abs(opt_final - ref_final) / ref_final < 1e-4 if ref_final != 0 else abs(opt_final - ref_final) < 1,
                  f"优化版={opt_final:.2f}, 参考实现={ref_final:.2f}")

    # 交易笔数一致
    opt_trades = len(opt_result['trades'])
    ref_trades = len(ref_result['trades'])
    t.assert_true("交易笔数一致", opt_trades == ref_trades,
                  f"优化版={opt_trades}, 参考={ref_trades}")

    # 净值曲线长度一致
    t.assert_true("净值曲线长度一致",
                  len(opt_result['equity_curve']) == len(ref_result['equity_curve']),
                  f"优化版={len(opt_result['equity_curve'])}, 参考={len(ref_result['equity_curve'])}")

    # PnL 不再是错误的 sell_amount-cost
    trades_df = opt_result['trades']
    if not trades_df.empty and 'action' in trades_df.columns:
        sell_trades = trades_df[trades_df['action'] == 'sell']
        if len(sell_trades) > 0:
            # 真实 PnL 应基于 cost_basis,可为负
            has_cost_basis = 'cost_basis' in sell_trades.columns
            t.assert_true("卖出交易含 cost_basis 字段", has_cost_basis)
            # PnL = amount - commission - tax - cost_basis*shares,应与记录一致
            if has_cost_basis and not sell_trades.empty:
                row = sell_trades.iloc[0]
                expected_pnl = row['amount'] - row['commission'] - row['tax'] - row['cost_basis'] * row['shares']
                t.assert_close("卖出 PnL 计算正确(基于成本基准)",
                               float(row['pnl']), float(expected_pnl), tol=1e-4)
    else:
        t.assert_true("回测产生交易记录", False, "trades 为空")


def test_backtest_performance(t: TestRunner):
    """测试2: 性能对比 — 预分组 vs 重复过滤"""
    print("\n=== 测试2: 回测性能 (预分组 vs 重复过滤) ===")
    data = make_synthetic_data(n_stocks=200, n_days=500)
    # 用每日信号(而非月度),最大化循环次数以放大重复过滤开销
    signals = make_signals_daily(data)

    opt = OptimizedBacktestEngine()

    # 优化版(预分组)
    t0 = time.time()
    opt.run_backtest(data, signals)
    opt_time = time.time() - t0

    # 模拟原实现(重复过滤) — 同样的逻辑但每次循环内 boolean filter
    t0 = time.time()
    _run_naive_filter(data, signals)
    naive_time = time.time() - t0

    speedup = naive_time / opt_time if opt_time > 0 else float('inf')
    t.assert_true("优化版快于重复过滤版", opt_time < naive_time,
                  f"优化版={opt_time:.3f}s, 重复过滤={naive_time:.3f}s, 加速比={speedup:.2f}x")
    t.assert_true("加速比 >= 1.5x", speedup >= 1.5,
                  f"加速比={speedup:.2f}x")


def make_signals_daily(data: pd.DataFrame, top_pct: float = 0.2) -> pd.DataFrame:
    """每日调仓信号(用于性能测试,最大化循环次数)"""
    df = data[data['code'] != '000300.SH'].copy()
    df = df.sort_values(['code', 'date'])
    df['mom_5'] = df.groupby('code')['close'].transform(lambda x: x.shift(1) / x.shift(6) - 1)
    df['rank'] = df.groupby('date')['mom_5'].rank(pct=True)
    signals = df[['code', 'date']].copy()
    signals['signal'] = 0
    signals.loc[df['rank'] >= (1 - top_pct), 'signal'] = 1
    signals.loc[df['rank'] <= top_pct, 'signal'] = -1
    return signals.sort_values(['date', 'code']).reset_index(drop=True)


def _run_naive_filter(data: pd.DataFrame, signals: pd.DataFrame):
    """
    模拟原 native_adapter 的重复过滤热循环(与优化版做等价工作)

    唯一区别: 用 `signals[signals['date']==dt]` 重复过滤(原实现)
    而非 预分组 dict 查找(优化版),其余逻辑(成本基准/滑点/T+1)保持一致
    以隔离"预分组"这一单一变量的性能影响
    """
    data = data.sort_values(['date', 'code']).reset_index(drop=True)
    signals = signals.sort_values(['date', 'code']).reset_index(drop=True)
    dates = sorted(signals['date'].unique())
    cash = 1e6
    positions = {}  # code -> {shares, today_bought, cost_basis}
    commission_rate = 0.00025
    stamp_tax_rate = 0.001
    slippage = 0.001

    for dt in dates:
        # 原实现: 每次循环内重复过滤 O(n) —— 这是被优化的热点
        day_signal = signals[signals['date'] == dt]
        day_data = data[data['date'] == dt]
        if day_data.empty:
            continue
        day_map = day_data.set_index('code')

        # 卖出(等价工作: 成本基准 + 滑点 + T+1)
        for _, row in day_signal.iterrows():
            code = row['code']
            sig = row.get('signal', 0)
            if sig >= 0:
                continue
            pos = positions.get(code)
            if not pos or pos['shares'] <= 0:
                continue
            closable = pos['shares'] - pos['today_bought']
            if closable <= 0:
                continue
            if code not in day_map.index:
                continue
            prow = day_map.loc[code]
            if prow.get('is_limit_down', False):
                continue
            price = float(prow['close']) * (1 - slippage)
            shares = closable
            amt = price * shares
            comm = max(amt * commission_rate, 5)
            tax = amt * stamp_tax_rate
            cash += amt - comm - tax
            pos['shares'] -= shares

        # 买入(等价工作)
        buy_codes = [r['code'] for _, r in day_signal.iterrows() if r.get('signal', 0) > 0]
        buy_codes = [c for c in buy_codes if c in day_map.index
                     and not day_map.loc[c].get('is_limit_up', False)]
        if buy_codes:
            budget = cash * 0.95 / len(buy_codes)
            for code in buy_codes:
                prow = day_map.loc[code]
                price = float(prow['close']) * (1 + slippage)
                shares = int(budget / price / 100) * 100
                if shares <= 0:
                    continue
                amt = price * shares
                comm = max(amt * commission_rate, 5)
                cost = amt + comm
                if cost > cash:
                    shares = int((cash * 0.98) / price / 100) * 100
                    if shares <= 0:
                        continue
                    amt = price * shares
                    comm = max(amt * commission_rate, 5)
                    cost = amt + comm
                cash -= cost
                pos = positions.setdefault(code, {'shares': 0, 'today_bought': 0, 'cost_basis': 0.0})
                total_cost = pos['cost_basis'] * pos['shares'] + price * shares
                pos['shares'] += shares
                pos['today_bought'] += shares
                pos['cost_basis'] = total_cost / pos['shares'] if pos['shares'] > 0 else 0

        # 估值(等价工作)
        mv = 0
        for code, pos in positions.items():
            if pos['shares'] <= 0:
                continue
            if code in day_map.index:
                mv += pos['shares'] * float(day_map.loc[code, 'close'])

        # T+1 结算
        for pos in positions.values():
            pos['today_bought'] = 0


def test_backtest_boundary(t: TestRunner):
    """测试3: 边界条件"""
    print("\n=== 测试3: 回测边界条件 ===")
    opt = OptimizedBacktestEngine()

    # 空数据
    r = opt.run_backtest(pd.DataFrame(), pd.DataFrame())
    t.assert_true("空数据返回空结果", r['metrics'] == {} and r['trades'].empty)

    # 单只股票
    data = make_synthetic_data(n_stocks=1, n_days=60, include_benchmark=False)
    sig = pd.DataFrame({'code': [data['code'].iloc[0]], 'date': [data['date'].iloc[0]], 'signal': [1]})
    r = opt.run_backtest(data, sig)
    t.assert_true("单只股票回测不崩溃", 'metrics' in r and r['equity_curve'] is not None)
    t.assert_true("单只股票产生买入交易", len(r['trades']) > 0)

    # 信号无匹配数据(停牌场景)
    data = make_synthetic_data(n_stocks=5, n_days=30, include_benchmark=False)
    sig = pd.DataFrame({'code': ['999999.SH'], 'date': [data['date'].iloc[0]], 'signal': [1]})
    r = opt.run_backtest(data, sig)
    t.assert_true("无匹配数据的信号不崩溃", r['equity_curve'] is not None)

    # T+1 约束: 同日买入同日卖出信号应被阻止
    data = make_synthetic_data(n_stocks=3, n_days=10, include_benchmark=False)
    dt = data['date'].iloc[0]
    code = data['code'].iloc[0]
    # 同一天先买后卖信号
    sig = pd.DataFrame([
        {'date': dt, 'code': code, 'signal': 1},
        {'date': dt, 'code': code, 'signal': -1},
    ])
    r = opt.run_backtest(data, sig, t_plus_1=True)
    # T+1 下,买入当日不可卖
    sell_trades = r['trades'][r['trades']['action'] == 'sell'] if not r['trades'].empty and 'action' in r['trades'].columns else pd.DataFrame()
    t.assert_true("T+1 阻止当日买入当日卖出", len(sell_trades) == 0 or sell_trades['shares'].sum() == 0,
                  f"卖出交易数={len(sell_trades)}")

    # Position T+1 单元测试
    pos = Position()
    pos.buy(100, 10.0)
    t.assert_close("T+1: 今日买入不可卖", pos.closable, 0)
    pos.settle_day()
    t.assert_close("T+1: 次日可卖", pos.closable, 100)
    pos.sell(50)
    t.assert_close("卖出后持仓正确", pos.shares, 50)
    t.assert_close("加权成本正确", pos.cost_basis, 10.0)

    # 涨跌停校验
    pos2 = Position()
    pos2.buy(100, 10.0)
    pos2.settle_day()
    v = PriceLimitValidator()
    t.assert_true("涨停拒绝买入", v.validate(OrderRequest('A', 'buy', 100, 10, 1000), {'is_limit_up': True}) is not None)
    t.assert_true("非涨停允许买入", v.validate(OrderRequest('A', 'buy', 100, 10, 1000), {'is_limit_up': False}) is None)
    t.assert_true("跌停拒绝卖出", v.validate(OrderRequest('A', 'sell', 100, 10, 1000), {'is_limit_down': True}) is not None)


def test_backtest_benchmark(t: TestRunner):
    """测试4: 基准跟踪指标"""
    print("\n=== 测试4: 基准跟踪 (alpha/beta/IR) ===")
    data = make_synthetic_data(n_stocks=30, n_days=200)
    signals = make_signals(data)
    opt = OptimizedBacktestEngine()
    r = opt.run_backtest(data, signals, benchmark="000300.SH")
    m = r['metrics']
    t.assert_true("含基准总收益", 'benchmark_total_return' in m, f"metrics keys={list(m.keys())}")
    t.assert_true("含 alpha", 'alpha' in m)
    t.assert_true("含 beta", 'beta' in m)
    t.assert_true("含 information_ratio", 'information_ratio' in m)
    t.assert_true("含 tracking_error", 'tracking_error' in m)
    t.assert_true("beta 在合理范围", -2 < m.get('beta', 0) < 3, f"beta={m.get('beta')}")


def test_factor_dsl_correctness(t: TestRunner):
    """测试5: 因子 DSL 正确性 — vs 手工 pandas 计算"""
    print("\n=== 测试5: 因子 DSL 正确性 (vs 手工计算) ===")
    data = make_synthetic_data(n_stocks=20, n_days=100, include_benchmark=False)
    engine = FactorExpressionEngine()

    # Mean($close, 20)
    result = engine.compute(data, ['Mean($close, 20)'])
    manual = data.sort_values(['code', 'date']).groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    ).reset_index(drop=True)
    dsl_vals = result['Mean($close,20)'].reset_index(drop=True)
    aligned = pd.concat([manual, dsl_vals], axis=1).dropna()
    t.assert_true("Mean($close,20) 与手工计算一致",
                  np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1], equal_nan=True, atol=1e-8))

    # Ref($close, 1) = 昨收
    result = engine.compute(data, ['Ref($close, 1)'])
    manual = data.sort_values(['code', 'date']).groupby('code')['close'].shift(1).reset_index(drop=True)
    dsl_vals = result['Ref($close,1)'].reset_index(drop=True)
    aligned = pd.concat([manual, dsl_vals], axis=1).dropna()
    t.assert_true("Ref($close,1) 与手工计算一致",
                  np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1], equal_nan=True, atol=1e-8))

    # $close / Ref($close, 20) - 1 = 20日收益率
    result = engine.compute(data, ['$close / Ref($close, 20) - 1'])
    df_s = data.sort_values(['code', 'date'])
    manual = (df_s['close'] / df_s.groupby('code')['close'].shift(20) - 1).reset_index(drop=True)
    dsl_vals = result['$close/Ref($close,20)-1'].reset_index(drop=True)
    aligned = pd.concat([manual, dsl_vals], axis=1).dropna()
    t.assert_true("复合表达式 $close/Ref($close,20)-1 正确",
                  np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1], equal_nan=True, atol=1e-8))

    # RSI($close, 14) — 与简单参考对比趋势
    result = engine.compute(data, ['RSI($close, 14)'])
    t.assert_true("RSI 计算产出非空值", result['RSI($close,14)'].notna().sum() > 50)
    t.assert_true("RSI 值在 [0,100] 范围",
                  result['RSI($close,14)'].dropna().between(0, 100).all())


def test_factor_dsl_performance(t: TestRunner):
    """测试6: 因子向量化性能 — vs 逐股循环"""
    print("\n=== 测试6: 因子向量化性能 (vs 逐股循环) ===")
    data = make_synthetic_data(n_stocks=200, n_days=300, include_benchmark=False)
    engine = FactorExpressionEngine()

    # 向量化 DSL
    t0 = time.time()
    engine.compute(data, ['Mean($close, 20)', 'Std($close, 20)', 'RSI($close, 14)'])
    vec_time = time.time() - t0

    # 逐股循环(模拟原 pandas_ta_calculator._calc_single)
    t0 = time.time()
    _naive_per_stock_factor(data)
    naive_time = time.time() - t0

    speedup = naive_time / vec_time if vec_time > 0 else float('inf')
    t.assert_true("向量化快于逐股循环", vec_time < naive_time,
                  f"向量化={vec_time:.3f}s, 逐股循环={naive_time:.3f}s, 加速比={speedup:.2f}x")


def _naive_per_stock_factor(data: pd.DataFrame):
    """模拟原 pandas_ta_calculator._calc_single 逐股循环"""
    data = data.sort_values(['code', 'date']).reset_index(drop=True)
    for code in data['code'].unique():
        mask = data['code'] == code
        sub = data[mask]
        _ = sub['close'].rolling(20, min_periods=10).mean()
        _ = sub['close'].rolling(20, min_periods=10).std()
        # RSI 简化
        diff = sub['close'].diff()
        gain = diff.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-diff.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        _ = 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def test_factor_cache(t: TestRunner):
    """测试7: 因子缓存"""
    print("\n=== 测试7: 因子缓存命中 ===")
    data = make_synthetic_data(n_stocks=20, n_days=100, include_benchmark=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FactorExpressionEngine(cache_dir=tmpdir)

        # 首次计算(写缓存)
        t0 = time.time()
        engine.compute(data, ['Mean($close, 20)'])
        first_time = time.time() - t0

        # 二次计算(读缓存)
        t0 = time.time()
        engine.compute(data, ['Mean($close, 20)'])
        cached_time = time.time() - t0

        t.assert_true("缓存文件已生成", len(os.listdir(tmpdir)) > 0)
        t.assert_true("缓存命中快于首次计算", cached_time < first_time,
                      f"首次={first_time:.4f}s, 缓存={cached_time:.4f}s")


def test_t_plus_1_label(t: TestRunner):
    """测试8: T+1 感知标签"""
    print("\n=== 测试8: T+1 感知标签 ===")
    data = make_synthetic_data(n_stocks=5, n_days=30, include_benchmark=False)

    t1 = t_plus_1_label(data)
    naive = naive_label(data)

    # T+1 标签: close[T+2]/close[T+1]-1, 末两行为 NaN
    df_s = data.sort_values(['code', 'date'])
    for code in df_s['code'].unique()[:1]:
        sub = df_s[df_s['code'] == code].reset_index(drop=True)
        if len(sub) >= 3:
            idx = sub.index[0]
            expected = sub['close'].iloc[2] / sub['close'].iloc[1] - 1
            actual = t1.iloc[df_s[df_s['code'] == code].index[0]]
            t.assert_close("T+1 标签 = close[T+2]/close[T+1]-1", float(actual), float(expected), tol=1e-6)

    # naive 标签: close[T+1]/close[T]-1
    for code in df_s['code'].unique()[:1]:
        sub = df_s[df_s['code'] == code].reset_index(drop=True)
        if len(sub) >= 2:
            expected = sub['close'].iloc[1] / sub['close'].iloc[0] - 1
            actual = naive.iloc[df_s[df_s['code'] == code].index[0]]
            t.assert_close("naive 标签 = close[T+1]/close[T]-1", float(actual), float(expected), tol=1e-6)

    # 两者应不同(T+1 标签错位一天)
    both = pd.concat([t1, naive], axis=1).dropna()
    t.assert_true("T+1 标签与 naive 标签不同(避免前视偏差)",
                  not np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-10))


def test_risk_fixes(t: TestRunner):
    """测试9: 风险引擎修复"""
    print("\n=== 测试9: 风险引擎修复 (HRP/CVaR) ===")
    data = make_synthetic_data(n_stocks=20, n_days=200, include_benchmark=False)
    pivot = data.pivot(index='date', columns='code', values='close')
    returns = pivot.pct_change().dropna()

    rf = RiskFixes()

    # HRP 修复
    weights, metrics = rf.optimize_hrp_fixed(returns)
    t.assert_true("HRP 权重非空", len(weights) > 0)
    t.assert_true("HRP 权重和≈1", abs(weights.sum() - 1.0) < 0.01, f"sum={weights.sum():.4f}")
    t.assert_true("HRP 权重非负", (weights >= 0).all())
    if rf.has_pypfopt:
        t.assert_true("HRP 方法标记正确", metrics.get('method') == 'hrp')
    else:
        t.assert_true("HRP 无pypfopt时回退等权", metrics.get('method') == 'equal_weight')

    # CVaR 修复(若 pypfopt 可用)
    weights_cvar, metrics_cvar = rf.optimize_cvar_fixed(returns)
    t.assert_true("CVaR 权重非空", len(weights_cvar) > 0)
    t.assert_true("CVaR 权重和≈1", abs(weights_cvar.sum() - 1.0) < 0.01, f"sum={weights_cvar.sum():.4f}")

    # VaR/CVaR 计算
    port_ret = returns.dot(weights)
    vc = rf.calc_var_cvar(port_ret)
    t.assert_true("VaR 为负(损失)", vc['VaR'] < 0)
    t.assert_true("CVaR <= VaR(更极端)", vc['CVaR'] <= vc['VaR'] + 1e-9,
                  f"VaR={vc['VaR']}, CVaR={vc['CVaR']}")


def test_validator_composition(t: TestRunner):
    """测试10: 盘前风控校验器组合"""
    print("\n=== 测试10: 盘前风控校验器组合 ===")
    v1 = PriceLimitValidator()
    v2 = MaxPositionValidator(max_weight=0.05)
    order = OrderRequest('A', 'buy', 1000, 10, 10000)
    # 涨停 + 超仓位 → 应被拒绝
    ctx = {'is_limit_up': True, 'total_equity': 100000, 'held_amount': 0}
    reasons = [v.validate(order, ctx) for v in [v1, v2]]
    rejected = [r for r in reasons if r is not None]
    t.assert_true("组合校验: 涨停被拒绝", len(rejected) > 0)

    # 非涨停 + 未超仓位 → 通过
    ctx2 = {'is_limit_up': False, 'total_equity': 1000000, 'held_amount': 0}
    reasons2 = [v.validate(order, ctx2) for v in [v1, v2]]
    t.assert_true("组合校验: 正常订单通过", all(r is None for r in reasons2))

    # 超仓位拒绝
    ctx3 = {'is_limit_up': False, 'total_equity': 100000, 'held_amount': 50000}
    reasons3 = [v.validate(order, ctx3) for v in [v1, v2]]
    rejected3 = [r for r in reasons3 if r is not None]
    t.assert_true("组合校验: 超仓位被拒绝", len(rejected3) > 0)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("jingni-trader 优化验证测试套件")
    print("分支: feat/quant-opt-20260624")
    print("借鉴: Qlib (表达式DSL+缓存) / RQAlpha (T+1+EventBus+FrontendValidator)")
    print("=" * 60)

    t = TestRunner()
    test_backtest_correctness(t)
    test_backtest_performance(t)
    test_backtest_boundary(t)
    test_backtest_benchmark(t)
    test_factor_dsl_correctness(t)
    test_factor_dsl_performance(t)
    test_factor_cache(t)
    test_t_plus_1_label(t)
    test_risk_fixes(t)
    test_validator_composition(t)

    ok = t.summary()

    # 保存测试结果 JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(out_dir, "test_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            "passed": t.passed,
            "failed": t.failed,
            "total": t.passed + t.failed,
            "all_passed": ok,
            "details": t.results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n测试结果已保存: {result_path}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
