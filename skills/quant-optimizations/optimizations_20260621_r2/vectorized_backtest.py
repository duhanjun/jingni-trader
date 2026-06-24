"""
向量化回测引擎 + 扩展绩效指标（优化验证版）

借鉴来源：
- Microsoft Qlib: 顶层向量化回测，避免逐日 Python 循环
- Backtrader/QuantConnect: 完整绩效指标体系（alpha/beta/Sortino/turnover/information ratio）

针对 jingni-trader skills/backtest-engine/scripts/adapters/native_adapter.py 的优化点：
1. 原实现 for dt in dates: day_signal = signals[signals['date'] == dt] 是 O(N*M) 逐日过滤
   优化为：预先按 date 排序 + 一次 groupby，避免重复过滤
2. 原实现 _calc_metrics 仅返回 7 个指标，缺少 alpha/beta/Sortino/turnover/information_ratio
   优化为：补充完整指标体系，并加入基准对比
3. 原实现持仓市值计算用 Python dict 循环，优化为向量化持仓表 join

本模块仅用于性能/正确性对比验证，不修改 main 分支代码。
"""
from __future__ import annotations

import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 优化版：向量化回测引擎
# ----------------------------------------------------------------------

def run_backtest_vectorized(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    init_capital: float = 1_000_000.0,
    benchmark: str = "000300.SH",
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.001,
    t_plus_1: bool = True,
    price_limit: bool = True,
    slippage: float = 0.001,
    benchmark_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    向量化回测。

    核心优化：
    1. 预先按 date 排序，用 groupby 一次性获取每日信号/行情，避免循环内重复过滤
    2. 持仓市值用 DataFrame join 向量化计算，替代 Python dict 循环
    3. 交易记录用列表收集后一次构建 DataFrame

    注意：T+1、涨跌停、印花税等 A 股规则保留，确保与原实现语义一致。
    """
    if data.empty or signals.empty:
        return _empty_result()

    data = data.sort_values(['date', 'code']).reset_index(drop=True)
    signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

    # 预先按 date 分组为 dict（O(N) 一次构建），避免循环内 O(N*M) 过滤
    signal_groups = {dt: g for dt, g in signals.groupby('date', sort=True)}
    data_groups = {dt: g for dt, g in data.groupby('date', sort=True)}

    dates = sorted(signal_groups.keys())
    if not dates:
        return _empty_result()

    cash = init_capital
    positions: Dict[str, int] = {}
    equity_records = []
    trades = []

    for dt in dates:
        day_signal = signal_groups[dt]
        day_data = data_groups.get(dt)
        if day_data is None or day_data.empty:
            continue

        # 用 set_index 一次构建映射，替代原实现的逐行 .loc
        day_data_map = day_data.set_index('code')

        # 信号分类（向量化）
        sig_values = day_signal['signal'].values
        sig_codes = day_signal['code'].values
        # 数值化
        sig_numeric = pd.to_numeric(pd.Series(sig_values), errors='coerce').fillna(0).values
        buy_mask = sig_numeric > 0
        sell_mask = sig_numeric < 0
        buy_codes = sig_codes[buy_mask].tolist()
        sell_codes = sig_codes[sell_mask].tolist()

        # 卖出
        for code in sell_codes:
            if code not in positions or positions[code] <= 0:
                continue
            if code not in day_data_map.index:
                continue
            price_row = day_data_map.loc[code]
            if price_limit and _safe_bool(price_row.get('is_limit_down')):
                continue
            price = price_row['close']
            shares = positions[code]
            sell_amount = price * shares
            commission = max(sell_amount * commission_rate, 5)
            tax = sell_amount * stamp_tax_rate
            cost = commission + tax
            cash += sell_amount - cost
            trades.append({
                'date': dt, 'code': code, 'action': 'sell',
                'price': price, 'shares': shares, 'amount': sell_amount,
                'commission': commission, 'tax': tax,
            })
            positions[code] = 0

        # 买入
        if buy_codes:
            n_buy = len(buy_codes)
            budget_per_stock = cash * 0.95 / n_buy
            for code in buy_codes:
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                if price_limit and _safe_bool(price_row.get('is_limit_up')):
                    continue
                price = price_row['close'] * (1 + slippage)
                shares = int(budget_per_stock / price / 100) * 100
                if shares <= 0:
                    continue
                buy_amount = price * shares
                commission = max(buy_amount * commission_rate, 5)
                cost = buy_amount + commission
                if cost > cash:
                    shares = int((cash * 0.98) / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * commission_rate, 5)
                    cost = buy_amount + commission
                cash -= cost
                positions[code] = positions.get(code, 0) + shares
                trades.append({
                    'date': dt, 'code': code, 'action': 'buy',
                    'price': price, 'shares': shares, 'amount': buy_amount,
                    'commission': commission, 'tax': 0,
                })

        # 持仓市值：向量化计算
        active_codes = [c for c, s in positions.items() if s > 0]
        if active_codes:
            held = day_data_map.reindex(active_codes)
            held_shares = pd.Series(positions).reindex(active_codes).fillna(0)
            market_value = float((held['close'].fillna(0).values * held_shares.values).sum())
        else:
            market_value = 0.0

        total_equity = cash + market_value
        equity_records.append({
            'date': dt,
            'equity': total_equity,
            'cash': cash,
            'market_value': market_value,
            'position_count': len(active_codes),
        })

    equity_curve = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trades)

    if equity_curve.empty:
        return _empty_result()

    metrics = calc_extended_metrics(
        equity_curve=equity_curve,
        trades_df=trades_df,
        init_capital=init_capital,
        benchmark_data=benchmark_data,
        risk_free_rate=0.03,
    )

    return {
        "trades": trades_df,
        "positions": pd.DataFrame(
            list(positions.items()), columns=['code', 'shares']
        ),
        "equity_curve": equity_curve,
        "metrics": metrics,
        "report_path": "",
    }


def _safe_bool(val) -> bool:
    """安全转 bool，处理 NaN/None。"""
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    return bool(val)


def _empty_result() -> Dict[str, Any]:
    return {
        "trades": pd.DataFrame(),
        "positions": pd.DataFrame(),
        "equity_curve": pd.DataFrame(),
        "metrics": {},
        "report_path": "",
    }


# ----------------------------------------------------------------------
# 扩展绩效指标（借鉴 Qlib + QuantConnect）
# ----------------------------------------------------------------------

def calc_extended_metrics(
    equity_curve: pd.DataFrame,
    trades_df: pd.DataFrame,
    init_capital: float,
    benchmark_data: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    计算完整绩效指标体系。

    原实现 _calc_metrics 仅返回：
        total_return, annual_return, volatility, sharpe_ratio,
        max_drawdown, win_rate, calmar_ratio

    本实现补充（借鉴 Qlib/QuantConnect）：
        - alpha, beta（CAPM，需 benchmark）
        - information_ratio（相对基准）
        - sortino_ratio（下行风险）
        - turnover（年化换手率）
        - benchmark_return（基准收益）
        - excess_return（超额收益）
    """
    if equity_curve.empty or 'equity' not in equity_curve.columns:
        return {}

    eq = equity_curve.set_index('date')['equity']
    if len(eq) < 2:
        return {}

    returns = eq.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    volatility = returns.std() * np.sqrt(periods_per_year)
    max_drawdown = (eq / eq.cummax() - 1).min()
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Sortino：仅用下行波动
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

    # 换手率
    turnover = _calc_turnover(trades_df, init_capital, n_periods, periods_per_year)

    metrics = {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "calmar_ratio": float(calmar),
        "sortino_ratio": float(sortino),
        "turnover_annual": float(turnover),
    }

    # 基准相关指标
    if benchmark_data is not None and not benchmark_data.empty:
        bench_metrics = _calc_benchmark_metrics(
            returns, benchmark_data, risk_free_rate, periods_per_year
        )
        metrics.update(bench_metrics)

    return metrics


def _calc_turnover(
    trades_df: pd.DataFrame,
    init_capital: float,
    n_periods: int,
    periods_per_year: int,
) -> float:
    """计算年化换手率 = 总成交额 / 平均权益 * (252 / 交易天数)。"""
    if trades_df.empty or 'amount' not in trades_df.columns:
        return 0.0
    total_amount = float(trades_df['amount'].sum())
    avg_capital = init_capital  # 简化：用初始资金
    if avg_capital <= 0 or n_periods == 0:
        return 0.0
    daily_turnover = total_amount / avg_capital / n_periods
    return daily_turnover * periods_per_year


def _calc_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_data: pd.DataFrame,
    risk_free_rate: float,
    periods_per_year: int,
) -> Dict[str, float]:
    """计算 alpha/beta/information_ratio/benchmark_return。"""
    # 基准收益率序列
    if 'date' in benchmark_data.columns and 'close' in benchmark_data.columns:
        bench = benchmark_data.set_index('date').sort_index()['close']
    elif isinstance(benchmark_data, pd.Series):
        bench = benchmark_data.sort_index()
    else:
        return {}

    bench_returns = bench.pct_change().dropna()
    # 对齐
    common = strategy_returns.index.intersection(bench_returns.index)
    if len(common) < 2:
        return {}

    s = strategy_returns.loc[common]
    b = bench_returns.loc[common]

    # beta = Cov(s,b)/Var(b)，alpha = mean(s) - beta*mean(b)（年化）
    cov_matrix = np.cov(s.values, b.values)
    var_b = cov_matrix[1, 1]
    beta = float(cov_matrix[0, 1] / var_b) if var_b > 0 else 0.0
    alpha_daily = float(s.mean() - beta * b.mean())
    alpha_annual = alpha_daily * periods_per_year

    # 信息比率：超额收益 / 跟踪误差
    excess = s - b
    tracking_error = excess.std() * np.sqrt(periods_per_year)
    ir = float(excess.mean() * periods_per_year / tracking_error) if tracking_error > 0 else 0.0

    bench_total_return = float((1 + b).prod() - 1)
    bench_annual = float((1 + bench_total_return) ** (periods_per_year / len(b)) - 1)

    return {
        "alpha": alpha_annual,
        "beta": beta,
        "information_ratio": ir,
        "benchmark_total_return": bench_total_return,
        "benchmark_annual_return": bench_annual,
        "excess_total_return": float((1 + s).prod() - (1 + b).prod()),
        "tracking_error": float(tracking_error),
    }


# ----------------------------------------------------------------------
# 基准实现（复刻原 native_adapter.py 逻辑，用于对比）
# ----------------------------------------------------------------------

def run_backtest_baseline(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    init_capital: float = 1_000_000.0,
    benchmark: str = "000300.SH",
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.001,
    t_plus_1: bool = True,
    price_limit: bool = True,
    slippage: float = 0.001,
) -> Dict[str, Any]:
    """复刻 skills/backtest-engine/scripts/adapters/native_adapter.py 的逐日循环逻辑。"""
    if data.empty or signals.empty:
        return _empty_result()

    data = data.sort_values(['date', 'code']).reset_index(drop=True)
    signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

    dates = sorted(signals['date'].unique())
    if not dates:
        return _empty_result()

    cash = init_capital
    positions = {}
    equity_records = []
    trades = []

    for dt in dates:
        day_signal = signals[signals['date'] == dt]   # 原实现的逐日过滤
        day_data = data[data['date'] == dt]            # 原实现的逐日过滤

        if day_data.empty:
            continue

        day_data_map = day_data.set_index('code')

        sell_codes = []
        buy_codes = []
        for _, row in day_signal.iterrows():           # 原实现的逐行迭代
            code = row['code']
            sig = row.get('signal', 0)
            if isinstance(sig, (int, float, np.integer, np.floating)):
                sig = float(sig)
                if sig > 0:
                    buy_codes.append(code)
                elif sig < 0:
                    sell_codes.append(code)

        for code in sell_codes:
            if code not in positions or positions[code] <= 0:
                continue
            if code not in day_data_map.index:
                continue
            price_row = day_data_map.loc[code]
            if price_limit and price_row.get('is_limit_down', False):
                continue
            price = price_row['close']
            shares = positions[code]
            sell_amount = price * shares
            commission = max(sell_amount * commission_rate, 5)
            tax = sell_amount * stamp_tax_rate
            cost = commission + tax
            cash += sell_amount - cost
            trades.append({
                'date': dt, 'code': code, 'action': 'sell',
                'price': price, 'shares': shares, 'amount': sell_amount,
                'commission': commission, 'tax': tax, 'pnl': sell_amount - cost,
            })
            positions[code] = 0

        if buy_codes:
            n_buy = len(buy_codes)
            budget_per_stock = cash * 0.95 / n_buy
            for code in buy_codes:
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                if price_limit and price_row.get('is_limit_up', False):
                    continue
                price = price_row['close'] * (1 + slippage)
                shares = int(budget_per_stock / price / 100) * 100
                if shares <= 0:
                    continue
                buy_amount = price * shares
                commission = max(buy_amount * commission_rate, 5)
                cost = buy_amount + commission
                if cost > cash:
                    shares = int((cash * 0.98) / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * commission_rate, 5)
                    cost = buy_amount + commission
                cash -= cost
                positions[code] = positions.get(code, 0) + shares
                trades.append({
                    'date': dt, 'code': code, 'action': 'buy',
                    'price': price, 'shares': shares, 'amount': buy_amount,
                    'commission': commission, 'tax': 0, 'pnl': -buy_amount - commission,
                })

        market_value = 0
        for code, shares in list(positions.items()):    # 原实现的 dict 循环
            if shares <= 0:
                continue
            if code in day_data_map.index:
                market_value += shares * day_data_map.loc[code, 'close']
        total_equity = cash + market_value

        equity_records.append({
            'date': dt,
            'equity': total_equity,
            'cash': cash,
            'market_value': market_value,
            'position_count': sum(1 for s in positions.values() if s > 0),
        })

    equity_curve = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trades)

    if equity_curve.empty:
        return _empty_result()

    # 原实现的 _calc_metrics（仅 7 个指标）
    metrics = _calc_metrics_baseline(equity_curve, init_capital)

    return {
        "trades": trades_df,
        "positions": pd.DataFrame(list(positions.items()), columns=['code', 'shares']),
        "equity_curve": equity_curve,
        "metrics": metrics,
        "report_path": "",
    }


def _calc_metrics_baseline(equity_curve: pd.DataFrame, init_capital: float) -> Dict[str, float]:
    """复刻原 backtest-engine/engine.py _calc_metrics 的逻辑。"""
    if equity_curve.empty or 'equity' not in equity_curve.columns:
        return {}
    eq = equity_curve.set_index('date')['equity']
    if len(eq) < 2:
        return {}
    returns = eq.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    max_drawdown = (eq / eq.cummax() - 1).min()
    sharpe = (annual_return - 0.03) / volatility if volatility != 0 else 0
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "calmar_ratio": float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0,
    }