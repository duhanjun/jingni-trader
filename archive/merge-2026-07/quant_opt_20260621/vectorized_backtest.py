"""
向量化回测引擎（验证原型）

借鉴来源：
- VectorBT (https://vectorbt.dev): 将回测逻辑表示为 NumPy 数组运算，
  避免逐 bar 的 Python 循环，性能提升 100-1000 倍。
- AKQuant (https://github.com/akfamily/akquant): Rust + Python 混合，
  Zero-Copy 数据架构。

优化目标：
jingni-trader 现有 skills/backtest-engine/scripts/adapters/native_adapter.py
使用 `for dt in dates: signals[signals['date'] == dt]` 的逐日循环 +
DataFrame 过滤，复杂度 O(n_days * n_rows)，在大数据量下极慢。

本模块通过以下手段优化：
1. 一次性 pivot 为宽表（date × code 矩阵），消除重复过滤
2. 将核心模拟循环操作在 NumPy 数组上进行（避免 pandas 索引开销）
3. 信号 / 涨跌停 / 收盘价全部预对齐为同形状矩阵
4. 保留 A 股 T+1、涨跌停、印花税、滑点等业务规则

注意：回测本质存在路径依赖（cash/position 随时间变化），
完全向量化困难，因此采用「数据预对齐 + 紧凑 NumPy 循环」的混合策略，
这与 VectorBT 用 Numba 编译内核的思路一致。
"""
from __future__ import annotations
from typing import Dict, Any
import time
import numpy as np
import pandas as pd


class VectorizedBacktest:
    """向量化回测引擎（A 股规则）"""

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1e6,
        benchmark: str = "000300.SH",
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        slippage: float = 0.001,
    ) -> Dict[str, Any]:
        if data.empty or signals.empty:
            return self._empty_result()

        # ---- 1. 数据预对齐：pivot 为宽表 (date × code) ----
        # 一次 pivot 替代原实现里每个日期都做一次 df[df['date']==dt] 的 O(n) 过滤
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        close_wide = data.pivot(index='date', columns='code', values='close')
        # 对齐 signals 到 close_wide 的行列
        signal_wide = signals.pivot(index='date', columns='code', values='signal')
        signal_wide = signal_wide.reindex(index=close_wide.index, columns=close_wide.columns).fillna(0.0)

        # 涨跌停标记（可选列）
        if 'is_limit_up' in data.columns:
            limit_up_wide = data.pivot(index='date', columns='code', values='is_limit_up') \
                .reindex(index=close_wide.index, columns=close_wide.columns).fillna(False).astype(bool)
        else:
            limit_up_wide = pd.DataFrame(False, index=close_wide.index, columns=close_wide.columns)

        if 'is_limit_down' in data.columns:
            limit_down_wide = data.pivot(index='date', columns='code', values='is_limit_down') \
                .reindex(index=close_wide.index, columns=close_wide.columns).fillna(False).astype(bool)
        else:
            limit_down_wide = pd.DataFrame(False, index=close_wide.index, columns=close_wide.columns)

        # ---- 2. 转为 NumPy 数组，进入紧凑循环 ----
        dates = close_wide.index.values
        codes = close_wide.columns.values
        close_arr = close_wide.values.astype(float)          # (T, N)
        signal_arr = signal_wide.values.astype(float)        # (T, N)
        limit_up_arr = limit_up_wide.values                  # (T, N) bool
        limit_down_arr = limit_down_wide.values              # (T, N) bool

        n_dates, n_codes = close_arr.shape
        # 用 NaN 表示当日无行情
        close_arr = np.where(np.isfinite(close_arr), close_arr, np.nan)

        # ---- 3. 模拟核心：紧凑 NumPy 循环（无 pandas 开销）----
        cash = float(init_capital)
        positions = np.zeros(n_codes, dtype=np.float64)      # 持仓股数
        # T+1: 记录当日买入的股票，次日才能卖
        bought_today = np.zeros(n_codes, dtype=bool) if t_plus_1 else None

        equity_curve = np.zeros(n_dates, dtype=np.float64)
        cash_curve = np.zeros(n_dates, dtype=np.float64)
        mv_curve = np.zeros(n_dates, dtype=np.float64)
        pos_count_curve = np.zeros(n_dates, dtype=np.int64)

        trades = []  # list of dict

        for t in range(n_dates):
            close_t = close_arr[t]
            sig_t = signal_arr[t]
            lu_t = limit_up_arr[t]
            ld_t = limit_down_arr[t]

            # 当日有行情的股票
            has_quote = np.isfinite(close_t)

            # ---- 卖出（signal < 0）----
            sell_mask = (sig_t < 0) & (positions > 0) & has_quote
            if t_plus_1:
                sell_mask = sell_mask & (~bought_today)
            sell_mask = sell_mask & (~ld_t)  # 跌停不能卖

            sell_idx = np.where(sell_mask)[0]
            for i in sell_idx:
                price = float(close_t[i])
                shares = float(positions[i])
                if shares <= 0:
                    continue
                amount = price * shares
                commission = max(amount * commission_rate, 5.0)
                tax = amount * stamp_tax_rate
                cash += amount - commission - tax
                positions[i] = 0.0
                trades.append({
                    'date': dates[t], 'code': codes[i], 'action': 'sell',
                    'price': price, 'shares': int(shares), 'amount': amount,
                    'commission': commission, 'tax': tax,
                    'pnl': amount - commission - tax,
                })

            # ---- 买入（signal > 0）----
            buy_mask = (sig_t > 0) & has_quote & (~lu_t)  # 涨停不能买
            buy_idx = np.where(buy_mask)[0]
            if len(buy_idx) > 0:
                budget_per_stock = cash * 0.95 / len(buy_idx)
                for i in buy_idx:
                    price = float(close_t[i]) * (1.0 + slippage)
                    if not np.isfinite(price) or price <= 0:
                        continue
                    shares = int(budget_per_stock / price / 100) * 100
                    if shares <= 0:
                        continue
                    amount = price * shares
                    commission = max(amount * commission_rate, 5.0)
                    cost = amount + commission
                    if cost > cash:
                        shares = int((cash * 0.98) / price / 100) * 100
                        if shares <= 0:
                            continue
                        amount = price * shares
                        commission = max(amount * commission_rate, 5.0)
                        cost = amount + commission
                    cash -= cost
                    positions[i] += shares
                    if t_plus_1:
                        bought_today[i] = True
                    trades.append({
                        'date': dates[t], 'code': codes[i], 'action': 'buy',
                        'price': price, 'shares': int(shares), 'amount': amount,
                        'commission': commission, 'tax': 0.0,
                        'pnl': -amount - commission,
                    })

            # ---- 结算当日净值 ----
            market_value = float(np.nansum(positions * np.where(has_quote, close_t, 0.0)))
            equity_curve[t] = cash + market_value
            cash_curve[t] = cash
            mv_curve[t] = market_value
            pos_count_curve[t] = int(np.sum(positions > 0))

            # T+1: 次日清空 bought_today
            if t_plus_1:
                bought_today[:] = False

        # ---- 4. 组装结果 ----
        equity_df = pd.DataFrame({
            'date': dates,
            'equity': equity_curve,
            'cash': cash_curve,
            'market_value': mv_curve,
            'position_count': pos_count_curve,
        })

        trades_df = pd.DataFrame(trades)
        positions_df = pd.DataFrame({'code': codes, 'shares': positions})

        eq_series = pd.Series(equity_curve, index=dates)
        # 复用现有指标计算（保持与原实现一致）
        # 注意：skills/backtest-engine 目录含连字符无法直接 import，
        # 这里内联等价实现，避免对原包路径的硬依赖
        metrics = _calc_all_metrics(eq_series, trades_df)

        return {
            "trades": trades_df,
            "positions": positions_df,
            "equity_curve": equity_df,
            "metrics": metrics,
            "report_path": "",
        }

    @staticmethod
    def _empty_result():
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }


def run_backtest_timed(engine, data, signals, **kwargs) -> Dict[str, Any]:
    """带计时包装的回测，便于性能对比"""
    t0 = time.perf_counter()
    result = engine.run_backtest(data, signals, **kwargs)
    elapsed = time.perf_counter() - t0
    result["elapsed_seconds"] = elapsed
    return result


# ---------------- 绩效指标计算（与 BaseBacktestMetrics 等价） ----------------

def _calc_all_metrics(equity_curve: pd.Series, trades: pd.DataFrame,
                      risk_free: float = 0.03, trading_days: int = 252) -> Dict[str, Any]:
    """与 skills/backtest-engine base_backtest.py 中 calc_all_metrics 等价的实现"""
    from datetime import datetime

    def _total_return(eq):
        if len(eq) < 2:
            return 0.0
        return float(eq.iloc[-1] / eq.iloc[0] - 1)

    def _annual_return(eq):
        if len(eq) < 2:
            return 0.0
        total = eq.iloc[-1] / eq.iloc[0]
        n_years = len(eq) / trading_days
        if n_years <= 0:
            return 0.0
        return float(total ** (1 / n_years) - 1)

    def _volatility(returns):
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(trading_days))

    def _sharpe(returns):
        vol = _volatility(returns)
        if vol == 0:
            return 0.0
        ann = returns.mean() * trading_days
        return float((ann - risk_free) / vol)

    def _max_drawdown(eq):
        if len(eq) < 2:
            return 0.0
        cummax = eq.cummax()
        dd = (eq - cummax) / cummax
        return float(dd.min())

    def _calmar(eq):
        ann = _annual_return(eq)
        mdd = abs(_max_drawdown(eq))
        if mdd == 0:
            return 0.0
        return float(ann / mdd)

    def _win_rate(tr):
        if tr.empty:
            return 0.0
        winning = (tr["pnl"] > 0).sum()
        total = len(tr)
        return float(winning / total) if total > 0 else 0.0

    def _sortino(returns):
        neg = returns[returns < 0]
        if len(neg) < 2:
            return 0.0
        downside = neg.std() * np.sqrt(trading_days)
        if downside == 0:
            return 0.0
        ann = returns.mean() * trading_days
        return float((ann - risk_free) / downside)

    returns = equity_curve.pct_change().dropna()
    return {
        "total_return": _total_return(equity_curve),
        "annual_return": _annual_return(equity_curve),
        "volatility": _volatility(returns),
        "sharpe_ratio": _sharpe(returns),
        "max_drawdown": _max_drawdown(equity_curve),
        "calmar_ratio": _calmar(equity_curve),
        "sortino_ratio": _sortino(returns),
        "win_rate": _win_rate(trades),
        "total_trades": len(trades),
        "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
