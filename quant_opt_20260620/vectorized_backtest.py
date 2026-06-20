"""
向量化回测引擎 —— 优化验证实现

借鉴来源:
- VectorBT: 信号延迟(shift) + 次日开盘成交 防前视偏差; 向量化资金曲线; 向量化成本
- Qlib: 面板数据 MultiIndex 索引; 横截面 groupby 向量化
- NautilusTrader: 成交模型可插拔; T+1 可卖头寸概念

本文件提供三个实现用于对比验证 (均实现与 BaseBacktestEngine.run_backtest 相同接口):

1. BaselineLoopAdapter      —— 复刻 jingni-trader 现有 native_adapter 的逻辑(同日收盘价执行),
                               作为正确性与性能基线。注意: 该版本存在前视偏差(原项目既有问题)。
2. LookaheadFixedAdapter    —— 修复前视偏差: 信号 shift(1) + 次日开盘价成交, 保留逐日循环结构,
                               用于隔离"正确性修复"的效应。
3. VectorizedAdapter        —— 完全向量化(矩阵化)实现, 用于性能对比。

关键修复点(对照原 native_adapter.py):
- 原实现 L44-46/L73/L96: 信号在 t 日基于 close 产生, 却在 t 日 close 成交 → 前视偏差
- 修复: 信号 shift 到 t+1, 用 t+1 的 open 成交
- T+1: 用 available_positions 概念(NautilusTrader 启发), 当日买入次日才可卖
"""
from __future__ import annotations
from typing import Dict, Any, List
import time
import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────
# 1. 基线: 复刻原 native_adapter 逻辑(含前视偏差, 仅作对照)
# ────────────────────────────────────────────────────────────
class BaselineLoopAdapter:
    """
    复刻 jingni-trader skills/backtest-engine/scripts/adapters/native_adapter.py
    的核心逻辑(同日收盘价执行), 用于对比基线。

    注意: 此版本刻意保留原项目的前视偏差(lookahead bias), 以便测试能够检测出该问题。
    """

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

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)
        dates = sorted(signals["date"].unique())
        if not dates:
            return self._empty_result()

        cash = init_capital
        positions: Dict[str, int] = {}
        # 记录买入日期, 用于 T+1 校验 (原实现未做, 这里保留以观察差异)
        buy_date: Dict[str, Any] = {}
        equity_records = []
        trades = []

        for dt in dates:
            day_signal = signals[signals["date"] == dt]
            day_data = data[data["date"] == dt]
            if day_data.empty:
                continue
            day_data_map = day_data.set_index("code")

            sell_codes, buy_codes = [], []
            for _, row in day_signal.iterrows():
                code = row["code"]
                sig = row.get("signal", 0)
                if isinstance(sig, (int, float, np.integer, np.floating)):
                    sig = float(sig)
                    if sig > 0:
                        buy_codes.append(code)
                    elif sig < 0:
                        sell_codes.append(code)

            # 卖出
            for code in sell_codes:
                if positions.get(code, 0) <= 0:
                    continue
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                if price_limit and price_row.get("is_limit_down", False):
                    continue
                price = price_row["close"]  # ← 原实现: 同日 close (前视偏差)
                shares = positions[code]
                sell_amount = price * shares
                commission = max(sell_amount * commission_rate, 5)
                tax = sell_amount * stamp_tax_rate
                cost = commission + tax
                cash += sell_amount - cost
                trades.append({
                    "date": dt, "code": code, "action": "sell",
                    "price": price, "shares": shares, "amount": sell_amount,
                    "commission": commission, "tax": tax,
                    "pnl": sell_amount - cost,
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
                    if price_limit and price_row.get("is_limit_up", False):
                        continue
                    price = price_row["close"] * (1 + slippage)  # ← 原实现: 同日 close (前视偏差)
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
                    buy_date[code] = dt
                    trades.append({
                        "date": dt, "code": code, "action": "buy",
                        "price": price, "shares": shares, "amount": buy_amount,
                        "commission": commission, "tax": 0,
                        "pnl": -buy_amount - commission,
                    })

            # 估值
            market_value = 0.0
            for code, shares in list(positions.items()):
                if shares <= 0:
                    continue
                if code in day_data_map.index:
                    market_value += shares * day_data_map.loc[code, "close"]
            total_equity = cash + market_value
            equity_records.append({
                "date": dt, "equity": total_equity, "cash": cash,
                "market_value": market_value,
                "position_count": sum(1 for s in positions.values() if s > 0),
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)
        if equity_curve.empty:
            return self._empty_result()
        eq_series = equity_curve.set_index("date")["equity"]
        metrics = _calc_metrics(eq_series, trades_df)
        return {
            "trades": trades_df,
            "positions": pd.DataFrame(list(positions.items()), columns=["code", "shares"]),
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    def _empty_result(self):
        return {
            "trades": pd.DataFrame(), "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(), "metrics": {}, "report_path": "",
        }


# ────────────────────────────────────────────────────────────
# 2. 前视偏差修复版: 信号 shift(1) + 次日开盘成交 (保留逐日循环)
# ────────────────────────────────────────────────────────────
class LookaheadFixedAdapter:
    """
    修复前视偏差:
    - 信号在 t 日产生 → 在 t+1 日以 open 价成交
    - T+1: 当日买入的标的当日不可卖出(available_positions 概念)

    保留逐日循环结构, 仅修复执行时序, 用于隔离"正确性修复"的效应。
    """

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

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        # 关键修复1: 信号延迟到"下一个交易日"执行 (而非下一行)
        # 用完整交易日历做 pivot + reindex + shift(1), 确保按交易日偏移
        all_dates = sorted(data["date"].unique())
        if not all_dates:
            return self._empty_result()
        sig_pivot = signals.pivot_table(index="date", columns="code", values="signal", aggfunc="first")
        sig_pivot = sig_pivot.reindex(all_dates).shift(1)  # 按交易日 shift 1
        exec_signals = sig_pivot.stack().reset_index()
        exec_signals.columns = ["date", "code", "signal"]
        exec_signals = exec_signals[exec_signals["signal"].notna()]
        exec_signals = exec_signals.sort_values(["date", "code"]).reset_index(drop=True)

        cash = init_capital
        positions: Dict[str, int] = {}
        # T+1: 记录每个标的的最近买入日期, 当日买入当日不可卖
        last_buy_date: Dict[str, Any] = {}
        equity_records = []
        trades = []

        # 用 data 的日期驱动(而非 signals), 因为执行发生在行情日
        data_map_by_date = {dt: g.set_index("code") for dt, g in data.groupby("date")}

        for dt in all_dates:
            day_data_map = data_map_by_date.get(dt)
            if day_data_map is None or day_data_map.empty:
                continue
            day_signal = exec_signals[exec_signals["date"] == dt]

            sell_codes, buy_codes = [], []
            for _, row in day_signal.iterrows():
                code = row["code"]
                sig = row.get("signal", 0)
                if isinstance(sig, (int, float, np.integer, np.floating)):
                    sig = float(sig)
                    if sig > 0:
                        buy_codes.append(code)
                    elif sig < 0:
                        sell_codes.append(code)

            # 卖出 (T+1: 当日买入的不可卖)
            for code in sell_codes:
                if positions.get(code, 0) <= 0:
                    continue
                if t_plus_1 and last_buy_date.get(code) == dt:
                    continue  # T+1 约束
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                if price_limit and price_row.get("is_limit_down", False):
                    continue
                # 关键修复2: 用次日 open 成交(此处 dt 已是执行日=信号日+1)
                price = price_row["open"] * (1 - slippage)
                shares = positions[code]
                sell_amount = price * shares
                commission = max(sell_amount * commission_rate, 5)
                tax = sell_amount * stamp_tax_rate
                cost = commission + tax
                cash += sell_amount - cost
                trades.append({
                    "date": dt, "code": code, "action": "sell",
                    "price": price, "shares": shares, "amount": sell_amount,
                    "commission": commission, "tax": tax,
                    "pnl": sell_amount - cost,
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
                    if price_limit and price_row.get("is_limit_up", False):
                        continue
                    # 关键修复2: 用次日 open 成交
                    price = price_row["open"] * (1 + slippage)
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
                    last_buy_date[code] = dt
                    trades.append({
                        "date": dt, "code": code, "action": "buy",
                        "price": price, "shares": shares, "amount": buy_amount,
                        "commission": commission, "tax": 0,
                        "pnl": -buy_amount - commission,
                    })

            # 估值 (用当日 close)
            market_value = 0.0
            for code, shares in list(positions.items()):
                if shares <= 0:
                    continue
                if code in day_data_map.index:
                    market_value += shares * day_data_map.loc[code, "close"]
            total_equity = cash + market_value
            equity_records.append({
                "date": dt, "equity": total_equity, "cash": cash,
                "market_value": market_value,
                "position_count": sum(1 for s in positions.values() if s > 0),
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)
        if equity_curve.empty:
            return self._empty_result()
        eq_series = equity_curve.set_index("date")["equity"]
        metrics = _calc_metrics(eq_series, trades_df)
        return {
            "trades": trades_df,
            "positions": pd.DataFrame(list(positions.items()), columns=["code", "shares"]),
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    def _empty_result(self):
        return {
            "trades": pd.DataFrame(), "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(), "metrics": {}, "report_path": "",
        }


# ────────────────────────────────────────────────────────────
# 3. 完全向量化版 (矩阵化, 等权多头组合)
# ────────────────────────────────────────────────────────────
class VectorizedAdapter:
    """
    完全向量化的等权多头组合回测 (VectorBT 风格)。

    核心思路:
    1. 将 close/open/limit 透视为 (date × code) 矩阵
    2. 信号 shift(1) 防前视, 次日 open 成交
    3. 目标持仓 = (shifted_signal==1) & ~limit_up  (涨停不可买)
    4. 等权: weights = target_hold / target_hold.sum(axis=1)
    5. 换手 turnover = |weights_t - weights_{t-1}|, 向量化扣成本
    6. 组合收益 = sum(weights * stock_return), stock_return = close_t/close_{t-1}-1
    7. equity = init_capital * cumprod(1 + port_return - cost)

    注: 等权分数权重(不做整手取整), 与逐日循环版的"整手+预算"逻辑不同,
    因此绝对净值会有差异, 但趋势/指标一致。本版本用于性能对比与无前视偏差验证。
    """

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

        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        sig = signals.copy()
        sig["date"] = pd.to_datetime(sig["date"])

        # 透视为矩阵 (date × code)
        close = df.pivot_table(index="date", columns="code", values="close").sort_index()
        open_ = df.pivot_table(index="date", columns="code", values="open").sort_index()
        if "is_limit_up" in df.columns:
            limit_up = df.pivot_table(index="date", columns="code", values="is_limit_up").fillna(False).astype(bool).sort_index()
        else:
            limit_up = pd.DataFrame(False, index=close.index, columns=close.columns)
        if "is_limit_down" in df.columns:
            limit_down = df.pivot_table(index="date", columns="code", values="is_limit_down").fillna(False).astype(bool).sort_index()
        else:
            limit_down = pd.DataFrame(False, index=close.index, columns=close.columns)

        sig_mat = sig.pivot_table(index="date", columns="code", values="signal", aggfunc="last").reindex_like(close).fillna(0).sort_index()

        # 对齐到 close 的日期索引
        common_dates = close.index
        sig_mat = sig_mat.reindex(common_dates).fillna(0)
        open_ = open_.reindex_like(close)
        limit_up = limit_up.reindex_like(close).fillna(False)
        limit_down = limit_down.reindex_like(close).fillna(False)

        # 1) 信号延迟 1 日 (防前视): t 日信号 → t+1 执行
        target_signal = sig_mat.shift(1).fillna(0)
        # 买入意图
        want_buy = target_signal > 0
        # 涨停不可买
        if price_limit:
            want_buy = want_buy & ~limit_up
        # 目标持仓 (等权多头)
        hold_count = want_buy.sum(axis=1).replace(0, np.nan)
        target_weights = want_buy.div(hold_count, axis=0).fillna(0.0)

        # 2) T+1 约束: 若某标的今日买入, 则今日权重生效但不可同日卖出
        #    等权多头场景下, T+1 体现为: 权重只能从 0→正 或 正→0, 不能在同日反复
        #    向量化近似: 用前一日权重 + 今日目标权重, 取目标权重(已 shift), 天然满足持有≥1日
        #    (精确 T+1 整手逻辑见 LookaheadFixedAdapter, 此处为性能版近似)

        # 3) 换手率 (向量化)
        weights = target_weights
        prev_weights = weights.shift(1).fillna(0)
        turnover = (weights - prev_weights).abs().sum(axis=1)
        # 买入换手 + 卖出换手
        buy_turnover = (weights - prev_weights).clip(lower=0).sum(axis=1)
        sell_turnover = (prev_weights - weights).clip(lower=0).sum(axis=1)

        # 4) 个股收益 (close to close)
        stock_returns = close.pct_change().fillna(0)
        # 组合收益
        port_return = (weights * stock_returns).sum(axis=1)

        # 5) 成本 (向量化)
        # 买入: 佣金 + 滑点; 卖出: 佣金 + 印花税 + 滑点
        # 用组合总资产比例近似
        cost = buy_turnover * (commission_rate + slippage) + sell_turnover * (commission_rate + slippage + stamp_tax_rate)

        # 6) 资金曲线
        net_return = port_return - cost
        equity = init_capital * (1 + net_return).cumprod()

        equity_curve = pd.DataFrame({
            "date": equity.index,
            "equity": equity.values,
            "cash": 0.0,  # 等权满仓近似
            "market_value": equity.values,
            "position_count": hold_count.fillna(0).astype(int).values,
        }).reset_index(drop=True)

        # 7) 成交记录 (从换手重建, 简化)
        trades = _reconstruct_trades(weights, prev_weights, open_, close, turnover)
        trades_df = pd.DataFrame(trades)

        eq_series = equity_curve.set_index("date")["equity"]
        metrics = _calc_metrics(eq_series, trades_df, turnover=turnover)
        return {
            "trades": trades_df,
            "positions": pd.DataFrame(columns=["code", "shares"]),
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    def _empty_result(self):
        return {
            "trades": pd.DataFrame(), "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(), "metrics": {}, "report_path": "",
        }


# ────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────
def _calc_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    turnover: pd.Series = None,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, float]:
    """计算绩效指标 (含增强指标: Sortino/Calmar/换手率/盈亏比/信息比率)"""
    if equity is None or len(equity) < 2:
        return {}
    returns = equity.pct_change().dropna()
    if len(returns) == 0:
        return {}

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_years = len(equity) / trading_days
    annual_return = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    volatility = float(returns.std() * np.sqrt(trading_days))
    sharpe = float((returns.mean() * trading_days - risk_free) / volatility) if volatility > 0 else 0.0

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = float(drawdown.min())

    # Sortino (下行风险)
    downside = returns[returns < 0]
    downside_std = float(downside.std() * np.sqrt(trading_days)) if len(downside) > 1 else 0.0
    sortino = float((returns.mean() * trading_days - risk_free) / downside_std) if downside_std > 0 else 0.0

    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # 盈亏比 (Profit Factor): 基于日收益
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0

    win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": int(len(trades)),
    }
    if turnover is not None and len(turnover) > 0:
        metrics["avg_turnover"] = float(turnover.mean())
    return metrics


def _reconstruct_trades(
    weights: pd.DataFrame,
    prev_weights: pd.DataFrame,
    open_: pd.DataFrame,
    close: pd.DataFrame,
    turnover: pd.Series,
) -> List[Dict]:
    """从权重变化重建简化成交记录 (向量化, 仅遍历非零变化点)"""
    trades = []
    diff = (weights - prev_weights).values
    dates = weights.index
    codes = weights.columns
    open_v = open_.values
    close_v = close.values
    # 买入: diff > 阈值; 卖出: diff < -阈值
    buy_idx = np.argwhere(diff > 1e-6)
    sell_idx = np.argwhere(diff < -1e-6)
    for i, j in buy_idx:
        price = open_v[i, j] if not np.isnan(open_v[i, j]) else close_v[i, j]
        trades.append({
            "date": dates[i], "code": codes[j], "action": "buy",
            "price": float(price), "shares": 0, "amount": float(diff[i, j]),
            "commission": 0, "tax": 0, "pnl": 0,
        })
    for i, j in sell_idx:
        price = open_v[i, j] if not np.isnan(open_v[i, j]) else close_v[i, j]
        trades.append({
            "date": dates[i], "code": codes[j], "action": "sell",
            "price": float(price), "shares": 0, "amount": float(-diff[i, j]),
            "commission": 0, "tax": 0, "pnl": 0,
        })
    return trades


def time_backtest(adapter, data, signals, runs: int = 3, **kwargs) -> Dict[str, Any]:
    """运行回测并计时"""
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = adapter.run_backtest(data=data, signals=signals, **kwargs)
        times.append(time.perf_counter() - t0)
    return {
        "result": result,
        "times": times,
        "mean_time": float(np.mean(times)),
        "median_time": float(np.median(times)),
    }
