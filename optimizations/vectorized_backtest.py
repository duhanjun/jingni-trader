"""
向量化回测引擎（验证用）
========================
OPTIMIZATION 1: 用预分组字典消除 O(n²) 布尔掩码。

借鉴来源：
- VectorBT 的“向量化优先、避免朴素循环”哲学
- 但保留逐日循环（路径依赖逻辑：T+1、资金扣减、持仓状态无法纯向量化，且不引入 numba）

本文件提供两个实现，逻辑完全一致，仅“取当日数据”方式不同：
- run_original_backtest: 复刻 native_adapter.py 的原始 O(n²) 布尔掩码写法
- VectorizedBacktest.run_backtest: 预先一次性构建 data_by_date / signals_by_date 字典，
  循环体内仅做 O(1) 字典查找，消除重复布尔掩码

两者必须产出 IDENTICAL 的 equity_curve 与 trades（浮点误差 ~1e-10）。

原始交易逻辑（必须逐字镜像，来自 native_adapter.py）：
- data/signals 按 ['date','code'] 排序；dates = sorted(unique signal dates)
- cash=init_capital, positions={}, 逐 dt:
  - day_data_map = 当日数据 set_index('code')
  - 卖出阶段: signal<0 的 code；无持仓/跌停跳过；price=close；shares=持仓；
    sell_amount=price*shares；commission=max(sell_amount*rate,5)；tax=sell_amount*stamp；
    cost=commission+tax；cash+=sell_amount-cost；记录 trade(pnl=sell_amount-cost)；持仓清0
  - 买入阶段: signal>0 的 buy_codes；n_buy=len；budget=cash*0.95/n_buy；
    涨停跳过；price=close*(1+slippage)；shares=int(budget/price/100)*100；
    若 shares<=0 跳过；buy_amount=price*shares；commission=max(buy_amount*rate,5)；
    cost=buy_amount+commission；若 cost>cash 则 shares=int(cash*0.98/price/100)*100 重算；
    cash-=cost；positions[code]+=shares；记录 trade(pnl=-buy_amount-commission)
  - market_value = sum(持仓股数 * 当日 close)（仅当日有数据的持仓计入）
  - total_equity=cash+market_value；记录 equity
- equity_curve=DataFrame(equity_records)；trades_df=DataFrame(trades)
- metrics = calc_all_metrics(eq_series, trades_df)
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime


# -------------------------------------------------------------------
# 绩效指标计算（忠实复刻 backtest-engine base/base_backtest.py 的 BaseBacktestMetrics）
# 复制到此处以保证验证代码自包含，不依赖 skills 包的相对导入。
# -------------------------------------------------------------------
class _Metrics:
    """与 BaseBacktestMetrics 行为一致的轻量复刻（仅供验证用）"""

    @staticmethod
    def calc_total_return(equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    @staticmethod
    def calc_annual_return(equity_curve: pd.Series, trading_days: int = 252) -> float:
        if len(equity_curve) < 2:
            return 0.0
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        n_years = len(equity_curve) / trading_days
        if n_years <= 0:
            return 0.0
        return float(total_return ** (1 / n_years) - 1)

    @staticmethod
    def calc_volatility(returns: pd.Series, trading_days: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(trading_days))

    @staticmethod
    def calc_sharpe(returns: pd.Series, risk_free: float = 0.03, trading_days: int = 252) -> float:
        vol = _Metrics.calc_volatility(returns, trading_days)
        if vol == 0:
            return 0.0
        ann_return = returns.mean() * trading_days
        return float((ann_return - risk_free) / vol)

    @staticmethod
    def calc_max_drawdown(equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return float(drawdown.min())

    @staticmethod
    def calc_calmar(equity_curve: pd.Series, trading_days: int = 252) -> float:
        ann_return = _Metrics.calc_annual_return(equity_curve, trading_days)
        mdd = abs(_Metrics.calc_max_drawdown(equity_curve))
        if mdd == 0:
            return 0.0
        return float(ann_return / mdd)

    @staticmethod
    def calc_win_rate(trades: pd.DataFrame) -> float:
        # 注意：这里复刻的是“原始有 bug”的胜率（统计全部 trade，含买入）
        if trades.empty:
            return 0.0
        winning = (trades["pnl"] > 0).sum()
        total = len(trades)
        return float(winning / total) if total > 0 else 0.0

    @staticmethod
    def calc_sortino(returns: pd.Series, risk_free: float = 0.03, trading_days: int = 252) -> float:
        negative_returns = returns[returns < 0]
        if len(negative_returns) < 2:
            return 0.0
        downside_std = negative_returns.std() * np.sqrt(trading_days)
        if downside_std == 0:
            return 0.0
        ann_return = returns.mean() * trading_days
        return float((ann_return - risk_free) / downside_std)

    @staticmethod
    def calc_all_metrics(equity_curve: pd.Series, trades: pd.DataFrame,
                         risk_free: float = 0.03, trading_days: int = 252) -> Dict[str, Any]:
        returns = equity_curve.pct_change().dropna()
        return {
            "total_return": _Metrics.calc_total_return(equity_curve),
            "annual_return": _Metrics.calc_annual_return(equity_curve, trading_days),
            "volatility": _Metrics.calc_volatility(returns, trading_days),
            "sharpe_ratio": _Metrics.calc_sharpe(returns, risk_free, trading_days),
            "max_drawdown": _Metrics.calc_max_drawdown(equity_curve),
            "calmar_ratio": _Metrics.calc_calmar(equity_curve, trading_days),
            "sortino_ratio": _Metrics.calc_sortino(returns, risk_free, trading_days),
            "win_rate": _Metrics.calc_win_rate(trades),
            "total_trades": len(trades),
            "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


def _empty_result() -> Dict[str, Any]:
    return {
        "trades": pd.DataFrame(),
        "positions": pd.DataFrame(),
        "equity_curve": pd.DataFrame(),
        "metrics": {},
        "report_path": "",
    }


# -------------------------------------------------------------------
# 原始实现：逐字复刻 native_adapter.py 的 O(n²) 布尔掩码写法（用于对比基准）
# -------------------------------------------------------------------
def run_original_backtest(
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
    """原始 O(n²) 布尔掩码回测（与 native_adapter.run_backtest 逻辑完全一致）"""
    if data.empty or signals.empty:
        return _empty_result()

    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

    dates = sorted(signals["date"].unique())
    if not dates:
        return _empty_result()

    cash = init_capital
    positions: Dict[str, int] = {}
    equity_records = []
    trades = []

    for dt in dates:
        # ---- 原始写法：每次循环都做两次 O(n) 布尔掩码 ----
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
            if isinstance(sig, (int, float, np.integer, np.floating)):
                sig = float(sig)
                if sig > 0:
                    buy_codes.append(code)
                elif sig < 0:
                    sell_codes.append(code)

        # 卖出阶段
        for code in sell_codes:
            if code not in positions or positions[code] <= 0:
                continue
            if code not in day_data_map.index:
                continue
            price_row = day_data_map.loc[code]
            if price_limit and price_row.get("is_limit_down", False):
                continue
            price = price_row["close"]
            shares = positions[code]
            sell_amount = price * shares
            commission = max(sell_amount * commission_rate, 5)
            tax = sell_amount * stamp_tax_rate
            cost = commission + tax
            cash += sell_amount - cost
            trades.append({
                "date": dt, "code": code, "action": "sell",
                "price": price, "shares": shares, "amount": sell_amount,
                "commission": commission, "tax": tax, "pnl": sell_amount - cost,
            })
            positions[code] = 0

        # 买入阶段
        if buy_codes:
            n_buy = len(buy_codes)
            budget_per_stock = cash * 0.95 / n_buy
            for code in buy_codes:
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                if price_limit and price_row.get("is_limit_up", False):
                    continue
                price = price_row["close"] * (1 + slippage)
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
                    "date": dt, "code": code, "action": "buy",
                    "price": price, "shares": shares, "amount": buy_amount,
                    "commission": commission, "tax": 0, "pnl": -buy_amount - commission,
                })

        # 市值
        market_value = 0
        for code, shares in list(positions.items()):
            if shares <= 0:
                continue
            if code in day_data_map.index:
                market_value += shares * day_data_map.loc[code, "close"]
        total_equity = cash + market_value

        equity_records.append({
            "date": dt,
            "equity": total_equity,
            "cash": cash,
            "market_value": market_value,
            "position_count": sum(1 for s in positions.values() if s > 0),
        })

    equity_curve = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trades)

    if equity_curve.empty:
        return _empty_result()

    eq_series = equity_curve.set_index("date")["equity"]
    metrics = _Metrics.calc_all_metrics(eq_series, trades_df)

    return {
        "trades": trades_df,
        "positions": pd.DataFrame(list(positions.items()), columns=["code", "shares"]),
        "equity_curve": equity_curve,
        "metrics": metrics,
        "report_path": "",
    }


# -------------------------------------------------------------------
# 优化实现：预先一次性构建按日字典，循环体内仅 O(1) 查找
# -------------------------------------------------------------------
class VectorizedBacktest:
    """
    向量化回测引擎。

    核心优化：在进入逐日循环前，一次性把 data / signals 按 date 分组并构建字典：
        data_by_date[dt]    = 当日 DataFrame（已 set_index('code')）
        signals_by_date[dt] = 当日 signal DataFrame
    循环体内用字典查找替代原来的 signals[signals['date']==dt] / data[data['date']==dt]
    两次 O(n) 布尔掩码，把 O(n_days * n_rows) 的掩码开销降为 O(n_rows) 的一次性分组。

    逐日循环本身保留（路径依赖：cash/positions 状态机无法纯向量化）。
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
            return _empty_result()

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        dates = sorted(signals["date"].unique())
        if not dates:
            return _empty_result()

        # ---- 关键优化：一次性预分组，避免循环内重复布尔掩码 ----
        # data_by_date: {dt: 当日数据 set_index('code')}，等价于原始每次 day_data.set_index('code')
        data_by_date: Dict[Any, pd.DataFrame] = {
            dt: group.set_index("code")
            for dt, group in data.groupby("date", sort=False)
        }
        # signals_by_date: {dt: 当日 signal DataFrame}，行内顺序保持原始（已按 date,code 排序）
        signals_by_date: Dict[Any, pd.DataFrame] = {
            dt: group for dt, group in signals.groupby("date", sort=False)
        }

        cash = init_capital
        positions: Dict[str, int] = {}
        equity_records = []
        trades = []

        for dt in dates:
            # O(1) 字典查找替代 O(n) 布尔掩码
            day_data_map = data_by_date.get(dt)
            if day_data_map is None or day_data_map.empty:
                continue

            day_signal = signals_by_date.get(dt)
            sell_codes = []
            buy_codes = []
            if day_signal is not None:
                for _, row in day_signal.iterrows():
                    code = row["code"]
                    sig = row.get("signal", 0)
                    if isinstance(sig, (int, float, np.integer, np.floating)):
                        sig = float(sig)
                        if sig > 0:
                            buy_codes.append(code)
                        elif sig < 0:
                            sell_codes.append(code)

            # 卖出阶段（逻辑与原始完全一致）
            for code in sell_codes:
                if code not in positions or positions[code] <= 0:
                    continue
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                if price_limit and price_row.get("is_limit_down", False):
                    continue
                price = price_row["close"]
                shares = positions[code]
                sell_amount = price * shares
                commission = max(sell_amount * commission_rate, 5)
                tax = sell_amount * stamp_tax_rate
                cost = commission + tax
                cash += sell_amount - cost
                trades.append({
                    "date": dt, "code": code, "action": "sell",
                    "price": price, "shares": shares, "amount": sell_amount,
                    "commission": commission, "tax": tax, "pnl": sell_amount - cost,
                })
                positions[code] = 0

            # 买入阶段（逻辑与原始完全一致）
            if buy_codes:
                n_buy = len(buy_codes)
                budget_per_stock = cash * 0.95 / n_buy
                for code in buy_codes:
                    if code not in day_data_map.index:
                        continue
                    price_row = day_data_map.loc[code]
                    if price_limit and price_row.get("is_limit_up", False):
                        continue
                    price = price_row["close"] * (1 + slippage)
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
                        "date": dt, "code": code, "action": "buy",
                        "price": price, "shares": shares, "amount": buy_amount,
                        "commission": commission, "tax": 0, "pnl": -buy_amount - commission,
                    })

            # 市值（逻辑与原始完全一致）
            market_value = 0
            for code, shares in list(positions.items()):
                if shares <= 0:
                    continue
                if code in day_data_map.index:
                    market_value += shares * day_data_map.loc[code, "close"]
            total_equity = cash + market_value

            equity_records.append({
                "date": dt,
                "equity": total_equity,
                "cash": cash,
                "market_value": market_value,
                "position_count": sum(1 for s in positions.values() if s > 0),
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return _empty_result()

        eq_series = equity_curve.set_index("date")["equity"]
        metrics = _Metrics.calc_all_metrics(eq_series, trades_df)

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(list(positions.items()), columns=["code", "shares"]),
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_test_data

    data, signals = generate_test_data(n_stocks=20, n_days=120, seed=1)
    r_orig = run_original_backtest(data, signals)
    r_vec = VectorizedBacktest().run_backtest(data, signals)

    eq_o = r_orig["equity_curve"]["equity"].values
    eq_v = r_vec["equity_curve"]["equity"].values
    print("orig final equity:", eq_o[-1])
    print("vec  final equity:", eq_v[-1])
    print("max abs diff:", np.max(np.abs(eq_o - eq_v)))
    print("trades equal:", len(r_orig["trades"]) == len(r_vec["trades"]))
