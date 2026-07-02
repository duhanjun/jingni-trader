"""
增强回测引擎
借鉴: quant-stream Pathway-based streaming backtest engine

核心改进:
1. 前视偏差防护: 信号使用 t 时刻数据，交易在 t+1 时刻执行
2. 缺失价格处理: 停牌股票使用 last_known_price
3. 资本预留: cost_reserve 确保资金覆盖佣金税费
4. 涨跌停过滤: 涨跌停日无法成交时跳过
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .calendar import TradingCalendar
from .price_tracker import PriceTracker


@dataclass
class TradeRecord:
    code: str
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    signal: int
    execution_price: float
    quantity: int
    trade_value: float
    commission: float
    stamp_tax: float


@dataclass
class BacktestConfig:
    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025
    stamp_tax_rate: float = 0.001
    min_commission: float = 5.0
    slippage: float = 0.0001
    t_plus_1: bool = True
    price_limit: bool = True
    cost_reserve: float = 0.02
    max_single_weight: float = 0.10
    min_lot: int = 100
    short_funding_rate: float = 0.0


class EnhancedBacktestEngine:
    """增强回测引擎"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.calendar: Optional[TradingCalendar] = None
        self.price_tracker = PriceTracker()
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []
        self.positions: Dict[str, int] = {}
        self.cash: float = self.config.init_capital
        self._nav: float = self.config.init_capital

    def run(self, price_data: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        self.calendar = TradingCalendar.from_dataframe(price_data)
        price_data = price_data.sort_values(["code", "date"])
        signals = signals.sort_values(["code", "date"])

        signal_dates = sorted(signals["date"].unique())
        exec_map = self.calendar.build_execution_map([pd.Timestamp(d) for d in signal_dates])

        for signal_date in signal_dates:
            signal_date = pd.Timestamp(signal_date)
            self._update_daily_nav(price_data, signal_date)

            exec_date = exec_map.get(signal_date) if self.config.t_plus_1 else signal_date
            if exec_date is None:
                continue

            day_signals = signals[signals["date"] == signal_date]
            self._execute_trades(day_signals, price_data, signal_date, exec_date)

        return {
            "metrics": self._calculate_metrics(),
            "equity_curve": pd.DataFrame(self.equity_curve),
            "trades": self.trades,
            "final_nav": self._nav,
            "final_cash": self.cash,
        }

    def _update_daily_nav(self, price_data: pd.DataFrame, date: pd.Timestamp):
        day_data = price_data[price_data["date"] == date]
        total_value = self.cash

        for code, shares in list(self.positions.items()):
            if shares == 0:
                continue
            cd = day_data[day_data["code"] == code]
            if not cd.empty:
                price = cd["close"].iloc[0]
                self.price_tracker.update(code, price)
            else:
                price = self.price_tracker.get_price(code)
            if not np.isnan(price):
                total_value += shares * price

        self._nav = total_value
        self.equity_curve.append({"date": date, "equity": total_value, "cash": self.cash})

    def _execute_trades(self, signals, price_data, signal_date, exec_date):
        exec_data = price_data[price_data["date"] == exec_date]
        for _, row in signals.iterrows():
            code = row["code"]
            signal = row.get("signal", 0)
            if signal == 0:
                continue

            ce = exec_data[exec_data["code"] == code]
            if ce.empty:
                continue
            exec_price = ce["close"].iloc[0]

            # 涨跌停检查
            if self.config.price_limit:
                is_limit_up = ce["is_limit_up"].iloc[0] if "is_limit_up" in ce.columns else False
                is_limit_down = ce["is_limit_down"].iloc[0] if "is_limit_down" in ce.columns else False
                if (signal > 0 and is_limit_up) or (signal < 0 and is_limit_down):
                    continue

            # 滑点
            exec_price *= (1 + self.config.slippage) if signal > 0 else (1 - self.config.slippage)

            if signal > 0:
                max_val = self.cash * (1 - self.config.cost_reserve) * self.config.max_single_weight
                qty = int(max_val / exec_price / self.config.min_lot) * self.config.min_lot
                if qty <= 0:
                    continue
                trade_value = qty * exec_price
                commission = max(trade_value * self.config.commission_rate, self.config.min_commission)
                if trade_value + commission > self.cash:
                    qty = int((self.cash * 0.98) / exec_price / self.config.min_lot) * self.config.min_lot
                    if qty <= 0:
                        continue
                    trade_value = qty * exec_price
                    commission = max(trade_value * self.config.commission_rate, self.config.min_commission)
                self.cash -= trade_value + commission
                self.positions[code] = self.positions.get(code, 0) + qty
                stamp_tax = 0.0
            else:
                current = self.positions.get(code, 0)
                if current <= 0:
                    continue
                qty = current
                trade_value = qty * exec_price
                commission = max(trade_value * self.config.commission_rate, self.config.min_commission)
                stamp_tax = trade_value * self.config.stamp_tax_rate
                self.cash += trade_value - commission - stamp_tax
                del self.positions[code]

            self.price_tracker.update(code, exec_price)
            self.trades.append(TradeRecord(code, signal_date, exec_date, signal, exec_price, qty, trade_value if signal > 0 else -trade_value, commission, stamp_tax))

    def _calculate_metrics(self) -> Dict[str, float]:
        if not self.equity_curve:
            return {}
        df = pd.DataFrame(self.equity_curve).set_index("date")
        returns = df["equity"].pct_change().dropna()
        if len(returns) < 2:
            return {}

        total_return = float(df["equity"].iloc[-1] / df["equity"].iloc[0] - 1)
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        volatility = float(returns.std() * np.sqrt(252))
        max_dd = float((df["equity"] / df["equity"].cummax() - 1).min())
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        sortino_vol = float(returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else volatility
        sortino = (annual_return - 0.03) / sortino_vol if sortino_vol > 0 else 0
        win_rate = float((returns > 0).mean())
        pf_num = returns[returns > 0].sum()
        pf_den = abs(returns[returns < 0].sum())
        profit_factor = float(pf_num / pf_den) if pf_den != 0 else 0

        return {
            "total_return": total_return,
            "annual_return": float(annual_return),
            "volatility": volatility,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_dd,
            "calmar_ratio": float(annual_return / abs(max_dd)) if max_dd != 0 else 0,
            "sortino_ratio": float(sortino),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": len(self.trades),
            "n_trading_days": n_days,
        }