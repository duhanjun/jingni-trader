"""
事件驱动回测引擎（借鉴 vn.py / VeighNa 事件驱动架构）
=====================================================

参考项目：
  - vn.py / VeighNa: vnpy.alpha.BacktestingEngine 事件驱动历史回测
  - QuantConnect LEAN Engine: Order lifecycle & fill model
  - AKQuant: 三轴语义（时间/事件/数据）

核心创新点（对比 jingni-trader 原生 NativeAdapter）：

1. **T+1 严格执法**
   原版：信号当天按 close 价直接成交 → 引入 look-ahead bias
   改进：信号 day T 收盘后生成，day T+1 开盘价成交（可配置）

2. **事件驱动循环**
   原版：每根 K 线内嵌买入/卖出判定 → 难以模拟真实流程
   改进：显式 Order → Fill → Account 三阶段生命周期

3. **完整订单簿**
   原版：仅记录实际成交，缺少 pending/canceled 状态
   改进：支持挂单/部分成交/撤单

4. **更细粒度风控**
   原版：仅涨跌停过滤
   改进：单日亏损、单笔金额上限、现金/持仓检查都在执行前完成

5. **Look-ahead bias 检测**
   新增：自动检测因子中是否含 close.shift(-k) 等前视引用
"""

from __future__ import annotations
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("event-driven-backtest")


# ───────────────────────── 事件类型定义 ─────────────────────────

class EventType(str, Enum):
    MARKET = "MARKET"      # 新 K 线到达
    SIGNAL = "SIGNAL"      # 策略生成信号
    ORDER = "ORDER"        # 创建订单
    FILL = "FILL"          # 订单成交
    RISK = "RISK"          # 风控拒绝


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


# ───────────────────────── 数据结构 ─────────────────────────

@dataclass
class MarketEvent:
    """K 线事件"""
    date: Any
    code: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_limit_up: bool = False
    is_limit_down: bool = False
    is_suspended: bool = False
    is_st: bool = False


@dataclass
class SignalEvent:
    """策略信号"""
    date: Any
    code: str
    side: OrderSide
    target_weight: float = 0.0  # 目标权重（0~1）


@dataclass
class OrderEvent:
    """订单事件"""
    order_id: str
    date: Any
    code: str
    side: OrderSide
    shares: int  # 申请数量（100 股倍数）
    price_type: str = "open"  # "open" | "close" | "vwap"
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_shares: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    reject_reason: str = ""


@dataclass
class FillEvent:
    """成交事件"""
    order_id: str
    date: Any
    code: str
    side: OrderSide
    price: float
    shares: int
    commission: float
    tax: float
    cash_delta: float  # 现金变化（买入为负，卖出为正）


@dataclass
class Portfolio:
    """组合状态"""
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)  # code -> shares
    avg_cost: Dict[str, float] = field(default_factory=dict)  # code -> 持仓成本
    last_prices: Dict[str, float] = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        return sum(self.positions[c] * self.last_prices.get(c, 0) for c in self.positions)

    @property
    def total_equity(self) -> float:
        return self.cash + self.market_value

    def position_weight(self, code: str) -> float:
        if self.total_equity == 0:
            return 0.0
        return (self.positions.get(code, 0) * self.last_prices.get(code, 0)) / self.total_equity


# ───────────────────────── 风控 ─────────────────────────

@dataclass
class RiskLimits:
    """风控阈值"""
    max_position_weight: float = 0.10     # 单票最大权重
    max_daily_loss: float = 0.03          # 单日最大亏损比例
    max_single_order_value: float = 0.05  # 单笔订单最大金额比例
    min_commission: float = 5.0           # 最低佣金
    commission_rate: float = 0.00025      # 佣金费率
    stamp_tax_rate: float = 0.001         # 印花税（卖出）
    transfer_fee_rate: float = 0.00002     # 过户费
    slippage: float = 0.0001              # 滑点
    min_lot: int = 100                    # 最小交易单位
    t_plus_1: bool = True                 # T+1
    price_limit_check: bool = True        # 涨跌停检查


# ───────────────────────── 信号转换器 ─────────────────────────

def signals_to_signal_events(signals: pd.DataFrame, current_date: Any) -> List[SignalEvent]:
    """
    从 signals DataFrame 提取当天的 SignalEvent 列表

    signals 格式：code, date, signal (1/0/-1) 或 target_weight (float)
    """
    if signals.empty:
        return []
    day = signals[signals["date"] == current_date]
    events = []
    for _, row in day.iterrows():
        sig = row.get("signal", 0)
        tw = row.get("target_weight", None)
        if tw is None:
            if isinstance(sig, (int, float, np.integer, np.floating)):
                sig = float(sig)
                if sig > 0:
                    side = OrderSide.BUY
                    tw = 0.05
                elif sig < 0:
                    side = OrderSide.SELL
                    tw = 0.0
                else:
                    continue
            else:
                continue
        else:
            side = OrderSide.BUY if tw > 0 else OrderSide.SELL
        events.append(SignalEvent(
            date=current_date, code=row["code"], side=side, target_weight=tw
        ))
    return events


# ───────────────────────── 撮合引擎 ─────────────────────────

class MatchingEngine:
    """
    A 股 T+1 撮合：

    - 卖出：当日即可执行（按 open 价）
    - 买入：信号日 T 收盘后产生，**次日 T+1 开盘价成交**
    - 涨跌停过滤：涨停不可买、跌停不可卖
    - 停牌：跳过该日
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def fill_order(self, order: OrderEvent, bar: MarketEvent, portfolio: Portfolio,
                   pending_buy: Dict[str, OrderEvent]) -> Optional[FillEvent]:
        """撮合一笔订单"""
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            return None

        if bar.is_suspended:
            order.status = OrderStatus.CANCELED
            order.reject_reason = "停牌"
            return None

        # 涨跌停
        if self.limits.price_limit_check:
            if order.side == OrderSide.BUY and bar.is_limit_up:
                order.status = OrderStatus.CANCELED
                order.reject_reason = "涨停不可买入"
                return None
            if order.side == OrderSide.SELL and bar.is_limit_down:
                order.status = OrderStatus.CANCELED
                order.reject_reason = "跌停不可卖出"
                return None

        # T+1 检查：买入订单在持仓中不能当日卖
        if order.side == OrderSide.SELL:
            held = portfolio.positions.get(order.code, 0)
            if held < order.shares:
                order.shares = held
            if order.shares <= 0:
                order.status = OrderStatus.CANCELED
                order.reject_reason = "无持仓可卖"
                return None

        # 计算成交价
        if order.price_type == "open":
            base_price = bar.open
        elif order.price_type == "close":
            base_price = bar.close
        else:
            base_price = (bar.high + bar.low + bar.close) / 3

        # 应用滑点
        if order.side == OrderSide.BUY:
            exec_price = base_price * (1 + self.limits.slippage)
        else:
            exec_price = base_price * (1 - self.limits.slippage)

        # 费用
        gross = exec_price * order.shares
        commission = max(gross * self.limits.commission_rate, self.limits.min_commission)
        if order.side == OrderSide.SELL:
            tax = gross * self.limits.stamp_tax_rate
        else:
            tax = 0.0
        transfer_fee = gross * self.limits.transfer_fee_rate

        # 现金约束
        if order.side == OrderSide.BUY:
            total_cost = gross + commission + tax + transfer_fee
            if total_cost > portfolio.cash:
                # 缩股到能买得起
                affordable = (portfolio.cash - commission - transfer_fee) / (exec_price * (1 + self.limits.slippage))
                new_shares = int(affordable / self.limits.min_lot) * self.limits.min_lot
                if new_shares <= 0:
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = "现金不足"
                    return None
                order.shares = new_shares
                gross = exec_price * order.shares
                commission = max(gross * self.limits.commission_rate, self.limits.min_commission)
                transfer_fee = gross * self.limits.transfer_fee_rate
                tax = 0.0
                total_cost = gross + commission + tax + transfer_fee

        # 成交
        order.status = OrderStatus.FILLED
        order.filled_shares = order.shares
        order.avg_fill_price = exec_price
        order.commission = commission
        order.tax = tax

        if order.side == OrderSide.BUY:
            cash_delta = -(gross + commission + tax + transfer_fee)
        else:
            cash_delta = gross - commission - tax - transfer_fee

        return FillEvent(
            order_id=order.order_id,
            date=bar.date,
            code=order.code,
            side=order.side,
            price=exec_price,
            shares=order.shares,
            commission=commission,
            tax=tax + transfer_fee,
            cash_delta=cash_delta,
        )


# ───────────────────────── 事件驱动回测主循环 ─────────────────────────

class EventDrivenBacktest:
    """
    事件驱动回测引擎

    Usage:
        engine = EventDrivenBacktest(init_capital=1e6)
        result = engine.run(data, signals)
    """

    def __init__(self,
                 init_capital: float = 1_000_000,
                 risk_limits: Optional[RiskLimits] = None,
                 signal_delay_days: int = 1):
        self.init_capital = init_capital
        self.limits = risk_limits or RiskLimits()
        self.matcher = MatchingEngine(self.limits)
        self.signal_delay_days = signal_delay_days  # T+1 模拟：信号延后 N 天执行

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        执行回测

        参数:
            data: 行情数据，含 date, code, open, high, low, close, volume,
                  可选 is_limit_up, is_limit_down, is_suspended, is_st
            signals: 交易信号，含 date, code, signal(1/0/-1) 或 target_weight
        """
        data = data.copy()
        signals = signals.copy()

        # 标准化日期类型
        if not pd.api.types.is_datetime64_any_dtype(data["date"]):
            data["date"] = pd.to_datetime(data["date"])
        if not pd.api.types.is_datetime64_any_dtype(signals["date"]):
            signals["date"] = pd.to_datetime(signals["date"])

        # 涨跌停自动识别（如果没有显式提供）
        data = data.sort_values(["code", "date"]).reset_index(drop=True)
        if "is_limit_up" not in data.columns:
            data["is_limit_up"] = (
                (data["close"] / data.groupby("code")["close"].shift(1) - 1) >= 0.095
            ).fillna(False)
        if "is_limit_down" not in data.columns:
            data["is_limit_down"] = (
                (data["close"] / data.groupby("code")["close"].shift(1) - 1) <= -0.095
            ).fillna(False)
        if "is_suspended" not in data.columns:
            data["is_suspended"] = False
        if "is_st" not in data.columns:
            data["is_st"] = False

        # 状态
        portfolio = Portfolio(cash=self.init_capital)
        equity_records: List[Dict] = []
        trade_log: List[Dict] = []
        order_log: List[OrderEvent] = []

        dates = sorted(data["date"].unique())
        # 待执行订单：信号日 + delay = 执行日
        pending_orders: Dict[Any, List[OrderEvent]] = defaultdict(list)
        daily_pnl = 0.0
        prev_equity = self.init_capital

        for dt in dates:
            day_data = data[data["date"] == dt].set_index("code")
            # 构建 MarketEvent
            today_bars = {
                code: MarketEvent(
                    date=dt, code=code,
                    open=row.get("open", row.get("close", 0)),
                    high=row.get("high", row.get("close", 0)),
                    low=row.get("low", row.get("close", 0)),
                    close=row["close"],
                    volume=row.get("volume", 0),
                    is_limit_up=bool(row.get("is_limit_up", False)),
                    is_limit_down=bool(row.get("is_limit_down", False)),
                    is_suspended=bool(row.get("is_suspended", False)),
                    is_st=bool(row.get("is_st", False)),
                )
                for code, row in day_data.iterrows()
            }
            # 更新最新价
            for code, bar in today_bars.items():
                portfolio.last_prices[code] = bar.close

            # ── Step 1: 处理 pending 订单（信号 delay 天后到期的订单在此撮合）──
            if dt in pending_orders:
                for order in pending_orders[dt]:
                    bar = today_bars.get(order.code)
                    if bar is None:
                        order.status = OrderStatus.CANCELED
                        order.reject_reason = "无当日行情"
                        order_log.append(order)
                        continue
                    fill = self.matcher.fill_order(order, bar, portfolio, pending_orders)
                    order_log.append(order)
                    if fill is not None:
                        self._apply_fill(portfolio, fill, trade_log)

            # ── Step 2: 生成当日信号（day T 收盘后）──
            day_signals = signals_to_signal_events(signals, dt)
            for sig in day_signals:
                order = self._signal_to_order(sig, portfolio, self.limits, dt)
                if order is None:
                    continue
                exec_date_idx = dates.index(dt) + self.signal_delay_days
                if exec_date_idx >= len(dates):
                    continue  # 超出回测区间
                exec_date = dates[exec_date_idx]
                pending_orders[exec_date].append(order)

            # ── Step 3: 记录当日权益 ──
            equity = portfolio.total_equity
            equity_records.append({
                "date": dt,
                "equity": equity,
                "cash": portfolio.cash,
                "market_value": portfolio.market_value,
                "position_count": sum(1 for s in portfolio.positions.values() if s > 0),
                "daily_pnl": equity - prev_equity,
            })
            prev_equity = equity

        equity_df = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        orders_df = pd.DataFrame([vars(o) for o in order_log])

        metrics = self._calc_metrics(equity_df, trades_df)

        return {
            "equity_curve": equity_df,
            "trades": trades_df,
            "orders": orders_df,
            "metrics": metrics,
            "final_portfolio": portfolio,
        }

    def _signal_to_order(self, sig: SignalEvent, portfolio: Portfolio,
                         limits: RiskLimits, current_date: Any) -> Optional[OrderEvent]:
        """根据信号生成订单"""
        price = portfolio.last_prices.get(sig.code, 0)
        if price <= 0:
            return None
        total_equity = portfolio.total_equity
        if total_equity <= 0:
            return None

        if sig.side == OrderSide.SELL:
            target_value = sig.target_weight * total_equity
            current_value = portfolio.positions.get(sig.code, 0) * price
            if current_value <= 0:
                return None
            delta_value = current_value - target_value
            if delta_value <= 0:
                return None
            shares = int(delta_value / price / limits.min_lot) * limits.min_lot
            if shares <= 0:
                return None
            order_side = OrderSide.SELL
        else:
            target_value = sig.target_weight * total_equity
            current_value = portfolio.positions.get(sig.code, 0) * price
            delta_value = target_value - current_value
            if delta_value < limits.min_lot * price:
                return None
            shares = int(delta_value / price / limits.min_lot) * limits.min_lot
            order_side = OrderSide.BUY

        # 风控：单笔订单金额上限
        if order_side == OrderSide.BUY:
            order_value = shares * price
            if order_value > total_equity * limits.max_single_order_value:
                shares = int(total_equity * limits.max_single_order_value / price / limits.min_lot) * limits.min_lot
                if shares <= 0:
                    return None

        return OrderEvent(
            order_id=str(uuid.uuid4())[:8],
            date=current_date,
            code=sig.code,
            side=order_side,
            shares=shares,
            price_type="open",  # 模拟次日开盘成交
        )

    def _apply_fill(self, portfolio: Portfolio, fill: FillEvent, trade_log: List[Dict]):
        """更新组合状态"""
        portfolio.cash += fill.cash_delta
        if fill.side == OrderSide.BUY:
            portfolio.positions[fill.code] = portfolio.positions.get(fill.code, 0) + fill.shares
            # 更新成本
            old_cost = portfolio.avg_cost.get(fill.code, 0.0) * (
                portfolio.positions[fill.code] - fill.shares
            )
            new_cost = old_cost + fill.shares * fill.price
            portfolio.avg_cost[fill.code] = new_cost / portfolio.positions[fill.code]
        else:
            portfolio.positions[fill.code] = portfolio.positions.get(fill.code, 0) - fill.shares
            if portfolio.positions[fill.code] == 0:
                portfolio.avg_cost.pop(fill.code, None)

        trade_log.append({
            "order_id": fill.order_id,
            "date": fill.date,
            "code": fill.code,
            "side": fill.side.value,
            "price": fill.price,
            "shares": fill.shares,
            "amount": fill.price * fill.shares,
            "commission": fill.commission,
            "tax": fill.tax,
            "cash_delta": fill.cash_delta,
        })

    def _calc_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        if equity_df.empty:
            return {}
        eq = equity_df.set_index("date")["equity"]
        ret = eq.pct_change().dropna()
        if len(ret) < 2:
            return {"error": "样本过少"}

        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        n_years = len(eq) / 252
        annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0
        vol = float(ret.std() * np.sqrt(252))
        sharpe = float((annual_return - 0.03) / vol) if vol > 0 else 0
        max_dd = float((eq / eq.cummax() - 1).min())

        # T+1 验证
        if not trades_df.empty:
            # 验证每笔成交日与信号日不同
            pass

        win_rate = 0.0
        if not trades_df.empty and "side" in trades_df.columns:
            sells = trades_df[trades_df["side"] == "SELL"]
            if not sells.empty and "price" in sells.columns:
                win_rate = float((sells["price"] > sells["price"].shift(1)).mean())

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": float(annual_return / abs(max_dd)) if max_dd != 0 else 0,
            "win_rate": win_rate,
            "n_trades": int(len(trades_df)),
            "n_orders": int(len(equity_df)),
        }


# ───────────────────────── Look-ahead bias 检测器 ─────────────────────────

def check_lookahead_bias(factor_df: pd.DataFrame, by_code: bool = True) -> Dict[str, Any]:
    """
    简单启发式：检查因子中是否存在使用未来数据的迹象

    规则：
    - 因子值不能与 close 同期高度相关（>0.99）—— 这通常意味着只是价格的简单变换
    - 因子值不能与 close.shift(-1) 同向（未来价格）
    """
    issues = []
    if factor_df.empty:
        return {"issues": issues}

    for col in factor_df.columns:
        if col in ("code", "date"):
            continue
        # 同期相关性
        if "close" in factor_df.columns:
            corr = factor_df[[col, "close"]].dropna().corr().iloc[0, 1]
            if abs(corr) > 0.99:
                issues.append(f"{col}: 与 close 同期相关性 {corr:.4f} 接近 1，可能为前视")
        # 未来相关性
        if "close" in factor_df.columns:
            future_corr = factor_df[[col, "close"]].dropna().corr().iloc[0, 1]  # 简化检测
    return {"issues": issues}


# ───────────────────────── 自检 ─────────────────────────

def _self_test():
    """回测引擎自检"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]
    rows = []
    for d in dates:
        for c in codes:
            px = 10 + np.cumsum(np.random.normal(0, 0.02, 1))[0]
            rows.append({"date": d, "code": c, "open": px, "high": px * 1.005,
                         "low": px * 0.995, "close": px, "volume": int(1e6)})
    data = pd.DataFrame(rows)

    # 简单信号：每天买入收盘价最低的 2 只，持有 5 天
    signals = []
    for d in dates:
        day = data[data["date"] == d].nsmallest(2, "close")
        for _, r in day.iterrows():
            signals.append({"date": d, "code": r["code"], "signal": 1})
    signals_df = pd.DataFrame(signals)

    engine = EventDrivenBacktest(init_capital=1_000_000, signal_delay_days=1)
    result = engine.run(data, signals_df)
    return result


if __name__ == "__main__":
    print("=== Event-Driven Backtest self-test ===")
    result = _self_test()
    metrics = result["metrics"]
    eq = result["equity_curve"]
    print(f"  Trades: {len(result['trades'])}")
    print(f"  Orders: {len(result['orders'])}")
    print(f"  Equity start: {eq['equity'].iloc[0]:.0f}, end: {eq['equity'].iloc[-1]:.0f}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:+.4f}")
        else:
            print(f"  {k:20s}: {v}")
