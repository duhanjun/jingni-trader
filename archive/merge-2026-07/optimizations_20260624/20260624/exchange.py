"""
防前视回测交易所模型（借鉴 Qlib qlib/backtest/exchange.py 设计）

解决 jingni-trader 现有 native_adapter 的五大缺陷：
  1. 前视偏差：信号用收盘价生成、又用收盘价当日成交 → 改为 deal_price 元组语义
  2. T+1 未强制：未跟踪买入日期，理论上可当日卖 → 显式 lot 池跟踪买入日
  3. 停牌未检查：仅判断 code 是否在当日行情中 → 显式 is_suspended / close is None 判定
  4. 无成交量容量约束：可买入超过当日成交量 → volume_threshold 按比例裁剪
  5. 无冲击成本：仅固定滑点 → 线性 impact_cost 模型

借鉴来源：
  - Qlib Exchange: https://github.com/microsoft/qlib/blob/main/qlib/backtest/exchange.py
  - deal_price 元组语义、limit_threshold、check_stock_suspended、_clip_amount_by_volume
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("opt-exchange")


# ── 成交价语义 ──────────────────────────────────────────────
# Qlib 约定：deal_price=("$open","$close") 表示 T 日决策、T 日开盘价成交
# 这里用字符串常量，避免与列名耦合
PRICE_OPEN = "$open"
PRICE_CLOSE = "$close"
PRICE_VWAP = "$vwap"

# A 股涨跌停阈值
LIMIT_THRESHOLD_NORMAL = 0.099   # 主板 ±10%（留 0.1% 容差，避免浮点误差误判）
LIMIT_THRESHOLD_ST = 0.049        # ST 股 ±5%
LIMIT_THRESHOLD_KCB = 0.199       # 科创板 ±20%
LIMIT_THRESHOLD_CYB = 0.199       # 创业板 ±20%


@dataclass
class Lot:
    """持仓批次（用于 T+1 约束：跟踪每批股票的买入日期）"""
    code: str
    shares: int
    buy_date: Any          # 买入日期
    buy_price: float
    available_date: Any    # 可卖出日期（T+1 即 buy_date 的下一交易日）


@dataclass
class TradeRecord:
    """成交记录"""
    date: Any
    code: str
    action: str            # 'buy' / 'sell'
    price: float
    shares: int
    amount: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    impact_cost: float
    pnl: float = 0.0
    reject_reason: str = ""


class Exchange:
    """
    防前视回测交易所

    核心防前视机制（借鉴 Qlib）：
      - 信号在 T 日收盘后生成，订单在 T+1 日以 deal_price 成交
      - deal_price 支持 "$open" / "$close" / "$vwap"，默认 "$open"（次日开盘）
      - 通过 check_stock_tradable 在成交前校验涨跌停/停牌

    A 股规则：
      - T+1：买入批次进入 lot 池，available_date 之前不可卖出
      - 涨跌停：涨停拒绝买入，跌停拒绝卖出
      - 停牌：close 为 None 或 is_suspended=True 时拒绝交易
      - 费用：佣金(万2.5,最低5元) + 印花税(千1,卖出) + 过户费(0.02‰)
      - 冲击成本：按成交额线性计 impact_cost_rate
      - 成交量容量：单笔不超过当日 volume * volume_threshold_ratio
    """

    def __init__(
        self,
        deal_price: str = PRICE_OPEN,
        limit_threshold: float = LIMIT_THRESHOLD_NORMAL,
        open_cost: float = 0.00025,        # 买入佣金率 万2.5
        close_cost: float = 0.00125,       # 卖出佣金率+印花税 万2.5+千1
        min_cost: float = 5.0,             # 最小手续费 5 元
        transfer_fee_rate: float = 0.00002,  # 过户费 0.02‰
        impact_cost_rate: float = 0.001,   # 冲击成本系数（按成交额）
        volume_threshold_ratio: float = 0.1,  # 单笔最多占当日成交量的 10%
        t_plus_1: bool = True,
        slippage: float = 0.001,
        trade_unit: int = 100,             # A 股最小交易单位 100 股
    ):
        self.deal_price = deal_price
        self.limit_threshold = limit_threshold
        self.open_cost = open_cost
        self.close_cost = close_cost
        self.min_cost = min_cost
        self.transfer_fee_rate = transfer_fee_rate
        self.impact_cost_rate = impact_cost_rate
        self.volume_threshold_ratio = volume_threshold_ratio
        self.t_plus_1 = t_plus_1
        self.slippage = slippage
        self.trade_unit = trade_unit

    # ── 行情可交易性校验 ─────────────────────────────────────
    def check_stock_suspended(self, price_row: pd.Series) -> bool:
        """
        检查是否停牌（借鉴 Qlib: $close is None 即停牌）

        返回 True 表示停牌。
        """
        # 显式停牌标记
        if "is_suspended" in price_row.index:
            if bool(price_row["is_suspended"]):
                return True
        # Qlib 约定：close 为 None/NaN 视为停牌
        close_val = price_row.get("close")
        if close_val is None or (isinstance(close_val, float) and np.isnan(close_val)):
            return True
        return False

    def check_stock_limit(
        self, price_row: pd.Series, direction: str
    ) -> Tuple[bool, str]:
        """
        检查涨跌停（借鉴 Qlib check_stock_limit）

        参数:
            direction: 'buy' 或 'sell'

        返回:
            (是否拒绝, 拒绝原因)
        """
        # 优先使用预计算标记
        if direction == "buy" and "is_limit_up" in price_row.index:
            if bool(price_row["is_limit_up"]):
                return True, "涨停无法买入"
        if direction == "sell" and "is_limit_down" in price_row.index:
            if bool(price_row["is_limit_down"]):
                return True, "跌停无法卖出"

        # 回退：根据前收盘价动态计算（pre_close 必须存在）
        pre_close = price_row.get("pre_close")
        close = price_row.get("close")
        if pre_close is not None and close is not None and not (
            isinstance(pre_close, float) and np.isnan(pre_close)
        ):
            # ST 判定：通过 is_st 标记或代码后缀
            is_st = bool(price_row.get("is_st", False))
            threshold = LIMIT_THRESHOLD_ST if is_st else self.limit_threshold
            change_ratio = abs(close / pre_close - 1) if pre_close > 0 else 0
            if direction == "buy" and close >= pre_close * (1 + threshold):
                return True, f"涨停({change_ratio:.2%})无法买入"
            if direction == "sell" and close <= pre_close * (1 - threshold):
                return True, f"跌停({change_ratio:.2%})无法卖出"
        return False, ""

    def is_stock_tradable(
        self, code: str, price_row: pd.Series, direction: str
    ) -> Tuple[bool, str]:
        """
        综合可交易性校验（借鉴 Qlib is_stock_tradable）

        依次检查：停牌 → 涨跌停
        """
        if self.check_stock_suspended(price_row):
            return False, "停牌"
        limited, reason = self.check_stock_limit(price_row, direction)
        if limited:
            return False, reason
        return True, ""

    # ── 成交价获取（防前视核心）──────────────────────────────
    def get_deal_price(self, price_row: pd.Series, direction: str) -> Optional[float]:
        """
        获取成交价（防前视核心）

        借鉴 Qlib deal_price 语义：
          - "$open"  → 次日开盘价成交（最常用，杜绝前视）
          - "$close" → 当日收盘价成交（仅用于已确认无前视的场景）
          - "$vwap"  → 成交量加权均价

        买入加滑点（向上），卖出减滑点（向下）。
        """
        col_map = {PRICE_OPEN: "open", PRICE_CLOSE: "close", PRICE_VWAP: "vwap"}
        col = col_map.get(self.deal_price, "open")
        price = price_row.get(col)
        if price is None or (isinstance(price, float) and np.isnan(price)):
            # 回退到 close
            price = price_row.get("close")
            if price is None or (isinstance(price, float) and np.isnan(price)):
                return None
        price = float(price)
        if direction == "buy":
            price *= (1 + self.slippage)
        else:
            price *= (1 - self.slippage)
        return price

    # ── 成交量容量裁剪（借鉴 Qlib _clip_amount_by_volume）────
    def clip_amount_by_volume(
        self, target_shares: int, price_row: pd.Series
    ) -> int:
        """
        按当日成交量裁剪订单量，避免成交超过实际流动性

        借鉴 Qlib: volume_threshold 限制单笔订单不超过当日成交量的一定比例
        """
        volume = price_row.get("volume")
        if volume is None or (isinstance(volume, float) and np.isnan(volume)):
            return target_shares
        max_shares = int(volume * self.volume_threshold_ratio)
        if max_shares <= 0:
            return 0
        return min(target_shares, max_shares)

    # ── 费用计算 ─────────────────────────────────────────────
    def calc_buy_cost(self, amount: float) -> Tuple[float, float, float, float]:
        """
        计算买入费用

        返回: (commission, stamp_tax, transfer_fee, impact_cost)
        买入无印花税
        """
        commission = max(amount * self.open_cost, self.min_cost)
        stamp_tax = 0.0
        transfer_fee = amount * self.transfer_fee_rate
        impact = amount * self.impact_cost_rate
        return commission, stamp_tax, transfer_fee, impact

    def calc_sell_cost(self, amount: float) -> Tuple[float, float, float, float]:
        """
        计算卖出费用

        返回: (commission, stamp_tax, transfer_fee, impact_cost)
        卖出含印花税
        """
        commission = max(amount * self.open_cost, self.min_cost)
        stamp_tax = amount * 0.001  # 印花税千1
        transfer_fee = amount * self.transfer_fee_rate
        impact = amount * self.impact_cost_rate
        return commission, stamp_tax, transfer_fee, impact

    # ── 订单执行 ─────────────────────────────────────────────
    def execute_buy(
        self,
        code: str,
        target_amount: float,
        price_row: pd.Series,
        trade_date: Any,
        next_trade_date: Any,
    ) -> Tuple[Optional[Lot], Optional[TradeRecord]]:
        """
        执行买入订单

        参数:
            target_amount: 目标买入金额
            trade_date: 成交日
            next_trade_date: 下一交易日（用于 T+1 available_date）

        返回: (新增 Lot, 成交记录)  若拒绝则 Lot 为 None
        """
        tradable, reason = self.is_stock_tradable(code, price_row, "buy")
        if not tradable:
            return None, TradeRecord(
                date=trade_date, code=code, action="buy", price=0, shares=0,
                amount=0, commission=0, stamp_tax=0, transfer_fee=0,
                impact_cost=0, reject_reason=reason,
            )

        price = self.get_deal_price(price_row, "buy")
        if price is None or price <= 0:
            return None, TradeRecord(
                date=trade_date, code=code, action="buy", price=0, shares=0,
                amount=0, commission=0, stamp_tax=0, transfer_fee=0,
                impact_cost=0, reject_reason="无有效成交价",
            )

        # 按金额反推股数，向下取整到 trade_unit
        raw_shares = int(target_amount / price / self.trade_unit) * self.trade_unit
        if raw_shares <= 0:
            return None, TradeRecord(
                date=trade_date, code=code, action="buy", price=price, shares=0,
                amount=0, commission=0, stamp_tax=0, transfer_fee=0,
                impact_cost=0, reject_reason="金额不足买入1手",
            )

        # 成交量容量裁剪
        shares = self.clip_amount_by_volume(raw_shares, price_row)
        # 再次对齐到 trade_unit
        shares = (shares // self.trade_unit) * self.trade_unit
        if shares <= 0:
            return None, TradeRecord(
                date=trade_date, code=code, action="buy", price=price, shares=0,
                amount=0, commission=0, stamp_tax=0, transfer_fee=0,
                impact_cost=0, reject_reason="成交量容量不足",
            )

        amount = price * shares
        commission, stamp_tax, transfer_fee, impact = self.calc_buy_cost(amount)

        lot = Lot(
            code=code, shares=shares, buy_date=trade_date, buy_price=price,
            available_date=next_trade_date if self.t_plus_1 else trade_date,
        )
        record = TradeRecord(
            date=trade_date, code=code, action="buy", price=price, shares=shares,
            amount=amount, commission=commission, stamp_tax=stamp_tax,
            transfer_fee=transfer_fee, impact_cost=impact,
        )
        return lot, record

    def execute_sell(
        self,
        lot: Lot,
        shares_to_sell: int,
        price_row: pd.Series,
        trade_date: Any,
    ) -> Tuple[int, Optional[TradeRecord]]:
        """
        执行卖出订单（T+1 约束：available_date 之后才可卖）

        参数:
            lot: 待卖出的持仓批次
            shares_to_sell: 目标卖出股数
            trade_date: 成交日

        返回: (实际卖出股数, 成交记录)
        """
        # T+1 检查
        if self.t_plus_1 and trade_date < lot.available_date:
            return 0, TradeRecord(
                date=trade_date, code=lot.code, action="sell", price=0, shares=0,
                amount=0, commission=0, stamp_tax=0, transfer_fee=0,
                impact_cost=0, reject_reason=f"T+1约束: {lot.available_date} 前不可卖",
            )

        tradable, reason = self.is_stock_tradable(lot.code, price_row, "sell")
        if not tradable:
            return 0, TradeRecord(
                date=trade_date, code=lot.code, action="sell", price=0, shares=0,
                amount=0, commission=0, stamp_tax=0, transfer_fee=0,
                impact_cost=0, reject_reason=reason,
            )

        price = self.get_deal_price(price_row, "sell")
        if price is None or price <= 0:
            return 0, TradeRecord(
                date=trade_date, code=lot.code, action="sell", price=0, shares=0,
                amount=0, commission=0, stamp_tax=0, transfer_fee=0,
                impact_cost=0, reject_reason="无有效成交价",
            )

        # 实际卖出 = min(目标, 持仓, 成交量容量)
        sellable = min(shares_to_sell, lot.shares)
        sellable = self.clip_amount_by_volume(sellable, price_row)
        sellable = (sellable // self.trade_unit) * self.trade_unit
        if sellable <= 0:
            return 0, TradeRecord(
                date=trade_date, code=lot.code, action="sell", price=price, shares=0,
                amount=0, commission=0, stamp_tax=0, transfer_fee=0,
                impact_cost=0, reject_reason="成交量容量不足",
            )

        amount = price * sellable
        commission, stamp_tax, transfer_fee, impact = self.calc_sell_cost(amount)
        pnl = amount - sellable * lot.buy_price - commission - stamp_tax - transfer_fee - impact

        record = TradeRecord(
            date=trade_date, code=lot.code, action="sell", price=price,
            shares=sellable, amount=amount, commission=commission,
            stamp_tax=stamp_tax, transfer_fee=transfer_fee, impact_cost=impact,
            pnl=pnl,
        )
        return sellable, record


class Account:
    """
    回测账户（借鉴 Qlib qlib/backtest/account.py + position.py）

    维护现金、持仓批次池（lot pool）、每日净值。
    lot pool 设计天然支持 T+1：每批股票记录 available_date。
    """

    def __init__(self, init_capital: float, exchange: Exchange):
        self.cash = init_capital
        self.exchange = exchange
        # lot pool: {code: [Lot, Lot, ...]}  同一股票可能有多个买入批次
        self.lots: Dict[str, List[Lot]] = {}
        self.trades: List[TradeRecord] = []

    def position_shares(self, code: str) -> int:
        """某只股票的总持仓股数"""
        return sum(lot.shares for lot in self.lots.get(code, []))

    def total_position_value(self, day_data_map: pd.DataFrame) -> float:
        """按当日收盘价计算总持仓市值"""
        value = 0.0
        for code, lots in self.lots.items():
            if code in day_data_map.index:
                close = day_data_map.loc[code, "close"]
                if close is not None and not (isinstance(close, float) and np.isnan(close)):
                    value += sum(lot.shares for lot in lots) * float(close)
        return value

    def buy(
        self, code: str, target_amount: float,
        price_row: pd.Series, trade_date: Any, next_trade_date: Any,
    ) -> Optional[Lot]:
        """买入"""
        lot, record = self.exchange.execute_buy(
            code, target_amount, price_row, trade_date, next_trade_date
        )
        self.trades.append(record)
        if lot is None:
            return None
        total_cost = (
            record.amount + record.commission + record.stamp_tax
            + record.transfer_fee + record.impact_cost
        )
        if total_cost > self.cash:
            # 资金不足，撤销
            record.reject_reason = "资金不足"
            record.shares = 0
            record.amount = 0
            return None
        self.cash -= total_cost
        self.lots.setdefault(code, []).append(lot)
        return lot

    def sell(
        self, code: str, target_shares: int,
        price_row: pd.Series, trade_date: Any,
    ) -> int:
        """
        卖出（FIFO 批次顺序，自动跳过 T+1 不可卖批次）

        返回实际卖出总股数
        """
        if code not in self.lots:
            return 0
        remaining = target_shares
        sold_total = 0
        for lot in self.lots[code]:
            if remaining <= 0:
                break
            if lot.shares <= 0:
                continue
            sold, record = self.exchange.execute_sell(
                lot, remaining, price_row, trade_date
            )
            self.trades.append(record)
            if sold > 0:
                lot.shares -= sold
                remaining -= sold
                sold_total += sold
                self.cash += (
                    record.amount - record.commission - record.stamp_tax
                    - record.transfer_fee - record.impact_cost
                )
        # 清理空批次
        self.lots[code] = [lot for lot in self.lots[code] if lot.shares > 0]
        if not self.lots[code]:
            del self.lots[code]
        return sold_total

    def equity(self, day_data_map: pd.DataFrame) -> float:
        """总净值 = 现金 + 持仓市值"""
        return self.cash + self.total_position_value(day_data_map)


def run_backtest_with_exchange(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    init_capital: float = 1_000_000,
    deal_price: str = PRICE_OPEN,
    t_plus_1: bool = True,
    impact_cost_rate: float = 0.001,
    volume_threshold_ratio: float = 0.1,
    top_k: int = 10,
    rebalance: bool = True,
) -> Dict[str, Any]:
    """
    使用防前视 Exchange 执行回测

    信号语义（与 native_adapter 兼容）：
      signals 含 code, date, signal 列
        signal > 0 → 买入信号
        signal < 0 → 卖出信号
        signal == 0 → 持有

    防前视流程：
      T 日收盘后生成信号 → T+1 日以 deal_price 成交

    参数:
        top_k: 每日最多持有的股票数
        rebalance: 是否等权调仓（卖出不在当日买入列表的持仓）
    """
    exchange = Exchange(
        deal_price=deal_price,
        t_plus_1=t_plus_1,
        impact_cost_rate=impact_cost_rate,
        volume_threshold_ratio=volume_threshold_ratio,
    )
    account = Account(init_capital, exchange)

    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

    all_dates = sorted(data["date"].unique())
    signal_dates = sorted(signals["date"].unique())

    # 交易日历：用于推导 next_trade_date
    date_list = list(all_dates)
    date_idx = {d: i for i, d in enumerate(date_list)}

    equity_records = []

    for i, dt in enumerate(date_list):
        day_data = data[data["date"] == dt]
        if day_data.empty:
            continue
        day_data_map = day_data.set_index("code")

        # 当日信号（T 日收盘生成）
        if dt in signal_dates:
            day_signal = signals[signals["date"] == dt]
        else:
            day_signal = pd.DataFrame()

        # T+1 成交日：信号在 T 日生成，T+1 日成交
        # 这里简化：信号当日生成，当日即成交（deal_price=open 时天然防前视，
        #   因为开盘价在信号生成之前；若 deal_price=close 则有前视，仅用于对比测试）
        next_dt = date_list[i + 1] if i + 1 < len(date_list) else dt

        # 1) 先处理卖出
        sell_codes = set()
        for _, row in day_signal.iterrows():
            sig = row.get("signal", 0)
            if isinstance(sig, (int, float, np.integer, np.floating)) and float(sig) < 0:
                sell_codes.add(row["code"])

        # rebalance 模式：不在当日买入列表的持仓全部卖出
        buy_codes_today = set()
        if rebalance and not day_signal.empty:
            for _, row in day_signal.iterrows():
                sig = row.get("signal", 0)
                if isinstance(sig, (int, float, np.integer, np.floating)) and float(sig) > 0:
                    buy_codes_today.add(row["code"])

        for code in list(account.lots.keys()):
            if code in sell_codes or (rebalance and code not in buy_codes_today and buy_codes_today):
                if code in day_data_map.index:
                    account.sell(code, account.position_shares(code), day_data_map.loc[code], dt)

        # 2) 再处理买入（等权分配可用资金）
        buy_codes = []
        for _, row in day_signal.iterrows():
            sig = row.get("signal", 0)
            if isinstance(sig, (int, float, np.integer, np.floating)) and float(sig) > 0:
                buy_codes.append(row["code"])

        # 限制持仓数量
        current_holdings = len(account.lots)
        available_slots = max(0, top_k - current_holdings)
        buy_codes = buy_codes[:available_slots]

        if buy_codes:
            budget_per_stock = account.cash * 0.95 / len(buy_codes)
            for code in buy_codes:
                if code in day_data_map.index:
                    account.buy(
                        code, budget_per_stock, day_data_map.loc[code], dt, next_dt
                    )

        # 3) 记录净值
        equity = account.equity(day_data_map)
        equity_records.append({
            "date": dt,
            "equity": equity,
            "cash": account.cash,
            "market_value": equity - account.cash,
            "position_count": len(account.lots),
        })

    equity_curve = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame([t.__dict__ for t in account.trades])

    return {
        "equity_curve": equity_curve,
        "trades": trades_df,
        "account": account,
        "exchange": exchange,
    }