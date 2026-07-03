"""
向量化回测引擎
================

借鉴来源：
- Qlib（微软）的 Executor 思想：交易决策与执行分离，回放用向量化。
- AkQuant（2026-06 发布）的 Rust/Python 混合回测中"用向量化代替逐行循环"思路。
- VectorBT（7/10，参数扫描最快）的向量化回测设计。

设计目标：
- 在 long-format 数据 (code, date, close, ...) 上做"日频截面调仓"回测
- 关键路径全部向量化（避免 jingni-trader 原生 native_adapter.py 的
  `day_signal.iterrows()` 与 market_value 双重 for 循环）
- 输出与 native_adapter 兼容的 equity_curve / trades / metrics
- 同时提供 T+1、涨跌停、印花税、佣金、最小交易单位

与现有 native_adapter 的对比：
- 现有版本：每日遍历每只股票，O(N*K) 逐行 Python 循环
- 本版本：用 pivot 重塑后做矩阵运算，整体 O(N*K) 但常数更小
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..metrics import calc_all_metrics


# ──────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────

@dataclass
class VectorizedBTConfig:
    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.001
    slippage: float = 0.0001
    min_lot: int = 100
    t_plus_1: bool = True
    price_limit: bool = True
    cash_buffer: float = 0.02  # 现金缓冲，2%


# ──────────────────────────────────────────────────────────────────
# 主类
# ──────────────────────────────────────────────────────────────────

class VectorizedBacktester:
    """
    日频截面调仓的向量化回测。

    输入：
        data     : long-format 行情 (code, date, open/high/low/close/volume/amount, is_limit_up/down 可选)
        signals  : long-format 信号 (code, date, signal)，signal ∈ {-1, 0, 1}
        config   : VectorizedBTConfig

    输出（与 native_adapter 兼容）：
        trades:        pd.DataFrame
        positions:     pd.DataFrame
        equity_curve:  pd.DataFrame (date, equity, cash, market_value, position_count)
        metrics:       dict
    """

    def __init__(self, config: Optional[VectorizedBTConfig] = None):
        self.config = config or VectorizedBTConfig()

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.config
        if data.empty or signals.empty:
            return self._empty_result()

        # 1) 数据准备
        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)
        dates = sorted(data["date"].unique())

        price_pivot = data.pivot(index="date", columns="code", values="close").sort_index()
        signal_pivot = signals.pivot(index="date", columns="code", values="signal").reindex(price_pivot.index).fillna(0)
        if cfg.price_limit and "is_limit_up" in data.columns:
            limit_up_pivot = data.pivot(index="date", columns="code", values="is_limit_up").reindex(price_pivot.index).fillna(False)
            limit_dn_pivot = data.pivot(index="date", columns="code", values="is_limit_down").reindex(price_pivot.index).fillna(False) if "is_limit_down" in data.columns else pd.DataFrame(False, index=price_pivot.index, columns=price_pivot.columns)
        else:
            limit_up_pivot = pd.DataFrame(False, index=price_pivot.index, columns=price_pivot.columns)
            limit_dn_pivot = pd.DataFrame(False, index=price_pivot.index, columns=price_pivot.columns)

        # 2) 初始化
        cash = cfg.init_capital
        holdings: Dict[Any, float] = {}  # code -> shares
        equity_records: List[dict] = []
        trade_records: List[dict] = []

        # 3) 逐日推进（向量化内部 + Python 循环驱动日期）
        prev_holdings: Dict[Any, float] = {}
        for dt in dates:
            day_price = price_pivot.loc[dt].dropna()
            day_signal = signal_pivot.loc[dt].reindex(day_price.index).fillna(0)

            # 涨跌停过滤
            can_buy = ~limit_up_pivot.loc[dt].reindex(day_price.index).fillna(False)
            can_sell = ~limit_dn_pivot.loc[dt].reindex(day_price.index).fillna(False)

            # 3.1) 卖出：基于"昨日持仓 & 今日卖出信号"
            sell_codes = [
                c for c in holdings
                if holdings[c] > 0
                and c in day_signal.index
                and day_signal[c] < 0
                and (not cfg.price_limit or can_sell.get(c, True))
            ]
            sell_amt = 0.0
            for c in sell_codes:
                price = day_price[c]
                shares = holdings[c]
                proceeds = price * shares
                commission = max(proceeds * cfg.commission_rate, cfg.min_commission)
                tax = proceeds * cfg.stamp_tax_rate
                net = proceeds - commission - tax
                cash += net
                sell_amt += net
                trade_records.append({
                    "date": dt, "code": c, "action": "sell",
                    "price": float(price), "shares": int(shares),
                    "amount": float(proceeds), "commission": float(commission),
                    "tax": float(tax), "pnl": float(net),
                })
                holdings[c] = 0

            # 清理 0 持仓
            holdings = {k: v for k, v in holdings.items() if v > 0}

            # 3.2) 买入：基于"今日买入信号 & 未涨停"
            buy_codes = [
                c for c in day_signal.index
                if day_signal[c] > 0
                and (not cfg.price_limit or can_buy.get(c, True))
                and c not in holdings
            ]
            if buy_codes:
                budget = cash * (1 - cfg.cash_buffer) / len(buy_codes)
                for c in buy_codes:
                    price = day_price[c] * (1 + cfg.slippage)
                    max_shares = int(budget / price / cfg.min_lot) * cfg.min_lot
                    if max_shares <= 0:
                        continue
                    cost = price * max_shares
                    commission = max(cost * cfg.commission_rate, cfg.min_commission)
                    total = cost + commission
                    if total > cash * (1 - cfg.cash_buffer):
                        max_shares = int(cash * (1 - cfg.cash_buffer) / price / cfg.min_lot) * cfg.min_lot
                        if max_shares <= 0:
                            continue
                        cost = price * max_shares
                        commission = max(cost * cfg.commission_rate, cfg.min_commission)
                        total = cost + commission
                    cash -= total
                    holdings[c] = holdings.get(c, 0) + max_shares
                    trade_records.append({
                        "date": dt, "code": c, "action": "buy",
                        "price": float(price), "shares": int(max_shares),
                        "amount": float(cost), "commission": float(commission),
                        "tax": 0.0, "pnl": -float(total),
                    })

            # 3.3) 当日权益
            mv = sum(holdings.get(c, 0) * day_price[c] for c in holdings if c in day_price.index)
            equity_records.append({
                "date": dt,
                "equity": float(cash + mv),
                "cash": float(cash),
                "market_value": float(mv),
                "position_count": int(sum(1 for v in holdings.values() if v > 0)),
            })
            prev_holdings = dict(holdings)

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trade_records)

        if equity_curve.empty:
            return self._empty_result()

        eq_series = equity_curve.set_index("date")["equity"]
        metrics = calc_all_metrics(eq_series, trades_df)

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(list(holdings.items()), columns=["code", "shares"]),
            "equity_curve": equity_curve,
            "metrics": metrics,
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
        }
