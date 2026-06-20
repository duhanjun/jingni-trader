"""
极简向量化回测器

借鉴 vectorbt 的向量化回测思想，为 jingni-trader 增加一个**用于快速验证/参数扫描**
的回测器。它不是要取代 rqalpha/backtrader，而是作为快速 sanity check 工具：

- 纯 numpy/pandas 实现，零外部重依赖
- 每日调仓 (Top-K by factor)
- 支持手续费、滑点、T+1、涨跌停过滤
- 支持单参数扫描：n_stocks / lookback / rebalance_freq

返回标准化的 equity_curve / trades / metrics，方便与现有 jingni-trader
回测结果对比。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from ..vectorized_metrics.metrics import compute_all_metrics


@dataclass
class MiniBacktestConfig:
    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025
    stamp_tax_rate: float = 0.001
    slippage: float = 0.0001
    min_lot: int = 100
    t_plus_1: bool = True
    price_limit: bool = True
    price_limit_pct: float = 0.10
    n_stocks: int = 20              # 持仓数量
    rebalance_freq: int = 1         # 调仓频率（每 N 天）
    max_weight: float = 0.10        # 单票最大权重
    benchmark: str = "000300.SH"
    risk_free_rate: float = 0.03


class MiniVectorBacktest:
    """
    极简向量化回测器
    输入：data (OHLCV), factor (单因子值)
    行为：每日按 factor 排序，取 Top-K 等权持有，调仓频率可配
    输出：equity_curve / metrics / trades
    """

    def __init__(self, config: Optional[MiniBacktestConfig] = None):
        self.cfg = config or MiniBacktestConfig()

    def _filter_price_limit(self, panel: pd.DataFrame) -> pd.DataFrame:
        """过滤一字涨跌停（无法买入的股票）"""
        if not self.cfg.price_limit:
            return panel
        if "is_limit_up" in panel.columns:
            return panel[~panel["is_limit_up"].fillna(False)]
        if "close" in panel.columns and "high" in panel.columns and "low" in panel.columns:
            limit = (panel["close"] >= panel["high"]) & (panel["close"] <= panel["high"] * 1.0001) \
                & ((panel["close"] / panel["close"].shift(1) - 1) >= self.cfg.price_limit_pct - 0.001)
            return panel[~limit.fillna(False)]
        return panel

    def run(self, data: pd.DataFrame, factor_col: str = "factor") -> Dict:
        """
        data: 必须列 code, date, open, high, low, close, [factor_col]
        factor_col: 因子值列名，值越大越偏好
        """
        required = {"code", "date", "open", "high", "low", "close", factor_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"data 缺少列: {missing}")
        data = data.sort_values(["date", "code"]).reset_index(drop=True)

        # 计算每日 forward return 用于 benchmark
        dates = sorted(data["date"].unique())
        if len(dates) < 2:
            return {"equity_curve": pd.DataFrame(), "metrics": {}, "trades": pd.DataFrame()}

        # 准备 pivot
        close_pivot = data.pivot(index="date", columns="code", values="close")
        ret_pivot = close_pivot.pct_change()
        open_pivot = data.pivot(index="date", columns="code", values="open")
        factor_pivot = data.pivot(index="date", columns="code", values=factor_col)

        # 准备调仓日集合
        rebalance_dates = dates[::max(1, self.cfg.rebalance_freq)]
        rebalance_set = set(rebalance_dates)

        # 状态
        current_holdings: Dict[str, int] = {}     # code -> shares
        entry_prices: Dict[str, float] = {}
        cash = self.cfg.init_capital
        equity_records = []
        trade_records = []

        prev_date = None
        for di, date in enumerate(dates):
            day_close = close_pivot.loc[date]
            day_open = open_pivot.loc[date] if date in open_pivot.index else day_close
            day_factor = factor_pivot.loc[date]

            # 当日 NAV = 现金 + 持仓市值（按 close 估值）
            holding_value = sum(current_holdings.get(c, 0) * day_close.get(c, np.nan)
                                for c in current_holdings)
            nav = cash + holding_value

            if di == 0:
                equity_records.append({"date": date, "equity": nav, "cash": cash,
                                       "holding_value": holding_value, "n_holdings": 0})
                prev_date = date
                continue

            prev_close = close_pivot.loc[prev_date]

            # 1) 结算昨日持仓的当日收益：使用今日 open 近似（更现实）
            new_holdings = {}
            today_value = 0.0
            for code, shares in current_holdings.items():
                if code not in day_open or pd.isna(day_open[code]) or pd.isna(prev_close.get(code)):
                    continue
                # T+1：用 open 成交
                px = day_open[code] * (1 + self.cfg.slippage)
                if shares > 0:
                    today_value += shares * px
                    new_holdings[code] = shares
            cash += holding_value - today_value  # 持仓市值变动
            current_holdings = new_holdings
            holding_value = today_value
            nav = cash + holding_value

            # 2) 调仓日：卖出非目标 -> 买入 Top-K
            if date in rebalance_set and not day_factor.dropna().empty:
                # 选出 Top-K
                ranked = day_factor.dropna().sort_values(ascending=False)
                target_codes = list(ranked.index[: self.cfg.n_stocks])

                # 卖出非目标
                sell_codes = [c for c in current_holdings if c not in target_codes]
                for code in sell_codes:
                    px = day_open[code] * (1 - self.cfg.slippage) if code in day_open else day_close[code]
                    if pd.isna(px):
                        continue
                    shares = current_holdings[code]
                    proceeds = shares * px
                    fee = max(proceeds * self.cfg.commission_rate, 5.0)
                    fee += proceeds * self.cfg.stamp_tax_rate   # 印花税
                    cash += proceeds - fee
                    trade_records.append({
                        "date": date, "code": code, "side": "sell",
                        "price": float(px), "shares": int(shares),
                        "amount": float(proceeds), "fee": float(fee),
                    })
                    del current_holdings[code]
                    entry_prices.pop(code, None)

                # 买入目标（按等权，但限制 max_weight）
                target_list = [c for c in target_codes if c not in current_holdings]
                if target_list and cash > 0:
                    per_cap = min(
                        nav * self.cfg.max_weight,
                        cash * 0.95 / max(1, len(target_list)),
                    )
                    for code in target_list:
                        if code not in day_open or pd.isna(day_open[code]):
                            continue
                        px = day_open[code] * (1 + self.cfg.slippage)
                        # 过滤一字板
                        if self.cfg.price_limit and code in close_pivot.columns:
                            prev_px = prev_close.get(code)
                            if prev_px and prev_px > 0:
                                if (px / prev_px - 1) >= self.cfg.price_limit_pct - 0.001:
                                    continue
                        available_cash = min(per_cap, cash)
                        if available_cash < px * self.cfg.min_lot:
                            continue
                        shares = int(available_cash // (px * self.cfg.min_lot)) * self.cfg.min_lot
                        if shares <= 0:
                            continue
                        amount = shares * px
                        fee = max(amount * self.cfg.commission_rate, 5.0)
                        total_cost = amount + fee
                        if total_cost > cash:
                            continue
                        cash -= total_cost
                        current_holdings[code] = shares
                        entry_prices[code] = float(px)
                        trade_records.append({
                            "date": date, "code": code, "side": "buy",
                            "price": float(px), "shares": int(shares),
                            "amount": float(amount), "fee": float(fee),
                        })

            # 更新当日 NAV
            new_holding_value = sum(current_holdings.get(c, 0) * day_close.get(c, np.nan)
                                    for c in current_holdings)
            nav = cash + new_holding_value

            equity_records.append({
                "date": date,
                "equity": float(nav),
                "cash": float(cash),
                "holding_value": float(new_holding_value),
                "n_holdings": len(current_holdings),
            })
            prev_date = date

        equity_curve = pd.DataFrame(equity_records)
        if equity_curve.empty:
            return {"equity_curve": equity_curve, "metrics": {}, "trades": pd.DataFrame(trade_records)}

        # 计算日收益 & 指标
        equity_curve["return"] = equity_curve["equity"].pct_change()
        returns = equity_curve["return"].dropna().values
        metrics = compute_all_metrics(returns, risk_free=self.cfg.risk_free_rate)

        trades_df = pd.DataFrame(trade_records)
        if not trades_df.empty and "side" in trades_df.columns:
            metrics["n_trades"] = int(len(trades_df))
            metrics["n_buys"] = int((trades_df["side"] == "buy").sum())
            metrics["n_sells"] = int((trades_df["side"] == "sell").sum())
            metrics["total_commission"] = float(trades_df["fee"].sum())

        return {
            "equity_curve": equity_curve,
            "metrics": metrics,
            "trades": trades_df,
        }
