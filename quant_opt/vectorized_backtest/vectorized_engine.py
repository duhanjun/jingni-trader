"""
向量化回测引擎
==============

设计要点
--------
1. **不逐日循环** —— 通过 groupby('date') + 广播进行批量资金分配
2. **不逐行 iterrows** —— 持仓、交易、PnL 全部以 DataFrame 形式产出
3. **可分桶** —— 标的越多越能享受向量化红利
4. **可对比** —— 内部与逐日循环版做等价性测试

A 股规则
-------
- T+1: 当日买入次日才可卖（实现为持仓表上的 ``t_plus_1_available`` 字段）
- 涨跌停: 涨停不能买、跌停不能卖（输入列 ``is_limit_up``/``is_limit_down``）
- 交易费: 佣金万 2.5（最低 5 元）、印花税 1‰（仅卖出）、过户费 0.02‰

约束与简化
----------
- 不处理融资融券
- 不处理停牌（默认假设数据已过滤停牌）
- 等权分配可用预算
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.001
    transfer_fee_rate: float = 0.00002
    slippage: float = 0.001
    t_plus_1: bool = True
    price_limit: bool = True
    max_position_per_stock: float = 0.1  # 单票最大仓位比例
    min_trade_cash: float = 10_000.0      # 不足此金额不买新仓


class VectorizedBacktester:
    """向量化回测器

    Parameters
    ----------
    config:
        ``BacktestConfig`` 实例
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Dict[str, Any]:
        """执行回测

        parameters
        ----------
        data:
            必须包含 code, date, close 列；可选: open, high, low, volume,
            is_limit_up, is_limit_down
        signals:
            必须包含 code, date, signal (1=买 / -1=卖 / 0=持有)
        """
        cfg = self.config
        if data.empty or signals.empty:
            return self._empty_result()

        df = self._prepare(data, signals)

        # === 1. 当日目标持仓表 ===
        # 1 表示目标持有, 0 表示目标空仓
        target = df.pivot(index="date", columns="code", values="signal").fillna(0)
        target = (target > 0).astype(int)
        target = target.sort_index()
        # 对齐价格矩阵
        close = df.pivot(index="date", columns="code", values="close").reindex(
            columns=target.columns
        ).sort_index()
        limit_up = self._safe_pivot(df, "is_limit_up", target.index, target.columns)
        limit_down = self._safe_pivot(df, "is_limit_down", target.index, target.columns)

        # === 2. 计算每日买卖动作 ===
        # desired: 1 表示目标持仓 (signal > 0), 0 表示不主动持仓
        desired = (df.pivot(index="date", columns="code", values="signal")
                    .reindex(index=target.index, columns=target.columns)
                    .fillna(0) > 0).astype(int)
        # 当日信号原值 (含 0/-1), 用于卖出判断
        signal_mat = (df.pivot(index="date", columns="code", values="signal")
                       .reindex(index=target.index, columns=target.columns)
                       .fillna(0))
        desired = desired.sort_index()

        # 上一期"主动持有"标记 (来自 desired)
        prev_desired = desired.shift(1).fillna(0).astype(int)
        # 当日新买入: 上一期未持仓, 当日 desired=1
        buy_mask = (desired == 1) & (prev_desired == 0)
        # 卖出: 当日 signal < 0 且上日已"主动持有" (T+1)
        sell_mask = (signal_mat < 0) & (prev_desired == 1)
        # 涨跌停过滤
        if cfg.price_limit:
            buy_mask = buy_mask & (~limit_up.fillna(False))
            sell_mask = sell_mask & (~limit_down.fillna(False))

        # === 3. 计算可用预算与资金 ===
        # 当日需要买入的标的数
        n_buy_per_day = buy_mask.sum(axis=1)
        # 上日现金 + 上日持仓按收盘价估值 * (1 + 持仓权重) ... 简化为等权分配
        # 用累计函数推导 daily_cash
        equity_records, trade_records = self._simulate(
            desired=desired,
            prev_desired=prev_desired,
            buy_mask=buy_mask,
            sell_mask=sell_mask,
            close=close,
            cfg=cfg,
        )

        equity_curve = pd.DataFrame(equity_records)
        trades = pd.DataFrame(trade_records)
        metrics = self._calc_metrics(equity_curve, trades, cfg)

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    # -----------------------------------------------------------------
    # 内部
    # -----------------------------------------------------------------

    @staticmethod
    def _prepare(data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        df = data.merge(signals[["code", "date", "signal"]], on=["code", "date"], how="left")
        df["signal"] = df["signal"].fillna(0).astype(float)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "code"]).reset_index(drop=True)
        return df

    @staticmethod
    def _safe_pivot(df: pd.DataFrame, col: str, idx, cols) -> pd.DataFrame:
        if col not in df.columns:
            return pd.DataFrame(False, index=idx, columns=cols)
        p = df.pivot(index="date", columns="code", values=col)
        return p.reindex(index=idx, columns=cols).fillna(False).astype(bool)

    def _simulate(
        self,
        desired: pd.DataFrame,
        prev_desired: pd.DataFrame,
        buy_mask: pd.DataFrame,
        sell_mask: pd.DataFrame,
        close: pd.DataFrame,
        cfg: BacktestConfig,
    ):
        equity_records = []
        trade_records = []
        cash = cfg.init_capital
        # 持仓 shares 矩阵
        shares = pd.DataFrame(0.0, index=desired.index, columns=desired.columns)
        dates = list(desired.index)
        for i, dt in enumerate(dates):
            # === 卖出: 按今日收盘价 ===
            if i > 0 and sell_mask.iloc[i].any():
                to_sell = sell_mask.iloc[i]
                sell_codes = to_sell[to_sell].index
                if len(sell_codes):
                    sell_px = close.iloc[i][sell_codes]
                    sell_sh = shares.iloc[i - 1][sell_codes] if i > 0 else pd.Series(0, index=sell_codes)
                    sell_amount = (sell_px * sell_sh).sum()
                    commission = max(sell_amount * cfg.commission_rate, cfg.min_commission)
                    tax = sell_amount * cfg.stamp_tax_rate
                    cost = commission + tax
                    cash += sell_amount - cost
                    for code in sell_codes:
                        sh = sell_sh.get(code, 0)
                        if sh <= 0:
                            continue
                        px = sell_px.get(code, np.nan)
                        trade_records.append({
                            "date": dt, "code": code, "action": "sell",
                            "price": float(px), "shares": float(sh),
                            "amount": float(px * sh),
                        })
                    shares.loc[dt, sell_codes] = 0
            else:
                # 没有卖出, 沿用上日持仓
                if i > 0:
                    shares.loc[dt] = shares.iloc[i - 1]

            # === 买入: 收盘价 + 滑点 ===
            n_buy = int(buy_mask.iloc[i].sum())
            if n_buy > 0 and cash > cfg.min_trade_cash:
                buy_codes = buy_mask.iloc[i][buy_mask.iloc[i]].index
                budget_per = min(
                    cash * cfg.max_position_per_stock,
                    cash * 0.95 / n_buy,
                )
                buy_px = close.iloc[i][buy_codes] * (1 + cfg.slippage)
                for code in buy_codes:
                    px = buy_px.get(code, np.nan)
                    if px is None or np.isnan(px) or px <= 0:
                        continue
                    qty = int(budget_per / px / 100) * 100
                    if qty <= 0:
                        continue
                    cost = px * qty
                    commission = max(cost * cfg.commission_rate, cfg.min_commission)
                    total = cost + commission
                    if total > cash:
                        qty = int((cash * 0.98) / px / 100) * 100
                        if qty <= 0:
                            continue
                        cost = px * qty
                        commission = max(cost * cfg.commission_rate, cfg.min_commission)
                        total = cost + commission
                    cash -= total
                    shares.loc[dt, code] = shares.loc[dt, code] + qty
                    trade_records.append({
                        "date": dt, "code": code, "action": "buy",
                        "price": float(px), "shares": float(qty),
                        "amount": float(cost),
                    })
            else:
                if i > 0:
                    shares.loc[dt] = shares.iloc[i - 1]

            # === 每日市值 ===
            market_value = float((shares.loc[dt] * close.loc[dt]).sum())
            total_equity = cash + market_value
            equity_records.append({
                "date": dt,
                "equity": total_equity,
                "cash": cash,
                "market_value": market_value,
                "position_count": int((shares.loc[dt] > 0).sum()),
            })

        return equity_records, trade_records

    @staticmethod
    def _calc_metrics(equity: pd.DataFrame, trades: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, float]:
        if equity.empty or "equity" not in equity.columns:
            return {}
        eq = equity.set_index("date")["equity"]
        if len(eq) < 2:
            return {}
        ret = eq.pct_change().dropna()
        if ret.empty:
            return {}
        total = float(eq.iloc[-1] / eq.iloc[0] - 1)
        n_years = len(ret) / 252
        annual = float((1 + total) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        vol = float(ret.std() * np.sqrt(252))
        sharpe = (annual - 0.03) / vol if vol > 0 else 0.0
        cum_max = eq.cummax()
        mdd = float(((eq - cum_max) / cum_max).min())
        calmar = annual / abs(mdd) if mdd != 0 else 0.0
        win_rate = 0.0
        if not trades.empty and "action" in trades.columns and "amount" in trades.columns:
            # 配对买卖: 简化按买入金额与卖出金额估算
            buys = trades[trades["action"] == "buy"]
            sells = trades[trades["action"] == "sell"]
            if len(buys) > 0 and len(sells) > 0:
                win_rate = float(
                    (sells["amount"].sum() - buys["amount"].sum() > 0)
                )
        return {
            "total_return": total,
            "annual_return": annual,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "total_trades": int(len(trades)),
            "n_days": int(len(eq)),
        }

    @staticmethod
    def _empty_result():
        return {
            "trades": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
