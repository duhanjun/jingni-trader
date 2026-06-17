"""
向量化回测引擎 (Vectorized Backtest Engine)
=====================================

借鉴项目：polakowo/vectorbt
    - 官方文档: https://vectorbt.dev/
    - Star: 6.5k+
    - 核心创新：全数据向量化运算 + Numba/Rust 加速，100x-1000x 提速

借鉴点：
    1. 避免逐日 Python 循环（原生 native_adapter 中 for dt in dates + iterrows）
    2. 持仓和资金作为二维矩阵（dates x codes）一次性计算
    3. 信号 → 目标权重 → 实际成交的链式向量化

本模块定位：
    - 与 jingni-trader 现有 BaseBacktestEngine 接口对齐
    - 在 [quant_opt_20260617/] 目录下独立验证，不修改主分支
    - 不引入 vectorbt 重依赖，使用纯 numpy/pandas + 可选 numba

设计取舍：
    - 完全向量化：股票池/日期二维矩阵一次计算
    - T+1 处理：以 shift(1) 表达"次日才能卖"
    - 涨跌停处理：mask 矩阵提前准备
    - 现金/持仓：用累计 numpy 数组，避免 DataFrame.iterrows

约束：
    - 保留 A 股佣金 / 印花税 / 滑点等本地参数
    - 输出 schema 与 native_adapter 一致：trades / positions / equity_curve / metrics
"""
from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# 尝试启用 numba 加速（可选依赖）
try:
    from numba import njit  # type: ignore

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _to_pivot(df: pd.DataFrame, value_col: str, dates: np.ndarray, codes: np.ndarray,
              fill_value: float = np.nan) -> np.ndarray:
    """把 long-form DataFrame 转成 (T, N) 二维矩阵，缺失位置填 fill_value"""
    pivot = (df.pivot_table(index="date", columns="code", values=value_col, aggfunc="first")
             .reindex(index=dates, columns=codes))
    return pivot.to_numpy()


class VectorizedBacktestEngine:
    """
    向量化回测引擎 (A股版)

    与 native_adapter 的关键差异：
    - 把信号矩阵、持仓矩阵、净值矩阵都展开为 (T, N) 二维数组
    - 核心推进用 np.cumsum / np.where，避免 Python 循环
    - 一次性输出整段权益曲线，便于后续多参数扫描
    """

    def __init__(
        self,
        init_capital: float = 1_000_000.0,
        commission_rate: float = 0.00025,
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        lot_size: int = 100,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit
        self.lot_size = lot_size

    # ----------------------------------------------------------------------
    # 公共入口
    # ----------------------------------------------------------------------

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        benchmark: str = "000300.SH",
    ) -> Dict[str, Any]:
        """
        参数:
            data:    包含 code, date, open, high, low, close, volume,
                     is_limit_up, is_limit_down, change_pct
            signals: 包含 code, date, signal
                     signal: 1=买入, -1=卖出, 0=不动
        返回:
            字典，包含 trades / positions / equity_curve / metrics
        """
        if data.empty or signals.empty:
            return self._empty_result()

        # 统一类型
        data = data.copy()
        signals = signals.copy()
        data["date"] = pd.to_datetime(data["date"])
        signals["date"] = pd.to_datetime(signals["date"])

        # 取得全局交易日 / 股票池
        dates = np.array(sorted(data["date"].unique()))
        codes = np.array(sorted(data["code"].unique()))
        if len(dates) < 2 or len(codes) == 0:
            return self._empty_result()

        date_to_idx = {d: i for i, d in enumerate(dates)}
        code_to_idx = {c: i for i, c in enumerate(codes)}

        T, N = len(dates), len(codes)

        # ---- 构造价格矩阵 (T, N) ----
        close = _to_pivot(data, "close", dates, codes)
        limit_up = _to_pivot(data, "is_limit_up", dates, codes, fill_value=False)
        limit_down = _to_pivot(data, "is_limit_down", dates, codes, fill_value=False)

        # ---- 构造信号矩阵 (T, N)，1/0/-1 ----
        sig_mat = np.zeros((T, N), dtype=np.int8)
        sig_df = signals.copy()
        sig_df["date_idx"] = sig_df["date"].map(date_to_idx)
        sig_df["code_idx"] = sig_df["code"].map(code_to_idx)
        sig_df = sig_df.dropna(subset=["date_idx", "code_idx"])
        sig_df["date_idx"] = sig_df["date_idx"].astype(int)
        sig_df["code_idx"] = sig_df["code_idx"].astype(int)
        sig_df["signal"] = sig_df["signal"].astype(int)
        sig_mat[sig_df["date_idx"].to_numpy(), sig_df["code_idx"].to_numpy()] = sig_df["signal"].to_numpy()

        # ---- 调用向量化核心 ----
        equity, cash_arr, holdings_arr, trade_log = self._vectorized_loop(
            close=close,
            signals=sig_mat,
            limit_up=limit_up,
            limit_down=limit_down,
        )

        # ---- 整理回结果 ----
        equity_curve = pd.DataFrame({
            "date": dates,
            "equity": equity,
            "cash": cash_arr,
            "market_value": equity - cash_arr,
        })
        trades_df = pd.DataFrame(trade_log, columns=[
            "date", "code", "action", "price", "shares",
            "amount", "commission", "tax", "pnl",
        ]) if trade_log else pd.DataFrame(columns=[
            "date", "code", "action", "price", "shares",
            "amount", "commission", "tax", "pnl",
        ])

        # 末次持仓
        final_pos = holdings_arr[-1]
        positions = pd.DataFrame({
            "code": codes[final_pos > 0],
            "shares": final_pos[final_pos > 0],
        })

        metrics = self._calc_metrics(equity_curve, trades_df)
        return {
            "trades": trades_df,
            "positions": positions,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    # ----------------------------------------------------------------------
    # 核心推进 (向量化)
    # ----------------------------------------------------------------------

    def _vectorized_loop(
        self,
        close: np.ndarray,
        signals: np.ndarray,
        limit_up: np.ndarray,
        limit_down: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        核心循环：用 numpy 数组做"伪循环"——按时间步推进，但用向量化
        计算当日所有股票的买卖。

        返回:
            equity:     (T,) 每日权益
            cash_arr:   (T,) 每日现金
            holdings:   (T, N) 每日持仓股数
            trade_log:  list of dict
        """
        T, N = close.shape
        cash = self.init_capital
        holdings = np.zeros((T + 1, N), dtype=np.int64)  # 末端状态对齐
        cash_arr = np.zeros(T, dtype=np.float64)
        equity_arr = np.zeros(T, dtype=np.float64)
        trade_log: list = []

        # 当日"可买可卖" mask
        for t in range(T):
            # 上一日持仓
            prev_h = holdings[t]

            # 计算当前权益（按昨日持仓 × 今日开盘前可用 close 估值）
            if t == 0:
                mkt_value = 0.0
            else:
                mkt_value = float(np.nansum(prev_h * close[t]))
            equity_arr[t] = cash + mkt_value
            cash_arr[t] = cash

            # ---- 1) 处理卖出信号 ----
            sell_mask = (signals[t] == -1) & (prev_h > 0)
            if self.price_limit:
                sell_mask &= ~limit_down[t].astype(bool)
            sell_mask &= ~np.isnan(close[t])

            if sell_mask.any():
                sell_codes = np.where(sell_mask)[0]
                sell_prices = close[t, sell_codes]
                sell_shares = prev_h[sell_codes]
                sell_amounts = sell_prices * sell_shares
                commissions = np.maximum(sell_amounts * self.commission_rate, self.min_commission)
                taxes = sell_amounts * self.stamp_tax_rate
                proceeds = sell_amounts - commissions - taxes
                cash += float(proceeds.sum())

                # 清仓这些代码
                holdings[t, sell_codes] = 0

                for j, code_idx in enumerate(sell_codes):
                    trade_log.append({
                        "date_idx": t,
                        "code_idx": int(code_idx),
                        "action": "sell",
                        "price": float(sell_prices[j]),
                        "shares": int(sell_shares[j]),
                        "amount": float(sell_amounts[j]),
                        "commission": float(commissions[j]),
                        "tax": float(taxes[j]),
                        "pnl": float(proceeds[j]),
                    })

            # ---- 2) 处理买入信号 ----
            # 若启用 T+1：昨日信号今日才能买（用 signal[t-1] 而非 signal[t]）
            if self.t_plus_1:
                # 改为上一天的信号驱动今天的成交
                if t == 0:
                    buy_mask = np.zeros(N, dtype=bool)
                else:
                    buy_mask = (signals[t - 1] == 1)
            else:
                buy_mask = (signals[t] == 1)

            buy_mask &= (prev_h == 0)  # 已有持仓的不重复买
            if self.price_limit:
                buy_mask &= ~limit_up[t].astype(bool)
            buy_mask &= ~np.isnan(close[t])

            if buy_mask.any():
                buy_codes = np.where(buy_mask)[0]
                buy_prices = close[t, buy_codes] * (1.0 + self.slippage)
                n_buy = len(buy_codes)
                budget_per = cash * 0.95 / max(n_buy, 1)
                raw_shares = np.floor(budget_per / np.maximum(buy_prices, 1e-6) / self.lot_size) * self.lot_size
                raw_shares = raw_shares.astype(np.int64)
                # 资金够的部分
                buy_amounts = raw_shares * buy_prices
                commissions = np.maximum(buy_amounts * self.commission_rate, self.min_commission)
                costs = buy_amounts + commissions
                affordable = costs <= cash * 0.98
                # 不可负担的剔除
                if not affordable.all():
                    # 简化处理：剔除后总成本
                    cost_total = float(costs[affordable].sum())
                    if cost_total > cash:
                        # 再降一次预算
                        scale = cash * 0.98 / cost_total
                        raw_shares[affordable] = np.floor(
                            raw_shares[affordable] * scale / self.lot_size
                        ) * self.lot_size
                        buy_amounts = raw_shares * buy_prices
                        commissions = np.maximum(buy_amounts * self.commission_rate, self.min_commission)
                        costs = buy_amounts + commissions
                    raw_shares[~affordable] = 0
                    buy_amounts[~affordable] = 0
                    commissions[~affordable] = 0
                    costs[~affordable] = 0
                # 实际扣款
                cash -= float(costs.sum())
                holdings[t, buy_codes] = raw_shares

                for j, code_idx in enumerate(buy_codes):
                    if raw_shares[j] <= 0:
                        continue
                    trade_log.append({
                        "date_idx": t,
                        "code_idx": int(code_idx),
                        "action": "buy",
                        "price": float(buy_prices[j]),
                        "shares": int(raw_shares[j]),
                        "amount": float(buy_amounts[j]),
                        "commission": float(commissions[j]),
                        "tax": 0.0,
                        "pnl": float(-(buy_amounts[j] + commissions[j])),
                    })

            # ---- 3) 复制到 t+1 ----
            holdings[t + 1] = holdings[t]

        # 去掉最后一行占位
        holdings = holdings[:T]
        return equity_arr, cash_arr, holdings, trade_log

    # ----------------------------------------------------------------------
    # 绩效指标
    # ----------------------------------------------------------------------

    @staticmethod
    def _calc_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, float]:
        if equity_curve.empty or "equity" not in equity_curve.columns:
            return {}
        eq = equity_curve["equity"].astype(float)
        if len(eq) < 2:
            return {}
        ret = eq.pct_change().dropna()
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        n_years = len(eq) / 252
        ann_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        vol = float(ret.std() * math.sqrt(252))
        sharpe = float((ret.mean() * 252 - 0.03) / vol) if vol > 0 else 0.0
        cum_max = eq.cummax()
        mdd = float((eq / cum_max - 1).min())
        calmar = float(ann_return / abs(mdd)) if mdd < 0 else 0.0
        downside = ret[ret < 0]
        downside_std = float(downside.std() * math.sqrt(252)) if len(downside) > 1 else 0.0
        sortino = float((ret.mean() * 252 - 0.03) / downside_std) if downside_std > 0 else 0.0
        win_rate = 0.0
        if not trades.empty and "pnl" in trades.columns:
            sell_trades = trades[trades["action"] == "sell"]
            if not sell_trades.empty:
                win_rate = float((sell_trades["pnl"] > 0).mean())
        return {
            "total_return": total_return,
            "annual_return": ann_return,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "total_trades": int(len(trades)),
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        empty = pd.DataFrame()
        return {
            "trades": empty,
            "positions": empty,
            "equity_curve": empty,
            "metrics": {},
            "report_path": "",
        }


__all__ = ["VectorizedBacktestEngine", "HAS_NUMBA"]
