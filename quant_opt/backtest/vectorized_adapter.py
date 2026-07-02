"""
vectorized_adapter.py
=====================

向量化回测适配器，参考 vectorbt 与 backtesting.py 的向量化思路。

痛点：
- 现有 native_adapter.py 是 O(N×T) 逐日循环
  (skills/backtest-engine/scripts/adapters/native_adapter.py)
- 当回测周期长 (3 年 ≈ 750 bar) 且股票池大 (5000+ 只) 时，回测耗时常 > 30s
- 向量化后理论可降到亚秒级 (类似 vectorbt 的水平)

设计：
- 输入: price DataFrame (index=date, columns=code), signal DataFrame
- 信号类型:
    - 'target_percent' (0~1): 目标持仓权重
    - 'target_amount' (float): 目标持仓金额
    - 'binary' (0/1): 二元信号
- 内部以 numpy 矩阵运算为主
- 严格 T+1：当日信号次日开盘成交
- 支持 A 股涨跌停、T+1、印花税
- 输出: 与 native_adapter 一致的结构 {trades, equity_curve, metrics, ...}

可对比项：
- 速度：在 100 只股票 / 3 年数据上，循环 vs 向量化耗时对比
- 数值一致性：相同输入下结果应得到近似一致的最终收益
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from comprehensive_metrics import compute_full_metrics


class VectorizedAdapter:
    """向量化回测适配器。"""

    def __init__(
        self,
        init_capital: float = 1_000_000.0,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        lot_size: int = 100,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit
        self.lot_size = lot_size

    def _normalize_signals(
        self,
        signals: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        将任意信号格式归一化为 'target_percent' (0~1) 格式。
        输入必须含 'date' 和 'code' 列。
        """
        df = signals.copy()
        if "date" not in df.columns or "code" not in df.columns:
            raise ValueError("signals must contain 'date' and 'code' columns")

        if "target_percent" in df.columns:
            w = df[["date", "code", "target_percent"]].copy()
        elif "target_amount" in df.columns:
            tmp = df[["date", "code", "target_amount"]].copy()
            total_per_date = tmp.groupby("date")["target_amount"].transform("sum")
            w = tmp.copy()
            w["target_percent"] = (tmp["target_amount"] / total_per_date.replace(0, np.nan)).fillna(0.0)
            w = w[["date", "code", "target_percent"]]
        elif "signal" in df.columns:
            w = df[["date", "code", "signal"]].copy()
            w["target_percent"] = w["signal"].astype(float).clip(lower=0)
            w = w[["date", "code", "target_percent"]]
        elif "signal_strength" in df.columns:
            tmp = df[["date", "code", "signal_strength"]].copy()
            tmp["rank"] = tmp.groupby("date")["signal_strength"].rank(pct=True)
            tmp["target_percent"] = (tmp["rank"] > 0.8).astype(float)
            w = tmp[["date", "code", "target_percent"]]
        else:
            raise ValueError(
                f"signals must contain one of: target_percent / target_amount / signal / signal_strength. "
                f"Got: {[c for c in df.columns if c not in ('date', 'code')]}"
            )

        # 截面归一化
        sums = w.groupby("date")["target_percent"].transform("sum")
        w["target_percent"] = (w["target_percent"] / sums.replace(0, np.nan)).fillna(0.0)
        return w

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        benchmark_close: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        执行向量化回测。

        参数：
            data: ['date', 'code', 'open', 'close', ...] 行情
            signals: 至少含 'date', 'code' + 一种信号列
            benchmark_close: 基准收盘价序列 (pd.Series)，用于 alpha/beta

        返回：
            dict，含 trades (DataFrame), positions, equity_curve, metrics
        """
        if data.empty or signals.empty:
            return self._empty_result()

        # 准备行情矩阵
        px = data.pivot_table(index="date", columns="code", values="close").sort_index()
        opx = data.pivot_table(index="date", columns="code", values="open").sort_index()
        is_limit_up = data.pivot_table(index="date", columns="code", values="is_limit_up").sort_index() \
            if "is_limit_up" in data.columns else pd.DataFrame(False, index=px.index, columns=px.columns)
        is_limit_down = data.pivot_table(index="date", columns="code", values="is_limit_down").sort_index() \
            if "is_limit_down" in data.columns else pd.DataFrame(False, index=px.index, columns=px.columns)

        is_limit_up = is_limit_up.reindex(px.index).reindex(columns=px.columns).astype(bool).fillna(False)
        is_limit_down = is_limit_down.reindex(px.index).reindex(columns=px.columns).astype(bool).fillna(False)
        opx = opx.reindex(px.index).reindex(columns=px.columns)
        px = px.reindex(columns=opx.columns)

        # 信号归一化 + 矩阵化
        w = self._normalize_signals(signals).pivot_table(
            index="date", columns="code", values="target_percent"
        ).sort_index()
        w = w.reindex(columns=px.columns).reindex(index=px.index).fillna(0.0)

        # T+1 移位：当日权重次日生效
        if self.t_plus_1:
            target_w = w.shift(1).fillna(0.0)
        else:
            target_w = w.copy()

        # 涨停日不能建仓
        if self.price_limit:
            target_w = target_w.where(~is_limit_up, 0.0)

        n_dates, n_codes = px.shape
        px_arr = px.to_numpy()
        opx_arr = opx.to_numpy()
        target_arr = target_w.to_numpy()
        is_lu_arr = is_limit_up.to_numpy()
        is_ld_arr = is_limit_down.to_numpy()
        LOT = self.lot_size

        # 状态: 现金 + 持仓股数
        cash = self.init_capital
        shares = np.zeros(n_codes, dtype=np.int64)
        equity_arr = np.zeros(n_dates)
        cash_arr = np.zeros(n_dates)
        market_val_arr = np.zeros(n_dates)
        pos_count_arr = np.zeros(n_dates, dtype=int)
        shares_history = np.zeros((n_dates, n_codes), dtype=np.int64)
        trades_records = []

        def rebalance_on(date_idx: int, target_idx: int):
            """调仓：在 date_idx 开盘按 target_arr[target_idx] 调仓。"""
            nonlocal cash, shares
            next_opx = np.where(np.isnan(opx_arr[date_idx]) | (opx_arr[date_idx] <= 0),
                                px_arr[date_idx - 1] if date_idx > 0 else 1.0,
                                opx_arr[date_idx])
            next_opx = np.where(next_opx <= 0, 0.0, next_opx)
            buy_price = next_opx * (1 + self.slippage)
            sell_price = next_opx * (1 - self.slippage)
            valid = next_opx > 0

            # 用前一日 bar 末的权益作为调仓基准
            prev_close = np.where(np.isnan(px_arr[date_idx - 1]) | (px_arr[date_idx - 1] <= 0),
                                  next_opx, px_arr[date_idx - 1])
            prev_close = np.where(prev_close <= 0, 0.0, prev_close)
            market_value_prev = (shares.astype(np.float64) * prev_close).sum()
            equity = cash + market_value_prev

            # 目标股数 (整手)
            target_value = equity * target_arr[target_idx]
            target_value = np.where(np.isnan(target_value), 0.0, target_value)
            target_value = np.maximum(target_value, 0.0)
            target_shares_f = np.where(
                buy_price > 0,
                target_value / buy_price,
                0.0
            )
            target_shares = (np.floor(target_shares_f / LOT) * LOT).astype(np.int64)
            target_shares = np.where(valid, target_shares, 0)
            if self.price_limit:
                target_shares = np.where(is_lu_arr[date_idx], 0, target_shares)

            # 调仓 delta
            delta = target_shares - shares

            sell_mask = (delta < 0) & valid
            buy_mask = (delta > 0) & valid

            # 跌停日不允许卖出
            if self.price_limit:
                sell_mask = sell_mask & (~is_ld_arr[date_idx])

            # 卖出
            sell_qty = -delta[sell_mask]
            if sell_qty.size > 0:
                sell_pr = sell_price[sell_mask]
                sell_amt = sell_qty * sell_pr
                sell_comm = np.maximum(sell_amt * self.commission_rate, self.min_commission)
                sell_tax = sell_amt * self.stamp_tax_rate
                net_proceeds = (sell_amt - sell_comm - sell_tax).sum()
                cash += net_proceeds
                cur_sell = np.zeros(n_codes, dtype=np.int64)
                cur_sell[sell_mask] = sell_qty
                # 记录卖出交易
                cur_sell_indices = np.where(sell_mask)[0]
                for j in cur_sell_indices:
                    qty = int(sell_qty[np.where(cur_sell_indices == j)[0][0]])
                    if qty > 0:
                        trades_records.append({
                            "date": px.index[date_idx], "code": px.columns[j],
                            "action": "sell",
                            "price": float(sell_price[j]),
                            "shares": qty,
                            "amount": float(qty * sell_price[j]),
                            "commission": float(np.maximum(qty * sell_price[j] * self.commission_rate, self.min_commission)),
                            "tax": float(qty * sell_price[j] * self.stamp_tax_rate),
                        })
                shares = shares - cur_sell
                shares = np.maximum(shares, 0)

            # 买入 (在卖出后剩余的现金内，按目标 delta 等比分配)
            buy_qty_needed = delta.copy()
            buy_qty_needed[~buy_mask] = 0
            total_buy_amt_needed = buy_qty_needed[buy_mask] * buy_price[buy_mask]
            if buy_mask.any() and cash > 0:
                # 限制总买入不超过现金 (扣佣金)
                max_cash_for_buy = cash * 0.99  # 留 1% buffer
                if total_buy_amt_needed.sum() > max_cash_for_buy:
                    scale = max_cash_for_buy / total_buy_amt_needed.sum()
                    buy_qty_needed = (buy_qty_needed.astype(np.float64) * scale).astype(np.int64)
                    # 整手化
                    buy_qty_needed = (buy_qty_needed // LOT) * LOT
                buy_qty_needed = np.maximum(buy_qty_needed, 0)

                buy_indices = np.where(buy_qty_needed > 0)[0]
                for j in buy_indices:
                    qty = int(buy_qty_needed[j])
                    pr = float(buy_price[j])
                    cost = qty * pr
                    comm = max(cost * self.commission_rate, self.min_commission)
                    if cost + comm <= cash:
                        cash -= cost + comm
                        shares[j] = shares[j] + qty
                        trades_records.append({
                            "date": px.index[date_idx], "code": px.columns[j],
                            "action": "buy",
                            "price": pr,
                            "shares": qty,
                            "amount": float(cost),
                            "commission": float(comm),
                            "tax": 0.0,
                        })
            return equity

        # 主循环
        for i in range(n_dates):
            cur_px = np.where(np.isnan(px_arr[i]) | (px_arr[i] <= 0), 0.0, px_arr[i])
            # 用 prev_close 重估当日权益
            if i > 0:
                prev_close = np.where(np.isnan(px_arr[i - 1]) | (px_arr[i - 1] <= 0), 0.0, px_arr[i - 1])
                market_value = (shares.astype(np.float64) * prev_close).sum()
                equity = cash + market_value
            else:
                market_value = 0.0
                equity = self.init_capital
            equity_arr[i] = equity
            cash_arr[i] = cash
            market_val_arr[i] = market_value
            pos_count_arr[i] = int((shares > 0).sum())
            shares_history[i] = shares

            # 在下一根 bar 的开盘调仓 (如果存在)
            if i < n_dates - 1:
                rebalance_on(date_idx=i + 1, target_idx=i)

        eq_df = pd.DataFrame({
            "date": px.index,
            "equity": equity_arr,
            "cash": cash_arr,
            "market_value": market_val_arr,
            "position_count": pos_count_arr,
        })
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        equity_series = eq_df.set_index("date")["equity"]

        trades_df = pd.DataFrame(trades_records)
        if trades_df.empty:
            trades_df = pd.DataFrame(columns=["date", "code", "action", "price", "shares", "amount", "commission", "tax", "pnl"])
        else:
            trades_df["pnl"] = 0.0

        position_series = pd.Series(pos_count_arr, index=px.index)
        any_holding = (shares_history > 0).any(axis=1).astype(int)
        position_series = pd.Series(any_holding, index=px.index)

        metrics = compute_full_metrics(
            equity=equity_series,
            trades=trades_df,
            positions=position_series,
            benchmark_close=benchmark_close,
        )

        return {
            "trades": trades_df,
            "equity_curve": eq_df,
            "metrics": metrics,
            "positions": pd.DataFrame(shares_history, index=px.index, columns=px.columns),
            "report_path": "",
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "trades": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "positions": pd.DataFrame(),
            "report_path": "",
        }


def build_test_data(n_stocks: int = 50, n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """生成模拟 A 股行情数据。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    rows = []
    for code in codes:
        ret = rng.normal(0.0008, 0.02, n_days)
        close = 10 * np.exp(np.cumsum(ret))
        open_ = close * (1 + rng.normal(0, 0.002, n_days))
        high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.003, n_days)))
        low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.003, n_days)))
        volume = rng.integers(1_000_000, 10_000_000, n_days)
        is_lu = rng.random(n_days) < 0.02
        is_ld = rng.random(n_days) < 0.02
        for i, dt in enumerate(dates):
            rows.append({
                "date": dt,
                "code": code,
                "open": float(open_[i]),
                "close": float(close[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "volume": int(volume[i]),
                "is_limit_up": bool(is_lu[i]),
                "is_limit_down": bool(is_ld[i]),
            })
    return pd.DataFrame(rows)


def build_test_signals(
    data: pd.DataFrame, top_pct: float = 0.2,
) -> pd.DataFrame:
    """生成截面动量信号：每期取 20 日动量最高的 20% 股票。"""
    df = data.sort_values(["code", "date"]).copy()
    df["ret_20d"] = df.groupby("code")["close"].pct_change(20)
    df["rank"] = df.groupby("date")["ret_20d"].rank(pct=True)
    df["signal"] = (df["rank"] > (1 - top_pct)).astype(int)
    return df[["date", "code", "signal"]].dropna()


if __name__ == "__main__":
    import time

    data = build_test_data(n_stocks=30, n_days=252)
    signals = build_test_signals(data, top_pct=0.2)
    adapter = VectorizedAdapter()
    t0 = time.perf_counter()
    result = adapter.run_backtest(data, signals)
    elapsed = time.perf_counter() - t0
    print(f"Vectorized backtest took: {elapsed:.4f}s for {data['code'].nunique()} stocks × {data['date'].nunique()} days")
    print("Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Equity final: {result['equity_curve']['equity'].iloc[-1]:,.0f}")
    print(f"Trades count: {len(result['trades'])}")
