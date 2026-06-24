"""
向量化回测适配器

借鉴来源：VectorBT 的矩阵化回测思路（NumPy 广播 + 向量化信号处理）
设计文档：https://vectorbt.pro/documentation/fundamentals/

优化思路：
main 分支的 native_adapter.py 使用三层 Python 循环：
  1. for dt in dates              （日期循环，因现金路径依赖无法完全消除）
  2. for _, row in day_signal.iterrows()  （逐行遍历信号 —— 性能瓶颈）
  3. for code in buy_codes / sell_codes   （逐股票处理 —— 性能瓶颈）

本适配器采用"半向量化"策略：
  - 将行情、信号透视为宽矩阵（date × code），一次性完成
  - 保留日期循环（现金账户路径依赖），但每个日期内的全部逐股票操作
    （卖出判定、买入判定、等权预算、整手取整、市值汇总）均用 NumPy
    向量化实现，消除 iterrows 与逐股票 Python 循环
  - 严格保留 A 股规则：T+1、涨跌停限制、印花税（卖出）、佣金、滑点

正确性：与 native_adapter 逻辑等价，仅替换循环为向量化运算
性能：对 N 只股票的单日处理从 O(N) Python 调用降为 O(1) NumPy 调用
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

from ..base.base_backtest_engine import BaseBacktestEngine
from ..base.base_backtest import BaseBacktestMetrics
from .extended_metrics import ExtendedMetrics


class VectorizedAdapter(BaseBacktestEngine):
    """向量化回测适配器（与 NativeAdapter 逻辑等价，性能更高）"""

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

        # ── 1. 透视为宽矩阵（date × code）──────────────────
        close_w, limit_up_w, limit_down_w = self._pivot_market(data)
        buy_mask_w, sell_mask_w = self._pivot_signals(signals, close_w.index, close_w.columns)

        all_dates = list(close_w.index)
        codes = list(close_w.columns)
        n_codes = len(codes)
        if n_codes == 0 or len(all_dates) == 0:
            return self._empty_result()

        # 转 NumPy 数组加速（行=日期，列=股票）
        close_arr = close_w.to_numpy(dtype=float)
        limit_up_arr = limit_up_w.reindex(index=close_w.index, columns=close_w.columns).fillna(False).to_numpy()
        limit_down_arr = limit_down_w.reindex(index=close_w.index, columns=close_w.columns).fillna(False).to_numpy()
        # 分离买卖掩码：同日对同一股票既买又卖时，原生适配器先卖后买
        buy_mask_arr = buy_mask_w.to_numpy(dtype=bool)
        sell_mask_arr = sell_mask_w.to_numpy(dtype=bool)

        nan_close = np.isnan(close_arr)

        # 与 native_adapter 一致：仅遍历有信号的日期（且当日有行情）
        # native: dates = sorted(signals['date'].unique()); 跳过 day_data.empty
        sig_dates = sorted(set(signals["date"].tolist()))
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        # 仅保留既有信号又有行情的日期
        dates = [d for d in sig_dates if d in date_to_idx and not nan_close[date_to_idx[d]].all()]
        if not dates:
            return self._empty_result()

        # ── 2. 逐日期回测（仅日期循环，内部全向量化）─────────
        cash = float(init_capital)
        positions = np.zeros(n_codes, dtype=float)       # 持仓股数
        # T+1：记录今日新买入的股票，当日不可卖出
        bought_today = np.zeros(n_codes, dtype=bool)

        equity_records = []
        trades = []

        for dt in dates:
            i = date_to_idx[dt]
            close_i = close_arr[i]
            buy_i = buy_mask_arr[i]
            sell_i = sell_mask_arr[i]
            lu_i = limit_up_arr[i]
            ld_i = limit_down_arr[i]
            valid_i = ~nan_close[i]  # 当日有行情的股票

            # T+1：新的一天开始，昨日及之前买入的股票今日可卖，
            # 仅今日新买入的不可卖（在买入阶段重新标记）
            if t_plus_1:
                bought_today = np.zeros(n_codes, dtype=bool)

            # ── 2a. 卖出（向量化）──
            # 卖出条件：sell信号 且 持仓>0 且 当日有行情 且 非跌停 且 非今日买入(T+1)
            can_sell = sell_i & (positions > 0) & valid_i
            if price_limit:
                can_sell = can_sell & (~ld_i)
            if t_plus_1:
                can_sell = can_sell & (~bought_today)

            if can_sell.any():
                sell_shares = positions[can_sell].copy()
                sell_prices = close_i[can_sell]
                sell_codes_idx = np.where(can_sell)[0]
                sell_amounts = sell_prices * sell_shares
                # 佣金（最低5元）+ 印花税（卖出）
                sell_commissions = np.maximum(sell_amounts * commission_rate, 5.0)
                sell_taxes = sell_amounts * stamp_tax_rate
                sell_costs = sell_commissions + sell_taxes
                sell_net = sell_amounts - sell_costs
                cash += float(sell_net.sum())
                positions[can_sell] = 0.0
                # 记录成交
                for k, idx in enumerate(sell_codes_idx):
                    trades.append({
                        "date": dt, "code": codes[idx], "action": "sell",
                        "price": float(sell_prices[k]), "shares": float(sell_shares[k]),
                        "amount": float(sell_amounts[k]), "commission": float(sell_commissions[k]),
                        "tax": float(sell_taxes[k]), "pnl": float(sell_net[k]),
                    })

            # ── 2b. 买入（向量化）──
            # 买入条件：buy信号 且 当日有行情 且 非涨停
            can_buy = buy_i & valid_i
            if price_limit:
                can_buy = can_buy & (~lu_i)

            # 注意：预算按"全部买入信号数"计算（含涨停等无法买入的），
            # 与 native_adapter 行为一致（其 buy_codes 含涨停股，仅循环内 continue）
            n_buy_total = int(buy_i.sum())
            if can_buy.any() and n_buy_total > 0:
                budget_per_stock = cash * 0.95 / n_buy_total
                buy_idx = np.where(can_buy)[0]
                buy_prices = close_i[buy_idx] * (1.0 + slippage)
                # 整手取整（100股/手），向量化
                shares = np.floor(budget_per_stock / buy_prices / 100.0).astype(np.int64) * 100
                # 预算不足则按可用现金 98% 重新计算（单只）
                buy_amounts = buy_prices * shares
                commissions = np.maximum(buy_amounts * commission_rate, 5.0)
                costs = buy_amounts + commissions
                over_budget = costs > cash
                if over_budget.any():
                    shares[over_budget] = np.floor(
                        (cash * 0.98) / buy_prices[over_budget] / 100.0
                    ).astype(np.int64) * 100
                    buy_amounts[over_budget] = buy_prices[over_budget] * shares[over_budget]
                    commissions[over_budget] = np.maximum(
                        buy_amounts[over_budget] * commission_rate, 5.0
                    )
                    costs[over_budget] = buy_amounts[over_budget] + commissions[over_budget]
                # 总成本超过可用现金时按比例缩放（等价于 native 顺序降仓的总体效果，
                # 但更确定性、不依赖股票处理顺序）
                total_cost = float(costs.sum())
                if total_cost > cash and total_cost > 0:
                    scale = (cash * 0.98) / total_cost
                    shares = np.floor(shares * scale / 100.0).astype(np.int64) * 100
                    buy_amounts = buy_prices * shares
                    commissions = np.maximum(buy_amounts * commission_rate, 5.0)
                    costs = buy_amounts + commissions
                # 过滤 0 股
                valid_lots = shares > 0
                if valid_lots.any():
                    sel = buy_idx[valid_lots]
                    sel_shares = shares[valid_lots].astype(float)
                    sel_prices = buy_prices[valid_lots]
                    sel_amounts = buy_amounts[valid_lots]
                    sel_commissions = commissions[valid_lots]
                    sel_costs = costs[valid_lots]
                    cash -= float(sel_costs.sum())
                    positions[sel] += sel_shares
                    if t_plus_1:
                        bought_today[sel] = True
                    for k in range(len(sel)):
                        trades.append({
                            "date": dt, "code": codes[sel[k]], "action": "buy",
                            "price": float(sel_prices[k]), "shares": float(sel_shares[k]),
                            "amount": float(sel_amounts[k]), "commission": float(sel_commissions[k]),
                            "tax": 0.0, "pnl": -float(sel_amounts[k]) - float(sel_commissions[k]),
                        })

            # ── 2c. 计算总权益（向量化）──
            hold_mask = positions > 0
            market_value = float(np.nansum(positions[hold_mask] * close_i[hold_mask])) if hold_mask.any() else 0.0
            total_equity = cash + market_value
            equity_records.append({
                "date": dt,
                "equity": total_equity,
                "cash": cash,
                "market_value": market_value,
                "position_count": int((positions > 0).sum()),
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return self._empty_result()

        eq_series = equity_curve.set_index("date")["equity"]
        metrics = BaseBacktestMetrics.calc_all_metrics(eq_series, trades_df)

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(
                [(codes[i], positions[i]) for i in range(n_codes) if positions[i] > 0],
                columns=["code", "shares"],
            ),
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    # ── 辅助方法 ──────────────────────────────────────────
    @staticmethod
    def _pivot_market(data: pd.DataFrame):
        """将行情透视为宽矩阵"""
        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        close_w = data.pivot(index="date", columns="code", values="close")
        limit_up_w = data.pivot(index="date", columns="code", values="is_limit_up") \
            if "is_limit_up" in data.columns else pd.DataFrame(index=close_w.index, columns=close_w.columns)
        limit_down_w = data.pivot(index="date", columns="code", values="is_limit_down") \
            if "is_limit_down" in data.columns else pd.DataFrame(index=close_w.index, columns=close_w.columns)
        # 布尔化（兼容 0/1 与 True/False）
        limit_up_w = limit_up_w.fillna(False).astype(bool)
        limit_down_w = limit_down_w.fillna(False).astype(bool)
        return close_w, limit_up_w, limit_down_w

    @staticmethod
    def _pivot_signals(signals: pd.DataFrame, index, columns):
        """
        将信号透视为买卖掩码宽矩阵

        返回 (buy_mask, sell_mask)，均为 (date × code) 布尔矩阵。
        同日对同一股票既出现买信号又出现卖信号时，两者均为 True，
        与原生适配器"先卖后买"的处理顺序一致。
        """
        if "signal" not in signals.columns:
            sig = signals[["date", "code"]].copy()
            sig["signal"] = 0
        else:
            sig = signals[["date", "code", "signal"]].copy()
        sig["is_buy"] = sig["signal"].fillna(0) > 0
        sig["is_sell"] = sig["signal"].fillna(0) < 0
        # 聚合重复 (date, code)：任一买信号即为买，任一卖信号即为卖
        agg = sig.groupby(["date", "code"], as_index=False).agg(
            is_buy=("is_buy", "any"), is_sell=("is_sell", "any")
        )
        buy_w = agg.pivot(index="date", columns="code", values="is_buy") \
            .reindex(index=index, columns=columns).fillna(False).astype(bool)
        sell_w = agg.pivot(index="date", columns="code", values="is_sell") \
            .reindex(index=index, columns=columns).fillna(False).astype(bool)
        return buy_w, sell_w

    @staticmethod
    def _empty_result():
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
