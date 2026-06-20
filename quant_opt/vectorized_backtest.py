"""
向量化回测引擎 —— 借鉴 vectorbt 向量化回测范式

jingni-trader 现有 native_adapter.py 采用逐日 Python for 循环：
    for dt in dates:
        day_data = data[data['date'] == dt]   # O(n) 过滤，循环内重复执行
        ...逐只股票处理买卖...
这种事件驱动方式准确但极慢（O(交易日 × 股票数) 的 Python 层开销）。

vectorbt 的核心思想：将回测转化为矩阵运算，用 NumPy/Pandas 向量化一次性计算，
性能提升 100-1000 倍。借鉴来源：
- vectorbt: https://github.com/polakowo/vectorbt
- "Backtesting at Scale": https://www.finantrix.com/...

本模块实现一个目标权重（target-weight）向量化回测：
1. 将信号转为目标权重矩阵（date × code）
2. T+1 滞后：今日信号 -> 次日持仓
3. 涨跌停过滤：涨停不买入，跌停不卖出
4. 向量化计算持仓市值、现金、净值曲线
5. 向量化计算交易成本（佣金 + 印花税 + 滑点）

与 native_adapter 的差异：
- native_adapter 用"等额预算"建仓（cash * 0.95 / n_buy）
- 本引擎用"等权目标权重"建仓（1/n_buy），更接近实盘量化常用范式
- 两者绩效指标会有差异，但趋势一致；本引擎优势在速度
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("vectorized-backtest")


class VectorizedBacktestEngine:
    """
    向量化回测引擎

    用法:
        engine = VectorizedBacktestEngine()
        result = engine.run(data, signals)
        # result 含 equity_curve, trades, metrics
    """

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1e6,
        commission_rate: float = 0.00025,
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        max_position_pct: float = 0.05,
    ) -> Dict[str, Any]:
        """
        向量化回测

        参数:
            data: 行情数据，含 code, date, close, 以及可选 is_limit_up, is_limit_down
            signals: 交易信号，含 code, date, signal (1买入, -1卖出, 0持仓)
            init_capital: 初始资金
            commission_rate: 佣金费率（双边）
            min_commission: 最小佣金（元）
            stamp_tax_rate: 印花税率（仅卖出）
            slippage: 滑点比例
            t_plus_1: 是否启用 T+1（今日买入次日可卖）
            price_limit: 是否启用涨跌停限制
            max_position_pct: 单只股票最大权重

        返回:
            {
                "equity_curve": DataFrame(date, equity, cash, market_value, position_count),
                "trades": DataFrame,
                "positions": DataFrame,
                "metrics": dict,
            }
        """
        if data.empty or signals.empty:
            return self._empty_result()

        # ── 1. 数据预处理：构建 (date × code) 透视表 ──
        data = data.sort_values(["date", "code"]).copy()
        signals = signals.sort_values(["date", "code"]).copy()

        # 确保日期类型一致
        data["date"] = pd.to_datetime(data["date"])
        signals["date"] = pd.to_datetime(signals["date"])

        # 对齐日期范围：取信号覆盖的交易日
        all_dates = sorted(data["date"].unique())
        sig_dates = sorted(signals["date"].unique())
        # 用信号的日期范围，但限制在数据范围内
        start = max(min(all_dates), min(sig_dates))
        end = min(max(all_dates), max(sig_dates))
        dates = [d for d in all_dates if start <= d <= end]
        if not dates:
            return self._empty_result()

        # 透视收盘价 (date × code)
        close_pivot = data.pivot_table(index="date", columns="code", values="close")
        close_pivot = close_pivot.reindex(dates)

        # 涨跌停标记（若存在）
        if price_limit and "is_limit_up" in data.columns:
            limit_up = data.pivot_table(
                index="date", columns="code", values="is_limit_up", aggfunc="max"
            ).reindex(dates).fillna(False).astype(bool)
        else:
            limit_up = pd.DataFrame(False, index=dates, columns=close_pivot.columns)

        if price_limit and "is_limit_down" in data.columns:
            limit_down = data.pivot_table(
                index="date", columns="code", values="is_limit_down", aggfunc="max"
            ).reindex(dates).fillna(False).astype(bool)
        else:
            limit_down = pd.DataFrame(False, index=dates, columns=close_pivot.columns)

        # ── 2. 信号转目标权重矩阵 ──
        # signal: 1 -> 买入目标, -1 -> 清仓, 0 -> 维持
        sig_pivot = signals.pivot_table(
            index="date", columns="code", values="signal", aggfunc="last"
        ).reindex(dates).fillna(0)

        # 对齐列
        common_codes = close_pivot.columns.intersection(sig_pivot.columns)
        close_pivot = close_pivot[common_codes]
        sig_pivot = sig_pivot[common_codes]
        limit_up = limit_up[common_codes] if not limit_up.empty else limit_up
        limit_down = limit_down[common_codes] if not limit_down.empty else limit_down

        # 计算每日目标权重：等权分配给买入信号
        # buy_mask: 当日有买入信号的股票
        buy_mask = (sig_pivot > 0).astype(float)
        sell_mask = (sig_pivot < 0).astype(float)

        # 每日买入股票数
        n_buy_per_day = buy_mask.sum(axis=1).replace(0, np.nan)
        # 目标权重：等权，受 max_position_pct 上限
        target_weight = buy_mask.div(n_buy_per_day, axis=0).fillna(0)
        target_weight = target_weight.clip(upper=max_position_pct)

        # ── 3. T+1 滞后与涨跌停过滤 ──
        # 今日信号决定明日目标持仓
        if t_plus_1:
            # 目标权重滞后一天生效
            effective_weight = target_weight.shift(1).fillna(0)
            # 卖出信号也滞后一天
            effective_sell = sell_mask.shift(1).fillna(0)
        else:
            effective_weight = target_weight.copy()
            effective_sell = sell_mask.copy()

        # 涨跌停过滤：涨停日不买入（目标权重置0），跌停日不卖出（保留原持仓）
        # 注意：effective_weight 是"想买入"的权重，若当日涨停则无法买入
        if price_limit:
            # 涨停无法买入：将涨停日的目标权重清零
            buy_blocked = limit_up.reindex(
                index=effective_weight.index, columns=effective_weight.columns
            ).fillna(False).astype(bool)
            effective_weight = effective_weight.where(~buy_blocked, 0)
            # 跌停无法卖出：将跌停日的卖出信号清零（保留持仓）
            sell_blocked = limit_down.reindex(
                index=effective_sell.index, columns=effective_sell.columns
            ).fillna(False).astype(bool)
            effective_sell = effective_sell.where(~sell_blocked, 0)

        # ── 4. 计算实际持仓矩阵（向量化）──
        # 持仓权重演化：遇到卖出信号则清仓，否则维持或建仓
        # 用迭代方式处理（持仓有状态依赖），但仅在权重矩阵上迭代，远快于逐股票逐日
        n_dates = len(dates)
        n_codes = len(common_codes)
        held_weight = np.zeros((n_dates, n_codes))
        eff_w = effective_weight.values
        eff_s = effective_sell.values

        for i in range(n_dates):
            if i == 0:
                held_weight[i] = eff_w[i]
            else:
                # 先处理卖出：卖出信号位置清零
                prev = held_weight[i - 1].copy()
                sold = eff_s[i] > 0
                prev[sold] = 0
                # 再处理买入：有目标权重的建仓（覆盖）
                has_target = eff_w[i] > 0
                prev[has_target] = eff_w[i][has_target]
                held_weight[i] = prev

        held_weight_df = pd.DataFrame(held_weight, index=dates, columns=common_codes)

        # ── 5. 逐日计算权益曲线（跟踪持仓资金随价格变化）──
        # 关键修正：持仓市值必须随价格变动，而非固定为 weight × equity
        # 跟踪每只股票的持仓资金（position_capital），非信号日按价格涨跌更新
        close_filled = pd.DataFrame(
            close_pivot.values, index=dates, columns=common_codes
        ).ffill()
        price_arr = close_filled.values  # (n_dates, n_codes)

        equity = init_capital
        cash = init_capital
        # 持仓资金矩阵（每只股票当前投入的资金）
        pos_capital = np.zeros(n_codes)
        # 当前持仓权重（用于判断是否持有）
        held_w = np.zeros(n_codes)
        prev_prices = None

        equity_records = []
        trade_records = []

        for i, dt in enumerate(dates):
            prices = price_arr[i]

            # 5a. 非首日：按价格变化更新持仓资金（市值随价格涨跌）
            if prev_prices is not None:
                # 避免除零：停牌时价格不变
                safe_prev = np.where(prev_prices > 0, prev_prices, 1.0)
                price_ratio = np.where(
                    np.isfinite(prices) & np.isfinite(prev_prices),
                    prices / safe_prev,
                    1.0,
                )
                pos_capital = pos_capital * price_ratio

            # 5b. 当前总权益 = 现金 + 持仓总市值
            position_value = pos_capital.sum()
            equity = cash + position_value

            # 5c. 处理调仓（仅信号日触发交易）
            # 注意：用 effective_weight（仅信号日非零），而非 held_weight_df（持久化）
            cur_eff_w = effective_weight.loc[dt].values  # 信号日才 > 0
            cur_eff_s = effective_sell.loc[dt].values if not effective_sell.empty else np.zeros(n_codes)

            # 卖出：卖出信号位置清仓
            sell_mask = cur_eff_s > 0
            if sell_mask.any():
                sell_amt_arr = pos_capital[sell_mask].copy()
                sell_cost_arr = np.maximum(sell_amt_arr * commission_rate, min_commission)
                sell_tax_arr = sell_amt_arr * stamp_tax_rate
                cash += sell_amt_arr.sum() - sell_cost_arr.sum() - sell_tax_arr.sum()
                for j in np.where(sell_mask)[0]:
                    if pos_capital[j] > 0:
                        trade_records.append({
                            "date": dt, "code": common_codes[j], "action": "sell",
                            "amount": float(pos_capital[j]),
                            "commission": float(max(pos_capital[j] * commission_rate, min_commission)),
                            "tax": float(pos_capital[j] * stamp_tax_rate),
                            "slippage": 0.0,
                        })
                pos_capital[sell_mask] = 0
                held_w[sell_mask] = 0

            # 买入/调仓：有目标权重的股票调至目标权重
            buy_mask = cur_eff_w > 0
            if buy_mask.any():
                # 重新计算权益（卖出后）
                equity = cash + pos_capital.sum()
                target_capital = cur_eff_w * equity
                # 仅对信号标记的股票调仓至目标
                for j in np.where(buy_mask)[0]:
                    target = target_capital[j]
                    current = pos_capital[j]
                    delta = target - current
                    if abs(delta) < 1:  # 忽略微小调整
                        pos_capital[j] = target
                        continue
                    if delta > 0:
                        # 买入
                        cost = max(delta * commission_rate, min_commission) + delta * slippage
                        cash -= delta + cost
                        trade_records.append({
                            "date": dt, "code": common_codes[j], "action": "buy",
                            "amount": float(delta),
                            "commission": float(max(delta * commission_rate, min_commission)),
                            "tax": 0.0,
                            "slippage": float(delta * slippage),
                        })
                    else:
                        # 卖出超配部分
                        sell_delta = -delta
                        cost = max(sell_delta * commission_rate, min_commission)
                        tax = sell_delta * stamp_tax_rate
                        cash += sell_delta - cost - tax
                        trade_records.append({
                            "date": dt, "code": common_codes[j], "action": "sell",
                            "amount": float(sell_delta),
                            "commission": float(cost),
                            "tax": float(tax),
                            "slippage": 0.0,
                        })
                    pos_capital[j] = target
                    held_w[j] = cur_eff_w[j]

            # 5d. 记录当日权益
            position_value = pos_capital.sum()
            equity = cash + position_value

            equity_records.append({
                "date": dt,
                "equity": float(equity),
                "cash": float(cash),
                "market_value": float(position_value),
                "position_count": int((pos_capital > 0).sum()),
            })

            prev_prices = prices.copy()

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trade_records)

        # ── 7. 计算绩效指标 ──
        metrics = self._calc_metrics(equity_curve, trades_df, init_capital)

        # 持仓快照（最后一日）
        last_weight = held_weight_df.iloc[-1]
        positions = pd.DataFrame({
            "code": common_codes,
            "weight": last_weight.values,
        })
        positions = positions[positions["weight"] > 0]

        return {
            "equity_curve": equity_curve,
            "trades": trades_df,
            "positions": positions,
            "metrics": metrics,
        }

    @staticmethod
    def _calc_metrics(
        equity_curve: pd.DataFrame, trades: pd.DataFrame, init_capital: float
    ) -> Dict[str, Any]:
        """计算绩效指标（与 base_backtest.py 对齐）"""
        if equity_curve.empty or len(equity_curve) < 2:
            return {}

        eq = equity_curve.set_index("date")["equity"]
        returns = eq.pct_change().dropna()

        total_return = eq.iloc[-1] / eq.iloc[0] - 1
        n_years = len(eq) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        max_drawdown = (eq / eq.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino
        neg_ret = returns[returns < 0]
        downside_std = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 1 else 0
        sortino = (annual_return - 0.03) / downside_std if downside_std > 0 else 0

        # 胜率（基于日收益）
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "sortino_ratio": float(sortino),
            "win_rate": float(win_rate),
            "total_trades": int(len(trades)),
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "equity_curve": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "metrics": {},
        }
