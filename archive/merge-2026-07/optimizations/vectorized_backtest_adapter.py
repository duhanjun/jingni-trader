"""
向量化回测引擎适配器（VectorBT 思想）

借鉴来源：VectorBT —— 将策略表示为 NumPy 矩阵，用向量化运算替代逐 bar 循环。
参考：https://vectorbt.dev/  (Portfolio.from_signals 矩阵化回测)

核心思想：
    传统事件驱动回测（如 jingni-trader 的 native_adapter.py）通过
    `for dt in dates: for row in day_signal.iterrows():` 逐行处理，
    在大股票池 + 长回测周期下性能极差。

    本实现将行情与信号 pivot 为 (date × code) 矩阵，用 pandas/numpy
    的 groupby/矩阵运算一次性计算目标持仓、净值曲线，避免 Python 层循环。

与原 native_adapter 的差异（重要）：
    - 原实现：买入预算 = cash * 0.95 / n_buy（路径依赖，逐日递推现金）
    - 本实现：目标权重 = 等权 1/n_held（目标组合法，Qlib TopkDropout 同款）
      这是向量化回测的标准做法，消除了现金路径依赖，可整体向量化。
    - 两者均为合法回测方法，本实现侧重“研究阶段快速验证”，
      实盘/精确资金曲线仍建议用事件驱动引擎。

支持 A 股规则：
    - T+1：当日买入次日才能卖出（通过持仓矩阵 shift 实现）
    - 涨跌停过滤：涨停不买、跌停不卖
    - 佣金 / 印花税 / 滑点
"""
from typing import Dict, Any
import numpy as np
import pandas as pd


class VectorizedAdapter:
    """向量化回测适配器"""

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
        max_positions: int = 50,
    ) -> Dict[str, Any]:
        if data.empty or signals.empty:
            return self._empty_result()

        # ── 1. pivot 为 (date × code) 矩阵 ──────────────────────
        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        close = data.pivot_table(index="date", columns="code", values="close")
        # 涨跌停标记（若不存在则视为非涨跌停）
        # 注意：pivot_table 后 fillna(False) 可能产生 object dtype，需显式转 bool
        if "is_limit_up" in data.columns:
            limit_up = data.pivot_table(
                index="date", columns="code", values="is_limit_up"
            ).fillna(False).astype(bool)
        else:
            limit_up = pd.DataFrame(False, index=close.index, columns=close.columns)
        if "is_limit_down" in data.columns:
            limit_down = data.pivot_table(
                index="date", columns="code", values="is_limit_down"
            ).fillna(False).astype(bool)
        else:
            limit_down = pd.DataFrame(False, index=close.index, columns=close.columns)

        # 信号矩阵：1 买入持有, -1 卖出, 0 无操作
        sig = signals.pivot_table(
            index="date", columns="code", values="signal", aggfunc="last"
        ).reindex(index=close.index, columns=close.columns).fillna(0)

        # ── 2. 计算目标持仓（向量化）────────────────────────────
        # hold_mask: True 表示该日持有该股票
        # 逻辑：买入信号 -> 进入持有；卖出信号 -> 退出持有
        buy_sig = (sig > 0).astype(int)
        sell_sig = (sig < 0).astype(int)

        # 持有状态：用 cummax 模拟“一旦买入则持有，直到卖出”
        # enter = 1, exit = -1；持有 = (累计 enter > 累计 exit)
        hold_raw = buy_sig - sell_sig
        # 每只股票的累计净买入计数 > 0 即视为持有
        cum_hold = hold_raw.cumsum().clip(lower=0)
        hold_mask = cum_hold > 0

        # T+1：当日买入当日不可卖 -> 持仓状态延后一日生效用于“可卖”判定
        # 这里用 shift(1) 表示“昨日已持有”才允许今日按卖出信号退出
        if t_plus_1:
            sellable = hold_mask.shift(1).fillna(False)
            # 重新计算 hold_mask：只有 sellable 的卖出才生效
            effective_sell = sell_sig & sellable
            hold_raw2 = buy_sig - effective_sell.astype(int)
            cum_hold2 = hold_raw2.cumsum().clip(lower=0)
            hold_mask = cum_hold2 > 0

        # 涨跌停过滤：涨停日不买入（从 hold_mask 剔除当日新增），
        # 跌停日不卖出（保留持仓）
        if price_limit:
            # 涨停日不可建仓 -> 把该日 buy_sig 置 0 后重算
            buy_sig_filtered = buy_sig.where(~limit_up, 0)
            # 跌停日卖出信号失效
            sell_sig_filtered = sell_sig.where(~limit_down, 0)
            if t_plus_1:
                effective_sell = sell_sig_filtered & sellable
                hold_raw2 = buy_sig_filtered - effective_sell.astype(int)
            else:
                hold_raw2 = buy_sig_filtered - sell_sig_filtered.astype(int)
            cum_hold2 = hold_raw2.cumsum().clip(lower=0)
            hold_mask = cum_hold2 > 0

        # 限制最大持仓数：每日只保留前 max_positions 只（按信号优先）
        if max_positions and max_positions > 0:
            # 每日持有数 > max_positions 时，保留 buy_sig 优先的
            n_held = hold_mask.sum(axis=1)
            over_mask = n_held > max_positions
            if over_mask.any():
                # 对超额日，按 close 排序保留前 max_positions（确定性）
                for dt in close.index[over_mask]:
                    held_codes = hold_mask.loc[dt][hold_mask.loc[dt]].index
                    keep = held_codes[:max_positions]
                    hold_mask.loc[dt] = False
                    hold_mask.loc[dt, keep] = True

        # ── 3. 目标权重（等权）──────────────────────────────────
        n_held = hold_mask.sum(axis=1).replace(0, np.nan)
        target_weight = hold_mask.div(n_held, axis=0).fillna(0)

        # T+1：实际持仓 = 昨日目标权重（今日开盘才能调整）
        # 用 shift(1) 模拟“信号在 T 日收盘产生，T+1 日开盘执行”
        actual_weight = target_weight.shift(1).fillna(0)

        # ── 4. 净值曲线（向量化）────────────────────────────────
        # 个股日收益率
        stock_ret = close.pct_change().fillna(0)
        # 组合日收益率 = sum(weight_t-1 * ret_t)
        port_ret = (actual_weight * stock_ret).sum(axis=1)
        # 扣除交易成本：权重变动 * (commission + slippage) + 卖出 * stamp_tax
        turnover = actual_weight.diff().abs().fillna(actual_weight.iloc[0])
        # 买入成本 = turnover 的正部分 * (commission_rate + slippage)
        # 卖出成本 = turnover 的负部分 * (commission_rate + stamp_tax_rate + slippage)
        weight_delta = actual_weight.diff().fillna(actual_weight.iloc[0])
        buy_turnover = turnover.where(weight_delta > 0, 0)
        sell_turnover = turnover.where(weight_delta < 0, 0)
        cost = (
            buy_turnover * (commission_rate + slippage)
            + sell_turnover * (commission_rate + stamp_tax_rate + slippage)
        ).sum(axis=1)  # 汇总为每日总成本（Series）
        port_ret_net = port_ret - cost

        # 净值
        equity = init_capital * (1 + port_ret_net).cumprod()

        equity_curve = pd.DataFrame({
            "date": equity.index,
            "equity": equity.values,
            "cash": (init_capital * (1 - actual_weight.sum(axis=1))).values,
            "market_value": (
                init_capital * actual_weight.shift(1).fillna(0) * (1 + stock_ret)
            ).sum(axis=1).values,
            "position_count": hold_mask.sum(axis=1).values,
        }).reset_index(drop=True)

        # ── 5. 成交记录（从权重变动还原）────────────────────────
        trades = self._extract_trades(
            actual_weight, close, init_capital,
            commission_rate, stamp_tax_rate, slippage,
        )

        # ── 6. 绩效指标 ────────────────────────────────────────
        eq_series = equity_curve.set_index("date")["equity"]
        metrics = self._calc_metrics(eq_series, trades, init_capital)

        return {
            "trades": trades,
            "positions": pd.DataFrame(),
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    def _extract_trades(
        self, weight: pd.DataFrame, close: pd.DataFrame, init_capital: float,
        commission_rate: float, stamp_tax_rate: float, slippage: float,
    ) -> pd.DataFrame:
        """从权重变动还原成交记录（向量化）"""
        delta = weight.diff().fillna(weight.iloc[0])
        # 堆叠为长表
        delta_long = delta.stack().reset_index()
        delta_long.columns = ["date", "code", "weight_delta"]
        delta_long = delta_long[delta_long["weight_delta"].abs() > 1e-8].copy()
        if delta_long.empty:
            return pd.DataFrame(
                columns=["date", "code", "action", "price", "shares",
                         "amount", "commission", "tax", "pnl"]
            )
        close_long = close.stack().reset_index()
        close_long.columns = ["date", "code", "close"]
        delta_long = delta_long.merge(close_long, on=["date", "code"], how="left")

        # 金额 = 权重变动 * 总资产（用 init_capital 近似，避免路径依赖）
        delta_long["amount"] = (delta_long["weight_delta"].abs()
                                * init_capital).round(2)
        delta_long["action"] = np.where(
            delta_long["weight_delta"] > 0, "buy", "sell"
        )
        delta_long["price"] = delta_long["close"]
        delta_long["shares"] = (delta_long["amount"] / delta_long["price"]).astype(int)
        delta_long["commission"] = np.maximum(
            delta_long["amount"] * commission_rate, 5
        )
        delta_long["tax"] = np.where(
            delta_long["action"] == "sell",
            delta_long["amount"] * stamp_tax_rate,
            0,
        )
        delta_long["pnl"] = np.where(
            delta_long["action"] == "sell", delta_long["amount"], -delta_long["amount"]
        )
        return delta_long[
            ["date", "code", "action", "price", "shares",
             "amount", "commission", "tax", "pnl"]
        ]

    def _calc_metrics(
        self, equity: pd.Series, trades: pd.DataFrame, init_capital: float
    ) -> Dict[str, Any]:
        from datetime import datetime
        if len(equity) < 2:
            return {}
        returns = equity.pct_change().dropna()
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        n_years = len(equity) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        max_dd = (equity / equity.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        neg = returns[returns < 0]
        downside = neg.std() * np.sqrt(252) if len(neg) >= 2 else 0
        sortino = (annual_return - 0.03) / downside if downside > 0 else 0
        win_rate = (trades["pnl"] > 0).mean() if not trades.empty else 0
        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "sortino_ratio": float(sortino),
            "win_rate": float(win_rate),
            "total_trades": int(len(trades)),
            "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _empty_result(self):
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
