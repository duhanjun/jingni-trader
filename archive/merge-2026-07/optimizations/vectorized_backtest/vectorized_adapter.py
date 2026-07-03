"""
向量化回测引擎 — 借鉴 VectorBT 的设计思想
================================================

借鉴来源: VectorBT (https://vectorbt.dev)
核心思想: 将策略状态表示为 2D 数组 (日期 × 股票)，用 pandas/numpy 向量化
         操作替代 Python 逐日逐股循环，从而获得数量级性能提升。

与 jingni-trader 现有 native_adapter.py 的对比:
  - 现有实现: `for dt in dates: for code in stocks: ...` 双重 Python 循环
  - 本实现:   单次向量化矩阵运算，无 Python 级别逐股循环

适用场景: 周期性调仓型策略 (rebalance to target weights)，这是 A 股多因子
         选股策略最常见的形态。对于需要复杂订单逻辑 (限价单、部分成交) 的
         高频策略，仍建议使用事件驱动引擎。

本模块为验证代码，独立于 main 分支的现有引擎，不修改任何现有文件。
"""
from __future__ import annotations

import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


class VectorizedBacktester:
    """
    向量化回测器

    通过目标权重矩阵 + 价格矩阵一次性计算整段回测的权益曲线，
    避免逐日逐股的 Python 循环。

    交易模型说明 (与 native_adapter 保持一致的语义):
      - 调仓日在收盘价执行 (close price)
      - 买入加滑点: 实际成交价 = close * (1 + slippage)
      - 卖出减滑点: 实际成交价 = close * (1 - slippage)
      - 佣金: 双边 max(amount * commission_rate, min_commission)
      - 印花税: 仅卖出 amount * stamp_tax_rate
      - T+1: 通过权重滞后一天实现 (今日信号 → 明日开盘生效，
             这里简化为今日信号 → 今日收盘调仓，与 native 一致)
    """

    def __init__(
        self,
        init_capital: float = 1_000_000.0,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.0001,
        risk_free_rate: float = 0.03,
        trading_days: int = 252,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def run(
        self,
        prices: pd.DataFrame,
        target_weights: pd.DataFrame,
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, Any]:
        """
        执行向量化回测

        参数:
            prices: 收盘价矩阵 (日期 × 股票)，列为股票代码，行为日期
            target_weights: 目标权重矩阵 (日期 × 股票)。
                            仅在调仓日有值，其余位置可为 NaN。
                            权重之和应接近 1.0 (满仓) 或 <1.0 (留现金)。
            rebalance_dates: 显式调仓日。若为 None，则取 target_weights
                             中非全 NaN 的日期。

        返回:
            dict 包含:
              - equity_curve: pd.DataFrame(date, equity, cash, market_value)
              - returns: pd.Series 日收益率
              - turnover: pd.Series 调仓换手率
              - metrics: dict 绩效指标
              - trades: pd.DataFrame 成交记录 (调仓快照)
        """
        # ---- 1. 对齐索引/列 ----
        common_codes = prices.columns.intersection(target_weights.columns)
        if len(common_codes) == 0:
            raise ValueError("prices 与 target_weights 无共同股票代码")
        prices = prices[common_codes].sort_index()
        target_weights = target_weights[common_codes].reindex(prices.index)

        # ---- 2. 确定调仓日 & 前向填充权重 (持仓不动) ----
        if rebalance_dates is None:
            # 调仓日 = target_weights 中至少有一个非 NaN 的日期
            has_weight = target_weights.notna().any(axis=1)
            rebalance_dates = prices.index[has_weight]
        rebalance_mask = pd.Series(False, index=prices.index)
        rebalance_mask.loc[rebalance_dates] = True

        # 前向填充权重: 调仓后持有到下一次调仓
        weights = target_weights.ffill().fillna(0.0)
        # 现金权重 = 1 - 股票权重和
        stock_weight_sum = weights.sum(axis=1)
        cash_weight = (1.0 - stock_weight_sum).clip(lower=0.0)

        # ---- 3. 个股日收益率 ----
        daily_returns = prices.pct_change().fillna(0.0)

        # ---- 4. 组合日收益率 (向量化核心) ----
        # 持仓权重用 *上一日* 的权重 (今日收益由昨日持仓赚取)
        held_weights = weights.shift(1).fillna(0.0)
        held_cash_weight = cash_weight.shift(1).fillna(1.0)

        # 股票部分收益
        stock_pnl = (held_weights * daily_returns).sum(axis=1)
        # 现金无收益 (简化; 如需计入货币基金收益可在此加 cash_return)
        portfolio_gross_return = stock_pnl

        # ---- 5. 交易成本 (仅调仓日) ----
        prev_weights = weights.shift(1).fillna(0.0)
        # 单边换手率 = sum(|w_t - w_{t-1}|) / 2
        turnover = (weights.subtract(prev_weights).abs().sum(axis=1) / 2.0)
        turnover = turnover.where(rebalance_mask, 0.0)

        # 成本率: 买入侧 (佣金+滑点) + 卖出侧 (佣金+印花税+滑点)
        # 简化为换手率 * 综合成本率 (双边)
        # 买入成本 ≈ turnover_buy * (commission + slippage)
        # 卖出成本 ≈ turnover_sell * (commission + stamp_tax + slippage)
        # 对称近似: turnover * (2*commission + stamp_tax + 2*slippage)
        buy_turnover = weights.where(weights > prev_weights, 0.0).subtract(
            prev_weights, fill_value=0.0
        )
        buy_turnover = buy_turnover.clip(lower=0.0).sum(axis=1)
        sell_turnover = turnover  # 单边总和近似

        # 用矩阵算成本率
        cost_rate = (
            buy_turnover * (self.commission_rate + self.slippage)
            + sell_turnover * (self.commission_rate + self.stamp_tax_rate + self.slippage)
        )
        cost_rate = cost_rate.where(rebalance_mask, 0.0)

        portfolio_net_return = portfolio_gross_return - cost_rate

        # ---- 6. 权益曲线 ----
        equity = self.init_capital * (1.0 + portfolio_net_return).cumprod()
        # 市值 / 现金拆分 (近似: 用当日权重 × 当日权益)
        market_value = equity * stock_weight_sum
        cash = equity * cash_weight

        equity_curve = pd.DataFrame({
            "date": equity.index,
            "equity": equity.values,
            "cash": cash.values,
            "market_value": market_value.values,
        }).reset_index(drop=True)

        returns = portfolio_net_return.rename("return")

        # ---- 7. 调仓快照 (作为 trades 记录) ----
        trade_rows = []
        for dt in rebalance_dates:
            w = weights.loc[dt]
            active = w[w.abs() > 1e-6]
            eq = equity.loc[dt]
            for code, weight in active.items():
                trade_rows.append({
                    "date": dt,
                    "code": code,
                    "weight": float(weight),
                    "target_value": float(eq * weight),
                })
        trades_df = pd.DataFrame(trade_rows)

        # ---- 8. 绩效指标 ----
        metrics = self._calc_metrics(equity, returns, turnover)

        return {
            "equity_curve": equity_curve,
            "returns": returns,
            "turnover": turnover,
            "metrics": metrics,
            "trades": trades_df,
            "rebalance_dates": list(rebalance_dates),
            "n_rebalances": len(rebalance_dates),
        }

    # ------------------------------------------------------------------
    # 绩效指标 (与 BaseBacktestMetrics 对齐，并补充 Sortino/换手率)
    # ------------------------------------------------------------------
    def _calc_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        turnover: pd.Series,
    ) -> Dict[str, float]:
        if len(equity) < 2:
            return {}

        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        n_years = len(equity) / self.trading_days
        annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        vol = float(returns.std() * np.sqrt(self.trading_days))
        sharpe = float((annual_return - self.risk_free_rate) / vol) if vol > 0 else 0.0

        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = float(drawdown.min())
        calmar = float(annual_return / abs(max_dd)) if max_dd < 0 else 0.0

        neg = returns[returns < 0]
        downside = float(neg.std() * np.sqrt(self.trading_days)) if len(neg) > 1 else 0.0
        sortino = float((annual_return - self.risk_free_rate) / downside) if downside > 0 else 0.0

        win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

        # 年化换手率 (单边)
        rebal_turnover = turnover[turnover > 0]
        avg_turnover = float(rebal_turnover.mean()) if len(rebal_turnover) > 0 else 0.0
        n_rebal = len(rebal_turnover)
        annual_turnover = avg_turnover * (n_rebal / n_years) if n_years > 0 else 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "avg_turnover": avg_turnover,
            "annual_turnover": annual_turnover,
            "n_rebalances": int(n_rebal),
        }


# ----------------------------------------------------------------------
# 辅助: 从信号 DataFrame 构造目标权重矩阵
# ----------------------------------------------------------------------
def signals_to_target_weights(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    top_pct: float = 0.2,
    equal_weight: bool = True,
) -> pd.DataFrame:
    """
    将 (code, date, signal) 长表信号转换为 (date × code) 目标权重矩阵

    策略: 每个 date 取 signal 排名前 top_pct 的股票，等权配置。

    参数:
        signals: 长表，至少含 code, date, signal 列
        prices:  价格矩阵 (用于对齐 index/columns)
        top_pct: 选取前多少比例的股票
        equal_weight: 是否等权; False 时按 signal 大小加权
    """
    pivot = signals.pivot_table(index="date", columns="code", values="signal")
    pivot = pivot.reindex(index=prices.index, columns=prices.columns)

    # 每个 date 的排名 (pct)
    rank = pivot.rank(axis=1, pct=True)
    # 选股掩码: 排名 >= 1 - top_pct
    selected = rank >= (1.0 - top_pct)

    weights = pd.DataFrame(0.0, index=pivot.index, columns=pivot.columns)
    if equal_weight:
        # 等权: 每行 1/选中数
        n_selected = selected.sum(axis=1).replace(0, np.nan)
        weights = selected.astype(float).div(n_selected, axis=0).fillna(0.0)
    else:
        # signal 加权
        pos_signal = pivot.clip(lower=0.0)
        row_sum = pos_signal.where(selected).sum(axis=1).replace(0, np.nan)
        weights = pos_signal.where(selected).div(row_sum, axis=0).fillna(0.0)

    # 仅在调仓日保留权重，其余置 NaN (供 ffill 使用)
    # 调仓日 = 有任何选中的日期
    has_selection = selected.any(axis=1)
    weights = weights.where(has_selection, np.nan)
    return weights


if __name__ == "__main__":
    # 简易自测: 构造合成数据并跑通
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    codes = [f"S{i:03d}.SZ" for i in range(20)]
    px = pd.DataFrame(
        10 * np.cumprod(1 + np.random.normal(0, 0.01, (len(dates), len(codes))), axis=0),
        index=dates, columns=codes,
    )
    sig = pd.DataFrame({
        "date": np.tile(dates, len(codes)),
        "code": np.repeat(codes, len(dates)),
        "signal": np.random.uniform(-1, 1, len(dates) * len(codes)),
    })
    tw = signals_to_target_weights(sig, px, top_pct=0.2)
    bt = VectorizedBacktester()
    res = bt.run(px, tw)
    print("metrics:", res["metrics"])
    print("equity tail:\n", res["equity_curve"].tail())
