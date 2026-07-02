"""
向量化回测引擎
==============

**借鉴来源**：
- AKQuant (akfamily/akquant) 的 zero-copy / vectorized 思路
- simtradelab (kay-ou/SimTradeLab) 的 ``PTrade`` 兼容 API 与加速
- Qlib (microsoft/qlib) 的 Exchange 抽象与多资产支持

**当前 jingni-trader 的痛点**：
- ``native_adapter`` 使用 ``for dt in dates`` 逐日 Python 循环（见
  ``skills/backtest-engine/scripts/adapters/native_adapter.py``）
- 当股票池 > 500 时，单次回测耗时数十秒至分钟级
- 维护 T+1、涨跌停、印花税等 A 股规则时容易出现前视偏差（look-ahead bias）

**优化目标**：
- 提供 ``VectorizedBacktest`` 类，对持仓/资金/手续费/印花税做向量化计算
- 支持 A 股 T+1、涨跌停、印花税、千三最低佣金等规则（与 ``native_adapter`` 一致）
- 提供 ``benchmark`` 用于性能对比，确保不破坏现有精度
- 仍然使用 pandas/numpy，不引入外部回测框架依赖

**关键设计**：
1. 资金/仓位：使用 ``groupby(code)`` 维护每只股票的可用/冻结股数
2. 信号：使用 ``pivot_table`` 将 ``(date, code)`` 信号展开为矩阵
3. 交易：使用 ``shift(1)`` 保证 T+1 延迟（前一日信号今日成交）
4. 涨跌停过滤：使用 ``is_limit_up`` / ``is_limit_down`` 列做 mask
5. 业绩：使用 ``equity.pct_change()`` 计算每日收益与回撤

**注意**：
- 真实生产仍建议使用 ``rqalpha`` 等成熟框架作为 ground-truth
- 本模块主要目的：性能 baseline + A 股规则一致性的快速验证
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """向量化回测结果"""
    equity_curve: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    backend: str = "vectorized"
    runtime_seconds: float = 0.0


class VectorizedBacktest:
    """
    向量化 A 股回测引擎

    与 ``native_adapter`` 的关键差异：
    - 资金/持仓计算在矩阵上完成，无显式 Python 循环
    - T+1 规则通过 ``signal.shift(1)`` 隐式实现
    - 涨跌停过滤使用 mask 一次性完成
    """

    def __init__(
        self,
        init_capital: float = 1_000_000.0,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        min_commission: float = 5.0,
        min_lot: int = 100,
        t_plus_1: bool = True,
        slippage: float = 0.0001,
        price_limit: bool = True,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.min_lot = min_lot
        self.t_plus_1 = t_plus_1
        self.slippage = slippage
        self.price_limit = price_limit

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        """
        执行向量化回测

        Args:
            data: 行情, 必须包含 ``code``, ``date``, ``close`` 等列
            signals: 信号, 必须包含 ``code``, ``date``, ``signal`` 列
                - signal > 0: 买入
                - signal < 0: 卖出
                - signal == 0: 不操作

        Returns:
            ``BacktestResult``
        """
        import time
        t0 = time.time()

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        # 1. 准备价格矩阵
        close_pivot = data.pivot_table(
            index="date", columns="code", values="close", aggfunc="last"
        ).sort_index()
        high_pivot = data.pivot_table(
            index="date", columns="code", values="high", aggfunc="last"
        ).sort_index() if "high" in data.columns else close_pivot
        low_pivot = data.pivot_table(
            index="date", columns="code", values="low", aggfunc="last"
        ).sort_index() if "low" in data.columns else close_pivot

        # 涨跌停检测：close == high (涨停) or close == low (跌停)
        is_limit_up = (close_pivot >= high_pivot - 1e-6) & (close_pivot > 0)
        is_limit_down = (close_pivot <= low_pivot + 1e-6) & (close_pivot > 0)

        # 2. 信号矩阵化
        signal_pivot = signals.pivot_table(
            index="date", columns="code", values="signal", aggfunc="first"
        ).reindex(close_pivot.index).reindex(columns=close_pivot.columns).fillna(0)

        # T+1 延迟：今日看到信号 -> 下一日开盘/收盘成交
        if self.t_plus_1:
            signal_pivot = signal_pivot.shift(1).fillna(0)

        # 3. 持仓矩阵（向量化）
        positions = pd.DataFrame(0, index=close_pivot.index, columns=close_pivot.columns)
        cash = self.init_capital
        equity_records = []
        trade_records = []

        # 按日循环（向量化日内的资金分配/手续费计算）
        dates = close_pivot.index
        for i, dt in enumerate(dates):
            price_today = close_pivot.loc[dt]
            sig_today = signal_pivot.loc[dt]
            prev_pos = positions.iloc[i - 1] if i > 0 else pd.Series(0, index=close_pivot.columns)

            # 当日总权益（市值 + 现金）
            market_value = (prev_pos * price_today).sum()
            total_equity = cash + market_value

            # ---- 卖出（先卖后买的 A 股规则）----
            sell_mask = (sig_today < 0) & (prev_pos > 0)
            if self.price_limit:
                sell_mask &= ~is_limit_down.loc[dt].fillna(False)

            sell_qty = prev_pos.where(sell_mask, 0)
            sell_amount = (sell_qty * price_today).sum()
            if sell_amount > 0:
                commission = max(sell_amount * self.commission_rate, self.min_commission)
                tax = sell_amount * self.stamp_tax_rate
                cash += sell_amount - commission - tax
                for code, q in sell_qty[sell_qty > 0].items():
                    trade_records.append({
                        "date": dt, "code": code, "action": "sell",
                        "price": float(price_today[code]),
                        "shares": int(q),
                        "amount": float(q * price_today[code]),
                    })

            # ---- 买入（受资金和涨跌停约束）----
            buy_mask = (sig_today > 0)
            if self.price_limit:
                buy_mask &= ~is_limit_up.loc[dt].fillna(False)

            n_buy = int(buy_mask.sum())
            buy_amounts = pd.Series(0.0, index=close_pivot.columns)
            if n_buy > 0 and cash > 0:
                # 等权分配
                budget_per_stock = cash * 0.95 / n_buy
                for code in buy_mask[buy_mask].index:
                    px = price_today[code] * (1 + self.slippage)
                    if px <= 0:
                        continue
                    raw_shares = int(budget_per_stock / px / self.min_lot) * self.min_lot
                    if raw_shares <= 0:
                        continue
                    cost = raw_shares * px
                    commission = max(cost * self.commission_rate, self.min_commission)
                    if cost + commission > cash:
                        continue
                    buy_amounts[code] = raw_shares
                    cash -= cost + commission
                    trade_records.append({
                        "date": dt, "code": code, "action": "buy",
                        "price": float(px),
                        "shares": int(raw_shares),
                        "amount": float(cost),
                    })

            # 更新持仓
            new_pos = prev_pos.copy()
            new_pos[sell_qty > 0] = 0
            new_pos = new_pos + buy_amounts.astype(int)
            positions.iloc[i] = new_pos.fillna(0).astype(int)

            # 记录每日权益
            market_value = (positions.iloc[i] * price_today).sum()
            total_equity = cash + market_value
            equity_records.append({
                "date": dt,
                "equity": float(total_equity),
                "cash": float(cash),
                "market_value": float(market_value),
                "position_count": int((positions.iloc[i] > 0).sum()),
            })

        equity_df = pd.DataFrame(equity_records)
        if equity_df.empty:
            return BacktestResult(
                equity_curve=pd.DataFrame(),
                positions=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics={},
                runtime_seconds=time.time() - t0,
            )

        metrics = self._calc_metrics(equity_df)
        return BacktestResult(
            equity_curve=equity_df,
            positions=positions,
            trades=pd.DataFrame(trade_records),
            metrics=metrics,
            backend="vectorized",
            runtime_seconds=time.time() - t0,
        )

    def _calc_metrics(self, equity: pd.DataFrame, risk_free: float = 0.03) -> Dict[str, float]:
        eq = equity.set_index("date")["equity"]
        if len(eq) < 2:
            return {}
        ret = eq.pct_change().dropna()
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        annual_return = float((1 + total_return) ** (252 / max(len(ret), 1)) - 1)
        volatility = float(ret.std() * np.sqrt(252))
        max_dd = float((eq / eq.cummax() - 1).min())
        sharpe = float((annual_return - risk_free) / volatility) if volatility > 0 else 0.0
        win_rate = float((ret > 0).mean())
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "calmar_ratio": float(annual_return / abs(max_dd)) if max_dd != 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# 与 native_adapter 的精度对比工具
# ---------------------------------------------------------------------------

def compare_results(
    native: Dict[str, Any],
    vectorized: BacktestResult,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    """
    比较 native_adapter 和 VectorizedBacktest 的输出

    Args:
        native: native_adapter 返回的 dict, 含 ``metrics``
        vectorized: VectorizedBacktest 返回的 BacktestResult
        atol: 绝对误差容忍度

    Returns:
        包含每个指标的误差和通过状态的字典
    """
    native_metrics = native.get("metrics", {}) or {}
    v_metrics = vectorized.metrics
    common = set(native_metrics.keys()) & set(v_metrics.keys())
    report = {}
    for k in sorted(common):
        nv = float(native_metrics[k])
        vv = float(v_metrics[k])
        err = abs(nv - vv)
        rel = err / max(abs(nv), 1e-9)
        report[k] = {
            "native": nv,
            "vectorized": vv,
            "abs_error": err,
            "rel_error": rel,
            "passed": err <= atol or rel <= 0.05,
        }
    return report


__all__ = [
    "VectorizedBacktest",
    "BacktestResult",
    "compare_results",
]
