"""
向量化回测引擎（验证版）

借鉴来源：VectorBT (https://vectorbt.dev)
核心思想：将逐 bar 的 Python 循环改造为 NumPy 矩阵运算，
         把"日期 × 股票"的数据表示为二维数组，用数组操作替代 iterrows/dict。

对比 jingni-trader 现有 native_adapter.py：
  - 现有实现：for dt in dates: for _, row in day_signal.iterrows(): ... （逐 bar Python 循环）
  - 本实现：  pivot 为宽表后，用矩阵运算一次性计算持仓收益、换手、成本、净值

支持的 A 股规则：T+1、涨跌停限制、佣金、印花税、滑点。
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


class VectorizedBacktester:
    """基于目标权重的向量化回测器。

    交易模型（与 Qlib / Zipline 的 Pipeline 回测一致）：
      - 每个交易日根据 ``target_weight`` 矩阵调仓到目标权重；
      - T+1：当日信号产生的目标权重在下一交易日才生效（shift 1）；
      - 涨跌停：被标记为不可交易的标的，其目标权重置 0；
      - 成本：换手率 × 综合费率（佣金 + 印花税的一半，近似买卖双边）；
      - 净值：init_capital × (1 + 日净收益).cumprod()。
    """

    def run_from_weights(
        self,
        close: pd.DataFrame,
        target_weight: pd.DataFrame,
        init_capital: float = 1e6,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.001,
        t_plus_1: bool = True,
        tradable: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """执行向量化回测。

        参数:
            close:          宽表收盘价 (index=date, columns=code)
            target_weight:  宽表目标权重 (index=date, columns=code)，每行和约为 1
            init_capital:   初始资金
            commission_rate: 佣金费率（双边）
            stamp_tax_rate:  印花税率（仅卖出）
            slippage:        滑点比例
            t_plus_1:        是否启用 T+1（信号次日生效）
            tradable:        可交易掩码宽表 (True=可交易)，None 表示全部可交易

        返回:
            dict: equity_curve, daily_returns, turnover, metrics, positions
        """
        # 对齐索引与列
        close = close.sort_index()
        target_weight = target_weight.reindex(index=close.index, columns=close.columns).fillna(0.0)

        if tradable is None:
            tradable = pd.DataFrame(True, index=close.index, columns=close.columns)
        else:
            tradable = tradable.reindex(index=close.index, columns=close.columns).fillna(True)

        # 日收益率矩阵
        returns = close.pct_change().fillna(0.0)

        # 目标权重：涨跌停不可交易 → 置 0
        eff_w = target_weight.where(tradable, 0.0)

        # 行归一化（保证权重和 <= 1，空仓日保持 0）
        row_sum = eff_w.sum(axis=1)
        safe_sum = row_sum.where(row_sum > 0, 1.0)
        eff_w = eff_w.div(safe_sum, axis=0)
        # 允许保留部分现金：若原始行和 < 1，归一化后仍 < 1（这里行和>0时归一到1）
        # 为保留"不满仓"语义，改为按原始行和缩放但不归一到1：
        eff_w = target_weight.where(tradable, 0.0)
        # 裁剪单标的最大权重，防止行和超过 1
        eff_w = eff_w.clip(upper=1.0)
        row_sum = eff_w.sum(axis=1)
        over = row_sum > 1.0
        if over.any():
            eff_w.loc[over] = eff_w.loc[over].div(row_sum.loc[over], axis=0)

        # T+1：今日信号 → 明日持仓
        if t_plus_1:
            held_w = eff_w.shift(1).fillna(0.0)
        else:
            held_w = eff_w.copy()

        # 换手率 = |w_t - w_{t-1}| 的行和
        turnover = held_w.diff().abs().sum(axis=1).fillna(0.0)

        # 滑点调整后的实际成交收益（近似：买入按更高价、卖出按更低价）
        # 这里用滑点等价于对换手部分扣除 slippage 成本
        # 综合成本率：佣金双边 + 印花税（仅卖出约占换手一半）+ 滑点
        cost_rate = commission_rate * 2.0 + stamp_tax_rate * 0.5 + slippage
        daily_cost = turnover * cost_rate

        # 组合日收益 = 昨日持仓权重 × 今日标的收益
        gross_ret = (held_w.shift(1) * returns).sum(axis=1).fillna(0.0)
        net_ret = gross_ret - daily_cost

        equity = init_capital * (1.0 + net_ret).cumprod()

        equity_curve = pd.DataFrame({
            "date": equity.index,
            "equity": equity.values,
            "daily_return": net_ret.values,
            "turnover": turnover.values,
            "daily_cost": daily_cost.values,
        }).reset_index(drop=True)

        positions = held_w.copy()

        return {
            "equity_curve": equity_curve,
            "positions": positions,
            "daily_returns": net_ret,
            "turnover": turnover,
            "metrics": self._calc_metrics(equity, net_ret, turnover),
        }

    @staticmethod
    def signals_to_weights(
        signals: pd.DataFrame,
        close: pd.DataFrame,
        max_weight: float = 0.1,
    ) -> pd.DataFrame:
        """把 (1/-1) 信号宽表转换为等权目标权重宽表。

        约定：signal==1 的标的等权持有，signal==-1 的标的权重置 0。
        用于与 native_adapter 的信号语义做近似对比。
        """
        sig_w = signals.reindex(index=close.index, columns=close.columns).fillna(0)
        buy_mask = (sig_w > 0).astype(float)
        n_buys = buy_mask.sum(axis=1)
        n_buys = n_buys.where(n_buys > 0, 1.0)
        w = buy_mask.div(n_buys, axis=0)
        w = w.clip(upper=max_weight)
        return w

    @staticmethod
    def _calc_metrics(equity: pd.Series, daily_ret: pd.Series, turnover: pd.Series) -> Dict[str, Any]:
        """计算核心绩效指标（与现有 BaseBacktestMetrics 口径一致以便对比）。"""
        trading_days = 252
        if len(equity) < 2:
            return {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0}
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        n_years = len(equity) / trading_days
        annual_return = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = float(drawdown.min())
        vol = float(daily_ret.std() * np.sqrt(trading_days)) if len(daily_ret) > 1 else 0.0
        sharpe = float((daily_ret.mean() * trading_days - 0.03) / vol) if vol > 0 else 0.0
        neg = daily_ret[daily_ret < 0]
        downside = float(neg.std() * np.sqrt(trading_days)) if len(neg) > 1 else 0.0
        sortino = float((daily_ret.mean() * trading_days - 0.03) / downside) if downside > 0 else 0.0
        avg_turnover = float(turnover.mean()) if len(turnover) > 0 else 0.0
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "avg_turnover": avg_turnover,
        }
