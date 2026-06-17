"""
向量化回测引擎（验证版）
============================

借鉴来源：
- VectorBT（开源 MIT 协议）：通过 NumPy/Pandas 矢量化运算大幅提升回测速度
- 微软 Qlib 的 TopkDropoutStrategy：选股+调仓的向量化建模思路
- 现有 jingni-trader 的 native_adapter.py（事件循环实现）

设计目标：
1. 验证在 A 股多标的（500+ 只）日频回测场景下，纯 NumPy 向量化方案
   相比 `iterrows()` 事件循环的性能提升幅度。
2. 保持与现有 `BaseBacktestEngine` 接口兼容，可作为 `BACKTEST_BACKEND` 的
   新选项 "vectorized" 接入。
3. 简化实盘：只支持等权目标权重（top-k 选股 + 月度/周度调仓），对个人
   投研者最常见的场景即可。

注意：本文件仅作验证/性能对比，不会直接替换 main 分支代码。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# 兼容 hyphen 命名的子包
import importlib
import sys as _sys

for _pkg in ("skills.backtest-engine.scripts.base", "skills.backtest-engine.scripts.adapters"):
    _mod = importlib.import_module(_pkg)
    _sys.modules[_pkg.replace("-", "_")] = _mod

from skills.backtest_engine.scripts.base.base_backtest import BaseBacktestMetrics  # noqa: E402
from skills.backtest_engine.scripts.base.base_backtest_engine import BaseBacktestEngine  # noqa: E402


@dataclass
class VectorizedBacktestConfig:
    """向量化回测配置"""

    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025
    stamp_tax_rate: float = 0.001
    slippage: float = 0.001
    rebalance_freq: str = "W-FRI"  # 调仓频率：W-FRI 月末/周五
    top_k: int = 20  # 持仓数量
    weight_scheme: str = "equal"  # equal / score_prop
    min_lot: int = 100  # A 股最小交易单位
    cash_buffer: float = 0.02  # 现金缓冲（防止资金不足）


class VectorizedBacktestEngine(BaseBacktestEngine):
    """向量化回测引擎（验证版）"""

    name = "vectorized"

    def __init__(self, config: Optional[VectorizedBacktestConfig] = None):
        self.config = config or VectorizedBacktestConfig()

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: Optional[float] = None,
        benchmark: str = "000300.SH",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """执行向量化回测

        参数:
            data: 包含 date/code/open/close 等字段的日频行情
            signals: 包含 date/code/signal（越大越优先买入）的目标信号
        """
        if data.empty or signals.empty:
            return self._empty_result()

        cfg = self.config
        if init_capital is not None:
            cfg = VectorizedBacktestConfig(**{**self.config.__dict__, "init_capital": init_capital})

        # === 1. 数据清洗 & 透视 ===
        # 统一日期为 datetime
        work = data.copy()
        work["date"] = pd.to_datetime(work["date"])
        sig = signals.copy()
        sig["date"] = pd.to_datetime(sig["date"])

        prices = (
            work.sort_values(["date", "code"])
            .pivot(index="date", columns="code", values="close")
            .astype(float)
            .ffill()
        )
        signal_pivot = (
            sig.sort_values(["date", "code"])
            .pivot(index="date", columns="code", values="signal")
            .reindex(prices.index)
        )

        # 调仓日：使用 signal_pivot 中非空日期作为调仓日
        sig_present = signal_pivot.notna().any(axis=1)
        rebalance_dates = signal_pivot.index[sig_present].tolist()
        if not rebalance_dates:
            return self._empty_result()
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for d in rebalance_dates:
            if d not in signal_pivot.index:
                continue
            row = signal_pivot.loc[d].dropna()
            if row.empty:
                continue
            top = row.nlargest(cfg.top_k)
            if top.empty:
                continue
            if cfg.weight_scheme == "equal":
                w = pd.Series(1.0 / len(top), index=top.index)
            else:  # score_prop: 分数占比
                pos = top.clip(lower=0)
                if pos.sum() <= 0:
                    w = pd.Series(1.0 / len(top), index=top.index)
                else:
                    w = pos / pos.sum()
            weights.loc[d, w.index] = w.values

        weights = weights.ffill().fillna(0.0)

        # === 2. 日度收益与组合净值 ===
        daily_ret = prices.pct_change().fillna(0.0)
        # 调仓日的组合收益应当使用「开盘→次日开盘」或「当日开盘→收盘」
        # 简化处理：使用当日 close→次日 close，与现有 native_adapter 一致
        port_ret = (weights.shift(1).fillna(0.0) * daily_ret).sum(axis=1)
        equity = cfg.init_capital * (1.0 + port_ret).cumprod()

        # === 3. 调仓日换手与交易成本 ===
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        # 交易成本：佣金双向 + 印花税仅卖出 + 滑点
        # 只在 turnover > 0 的调仓日计入成本（避免逐日重复扣费）
        sell_turnover = weights.diff().clip(upper=0).abs().sum(axis=1).fillna(0.0)
        cost = (
            turnover * (cfg.commission_rate + cfg.slippage)
            + sell_turnover * cfg.stamp_tax_rate
        )
        # 成本从当日收益中扣除
        port_ret_net = port_ret - cost
        equity_net = cfg.init_capital * (1.0 + port_ret_net).cumprod()

        # === 4. 构造 trades / positions 模拟（用于指标计算接口兼容） ===
        trades = self._approximate_trades(weights, prices, cfg)
        positions = weights.iloc[-1][weights.iloc[-1] > 0].reset_index()
        positions.columns = ["code", "weight"]

        # === 5. 绩效指标 ===
        equity_curve = pd.DataFrame(
            {
                "date": equity_net.index,
                "equity": equity_net.values,
                "cash": 0.0,  # 简化：仅记录净值
                "market_value": equity_net.values,
                "position_count": (weights > 0).sum(axis=1).values,
            }
        )
        metrics = BaseBacktestMetrics.calc_all_metrics(equity_net, trades)

        return {
            "trades": trades,
            "positions": positions,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
            "extra": {
                "engine": self.name,
                "rebalance_freq": cfg.rebalance_freq,
                "top_k": cfg.top_k,
                "avg_turnover": float(turnover.mean()),
                "max_turnover": float(turnover.max()),
            },
        }

    @staticmethod
    def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> List[pd.Timestamp]:
        """生成调仓日序列（每个 freq 周期最后一行）"""
        df = pd.DataFrame(index=index)
        grouped = df.groupby(pd.Grouper(freq=freq))
        last = grouped.size().cumsum()
        out = []
        cum = 0
        for _period, n in grouped.size().items():
            cum += n
            if n == 0:
                continue
            out.append(index[cum - 1])
        return out

    @staticmethod
    def _approximate_trades(
        weights: pd.DataFrame, prices: pd.DataFrame, cfg: VectorizedBacktestConfig
    ) -> pd.DataFrame:
        """根据权重变化生成近似 trades 记录（用于胜率等指标）"""
        diff = weights.diff().fillna(weights)
        records: List[Dict[str, Any]] = []
        for d, row in diff.iterrows():
            for code, dw in row.items():
                if abs(dw) < 1e-9:
                    continue
                price = prices.at[d, code] if (d in prices.index and code in prices.columns) else np.nan
                if pd.isna(price):
                    continue
                shares = int(dw * cfg.init_capital / max(price, 1e-6) / cfg.min_lot) * cfg.min_lot
                amount = shares * price
                pnl = 0.0  # 单笔不计算盈亏，胜率仅按调仓成功与否统计
                records.append(
                    {
                        "date": d,
                        "code": code,
                        "action": "buy" if dw > 0 else "sell",
                        "price": price,
                        "shares": shares,
                        "amount": amount,
                        "commission": 0.0,
                        "tax": 0.0,
                        "pnl": pnl,
                    }
                )
        return pd.DataFrame(records)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
            "extra": {"engine": self.name},
        }
