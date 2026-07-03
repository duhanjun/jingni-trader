"""
向量化回测引擎

【借鉴来源】
- backtesting.py (kernc/backtesting.py): 利用 self.I() 预计算 + 向量化执行
- Qlib (microsoft/qlib): 表达式引擎 + Point-in-Time 数据
- QUANTAXIS: 适配器模式 + 桥接层抽象

【问题背景】
原 backtest-engine/scripts/adapters/native_adapter.py 存在严重的性能瓶颈：
  - 对每个日期 for 循环 (O(D))
  - 每个日期内对每条信号用 day_data_map.loc[code] 反复查表 (O(N))
  - 整个回测复杂度 O(D * N)，5000 股 × 1000 日 = 5e6 次查表

【设计目标】
1. 性能提升 10-100 倍（向量化后预期 100x）
2. 保持语义一致：T+1、涨跌停、最低佣金、印花税
3. 输出与原引擎对齐的 metrics，便于替换对比
4. 不引入额外重型依赖（仅 pandas + numpy）

【关键技巧】
- 用 pivot_table 将 OHLCV 转成 (date × code) 矩阵
- 用 shift 实现 T+1 信号延迟
- 用 groupby + agg 计算组合权益
- 向量化订单簿：按 (date, code) 一次性结算
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── A 股交易规则常量 ──────────────────────────────────────
DEFAULT_BENCHMARK = "000300.SH"
DEFAULT_TRADING_DAYS = 252
DEFAULT_RISK_FREE = 0.03


@dataclass
class BacktestConfig:
    """回测配置（与 jingni-trader config.py 对齐）"""
    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.001          # 仅卖出收取
    transfer_fee_rate: float = 0.00002
    slippage: float = 0.0001
    t_plus_1: bool = True
    price_limit: bool = True
    benchmark: str = DEFAULT_BENCHMARK
    risk_free_rate: float = DEFAULT_RISK_FREE
    trading_days: int = DEFAULT_TRADING_DAYS
    lot_size: int = 100


class VectorizedBacktestEngine:
    """
    向量化回测引擎

    输入:
        data    : 标准化行情 DataFrame (code, date, open, high, low, close, volume, ...)
        signals : 信号 DataFrame (code, date, signal)，signal ∈ {-1, 0, +1}
                  或连续权重 (0..1 表示目标仓位)
    输出:
        包含 equity_curve / trades / metrics 的字典
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    # ── 公开 API ─────────────────────────────────────────
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Dict[str, Any]:
        """执行回测"""
        if data.empty or signals.empty:
            return self._empty_result()

        # 1. 准备面板
        prices = self._build_price_panel(data)
        sig = self._align_signals(signals, prices)

        # 2. 涨跌停标记
        if self.config.price_limit and "is_limit_up" not in prices.columns:
            prices = self._mark_price_limits(prices)

        # 3. T+1 处理：信号在 t 日发出，t+1 日开盘成交
        #    为简化，假设按下一交易日 close 结算（与原 native_adapter 行为一致）
        if self.config.t_plus_1:
            target_weights = sig.shift(1).fillna(0)
        else:
            target_weights = sig

        # 4. 限价过滤（不能买入涨停、不能卖出跌停）
        target_weights = self._apply_price_limit_filter(target_weights, prices)

        # 5. 向量化计算组合权益
        equity_curve, trades, daily_stats = self._simulate(target_weights, prices)

        # 6. 绩效指标
        metrics = self._calc_metrics(equity_curve)

        return {
            "trades": trades,
            "equity_curve": equity_curve.reset_index(),
            "metrics": metrics,
            "daily_stats": daily_stats,
            "config": self.config.__dict__,
        }

    # ── 数据准备 ─────────────────────────────────────────
    def _build_price_panel(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        将长表 (code, date, ...) 转换为 (date, code) 面板
        索引为日期，列为股票代码
        """
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "code"]).drop_duplicates(["date", "code"])
        panel = df.pivot(index="date", columns="code", values="close").sort_index()
        # 处理可能的全空列
        panel = panel.dropna(axis=1, how="all")
        return panel

    def _align_signals(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        将信号表对齐到价格面板。
        若 signal 是离散 {-1,0,1}，转为等权（0.1/0/0.1 等权分配）。
        若 signal 是连续权重 (0..1)，直接使用。
        """
        sig = signals.copy()
        sig["date"] = pd.to_datetime(sig["date"])
        sig = sig.sort_values(["date", "code"])

        # 判断信号是离散还是连续
        is_continuous = False
        if "weight" in sig.columns:
            is_continuous = True
            value_col = "weight"
        elif "signal" in sig.columns:
            unique_vals = sig["signal"].dropna().unique()
            if len(unique_vals) > 4 or not set(unique_vals).issubset({-1, 0, 1, 0.0, 1.0, -1.0}):
                is_continuous = True
            value_col = "signal"
        else:
            raise ValueError("signals 必须包含 'signal' 或 'weight' 列")

        sig_pivot = sig.pivot(index="date", columns="code", values=value_col).sort_index()

        # 对齐到价格面板（缺失填 0）
        sig_pivot = sig_pivot.reindex(index=prices.index, columns=prices.columns).fillna(0)

        if not is_continuous:
            # 离散信号：每天把仓位等分给信号为 1 的股票，-1 的反向
            # 为简化，仅做多头（信号为 1 时等权买入）
            row_sums = sig_pivot.clip(lower=0).sum(axis=1).replace(0, np.nan)
            sig_pivot = sig_pivot.div(row_sums, axis=0).fillna(0)

        # clip 权重到合理范围
        sig_pivot = sig_pivot.clip(-1, 1).fillna(0)
        return sig_pivot

    def _mark_price_limits(self, prices: pd.DataFrame) -> pd.DataFrame:
        """标记涨跌停（基于 pre_close）

        关键：涨跌停标记存为 prices._limit_up / _limit_down（带下划线的内部属性），
        不混入 prices 列（避免影响 pct_change 等数值运算）。
        """
        pre_close = prices.shift(1)
        pct_change_series = (prices - pre_close) / pre_close.replace(0, np.nan)
        any_limit_up = (pct_change_series >= 0.099).any(axis=1)
        any_limit_down = (pct_change_series <= -0.099).any(axis=1)
        prices = prices.copy()
        # 存为 Series，索引与 prices 对齐
        prices["_limit_up"] = any_limit_up
        prices["_limit_down"] = any_limit_down
        return prices

    def _apply_price_limit_filter(
        self,
        target_weights: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        价格限制过滤：股票当天涨停不能买入，跌停不能卖出。
        简化处理：若当日全市场有涨停/跌停，则禁止正向/负向调仓。
        """
        if not self.config.price_limit:
            return target_weights
        if "_limit_up" not in prices.columns:
            return target_weights
        # 用 reindex 让它与 target_weights 索引对齐（行级广播）
        limit_up_mask = prices["_limit_up"].reindex(
            index=target_weights.index, fill_value=False
        )
        limit_down_mask = prices["_limit_down"].reindex(
            index=target_weights.index, fill_value=False
        )
        filtered = target_weights.copy()
        filtered.loc[limit_up_mask, :] = filtered.loc[limit_up_mask, :].clip(upper=0)
        filtered.loc[limit_down_mask, :] = filtered.loc[limit_down_mask, :].clip(lower=0)
        return filtered

    # ── 核心：向量化模拟 ───────────────────────────────
    def _simulate(
        self,
        target_weights: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        核心向量化回测模拟器。

        关键步骤：
        1. 实际仓位按"目标权重 * 当前权益"计算
        2. 调仓成本按换手率 * 交易金额 * 费率计算
        3. 收益 = (next_close - close) / close * 持仓权重

        简化：
        - 不模拟整数手（按允许碎股交易，借鉴 backtesting.py 的 FractionalBacktest）
        - 不模拟逐笔交易，仅按净值曲线和换手率推算成本
        """
        cfg = self.config
        # 1. 计算每日收益（仅用数值列，过滤 _limit_up 等内部标记列）
        numeric_prices = prices[[c for c in prices.columns if not c.startswith("_")]]
        numeric_prices = numeric_prices.replace(0, np.nan).ffill()
        rets = numeric_prices.pct_change().shift(-1)  # t → t+1
        rets = rets.fillna(0)  # 最后一行无未来收益

        # 2. 实际持仓 = 上一日目标权重（已 shift(1)）
        actual_weights = target_weights.shift(1).fillna(0)
        if len(actual_weights) > 0:
            actual_weights.iloc[0] = 0  # 第一日无持仓

        # 3. 组合收益
        port_ret = (actual_weights * rets).sum(axis=1).fillna(0)

        # 4. 交易成本
        turnover = (target_weights - actual_weights).abs().sum(axis=1)
        blended_cost_rate = cfg.commission_rate + cfg.stamp_tax_rate * 0.5 + cfg.slippage
        cost = turnover * blended_cost_rate

        net_ret = port_ret - cost

        # 5. 累计权益
        equity = (1 + net_ret).cumprod() * cfg.init_capital
        equity = pd.concat([pd.Series([cfg.init_capital], index=[prices.index[0]]), equity])
        equity = equity[~equity.index.duplicated(keep="last")].sort_index()

        # 6. 模拟交易记录
        trade_records = self._extract_trades(target_weights, equity, numeric_prices)

        # 7. 每日统计
        daily_stats = pd.DataFrame({
            "port_ret": port_ret,
            "turnover": turnover,
            "cost": cost,
            "net_ret": net_ret,
            "n_long": (actual_weights > 0).sum(axis=1),
            "n_short": (actual_weights < 0).sum(axis=1),
        })
        daily_stats.index = target_weights.index

        return equity, trade_records, daily_stats

    def _extract_trades(
        self,
        target_weights: pd.DataFrame,
        equity: pd.Series,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """从权重变化中提取交易记录"""
        delta = target_weights.diff().fillna(target_weights)
        records = []
        for dt in target_weights.index:
            row = delta.loc[dt]
            equity_t = equity.get(dt, equity.iloc[0] if len(equity) else 0)
            for code, w in row.items():
                if abs(w) < 1e-6:
                    continue
                price = prices.at[dt, code] if dt in prices.index and code in prices.columns else np.nan
                if pd.isna(price):
                    continue
                amount = w * equity_t
                records.append({
                    "date": dt, "code": code, "weight": w,
                    "price": price, "amount": amount,
                    "action": "buy" if w > 0 else "sell",
                })
        return pd.DataFrame(records)

    # ── 绩效指标 ─────────────────────────────────────────
    def _calc_metrics(self, equity: pd.Series) -> Dict[str, float]:
        if len(equity) < 2:
            return {}

        rets = equity.pct_change().dropna()
        if rets.empty:
            return {}

        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        n_days = max(len(rets), 1)
        annual_return = float((1 + total_return) ** (self.config.trading_days / n_days) - 1)
        volatility = float(rets.std() * np.sqrt(self.config.trading_days))
        sharpe = float((annual_return - self.config.risk_free_rate) / volatility) if volatility > 0 else 0.0

        cumulative_max = equity.cummax()
        drawdown = equity / cumulative_max - 1
        max_drawdown = float(drawdown.min())

        win_rate = float((rets > 0).mean())
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        # Sortino
        neg_rets = rets[rets < 0]
        downside_std = float(neg_rets.std() * np.sqrt(self.config.trading_days)) if len(neg_rets) > 1 else 0.0
        sortino = float((annual_return - self.config.risk_free_rate) / downside_std) if downside_std > 0 else 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "n_trading_days": int(n_days),
            "engine": "vectorized",
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "trades": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "daily_stats": pd.DataFrame(),
            "config": self.config.__dict__,
        }


# ── 与 jingni-trader 原生引擎的等价实现（用于对比） ─────
class LoopBacktestEngine:
    """
    原 jingni-trader 的事件循环式回测逻辑（精简复刻版）
    保留 O(D*N) 循环结构，用于性能对比基准。
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        if data.empty or signals.empty:
            return self._empty_result()

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)
        dates = sorted(signals["date"].unique())

        cfg = self.config
        cash = cfg.init_capital
        positions: Dict[str, int] = {}
        equity_records = []
        trades = []

        for dt in dates:
            day_signal = signals[signals["date"] == dt]
            day_data = data[data["date"] == dt]
            if day_data.empty:
                continue

            day_data_map = day_data.set_index("code")
            buy_codes, sell_codes = [], []
            for _, row in day_signal.iterrows():
                code = row["code"]
                sig = float(row.get("signal", 0))
                if sig > 0:
                    buy_codes.append(code)
                elif sig < 0:
                    sell_codes.append(code)

            # 卖出
            for code in sell_codes:
                if code not in positions or positions[code] <= 0:
                    continue
                if code not in day_data_map.index:
                    continue
                price = day_data_map.loc[code, "close"]
                shares = positions[code]
                amount = price * shares
                commission = max(amount * cfg.commission_rate, cfg.min_commission)
                tax = amount * cfg.stamp_tax_rate
                cash += amount - commission - tax
                trades.append({"date": dt, "code": code, "action": "sell",
                               "price": price, "shares": shares, "amount": amount})
                positions[code] = 0

            # 买入
            if buy_codes:
                budget_per_stock = cash * 0.95 / len(buy_codes)
                for code in buy_codes:
                    if code not in day_data_map.index:
                        continue
                    price = day_data_map.loc[code, "close"] * (1 + cfg.slippage)
                    shares = int(budget_per_stock / price / cfg.lot_size) * cfg.lot_size
                    if shares <= 0:
                        continue
                    amount = price * shares
                    commission = max(amount * cfg.commission_rate, cfg.min_commission)
                    cost = amount + commission
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[code] = positions.get(code, 0) + shares
                    trades.append({"date": dt, "code": code, "action": "buy",
                                   "price": price, "shares": shares, "amount": amount})

            # 估值
            market_value = 0
            for code, shares in positions.items():
                if shares > 0 and code in day_data_map.index:
                    market_value += shares * day_data_map.loc[code, "close"]
            equity_records.append({"date": dt, "equity": cash + market_value, "cash": cash, "market_value": market_value})

        eq_df = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)
        if eq_df.empty:
            return self._empty_result()

        # 简化指标
        eq = eq_df.set_index("date")["equity"]
        rets = eq.pct_change().dropna()
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        n_days = max(len(rets), 1)
        annual_return = float((1 + total_return) ** (cfg.trading_days / n_days) - 1)
        vol = float(rets.std() * np.sqrt(cfg.trading_days)) if len(rets) > 1 else 0.0
        sharpe = float((annual_return - cfg.risk_free_rate) / vol) if vol > 0 else 0.0
        mdd = float((eq / eq.cummax() - 1).min())

        return {
            "trades": trades_df,
            "equity_curve": eq_df,
            "metrics": {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": mdd,
                "win_rate": float((rets > 0).mean()) if len(rets) > 0 else 0.0,
                "n_trading_days": int(n_days),
                "engine": "loop",
            },
        }

    def _empty_result(self):
        return {"trades": pd.DataFrame(), "equity_curve": pd.DataFrame(), "metrics": {}}
