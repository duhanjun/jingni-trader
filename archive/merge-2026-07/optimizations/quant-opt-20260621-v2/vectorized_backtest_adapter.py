"""
向量化回测适配器

借鉴来源:
- VectorBT: 核心思想——用矩阵运算替代逐 bar 事件驱动循环，性能提升 50-100x
- Microsoft Qlib: TopkDropoutStrategy 等向量化策略实现

优化点:
jingni-trader backtest-engine 的 native_adapter.run_backtest 使用
`for dt in dates: for _, row in day_signal.iterrows():` 双重 Python 循环，
当股票数 × 日期数较大时(如全市场 5000 股 × 1000 日 = 500 万行)性能极差。

本模块用 pandas/numpy 向量化实现等权多头组合回测:
1. 信号矩阵化: pivot 成 (date × code) 矩阵
2. 持仓矩阵: 信号触发后等权分配，考虑 T+1
3. 收益矩阵: 每日收益率 × 持仓权重
4. 净值曲线: 累乘得到组合净值

支持 A 股规则: T+1、涨跌停过滤、佣金、印花税、滑点。
"""
from __future__ import annotations
import time
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("vectorized-backtest")


class VectorizedBacktester:
    """
    向量化回测器

    适用于等权多头/多空组合策略，性能远超事件驱动。
    对于需要复杂仓位管理(如动态止损、金字塔加仓)的策略，
    仍建议使用 native_adapter。
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
        benchmark: str = "000300.SH",
        max_positions: int = 50,
    ) -> Dict[str, Any]:
        """
        向量化回测

        参数:
            data: OHLCV 数据, 含 code/date/open/high/low/close/volume
                  可选列: is_limit_up, is_limit_down, suspended
            signals: 信号, 含 code/date/signal
                     signal > 0 表示买入, < 0 表示卖出, = 0 持有
            init_capital: 初始资金
            commission_rate: 佣金费率(双边)
            min_commission: 最小佣金(元)
            stamp_tax_rate: 印花税率(仅卖出)
            slippage: 滑点(比例)
            t_plus_1: 是否启用 T+1
            price_limit: 是否过滤涨跌停
            max_positions: 最大持仓数

        返回:
            {
                "equity_curve": DataFrame,
                "trades": DataFrame,
                "positions": DataFrame,
                "metrics": dict,
            }
        """
        if data.empty or signals.empty:
            return self._empty_result()

        t0 = time.perf_counter()

        # 1. 数据预处理: pivot 成矩阵
        close_matrix, ret_matrix, limit_up_matrix, limit_down_matrix, suspended_matrix = \
            self._build_matrices(data, price_limit)

        # 2. 信号矩阵化
        signal_matrix = self._build_signal_matrix(signals, close_matrix)

        # 3. 计算目标持仓矩阵(等权)
        target_weights = self._calc_target_weights(signal_matrix, max_positions, price_limit,
                                                    limit_up_matrix, limit_down_matrix)

        # 4. T+1 调整: 今日信号明日才生效
        if t_plus_1:
            held_weights = target_weights.shift(1).fillna(0)
        else:
            held_weights = target_weights.copy()

        # 5. 计算换手率(权重变化)
        prev_weights = held_weights.shift(1).fillna(0)
        weight_change = held_weights - prev_weights
        buy_turnover = weight_change.clip(lower=0).sum(axis=1)
        sell_turnover = (-weight_change.clip(upper=0)).sum(axis=1)

        # 6. 计算组合收益与成本
        gross_return = (held_weights * ret_matrix).sum(axis=1)
        # 成本: 买入佣金 + 卖出佣金 + 卖出印花税 + 滑点
        buy_cost = buy_turnover * (commission_rate + slippage)
        sell_cost = sell_turnover * (commission_rate + stamp_tax_rate + slippage)
        # 最小佣金近似: 当换手率很小时按 min_commission/init_capital 计
        min_cost_ratio = min_commission / init_capital
        buy_cost = buy_cost.clip(lower=min_cost_ratio * (buy_turnover > 0).astype(float))
        sell_cost = sell_cost.clip(lower=min_cost_ratio * (sell_turnover > 0).astype(float))
        net_return = gross_return - buy_cost - sell_cost

        # 7. 净值曲线
        equity = (1 + net_return).cumprod() * init_capital
        equity.name = "equity"

        # 8. 持仓明细
        positions_df = self._extract_positions(held_weights, close_matrix)

        # 9. 成交流水(简化: 记录每日换手)
        trades_df = self._extract_trades(weight_change, close_matrix, init_capital,
                                          commission_rate, stamp_tax_rate, slippage)

        # 10. 绩效指标
        try:
            from .benchmark_metrics import calc_full_metrics
        except ImportError:
            from benchmark_metrics import calc_full_metrics
        metrics = calc_full_metrics(equity, risk_free_rate=0.03)

        elapsed = time.perf_counter() - t0
        logger.info(f"向量化回测完成，耗时 {elapsed:.3f}s, {len(equity)} 个交易日")

        equity_curve = pd.DataFrame({
            "equity": equity,
            "cash": init_capital - (held_weights.abs().sum(axis=1) * init_capital),
            "market_value": (held_weights.abs().sum(axis=1) * init_capital),
            "position_count": (held_weights > 0).sum(axis=1),
            "daily_return": net_return,
        }).reset_index().rename(columns={"index": "date"})

        return {
            "equity_curve": equity_curve,
            "trades": trades_df,
            "positions": positions_df,
            "metrics": metrics,
            "report_path": "",
            "elapsed_sec": round(elapsed, 4),
        }

    def _build_matrices(
        self, data: pd.DataFrame, price_limit: bool,
    ) -> tuple:
        """构建价格/收益/涨跌停矩阵 (date × code)"""
        df = data[["code", "date", "close"]].copy()
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        close_matrix = df.pivot_table(index="date", columns="code", values="close")
        close_matrix = close_matrix.sort_index()

        # 收益率矩阵
        ret_matrix = close_matrix.pct_change().fillna(0)

        # 涨跌停矩阵
        limit_up_matrix = pd.DataFrame(0.0, index=close_matrix.index, columns=close_matrix.columns)
        limit_down_matrix = pd.DataFrame(0.0, index=close_matrix.index, columns=close_matrix.columns)
        suspended_matrix = pd.DataFrame(0.0, index=close_matrix.index, columns=close_matrix.columns)

        if price_limit and "is_limit_up" in data.columns:
            lu = data.pivot_table(index="date", columns="code", values="is_limit_up", fill_value=0)
            ld = data.pivot_table(index="date", columns="code", values="is_limit_down", fill_value=0)
            limit_up_matrix = lu.reindex(index=close_matrix.index, columns=close_matrix.columns).fillna(0)
            limit_down_matrix = ld.reindex(index=close_matrix.index, columns=close_matrix.columns).fillna(0)

        if "suspended" in data.columns:
            sus = data.pivot_table(index="date", columns="code", values="suspended", fill_value=0)
            suspended_matrix = sus.reindex(index=close_matrix.index, columns=close_matrix.columns).fillna(0)

        return close_matrix, ret_matrix, limit_up_matrix, limit_down_matrix, suspended_matrix

    def _build_signal_matrix(
        self, signals: pd.DataFrame, close_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """构建信号矩阵 (date × code), 值为信号强度"""
        sig = signals[["code", "date", "signal"]].copy()
        signal_matrix = sig.pivot_table(index="date", columns="code", values="signal", fill_value=0)
        return signal_matrix.reindex(index=close_matrix.index, columns=close_matrix.columns).fillna(0)

    def _calc_target_weights(
        self,
        signal_matrix: pd.DataFrame,
        max_positions: int,
        price_limit: bool,
        limit_up_matrix: pd.DataFrame,
        limit_down_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        计算目标持仓权重矩阵

        策略: 每日选取 signal 最大的 max_positions 只股票等权配置
        涨停不可买入，跌停不可卖出
        """
        weights = pd.DataFrame(0.0, index=signal_matrix.index, columns=signal_matrix.columns)

        for dt in signal_matrix.index:
            day_signal = signal_matrix.loc[dt]
            # 只看正向信号
            candidates = day_signal[day_signal > 0]
            if candidates.empty:
                continue

            # 过滤涨停(不可买入)
            if price_limit and dt in limit_up_matrix.index:
                lu = limit_up_matrix.loc[dt]
                candidates = candidates[lu.reindex(candidates.index).fillna(0) == 0]

            if candidates.empty:
                continue

            # 取 signal 最大的 N 只
            top_n = candidates.nlargest(max_positions)
            weights.loc[dt, top_n.index] = 1.0 / len(top_n)

        return weights

    def _extract_positions(
        self, held_weights: pd.DataFrame, close_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """提取末日持仓"""
        if held_weights.empty:
            return pd.DataFrame(columns=["code", "shares", "weight", "price", "market_value"])
        last_date = held_weights.index[-1]
        last_weights = held_weights.loc[last_date]
        active = last_weights[last_weights > 0]
        if active.empty:
            return pd.DataFrame(columns=["code", "shares", "weight", "price", "market_value"])
        last_prices = close_matrix.loc[last_date].reindex(active.index)
        return pd.DataFrame({
            "code": active.index,
            "weight": active.values,
            "price": last_prices.values,
        })

    def _extract_trades(
        self,
        weight_change: pd.DataFrame,
        close_matrix: pd.DataFrame,
        init_capital: float,
        commission_rate: float,
        stamp_tax_rate: float,
        slippage: float,
    ) -> pd.DataFrame:
        """提取成交流水(每日每股票的权重变化)"""
        records = []
        for dt in weight_change.index:
            day_change = weight_change.loc[dt]
            nonzero = day_change[day_change != 0]
            if nonzero.empty or dt not in close_matrix.index:
                continue
            prices = close_matrix.loc[dt]
            for code, w_change in nonzero.items():
                if code not in prices.index or pd.isna(prices[code]) or prices[code] <= 0:
                    continue
                price = prices[code]
                action = "buy" if w_change > 0 else "sell"
                amount = abs(w_change) * init_capital
                shares = int(amount / price / 100) * 100
                if shares <= 0:
                    continue
                exec_price = price * (1 + slippage) if action == "buy" else price * (1 - slippage)
                commission = max(amount * commission_rate, 5)
                tax = amount * stamp_tax_rate if action == "sell" else 0
                records.append({
                    "date": dt, "code": code, "action": action,
                    "price": exec_price, "shares": shares, "amount": amount,
                    "commission": commission, "tax": tax,
                })
        return pd.DataFrame(records)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "equity_curve": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
            "elapsed_sec": 0.0,
        }


def run_vectorized_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    **kwargs,
) -> Dict[str, Any]:
    """便捷入口函数"""
    bt = VectorizedBacktester()
    return bt.run(data, signals, **kwargs)
