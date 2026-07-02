"""
向量化回测引擎（验证模块 2）

借鉴来源：
- VectorBT / VectorBT PRO：
    https://github.com/polakowo/vectorbt
    核心思想：把策略表达为 boolean 矩阵 → 用 NumPy / Numba 向量化模拟
- VectorBT fundamentals：
    "treat matrices as first-class citizens"
- AKQuant 内部文档：
    "Core Calculation Sinking (Rust Core): 事件循环、订单撮合、风险检查都在 Rust/Numba 层"
- 本实现：用 Numba @njit 编译回测主循环，绕开 Python GIL 与逐行解释开销

相对 jingni-trader 现有实现 (skills/backtest-engine/scripts/adapters/native_adapter.py)
的改进：
- 原实现：纯 Python for dt in dates: 逐日循环
- 本实现：Numba JIT 编译的主循环 + 向量化持仓/资金管理
- 新增：参数 sweep 一次性跑多组策略（VectorBT 标志性能力）
- 保留：A 股 T+1、涨跌停、印花税
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False
    njit = lambda *a, **k: (lambda f: f)  # no-op decorator fallback
    prange = range


# ---------------------------------------------------------------------------
# Numba 加速的回测主循环
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _simulate_loop_nb(
    close: np.ndarray,         # (T, N) 收盘价
    open_: np.ndarray,         # (T, N) 开盘价
    is_limit_up: np.ndarray,   # (T, N) bool
    is_limit_down: np.ndarray, # (T, N) bool
    signal: np.ndarray,        # (T, N) 0/1/2  (0=空仓 1=持有 2=换仓)
    n_stocks_per_buy: int,
    init_capital: float,
    commission_rate: float,
    stamp_tax_rate: float,
    min_commission: float,
    slippage: float,
    cash_buffer: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A 股规则下的向量化回测主循环（Numba 加速）

    关键规则：
    - T+1: 当日买入次日才能卖（实现在循环中：卖只在 next-bar 撮合）
    - 涨停不买入、跌停不卖出
    - 现金 95% 分配、保留 5% buffer
    - 100 股一手、向下取整

    返回:
        equity (T,)         : 每日总资产
        cash  (T,)         : 每日现金
        market_value (T,)  : 每日持仓市值
        position_count (T,): 每日持仓只数
    """
    T, N = close.shape
    cash = init_capital
    positions = np.zeros(N, dtype=np.float64)
    cash_arr = np.zeros(T, dtype=np.float64)
    mv_arr = np.zeros(T, dtype=np.float64)
    eq_arr = np.zeros(T, dtype=np.float64)
    pc_arr = np.zeros(T, dtype=np.float64)

    for t in range(T):
        # 1) 当日撮合：用前一日信号 → 当日开盘价成交（更接近实盘）
        if t > 0:
            prev_sig = signal[t - 1]
            # 先卖：清空所有前日持有但今日信号为 0 的
            for i in range(N):
                if positions[i] > 0 and prev_sig[i] == 0:
                    if not is_limit_down[t, i]:
                        px = open_[t, i] * (1 - slippage)
                        proceeds = positions[i] * px
                        fee = max(proceeds * commission_rate, min_commission) + proceeds * stamp_tax_rate
                        cash += proceeds - fee
                        positions[i] = 0.0
            # 再买：信号为 1 但当前未持有
            # 计算今日要买的数量
            buy_codes = []
            for i in range(N):
                if prev_sig[i] == 1 and positions[i] == 0 and not is_limit_up[t, i]:
                    buy_codes.append(i)
            n_buy = len(buy_codes)
            if n_buy > 0:
                per_budget = cash * (1 - cash_buffer) / n_buy
                for k in range(n_buy):
                    i = buy_codes[k]
                    px = open_[t, i] * (1 + slippage)
                    # 100 股一手
                    shares = (np.floor(per_budget / px / 100.0) * 100.0)
                    if shares <= 0:
                        continue
                    cost = shares * px
                    fee = max(cost * commission_rate, min_commission)
                    total = cost + fee
                    if total > cash:
                        shares = (np.floor(cash * (1 - cash_buffer) / px / 100.0) * 100.0)
                        if shares <= 0:
                            continue
                        cost = shares * px
                        fee = max(cost * commission_rate, min_commission)
                        total = cost + fee
                    cash -= total
                    positions[i] += shares

        # 2) 按当日收盘价计算市值
        mv = 0.0
        pc = 0
        for i in range(N):
            if positions[i] > 0:
                mv += positions[i] * close[t, i]
                pc += 1

        cash_arr[t] = cash
        mv_arr[t] = mv
        eq_arr[t] = cash + mv
        pc_arr[t] = pc

    return eq_arr, cash_arr, mv_arr, pc_arr


# ---------------------------------------------------------------------------
# Python 包装层
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """回测结果"""
    equity: np.ndarray             # (T,)
    cash: np.ndarray               # (T,)
    market_value: np.ndarray       # (T,)
    position_count: np.ndarray     # (T,)
    dates: np.ndarray              # (T,)
    metrics: Dict[str, float] = field(default_factory=dict)


class VectorizedBacktester:
    """
    向量化回测器

    用法:
        bt = VectorizedBacktester(init_capital=1_000_000)
        result = bt.run(data_df, signals_df)
        print(result.metrics)
    """

    def __init__(
        self,
        init_capital: float = 1_000_000.0,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.001,
        cash_buffer: float = 0.05,
        n_stocks_per_buy: int = 0,  # 0 表示按信号平均分配
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.cash_buffer = cash_buffer
        self.n_stocks_per_buy = n_stocks_per_buy

    def _prepare_arrays(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """
        把 (date, code) 面板数据 → (T, N) 矩阵
        """
        df = data.merge(signals, on=["date", "code"], how="outer")
        df = df.sort_values(["date", "code"]).reset_index(drop=True)
        df["signal"] = df["signal"].fillna(0).astype(np.int8)
        # 重塑为 2D 矩阵
        pivot_close = df.pivot(index="date", columns="code", values="close").sort_index()
        pivot_open = df.pivot(index="date", columns="code", values="open").sort_index()
        pivot_lim_up = df.pivot(index="date", columns="code", values="is_limit_up").sort_index()
        pivot_lim_dn = df.pivot(index="date", columns="code", values="is_limit_down").sort_index()
        pivot_sig = df.pivot(index="date", columns="code", values="signal").sort_index()

        # 对齐 index & columns
        idx = pivot_close.index
        cols = pivot_close.columns
        pivot_open = pivot_open.reindex(index=idx, columns=cols)
        pivot_lim_up = pivot_lim_up.reindex(index=idx, columns=cols).fillna(False)
        pivot_lim_dn = pivot_lim_dn.reindex(index=idx, columns=cols).fillna(False)
        pivot_sig = pivot_sig.reindex(index=idx, columns=cols).fillna(0)

        close = pivot_close.values.astype(np.float64)
        open_ = pivot_open.values.astype(np.float64)
        is_up = pivot_lim_up.values.astype(np.bool_)
        is_dn = pivot_lim_dn.values.astype(np.bool_)
        sig = pivot_sig.values.astype(np.int8)
        return close, open_, is_up, is_dn, sig, idx.values, cols.values

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        """
        执行单次回测

        参数:
            data:    ['date', 'code', 'open', 'close', 'is_limit_up', 'is_limit_down']
            signals: ['date', 'code', 'signal']  signal ∈ {0, 1, 2}
                     0=空仓 1=目标买入 2=换仓（卖出旧仓再买新仓）
        """
        if data.empty or signals.empty:
            return self._empty_result()

        close, open_, is_up, is_dn, sig, dates, codes = self._prepare_arrays(data, signals)

        # 触发 Numba JIT 编译（首次会稍慢）
        if HAS_NUMBA and not _simulate_loop_nb.py_func.__name__.startswith("_simulate_loop_nb"):
            pass  # numba 自动缓存

        t0 = time.perf_counter()
        eq, cash, mv, pc = _simulate_loop_nb(
            close, open_, is_up, is_dn, sig,
            self.n_stocks_per_buy, self.init_capital,
            self.commission_rate, self.stamp_tax_rate,
            self.min_commission, self.slippage, self.cash_buffer,
        )
        elapsed = time.perf_counter() - t0

        metrics = self._calc_metrics(eq, dates)
        metrics["backtest_seconds"] = round(elapsed, 4)
        metrics["T_days"] = int(len(eq))
        metrics["N_stocks"] = int(len(codes))

        return BacktestResult(
            equity=eq, cash=cash, market_value=mv,
            position_count=pc, dates=dates, metrics=metrics,
        )

    def parameter_sweep(
        self,
        data: pd.DataFrame,
        signal_func,        # callable(topk) -> signals_df
        param_grid: Dict[str, List],
    ) -> pd.DataFrame:
        """
        参数扫描（VectorBT 标志性能力）
        一次性跑多组超参，对比结果

        参数:
            data:         同 run
            signal_func:  接受 param 字典，返回 signals DataFrame
            param_grid:   {'topk': [10,20,30], 'hold_days': [5,10]}

        返回:
            DataFrame, index=参数组合, columns=metrics
        """
        from itertools import product

        keys = list(param_grid.keys())
        rows = []
        for combo in product(*[param_grid[k] for k in keys]):
            params = dict(zip(keys, combo))
            sigs = signal_func(**params)
            res = self.run(data, sigs)
            row = {**params, **res.metrics}
            rows.append(row)
        return pd.DataFrame(rows).set_index(keys)

    @staticmethod
    def _calc_metrics(equity: np.ndarray, dates: np.ndarray) -> Dict[str, float]:
        """全面绩效指标（与现有 BacktestEngine 对齐）"""
        if len(equity) < 2:
            return {}
        eq = pd.Series(equity, index=pd.to_datetime(dates))
        returns = eq.pct_change().dropna()
        if returns.empty:
            return {}
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        n_days = len(returns)
        annual_return = float((1 + total_return) ** (252 / n_days) - 1) if n_days > 0 else 0.0
        volatility = float(returns.std() * np.sqrt(252))
        sharpe = float(annual_return / volatility) if volatility > 0 else 0.0
        max_dd = float((eq / eq.cummax() - 1).min())
        win_rate = float((returns > 0).mean())
        calmar = float(annual_return / abs(max_dd)) if max_dd != 0 else 0.0
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "calmar_ratio": calmar,
        }

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            equity=np.array([]), cash=np.array([]),
            market_value=np.array([]), position_count=np.array([]),
            dates=np.array([]), metrics={},
        )


# ---------------------------------------------------------------------------
# 顶层便捷函数
# ---------------------------------------------------------------------------

def run_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    init_capital: float = 1_000_000.0,
    **kwargs,
) -> BacktestResult:
    """便捷入口"""
    bt = VectorizedBacktester(init_capital=init_capital, **kwargs)
    return bt.run(data, signals)
