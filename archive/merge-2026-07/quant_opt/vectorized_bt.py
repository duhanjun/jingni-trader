"""
vectorized_bt - 向量化快速回测器

借鉴来源:
  - vectorbt (vectorbt.dev) - 从 signals/orders 一次性向量化计算, 10x+ 快于事件驱动
  - vectorbt PRO 的 from_signals / from_orders API
  - zipline-reloaded 的 Pipeline 思想

设计目标:
  1. 不依赖 vectorbt 库本身, 用 numpy/pandas 复刻核心逻辑 (可作为降级)
  2. 当 vectorbt 可用时, 优先使用 (speed boost)
  3. 与 jingni-trader 的 signals DataFrame [date, code, signal] 对齐
  4. 输出与 native_adapter 等价的 metrics dict, 可直接替换

性能:
  在 1k 票 x 5y 日频数据上, 应比 native_adapter 快 5-10x
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_opt.vectorized_bt")

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    logger.info("vectorbt 未安装, 将使用纯 pandas/numpy 实现")


# ============================================================================
# 1. 纯 numpy/pandas 实现 (降级方案)
# ============================================================================

def _vbt_run_pure(
    prices: pd.DataFrame,        # index=date, columns=code, value=close
    signals: pd.DataFrame,       # index=date, columns=code, value in {0, 1, -1}
    init_cash: float = 1e6,
    commission: float = 0.00025,
    stamp_tax: float = 0.001,
    slippage: float = 0.0001,
    top_k: Optional[int] = None,  # 每日持仓上限
    long_only: bool = True,
) -> Dict[str, Any]:
    """
    纯 pandas 向量化回测

    核心思想:
      - 每日信号 -> 次日开盘调仓 (避免未来数据)
      - 权重 = 等权 (top_k 信号) / NaN (无信号)
      - 收益 = sum(weight_t * return_{t+1})
      - 成本 = 调仓金额 * 费率
    """
    if prices.empty or signals.empty:
        return _empty_result()

    prices = prices.sort_index()
    signals = signals.sort_index().reindex(prices.index).fillna(0).astype(int)

    # 次日收益
    rets = prices.pct_change().shift(-1).fillna(0.0)  # t 日信号 -> t+1 日收益
    if long_only:
        signals = signals.clip(lower=0)

    # 仅保留正值信号 (买入)
    entries = signals > 0

    if top_k is not None and top_k > 0:
        # 每日取信号值为 1 的代码中随机选 top_k (简化: 取前 top_k 个)
        # 更精细: 按 signal 大小排序, 但本引擎假设信号是 0/1
        # 实现: 保留每日每行, 但加 mask
        mask = entries.apply(
            lambda row: pd.Series(
                np.where(row.cumsum() <= top_k, True, False),
                index=row.index
            ),
            axis=1,
        )
        # fallback: 用 numpy 加速
        entry_arr = entries.values.astype(bool)
        cum = np.cumsum(entry_arr, axis=1)
        mask_arr = (cum <= top_k) & entry_arr
        weights_arr = mask_arr.astype(np.float64) / max(1, top_k)
    else:
        # 等权: 每日有信号的代码平均分配 100% 仓位
        n_active = entries.sum(axis=1).replace(0, np.nan)
        weights_arr = (entries.div(n_active, axis=0)).fillna(0.0).values

    # 收益: 每日组合收益
    rets_arr = rets.fillna(0.0).values
    port_rets = (weights_arr * rets_arr).sum(axis=1)

    # 交易成本: 权重变化 * 费率 (简化: 用换手率近似)
    w_df = pd.DataFrame(weights_arr, index=prices.index, columns=prices.columns)
    turnover = w_df.diff().abs().sum(axis=1).fillna(w_df.iloc[0].abs().sum()) / 2
    # 买入收佣金 + 滑点, 卖出收佣金 + 滑点 + 印花税 (近似为综合费率)
    cost = turnover.values * (commission * 2 + slippage * 2 + stamp_tax * 0.5)

    net_rets = port_rets - cost
    equity = pd.Series((1 + net_rets).cumprod() * init_cash, index=prices.index)

    # 交易记录 (简化: 仅记录权重变化点)
    weight_change = w_df.diff().fillna(w_df)
    trade_mask = weight_change.abs() > 1e-9
    n_trades = int(trade_mask.values.sum())

    return {
        "equity": equity,
        "returns": pd.Series(net_rets, index=prices.index),
        "weights": w_df,
        "metrics": {
            "total_return": float(equity.iloc[-1] / init_cash - 1),
            "annual_return": float(equity.iloc[-1] / init_cash) ** (252 / len(equity)) - 1,
            "annual_vol": float(np.std(net_rets) * np.sqrt(252)),
            "sharpe_ratio": float(np.mean(net_rets) / np.std(net_rets) * np.sqrt(252)) if np.std(net_rets) > 0 else 0.0,
            "max_drawdown": float(((equity / equity.cummax()) - 1).min()),
            "n_trades": n_trades,
            "avg_daily_turnover": float(turnover.mean()),
        },
    }


def _vbt_run_with_vbt(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    init_cash: float = 1e6,
    commission: float = 0.00025,
    stamp_tax: float = 0.001,
    slippage: float = 0.0001,
    top_k: int = 30,
) -> Dict[str, Any]:
    """使用 vectorbt 库的高性能回测"""
    close = prices.iloc[:, 0] if prices.shape[1] == 1 else prices
    if hasattr(close, 'columns'):
        # 多资产: 用 vectorbt 的 Portfolio.from_signals 需要长表 -> pivot
        # 简化: 取等权组合 (一个虚拟 index 价格)
        if close.shape[1] > 1:
            return _vbt_run_pure(prices, signals, init_cash, commission, stamp_tax, slippage, top_k)
        close = close.iloc[:, 0]

    # signal -> entries/exits
    sig = signals.reindex(close.index).fillna(0).astype(int)
    if hasattr(sig, 'columns'):
        sig = sig.iloc[:, 0]
    entries = sig > 0
    exits = sig < 0

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=commission,
        slippage=slippage,
        freq="1D",
    )
    stats = pf.stats()
    return {
        "equity": pf.value(),
        "returns": pf.returns(),
        "metrics": {
            "total_return": float(pf.total_return()),
            "annual_return": float(stats.get("Annualized Return", 0.0) or 0.0),
            "annual_vol": float(stats.get("Annualized Volatility", 0.0) or 0.0),
            "sharpe_ratio": float(stats.get("Sharpe Ratio", 0.0) or 0.0),
            "max_drawdown": float(pf.max_drawdown()),
            "n_trades": int(stats.get("Total Trades", 0) or 0),
        },
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "equity": pd.Series(dtype=float),
        "returns": pd.Series(dtype=float),
        "weights": pd.DataFrame(),
        "metrics": {},
    }


# ============================================================================
# 2. 统一入口
# ============================================================================

def run_vectorized_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    init_cash: float = 1e6,
    commission: float = 0.00025,
    stamp_tax: float = 0.001,
    slippage: float = 0.0001,
    top_k: int = 30,
    use_vbt: bool = True,
) -> Dict[str, Any]:
    """
    统一向量化回测入口

    参数:
        data: [code, date, close, ...] 行情数据
        signals: [code, date, signal] 信号数据
        use_vbt: 是否使用 vectorbt (如可用)
    """
    if data.empty or signals.empty:
        return _empty_result()

    # pivot: date x code
    price_pivot = data.pivot_table(index='date', columns='code', values='close').sort_index()
    sig_pivot = signals.pivot_table(index='date', columns='code', values='signal',
                                     aggfunc='last').sort_index()
    # 对齐
    sig_pivot = sig_pivot.reindex(price_pivot.index).fillna(0)

    if use_vbt and HAS_VBT and price_pivot.shape[1] == 1:
        return _vbt_run_with_vbt(price_pivot, sig_pivot, init_cash, commission, stamp_tax, slippage, top_k)
    return _vbt_run_pure(price_pivot, sig_pivot, init_cash, commission, stamp_tax, slippage, top_k)


# ============================================================================
# 3. CLI 自检
# ============================================================================

def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n-stocks", type=int, default=20)
    ap.add_argument("--n-days", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    if args.self_test:
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=args.n_days, freq="B")
        codes = [f"{i:06d}.SH" for i in range(1, args.n_stocks + 1)]
        rows = []
        for c in codes:
            price = 10 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, args.n_days)))
            for d, p in zip(dates, price):
                rows.append({"date": d, "code": c, "close": p})
        data = pd.DataFrame(rows)
        # 简单动量信号
        pivot = data.pivot_table(index='date', columns='code', values='close')
        ret_5 = pivot.pct_change(5)
        sig = (ret_5.rank(axis=1, ascending=False) <= args.top_k).astype(int)
        sig_long = sig.stack().reset_index()
        sig_long.columns = ['date', 'code', 'signal']
        out = run_vectorized_backtest(data, sig_long, top_k=args.top_k, use_vbt=False)
        print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    _cli()
