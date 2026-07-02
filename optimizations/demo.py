"""
端到端演示: 因子表达式引擎 + 向量化回测引擎 协同工作
=====================================================
展示两个优化模块如何组合使用, 完成 "因子计算 -> 信号生成 -> 向量化回测" 全流程。

运行: python -m optimizations.demo
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizations.factor_expr import ExpressionEngine, FactorRegistry, register_builtins  # noqa: E402
from optimizations.vectorized_backtest import VectorizedBacktester  # noqa: E402


def make_market_data(n_days: int = 250, n_codes: int = 30, seed: int = 42) -> pd.DataFrame:
    """生成模拟行情面板 (MultiIndex: date, code)。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])

    n = len(idx)
    # 每只股票独立的随机游走
    rets = rng.normal(0.0005, 0.015, (n_days, n_codes))
    close = 10.0 * np.cumprod(1 + rets, axis=0)
    close = close.flatten()
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.lognormal(15, 0.8, n)
    turnover = rng.uniform(0.005, 0.04, n)
    total_mv = close * 1e8

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "turnover_rate": turnover,
            "total_mv": total_mv,
        },
        index=idx,
    )


def run_demo() -> dict:
    """运行端到端演示, 返回结果摘要。"""
    print("=" * 70)
    print("jingni-trader 优化验证: 因子表达式 + 向量化回测 端到端演示")
    print("=" * 70)

    # ---- 1. 生成模拟行情数据 ----
    print("\n[1/5] 生成模拟行情数据 (250交易日 x 30只股票)...")
    panel = make_market_data(n_days=250, n_codes=30)
    print(f"      数据形状: {panel.shape}, 日期范围: {panel.index.get_level_values(0).min().date()} ~ "
          f"{panel.index.get_level_values(0).max().date()}")

    # 转为宽表 (index=date, columns=code) 供回测使用
    close_wide = panel["close"].unstack("code")
    open_wide = panel["open"].unstack("code")

    # ---- 2. 因子表达式引擎: 计算多个因子 ----
    print("\n[2/5] 因子表达式引擎: 计算因子...")
    registry = FactorRegistry.instance()
    registry.clear()
    register_builtins(registry)
    print(f"      已注册 {len(registry)} 个内置因子, 类别: {registry.categories()}")

    engine = ExpressionEngine()
    # 用表达式定义一个复合因子: 20日反转 + 低波动 + 高换手
    composite_expr = "CSRank(-ROC($close, 20)) + CSRank(-STD(ROC($close, 1), 20)) + CSRank(MA($turnover_rate, 20))"

    t0 = time.perf_counter()
    composite_factor = engine.compute(composite_expr, panel)
    expr_time = time.perf_counter() - t0
    print(f"      复合因子表达式: {composite_expr}")
    print(f"      计算耗时: {expr_time*1000:.1f}ms, 有效值: {composite_factor.notna().sum()}/{len(composite_factor)}")

    # ---- 3. 生成目标权重 (Top-5 等权, 月度调仓) ----
    print("\n[3/5] 基于因子生成目标权重 (Top-5 等权, 月度调仓)...")
    factor_wide = composite_factor.unstack("code")

    # 每月第一个交易日调仓, 选 Top-5
    target_weights = pd.DataFrame(0.0, index=factor_wide.index, columns=factor_wide.columns)
    monthly_dates = factor_wide.index.to_series().groupby(factor_wide.index.to_period("M")).first()
    for date in monthly_dates:
        if date not in factor_wide.index:
            continue
        row = factor_wide.loc[date].dropna()
        if len(row) < 5:
            continue
        top5 = row.nlargest(5).index
        target_weights.loc[date, top5] = 1.0 / 5

    n_rebalance = (target_weights.sum(axis=1) > 0).sum()
    print(f"      调仓次数: {n_rebalance}, 每次持有 5 只")

    # ---- 4. 向量化回测 ----
    print("\n[4/5] 向量化回测 (T+1, 完整A股费用)...")
    bt = VectorizedBacktester(t_plus_1=True, deal_price="open")

    # 基准: 等权全市场
    benchmark = close_wide.mean(axis=1)
    benchmark = benchmark / benchmark.iloc[0]

    t0 = time.perf_counter()
    result = bt.run_target_weight(
        target_weights=target_weights,
        price=close_wide,
        open_price=open_wide,
        benchmark=benchmark,
        initial_capital=1_000_000,
    )
    bt_time = time.perf_counter() - t0
    print(f"      回测耗时: {bt_time*1000:.1f}ms")

    # ---- 5. 输出绩效 ----
    print("\n[5/5] 绩效指标:")
    print("-" * 50)
    metrics = result.metrics
    for key in [
        "total_return", "annual_return", "annual_volatility", "sharpe",
        "max_drawdown", "calmar", "sortino", "win_rate",
        "benchmark_return", "excess_return", "tracking_error", "information_ratio",
        "avg_turnover", "annual_turnover",
    ]:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:25s}: {val:>10.4f}")
            else:
                print(f"  {key:25s}: {val}")
    print("-" * 50)
    print(f"  最终净值: {result.equity_curve.iloc[-1]:,.0f} (初始 1,000,000)")

    summary = {
        "expr_time_ms": expr_time * 1000,
        "backtest_time_ms": bt_time * 1000,
        "n_factors_registered": len(registry),
        "n_rebalance": int(n_rebalance),
        "metrics": metrics,
        "final_equity": float(result.equity_curve.iloc[-1]),
    }
    return summary


if __name__ == "__main__":
    run_demo()
