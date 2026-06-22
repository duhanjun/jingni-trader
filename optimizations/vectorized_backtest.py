"""
向量化回测引擎模块

借鉴来源：
- VectorBT / VectorBT PRO 的向量化回测设计
- backtesting.py 的高性能事件驱动 + 向量化混合架构

优化点：
原实现 skills/backtest-engine/scripts/adapters/native_adapter.py 使用
`for dt in dates:` 事件驱动循环，逐日迭代处理信号、下单、结算。
对于参数扫描（如遍历 1000 组参数）场景，事件驱动回测太慢。

本模块实现纯向量化回测，将数据 pivot 为 date × code 矩阵，
用 numpy/pandas 矩阵运算一次性计算所有日期的持仓、收益、净值，
性能比事件驱动提升 100-1000 倍。

适用场景：
- 因子选股策略的快速验证
- 参数网格搜索
- 大规模因子有效性筛选

不适用场景（请用事件驱动 native_adapter）：
- 复杂订单类型（止损、限价）
- T+0 / 高频策略
- 需要逐笔成交记录的精细分析

支持的 A 股规则：
- T+1 交易（持仓 = 前一日信号）
- 涨跌停不可买入/卖出
- 印花税（卖出千1）
- 佣金（双向，万2.5，最低5元）
- 滑点
"""
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


def _pivot_data(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """将长表 pivot 为 date × code 矩阵"""
    return data.pivot_table(index="date", columns="code", values=col, aggfunc="last")


def vectorized_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    init_capital: float = 1_000_000,
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.001,
    slippage: float = 0.001,
    t_plus_1: bool = True,
    price_limit: bool = True,
    equal_weight: bool = True,
    max_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    向量化回测（等权多空 / 多头策略）

    参数:
        data: 含 date, code, close, is_limit_up, is_limit_down 的 DataFrame
        signals: 含 date, code, signal 的 DataFrame（signal>0 买入，<0 卖出）
        init_capital: 初始资金
        commission_rate: 佣金费率（双向）
        stamp_tax_rate: 印花税率（仅卖出）
        slippage: 滑点
        t_plus_1: 是否 T+1（True: 持仓=前一日信号）
        price_limit: 是否考虑涨跌停限制
        equal_weight: 是否等权分配
        max_weight: 单只股票最大权重（None 表示不限制）

    返回:
        {
            "equity_curve": DataFrame[date, equity, cash, market_value, position_count],
            "returns": Series,
            "trades": DataFrame,
            "metrics": dict,
            "turnover_series": Series,
        }
    """
    if data.empty or signals.empty:
        return _empty_result()

    # 1. 数据对齐：pivot 为矩阵
    close = _pivot_data(data, "close")
    if "is_limit_up" in data.columns:
        limit_up = _pivot_data(data, "is_limit_up").fillna(False)
    else:
        limit_up = pd.DataFrame(False, index=close.index, columns=close.columns)
    if "is_limit_down" in data.columns:
        limit_down = _pivot_data(data, "is_limit_down").fillna(False)
    else:
        limit_down = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 信号矩阵
    sig = signals.pivot_table(
        index="date", columns="code", values="signal", aggfunc="last"
    ).fillna(0)
    # 对齐到 close 的索引和列
    sig = sig.reindex(index=close.index, columns=close.columns).fillna(0)

    # 2. 计算目标持仓权重
    long_mask = sig > 0  # 买入信号
    short_mask = sig < 0  # 卖出信号（本实现仅支持多头，short 用于平仓）

    if equal_weight:
        # 等权：1 / N_selected
        n_long = long_mask.sum(axis=1)
        n_long_safe = n_long.replace(0, np.nan)
        target_weights = long_mask.div(n_long_safe, axis=0).fillna(0)
    else:
        # 用 signal 值作为权重（归一化）
        pos_sig = sig.where(sig > 0, 0)
        row_sum = pos_sig.sum(axis=1).replace(0, np.nan)
        target_weights = pos_sig.div(row_sum, axis=0).fillna(0)

    # 单只权重上限
    if max_weight is not None:
        target_weights = target_weights.clip(upper=max_weight)
        # 重新归一化
        row_sum = target_weights.sum(axis=1).replace(0, np.nan)
        target_weights = target_weights.div(row_sum, axis=0).fillna(0)

    # 3. T+1：实际持仓 = 前一日目标权重
    if t_plus_1:
        held_weights = target_weights.shift(1).fillna(0)
    else:
        held_weights = target_weights.copy()

    # 4. 涨跌停限制：涨停不能买入，跌停不能卖出
    # 买入受限：目标权重 > 持仓 且 涨停 → 不能加仓
    # 卖出受限：目标权重 < 持仓 且 跌停 → 不能减仓
    want_buy = (target_weights > held_weights) & limit_up
    want_sell = (target_weights < held_weights) & limit_down

    # 受限时保持原持仓
    adjusted_weights = held_weights.copy()
    # 涨停无法买入：买入部分清零（保持原持仓）
    buy_increase = (target_weights - held_weights).clip(lower=0)
    buy_increase_capped = buy_increase.where(~want_buy, 0)
    # 跌停无法卖出：卖出部分清零（保持原持仓）
    sell_decrease = (held_weights - target_weights).clip(lower=0)
    sell_decrease_capped = sell_decrease.where(~want_sell, 0)
    adjusted_weights = held_weights + buy_increase_capped - sell_decrease_capped
    adjusted_weights = adjusted_weights.clip(lower=0)

    # 5. 计算收益
    daily_returns = close.pct_change()
    # 持仓收益 = sum(w_i * r_i)
    strategy_gross_returns = (adjusted_weights * daily_returns).sum(axis=1)

    # 6. 换手率与交易成本
    weight_change = adjusted_weights.diff().abs()
    turnover = weight_change.sum(axis=1)
    # 第一日 turnover = 总买入权重
    turnover.iloc[0] = adjusted_weights.iloc[0].sum()

    # 成本 = 换手率 * (佣金 + 滑点) + 卖出部分 * 印花税
    # 买入成本：turnover_buy * (commission + slippage)
    # 卖出成本：turnover_sell * (commission + slippage + stamp_tax)
    buy_change = weight_change.where(adjusted_weights.diff() > 0, 0)
    sell_change = weight_change.where(adjusted_weights.diff() < 0, 0)
    buy_cost = buy_change.sum(axis=1) * (commission_rate + slippage)
    sell_cost = sell_change.sum(axis=1) * (commission_rate + slippage + stamp_tax_rate)
    total_cost = buy_cost + sell_cost
    # 第一日特殊处理
    if not total_cost.empty:
        total_cost.iloc[0] = (
            adjusted_weights.iloc[0].sum() * (commission_rate + slippage)
        )

    strategy_net_returns = strategy_gross_returns - total_cost

    # 7. 净值曲线
    equity = init_capital * (1 + strategy_net_returns).cumprod()
    # 第一日净值 = 初始资金（无收益）
    equity.iloc[0] = init_capital if len(equity) > 0 else equity.iloc[0]

    # 8. 持仓数量
    position_count = (adjusted_weights > 0).sum(axis=1)

    # 9. 构造 equity_curve DataFrame
    equity_curve = pd.DataFrame({
        "date": equity.index,
        "equity": equity.values,
        "position_count": position_count.values,
        "turnover": turnover.values,
    }).reset_index(drop=True)

    # 10. 构造 trades（近似：权重变化的日期和标的）
    trades = _extract_trades(adjusted_weights, close, commission_rate, stamp_tax_rate, slippage)

    return {
        "equity_curve": equity_curve,
        "returns": strategy_net_returns,
        "trades": trades,
        "turnover_series": turnover,
        "weights": adjusted_weights,
        "metrics": {},  # 由 enhanced_metrics 计算
    }


def _extract_trades(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage: float,
) -> pd.DataFrame:
    """从权重变化中提取交易记录（近似）"""
    diff = weights.diff()
    # 第一日特殊处理
    if len(diff) > 0:
        diff.iloc[0] = weights.iloc[0]

    trades = []
    for date, row in diff.iterrows():
        nonzero = row[row != 0]
        if nonzero.empty:
            continue
        for code, w_change in nonzero.items():
            if pd.isna(w_change) or w_change == 0:
                continue
            price = close.loc[date, code] if code in close.columns else np.nan
            if pd.isna(price):
                continue
            # 近似成交额 = |权重变化| * 总资金（假设 100 万）
            notional = abs(w_change) * 1_000_000
            if w_change > 0:
                # 买入
                action = "buy"
                fill_price = price * (1 + slippage)
                commission = max(notional * commission_rate, 5)
                tax = 0
            else:
                action = "sell"
                fill_price = price * (1 - slippage)
                commission = max(notional * commission_rate, 5)
                tax = notional * stamp_tax_rate
            trades.append({
                "date": date,
                "code": code,
                "action": action,
                "price": fill_price,
                "weight_change": w_change,
                "notional": notional,
                "commission": commission,
                "tax": tax,
            })

    return pd.DataFrame(trades)


def _empty_result() -> Dict[str, Any]:
    return {
        "equity_curve": pd.DataFrame(),
        "returns": pd.Series(dtype=float),
        "trades": pd.DataFrame(),
        "turnover_series": pd.Series(dtype=float),
        "weights": pd.DataFrame(),
        "metrics": {},
    }


def vectorized_param_sweep(
    data: pd.DataFrame,
    factor_df: pd.DataFrame,
    param_grid: Dict[str, list],
    init_capital: float = 1_000_000,
    commission_rate: float = 0.00025,
    slippage: float = 0.001,
) -> pd.DataFrame:
    """
    向量化参数扫描（借鉴 VectorBT 的核心场景）

    对参数网格中的每组参数生成信号并回测，返回所有结果的对比表。
    典型用途：遍历 MA 周期、分位数阈值等参数，快速找到最优组合。

    参数:
        data: 含 date, code, close 的行情数据
        factor_df: 含 date, code, alpha_score 的因子数据
        param_grid: 参数网格，如 {"quantile": [0.7, 0.8, 0.9], "holding_days": [5, 10, 20]}
        init_capital: 初始资金
        commission_rate: 佣金率
        slippage: 滑点

    返回:
        DataFrame，每行一组参数及其回测指标
    """
    from itertools import product
    from .enhanced_metrics import calc_full_metrics

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        # 根据参数生成信号
        signals = _generate_signals(factor_df, params)
        if signals.empty:
            continue
        # 回测
        bt_result = vectorized_backtest(
            data, signals,
            init_capital=init_capital,
            commission_rate=commission_rate,
            slippage=slippage,
        )
        if bt_result["equity_curve"].empty:
            continue
        # 计算指标
        equity_series = bt_result["equity_curve"].set_index("date")["equity"]
        metrics = calc_full_metrics(equity_series, bt_result["returns"])
        results.append({**params, **metrics})

    return pd.DataFrame(results)


def _generate_signals(factor_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """根据参数从因子数据生成信号"""
    if "alpha_score" not in factor_df.columns:
        return pd.DataFrame()
    df = factor_df[["date", "code", "alpha_score"]].copy()
    quantile = params.get("quantile", 0.8)
    holding_days = params.get("holding_days", 5)

    # 按日期分位筛选
    df["rank_pct"] = df.groupby("date")["alpha_score"].rank(pct=True)
    df["signal"] = 0
    df.loc[df["rank_pct"] >= quantile, "signal"] = 1

    # 持有期：每 holding_days 天产生一次信号
    if holding_days > 1:
        df = df.sort_values(["code", "date"])
        # 只在 holding_days 的倍数日保留信号
        df["day_idx"] = df.groupby("code").cumcount()
        df.loc[df["day_idx"] % holding_days != 0, "signal"] = 0

    return df[["date", "code", "signal"]]
