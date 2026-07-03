"""
增强回测绩效指标

借鉴来源：
- Qlib backtest.performance：turnover / alpha / beta / information_ratio
- QuantStats：max_drawdown_duration / profit_factor / sortino 等

优化点：
jingni-trader 现有 `BaseBacktestMetrics.calc_all_metrics` 已含基础指标
（total_return / annual_return / volatility / sharpe / max_drawdown /
calmar / sortino / win_rate），但缺少以下对策略评估至关重要的指标：

  - turnover: 换手率（衡量交易成本敏感度，越高成本侵蚀越严重）
  - alpha / beta: 相对基准的超额收益与市场敏感度（CAPM）
  - information_ratio: 信息比率（超额收益 / 跟踪误差）
  - max_drawdown_duration: 最大回撤持续期（资金被套时间）

这些指标在 Qlib / QuantStats 中均为标配，是机构量化评估的硬性要求。
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


def calc_turnover(
    positions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    trading_days: int = 252,
) -> float:
    """
    计算日均换手率（年化）。

    换手率 = 单边交易额 / 平均持仓市值，反映策略交易频繁程度。

    参数:
        positions: 各日持仓明细，至少含 date, code, market_value（或 shares*price）
        equity_curve: 含 date, equity 的净值曲线
        trading_days: 年化天数

    返回:
        年化换手率（单边）
    """
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return 0.0

    eq = equity_curve.set_index("date")["equity"]
    if len(eq) < 2:
        return 0.0

    # 若有 positions 明细，按日聚合持仓市值变化计算换手
    if (
        positions is not None
        and not positions.empty
        and {"date", "code", "market_value"}.issubset(positions.columns)
    ):
        # 每日每股票市值
        daily_mv = positions.groupby(["date", "code"])["market_value"].sum().unstack(fill_value=0)
        # 对齐到 equity 日期
        daily_mv = daily_mv.reindex(eq.index, fill_value=0)
        # 单边换手 = 当日买入额 + 当日卖出额 的一半（单边）
        # 持仓市值增加部分视为买入，减少部分视为卖出
        # 首日 diff 为 NaN（初始建仓不计入换手），需保留 NaN 以便 mean() 跳过
        diff = daily_mv.diff()
        valid_mask = diff.notna().any(axis=1)
        buy = diff.where(diff > 0, 0).sum(axis=1).where(valid_mask)
        sell = (-diff).where(diff < 0, 0).sum(axis=1).where(valid_mask)
        # 单边换手额 = (buy+sell)/2
        daily_turnover = (buy + sell) / 2 / eq.replace(0, np.nan)
        avg_daily_turnover = daily_turnover.mean()
    else:
        # 无持仓明细时，用净值变化粗估（仅作兜底，精度有限）
        returns = eq.pct_change().dropna()
        avg_daily_turnover = returns.abs().mean()

    if pd.isna(avg_daily_turnover):
        return 0.0
    return float(avg_daily_turnover * trading_days)


def calc_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Tuple[float, float]:
    """
    计算 CAPM Alpha 与 Beta（相对基准）。

    Alpha = 年化超额收益（剥离市场风险后的纯 α）
    Beta = 策略对基准的敏感度

    参数:
        strategy_returns: 策略日收益率 Series
        benchmark_returns: 基准日收益率 Series（index 对齐）
        risk_free: 年化无风险利率
        trading_days: 年化天数

    返回:
        (alpha, beta)，均已年化
    """
    if strategy_returns is None or benchmark_returns is None:
        return 0.0, 0.0
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0, 0.0

    # 对齐 index
    df = pd.DataFrame({"s": strategy_returns, "b": benchmark_returns}).dropna()
    if len(df) < 2:
        return 0.0, 0.0

    rf_daily = risk_free / trading_days
    s_excess = df["s"] - rf_daily
    b_excess = df["b"] - rf_daily

    # OLS: s_excess = alpha_daily + beta * b_excess
    x = b_excess.to_numpy()
    y = s_excess.to_numpy()
    x_with_const = np.column_stack([np.ones(len(x)), x])
    try:
        coef, *_ = np.linalg.lstsq(x_with_const, y, rcond=None)
        alpha_daily, beta = float(coef[0]), float(coef[1])
    except np.linalg.LinAlgError:
        return 0.0, 0.0

    # 年化 alpha
    alpha = alpha_daily * trading_days
    return alpha, beta


def calc_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    计算信息比率（IR）= 年化超额收益 / 年化跟踪误差。

    IR 衡量主动管理的风险调整后收益，是主动策略的核心评估指标。
    """
    if strategy_returns is None or benchmark_returns is None:
        return 0.0
    df = pd.DataFrame({"s": strategy_returns, "b": benchmark_returns}).dropna()
    if len(df) < 2:
        return 0.0

    excess = df["s"] - df["b"]
    tracking_error = excess.std() * np.sqrt(trading_days)
    if tracking_error == 0:
        return 0.0
    annual_excess = excess.mean() * trading_days
    return float(annual_excess / tracking_error)


def calc_max_drawdown_duration(equity_curve: pd.Series) -> int:
    """
    计算最大回撤持续期（交易日数）。

    从净值创新高到回到前高的最长天数，反映资金被套时间。
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 0
    cumulative_max = equity_curve.cummax()
    underwater = equity_curve < cumulative_max

    max_duration = 0
    current_duration = 0
    for u in underwater:
        if u:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return int(max_duration)


def calc_all_enhanced_metrics(
    equity_curve: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    positions: Optional[pd.DataFrame] = None,
    equity_df: Optional[pd.DataFrame] = None,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """
    计算增强绩效指标（补充现有 BaseBacktestMetrics.calc_all_metrics）。

    参数:
        equity_curve: 净值 Series（index 为日期）
        benchmark_returns: 基准日收益率 Series（可选，用于 alpha/beta/IR）
        positions: 持仓明细 DataFrame（可选，用于精确换手率）
        equity_df: 含 date/equity 列的 DataFrame（可选，用于换手率）
        risk_free: 年化无风险利率
        trading_days: 年化天数

    返回:
        含 turnover / alpha / beta / information_ratio / max_drawdown_duration 的字典
    """
    if equity_curve is None or len(equity_curve) < 2:
        return {
            "turnover": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "information_ratio": 0.0,
            "max_drawdown_duration": 0,
        }

    returns = equity_curve.pct_change().dropna()

    # 换手率
    turnover = 0.0
    if equity_df is not None and not equity_df.empty:
        turnover = calc_turnover(positions, equity_df, trading_days)
    elif positions is not None and not positions.empty:
        turnover = calc_turnover(positions, pd.DataFrame(), trading_days)

    # Alpha / Beta / IR（需基准）
    alpha, beta, ir = 0.0, 0.0, 0.0
    if benchmark_returns is not None and not benchmark_returns.empty:
        alpha, beta = calc_alpha_beta(returns, benchmark_returns, risk_free, trading_days)
        ir = calc_information_ratio(returns, benchmark_returns, trading_days)

    # 最大回撤持续期
    mdd_duration = calc_max_drawdown_duration(equity_curve)

    return {
        "turnover": round(float(turnover), 6),
        "alpha": round(float(alpha), 6),
        "beta": round(float(beta), 6),
        "information_ratio": round(float(ir), 6),
        "max_drawdown_duration": int(mdd_duration),
    }
