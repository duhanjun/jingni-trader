"""
comprehensive_metrics.py
========================

扩展 jingni-trader 现有 BaseBacktestMetrics，增加 15+ 个业界标准绩效指标。

灵感来源：
- backtesting.py 的 compute_stats() (https://github.com/kernc/backtesting.py)
- Qlib 的 port_analysis_report
- 业界标准 quantstats / pyfolio

设计原则：
- 纯函数 + 静态方法，可独立调用，方便单元测试
- 兼容现有 BaseBacktestMetrics 行为
- 支持 benchmark 对比（Alpha / Beta / Information Ratio）
- 全部基于 pd.Series / np.ndarray，不依赖外部状态

注意：
- 本文件为验证性代码，不修改 main 分支的 BaseBacktestMetrics
- 适用于回测/绩效报告的指标计算
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，b 为 0 时返回 default。"""
    if b == 0 or np.isnan(b) or np.isinf(b):
        return default
    return a / b


def geometric_mean(returns: pd.Series) -> float:
    """几何平均收益率（对数累计等效）。"""
    arr = returns.fillna(0).to_numpy() + 1
    if (arr <= 0).any():
        return 0.0
    return float(np.exp(np.log(arr).sum() / max(len(arr), 1)) - 1)


def calc_total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def calc_cagr(equity: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(equity) < 2:
        return 0.0
    n_years = len(equity) / trading_days
    if n_years <= 0:
        return 0.0
    eq0 = float(equity.iloc[0])
    eqT = float(equity.iloc[-1])
    if eq0 <= 0 or eqT <= 0:
        return 0.0
    try:
        return float((eqT ** (1 / n_years)) / (eq0 ** (1 / n_years)) - 1)
    except (ValueError, OverflowError):
        return 0.0


def calc_annual_return_from_returns(returns: pd.Series,
                                    trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """年化收益率（几何平均法，更稳健）。"""
    if len(returns) == 0:
        return 0.0
    g = geometric_mean(returns)
    return float((1 + g) ** trading_days - 1)


def calc_volatility(returns: pd.Series,
                    trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def calc_sharpe(returns: pd.Series, risk_free: float = 0.03,
                trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    vol = calc_volatility(returns, trading_days)
    ann_ret = calc_annual_return_from_returns(returns, trading_days)
    return _safe_div(ann_ret - risk_free, vol)


def calc_sortino(returns: pd.Series, risk_free: float = 0.03,
                 trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """索提诺比率，仅用下行波动率。"""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    neg = r[r < 0]
    if len(neg) < 2:
        return 0.0
    downside = float(neg.std(ddof=1) * np.sqrt(trading_days))
    ann_ret = calc_annual_return_from_returns(returns, trading_days)
    return _safe_div(ann_ret - risk_free, downside)


def calc_max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def calc_max_drawdown_duration(equity: pd.Series) -> int:
    """最大回撤持续期（bar 数）。"""
    if len(equity) < 2:
        return 0
    dd = equity / equity.cummax() - 1.0
    in_dd = dd < 0
    if not in_dd.any():
        return 0
    groups = (in_dd != in_dd.shift()).cumsum()
    durations = in_dd.groupby(groups).cumsum()
    return int(durations.max())


def calc_calmar(equity: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    cagr = calc_cagr(equity, trading_days)
    mdd = abs(calc_max_drawdown(equity))
    return _safe_div(cagr, mdd)


def calc_win_rate(trades: Union[pd.DataFrame, pd.Series]) -> float:
    if isinstance(trades, pd.DataFrame) and "pnl" in trades.columns:
        if len(trades) == 0:
            return 0.0
        return float((trades["pnl"] > 0).sum() / len(trades))
    if isinstance(trades, pd.Series) and len(trades) > 0:
        return float((trades > 0).sum() / len(trades))
    return 0.0


def calc_profit_factor(trades: Union[pd.DataFrame, pd.Series]) -> float:
    if isinstance(trades, pd.DataFrame) and "pnl" in trades.columns:
        pnl = trades["pnl"]
    elif isinstance(trades, pd.Series):
        pnl = trades
    else:
        return 0.0
    if len(pnl) == 0:
        return 0.0
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    return _safe_div(float(gross_profit), float(gross_loss))


def calc_expectancy(trades: Union[pd.DataFrame, pd.Series]) -> float:
    """平均每笔交易的预期收益 (绝对值)。"""
    if isinstance(trades, pd.DataFrame) and "pnl" in trades.columns:
        pnl = trades["pnl"]
    elif isinstance(trades, pd.Series):
        pnl = trades
    else:
        return 0.0
    if len(pnl) == 0:
        return 0.0
    return float(pnl.mean())


def calc_sqn(trades: Union[pd.DataFrame, pd.Series]) -> float:
    """System Quality Number = sqrt(N) * mean(pnl) / std(pnl)。"""
    if isinstance(trades, pd.DataFrame) and "pnl" in trades.columns:
        pnl = trades["pnl"]
    elif isinstance(trades, pd.Series):
        pnl = trades
    else:
        return 0.0
    if len(pnl) < 2:
        return 0.0
    return _safe_div(
        float(np.sqrt(len(pnl)) * pnl.mean()),
        float(pnl.std(ddof=1)),
    )


def calc_kelly(trades: Union[pd.DataFrame, pd.Series]) -> float:
    """Kelly 凯利公式：胜率 - (1-胜率) / 盈亏比。"""
    if isinstance(trades, pd.DataFrame) and "pnl" in trades.columns:
        pnl = trades["pnl"]
    elif isinstance(trades, pd.Series):
        pnl = trades
    else:
        return 0.0
    if len(pnl) == 0:
        return 0.0
    win = pnl[pnl > 0]
    loss = pnl[pnl < 0]
    if len(win) == 0 or len(loss) == 0:
        return 0.0
    p = len(win) / len(pnl)
    b = float(win.mean() / abs(loss.mean()))
    return float(p - (1 - p) / b)


def calc_exposure_time(positions: pd.Series) -> float:
    """持仓时间占比（positions > 0 的时间比例）。"""
    if len(positions) == 0:
        return 0.0
    return float((positions > 0).mean())


def calc_buy_and_hold_return(close: pd.Series) -> float:
    """买入持有基准收益。"""
    if len(close) < 2:
        return 0.0
    return float(close.iloc[-1] / close.iloc[0] - 1)


def calc_alpha_beta(returns: pd.Series, bench_returns: pd.Series,
                    risk_free: float = 0.03) -> Dict[str, float]:
    """计算 Jensen Alpha / Beta（CAPM）。"""
    r = returns.dropna()
    b = bench_returns.dropna()
    common = r.index.intersection(b.index)
    if len(common) < 2:
        return {"alpha": 0.0, "beta": 0.0, "alpha_annualized": 0.0}
    r = r.loc[common]
    b = b.loc[common]
    cov = np.cov(r, b, ddof=1)
    beta = _safe_div(float(cov[0, 1]), float(cov[1, 1]))
    ann_r = calc_annual_return_from_returns(r)
    ann_b = calc_annual_return_from_returns(b)
    alpha = ann_r - (risk_free + beta * (ann_b - risk_free))
    return {"alpha": float(alpha), "beta": float(beta), "alpha_annualized": float(alpha)}


def calc_information_ratio(returns: pd.Series, bench_returns: pd.Series,
                           trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Information Ratio = 超额收益年化 / Tracking Error。"""
    r = returns.dropna()
    b = bench_returns.dropna()
    common = r.index.intersection(b.index)
    if len(common) < 2:
        return 0.0
    excess = r.loc[common] - b.loc[common]
    if excess.std(ddof=1) == 0:
        return 0.0
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(trading_days))


def calc_trade_pnl_stats(trades: pd.DataFrame) -> Dict[str, float]:
    """单笔交易统计。"""
    if trades.empty or "pnl" not in trades.columns:
        return {
            "best_trade": 0.0, "worst_trade": 0.0,
            "avg_trade": 0.0, "median_trade": 0.0,
            "n_trades": 0,
        }
    pnl = trades["pnl"]
    return {
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "avg_trade": float(pnl.mean()),
        "median_trade": float(pnl.median()),
        "n_trades": int(len(trades)),
    }


def compute_full_metrics(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    positions: Optional[pd.Series] = None,
    benchmark_close: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, Any]:
    """
    一次性计算业界标准的全套绩效指标。

    参数：
        equity: 资金曲线 (pd.Series, index=日期)
        trades: 交易记录 DataFrame，需含 'pnl' 列（可选）
        positions: 每日持仓数量 (pd.Series, 可选)
        benchmark_close: 基准收盘价 (pd.Series, 可选)
        risk_free: 无风险利率
        trading_days: 年交易日数

    返回：
        包含 25+ 指标的字典
    """
    returns = equity.pct_change().dropna()

    bench_returns = None
    if benchmark_close is not None and len(benchmark_close) >= 2:
        bench_returns = benchmark_close.pct_change().dropna()
        bh_ret = calc_buy_and_hold_return(benchmark_close)
    else:
        bh_ret = 0.0

    metrics: Dict[str, Any] = {
        # 收益类
        "total_return": calc_total_return(equity),
        "cagr": calc_cagr(equity, trading_days),
        "annual_return": calc_annual_return_from_returns(returns, trading_days),
        "buy_hold_return": bh_ret,
        # 风险类
        "volatility_annual": calc_volatility(returns, trading_days),
        "max_drawdown": calc_max_drawdown(equity),
        "max_drawdown_duration": calc_max_drawdown_duration(equity),
        # 风险调整收益
        "sharpe_ratio": calc_sharpe(returns, risk_free, trading_days),
        "sortino_ratio": calc_sortino(returns, risk_free, trading_days),
        "calmar_ratio": calc_calmar(equity, trading_days),
    }

    # 交易统计
    if trades is not None and not trades.empty:
        pnl = trades["pnl"] if "pnl" in trades.columns else pd.Series(dtype=float)
        metrics.update({
            "win_rate": calc_win_rate(pnl),
            "profit_factor": calc_profit_factor(pnl),
            "expectancy": calc_expectancy(pnl),
            "sqn": calc_sqn(pnl),
            "kelly_criterion": calc_kelly(pnl),
            **calc_trade_pnl_stats(trades if "pnl" in trades.columns else pd.DataFrame()),
        })
    else:
        metrics.update({
            "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
            "sqn": 0.0, "kelly_criterion": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
            "avg_trade": 0.0, "median_trade": 0.0, "n_trades": 0,
        })

    # 持仓时间
    if positions is not None and len(positions) > 0:
        metrics["exposure_time"] = calc_exposure_time(positions)
    else:
        metrics["exposure_time"] = 0.0

    # 基准对比
    if bench_returns is not None:
        ab = calc_alpha_beta(returns, bench_returns, risk_free)
        metrics["alpha"] = ab["alpha"]
        metrics["beta"] = ab["beta"]
        metrics["information_ratio"] = calc_information_ratio(returns, bench_returns, trading_days)
    else:
        metrics["alpha"] = 0.0
        metrics["beta"] = 0.0
        metrics["information_ratio"] = 0.0

    return metrics


if __name__ == "__main__":
    # 简单自测
    rng = np.random.default_rng(42)
    n = 252
    ret = pd.Series(rng.normal(0.0005, 0.015, n))
    equity = (1 + ret).cumprod() * 1_000_000
    trades = pd.DataFrame({"pnl": rng.normal(100, 1000, 20)})
    print(compute_full_metrics(equity, trades=trades, risk_free=0.03))
