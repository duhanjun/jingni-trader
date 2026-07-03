"""
metrics - 扩展绩效指标库

借鉴来源:
  - vectorbt / vectorbt PRO (Portfolio.stats 接口)
    https://vectorbt.dev/api/portfolio/base/
  - Riskfolio-Lib 的下行/上行情形指标
  - 行业实践: Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)

补强点 (相对 jingni-trader 现有 BaseBacktestMetrics):
  1. 信息比率 IR (excess return / tracking error)
  2. Alpha / Beta (相对基准)
  3. Deflated Sharpe Ratio (对过拟合/多重检验修正)
  4. 年化换手率 / 平均持仓周期
  5. 下行捕获 / 上行捕获 (Up/Down capture)
  6. 回撤序列 (含持续期、修复期)
  7. 收益分布 (skew, kurtosis, VaR/CVaR)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger("quant_opt.metrics")

TRADING_DAYS = 252


# ============================================================================
# 1. 基础收益与风险
# ============================================================================

def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def annualized_return(equity: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    years = len(equity) / trading_days
    if years <= 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def annualized_vol(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(trading_days))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.03,
                 trading_days: int = TRADING_DAYS) -> float:
    vol = annualized_vol(returns, trading_days)
    if vol == 0 or not np.isfinite(vol):
        return 0.0
    ann = returns.mean() * trading_days
    sr = (ann - risk_free) / vol
    if not np.isfinite(sr):
        return 0.0
    return float(sr)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.03,
                  trading_days: int = TRADING_DAYS) -> float:
    """下行风险调整的 Sharpe"""
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std() * np.sqrt(trading_days)
    if downside_std == 0:
        return 0.0
    ann = returns.mean() * trading_days
    return float((ann - risk_free) / downside_std)


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def calmar_ratio(equity: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(annualized_return(equity, trading_days) / mdd)


# ============================================================================
# 2. 基准相关
# ============================================================================

def alpha_beta(returns: pd.Series, bench_returns: pd.Series,
               risk_free: float = 0.03, trading_days: int = TRADING_DAYS) -> Tuple[float, float]:
    """CAPM 回归: r_p - rf = alpha + beta * (r_b - rf) + eps"""
    aligned = pd.concat([returns, bench_returns], axis=1, join='inner').dropna()
    if len(aligned) < 20:
        return 0.0, 0.0
    aligned.columns = ['p', 'b']
    aligned['ex_p'] = aligned['p'] - risk_free / trading_days
    aligned['ex_b'] = aligned['b'] - risk_free / trading_days
    cov = aligned[['ex_p', 'ex_b']].cov()
    var_b = cov.loc['ex_b', 'ex_b']
    if var_b == 0 or np.isnan(var_b):
        return 0.0, 0.0
    beta = float(cov.loc['ex_p', 'ex_b'] / var_b)
    alpha_daily = aligned['ex_p'].mean() - beta * aligned['ex_b'].mean()
    alpha_ann = float(alpha_daily * trading_days)
    return alpha_ann, beta


def information_ratio(returns: pd.Series, bench_returns: pd.Series,
                      trading_days: int = TRADING_DAYS) -> float:
    """主动收益 / 跟踪误差"""
    diff = returns - bench_returns
    diff = diff.dropna()
    if len(diff) < 2:
        return 0.0
    te = diff.std() * np.sqrt(trading_days)
    if te == 0:
        return 0.0
    return float(diff.mean() * trading_days / te)


def up_down_capture(returns: pd.Series, bench_returns: pd.Series) -> Tuple[float, float]:
    """上行捕获 / 下行捕获"""
    df = pd.concat([returns, bench_returns], axis=1, join='inner').dropna()
    df.columns = ['p', 'b']
    up = df[df['b'] > 0]
    dn = df[df['b'] < 0]
    up_cap = (up['p'].mean() / up['b'].mean()) if len(up) > 0 and up['b'].mean() != 0 else 0.0
    dn_cap = (dn['p'].mean() / dn['b'].mean()) if len(dn) > 0 and dn['b'].mean() != 0 else 0.0
    return float(up_cap), float(dn_cap)


# ============================================================================
# 3. 分布 / 极端风险
# ============================================================================

def var_cvar(returns: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
    """VaR / CVaR (历史法, 损失为正)"""
    if len(returns) < 20:
        return 0.0, 0.0
    q = (1 - confidence) * 100
    var = float(-np.percentile(returns, q))
    cvar = float(-returns[returns <= np.percentile(returns, q)].mean())
    return var, cvar


def deflated_sharpe(observed_sharpe: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
    调整多重检验造成的 Sharpe 虚高。

    返回的是: (observed_sharpe - e_max_sharpe) / SE_of_sharpe
    当 n_trials=1 时, e_max_sharpe ≈ 0, 退化为普通 Sharpe
    当 n_trials 增大, e_max_sharpe 增大, deflated 下降
    """
    if n_trials <= 1 or n_obs <= 1:
        return observed_sharpe
    e_max_sharpe = expected_max_sharpe(n_trials, n_obs)
    if not np.isfinite(e_max_sharpe):
        return observed_sharpe
    # SE of Sharpe, 考虑偏度/峰度
    var = ((1 + 0.5 * observed_sharpe ** 2
            - skew * observed_sharpe
            + (kurt - 3) / 4 * observed_sharpe ** 2) / (n_obs - 1))
    if var <= 0 or not np.isfinite(var):
        return float(observed_sharpe - e_max_sharpe)
    se = np.sqrt(var)
    if se == 0:
        return float(observed_sharpe - e_max_sharpe)
    return float((observed_sharpe - e_max_sharpe) / se)


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """E[max(SR_i)] 的渐近估计"""
    if n_trials <= 0:
        return 0.0
    # 渐近 (n_trials 较大): E[max] ≈ (1 - gamma) * Phi^(-1)(1 - 1/n) + gamma * Phi^(-1)(1 - 1/(n*e))
    # 简化使用 Euler-Mascheroni 近似
    gamma = 0.5772156649
    p1 = 1 - 1.0 / n_trials
    p2 = 1 - 1.0 / (n_trials * math.e)
    if not (0 < p1 < 1) or not (0 < p2 < 1):
        return 0.0
    z1 = sp_stats.norm.ppf(p1)
    z2 = sp_stats.norm.ppf(p2)
    return float((1 - gamma) * z1 + gamma * z2) * np.sqrt(1.0 / n_obs)


# ============================================================================
# 4. 回撤细节
# ============================================================================

def drawdown_series(equity: pd.Series) -> pd.Series:
    if len(equity) < 2:
        return pd.Series(dtype=float)
    return equity / equity.cummax() - 1


def drawdown_summary(equity: pd.Series) -> Dict[str, float]:
    """回撤事件统计: 总数 / 平均深度 / 最长持续 / 最长修复"""
    if len(equity) < 2:
        return {"dd_count": 0, "avg_depth": 0.0, "max_duration": 0,
                "max_recovery": 0, "max_depth": 0.0}

    dd = drawdown_series(equity)
    underwater = dd < 0
    # 找回撤事件
    events = []
    in_event = False
    start = 0
    for i, v in enumerate(underunder := underwater.values if hasattr(underwater, 'values') else underwater):
        if v and not in_event:
            in_event = True
            start = i
        elif not v and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(equity) - 1))

    if not events:
        return {"dd_count": 0, "avg_depth": 0.0, "max_duration": 0,
                "max_recovery": 0, "max_depth": 0.0}

    depths = [dd.iloc[s:e + 1].min() for s, e in events]
    durations = [e - s + 1 for s, e in events]
    # 修复期: 回到前高的天数
    recoveries = []
    for s, e in events:
        peak = equity.iloc[:s + 1].max()
        post = equity.iloc[e + 1:]
        rec_idx = post[post >= peak].index
        if len(rec_idx) > 0:
            rec_pos = equity.index.get_loc(rec_idx[0])
            recoveries.append(rec_pos - e)
        else:
            recoveries.append(len(equity) - e)  # 未修复

    return {
        "dd_count": len(events),
        "avg_depth": float(np.mean(depths)),
        "max_depth": float(min(depths)),
        "max_duration": int(max(durations)),
        "max_recovery": int(max(recoveries)),
    }


# ============================================================================
# 5. 换手 / 行为指标
# ============================================================================

def turnover_stats(weights: pd.DataFrame) -> Dict[str, float]:
    """
    权重矩阵: index=date, columns=code, value=weight
    计算每日换手率与年化换手率
    """
    if weights.empty or len(weights) < 2:
        return {"avg_daily_turnover": 0.0, "annual_turnover": 0.0}

    # 填充 NaN 为 0 (不进持仓视为 0 权重)
    w = weights.fillna(0.0)
    # 换手率 = sum(|w_t - w_{t-1}|) / 2
    delta = w.diff().iloc[1:].abs().sum(axis=1) / 2
    return {
        "avg_daily_turnover": float(delta.mean()),
        "annual_turnover": float(delta.mean() * TRADING_DAYS),
    }


def trade_stats(trades: pd.DataFrame) -> Dict[str, float]:
    """从成交记录推导行为指标"""
    if trades.empty:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_holding_days": 0.0, "buy_count": 0, "sell_count": 0}
    pnl = trades.get("pnl")
    if pnl is not None:
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
        profit_factor = (
            wins.sum() / abs(losses.sum())
            if len(losses) > 0 and losses.sum() != 0
            else float("inf") if len(wins) > 0 else 0.0
        )
    else:
        win_rate = 0.0
        profit_factor = 0.0

    actions = trades["action"].astype(str).str.lower() if "action" in trades.columns else pd.Series(dtype=str)
    buy_count = int((actions == "buy").sum())
    sell_count = int((actions == "sell").sum())

    avg_holding = 0.0
    if "action" in trades.columns and "date" in trades.columns and "code" in trades.columns:
        try:
            t = trades.copy()
            t["date"] = pd.to_datetime(t["date"])
            t = t.sort_values(["code", "date"])
            buy_map: Dict[str, pd.Timestamp] = {}
            hold_days = []
            for _, row in t.iterrows():
                if str(row["action"]).lower() == "buy":
                    buy_map[row["code"]] = row["date"]
                elif str(row["action"]).lower() == "sell" and row["code"] in buy_map:
                    hold_days.append((row["date"] - buy_map.pop(row["code"])).days)
            if hold_days:
                avg_holding = float(np.mean(hold_days))
        except Exception as e:
            logger.debug("compute avg_holding_days failed: %s", e)

    return {
        "total_trades": int(len(trades)),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_holding_days": float(avg_holding),
    }


# ============================================================================
# 6. 一次性计算全套指标
# ============================================================================

@dataclass
class MetricReport:
    """所有绩效指标的结构化输出"""
    metrics: Dict[str, float] = field(default_factory=dict)
    drawdown: Dict[str, float] = field(default_factory=dict)
    trades: Dict[str, float] = field(default_factory=dict)
    turnover: Dict[str, float] = field(default_factory=dict)
    benchmark: Optional[Dict[str, float]] = None  # alpha/beta/IR/capture
    n_trials: int = 1  # 用于 deflated_sharpe

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def full_report(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    weights: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    n_trials: int = 1,
) -> MetricReport:
    """一次性输出完整绩效报告"""
    equity = equity.dropna()
    if len(equity) < 2:
        return MetricReport()

    returns = equity.pct_change().dropna()
    metrics: Dict[str, float] = {
        "total_return": total_return(equity),
        "annual_return": annualized_return(equity),
        "annual_vol": annualized_vol(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free),
        "sortino_ratio": sortino_ratio(returns, risk_free),
        "calmar_ratio": calmar_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "skewness": float(returns.skew()) if len(returns) > 2 else 0.0,
        "kurtosis": float(returns.kurt()) if len(returns) > 3 else 0.0,
        "best_day": float(returns.max()) if len(returns) > 0 else 0.0,
        "worst_day": float(returns.min()) if len(returns) > 0 else 0.0,
    }
    var95, cvar95 = var_cvar(returns, 0.95)
    metrics["var_95"] = var95
    metrics["cvar_95"] = cvar95
    metrics["n_obs"] = int(len(returns))

    # Deflated Sharpe
    metrics["deflated_sharpe"] = deflated_sharpe(
        metrics["sharpe_ratio"], n_trials, len(returns),
        skew=metrics["skewness"], kurt=metrics["kurtosis"]
    )

    report = MetricReport(
        metrics=metrics,
        drawdown=drawdown_summary(equity),
        trades=trade_stats(trades) if trades is not None else {},
        turnover=turnover_stats(weights) if weights is not None else {},
        n_trials=n_trials,
    )

    if benchmark_returns is not None and len(benchmark_returns) > 1:
        alpha, beta = alpha_beta(returns, benchmark_returns, risk_free)
        ir = information_ratio(returns, benchmark_returns)
        up_cap, dn_cap = up_down_capture(returns, benchmark_returns)
        report.benchmark = {
            "alpha_annual": alpha,
            "beta": beta,
            "information_ratio": ir,
            "up_capture": up_cap,
            "down_capture": dn_cap,
        }

    return report


# ============================================================================
# 7. CLI
# ============================================================================

def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="metrics 自检")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        # 简单合成数据
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.0005, 0.015, 500))
        eq = (1 + rets).cumprod() * 1e6
        r = full_report(eq)
        print(json.dumps(r.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    _cli()