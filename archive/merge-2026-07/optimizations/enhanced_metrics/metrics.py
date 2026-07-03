"""
增强版绩效指标计算（借鉴 VectorBT 50+ 指标体系）

核心优化点：
1. 从 ~7 个指标扩展到 30+ 指标
2. 新增下行风险类：Sortino, Omega, VaR, CVaR, Calmar
3. 新增相对基准类：Alpha, Beta, Information Ratio, Up/Down Capture
4. 新增分布类：偏度、峰度、最大连续亏损天数
5. 新增交易类：盈亏比、平均持仓天数、换手率

借鉴来源：VectorBT (https://vectorbt.dev)
  - Portfolio.stats() 的 50+ 指标体系
  - 年化收益、波动、夏普、索提诺、卡尔玛的标准化计算
  - Value at Risk (VaR) 与 Conditional VaR (CVaR)

对照 jingni-trader 现有 base_backtest.py 的改进：
  - 现有：total_return, annual_return, volatility, sharpe, max_drawdown,
          calmar, sortino, win_rate, total_trades (9 个)
  - 优化：30+ 指标，覆盖下行风险、相对基准、分布特征、交易质量
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


class EnhancedMetrics:
    """增强版绩效指标计算器"""

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        trading_days: int = 252,
        var_confidence: float = 0.95,
    ):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.var_confidence = var_confidence

    def calc_all(
        self,
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        一次性计算全部绩效指标

        参数:
            equity_curve: 净值序列（绝对值，非收益率）
            trades: 成交记录，含 date, code, action, pnl, amount 等列
            benchmark: 基准净值序列（与 equity_curve 对齐）

        返回:
            包含 30+ 指标的字典
        """
        if equity_curve is None or len(equity_curve) < 2:
            return {}

        eq = pd.Series(equity_curve).dropna()
        returns = eq.pct_change().dropna()
        if len(returns) < 2:
            return {}

        metrics: Dict[str, Any] = {}

        # ---- 1. 收益类指标 ----
        metrics.update(self._return_metrics(eq, returns))

        # ---- 2. 风险类指标 ----
        metrics.update(self._risk_metrics(returns))

        # ---- 3. 风险调整收益指标 ----
        metrics.update(self._risk_adjusted_metrics(eq, returns))

        # ---- 4. 回撤类指标 ----
        metrics.update(self._drawdown_metrics(eq))

        # ---- 5. 分布类指标 ----
        metrics.update(self._distribution_metrics(returns))

        # ---- 6. 相对基准类指标 ----
        if benchmark is not None and len(benchmark) > 1:
            bench_returns = pd.Series(benchmark).pct_change().dropna()
            # 对齐
            common = returns.index.intersection(bench_returns.index)
            if len(common) > 10:
                metrics.update(
                    self._benchmark_metrics(
                        returns.loc[common], bench_returns.loc[common]
                    )
                )

        # ---- 7. 交易类指标 ----
        if trades is not None and not trades.empty:
            metrics.update(self._trade_metrics(trades))

        return metrics

    # ---------- 收益类 ----------

    def _return_metrics(self, eq: pd.Series, returns: pd.Series) -> Dict[str, float]:
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        n_years = len(returns) / self.trading_days
        annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        daily_mean = float(returns.mean())
        # 月度收益
        try:
            monthly = eq.resample("ME").last().pct_change().dropna()
            best_month = float(monthly.max()) if len(monthly) > 0 else 0.0
            worst_month = float(monthly.min()) if len(monthly) > 0 else 0.0
            positive_months = float((monthly > 0).mean()) if len(monthly) > 0 else 0.0
        except Exception:
            best_month = worst_month = positive_months = 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_mean_return": daily_mean,
            "best_month": best_month,
            "worst_month": worst_month,
            "positive_month_ratio": positive_months,
        }

    # ---------- 风险类 ----------

    def _risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        volatility = float(returns.std() * np.sqrt(self.trading_days))
        downside = returns[returns < 0]
        downside_std = float(downside.std() * np.sqrt(self.trading_days)) if len(downside) > 1 else 0.0
        # VaR (历史模拟法)
        var_pct = 1 - self.var_confidence
        var = float(returns.quantile(var_pct))
        # CVaR / Expected Shortfall
        cvar = float(returns[returns <= var].mean()) if (returns <= var).any() else var
        # 最大单日跌幅
        max_daily_loss = float(returns.min())

        return {
            "volatility": volatility,
            "downside_volatility": downside_std,
            "var_95": var,
            "cvar_95": cvar,
            "max_daily_loss": max_daily_loss,
        }

    # ---------- 风险调整收益 ----------

    def _risk_adjusted_metrics(self, eq: pd.Series, returns: pd.Series) -> Dict[str, float]:
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        n_years = len(returns) / self.trading_days
        annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        volatility = float(returns.std() * np.sqrt(self.trading_days))
        downside = returns[returns < 0]
        downside_std = float(downside.std() * np.sqrt(self.trading_days)) if len(downside) > 1 else 0.0
        max_dd = float((eq / eq.cummax() - 1).min())

        sharpe = float((annual_return - self.risk_free_rate) / volatility) if volatility > 0 else 0.0
        sortino = float((annual_return - self.risk_free_rate) / downside_std) if downside_std > 0 else 0.0
        calmar = float(annual_return / abs(max_dd)) if max_dd != 0 else 0.0

        # Omega 比率（以 0 为阈值）
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        omega = float(gains / losses) if losses > 0 else float("inf")

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "omega_ratio": omega,
        }

    # ---------- 回撤类 ----------

    def _drawdown_metrics(self, eq: pd.Series) -> Dict[str, float]:
        cummax = eq.cummax()
        drawdown = (eq - cummax) / cummax
        max_dd = float(drawdown.min())

        # 最大回撤持续天数
        in_dd = drawdown < 0
        max_dd_duration = 0
        current = 0
        for v in in_dd:
            if v:
                current += 1
                max_dd_duration = max(max_dd_duration, current)
            else:
                current = 0

        # 回撤恢复天数（从最大回撤低点到回到前高）
        dd_min_idx = drawdown.idxmin()
        try:
            recovery_mask = (eq.index > dd_min_idx) & (eq >= cummax.loc[dd_min_idx])
            if recovery_mask.any():
                recovery_idx = eq.index[recovery_mask][0]
                recovery_days = int(eq.index.get_loc(recovery_idx) - eq.index.get_loc(dd_min_idx))
            else:
                recovery_days = -1  # 未恢复
        except Exception:
            recovery_days = -1

        return {
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": int(max_dd_duration),
            "drawdown_recovery_days": int(recovery_days),
        }

    # ---------- 分布类 ----------

    def _distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        skewness = float(scipy_stats.skew(returns))
        kurtosis = float(scipy_stats.kurtosis(returns))
        # Jarque-Bera 正态性检验
        try:
            jb_stat, jb_p = scipy_stats.jarque_bera(returns)
            jb_stat = float(jb_stat)
            jb_p = float(jb_p)
        except Exception:
            jb_stat = jb_p = 0.0

        # 最大连续亏损天数
        max_consec_loss = 0
        current = 0
        for r in returns:
            if r < 0:
                current += 1
                max_consec_loss = max(max_consec_loss, current)
            else:
                current = 0

        # 最大连续盈利天数
        max_consec_win = 0
        current = 0
        for r in returns:
            if r > 0:
                current += 1
                max_consec_win = max(max_consec_win, current)
            else:
                current = 0

        return {
            "return_skewness": skewness,
            "return_kurtosis": kurtosis,
            "jarque_bera_stat": jb_stat,
            "jarque_bera_pvalue": jb_p,
            "max_consecutive_loss_days": int(max_consec_loss),
            "max_consecutive_win_days": int(max_consec_win),
        }

    # ---------- 相对基准类 ----------

    def _benchmark_metrics(
        self, returns: pd.Series, bench_returns: pd.Series
    ) -> Dict[str, float]:
        # Beta, Alpha (CAPM)
        cov = float(np.cov(returns, bench_returns)[0, 1])
        var_b = float(bench_returns.var())
        beta = float(cov / var_b) if var_b > 0 else 0.0
        alpha = float(returns.mean() - beta * bench_returns.mean()) * self.trading_days

        # Information Ratio
        excess = returns - bench_returns
        tracking_error = float(excess.std() * np.sqrt(self.trading_days))
        ir = float(excess.mean() * self.trading_days / tracking_error) if tracking_error > 0 else 0.0

        # Up/Down Capture Ratio
        up_market = bench_returns > 0
        down_market = bench_returns < 0
        up_capture = (
            float(returns[up_market].mean() / bench_returns[up_market].mean())
            if up_market.any() and bench_returns[up_market].mean() != 0 else 0.0
        )
        down_capture = (
            float(returns[down_market].mean() / bench_returns[down_market].mean())
            if down_market.any() and bench_returns[down_market].mean() != 0 else 0.0
        )

        # 超额收益
        excess_return = float((1 + returns).prod() - (1 + bench_returns).prod())

        return {
            "alpha": alpha,
            "beta": beta,
            "information_ratio": ir,
            "tracking_error": tracking_error,
            "up_capture": up_capture,
            "down_capture": down_capture,
            "excess_return": excess_return,
        }

    # ---------- 交易类 ----------

    def _trade_metrics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        result: Dict[str, Any] = {"total_trades": int(len(trades))}

        if "action" in trades.columns and "pnl" in trades.columns:
            sell_trades = trades[trades["action"] == "sell"]
            buy_trades = trades[trades["action"] == "buy"]
            result["n_sells"] = int(len(sell_trades))
            result["n_buys"] = int(len(buy_trades))

            if not sell_trades.empty:
                wins = sell_trades[sell_trades["pnl"] > 0]
                losses = sell_trades[sell_trades["pnl"] <= 0]
                result["win_rate"] = float(len(wins) / len(sell_trades))
                result["avg_win"] = float(wins["pnl"].mean()) if not wins.empty else 0.0
                result["avg_loss"] = float(losses["pnl"].mean()) if not losses.empty else 0.0
                result["profit_factor"] = (
                    float(wins["pnl"].sum() / abs(losses["pnl"].sum()))
                    if not losses.empty and losses["pnl"].sum() != 0
                    else float("inf")
                )
                result["payoff_ratio"] = (
                    float(abs(wins["pnl"].mean() / losses["pnl"].mean()))
                    if not wins.empty and not losses.empty and losses["pnl"].mean() != 0
                    else 0.0
                )
                result["max_win"] = float(sell_trades["pnl"].max())
                result["max_loss"] = float(sell_trades["pnl"].min())
                result["avg_pnl_per_trade"] = float(sell_trades["pnl"].mean())

        if "amount" in trades.columns and not trades.empty:
            result["total_turnover"] = float(trades["amount"].sum())

        return result
