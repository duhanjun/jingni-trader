"""
扩展绩效指标计算

借鉴来源：
- VectorBT 的 stats() 全面指标体系
- Qlib 的 risk analysis 模块
- 业界常用的 Information Ratio / Profit Factor / 盈亏比

在 main 分支 BaseBacktestMetrics 基础上补充：
- 信息比率 (Information Ratio，相对基准)
- 利润因子 (Profit Factor)
- 平均盈亏比 (Payoff Ratio)
- 最大连续亏损天数
- 年化下行波动率
- 基准相对指标 (Alpha / Beta / 跟踪误差)
- 单笔最大盈利 / 最大亏损
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class ExtendedMetrics:
    """扩展绩效指标计算器"""

    TRADING_DAYS = 252

    @staticmethod
    def calc_profit_factor(returns: pd.Series) -> float:
        """利润因子 = 总盈利 / 总亏损绝对值"""
        if len(returns) == 0:
            return 0.0
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    @staticmethod
    def calc_payoff_ratio(returns: pd.Series) -> float:
        """平均盈亏比 = 平均盈利 / 平均亏损绝对值"""
        if len(returns) == 0:
            return 0.0
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        avg_win = wins.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 0.0
        return float(avg_win / abs(avg_loss))

    @staticmethod
    def calc_max_consecutive_loss_days(returns: pd.Series) -> int:
        """最大连续亏损天数"""
        if len(returns) == 0:
            return 0
        is_loss = (returns < 0).astype(int)
        max_streak = 0
        cur = 0
        for v in is_loss:
            if v:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        return int(max_streak)

    @staticmethod
    def calc_downside_volatility(returns: pd.Series, trading_days: int = 252) -> float:
        """年化下行波动率（仅用负收益）"""
        neg = returns[returns < 0]
        if len(neg) < 2:
            return 0.0
        return float(neg.std() * np.sqrt(trading_days))

    @staticmethod
    def calc_alpha_beta(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free: float = 0.03,
        trading_days: int = 252,
    ) -> Dict[str, float]:
        """计算 CAPM Alpha / Beta"""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return {"alpha": 0.0, "beta": 0.0}
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
        if len(aligned) < 2:
            return {"alpha": 0.0, "beta": 0.0}
        r = aligned.iloc[:, 0]
        b = aligned.iloc[:, 1]
        var_b = b.var()
        if var_b == 0:
            return {"alpha": 0.0, "beta": 0.0}
        beta = float(r.cov(b) / var_b)
        # 年化 alpha
        rf_daily = (1 + risk_free) ** (1 / trading_days) - 1
        alpha = float((r.mean() - rf_daily) - beta * (b.mean() - rf_daily)) * trading_days
        return {"alpha": alpha, "beta": beta}

    @staticmethod
    def calc_information_ratio(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        trading_days: int = 252,
    ) -> float:
        """信息比率 = 超额收益年化 / 跟踪误差年化"""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
        if len(aligned) < 2:
            return 0.0
        excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        te = excess.std()
        if te == 0:
            return 0.0
        return float(excess.mean() / te * np.sqrt(trading_days))

    @staticmethod
    def calc_tracking_error(
        returns: pd.Series, benchmark_returns: pd.Series, trading_days: int = 252
    ) -> float:
        """年化跟踪误差"""
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
        if len(aligned) < 2:
            return 0.0
        excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        return float(excess.std() * np.sqrt(trading_days))

    @staticmethod
    def calc_max_single_trade_pnl(trades: pd.DataFrame) -> Dict[str, float]:
        """单笔最大盈利 / 最大亏损"""
        if trades.empty or "pnl" not in trades.columns:
            return {"max_win": 0.0, "max_loss": 0.0}
        pnl = trades["pnl"]
        return {
            "max_win": float(pnl.max()) if len(pnl) else 0.0,
            "max_loss": float(pnl.min()) if len(pnl) else 0.0,
        }

    @staticmethod
    def calc_all_extended_metrics(
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free: float = 0.03,
        trading_days: int = 252,
    ) -> Dict[str, Any]:
        """
        计算全部扩展指标

        参数:
            equity_curve: 净值序列（index=date）
            trades: 成交记录（含 pnl 列）
            benchmark_returns: 基准日收益率序列（可选）
            risk_free: 无风险利率
            trading_days: 年交易日
        """
        if len(equity_curve) < 2:
            return {}
        returns = equity_curve.pct_change().dropna()

        metrics: Dict[str, Any] = {
            "profit_factor": ExtendedMetrics.calc_profit_factor(returns),
            "payoff_ratio": ExtendedMetrics.calc_payoff_ratio(returns),
            "max_consecutive_loss_days": ExtendedMetrics.calc_max_consecutive_loss_days(returns),
            "downside_volatility": ExtendedMetrics.calc_downside_volatility(returns, trading_days),
        }

        # 单笔交易统计
        metrics.update(ExtendedMetrics.calc_max_single_trade_pnl(trades))

        # 基准相对指标
        if benchmark_returns is not None and len(benchmark_returns) >= 2:
            metrics["information_ratio"] = ExtendedMetrics.calc_information_ratio(
                returns, benchmark_returns, trading_days
            )
            metrics["tracking_error"] = ExtendedMetrics.calc_tracking_error(
                returns, benchmark_returns, trading_days
            )
            ab = ExtendedMetrics.calc_alpha_beta(
                returns, benchmark_returns, risk_free, trading_days
            )
            metrics["alpha"] = ab["alpha"]
            metrics["beta"] = ab["beta"]

        return metrics
