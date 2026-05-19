"""
组合优化与风控引擎主逻辑
负责权重优化、协方差估计、风险归因、VaR/CVaR、止损机制
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from .config import (
    PORTFOLIO_DIR, OPTIMIZATION_METHOD, RISK_FREE_RATE,
    ESTIMATION_PERIOD, MAX_SINGLE_STOCK_WEIGHT, MAX_INDUSTRY_DEVIATION,
    MAX_TURNOVER, COVARIANCE_METHOD, EXPECTED_RETURNS_METHOD,
    MAX_DAILY_LOSS_RATIO, INDIVIDUAL_STOP_LOSS,
    VAR_CONFIDENCE, CVAR_CONFIDENCE, BARRA_FACTORS, MIN_WEIGHT, BACKEND
)

logger = logging.getLogger("portfolio-risk-engine")

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt, CLA
    from pypfopt import plotting as pfopt_plot
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False

try:
    import riskfolio as rl
    HAS_RISKFOLIO = True
except ImportError:
    HAS_RISKFOLIO = False


class PortfolioOptimizer:
    """组合优化器，基于 PyPortfolioOpt"""

    def __init__(self):
        if not HAS_PYPFOPT:
            raise ImportError("PyPortfolioOpt 未安装，请 pip install PyPortfolioOpt")

    def estimate_expected_returns(
        self,
        returns: pd.DataFrame,
        method: str = EXPECTED_RETURNS_METHOD
    ) -> pd.Series:
        """估计预期收益"""
        if method == "ema_historical":
            return expected_returns.ema_historical_return(returns, frequency=252)
        elif method == "mean_historical":
            return expected_returns.mean_historical_return(returns, frequency=252)
        elif method == "capm_return":
            return expected_returns.capm_return(returns)
        else:
            return expected_returns.mean_historical_return(returns, frequency=252)

    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        method: str = COVARIANCE_METHOD
    ) -> pd.DataFrame:
        """估计协方差矩阵"""
        if method == "ledoit_wolf":
            return risk_models.CovarianceShrinkage(returns).ledoit_wolf()
        elif method == "sample_cov":
            return risk_models.sample_cov(returns)
        elif method == "shrinkage":
            return risk_models.CovarianceShrinkage(returns).ledoit_wolf(shrinkage_target="constant_correlation")
        else:
            return risk_models.CovarianceShrinkage(returns).ledoit_wolf()

    def optimize(
        self,
        expected_rets: pd.Series,
        cov_matrix: pd.DataFrame,
        method: str = OPTIMIZATION_METHOD,
        constraints: Optional[Dict[str, float]] = None,
        current_weights: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        执行组合优化

        参数:
            expected_rets: 预期收益 Series
            cov_matrix: 协方差矩阵 DataFrame
            method: 优化方法
            constraints: 约束字典
            current_weights: 当前权重

        返回:
            (weights, metrics)
        """
        if constraints is None:
            constraints = {}

        max_weight = constraints.get("max_weight", MAX_SINGLE_STOCK_WEIGHT)
        min_weight = constraints.get("min_weight", MIN_WEIGHT)

        if method == "risk_parity":
            return self._optimize_hrp(expected_rets, cov_matrix)

        ef = EfficientFrontier(expected_rets, cov_matrix, weight_bounds=(min_weight, max_weight))

        if current_weights is not None and len(current_weights) > 0:
            try:
                turnover = constraints.get("max_turnover", MAX_TURNOVER)
                ef.add_objective(
                    lambda w: np.sum(np.abs(w - current_weights.reindex(w.index, fill_value=0)))
                )
            except Exception as e:
                logger.warning(f"添加换手率约束失败: {e}")

        if method == "max_sharpe":
            weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        elif method == "min_variance":
            weights = ef.min_volatility()
        elif method == "max_return":
            weights = ef.max_quadratic_utility()
        elif method == "black_litterman":
            weights = self._black_litterman(expected_rets, cov_matrix, constraints)
            return weights, {}
        elif method == "cvar":
            return self._optimize_cvar(expected_rets, cov_matrix, constraints)
        else:
            weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)

        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        return pd.Series(cleaned), {
            "expected_return": float(perf[0]),
            "volatility": float(perf[1]),
            "sharpe_ratio": float(perf[2]),
        }

    def _optimize_hrp(
        self,
        expected_rets: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """分层风险平价"""
        returns = pd.DataFrame()
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        cleaned = hrp.clean_weights()
        return pd.Series(cleaned), {"method": "hrp"}

    def _optimize_cvar(
        self,
        expected_rets: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Dict[str, float]
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """CVaR 优化"""
        n = len(expected_rets)
        weights = pd.Series(1.0 / n, index=expected_rets.index)
        logger.warning("CVaR优化需要历史收益数据，当前返回等权组合")
        return weights, {"method": "cvar", "note": "简化实现，使用等权"}

    def _black_litterman(
        self,
        expected_rets: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Dict[str, float]
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """Black-Litterman 模型"""
        from pypfopt import black_litterman
        market_caps = pd.Series(1.0, index=expected_rets.index)
        bl = black_litterman.BlackLittermanModel(
            cov_matrix, pi="market", market_caps=market_caps,
            risk_aversion=2.5
        )
        rets_bl = bl.bl_returns()
        ef = EfficientFrontier(rets_bl, cov_matrix)
        weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        cleaned = ef.clean_weights()
        return pd.Series(cleaned), {"method": "black_litterman"}


class AShareConstraints:
    """A股组合约束管理"""

    def __init__(
        self,
        industry_map: Optional[pd.DataFrame] = None,
        benchmark_weights: Optional[pd.Series] = None,
    ):
        self.industry_map = industry_map
        self.benchmark_weights = benchmark_weights

    def validate_constraints(
        self,
        weights: pd.Series,
        constraints: Dict[str, float]
    ) -> Dict[str, bool]:
        """验证组合是否满足所有约束"""
        result = {}
        max_w = constraints.get("max_weight", MAX_SINGLE_STOCK_WEIGHT)
        result["max_single_weight"] = all(weights <= max_w)
        result["weights_sum_one"] = abs(weights.sum() - 1.0) < 0.001

        if self.industry_map is not None and not self.industry_map.empty:
            result["industry_deviation"] = self._check_industry_deviation(weights, constraints)

        return result

    def _check_industry_deviation(self, weights, constraints) -> bool:
        """检查行业偏离"""
        max_deviation = constraints.get("max_industry_deviation", MAX_INDUSTRY_DEVIATION)
        ind_map = self.industry_map.set_index('code')['industry'].to_dict()

        ind_weights = {}
        for code, w in weights.items():
            ind = ind_map.get(code, "其他")
            ind_weights[ind] = ind_weights.get(ind, 0) + w

        for ind, w in ind_weights.items():
            bench_w = 0
            if self.benchmark_weights is not None and ind in self.benchmark_weights.index:
                bench_w = self.benchmark_weights[ind]
            if abs(w - bench_w) > max_deviation:
                return False
        return True


class RiskManager:
    """风险计算与止损管理"""

    def __init__(self):
        self._daily_pnl = 0.0
        self._start_nav = 0.0

    def reset_daily(self, nav: float):
        """交易日开始时重置日损益"""
        self._daily_pnl = 0.0
        self._start_nav = nav

    def check_portfolio_stop(self, current_nav: float) -> Dict[str, Any]:
        """检查组合层面止损"""
        daily_return = (current_nav - self._start_nav) / self._start_nav if self._start_nav > 0 else 0
        triggered = daily_return <= -MAX_DAILY_LOSS_RATIO
        return {
            "triggered": triggered,
            "daily_return": float(daily_return),
            "threshold": MAX_DAILY_LOSS_RATIO,
            "reason": "单日亏损超过阈值" if triggered else "",
        }

    def check_individual_stop(
        self,
        current_prices: pd.Series,
        entry_prices: pd.Series,
    ) -> pd.Series:
        """检查个股止损信号"""
        returns = (current_prices - entry_prices) / entry_prices
        return returns <= -INDIVIDUAL_STOP_LOSS

    def calc_var(
        self,
        returns: pd.Series,
        confidence: float = VAR_CONFIDENCE
    ) -> float:
        """历史模拟法 VaR"""
        return float(np.percentile(returns, (1 - confidence) * 100))

    def calc_cvar(
        self,
        returns: pd.Series,
        confidence: float = CVAR_CONFIDENCE
    ) -> float:
        """CVaR (Expected Shortfall)"""
        var = np.percentile(returns, (1 - confidence) * 100)
        return float(returns[returns <= var].mean())

    def calc_portfolio_var_cvar(
        self,
        returns_df: pd.DataFrame,
        weights: pd.Series,
        confidence: float = VAR_CONFIDENCE
    ) -> Dict[str, float]:
        """计算组合的 VaR 和 CVaR"""
        aligned_weights = weights.reindex(returns_df.columns, fill_value=0)
        portfolio_returns = returns_df.dot(aligned_weights)
        return {
            "VaR": self.calc_var(portfolio_returns, confidence),
            "CVaR": self.calc_cvar(portfolio_returns, confidence),
            "confidence": confidence,
        }

    def barra_style_attribution(
        self,
        returns_df: pd.DataFrame,
        factor_data: pd.DataFrame,
        weights: pd.Series,
    ) -> Dict[str, float]:
        """
        简化版 Barra CNE5 风格因子归因

        参数:
            returns_df: 股票日收益 DataFrame
            factor_data: 因子暴露 DataFrame (index=date, columns 为各因子列)
            weights: 当前权重

        返回:
            各风格因子的暴露度
        """
        style_exposures = {}
        for style in BARRA_FACTORS.split(","):
            style = style.strip()
            exposures = []
            for code in weights.index:
                if code in factor_data.columns and style in factor_data.columns.levels if hasattr(factor_data.columns, 'levels') else False:
                    continue
            style_exposures[style] = 0.0

        logger.info("Barra归因需要完整的因子暴露数据，当前返回空结果")
        return style_exposures

    def generate_stop_loss_signals(
        self,
        data: pd.DataFrame,
        portfolio_weights: pd.Series,
        nav: float,
    ) -> Dict[str, Any]:
        """综合生成所有止损信号"""
        stop_result = self.check_portfolio_stop(nav)

        individual_stops = pd.Series(False, index=portfolio_weights.index)
        if 'close' in data.columns:
            latest_prices = data.groupby('code')['close'].last()
            entry_prices = data.groupby('code')['close'].apply(lambda x: x.iloc[-20] if len(x) >= 20 else x.iloc[0])
            aligned_indices = portfolio_weights.index.intersection(latest_prices.index).intersection(entry_prices.index)
            if len(aligned_indices) > 0:
                individual_stops.loc[aligned_indices] = self.check_individual_stop(
                    latest_prices.loc[aligned_indices],
                    entry_prices.loc[aligned_indices]
                )

        return {
            "portfolio_stop": stop_result,
            "individual_stops": individual_stops.to_dict(),
            "any_triggered": stop_result["triggered"] or individual_stops.any(),
        }


def run(ctx) -> Dict[str, Any]:
    """
    portfolio-risk-engine 的 run 函数

    参数:
        ctx: Context 对象，需包含:
            - artifacts['DATA']: 清洗后数据路径
            - artifacts['FACTOR']: 因子数据（获取 Alpha）
            - artifacts['BACKTEST']: 回测结果（获取收益率矩阵）
            - strategy_params: 可覆写优化参数

    返回:
        {
            "success": bool,
            "artifact_path": str,
            "metadata": {
                "weights": {...},
                "metrics": {...},
                "var_cvar": {...},
                "stop_signals": {...},
                "constraint_check": {...}
            },
            "error": str
        }
    """
    try:
        data_path = ctx.get_artifact("DATA")
        if not data_path or not os.path.exists(data_path):
            return {"success": False, "artifact_path": "", "metadata": {}, "error": "行情数据不存在"}

        data = pd.read_parquet(data_path)

        strategy_params = getattr(ctx, 'strategy_params', {}) or {}
        method = strategy_params.get("optimization_method", OPTIMIZATION_METHOD)

        # 构建收益率矩阵
        pivot_close = data.pivot(index='date', columns='code', values='close')
        returns = pivot_close.pct_change().dropna()

        if returns.empty:
            return {"success": False, "artifact_path": "", "metadata": {}, "error": "收益率数据为空"}

        optimizer = PortfolioOptimizer()

        # 获取预期收益 — 优先使用因子的 Alpha
        alpha_available = False
        factor_path = ctx.get_artifact("FACTOR")
        if factor_path and os.path.exists(factor_path):
            factor_df = pd.read_parquet(factor_path)
            if 'alpha_score' in factor_df.columns:
                latest_alphas = factor_df[factor_df['date'] == factor_df['date'].max()]
                if not latest_alphas.empty:
                    expected_rets = latest_alphas.set_index('code')['alpha_score']
                    expected_rets = expected_rets.reindex(returns.columns).fillna(0)
                    alpha_available = True
                    logger.info("使用因子 Alpha 作为预期收益")

        if not alpha_available:
            expected_rets = optimizer.estimate_expected_returns(returns)

        cov_matrix = optimizer.estimate_covariance(returns)

        valid_codes = expected_rets.index.intersection(cov_matrix.index)
        expected_rets = expected_rets.loc[valid_codes]
        cov_matrix = cov_matrix.loc[valid_codes, valid_codes]

        constraints = {
            "max_weight": strategy_params.get("max_weight", MAX_SINGLE_STOCK_WEIGHT),
            "min_weight": strategy_params.get("min_weight", MIN_WEIGHT),
            "max_turnover": strategy_params.get("max_turnover", MAX_TURNOVER),
            "max_industry_deviation": strategy_params.get("max_industry_deviation", MAX_INDUSTRY_DEVIATION),
        }

        weights, metrics = optimizer.optimize(expected_rets, cov_matrix, method=method, constraints=constraints)

        constraint_checker = AShareConstraints()
        constraint_result = constraint_checker.validate_constraints(weights, constraints)

        risk_mgr = RiskManager()
        var_cvar = {}
        if len(returns.columns) > 1:
            var_cvar = risk_mgr.calc_portfolio_var_cvar(returns, weights)

        nav = 1_000_000
        stop_signals = risk_mgr.generate_stop_loss_signals(data, weights, nav)

        weights_dict = {k: float(v) for k, v in weights.items() if v > 0.0001}
        weights_path = os.path.join(PORTFOLIO_DIR, "portfolio_weights.json")
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(weights_dict, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "artifact_path": weights_path,
            "metadata": {
                "weights": weights_dict,
                "metrics": metrics,
                "var_cvar": var_cvar,
                "stop_signals": stop_signals,
                "constraint_check": constraint_result,
                "optimization_method": method,
                "num_assets": len(weights_dict),
            },
            "error": ""
        }

    except Exception as e:
        logger.exception("组合优化引擎执行失败")
        return {"success": False, "artifact_path": "", "metadata": {}, "error": str(e)}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from quant_trading_master.scripts.context import Context

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            ctx = Context.from_dict(json.load(f))
    else:
        ctx = Context(
            task_id="test_portfolio",
            stock_pool=[],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        ctx.update_artifact("DATA", "./quant_workspace/data/cleaned_data.parquet")
        ctx.update_artifact("FACTOR", "./quant_workspace/factors/factor_data.parquet")

    result = run(ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))