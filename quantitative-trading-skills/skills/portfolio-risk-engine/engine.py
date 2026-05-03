import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict, Optional

from config import Config, get_config
from covariance import CovarianceEstimator
from optimization import PortfolioOptimizer
from attribution import BarraAttribution
from constraints import ConstraintHandler
from stop_loss import StopLossManager
from var import VaRCalculator
from monitoring import RiskMonitor


class PortfolioRiskEngine:
    """
    投资组合风险管理引擎主类
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        
        # 初始化各个模块
        self.covariance_estimator = CovarianceEstimator(
            ewma_span=self.config.EWMA_SPAN
        )
        
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=self.config.RISK_FREE_RATE
        )
        
        self.attributor = BarraAttribution()
        
        self.constraint_handler = ConstraintHandler(
            max_single_weight=self.config.MAX_SINGLE_WEIGHT,
            min_single_weight=self.config.MIN_SINGLE_WEIGHT,
            max_industry_deviation=self.config.MAX_INDUSTRY_DEVIATION
        )
        
        self.stop_loss_manager = StopLossManager(
            daily_drawdown_threshold=self.config.DAILY_DRAWDOWN_THRESHOLD,
            ma_period=self.config.MA_PERIOD,
            volume_ratio_threshold=self.config.VOLUME_RATIO_THRESHOLD
        )
        
        self.var_calculator = VaRCalculator(
            confidence_level=self.config.CONFIDENCE_LEVEL,
            horizon=self.config.VAR_HORIZON
        )
        
        self.risk_monitor = RiskMonitor(
            max_position_limit=self.config.MAX_POSITION_LIMIT,
            margin_ratio=self.config.MARGIN_RATIO,
            profit_warning_level=self.config.PROFIT_WARNING_LEVEL
        )

    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        估计协方差矩阵
        
        Args:
            returns: 收益率 DataFrame
            method: 估计方法（默认使用配置）
        
        Returns:
            协方差矩阵 DataFrame
        """
        method = method or self.config.COVARIANCE_METHOD
        return self.covariance_estimator.estimate(returns, method)

    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        method: Optional[str] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        max_weight: Optional[float] = None,
        min_weight: Optional[float] = None,
        apply_constraints: bool = True
    ) -> Dict:
        """
        优化投资组合
        
        Args:
            returns: 收益率 DataFrame
            method: 优化方法（默认使用配置）
            cov_matrix: 协方差矩阵（可选）
            max_weight: 单资产最大权重
            min_weight: 单资产最小权重
            apply_constraints: 是否应用约束
        
        Returns:
            优化结果字典
        """
        method = method or self.config.OPTIMIZATION_METHOD
        max_weight = max_weight or self.config.MAX_SINGLE_WEIGHT
        min_weight = min_weight or self.config.MIN_SINGLE_WEIGHT
        
        # 优化
        weights, performance = self.optimizer.optimize(
            returns, method, cov_matrix, max_weight, min_weight
        )
        
        # 应用约束
        if apply_constraints:
            weights = self.constraint_handler.apply_single_stock_constraint(weights)
        
        return {
            "weights": weights,
            "performance": performance
        }

    def calculate_var(
        self,
        returns: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        method: Optional[str] = None,
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        计算 VaR
        
        Args:
            returns: 收益率 DataFrame
            weights: 权重字典（可选）
            method: 计算方法（默认使用配置）
            cov_matrix: 协方差矩阵（仅参数法需要）
        
        Returns:
            VaR 结果字典
        """
        method = method or self.config.VAR_METHOD
        return self.var_calculator.calculate(returns, weights, method, cov_matrix)

    def analyze_risk(
        self,
        returns: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        factor_data: Optional[pd.DataFrame] = None,
        industry_data: Optional[pd.Series] = None,
        benchmark_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        全面风险分析
        
        Args:
            returns: 收益率 DataFrame
            weights: 权重字典
            factor_data: 因子暴露 DataFrame
            industry_data: 行业数据 Series
            benchmark_weights: 基准权重
        
        Returns:
            风险分析结果字典
        """
        if weights is None:
            n_assets = returns.shape[1]
            weights = {col: 1.0 / n_assets for col in returns.columns}
        
        # 协方差矩阵
        cov_matrix = self.estimate_covariance(returns)
        
        # VaR
        var_result = self.calculate_var(returns, weights, cov_matrix=cov_matrix)
        
        # 约束检查
        constraint_result = self.constraint_handler.check_all_constraints(
            weights, industry_data, benchmark_weights
        )
        
        # 因子归因（如果有数据）
        attribution_result = None
        if factor_data is not None:
            attribution_result = {
                "factor_exposures": self.attributor.calculate_factor_exposures(
                    factor_data, weights
                ).to_dict()
            }
            if industry_data is not None:
                attribution_result["industry_exposures"] = self.attributor.calculate_industry_exposures(
                    industry_data, weights, benchmark_weights
                ).to_dict()
        
        return {
            "weights": weights,
            "covariance_matrix": cov_matrix,
            "var": var_result,
            "constraints": constraint_result,
            "attribution": attribution_result
        }

    def check_stop_loss(
        self,
        portfolio_returns: pd.Series,
        individual_prices: Optional[Dict[str, pd.Series]] = None,
        individual_volumes: Optional[Dict[str, pd.Series]] = None
    ) -> Dict:
        """
        检查止损
        
        Args:
            portfolio_returns: 组合收益率 Series
            individual_prices: 个股价格字典
            individual_volumes: 个股成交量字典
        
        Returns:
            止损检查结果字典
        """
        return self.stop_loss_manager.check_portfolio_stop_loss(
            portfolio_returns, individual_prices, individual_volumes
        )

    def monitor_risk(
        self,
        positions: Dict[str, float],
        equity: float,
        returns: pd.Series,
        initial_investment: float = 1.0
    ) -> Dict:
        """
        风险监控
        
        Args:
            positions: 持仓市值字典
            equity: 权益
            returns: 收益率 Series
            initial_investment: 初始投资
        
        Returns:
            监控结果字典
        """
        return self.risk_monitor.monitor_all(
            positions, equity, returns, initial_investment
        )
