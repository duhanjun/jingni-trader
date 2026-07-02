import numpy as np
import pandas as pd
from pypfopt import expected_returns, EfficientFrontier, risk_models, HRPOpt
from typing import Dict, Optional, Tuple


class PortfolioOptimizer:
    """
    投资组合权重优化器，支持风险平价、最小方差、最大夏普比率
    """

    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    def optimize_min_variance(
        self,
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        最小方差组合优化
        
        Args:
            returns: 收益率 DataFrame
            cov_matrix: 可选的协方差矩阵，如果不提供则自动计算
            max_weight: 单资产最大权重
            min_weight: 单资产最小权重
        
        Returns:
            (权重字典, 性能指标字典)
        """
        if cov_matrix is None:
            cov_matrix = risk_models.sample_cov(returns)
        
        ef = EfficientFrontier(None, cov_matrix, weight_bounds=(min_weight, max_weight))
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        
        return dict(cleaned_weights), {
            "volatility": performance[1],
            "sharpe_ratio": performance[2]
        }

    def optimize_max_sharpe(
        self,
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        最大夏普比率组合优化
        
        Args:
            returns: 收益率 DataFrame
            cov_matrix: 可选的协方差矩阵
            max_weight: 单资产最大权重
            min_weight: 单资产最小权重
        
        Returns:
            (权重字典, 性能指标字典)
        """
        mu = expected_returns.mean_historical_return(returns)
        if cov_matrix is None:
            cov_matrix = risk_models.sample_cov(returns)
        
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(min_weight, max_weight))
        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate, verbose=False)
        
        return dict(cleaned_weights), {
            "expected_return": performance[0],
            "volatility": performance[1],
            "sharpe_ratio": performance[2]
        }

    def optimize_risk_parity(
        self,
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        风险平价组合优化（使用 HRP）
        
        Args:
            returns: 收益率 DataFrame
            cov_matrix: 可选的协方差矩阵
        
        Returns:
            (权重字典, 性能指标字典)
        """
        if cov_matrix is None:
            cov_matrix = risk_models.sample_cov(returns)
        
        hrp = HRPOpt(returns, cov_matrix)
        weights = hrp.optimize()
        cleaned_weights = hrp.clean_weights()
        performance = hrp.portfolio_performance(verbose=False)
        
        return dict(cleaned_weights), {
            "expected_return": performance[0],
            "volatility": performance[1],
            "sharpe_ratio": performance[2]
        }

    def optimize(
        self,
        returns: pd.DataFrame,
        method: str = "max_sharpe",
        cov_matrix: Optional[pd.DataFrame] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        优化投资组合
        
        Args:
            returns: 收益率 DataFrame
            method: 优化方法，"min_variance", "max_sharpe", "risk_parity"
            cov_matrix: 可选的协方差矩阵
            max_weight: 单资产最大权重
            min_weight: 单资产最小权重
        
        Returns:
            (权重字典, 性能指标字典)
        """
        if method == "min_variance":
            return self.optimize_min_variance(returns, cov_matrix, max_weight, min_weight)
        elif method == "max_sharpe":
            return self.optimize_max_sharpe(returns, cov_matrix, max_weight, min_weight)
        elif method == "risk_parity":
            return self.optimize_risk_parity(returns, cov_matrix)
        else:
            raise ValueError(f"Unsupported method: {method}")
