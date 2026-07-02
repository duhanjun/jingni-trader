import numpy as np
import pandas as pd
from pypfopt import risk_models
from typing import Optional


class CovarianceEstimator:
    """
    协方差矩阵估计器，支持历史协方差和指数加权协方差
    """

    def __init__(self, ewma_span: int = 60):
        self.ewma_span = ewma_span

    def estimate_historical(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        估计历史协方差矩阵
        
        Args:
            returns: 收益率 DataFrame，index 为日期，columns 为股票代码
        
        Returns:
            协方差矩阵 DataFrame
        """
        return risk_models.sample_cov(returns)

    def estimate_ewma(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        估计指数加权移动平均协方差矩阵（EWMA）
        
        Args:
            returns: 收益率 DataFrame，index 为日期，columns 为股票代码
        
        Returns:
            协方差矩阵 DataFrame
        """
        returns_clean = returns.dropna()
        if returns_clean.empty:
            raise ValueError("No valid returns data")
        
        # 计算指数权重
        lambda_decay = 2 / (self.ewma_span + 1)
        weights = np.array([(1 - lambda_decay) ** t for t in range(len(returns_clean))])
        weights = weights / weights.sum()
        weights = weights[::-1]  # 反转，最近的权重最大
        
        # 计算加权协方差
        mean_returns = np.average(returns_clean, weights=weights, axis=0)
        demeaned_returns = returns_clean - mean_returns
        weighted_demeaned = demeaned_returns * np.sqrt(weights)[:, np.newaxis]
        cov_matrix = weighted_demeaned.T @ weighted_demeaned * len(returns_clean)
        
        return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

    def estimate(self, returns: pd.DataFrame, method: str = "historical") -> pd.DataFrame:
        """
        估计协方差矩阵
        
        Args:
            returns: 收益率 DataFrame
            method: 估计方法，"historical" 或 "ewma"
        
        Returns:
            协方差矩阵 DataFrame
        """
        if method == "historical":
            return self.estimate_historical(returns)
        elif method == "ewma":
            return self.estimate_ewma(returns)
        else:
            raise ValueError(f"Unsupported method: {method}")
