"""
组合优化器抽象基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class BaseOptimizer(ABC):
    """组合优化器基类"""

    @abstractmethod
    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Dict[str, float],
        method: str = "max_sharpe",
        current_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        执行组合优化

        参数:
            expected_returns: 各资产预期收益 Series (index=code)
            cov_matrix: 协方差矩阵 DataFrame (index=code, columns=code)
            constraints: 约束字典
                - max_weight: 单一股票最大权重
                - min_weight: 单一股票最小权重
                - max_turnover: 最大换手率（若 current_weights 非空）
                - max_industry_deviation: 行业最大偏离
            method: 优化方法
            current_weights: 当前持仓权重（用于换手率约束）

        返回:
            优化后的权重 Series (index=code)
        """
        ...

    @abstractmethod
    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        method: str = "ledoit_wolf"
    ) -> pd.DataFrame:
        """
        协方差矩阵估计

        参数:
            returns: 收益率 DataFrame (index=date, columns=code)
            method: 估计方法 (sample, ledoit_wolf, shrinkage)

        返回:
            协方差矩阵 DataFrame
        """
        ...

    @abstractmethod
    def estimate_expected_returns(
        self,
        prices_or_returns: pd.DataFrame,
        method: str = "ema_historical"
    ) -> pd.Series:
        """
        预期收益估计

        参数:
            prices_or_returns: 价格或收益 DataFrame
            method: 估计方法 (mean_historical, ema_historical, capm)

        返回:
            预期收益 Series (index=code)
        """
        ...