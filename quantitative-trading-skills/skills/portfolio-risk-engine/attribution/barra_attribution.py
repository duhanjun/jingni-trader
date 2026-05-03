import numpy as np
import pandas as pd
from typing import Dict, Optional, List


class BarraAttribution:
    """
    Barra 风格因子归因分析（CNE5 模型简化版）
    """

    def __init__(self):
        # CNE5 风格因子（简化）
        self.style_factors = [
            "size", "beta", "momentum", "volatility",
            "value", "liquidity", "growth", "leverage"
        ]

    def calculate_factor_exposures(
        self,
        factor_data: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """
        计算组合的因子暴露
        
        Args:
            factor_data: 因子暴露 DataFrame，index 为股票代码，columns 为因子名
            weights: 权重字典
        
        Returns:
            组合因子暴露 Series
        """
        weight_series = pd.Series(weights)
        portfolio_exposures = factor_data.mul(weight_series, axis=0).sum()
        return portfolio_exposures

    def calculate_industry_exposures(
        self,
        industry_data: pd.Series,
        weights: Dict[str, float],
        benchmark_weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        计算行业暴露
        
        Args:
            industry_data: 行业数据 Series，index 为股票代码
            weights: 权重字典
            benchmark_weights: 基准权重（可选）
        
        Returns:
            行业暴露 Series
        """
        weight_series = pd.Series(weights)
        
        # 组合行业暴露
        portfolio_industry = pd.DataFrame({
            "weight": weight_series,
            "industry": industry_data
        }).groupby("industry")["weight"].sum()
        
        if benchmark_weights is not None:
            benchmark_series = pd.Series(benchmark_weights)
            benchmark_industry = pd.DataFrame({
                "weight": benchmark_series,
                "industry": industry_data
            }).groupby("industry")["weight"].sum()
            
            # 计算偏离
            industry_deviation = portfolio_industry.subtract(benchmark_industry, fill_value=0)
            return industry_deviation
        
        return portfolio_industry

    def attribute_returns(
        self,
        returns: pd.Series,
        factor_data: pd.DataFrame,
        weights: Dict[str, float],
        factor_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        收益归因分析
        
        Args:
            returns: 收益率 Series
            factor_data: 因子暴露 DataFrame
            weights: 权重字典
            factor_returns: 因子收益率 Series（可选）
        
        Returns:
            归因结果字典
        """
        weight_series = pd.Series(weights)
        
        # 组合总收益
        portfolio_return = (returns * weight_series).sum()
        
        # 因子暴露
        exposures = self.calculate_factor_exposures(factor_data, weights)
        
        result = {
            "total_return": portfolio_return,
            "factor_exposures": exposures.to_dict()
        }
        
        if factor_returns is not None:
            # 因子收益
            factor_contribution = exposures * factor_returns
            result["factor_contribution"] = factor_contribution.to_dict()
            result["specific_return"] = portfolio_return - factor_contribution.sum()
        
        return result
