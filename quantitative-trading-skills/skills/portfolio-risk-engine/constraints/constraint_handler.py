import numpy as np
import pandas as pd
from typing import Dict, Optional


class ConstraintHandler:
    """
    A 股约束处理器
    """

    def __init__(
        self,
        max_single_weight: float = 0.10,
        min_single_weight: float = 0.0,
        max_industry_deviation: float = 0.05
    ):
        self.max_single_weight = max_single_weight
        self.min_single_weight = min_single_weight
        self.max_industry_deviation = max_industry_deviation

    def check_single_stock_constraint(self, weights: Dict[str, float]) -> Dict:
        """
        检查单股票约束
        
        Args:
            weights: 权重字典
        
        Returns:
            检查结果字典
        """
        violations = []
        for stock, weight in weights.items():
            if weight > self.max_single_weight + 1e-8:
                violations.append({
                    "stock": stock,
                    "weight": weight,
                    "max": self.max_single_weight,
                    "type": "overweight"
                })
            elif weight < self.min_single_weight - 1e-8:
                violations.append({
                    "stock": stock,
                    "weight": weight,
                    "min": self.min_single_weight,
                    "type": "underweight"
                })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations
        }

    def check_industry_constraint(
        self,
        weights: Dict[str, float],
        industry_data: pd.Series,
        benchmark_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        检查行业约束
        
        Args:
            weights: 权重字典
            industry_data: 行业数据 Series
            benchmark_weights: 基准权重
        
        Returns:
            检查结果字典
        """
        weight_series = pd.Series(weights)
        
        portfolio_industry = pd.DataFrame({
            "weight": weight_series,
            "industry": industry_data
        }).groupby("industry")["weight"].sum()
        
        violations = []
        
        if benchmark_weights is not None:
            benchmark_series = pd.Series(benchmark_weights)
            benchmark_industry = pd.DataFrame({
                "weight": benchmark_series,
                "industry": industry_data
            }).groupby("industry")["weight"].sum()
            
            industry_deviation = portfolio_industry.subtract(benchmark_industry, fill_value=0)
            
            for industry, dev in industry_deviation.items():
                if abs(dev) > self.max_industry_deviation + 1e-8:
                    violations.append({
                        "industry": industry,
                        "deviation": dev,
                        "max_deviation": self.max_industry_deviation,
                        "portfolio_weight": portfolio_industry.get(industry, 0),
                        "benchmark_weight": benchmark_industry.get(industry, 0)
                    })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "portfolio_industry": portfolio_industry.to_dict()
        }

    def apply_single_stock_constraint(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        应用单股票约束（截断并重新归一化）
        
        Args:
            weights: 权重字典
        
        Returns:
            调整后的权重字典
        """
        # 截断
        adjusted_weights = {}
        for stock, weight in weights.items():
            adjusted = max(self.min_single_weight, min(weight, self.max_single_weight))
            adjusted_weights[stock] = adjusted
        
        # 重新归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            for stock in adjusted_weights:
                adjusted_weights[stock] /= total
        
        return adjusted_weights

    def check_all_constraints(
        self,
        weights: Dict[str, float],
        industry_data: Optional[pd.Series] = None,
        benchmark_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        检查所有约束
        
        Args:
            weights: 权重字典
            industry_data: 行业数据 Series
            benchmark_weights: 基准权重
        
        Returns:
            检查结果字典
        """
        single_stock_result = self.check_single_stock_constraint(weights)
        
        industry_result = None
        if industry_data is not None:
            industry_result = self.check_industry_constraint(weights, industry_data, benchmark_weights)
        
        all_valid = single_stock_result["valid"]
        if industry_result is not None:
            all_valid = all_valid and industry_result["valid"]
        
        return {
            "all_valid": all_valid,
            "single_stock": single_stock_result,
            "industry": industry_result
        }
