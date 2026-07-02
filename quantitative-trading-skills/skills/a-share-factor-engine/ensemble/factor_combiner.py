import pandas as pd
import numpy as np
from typing import Dict, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class FactorCombiner:
    """
    因子融合器
    """

    def __init__(self, config: Config):
        self.config = config

    def combine_equal_weight(
        self,
        factors: pd.DataFrame
    ) -> pd.Series:
        """
        等权融合
        
        Args:
            factors: 因子 DataFrame
        
        Returns:
            融合后的 Alpha 序列
        """
        factor_cols = [col for col in factors.columns if col not in ["code", "date"]]
        return factors[factor_cols].mean(axis=1)

    def combine_ic_weighted(
        self,
        factors: pd.DataFrame,
        ic_mean: Dict[str, float]
    ) -> pd.Series:
        """
        IC加权融合
        
        Args:
            factors: 因子 DataFrame
            ic_mean: IC均值字典
        
        Returns:
            融合后的 Alpha 序列
        """
        factor_cols = [col for col in factors.columns if col not in ["code", "date"]]
        weights = pd.Series(1.0, index=factor_cols)
        
        for factor in factor_cols:
            if factor in ic_mean:
                weights[factor] = max(ic_mean[factor], 0)
        
        total_weight = weights.sum()
        if total_weight > 0:
            weights = weights / total_weight
        
        return factors[factor_cols].mul(weights).sum(axis=1)

    def combine_risk_weighted(
        self,
        factors: pd.DataFrame
    ) -> pd.Series:
        """
        风险平价融合
        
        Args:
            factors: 因子 DataFrame
        
        Returns:
            融合后的 Alpha 序列
        """
        factor_cols = [col for col in factors.columns if col not in ["code", "date"]]
        factor_data = factors[factor_cols]
        
        volatility = factor_data.std()
        weights = 1 / volatility
        weights = weights / weights.sum()
        
        return factor_data.mul(weights).sum(axis=1)

    def combine_factors(
        self,
        factors: pd.DataFrame,
        method: Optional[str] = None,
        ic_mean: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        因子融合
        
        Args:
            factors: 因子 DataFrame
            method: 融合方法，可选 "equal", "ic_weighted", "risk_weighted"
            ic_mean: IC均值字典（IC加权需要）
        
        Returns:
            包含融合因子的 DataFrame
        """
        if method is None:
            method = self.config.ENSEMBLE_METHOD
        
        result = factors[["code", "date"]].copy()
        
        if method == "equal":
            result["combined_alpha"] = self.combine_equal_weight(factors)
        elif method == "ic_weighted":
            if ic_mean is None:
                raise ValueError("ic_mean is required for ic_weighted method")
            result["combined_alpha"] = self.combine_ic_weighted(factors, ic_mean)
        elif method == "risk_weighted":
            result["combined_alpha"] = self.combine_risk_weighted(factors)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return result
