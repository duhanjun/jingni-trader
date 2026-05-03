import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class CorrelationAnalyzer:
    """
    相关性分析器
    """

    def __init__(self, config: Config):
        self.config = config

    def compute_correlation_matrix(
        self,
        factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算相关性矩阵
        
        Args:
            factors: 因子 DataFrame
        
        Returns:
            相关性矩阵 DataFrame
        """
        if factors.empty:
            return pd.DataFrame()
        
        factor_cols = [col for col in factors.columns if col not in ["code", "date"]]
        factor_data = factors[factor_cols]
        
        return factor_data.corr()

    def get_redundant_factors(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        识别冗余因子对
        
        Args:
            corr_matrix: 相关性矩阵
            threshold: 相关性阈值
        
        Returns:
            冗余因子对列表 [(factor1, factor2, correlation), ...]
        """
        redundant = []
        factors = corr_matrix.columns.tolist()
        
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    redundant.append((factors[i], factors[j], corr))
        
        return sorted(redundant, key=lambda x: abs(x[2]), reverse=True)

    def suggest_reduction(
        self,
        factors: pd.DataFrame,
        corr_matrix: pd.DataFrame,
        ic_mean: Dict[str, float],
        threshold: float = 0.7
    ) -> Dict:
        """
        建议因子精简方案
        
        Args:
            factors: 因子 DataFrame
            corr_matrix: 相关性矩阵
            ic_mean: IC均值字典
            threshold: 相关性阈值
        
        Returns:
            建议字典
        """
        redundant_pairs = self.get_redundant_factors(corr_matrix, threshold)
        
        to_remove = set()
        
        for f1, f2, corr in redundant_pairs:
            if f1 not in to_remove and f2 not in to_remove:
                if ic_mean.get(f1, 0) > ic_mean.get(f2, 0):
                    to_remove.add(f2)
                else:
                    to_remove.add(f1)
        
        return {
            "redundant_pairs": redundant_pairs,
            "suggested_removal": list(to_remove),
            "remaining_factors": [f for f in corr_matrix.columns if f not in to_remove]
        }

    def analyze_correlation(
        self,
        factors: pd.DataFrame,
        ic_mean: Optional[Dict[str, float]] = None,
        threshold: float = 0.7
    ) -> Dict:
        """
        完整相关性分析
        
        Args:
            factors: 因子 DataFrame
            ic_mean: IC均值字典（可选，用于精简建议）
            threshold: 相关性阈值
        
        Returns:
            分析报告字典
        """
        corr_matrix = self.compute_correlation_matrix(factors)
        redundant = self.get_redundant_factors(corr_matrix, threshold)
        
        suggestion = None
        if ic_mean is not None:
            suggestion = self.suggest_reduction(factors, corr_matrix, ic_mean, threshold)
        
        return {
            "correlation_matrix": corr_matrix,
            "redundant_pairs": redundant,
            "suggestion": suggestion
        }
