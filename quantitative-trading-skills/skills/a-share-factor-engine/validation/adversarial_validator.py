import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class AdversarialValidator:
    """
    对抗性验证器
    """

    def __init__(self, config: Config):
        self.config = config

    def check_distribution_consistency(
        self,
        train_factors: pd.DataFrame,
        test_factors: pd.DataFrame
    ) -> Dict:
        """
        检查训练集和测试集分布一致性
        
        Args:
            train_factors: 训练集因子
            test_factors: 测试集因子
        
        Returns:
            一致性检查报告
        """
        factor_cols = [col for col in train_factors.columns if col not in ["code", "date"]]
        results = {}
        
        for factor in factor_cols:
            train_vals = train_factors[factor].dropna()
            test_vals = test_factors[factor].dropna()
            
            if len(train_vals) >= 30 and len(test_vals) >= 30:
                ks_stat, ks_pvalue = stats.ks_2samp(train_vals, test_vals)
                t_stat, t_pvalue = stats.ttest_ind(train_vals, test_vals, equal_var=False)
                
                results[factor] = {
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "t_statistic": t_stat,
                    "t_pvalue": t_pvalue,
                    "train_mean": train_vals.mean(),
                    "test_mean": test_vals.mean(),
                    "train_std": train_vals.std(),
                    "test_std": test_vals.std(),
                    "ks_passed": ks_pvalue > 0.05,
                    "t_passed": t_pvalue > 0.05
                }
        
        passed_count = sum(1 for r in results.values() if r["ks_passed"] and r["t_passed"])
        
        return {
            "factor_results": results,
            "passed_count": passed_count,
            "total_count": len(factor_cols),
            "overall_passed": passed_count / len(factor_cols) >= 0.8
        }

    def validate_factor_stability(
        self,
        factors: pd.DataFrame,
        window: int = 20
    ) -> Dict:
        """
        验证因子稳定性
        
        Args:
            factors: 因子 DataFrame
            window: 滚动窗口大小
        
        Returns:
            稳定性报告
        """
        if "date" not in factors.columns:
            return {}
        
        factor_cols = [col for col in factors.columns if col not in ["code", "date"]]
        df_sorted = factors.sort_values("date")
        
        stability = {}
        
        for factor in factor_cols:
            rolling_mean = df_sorted.groupby("date")[factor].mean().rolling(window).mean()
            rolling_std = df_sorted.groupby("date")[factor].std().rolling(window).std()
            
            stability[factor] = {
                "mean_stability": 1 - rolling_std.std() / (rolling_mean.abs().mean() + 1e-8),
                "std_stability": 1 - rolling_std.std() / (rolling_std.mean() + 1e-8)
            }
        
        return stability
