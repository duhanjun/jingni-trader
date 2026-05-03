import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class ICAnalyzer:
    """
    IC 分析器
    """

    def __init__(self, config: Config):
        self.config = config

    def compute_ic_series(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算 IC 时间序列
        
        Args:
            factors: 因子 DataFrame
            forward_returns: 未来收益 DataFrame，包含 code, date, forward_return 字段
        
        Returns:
            IC 时间序列 DataFrame
        """
        if factors.empty or forward_returns.empty:
            return pd.DataFrame()
        
        merged = pd.merge(factors, forward_returns, on=["code", "date"], how="inner")
        factor_cols = [col for col in factors.columns if col not in ["code", "date"]]
        ic_data = []
        
        for date in merged["date"].unique():
            daily_data = merged[merged["date"] == date]
            daily_ic = {"date": date}
            
            for factor in factor_cols:
                valid = daily_data[[factor, "forward_return"]].dropna()
                if len(valid) >= 10:
                    ic, _ = spearmanr(valid[factor], valid["forward_return"])
                    daily_ic[factor] = ic
                else:
                    daily_ic[factor] = np.nan
            
            ic_data.append(daily_ic)
        
        return pd.DataFrame(ic_data)

    def compute_icir(
        self,
        ic_series: pd.DataFrame
    ) -> Dict[str, float]:
        """
        计算 ICIR
        
        Args:
            ic_series: IC 时间序列 DataFrame
        
        Returns:
            ICIR 字典
        """
        icir_dict = {}
        factor_cols = [col for col in ic_series.columns if col != "date"]
        
        for factor in factor_cols:
            ic_values = ic_series[factor].dropna()
            if len(ic_values) > 0:
                ic_mean = ic_values.mean()
                ic_std = ic_values.std()
                if ic_std > 0:
                    icir_dict[factor] = ic_mean / ic_std
                else:
                    icir_dict[factor] = np.nan
            else:
                icir_dict[factor] = np.nan
        
        return icir_dict

    def neutralize_industry(
        self,
        factors: pd.DataFrame,
        industry_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        行业中性化
        
        Args:
            factors: 因子 DataFrame
            industry_data: 行业数据 DataFrame，包含 code, date, industry 字段
        
        Returns:
            中性化后的因子 DataFrame
        """
        if factors.empty or industry_data.empty:
            return factors
        
        merged = pd.merge(factors, industry_data, on=["code", "date"], how="inner")
        result = merged[["code", "date"]].copy()
        factor_cols = [col for col in factors.columns if col not in ["code", "date"]]
        
        for factor in factor_cols:
            neutralized = []
            for (date, industry), group in merged.groupby(["date", "industry"]):
                mean_val = group[factor].mean()
                neutral = group[factor] - mean_val
                neutralized.append(pd.DataFrame({
                    "code": group["code"],
                    "date": group["date"],
                    factor: neutral
                }))
            
            factor_result = pd.concat(neutralized, ignore_index=True)
            result = pd.merge(result, factor_result, on=["code", "date"], how="left")
        
        return result

    def analyze_ic(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.DataFrame,
        industry_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        完整 IC 分析
        
        Args:
            factors: 因子 DataFrame
            forward_returns: 未来收益 DataFrame
            industry_data: 行业数据（可选，用于中性化）
        
        Returns:
            分析报告字典
        """
        factors_to_use = factors.copy()
        
        if self.config.NEUTRALIZE_INDUSTRY and industry_data is not None:
            factors_to_use = self.neutralize_industry(factors_to_use, industry_data)
        
        ic_series = self.compute_ic_series(factors_to_use, forward_returns)
        icir = self.compute_icir(ic_series)
        
        ic_mean = ic_series.drop(columns=["date"]).mean().to_dict()
        ic_std = ic_series.drop(columns=["date"]).std().to_dict()
        
        return {
            "ic_series": ic_series,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir
        }
