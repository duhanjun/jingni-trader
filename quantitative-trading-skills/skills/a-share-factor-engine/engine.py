import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from config import Config, get_config
from factors import FactorBuilder
from ic_analysis import ICAnalyzer
from correlation import CorrelationAnalyzer
from ensemble import FactorCombiner
from storage import FactorStorage
from validation import AdversarialValidator


class AShareFactorEngine:
    """
    A股多因子引擎主类
    """

    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = get_config()
        self.config = config
        self.factor_builder = FactorBuilder(config)
        self.ic_analyzer = ICAnalyzer(config)
        self.corr_analyzer = CorrelationAnalyzer(config)
        self.factor_combiner = FactorCombiner(config)
        self.factor_storage = FactorStorage(config)
        self.adversarial_validator = AdversarialValidator(config)

    def compute_factors(
        self,
        price_data: pd.DataFrame,
        factor_list: Optional[List[str]] = None,
        winsorize: Optional[bool] = None,
        standardize: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        计算因子
        
        Args:
            price_data: 价格数据
            factor_list: 因子列表
            winsorize: 是否缩尾
            standardize: 是否标准化
        
        Returns:
            因子 DataFrame
        """
        if winsorize is None:
            winsorize = self.config.WINSORIZE_THRESHOLD > 0
        if standardize is None:
            standardize = self.config.STANDARDIZE
        
        factors = self.factor_builder.build_factors(price_data, factor_list)
        
        if winsorize:
            factors = self.factor_builder.winsorize(factors)
        if standardize:
            factors = self.factor_builder.standardize(factors)
        
        return factors

    def analyze_ic(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.DataFrame,
        industry_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        IC分析
        """
        return self.ic_analyzer.analyze_ic(factors, forward_returns, industry_data)

    def analyze_correlation(
        self,
        factors: pd.DataFrame,
        ic_mean: Optional[Dict[str, float]] = None,
        threshold: float = 0.7
    ) -> Dict:
        """
        相关性分析
        """
        return self.corr_analyzer.analyze_correlation(factors, ic_mean, threshold)

    def combine_factors(
        self,
        factors: pd.DataFrame,
        method: Optional[str] = None,
        ic_mean: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        因子融合
        """
        return self.factor_combiner.combine_factors(factors, method, ic_mean)

    def save_factors(
        self,
        factors: pd.DataFrame,
        filename: str,
        incremental: bool = False
    ) -> str:
        """
        保存因子
        """
        return self.factor_storage.save_factors(factors, filename, incremental)

    def load_factors(
        self,
        filename: str
    ) -> pd.DataFrame:
        """
        加载因子
        """
        return self.factor_storage.load_factors(filename)

    def check_distribution(
        self,
        train_factors: pd.DataFrame,
        test_factors: pd.DataFrame
    ) -> Dict:
        """
        分布一致性检查
        """
        return self.adversarial_validator.check_distribution_consistency(train_factors, test_factors)

    def full_workflow(
        self,
        price_data: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_list: Optional[List[str]] = None,
        save_filename: Optional[str] = None
    ) -> Dict:
        """
        完整工作流
        
        Args:
            price_data: 价格数据
            forward_returns: 未来收益
            factor_list: 因子列表
            save_filename: 保存文件名
        
        Returns:
            完整分析报告
        """
        factors = self.compute_factors(price_data, factor_list)
        ic_report = self.analyze_ic(factors, forward_returns)
        corr_report = self.analyze_correlation(factors, ic_report["ic_mean"])
        combined = self.combine_factors(factors, ic_mean=ic_report["ic_mean"])
        
        if save_filename is not None:
            self.save_factors(factors, save_filename)
        
        return {
            "factors": factors,
            "ic_report": ic_report,
            "corr_report": corr_report,
            "combined_alpha": combined
        }
