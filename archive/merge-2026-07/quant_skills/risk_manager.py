"""
风险控制与组合管理模块
提供风险分析、组合优化等功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple


class RiskManager:
    """
    风险管理工具类
    """
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        计算风险价值 (VaR)
        
        参数:
            returns: 收益率序列
            confidence_level: 置信水平
            method: 计算方法 ("historical", "variance_covariance")
            
        返回:
            float: VaR值 (负值表示损失)
        """
        if method == "historical":
            var = -np.percentile(returns.dropna(), (1 - confidence_level) * 100)
        elif method == "variance_covariance":
            mean = returns.mean()
            std = returns.std()
            z_score = np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100)
            var = -(mean + z_score * std)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        return var
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        计算条件风险价值 (CVaR)
        
        参数:
            returns: 收益率序列
            confidence_level: 置信水平
            
        返回:
            float: CVaR值
        """
        returns_clean = returns.dropna()
        var_threshold = np.percentile(returns_clean, (1 - confidence_level) * 100)
        tail_returns = returns_clean[returns_clean <= var_threshold]
        
        if len(tail_returns) > 0:
            cvar = -tail_returns.mean()
        else:
            cvar = self.calculate_var(returns, confidence_level)
        
        return cvar
    
    def calculate_max_drawdown(
        self,
        prices: pd.Series
    ) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        计算最大回撤
        
        参数:
            prices: 价格序列
            
        返回:
            Tuple: (最大回撤, 开始时间, 结束时间)
        """
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        
        max_dd = drawdown.min()
        end_date = drawdown.idxmin()
        start_date = cummax.loc[:end_date].idxmax()
        
        return max_dd, start_date, end_date
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods: int = 252
    ) -> float:
        """
        计算夏普比率
        
        参数:
            returns: 收益率序列
            risk_free_rate: 无风险利率 (年化)
            periods: 一年的期数
            
        返回:
            float: 夏普比率
        """
        excess_returns = returns - (risk_free_rate / periods)
        return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods: int = 252
    ) -> float:
        """
        计算索提诺比率
        
        参数:
            returns: 收益率序列
            risk_free_rate: 无风险利率 (年化)
            periods: 一年的期数
            
        返回:
            float: 索提诺比率
        """
        excess_returns = returns - (risk_free_rate / periods)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std > 0:
            return np.sqrt(periods) * excess_returns.mean() / downside_std
        else:
            return np.inf
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        periods: int = 252,
        window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        计算波动率
        
        参数:
            returns: 收益率序列
            periods: 一年的期数
            window: 滚动窗口大小 (None表示计算整个时期的波动率)
            
        返回:
            float or pd.Series: 波动率
        """
        if window is None:
            return returns.std() * np.sqrt(periods)
        else:
            return returns.rolling(window=window).std() * np.sqrt(periods)
    
    def calculate_beta(
        self,
        returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        计算Beta
        
        参数:
            returns: 资产收益率
            market_returns: 市场收益率
            
        返回:
            float: Beta值
        """
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        
        if market_variance > 0:
            return covariance / market_variance
        else:
            return 0.0
    
    def risk_summary(
        self,
        returns: pd.Series,
        market_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0
    ) -> Dict[str, Any]:
        """
        生成风险分析摘要
        
        参数:
            returns: 收益率序列
            market_returns: 市场收益率序列 (可选)
            risk_free_rate: 无风险利率
            
        返回:
            Dict: 风险指标摘要
        """
        summary = {
            "total_return": (1 + returns).prod() - 1,
            "annual_return": (1 + returns).prod() ** (252 / len(returns)) - 1,
            "annual_volatility": self.calculate_volatility(returns),
            "sharpe_ratio": self.calculate_sharpe_ratio(returns, risk_free_rate),
            "sortino_ratio": self.calculate_sortino_ratio(returns, risk_free_rate),
            "var_95": self.calculate_var(returns, 0.95),
            "var_99": self.calculate_var(returns, 0.99),
            "cvar_95": self.calculate_cvar(returns, 0.95),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
        }
        
        if market_returns is not None:
            summary["beta"] = self.calculate_beta(returns, market_returns)
        
        return summary


class PortfolioOptimizer:
    """
    组合优化工具类
    """
    
    def __init__(self):
        self._pypfopt_available = False
        
        try:
            from pypfopt import EfficientFrontier, risk_models, expected_returns
            from pypfopt import objective_functions
            self._ef = EfficientFrontier
            self._risk_models = risk_models
            self._expected_returns = expected_returns
            self._objective_functions = objective_functions
            self._pypfopt_available = True
        except ImportError:
            pass
    
    def calculate_min_variance_weights(
        self,
        returns: pd.DataFrame,
        allow_shorting: bool = False
    ) -> Dict[str, float]:
        """
        计算最小方差组合权重
        
        参数:
            returns: 收益率数据 (每列一个资产)
            allow_shorting: 是否允许做空
            
        返回:
            Dict: 资产权重
        """
        if self._pypfopt_available:
            # 使用PyPortfolioOpt
            cov_matrix = self._risk_models.sample_cov(returns)
            ef = self._ef(None, cov_matrix, weight_bounds=(0, 1) if not allow_shorting else (-1, 1))
            weights = ef.min_volatility()
            return ef.clean_weights()
        else:
            # 手动实现
            cov_matrix = returns.cov()
            n = len(cov_matrix)
            
            # 逆矩阵
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n)
            
            # 计算权重
            weights = inv_cov @ ones
            weights = weights / weights.sum()
            
            # 如果不允许做空，进行非负约束
            if not allow_shorting:
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
            
            return dict(zip(returns.columns, weights))
    
    def calculate_max_sharpe_weights(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        allow_shorting: bool = False
    ) -> Dict[str, float]:
        """
        计算最大夏普比率组合权重
        
        参数:
            returns: 收益率数据 (每列一个资产)
            risk_free_rate: 无风险利率
            allow_shorting: 是否允许做空
            
        返回:
            Dict: 资产权重
        """
        if self._pypfopt_available:
            # 使用PyPortfolioOpt
            mu = self._expected_returns.mean_historical_return(returns)
            cov_matrix = self._risk_models.sample_cov(returns)
            ef = self._ef(mu, cov_matrix, weight_bounds=(0, 1) if not allow_shorting else (-1, 1))
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            return ef.clean_weights()
        else:
            # 简化实现：等权重
            n = len(returns.columns)
            return {col: 1.0 / n for col in returns.columns}
    
    def calculate_equal_risk_contribution(
        self,
        returns: pd.DataFrame,
        allow_shorting: bool = False
    ) -> Dict[str, float]:
        """
        计算风险平价组合权重 (Equal Risk Contribution)
        
        参数:
            returns: 收益率数据 (每列一个资产)
            allow_shorting: 是否允许做空
            
        返回:
            Dict: 资产权重
        """
        # 简化实现：基于波动率倒数
        volatilities = returns.std()
        weights = 1 / volatilities
        weights = weights / weights.sum()
        
        # 如果不允许做空
        if not allow_shorting:
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
        
        return weights.to_dict()
    
    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        method: str = "min_variance",
        **kwargs
    ) -> Dict[str, float]:
        """
        优化投资组合
        
        参数:
            returns: 收益率数据
            method: 优化方法 ("min_variance", "max_sharpe", "equal_risk")
            **kwargs: 其他参数
            
        返回:
            Dict: 资产权重
        """
        if method == "min_variance":
            return self.calculate_min_variance_weights(returns, **kwargs)
        elif method == "max_sharpe":
            return self.calculate_max_sharpe_weights(returns, **kwargs)
        elif method == "equal_risk":
            return self.calculate_equal_risk_contribution(returns, **kwargs)
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def portfolio_performance(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算组合表现
        
        参数:
            returns: 收益率数据
            weights: 资产权重
            
        返回:
            Dict: 组合表现指标
        """
        # 计算组合收益率
        weights_series = pd.Series(weights).reindex(returns.columns)
        portfolio_returns = (returns * weights_series).sum(axis=1)
        
        # 计算指标
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        # 计算组合方差
        cov_matrix = returns.cov()
        weights_array = np.array([weights.get(col, 0) for col in returns.columns])
        port_variance = weights_array @ cov_matrix @ weights_array
        port_volatility = np.sqrt(port_variance) * np.sqrt(252)
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": annual_return / annual_vol if annual_vol > 0 else 0
        }
