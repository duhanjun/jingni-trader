import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler


class StockSelectionModel:
    """
    截面选股模型
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_type = None
    
    def _neutralize_industry(self, X: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
        """
        行业中性化
        
        Args:
            X: 因子数据
            industry: 行业标签
        
        Returns:
            中性化后的因子数据
        """
        X_neutralized = X.copy()
        for col in X.columns:
            industry_mean = X_neutralized.groupby(industry)[col].transform("mean")
            X_neutralized[col] = X_neutralized[col] - industry_mean
        return X_neutralized
    
    def train_lightgbm(self, X: pd.DataFrame, y: pd.Series, 
                      industry: Optional[pd.Series] = None,
                      params: Optional[Dict[str, Any]] = None) -> lgb.LGBMRegressor:
        """
        训练LightGBM模型
        
        Args:
            X: 因子数据
            y: 目标收益
            industry: 行业标签（用于中性化）
            params: 模型参数
        
        Returns:
            训练好的LightGBM模型
        """
        if params is None:
            params = self.config.LGBM_PARAMS
        
        if self.config.NEUTRALIZE_INDUSTRY and industry is not None:
            X = self._neutralize_industry(X, industry)
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_scaled, y)
        
        self.model = model
        self.feature_names = X.columns.tolist()
        self.model_type = "lightgbm"
        
        return model
    
    def train_catboost(self, X: pd.DataFrame, y: pd.Series,
                     industry: Optional[pd.Series] = None,
                     params: Optional[Dict[str, Any]] = None) -> cb.CatBoostRegressor:
        """
        训练CatBoost模型
        
        Args:
            X: 因子数据
            y: 目标收益
            industry: 行业标签（用于中性化）
            params: 模型参数
        
        Returns:
            训练好的CatBoost模型
        """
        if params is None:
            params = self.config.CATBOOST_PARAMS
        
        if self.config.NEUTRALIZE_INDUSTRY and industry is not None:
            X = self._neutralize_industry(X, industry)
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        model = cb.CatBoostRegressor(**params)
        model.fit(X_scaled, y, verbose=False)
        
        self.model = model
        self.feature_names = X.columns.tolist()
        self.model_type = "catboost"
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series,
            model_type: str = "lightgbm",
            industry: Optional[pd.Series] = None,
            params: Optional[Dict[str, Any]] = None):
        """
        训练模型
        
        Args:
            X: 因子数据
            y: 目标收益
            model_type: 模型类型 ('lightgbm' 或 'catboost')
            industry: 行业标签（用于中性化）
            params: 模型参数
        
        Returns:
            训练好的模型
        """
        if model_type == "lightgbm":
            return self.train_lightgbm(X, y, industry, params)
        elif model_type == "catboost":
            return self.train_catboost(X, y, industry, params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict(self, X: pd.DataFrame, industry: Optional[pd.Series] = None) -> pd.Series:
        """
        预测
        
        Args:
            X: 因子数据
            industry: 行业标签
        
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.config.NEUTRALIZE_INDUSTRY and industry is not None:
            X = self._neutralize_industry(X, industry)
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        predictions = self.model.predict(X_scaled)
        return pd.Series(predictions, index=X.index, name="predicted_return")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == "lightgbm":
            importance = self.model.feature_importances_
        elif self.model_type == "catboost":
            importance = self.model.get_feature_importance()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
