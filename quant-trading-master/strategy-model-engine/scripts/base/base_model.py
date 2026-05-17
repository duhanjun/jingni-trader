"""
模型基类定义
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd


class BaseModel(ABC):
    """模型基类"""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """预测"""
        pass

    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path: str) -> Any:
        """加载模型"""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        pass
