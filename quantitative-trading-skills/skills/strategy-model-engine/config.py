import os
from typing import Optional, Dict, Any


class Config:
    """
    配置类
    """

    def __init__(self, **kwargs):
        self.MLFLOW_TRACKING_URI: str = kwargs.get(
            "MLFLOW_TRACKING_URI",
            os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
        )
        
        self.EXPERIMENT_NAME: str = kwargs.get(
            "EXPERIMENT_NAME",
            os.environ.get("EXPERIMENT_NAME", "strategy_model_engine")
        )
        
        self.RANDOM_STATE: int = kwargs.get("RANDOM_STATE", 42)
        
        self.CV_TRAIN_WINDOW: int = kwargs.get("CV_TRAIN_WINDOW", 36)
        self.CV_VALID_WINDOW: int = kwargs.get("CV_VALID_WINDOW", 12)
        self.CV_TEST_WINDOW: int = kwargs.get("CV_TEST_WINDOW", 12)
        self.CV_PURGE_GAP: int = kwargs.get("CV_PURGE_GAP", 2)
        
        self.OPTIMIZATION_N_TRIALS: int = kwargs.get("OPTIMIZATION_N_TRIALS", 50)
        self.OPTIMIZATION_ALGORITHM: str = kwargs.get("OPTIMIZATION_ALGORITHM", "tpe")
        
        self.STOCK_SELECTION_MODEL: str = kwargs.get("STOCK_SELECTION_MODEL", "lightgbm")
        self.TIMING_MODEL: str = kwargs.get("TIMING_MODEL", "lightgbm")
        
        self.FORWARD_RETURN_DAYS: int = kwargs.get("FORWARD_RETURN_DAYS", 3)
        
        self.NEUTRALIZE_INDUSTRY: bool = kwargs.get("NEUTRALIZE_INDUSTRY", True)
        
        self.LGBM_PARAMS: Dict[str, Any] = kwargs.get("LGBM_PARAMS", {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "num_leaves": 31,
            "verbose": -1,
            "random_state": 42
        })
        
        self.CATBOOST_PARAMS: Dict[str, Any] = kwargs.get("CATBOOST_PARAMS", {
            "objective": "RMSE",
            "iterations": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "verbose": 0,
            "random_state": 42
        })


def get_config(**kwargs) -> Config:
    """
    获取配置实例
    
    Args:
        **kwargs: 可选的配置覆盖
    
    Returns:
        Config 实例
    """
    return Config(**kwargs)
