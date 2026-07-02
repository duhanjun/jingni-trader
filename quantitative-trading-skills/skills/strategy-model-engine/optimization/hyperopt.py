import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
import optuna
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit


class HyperparameterOptimizer:
    """
    超参数优化器 - 使用Optuna
    """
    
    def __init__(self, config):
        self.config = config
        self.study = None
        self.best_params = None
        self.best_value = None
    
    def _suggest_lightgbm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        为LightGBM模型建议超参数
        
        Args:
            trial: Optuna trial对象
        
        Returns:
            参数字典
        """
        return {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "verbose": -1,
            "random_state": self.config.RANDOM_STATE
        }
    
    def _suggest_catboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        为CatBoost模型建议超参数
        
        Args:
            trial: Optuna trial对象
        
        Returns:
            参数字典
        """
        return {
            "objective": "RMSE",
            "iterations": trial.suggest_int("iterations", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.0, 10.0),
            "verbose": 0,
            "random_state": self.config.RANDOM_STATE
        }
    
    def _objective_regression(self, trial: optuna.Trial, 
                             model_type: str,
                             X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        回归目标函数
        
        Args:
            trial: Optuna trial对象
            model_type: 模型类型
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
        
        Returns:
            验证集RMSE
        """
        if model_type == "lightgbm":
            params = self._suggest_lightgbm_params(trial)
            model = lgb.LGBMRegressor(**params)
        elif model_type == "catboost":
            params = self._suggest_catboost_params(trial)
            model = cb.CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse
    
    def _objective_classification(self, trial: optuna.Trial,
                                 model_type: str,
                                 X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        分类目标函数
        
        Args:
            trial: Optuna trial对象
            model_type: 模型类型
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
        
        Returns:
            负准确率（因为Optuna最小化）
        """
        if model_type == "lightgbm":
            params = self._suggest_lightgbm_params(trial)
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type for classification: {model_type}")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return -accuracy  # 最小化负准确率
    
    def optimize(self, X: pd.DataFrame, y: pd.Series,
                model_type: str = "lightgbm",
                task_type: str = "regression",
                n_trials: Optional[int] = None,
                val_size: float = 0.2,
                direction: str = "minimize") -> Dict[str, Any]:
        """
        优化超参数
        
        Args:
            X: 特征数据
            y: 目标数据
            model_type: 模型类型
            task_type: 任务类型 ('regression' 或 'classification')
            n_trials: 试验次数
            val_size: 验证集比例
            direction: 优化方向 ('minimize' 或 'maximize')
        
        Returns:
            最佳参数字典
        """
        if n_trials is None:
            n_trials = self.config.OPTIMIZATION_N_TRIALS
        
        # 划分训练集和验证集（时间序列划分）
        split_idx = int(len(X) * (1 - val_size))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 创建study
        self.study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler())
        
        # 定义目标函数
        if task_type == "regression":
            def objective(trial):
                return self._objective_regression(trial, model_type, X_train, y_train, X_val, y_val)
        elif task_type == "classification":
            def objective(trial):
                return self._objective_classification(trial, model_type, X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # 运行优化
        self.study.optimize(objective, n_trials=n_trials)
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return self.best_params
    
    def get_study_summary(self) -> Dict[str, Any]:
        """
        获取study摘要
        
        Returns:
            摘要字典
        """
        if self.study is None:
            raise ValueError("No optimization has been run yet")
        
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "trials_data": self.study.trials_dataframe()
        }
