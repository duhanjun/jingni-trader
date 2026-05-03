import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager

from config import Config, get_config
from strategy_templates import StrategyTemplateGenerator
from stock_selection import StockSelectionModel
from timing_models import TimingModel
from optimization import HyperparameterOptimizer
from experiment_management import ExperimentManager
from overfitting_protection import PurgedGroupTimeSeriesSplit


class StrategyModelEngine:
    """
    策略模型引擎 - 主入口
    整合了策略模板生成、截面选股、择时模型、参数优化、实验管理和过拟合防范
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化引擎
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        if config is None:
            config = get_config()
        self.config = config
        
        # 初始化各个组件
        self.strategy_generator = StrategyTemplateGenerator(config)
        self.stock_selector = StockSelectionModel(config)
        self.timing_model = TimingModel(config)
        self.optimizer = HyperparameterOptimizer(config)
        self.experiment_manager = ExperimentManager(config)
    
    def generate_strategy_template(self, strategy_type: str, **kwargs) -> Dict[str, Any]:
        """
        生成策略模板
        
        Args:
            strategy_type: 策略类型 ('trend_following', 'mean_reversion', 'pair_trading')
            **kwargs: 策略参数
        
        Returns:
            策略模板字典
        """
        return self.strategy_generator.generate_template(strategy_type, **kwargs)
    
    def train_stock_selection_model(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series,
                                   model_type: str = "lightgbm",
                                   industry: Optional[pd.Series] = None,
                                   params: Optional[Dict[str, Any]] = None):
        """
        训练截面选股模型
        
        Args:
            X: 因子数据
            y: 目标收益
            model_type: 模型类型 ('lightgbm' 或 'catboost')
            industry: 行业标签（用于中性化）
            params: 模型参数
        
        Returns:
            训练好的模型
        """
        return self.stock_selector.train(X, y, model_type, industry, params)
    
    def predict_stock_selection(self, 
                               X: pd.DataFrame, 
                               industry: Optional[pd.Series] = None) -> pd.Series:
        """
        截面选股预测
        
        Args:
            X: 因子数据
            industry: 行业标签
        
        Returns:
            预测结果
        """
        return self.stock_selector.predict(X, industry)
    
    def train_timing_model(self, 
                          minute_data: pd.DataFrame,
                          model_type: str = "lightgbm",
                          params: Optional[Dict[str, Any]] = None):
        """
        训练择时模型
        
        Args:
            minute_data: 分钟线数据，包含 open, high, low, close, volume
            model_type: 模型类型
            params: 模型参数
        
        Returns:
            训练好的模型
        """
        X, y = self.timing_model.prepare_features_and_target(minute_data)
        return self.timing_model.train(X, y, params)
    
    def predict_timing(self, minute_data: pd.DataFrame) -> pd.Series:
        """
        择时预测
        
        Args:
            minute_data: 分钟线数据
        
        Returns:
            预测结果
        """
        X, _ = self.timing_model.prepare_features_and_target(minute_data)
        return self.timing_model.predict(X)
    
    def optimize_hyperparameters(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                model_type: str = "lightgbm",
                                task_type: str = "regression",
                                n_trials: Optional[int] = None,
                                val_size: float = 0.2,
                                direction: str = "minimize") -> Dict[str, Any]:
        """
        超参数优化
        
        Args:
            X: 特征数据
            y: 目标数据
            model_type: 模型类型
            task_type: 任务类型 ('regression' 或 'classification')
            n_trials: 试验次数
            val_size: 验证集比例
            direction: 优化方向
        
        Returns:
            最佳参数字典
        """
        return self.optimizer.optimize(X, y, model_type, task_type, n_trials, val_size, direction)
    
    def get_purged_group_splits(self, 
                               X: pd.DataFrame,
                               groups: Optional[pd.Series] = None,
                               time_col: Optional[str] = None,
                               n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        获取Purged Group Time Series划分
        
        Args:
            X: 特征DataFrame
            groups: 分组Series（如行业）
            time_col: 时间列名
            n_splits: 折数
        
        Returns:
            划分列表
        """
        cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits)
        return cv.split(X, groups=groups, time_col=time_col)
    
    @contextmanager
    def start_experiment(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        开始实验（上下文管理器）
        
        Args:
            run_name: run名称
            tags: 标签字典
        """
        with self.experiment_manager.start_run(run_name, tags):
            yield
    
    def log_params(self, params: Dict[str, Any]):
        """
        记录参数
        """
        self.experiment_manager.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        记录指标
        """
        self.experiment_manager.log_metrics(metrics)
    
    def log_model(self, model: Any, model_name: str):
        """
        记录模型
        """
        self.experiment_manager.log_model(model, model_name)
    
    def log_factor_combination(self, factors: List[str]):
        """
        记录因子组合
        """
        self.experiment_manager.log_factor_combination(factors)
    
    def full_workflow(self, 
                     X: pd.DataFrame, 
                     y: pd.Series,
                     model_type: str = "lightgbm",
                     industry: Optional[pd.Series] = None,
                     factors: Optional[List[str]] = None,
                     run_name: Optional[str] = None,
                     optimize: bool = False,
                     use_cv: bool = True) -> Dict[str, Any]:
        """
        完整工作流
        
        Args:
            X: 因子数据
            y: 目标收益
            model_type: 模型类型
            industry: 行业标签
            factors: 因子列表（用于记录）
            run_name: run名称
            optimize: 是否进行超参数优化
            use_cv: 是否使用交叉验证
        
        Returns:
            结果字典
        """
        results = {}
        
        with self.start_experiment(run_name):
            # 记录因子组合
            if factors is not None:
                self.log_factor_combination(factors)
            
            # 超参数优化
            if optimize:
                best_params = self.optimize_hyperparameters(X, y, model_type)
                self.log_params(best_params)
            else:
                best_params = None
            
            # 训练模型
            model = self.train_stock_selection_model(X, y, model_type, industry, best_params)
            results["model"] = model
            
            # 记录模型
            self.log_model(model, "stock_selection_model")
            
            # 特征重要性
            feature_importance = self.stock_selector.get_feature_importance()
            self.experiment_manager.log_feature_importance(feature_importance)
            results["feature_importance"] = feature_importance
            
            # 预测
            predictions = self.predict_stock_selection(X, industry)
            results["predictions"] = predictions
        
        return results
