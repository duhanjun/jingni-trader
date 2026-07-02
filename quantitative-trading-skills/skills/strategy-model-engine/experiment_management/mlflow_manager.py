import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.catboost
import lightgbm as lgb
import catboost as cb
from contextlib import contextmanager


class ExperimentManager:
    """
    实验管理器 - 使用MLflow
    """
    
    def __init__(self, config):
        self.config = config
        self.tracking_uri = config.MLFLOW_TRACKING_URI
        self.experiment_name = config.EXPERIMENT_NAME
        self.active_run = None
        
        # 设置MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # 创建或获取实验
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        上下文管理器，开始一个run
        
        Args:
            run_name: run名称
            tags: 标签字典
        """
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        ) as run:
            self.active_run = run
            yield
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        记录参数
        
        Args:
            params: 参数字典
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 步骤
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model: Any, model_name: str, 
                 conda_env: Optional[Union[str, Dict[str, Any]]] = None):
        """
        记录模型
        
        Args:
            model: 模型对象
            model_name: 模型名称
            conda_env: conda环境
        """
        if isinstance(model, lgb.LGBMModel):
            mlflow.lightgbm.log_model(model, model_name, conda_env=conda_env)
        elif isinstance(model, cb.CatBoost):
            mlflow.catboost.log_model(model, model_name, conda_env=conda_env)
        else:
            mlflow.sklearn.log_model(model, model_name, conda_env=conda_env)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        记录artifact
        
        Args:
            local_path: 本地文件路径
            artifact_path: artifact路径
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dataframe(self, df: pd.DataFrame, artifact_path: str):
        """
        记录DataFrame为artifact
        
        Args:
            df: DataFrame
            artifact_path: artifact路径
        """
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, f"{artifact_path}.csv")
            df.to_csv(file_path, index=False)
            mlflow.log_artifact(file_path, artifact_path)
    
    def log_factor_combination(self, factors: List[str], prefix: str = "factor"):
        """
        记录因子组合
        
        Args:
            factors: 因子列表
            prefix: 参数前缀
        """
        for i, factor in enumerate(factors):
            mlflow.log_param(f"{prefix}_{i}", factor)
        mlflow.log_param(f"{prefix}_count", len(factors))
    
    def log_feature_importance(self, importance_df: pd.DataFrame, 
                               artifact_path: str = "feature_importance"):
        """
        记录特征重要性
        
        Args:
            importance_df: 特征重要性DataFrame
            artifact_path: artifact路径
        """
        self.log_dataframe(importance_df, artifact_path)
        
        # 记录最重要的几个特征
        top_features = importance_df.head(10)
        for i, row in top_features.iterrows():
            mlflow.log_param(f"top_feature_{i}", row["feature"])
            mlflow.log_metric(f"top_importance_{i}", row["importance"])
    
    def get_run_info(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取run信息
        
        Args:
            run_id: run ID，如果为None则使用当前active run
        
        Returns:
            run信息字典
        """
        if run_id is None:
            if self.active_run is None:
                raise ValueError("No active run and no run_id provided")
            run_id = self.active_run.info.run_id
        
        run = mlflow.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags
        }
    
    def search_runs(self, filter_string: Optional[str] = None, 
                   order_by: Optional[List[str]] = None,
                   max_results: int = 100) -> pd.DataFrame:
        """
        搜索runs
        
        Args:
            filter_string: 过滤字符串
            order_by: 排序字段
            max_results: 最大结果数
        
        Returns:
            runs DataFrame
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results
        )
        return runs
    
    def load_model(self, run_id: str, model_name: str = "model") -> Any:
        """
        加载模型
        
        Args:
            run_id: run ID
            model_name: 模型名称
        
        Returns:
            模型对象
        """
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.pyfunc.load_model(model_uri)
