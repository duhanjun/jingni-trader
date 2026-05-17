"""
策略模型引擎主逻辑
负责数据准备、模型训练、超参优化、实验管理
"""
import os
import json
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 模型库
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

import optuna
import mlflow

from .config import (
    MODEL_TYPE, MODEL_DIR,
    OPTUNA_TRIALS, OPTUNA_TIMEOUT,
    TRAIN_WINDOW_MONTHS, VALIDATION_WINDOW_MONTHS, TEST_WINDOW_MONTHS,
    PURGE_GAP_DAYS, FORWARD_PERIOD, LABEL_TYPE,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)

warnings.filterwarnings('ignore')
logger = logging.getLogger("strategy-model-engine")


class ModelEngine:
    """策略模型引擎"""

    def __init__(self):
        # 初始化MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── 数据准备 ──────────────────────────
    def prepare_data(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'forward_return'
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        准备训练数据

        参数:
            factor_df: 因子数据，含 code, date 和因子列
            price_df: 行情数据，含 code, date, close
            feature_cols: 特征列名
            label_col: 标签列名（未来收益），如果未提供则自动计算

        返回:
            X (DataFrame), y (Series), dates (Series)
        """
        logger.info("准备训练数据...")

        # 计算标签（未来N日收益）
        price_df = price_df.sort_values(['code', 'date'])
        price_df['forward_return'] = price_df.groupby('code')['close'].transform(
            lambda x: x.shift(-FORWARD_PERIOD) / x - 1
        )

        # 合并因子和标签
        data = factor_df[['code', 'date'] + feature_cols].merge(
            price_df[['code', 'date', 'forward_return']],
            on=['code', 'date'],
            how='inner'
        )

        # 删除缺失值
        data = data.dropna()

        # 如果是分类问题，将收益转为二分类
        if LABEL_TYPE == 'classification':
            # 大于中位数则为正类
            threshold = data.groupby('date')['forward_return'].transform('median')
            data['label'] = (data['forward_return'] > threshold).astype(int)
            y_col = 'label'
        else:
            y_col = 'forward_return'

        X = data[feature_cols]
        y = data[y_col]
        dates = data['date']

        logger.info(f"数据准备完成: {len(X)} 样本, {len(feature_cols)} 特征")
        return X, y, dates

    # ── 分组时序交叉验证 ────────────────────
    def purged_group_ts_split(
        self,
        dates: pd.Series,
        n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Purged Group Time Series Split
        防止使用未来信息

        参数:
            dates: 每个样本对应的日期
            n_splits: 折数

        返回:
            [(train_indices, val_indices), ...]
        """
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        splits = []
        test_size = n_dates // (n_splits + 1)

        for i in range(n_splits):
            # 训练集：最早的一些日期
            train_end_idx = n_dates - (n_splits - i) * test_size
            # 验证集：紧随训练集之后的一段，但中间隔 PURGE_GAP_DAYS
            val_start_idx = train_end_idx + 1
            val_end_idx = min(val_start_idx + test_size, n_dates)

            if val_start_idx >= n_dates:
                break

            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            # 应用 purge gap：从训练集末尾向前剔除 gap 内的日期
            if PURGE_GAP_DAYS > 0:
                purge_date = unique_dates[train_end_idx] - timedelta(days=PURGE_GAP_DAYS)
                train_dates = [d for d in train_dates if d <= purge_date]

            train_idx = dates[dates.isin(train_dates)].index.values
            val_idx = dates[dates.isin(val_dates)].index.values

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        logger.info(f"生成 {len(splits)} 个交叉验证分割")
        return splits

    # ── 模型创建 ────────────────────────────
    def create_model(self, trial: Optional[optuna.Trial] = None) -> Any:
        """根据配置和 Optuna trial 创建模型"""
        if MODEL_TYPE == 'lightgbm':
            if trial is not None:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': -1,
                }
            else:
                params = {'random_state': 42, 'n_jobs': -1, 'verbosity': -1}

            if LABEL_TYPE == 'classification':
                return lgb.LGBMClassifier(**params)
            else:
                return lgb.LGBMRegressor(**params)

        elif MODEL_TYPE == 'catboost':
            if trial is not None:
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 500),
                    'depth': trial.suggest_int('depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                    'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
                    'random_seed': 42,
                    'verbose': False,
                }
            else:
                params = {'random_seed': 42, 'verbose': False}

            if LABEL_TYPE == 'classification':
                return CatBoostClassifier(**params)
            else:
                return CatBoostRegressor(**params)

        elif MODEL_TYPE == 'logistic_regression':
            if trial is not None:
                params = {
                    'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': 'saga',
                    'max_iter': 1000,
                    'random_state': 42,
                }
            else:
                params = {'max_iter': 1000, 'random_state': 42}
            return LogisticRegression(**params)

        elif MODEL_TYPE == 'random_forest':
            if trial is not None:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42,
                    'n_jobs': -1,
                }
            else:
                params = {'random_state': 42, 'n_jobs': -1}
            if LABEL_TYPE == 'classification':
                return RandomForestClassifier(**params)
            else:
                return RandomForestRegressor(**params)
        else:
            raise ValueError(f"不支持的模型类型: {MODEL_TYPE}")

    # ── 超参数优化 ──────────────────────────
    def optimize_hyperparams(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        n_trials: int = OPTUNA_TRIALS
    ) -> Dict[str, Any]:
        """
        使用 Optuna 进行超参数优化

        返回:
            最优参数字典
        """
        logger.info(f"开始超参数优化 (模型: {MODEL_TYPE}, trials: {n_trials})")

        # 生成一个时序分割用于评估
        splits = self.purged_group_ts_split(dates, n_splits=3)
        if not splits:
            logger.warning("无法生成交叉验证分割，使用默认参数")
            return {}

        def objective(trial):
            model = self.create_model(trial)
            scores = []

            for train_idx, val_idx in splits:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)

                if LABEL_TYPE == 'classification':
                    score = model.score(X_val, y_val)  # accuracy
                else:
                    from sklearn.metrics import mean_squared_error
                    pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, pred)

                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
        study.optimize(objective, n_trials=n_trials, timeout=OPTUNA_TIMEOUT)

        logger.info(f"超参数优化完成，最佳分数: {study.best_value}")
        return study.best_params
