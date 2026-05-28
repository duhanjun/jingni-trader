"""
策略模型引擎主逻辑
负责数据准备、模型训练、超参优化、实验管理
"""
import os
import sys
import json
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from scripts.config import (
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
        if HAS_MLFLOW:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def prepare_data(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'forward_return'
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """准备训练数据"""
        logger.info("准备训练数据...")

        price_df = price_df.sort_values(['code', 'date'])
        price_df['forward_return'] = price_df.groupby('code')['close'].transform(
            lambda x: x.shift(-FORWARD_PERIOD) / x - 1
        )

        data = factor_df[['code', 'date'] + feature_cols].merge(
            price_df[['code', 'date', 'forward_return']],
            on=['code', 'date'],
            how='inner'
        )

        data = data.dropna()

        if LABEL_TYPE == 'classification':
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

    def purged_group_ts_split(
        self,
        dates: pd.Series,
        n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Purged Group Time Series Split"""
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        splits = []
        test_size = n_dates // (n_splits + 1)

        for i in range(n_splits):
            train_end_idx = n_dates - (n_splits - i) * test_size
            val_start_idx = train_end_idx + 1
            val_end_idx = min(val_start_idx + test_size, n_dates)

            if val_start_idx >= n_dates:
                break

            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            if PURGE_GAP_DAYS > 0:
                purge_date = unique_dates[train_end_idx] - timedelta(days=PURGE_GAP_DAYS)
                train_dates = [d for d in train_dates if d <= purge_date]

            train_idx = dates[dates.isin(train_dates)].index.values
            val_idx = dates[dates.isin(val_dates)].index.values

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        logger.info(f"生成 {len(splits)} 个交叉验证分割")
        return splits

    def create_model(self, trial: Optional['optuna.Trial'] = None) -> Any:
        """根据配置创建模型"""
        if MODEL_TYPE == 'lightgbm':
            if HAS_LGB:
                if trial is not None and HAS_OPTUNA:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
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
            else:
                raise ImportError("LightGBM 未安装")

        elif MODEL_TYPE == 'catboost':
            if HAS_CATBOOST:
                if trial is not None and HAS_OPTUNA:
                    params = {
                        'iterations': trial.suggest_int('iterations', 50, 500),
                        'depth': trial.suggest_int('depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'random_seed': 42,
                        'verbose': False,
                    }
                else:
                    params = {'random_seed': 42, 'verbose': False}

                if LABEL_TYPE == 'classification':
                    return CatBoostClassifier(**params)
                else:
                    return CatBoostRegressor(**params)
            else:
                raise ImportError("CatBoost 未安装")

        elif MODEL_TYPE == 'logistic_regression':
            if trial is not None and HAS_OPTUNA:
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
            if trial is not None and HAS_OPTUNA:
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

    def optimize_hyperparams(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        n_trials: int = OPTUNA_TRIALS
    ) -> Dict[str, Any]:
        """使用 Optuna 进行超参数优化"""
        if not HAS_OPTUNA:
            logger.warning("Optuna 未安装，跳过超参数优化")
            return {}

        logger.info(f"开始超参数优化 (模型: {MODEL_TYPE}, trials: {n_trials})")

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
                    score = model.score(X_val, y_val)
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

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        best_params: Dict[str, Any] = None,
        test_dates: pd.Series = None
    ) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
        """训练模型并评估"""
        logger.info("训练最终模型...")

        if HAS_MLFLOW:
            with mlflow.start_run():
                mlflow.log_params({
                    "model_type": MODEL_TYPE,
                    "label_type": LABEL_TYPE,
                    "forward_period": FORWARD_PERIOD,
                    "features": ", ".join(X.columns.tolist()),
                })
                if best_params:
                    mlflow.log_params(best_params)

        model = self.create_model()
        if best_params:
            model.set_params(**best_params)

        if test_dates is not None and len(test_dates) > 0:
            train_mask = ~X.index.isin(test_dates.index)
            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]
            X_test = X.loc[~train_mask]
            y_test = y.loc[~train_mask]
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None

        model.fit(X_train, y_train)

        metrics = {}
        predictions = None

        if X_test is not None:
            predictions = model.predict(X_test)

            if LABEL_TYPE == 'classification':
                from sklearn.metrics import accuracy_score, f1_score
                metrics['accuracy'] = accuracy_score(y_test, predictions)
                metrics['f1'] = f1_score(y_test, predictions, average='weighted')
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                metrics['mse'] = mean_squared_error(y_test, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y_test, predictions)

            pred_series = pd.Series(predictions, index=X_test.index)
            y_test_aligned = y_test.loc[X_test.index]
            metrics['ic'] = pred_series.corr(y_test_aligned)
        else:
            if hasattr(model, 'score'):
                metrics['train_score'] = model.score(X_train, y_train)

        if HAS_MLFLOW:
            mlflow.log_metrics(metrics)

            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(X.columns, model.feature_importances_))
                if HAS_MLFLOW:
                    with mlflow.start_run(nested=True):
                        mlflow.log_dict(importance_dict, "feature_importance.json")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        if HAS_JOBLIB:
            joblib.dump(model, model_path)
        if HAS_MLFLOW:
            mlflow.log_artifact(model_path)

        logger.info(f"模型训练完成，指标: {metrics}")
        return model, metrics, predictions

    def generate_rule_based_signal(
        self,
        factor_df: pd.DataFrame,
        strategy_type: str = "single_factor"
    ) -> pd.DataFrame:
        """生成基于规则的策略信号"""
        df = factor_df[['code', 'date']].copy()

        if strategy_type == "single_factor":
            if 'alpha_score' in factor_df.columns:
                factor_col = 'alpha_score'
            elif 'reversal_20d' in factor_df.columns:
                factor_col = 'reversal_20d'
            else:
                factor_col = [c for c in factor_df.columns if c not in ['code', 'date']][0]

            df['raw_score'] = factor_df[factor_col]
            df['rank_pct'] = df.groupby('date')['raw_score'].rank(pct=True)
            df['signal'] = 0
            df.loc[df['rank_pct'] > 0.8, 'signal'] = 1
            df.loc[df['rank_pct'] < 0.2, 'signal'] = -1
            df = df[['code', 'date', 'signal']]

        elif strategy_type == "mean_reversion":
            if 'ret_20d' in factor_df.columns:
                df['signal'] = -np.sign(factor_df['ret_20d'])
            df = df[['code', 'date', 'signal']]

        elif strategy_type == "trend_following":
            if 'ma_20' in factor_df.columns and 'close' in factor_df.columns:
                df['signal'] = (factor_df['close'] > factor_df['ma_20']).astype(int)
            df = df[['code', 'date', 'signal']]

        else:
            raise ValueError(f"未知策略类型: {strategy_type}")

        logger.info(f"规则型策略信号生成完成: {strategy_type}")
        return df


def run(ctx) -> Dict[str, Any]:
    """
    strategy-model-engine 的 run 函数

    参数:
        ctx: Context 对象，需包含:
            - artifacts['DATA']: 清洗后数据路径
            - artifacts['FACTOR']: 因子数据路径

    返回:
        {
            "success": bool,
            "artifact_path": str,
            "predictions_path": str,
            "metadata": {...},
            "error": str
        }
    """
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        factor_path = ctx.get_artifact("FACTOR")
        if not factor_path or not os.path.exists(factor_path):
            return {
                "success": False,
                "artifact_path": "",
                "metadata": {},
                "error": "因子产物不存在，请先运行 a-share-factor-engine"
            }

        existing = ctx.get_artifact("MODEL")
        if existing and os.path.exists(existing):
            return {
                "success": True,
                "artifact_path": existing,
                "metadata": {"source": "cache"},
                "error": ""
            }

        engine = ModelEngine()

        factor_df = pd.read_parquet(factor_path)

        data_path = ctx.get_artifact("DATA")
        if not data_path or not os.path.exists(data_path):
            return {"success": False, "artifact_path": "", "metadata": {}, "error": "行情数据不存在"}

        price_df = pd.read_parquet(data_path)

        strategy_name = getattr(ctx, 'strategy_name', None) or ctx.strategy_params.get('strategy_type', 'ml')

        if strategy_name in ['ml', 'model', 'lightgbm', 'catboost']:
            feature_cols = [c for c in factor_df.columns
                          if c not in ['code', 'date', 'industry', 'alpha_score']]
            if 'alpha_score' in factor_df.columns:
                feature_cols.append('alpha_score')

            X, y, dates = engine.prepare_data(factor_df, price_df, feature_cols)

            best_params = engine.optimize_hyperparams(X, y, dates)

            model, metrics, predictions = engine.train(X, y, best_params)

            if predictions is not None:
                signal_df = factor_df[['code', 'date']].copy()
                signal_df['pred'] = predictions
                signal_path = os.path.join(MODEL_DIR, "predictions.parquet")
                signal_df.to_parquet(signal_path, index=False)
            else:
                signal_path = ""

            model_path = os.path.join(MODEL_DIR, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            if HAS_JOBLIB:
                joblib.dump(model, model_path)

            return {
                "success": True,
                "artifact_path": model_path,
                "predictions_path": signal_path,
                "metadata": {
                    "model_type": MODEL_TYPE,
                    "metrics": metrics,
                    "feature_cols": feature_cols,
                },
                "error": ""
            }

        else:
            signal_df = engine.generate_rule_based_signal(factor_df, strategy_type=strategy_name)

            signal_path = os.path.join(MODEL_DIR, f"signal_{strategy_name}.parquet")
            signal_df.to_parquet(signal_path, index=False)

            return {
                "success": True,
                "artifact_path": signal_path,
                "predictions_path": signal_path,
                "metadata": {
                    "strategy_type": strategy_name,
                    "signal_count": len(signal_df),
                },
                "error": ""
            }

    except Exception as e:
        logger.exception("模型引擎执行失败")
        return {
            "success": False,
            "artifact_path": "",
            "metadata": {},
            "error": str(e)
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            ctx_dict = json.load(f)
        from scripts.context import Context
        ctx = Context.from_dict(ctx_dict)
    else:
        from scripts.context import Context
        ctx = Context(
            task_id="test_model",
            stock_pool=[],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        ctx.update_artifact("DATA", "./workspace/data/cleaned_data.parquet")
        ctx.update_artifact("FACTOR", "./workspace/factors/factor_data.parquet")

    result = run(ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
