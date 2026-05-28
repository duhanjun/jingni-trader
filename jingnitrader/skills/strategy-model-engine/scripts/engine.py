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
import joblib

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

    # ── 训练最终模型 ────────────────────────
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        best_params: Dict[str, Any] = None,
        test_dates: pd.Series = None
    ) -> Tuple[Any, Dict[str, float], Optional[np.ndarray]]:
        """
        训练模型并评估

        参数:
            X, y: 全部数据
            best_params: 最优参数
            test_dates: 测试集日期（用于样本外评估）

        返回:
            model, metrics, predictions
        """
        logger.info("训练最终模型...")

        with mlflow.start_run():
            # 日志参数
            mlflow.log_params({
                "model_type": MODEL_TYPE,
                "label_type": LABEL_TYPE,
                "forward_period": FORWARD_PERIOD,
                "features": ", ".join(X.columns.tolist()),
            })

            if best_params:
                mlflow.log_params(best_params)

            # 创建并训练模型
            model = self.create_model()
            if best_params:
                model.set_params(**best_params)

            # 如果指定了测试集日期，则用测试集之前的数据训练
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

            # 评估
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

                # IC 分析（预测值与实际收益的相关性）
                pred_series = pd.Series(predictions, index=X_test.index)
                y_test_aligned = y_test.loc[X_test.index]
                metrics['ic'] = pred_series.corr(y_test_aligned)

            else:
                # 在全部训练集上的拟合指标
                if hasattr(model, 'score'):
                    metrics['train_score'] = model.score(X_train, y_train)

            mlflow.log_metrics(metrics)

            # 日志特征重要性
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(X.columns, model.feature_importances_))
                # 保存为JSON
                with mlflow.start_run(nested=True):
                    mlflow.log_dict(importance_dict, "feature_importance.json")

            # 保存模型
            model_path = os.path.join(MODEL_DIR, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            logger.info(f"模型训练完成，指标: {metrics}")
            return model, metrics, predictions

    # ── 策略模板：规则型策略 ────────────────
    def generate_rule_based_signal(
        self,
        factor_df: pd.DataFrame,
        strategy_type: str = "single_factor"
    ) -> pd.DataFrame:
        """
        生成基于规则的策略信号（不涉及ML训练）

        参数:
            factor_df: 因子数据
            strategy_type: 策略类型
                - single_factor: 单因子选股（如反转因子）
                - mean_reversion: 均值回归
                - trend_following: 趋势跟踪

        返回:
            信号 DataFrame (code, date, signal)
        """
        df = factor_df[['code', 'date']].copy()

        if strategy_type == "single_factor":
            # 单因子：选因子值最低（或最高）的N只
            if 'alpha_score' in factor_df.columns:
                # 使用融合Alpha
                factor_col = 'alpha_score'
            elif 'reversal_20d' in factor_df.columns:
                factor_col = 'reversal_20d'
            else:
                factor_col = [c for c in factor_df.columns if c not in ['code', 'date']][0]

            # 每天选因子值最大的前20%做多（简化示例）
            df['raw_score'] = factor_df[factor_col]
            df['rank_pct'] = df.groupby('date')['raw_score'].rank(pct=True)
            df['signal'] = 0
            df.loc[df['rank_pct'] > 0.8, 'signal'] = 1  # 前20%买入
            df.loc[df['rank_pct'] < 0.2, 'signal'] = -1  # 后20%卖出（若允许融券）
            df = df[['code', 'date', 'signal']]

        elif strategy_type == "mean_reversion":
            # 均值回归：基于价格偏离均线
            if 'ret_20d' in factor_df.columns:
                df['signal'] = -np.sign(factor_df['ret_20d'])  # 跌多了买，涨多了卖
            df = df[['code', 'date', 'signal']]

        elif strategy_type == "trend_following":
            # 趋势跟踪：基于移动平均交叉
            if 'ma_20' in factor_df.columns and 'close' in factor_df.columns:
                df['signal'] = (factor_df['close'] > factor_df['ma_20']).astype(int)
            df = df[['code', 'date', 'signal']]

        else:
            raise ValueError(f"未知策略类型: {strategy_type}")

        logger.info(f"规则型策略信号生成完成: {strategy_type}")
        return df


# ── Skill 统一入口 ────────────────────────
def run(ctx) -> Dict[str, Any]:
    """
    strategy-model-engine 的 run 函数

    参数:
        ctx: Context 对象，需包含:
            - artifacts['DATA']: 清洗后数据路径
            - artifacts['FACTOR']: 因子数据路径
            - strategy_name: 策略类型 (可选)
            - strategy_params: 策略参数 (可选)

    返回:
        {
            "success": bool,
            "artifact_path": str,     # 模型文件路径 或 信号文件路径
            "predictions_path": str,  # 预测信号文件路径
            "metadata": {
                "model_type": str,
                "metrics": dict,
                "feature_importance": dict,
                ...
            },
            "error": str
        }
    """
    try:
        # 检查依赖产物
        factor_path = ctx.get_artifact("FACTOR")
        if not factor_path or not os.path.exists(factor_path):
            return {
                "success": False,
                "artifact_path": "",
                "metadata": {},
                "error": "因子产物不存在，请先运行 a-share-factor-engine"
            }

        # 检查模型产物是否已存在
        existing = ctx.get_artifact("MODEL")
        if existing and os.path.exists(existing):
            return {
                "success": True,
                "artifact_path": existing,
                "metadata": {"source": "cache"},
                "error": ""
            }

        engine = ModelEngine()

        # 加载因子数据
        factor_df = pd.read_parquet(factor_path)

        # 也加载行情数据（用于计算标签）
        data_path = ctx.get_artifact("DATA")
        if not data_path or not os.path.exists(data_path):
            return {"success": False, "artifact_path": "", "metadata": {}, "error": "行情数据不存在"}

        price_df = pd.read_parquet(data_path)

        # 确定策略类型
        strategy_name = getattr(ctx, 'strategy_name', None) or ctx.strategy_params.get('strategy_type', 'ml')

        if strategy_name in ['ml', 'model', 'lightgbm', 'catboost']:
            # ML模型训练
            feature_cols = [c for c in factor_df.columns
                          if c not in ['code', 'date', 'industry', 'alpha_score']]
            if 'alpha_score' in factor_df.columns:
                feature_cols.append('alpha_score')

            X, y, dates = engine.prepare_data(factor_df, price_df, feature_cols)

            # 超参数优化（可选）
            best_params = engine.optimize_hyperparams(X, y, dates)

            # 训练最终模型
            model, metrics, predictions = engine.train(X, y, best_params)

            # 保存预测信号
            if predictions is not None:
                signal_df = factor_df[['code', 'date']].copy()
                signal_df['pred'] = predictions
                signal_path = os.path.join(MODEL_DIR, "predictions.parquet")
                signal_df.to_parquet(signal_path, index=False)
            else:
                signal_path = ""

            model_path = os.path.join(MODEL_DIR, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
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
            # 规则型策略
            signal_df = engine.generate_rule_based_signal(factor_df, strategy_type=strategy_name)

            # 保存信号
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
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jingnitrader.scripts.context import Context

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            ctx = Context.from_dict(json.load(f))
    else:
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
