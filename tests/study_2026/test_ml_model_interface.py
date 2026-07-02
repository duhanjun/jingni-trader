"""
验证测试：标准化 ML 模型接口与超参数优化（Standardized ML Model Interface）

借鉴来源：Freqtrade + FreqAI (https://www.freqtrade.io/en/stable/freqai/)
  - FreqAI 通过 IFreqaiModel 抽象基类实现模型与策略的解耦：
    * train() → fit() → predict() 三步接口
    * DataKitchen/DataDrawer 负责数据管道（归一化、PCA、异常值检测）
    * Hyperopt (Optuna) 实现贝叶斯超参数优化
    * 滑动训练窗口 + 模型过期机制

优化方向：为 jingni-trader 的 strategy-model-engine 引入标准化 ML 模型接口，
  支持可插拔模型、自动化超参数搜索、数据预处理管道。

测试内容：
  1. 标准化 ML 模型接口（train/fit/predict）
  2. 数据预处理管道（DataPipeline）
  3. 超参数优化（Hyperopt via Optuna）
  4. 滑动窗口训练
  5. 模型持久化
  6. 边界条件
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import time
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SKPipeline


# ============================================================
# 标准化 ML 模型接口（借鉴 FreqAI 设计）
# ============================================================

class BaseQuantModel(ABC):
    """
    量化 ML 模型基类

    借鉴 FreqAI 的 IFreqaiModel 设计：
    - train() → fit() → predict() 三步接口
    - 支持自定义特征工程
    - 支持模型持久化
    """

    def __init__(self):
        self.model: Optional[BaseEstimator] = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """创建底层 sklearn 模型实例"""
        ...

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseQuantModel":
        """训练模型"""
        self.feature_names = list(X.columns)
        self.model = self._create_model()
        self.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """拟合模型（子类可重写以添加验证集等）"""
        self.model.fit(X.values, y.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        return self.model.predict(X.values)

    def save(self, path: str):
        import joblib
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }
        joblib.dump(data, path)

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.metadata = data.get("metadata", {})
        self.is_fitted = True


class LightGBMQuantModel(BaseQuantModel):
    """LightGBM 量化模型"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__()
        self.params = params or {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbosity": -1,
        }

    def _create_model(self) -> BaseEstimator:
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**self.params)
        except ImportError:
            # 回退到 sklearn RandomForest
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth", 5),
                random_state=self.params.get("random_state", 42),
            )


class XGBoostQuantModel(BaseQuantModel):
    """XGBoost 量化模型"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__()
        self.params = params or {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbosity": 0,
        }

    def _create_model(self) -> BaseEstimator:
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(**self.params)
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth", 5),
                random_state=self.params.get("random_state", 42),
            )


# ============================================================
# 数据预处理管道（借鉴 FreqAI DataKitchen 设计）
# ============================================================

class DataPipeline:
    """
    数据预处理管道

    借鉴 FreqAI 的 DataKitchen/DataDrawer 设计：
    - 可配置的预处理步骤（归一化、PCA、异常值检测）
    - 支持 fit/transform 分离，避免数据泄露
    - 处理金融时序特有的特征工程需求
    """

    def __init__(self, steps: Optional[List[Tuple[str, Any]]] = None):
        self.steps = steps or [
            ("scaler", StandardScaler()),
        ]
        self.pipeline = SKPipeline(self.steps)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        self.pipeline.fit(X)
        self.is_fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        transformed = self.pipeline.transform(X)
        # Handle column reduction (e.g. PCA)
        if transformed.shape[1] != X.shape[1]:
            new_cols = [f"pc_{i+1}" for i in range(transformed.shape[1])]
            return pd.DataFrame(transformed, index=X.index, columns=new_cols)
        return pd.DataFrame(transformed, index=X.index, columns=X.columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


# ============================================================
# 超参数优化（借鉴 Freqtrade Hyperopt + Optuna 设计）
# ============================================================

class HyperoptEngine:
    """
    超参数优化引擎

    借鉴 Freqtrade 的 Hyperopt 设计：
    - 基于 Optuna 的贝叶斯优化
    - 支持自定义损失函数
    - 支持时间序列交叉验证
    """

    def __init__(self, n_trials: int = 50, n_folds: int = 3):
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.best_params: Optional[Dict] = None
        self.best_score: float = float("-inf")
        self.study_results: List[Dict] = []

    def optimize(
        self,
        model_factory: callable,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        loss_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        执行超参数优化

        参数:
            model_factory: 接受参数字典并返回 BaseQuantModel 实例的工厂函数
            param_space: 参数空间定义
            X: 特征
            y: 标签
            loss_fn: 损失函数 (y_true, y_pred) → float

        返回:
            {"best_params": ..., "best_score": ..., "study_results": [...]}
        """
        try:
            import optuna
            return self._optimize_optuna(model_factory, param_space, X, y, loss_fn)
        except ImportError:
            return self._optimize_grid_search(model_factory, param_space, X, y, loss_fn)

    def _optimize_optuna(
        self,
        model_factory: callable,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        loss_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        import optuna

        if loss_fn is None:
            def default_loss(y_true, y_pred):
                ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
                return ic  # 最大化 IC
            loss_fn = default_loss

        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        def objective(trial):
            # 构建参数
            params = {}
            for name, spec in param_space.items():
                if spec["type"] == "int":
                    params[name] = trial.suggest_int(name, spec["low"], spec["high"])
                elif spec["type"] == "float":
                    params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
                elif spec["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, spec["choices"])

            # 交叉验证
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = model_factory(params)
                model.train(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(loss_fn(y_val.values, y_pred))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        self.best_score = study.best_value
        self.study_results = [
            {"params": t.params, "score": t.value}
            for t in study.trials
        ]

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(study.trials),
            "study_results": self.study_results,
        }

    def _optimize_grid_search(
        self,
        model_factory: callable,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        loss_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """无 Optuna 时的回退：网格搜索"""
        from itertools import product

        if loss_fn is None:
            def default_loss(y_true, y_pred):
                ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
                return ic
            loss_fn = default_loss

        # 生成候选参数组合
        keys = list(param_space.keys())
        values = []
        for spec in param_space.values():
            if spec["type"] == "int":
                values.append(list(range(spec["low"], spec["high"] + 1, max(1, (spec["high"] - spec["low"]) // 3))))
            elif spec["type"] == "float":
                values.append(np.linspace(spec["low"], spec["high"], 4).tolist())
            elif spec["type"] == "categorical":
                values.append(spec["choices"])

        best_params = None
        best_score = float("-inf")

        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        for combo in product(*values):
            params = dict(zip(keys, combo))
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = model_factory(params)
                model.train(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(loss_fn(y_val.values, y_pred))

            avg_score = np.mean(scores)
            self.study_results.append({"params": params, "score": avg_score})
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        self.best_params = best_params
        self.best_score = best_score

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.study_results),
            "study_results": self.study_results,
        }


# ============================================================
# 滑动窗口训练器（借鉴 FreqAI 设计）
# ============================================================

class SlidingWindowTrainer:
    """
    滑动窗口训练器

    借鉴 FreqAI 的滑动训练窗口设计：
    - train_period_days: 训练窗口大小
    - retrain_interval: 重训练间隔
    - 支持模型过期机制
    """

    def __init__(
        self,
        train_window: int = 252,
        retrain_interval: int = 20,
        model_factory: Optional[callable] = None,
    ):
        self.train_window = train_window
        self.retrain_interval = retrain_interval
        self.model_factory = model_factory or (lambda: LightGBMQuantModel())
        self.current_model: Optional[BaseQuantModel] = None
        self.last_train_date: Optional[pd.Timestamp] = None

    def should_retrain(self, current_date: pd.Timestamp) -> bool:
        if self.last_train_date is None:
            return True
        days_since = (current_date - self.last_train_date).days
        return days_since >= self.retrain_interval

    def train_on_window(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current_date: pd.Timestamp,
    ):
        """在当前日期滑动窗口上训练"""
        window_start = current_date - pd.Timedelta(days=self.train_window)
        mask = (X.index >= window_start) & (X.index <= current_date)
        X_window = X.loc[mask]
        y_window = y.loc[mask]

        if len(X_window) < 50:
            return

        self.current_model = self.model_factory()
        self.current_model.train(X_window, y_window)
        self.last_train_date = current_date

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.current_model is None:
            raise RuntimeError("No trained model available")
        return self.current_model.predict(X)


# ============================================================
# 测试用例
# ============================================================

class TestModelInterface(unittest.TestCase):
    """测试标准化 ML 模型接口"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 500
        cls.X = pd.DataFrame({
            "ret_1d": np.random.normal(0, 0.02, n),
            "ret_5d": np.random.normal(0, 0.04, n),
            "ret_20d": np.random.normal(0, 0.08, n),
            "vol_20d": np.abs(np.random.normal(0.02, 0.01, n)),
            "ma20_ratio": np.random.normal(1.0, 0.05, n),
        })
        cls.y = pd.Series(
            0.5 * cls.X["ret_1d"] - 0.3 * cls.X["ret_5d"] + np.random.normal(0, 0.01, n)
        )

    def test_train_predict_save_load(self):
        """测试完整训练-预测-保存-加载流程"""
        model = LightGBMQuantModel()
        model.train(self.X.iloc[:400], self.y.iloc[:400])
        self.assertTrue(model.is_fitted)

        preds = model.predict(self.X.iloc[400:])
        self.assertEqual(len(preds), 100)

        # 保存和加载
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            model2 = LightGBMQuantModel()
            model2.load(path)
            self.assertTrue(model2.is_fitted)
            preds2 = model2.predict(self.X.iloc[400:])
            np.testing.assert_array_almost_equal(preds, preds2)
        finally:
            os.unlink(path)

    def test_multiple_model_types(self):
        """测试多种模型类型"""
        models = [
            LightGBMQuantModel(),
            XGBoostQuantModel(),
        ]
        for model in models:
            model.train(self.X.iloc[:400], self.y.iloc[:400])
            preds = model.predict(self.X.iloc[-50:])
            self.assertEqual(len(preds), 50)

    def test_predict_before_train(self):
        """未训练时预测应报错"""
        model = LightGBMQuantModel()
        with self.assertRaises(RuntimeError):
            model.predict(self.X.iloc[:10])


class TestDataPipeline(unittest.TestCase):
    """测试数据预处理管道"""

    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame({
            "f1": np.random.normal(0, 1, 100),
            "f2": np.random.normal(10, 5, 100),
            "f3": np.random.lognormal(0, 1, 100),
        })

    def test_standard_scaler(self):
        pipeline = DataPipeline([("scaler", StandardScaler())])
        transformed = pipeline.fit_transform(self.X)
        self.assertEqual(transformed.shape, self.X.shape)
        # StandardScaler with ddof=0 on small samples, use looser delta
        self.assertAlmostEqual(transformed["f1"].mean(), 0, delta=0.1)
        self.assertAlmostEqual(transformed["f1"].std(), 1, delta=0.1)

    def test_pca_pipeline(self):
        pipeline = DataPipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2)),
        ])
        transformed = pipeline.fit_transform(self.X)
        self.assertEqual(transformed.shape[1], 2)

    def test_minmax_scaler(self):
        pipeline = DataPipeline([("scaler", MinMaxScaler())])
        transformed = pipeline.fit_transform(self.X)
        self.assertTrue((transformed["f1"] >= 0).all())
        self.assertTrue((transformed["f1"] <= 1).all())

    def test_transform_before_fit(self):
        pipeline = DataPipeline()
        with self.assertRaises(RuntimeError):
            pipeline.transform(self.X)


class TestHyperoptEngine(unittest.TestCase):
    """测试超参数优化"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 300
        cls.X = pd.DataFrame({
            "ret_1d": np.random.normal(0, 0.02, n),
            "ret_5d": np.random.normal(0, 0.04, n),
            "ret_20d": np.random.normal(0, 0.08, n),
        })
        cls.y = pd.Series(
            0.4 * cls.X["ret_1d"] + np.random.normal(0, 0.02, n)
        )

    def test_hyperopt_basic(self):
        """测试基本超参数搜索"""
        def factory(params):
            return LightGBMQuantModel(params)

        param_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
        }

        engine = HyperoptEngine(n_trials=20, n_folds=3)
        result = engine.optimize(factory, param_space, self.X, self.y)

        self.assertIn("best_params", result)
        self.assertIn("best_score", result)
        self.assertGreater(result["n_trials"], 0)

        print(f"\n    超参数优化结果:")
        print(f"    最佳参数: {result['best_params']}")
        print(f"    最佳 IC: {result['best_score']:.4f}")

    def test_hyperopt_with_custom_loss(self):
        """测试自定义损失函数"""
        def factory(params):
            return LightGBMQuantModel(params)

        def custom_loss(y_true, y_pred):
            # 使用 rank IC
            return np.corrcoef(
                pd.Series(y_true).rank(),
                pd.Series(y_pred).rank()
            )[0, 1]

        param_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 150},
            "max_depth": {"type": "int", "low": 3, "high": 6},
        }

        engine = HyperoptEngine(n_trials=10, n_folds=2)
        result = engine.optimize(factory, param_space, self.X, self.y, loss_fn=custom_loss)

        self.assertIsNotNone(result["best_params"])
        self.assertGreaterEqual(result["best_score"], -1.0)
        self.assertLessEqual(result["best_score"], 1.0)


class TestSlidingWindowTrainer(unittest.TestCase):
    """测试滑动窗口训练器"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
        n = len(dates)
        cls.X = pd.DataFrame(
            {"ret_1d": np.random.normal(0, 0.02, n)},
            index=dates,
        )
        cls.y = pd.Series(
            0.3 * cls.X["ret_1d"] + np.random.normal(0, 0.01, n),
            index=dates,
        )

    def test_retrain_on_interval(self):
        """测试按间隔重训练"""
        trainer = SlidingWindowTrainer(
            train_window=120,
            retrain_interval=60,
        )
        self.assertTrue(trainer.should_retrain(pd.Timestamp("2023-01-01")))

        # 第一次训练
        trainer.train_on_window(self.X, self.y, pd.Timestamp("2023-06-15"))
        self.assertFalse(trainer.should_retrain(pd.Timestamp("2023-06-20")))

        # 超过间隔后应重训练
        self.assertTrue(trainer.should_retrain(pd.Timestamp("2023-09-01")))

    def test_predict_after_training(self):
        """测试训练后预测"""
        trainer = SlidingWindowTrainer(train_window=120, retrain_interval=60)
        trainer.train_on_window(self.X, self.y, pd.Timestamp("2023-06-15"))

        preds = trainer.predict(self.X.iloc[-50:])
        self.assertEqual(len(preds), 50)

    def test_predict_before_training(self):
        """未训练时预测应报错"""
        trainer = SlidingWindowTrainer()
        with self.assertRaises(RuntimeError):
            trainer.predict(self.X.iloc[:10])


if __name__ == "__main__":
    unittest.main(verbosity=2)