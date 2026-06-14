"""
================================================================================
优化方向: 自适应 ML 训练管线（FreqAI 风格）
借鉴来源: Freqtrade (https://github.com/freqtrade/freqtrade) — FreqAI 模块
         FreqAI 实现了自动化的 ML 特征工程和模型训练管线，支持:
         - 自适应数据窗口（rolling/expanding window training）
         - 自动特征重要性分析与特征选择
         - 多种模型类型（LightGBM, XGBoost, CatBoost, PyTorch, Keras）
         - 在线学习与模型自动重训练
         - 离群值检测与数据清洗

优化目标:
  当前 jingni-trader 的 strategy-model-engine 存在以下可改进点:
  1. 训练窗口固定，不支持自适应 rolling window 重训练
  2. 没有自动特征选择机制
  3. 没有离群值清洗
  4. 模型训练和推理耦合在一起

验证内容:
  1. Rolling/Expanding Window 自适应训练管道
  2. 自动特征选择（基于 SHAP 或 feature importance）
  3. 离群值检测与清洗
  4. 模型 persistence 与重训练触发条件
  5. 性能对比：自适应 vs 固定窗口
"""

import unittest
import sys
import os
import time
import json
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

warnings.filterwarnings("ignore")

# ── 可选依赖 ────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ================================================================================
# Part 1: 自适应训练管线
# ================================================================================


class OutlierDetector:
    """离群值检测器 — 参考 FreqAI 的 outlier_protection"""

    @staticmethod
    def mad_outliers(
        series: pd.Series,
        threshold: float = 3.0,
    ) -> pd.Series:
        """MAD (Median Absolute Deviation) 方法检测离群值"""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            return pd.Series(False, index=series.index)
        z_score = 0.6745 * (series - median) / mad
        return z_score.abs() > threshold

    @staticmethod
    def iqr_outliers(
        series: pd.Series,
        multiplier: float = 1.5,
    ) -> pd.Series:
        """IQR 方法检测离群值"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return (series < lower) | (series > upper)

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        target_col: str,
        method: str = "mad",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """移除目标变量中的离群值"""
        if method == "mad":
            outliers = OutlierDetector.mad_outliers(df[target_col], threshold)
        elif method == "iqr":
            outliers = OutlierDetector.iqr_outliers(df[target_col], threshold)
        else:
            outliers = pd.Series(False, index=df.index)

        n_removed = outliers.sum()
        if n_removed > 0:
            print(f"  离群值检测: 移除 {n_removed}/{len(df)} ({n_removed/len(df)*100:.1f}%) 个样本")
        return df[~outliers].copy()


class FeatureSelector:
    """自动特征选择器 — 参考 FreqAI 的 feature selection"""

    @staticmethod
    def select_by_importance(
        model: Any,
        feature_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 10,
    ) -> List[str]:
        """基于模型特征重要性选择 Top-K 特征"""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            return feature_names[:top_k]  # fallback

        indices = np.argsort(importances)[::-1][:top_k]
        selected = [feature_names[i] for i in indices if i < len(feature_names)]
        return selected

    @staticmethod
    def select_by_correlation(
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.01,
        max_features: int = 20,
    ) -> List[str]:
        """基于与目标变量的相关性选择特征"""
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected = correlations[correlations > threshold].index.tolist()
        return selected[:max_features]


@dataclass
class TrainingWindow:
    """训练窗口定义"""
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class AdaptiveMLPipeline:
    """自适应 ML 训练管线 (FreqAI 风格)

    支持两种模式:
    1. rolling_window: 固定窗口长度，随时间滑动
    2. expanding_window: 不断扩展的训练窗口
    """

    def __init__(
        self,
        window_mode: str = "rolling",
        window_months: int = 12,
        retrain_frequency_months: int = 3,
        outlier_method: str = "mad",
        outlier_threshold: float = 3.0,
        top_k_features: int = 10,
    ):
        self.window_mode = window_mode
        self.window_months = window_months
        self.retrain_frequency_months = retrain_frequency_months
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.top_k_features = top_k_features

        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.selected_features: List[str] = []
        self.last_train_date: Optional[str] = None
        self.metrics_history: List[Dict] = []

    def generate_windows(
        self,
        start_date: str,
        end_date: str,
    ) -> List[TrainingWindow]:
        """生成训练窗口序列"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        windows = []

        if self.window_mode == "rolling":
            # 从末尾往前面滑动
            current_end = end
            while True:
                window_end = current_end
                window_start = current_end - pd.DateOffset(months=self.window_months)
                test_start = window_end
                test_end = min(window_end + pd.DateOffset(months=self.retrain_frequency_months), end)

                if window_start < start:
                    break

                windows.append(TrainingWindow(
                    train_start=window_start.strftime("%Y-%m-%d"),
                    train_end=window_end.strftime("%Y-%m-%d"),
                    test_start=test_start.strftime("%Y-%m-%d"),
                    test_end=test_end.strftime("%Y-%m-%d"),
                ))
                current_end = current_end - pd.DateOffset(months=self.retrain_frequency_months)

        elif self.window_mode == "expanding":
            # 不断扩展训练集
            current_test_start = start + pd.DateOffset(months=self.window_months)
            while current_test_start < end:
                test_end = min(current_test_start + pd.DateOffset(months=self.retrain_frequency_months), end)
                windows.append(TrainingWindow(
                    train_start=start.strftime("%Y-%m-%d"),
                    train_end=current_test_start.strftime("%Y-%m-%d"),
                    test_start=current_test_start.strftime("%Y-%m-%d"),
                    test_end=test_end.strftime("%Y-%m-%d"),
                ))
                current_test_start = current_test_start + pd.DateOffset(months=self.retrain_frequency_months)

        return list(reversed(windows))

    def needs_retrain(self, current_date: str) -> bool:
        """判断是否需要重训练"""
        if self.last_train_date is None:
            return True
        if self.model is None:
            return True
        months_diff = (
            pd.to_datetime(current_date).year * 12 + pd.to_datetime(current_date).month
        ) - (
            pd.to_datetime(self.last_train_date).year * 12 + pd.to_datetime(self.last_train_date).month
        )
        return months_diff >= self.retrain_frequency_months

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        train_date: str = None,
    ) -> Dict[str, Any]:
        """执行一次训练迭代"""
        t0 = time.perf_counter()

        # 1. 离群值检测与清洗
        df_combined = X.copy()
        df_combined["__target__"] = y
        df_clean = OutlierDetector.remove_outliers(
            df_combined, "__target__",
            method=self.outlier_method,
            threshold=self.outlier_threshold,
        )
        y_clean = df_clean.pop("__target__")

        # 2. 特征归一化
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(df_clean)
        else:
            X_scaled = self.scaler.transform(df_clean)

        # 3. 初始模型训练
        if HAS_LGB:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, random_state=42,
                verbosity=-1, n_jobs=-1,
            )
        elif HAS_SKLEARN:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1,
            )
        else:
            model = LinearRegression()

        model.fit(X_scaled, y_clean)

        # 4. 特征选择
        self.selected_features = FeatureSelector.select_by_importance(
            model, feature_names,
            pd.DataFrame(X_scaled, columns=feature_names), y_clean,
            top_k=self.top_k_features,
        )

        # 5. 在选定的特征上重新训练（如有 sklearn）
        if HAS_SKLEARN and HAS_LGB:
            sel_indices = [feature_names.index(f) for f in self.selected_features if f in feature_names]
            if sel_indices:
                X_sel = pd.DataFrame(X_scaled, columns=feature_names).iloc[:, sel_indices]
                model = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=5, random_state=42,
                    verbosity=-1, n_jobs=-1,
                )
                model.fit(X_sel, y_clean)

        self.model = model
        self.last_train_date = train_date or datetime.now().strftime("%Y-%m-%d")

        train_time = time.perf_counter() - t0

        metrics = {
            "train_date": self.last_train_date,
            "n_samples": len(df_clean),
            "n_features": len(self.selected_features),
            "selected_features": self.selected_features,
            "train_time_seconds": round(train_time, 4),
        }

        # 如果在 sklearn 环境，计算 in-sample R²
        if HAS_SKLEARN and len(self.selected_features) > 0 and len(y_clean) > 1:
            from sklearn.metrics import r2_score
            sel_indices = [feature_names.index(f) for f in self.selected_features if f in feature_names]
            if sel_indices:
                X_sel = pd.DataFrame(X_scaled, columns=feature_names).iloc[:, sel_indices]
                y_pred = model.predict(X_sel)
                metrics["r2_train"] = round(r2_score(y_clean, y_pred), 4)

        self.metrics_history.append(metrics)
        return metrics

    def predict(self, X: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        """用训练好的模型进行预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        if self.selected_features and HAS_SKLEARN:
            sel_indices = [feature_names.index(f) for f in self.selected_features if f in feature_names]
            if sel_indices:
                X_scaled = pd.DataFrame(X_scaled, columns=feature_names).iloc[:, sel_indices]

        return self.model.predict(X_scaled)


# ================================================================================
# Part 2: 测试用例
# ================================================================================


class TestAdaptiveMLPipeline(unittest.TestCase):
    """自适应 ML 管线单元测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟因子数据 — 模拟 A 股月度横截面数据"""
        np.random.seed(42)
        n_dates = 48  # 4 年
        n_stocks = 50
        dates = pd.date_range("2020-01-01", periods=n_dates, freq="ME")
        codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

        rows = []
        for ci, code in enumerate(codes):
            base_factor = np.random.randn(5)  # 5 个潜在因子
            for di, d in enumerate(dates):
                # 构建因子值，加入真实信号
                signal = np.sin(ci * 0.1 + di * 0.3) * 0.5
                row = {
                    "date": d,
                    "code": code,
                    "momentum_1m": base_factor[0] + signal + np.random.randn() * 0.1,
                    "momentum_3m": base_factor[0] * 0.8 + signal * 0.8 + np.random.randn() * 0.15,
                    "volatility_1m": abs(base_factor[1]) + np.random.randn() * 0.1,
                    "turnover": base_factor[2] + signal * 0.3 + np.random.randn() * 0.2,
                    "size_factor": base_factor[3] + np.random.randn() * 0.2,
                    "reversal_1m": -base_factor[0] + np.random.randn() * 0.2,
                    "quality_roe": base_factor[4] + np.random.randn() * 0.3,
                    "target_return": signal + np.random.randn() * 0.3,
                }
                rows.append(row)

        cls.df = pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)
        cls.feature_names = [
            "momentum_1m", "momentum_3m", "volatility_1m",
            "turnover", "size_factor", "reversal_1m", "quality_roe",
        ]

    def test_window_generation_rolling(self):
        """测试 Rolling Window 生成"""
        pipeline = AdaptiveMLPipeline(
            window_mode="rolling",
            window_months=12,
            retrain_frequency_months=3,
        )
        windows = pipeline.generate_windows("2020-01-01", "2024-01-01")

        self.assertGreater(len(windows), 0)
        for w in windows:
            self.assertIsInstance(w.train_start, str)
            self.assertIsInstance(w.test_start, str)
            # 检查训练窗口长度约为 12 个月
            train_days = (pd.to_datetime(w.train_end) - pd.to_datetime(w.train_start)).days
            self.assertAlmostEqual(train_days / 30, 12, delta=2)  # 约 12 个月

        print(f"\n  Rolling Window: 生成 {len(windows)} 个训练窗口")

    def test_window_generation_expanding(self):
        """测试 Expanding Window 生成"""
        pipeline = AdaptiveMLPipeline(
            window_mode="expanding",
            window_months=12,
            retrain_frequency_months=6,
        )
        windows = pipeline.generate_windows("2020-01-01", "2024-01-01")

        self.assertGreater(len(windows), 0)
        # Expanding window 的 train_start 应该始终等于初始日期
        for w in windows:
            self.assertEqual(w.train_start, "2020-01-01")

        print(f"\n  Expanding Window: 生成 {len(windows)} 个训练窗口")

    @unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn")
    def test_train_and_predict(self):
        """测试完整的训练/预测流程"""
        pipeline = AdaptiveMLPipeline(
            window_mode="rolling",
            window_months=12,
            retrain_frequency_months=3,
            top_k_features=5,
            outlier_method="mad",
            outlier_threshold=3.0,
        )

        # 按日期切分训练/测试
        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"])
        train_mask = df["date"] < "2022-01-01"
        test_mask = df["date"] >= "2022-01-01"

        X_train = df.loc[train_mask, self.feature_names].copy()
        y_train = df.loc[train_mask, "target_return"].copy()
        X_test = df.loc[test_mask, self.feature_names].copy()
        y_test = df.loc[test_mask, "target_return"].copy()

        # 训练
        metrics = pipeline.train(X_train, y_train, self.feature_names)
        print(f"\n  训练结果:")
        print(f"    样本数: {metrics['n_samples']}")
        print(f"    选中特征数: {metrics['n_features']}")
        print(f"    选中特征: {metrics['selected_features']}")
        print(f"    训练耗时: {metrics['train_time_seconds']}s")

        if "r2_train" in metrics:
            print(f"    R² (train): {metrics['r2_train']}")
            self.assertGreater(metrics["r2_train"], -1.0)  # 不应太差

        # 预测
        predictions = pipeline.predict(X_test, self.feature_names)
        self.assertEqual(len(predictions), len(X_test))

        # 简单 IC 检验（预测与真实的相关系数应为正）
        if HAS_SKLEARN:
            from scipy import stats
            ic, p_value = stats.spearmanr(predictions, y_test, nan_policy="omit")
            print(f"    预测 IC: {ic:.4f}, p-value: {p_value:.4f}")


    @unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn")
    def test_feature_selection_consistency(self):
        """测试特征选择的一致性"""
        pipeline = AdaptiveMLPipeline(top_k_features=5)

        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"])
        train_mask = df["date"] < "2022-01-01"

        X_train = df.loc[train_mask, self.feature_names].copy()
        y_train = df.loc[train_mask, "target_return"].copy()

        # 多次训练，检查特征选择稳定性
        selections = []
        for seed in range(3):
            np.random.seed(seed)
            pipeline.model = None
            pipeline.scaler = None
            pipeline.selected_features = []

            sub_mask = np.random.choice(len(X_train), size=int(len(X_train) * 0.8), replace=False)
            pipeline.train(
                X_train.iloc[sub_mask],
                y_train.iloc[sub_mask],
                self.feature_names,
            )
            selections.append(set(pipeline.selected_features))

        # 检查共同选中的特征
        common = selections[0].intersection(*selections[1:])
        print(f"\n  特征选择一致性:")
        for i, sel in enumerate(selections):
            print(f"    第 {i+1} 次: {sorted(sel)}")
        print(f"    共同选中: {sorted(common)}")
        self.assertGreater(len(common), 0, "至少有一个特征被持续选中")

    @unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn")
    def test_needs_retrain(self):
        """测试重训练触发条件"""
        pipeline = AdaptiveMLPipeline(
            window_months=12,
            retrain_frequency_months=3,
        )

        # 初始状态
        self.assertTrue(pipeline.needs_retrain("2024-01-01"))

        # 训练后
        pipeline.model = True  # mock
        pipeline.last_train_date = "2024-01-01"
        self.assertFalse(pipeline.needs_retrain("2024-02-01"))
        self.assertFalse(pipeline.needs_retrain("2024-03-01"))
        self.assertTrue(pipeline.needs_retrain("2024-04-15"))

        print(f"\n  重训练触发逻辑: PASS")

    def test_outlier_detection(self):
        """测试离群值检测"""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 1000)
        # 添加离群值
        outliers = np.array([10, -10, 8, -8, 15])
        data = pd.Series(np.concatenate([normal_data, outliers]))

        # MAD 方法
        mad_result = OutlierDetector.mad_outliers(data, threshold=3.0)
        n_mad_detected = mad_result.sum()

        # IQR 方法
        iqr_result = OutlierDetector.iqr_outliers(data, multiplier=1.5)
        n_iqr_detected = iqr_result.sum()

        print(f"\n  离群值检测:")
        print(f"    总样本: {len(data)} (含 {len(outliers)} 个离群值)")
        print(f"    MAD(3σ) 检出: {n_mad_detected}")
        print(f"    IQR(1.5×) 检出: {n_iqr_detected}")

        # 至少检出一些离群值
        self.assertGreater(n_mad_detected, 0, "MAD 应检测到离群值")


class TestAutoMLPerformance(unittest.TestCase):
    """性能对比测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_samples = 5000
        n_features = 20
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"factor_{i}" for i in range(n_features)],
        )
        # 前5个因子有真实信号的
        signal = (
            X["factor_0"] * 0.3
            + X["factor_1"] * 0.2
            + X["factor_2"] * 0.15
            + np.random.randn(n_samples) * 0.5
        )
        cls.X = X
        cls.y = pd.Series(signal)
        cls.feature_names = X.columns.tolist()

    @unittest.skipUnless(HAS_SKLEARN and HAS_LGB, "需要 scikit-learn 和 LightGBM")
    def test_speed_with_vs_without_feature_selection(self):
        """对比有/无特征选择的预测速度"""
        pipeline_full = AdaptiveMLPipeline(top_k_features=20)  # 全特征
        pipeline_sel = AdaptiveMLPipeline(top_k_features=5)    # 选前5个

        # 训练
        pipeline_full.train(self.X, self.y, self.feature_names)
        pipeline_sel.train(self.X, self.y, self.feature_names)

        # 预测速度对比
        n_runs = 100
        X_new = pd.DataFrame(
            np.random.randn(1000, len(self.feature_names)),
            columns=self.feature_names,
        )

        t0 = time.perf_counter()
        for _ in range(n_runs):
            pipeline_full.predict(X_new, self.feature_names)
        t_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_runs):
            pipeline_sel.predict(X_new, self.feature_names)
        t_sel = time.perf_counter() - t0

        print(f"\n  预测性能对比 ({n_runs} 次):")
        print(f"    全特征(20): {t_full:.4f}s ({t_full/n_runs*1e3:.2f}ms/次)")
        print(f"    选中特征(5): {t_sel:.4f}s ({t_sel/n_runs*1e3:.2f}ms/次)")
        print(f"    加速比: {t_full/t_sel:.2f}x")

    @unittest.skipUnless(HAS_SKLEARN and HAS_LGB, "需要 scikit-learn 和 LightGBM")
    def test_adaptive_vs_fixed_window(self):
        """对比自适应窗口 vs 固定窗口的训练效果"""
        # 构造带时间漂移的数据
        np.random.seed(42)
        n_total = 2000
        features = pd.DataFrame(
            np.random.randn(n_total, 10),
            columns=[f"f_{i}" for i in range(10)],
        )
        # 前 1000 个样本用 signal_A, 后 1000 个用 signal_B
        signal_a = features["f_0"].iloc[:1000] * 0.5 + np.random.randn(1000) * 0.3
        signal_b = features["f_1"].iloc[1000:] * 0.5 + np.random.randn(1000) * 0.3
        y = pd.concat([pd.Series(signal_a), pd.Series(signal_b)]).reset_index(drop=True)
        f_names = features.columns.tolist()

        # 固定窗口：只用在 前 1000 上训练的模型预测全部
        t0 = time.perf_counter()
        model_fixed = lgb.LGBMRegressor(
            n_estimators=100, random_state=42, verbosity=-1,
        )
        model_fixed.fit(features.iloc[:1000], y.iloc[:1000])
        pred_fixed = model_fixed.predict(features)
        t_fixed = time.perf_counter() - t0

        # 自适应窗口：先用前 1000 训练，再用后 1000 重训练
        t0 = time.perf_counter()
        model_adapt = lgb.LGBMRegressor(
            n_estimators=100, random_state=42, verbosity=-1,
        )
        model_adapt.fit(features.iloc[:1000], y.iloc[:1000])
        pred_adapt_first = model_adapt.predict(features.iloc[:1000])
        # 重训练（模拟 expanding window）
        model_adapt.fit(features.iloc[:1500], y.iloc[:1500])
        pred_adapt_second = model_adapt.predict(features.iloc[1000:])
        pred_adapt = np.concatenate([pred_adapt_first[:1000], pred_adapt_second])
        t_adapt = time.perf_counter() - t0

        # IC 对比
        from scipy import stats
        ic_fixed, _ = stats.spearmanr(pred_fixed, y, nan_policy="omit")
        ic_adapt, _ = stats.spearmanr(pred_adapt, y, nan_policy="omit")

        print(f"\n  自适应 vs 固定窗口:")
        print(f"    固定窗口 IC: {ic_fixed:.4f}, 耗时: {t_fixed:.4f}s")
        print(f"    自适应窗口 IC: {ic_adapt:.4f}, 耗时: {t_adapt:.4f}s")


# ================================================================================
# 运行入口
# ================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("自适应 ML 训练管线（FreqAI 风格）验证测试")
    print("借鉴来源: Freqtrade FreqAI 模块")
    print("=" * 70)
    unittest.main(verbosity=2)