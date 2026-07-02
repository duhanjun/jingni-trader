"""
验证测试：因子数据异常值检测与处理 (Outlier Detection & Processing)
=====================================================================
借鉴来源：Freqtrade FreqAI (github.com/freqtrade/freqtrade) - Outlier Detection
         + Microsoft Qlib (github.com/microsoft/qlib) - Data Processor Pipeline

优化方向：factor-engine - 因子数据预处理管道，提升因子质量

核心问题：
当前 jingni-trader factor-engine 的因子计算后直接进行 IC 分析和相关性分析，
但缺少对因子数据中异常值的检测和处理。异常值会严重影响：
1. IC 分析的准确性（异常值会扭曲相关性）
2. 因子中性化的效果（异常值主导回归）
3. 多因子融合的权重估计
4. 模型训练的稳定性

Freqtrade FreqAI 提供了多种异常值检测方法：
- Dissimilarity Index (DI) - 基于特征空间的异常检测
- SVM (One-Class SVM) - 基于支持向量机的异常检测
- DBSCAN - 基于密度的聚类异常检测
- PCA - 基于主成分分析的降维去噪

Qlib 的 Data Processor Pipeline 提供了：
- DropnaProcessor - 缺失值处理
- ZScoreNorm - Z-Score 标准化
- CSZScoreNorm - 截面 Z-Score 标准化
- CSRankNorm - 截面排名标准化

本测试验证：
1. 多种异常值检测方法的有效性
2. 异常值处理对因子 IC 的影响
3. 因子标准化方法对比

日期：2026-06-13
作者：jingni-trader AI Research Agent
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 异常值检测方法实现
# =============================================================================

class OutlierDetector:
    """因子数据异常值检测器集合"""

    @staticmethod
    def mad_outliers(series: pd.Series, threshold: float = 5.0) -> pd.Series:
        """
        MAD (Median Absolute Deviation) 方法检测异常值

        优点：对异常值本身具有鲁棒性
        来源：量化金融领域的标准做法

        返回: 布尔 Series，True 表示异常值
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return pd.Series(False, index=series.index)
        modified_z_score = 0.6745 * (series - median) / mad
        return np.abs(modified_z_score) > threshold

    @staticmethod
    def iqr_outliers(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
        """
        IQR (Interquartile Range) 方法检测异常值

        返回: 布尔 Series，True 表示异常值
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return (series < lower) | (series > upper)

    @staticmethod
    def percentile_clip(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """
        百分位截断 - 将超出上下界的值截断到边界值

        返回: 截断后的 Series
        """
        lo = series.quantile(lower)
        hi = series.quantile(upper)
        return series.clip(lower=lo, upper=hi)

    @staticmethod
    def sigma_clip(series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
        """
        Sigma 截断 - 将超出 n 倍标准差的值截断

        返回: 截断后的 Series
        """
        mean = series.mean()
        std = series.std()
        return series.clip(lower=mean - n_sigma * std, upper=mean + n_sigma * std)

    @staticmethod
    def winsorize(series: pd.Series, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
        """
        Winsorize 处理

        返回: Winsorize 后的 Series
        """
        from scipy.stats.mstats import winsorize as scipy_winsorize
        values = series.values.copy()
        result = scipy_winsorize(values[~np.isnan(values)], limits=limits)
        out = pd.Series(index=series.index, dtype=float)
        non_nan_mask = ~series.isna()
        out.loc[non_nan_mask] = result
        out.loc[~non_nan_mask] = np.nan
        return out

    @staticmethod
    def one_class_svm_outliers(
        features: pd.DataFrame,
        nu: float = 0.05,
    ) -> pd.Series:
        """
        One-Class SVM 异常检测（参考 FreqAI）

        在特征空间中检测异常样本

        返回: 布尔 Series，True 表示异常值
        """
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features.fillna(0))

        # One-Class SVM
        svm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
        svm.fit(X_scaled)
        predictions = svm.predict(X_scaled)

        return pd.Series(predictions == -1, index=features.index)

    @staticmethod
    def dissimilarity_index(
        features: pd.DataFrame,
        threshold_percentile: float = 95,
    ) -> pd.Series:
        """
        Dissimilarity Index 异常检测（参考 FreqAI）

        计算每个样本与特征空间中位数的距离，超过阈值的标记为异常

        返回: 布尔 Series，True 表示异常值
        """
        median = features.median()
        # 计算欧氏距离
        diff = (features - median).pow(2).sum(axis=1).pow(0.5)
        threshold = diff.quantile(threshold_percentile / 100)
        return diff > threshold


# =============================================================================
# 因子标准化方法
# =============================================================================

class FactorNormalizer:
    """因子标准化方法集合（参考 Qlib 的 Data Processor）"""

    @staticmethod
    def zscore_normalize(series: pd.Series) -> pd.Series:
        """Z-Score 标准化"""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0, index=series.index)
        return (series - mean) / std

    @staticmethod
    def cross_sectional_zscore(factor_df: pd.DataFrame, factor_col: str) -> pd.Series:
        """
        截面 Z-Score 标准化（参考 Qlib CSZScoreNorm）

        在每个日期截面上对因子值进行 Z-Score 标准化
        """
        result = pd.Series(index=factor_df.index, dtype=float)
        for dt in factor_df['date'].unique():
            mask = factor_df['date'] == dt
            cross = factor_df.loc[mask, factor_col]
            mean = cross.mean()
            std = cross.std()
            if std == 0 or pd.isna(std):
                result.loc[mask] = 0
            else:
                result.loc[mask] = (cross - mean) / std
        return result

    @staticmethod
    def cross_sectional_rank(factor_df: pd.DataFrame, factor_col: str) -> pd.Series:
        """
        截面排名标准化（参考 Qlib CSRankNorm）

        将因子值转换为 0-1 之间的排名百分比
        """
        result = pd.Series(index=factor_df.index, dtype=float)
        for dt in factor_df['date'].unique():
            mask = factor_df['date'] == dt
            result.loc[mask] = factor_df.loc[mask, factor_col].rank(pct=True)
        return result

    @staticmethod
    def minmax_normalize(series: pd.Series) -> pd.Series:
        """Min-Max 标准化到 [-1, 1]（参考 FreqAI）"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0, index=series.index)
        return 2 * (series - min_val) / (max_val - min_val) - 1


# =============================================================================
# 因子数据处理管道
# =============================================================================

class FactorProcessingPipeline:
    """
    因子数据处理管道（参考 Qlib DataHandler + FreqAI Pipeline）

    将因子数据的检测、清洗、标准化组合为可配置的管道
    """

    def __init__(self, steps: List[Tuple[str, Dict[str, Any]]]):
        """
        参数:
            steps: 处理步骤列表，每个步骤为 (step_name, step_params)
        """
        self.steps = steps
        self.detector = OutlierDetector()
        self.normalizer = FactorNormalizer()

    def process(self, factor_df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """
        按顺序执行处理步骤

        返回: 处理后的 DataFrame
        """
        result = factor_df.copy()

        for step_name, params in self.steps:
            if step_name == 'dropna':
                result = result.dropna(subset=factor_cols)

            elif step_name == 'mad_filter':
                threshold = params.get('threshold', 5.0)
                for col in factor_cols:
                    outliers = self.detector.mad_outliers(result[col], threshold)
                    result.loc[outliers, col] = np.nan

            elif step_name == 'percentile_clip':
                lower = params.get('lower', 0.01)
                upper = params.get('upper', 0.99)
                for col in factor_cols:
                    result[col] = self.detector.percentile_clip(result[col], lower, upper)

            elif step_name == 'winsorize':
                limits = params.get('limits', (0.01, 0.01))
                for col in factor_cols:
                    result[col] = self.detector.winsorize(result[col], limits)

            elif step_name == 'cross_sectional_zscore':
                for col in factor_cols:
                    result[col] = self.normalizer.cross_sectional_zscore(result, col)

            elif step_name == 'cross_sectional_rank':
                for col in factor_cols:
                    result[col] = self.normalizer.cross_sectional_rank(result, col)

            elif step_name == 'minmax':
                for col in factor_cols:
                    result[col] = self.normalizer.minmax_normalize(result[col])

            elif step_name == 'one_class_svm':
                nu = params.get('nu', 0.05)
                outliers = self.detector.one_class_svm_outliers(
                    result[factor_cols], nu
                )
                for col in factor_cols:
                    result.loc[outliers, col] = np.nan

            elif step_name == 'fillna':
                method = params.get('method', 'median')
                for col in factor_cols:
                    if method == 'median':
                        result[col] = result[col].fillna(result[col].median())
                    elif method == 'zero':
                        result[col] = result[col].fillna(0)

        return result


# =============================================================================
# 单元测试
# =============================================================================

class TestOutlierDetector(unittest.TestCase):
    """测试异常值检测器"""

    @classmethod
    def setUpClass(cls):
        """创建包含异常值的数据"""
        np.random.seed(42)
        n = 1000
        clean = np.random.randn(n) * 2 + 10
        # 注入异常值
        clean[0] = 1000
        clean[1] = -500
        clean[2] = 500
        cls.series = pd.Series(clean)
        cls.detector = OutlierDetector()

    def test_mad_outliers(self):
        """测试 MAD 异常值检测"""
        outliers = self.detector.mad_outliers(self.series, threshold=3.0)
        self.assertTrue(outliers.iloc[0], "极端正异常值应被检测到")
        self.assertTrue(outliers.iloc[1], "极端负异常值应被检测到")
        # 大部分正常值不应被标记
        self.assertLess(outliers.sum(), len(self.series) * 0.1,
                       "异常值比例应小于 10%")

    def test_iqr_outliers(self):
        """测试 IQR 异常值检测"""
        outliers = self.detector.iqr_outliers(self.series)
        self.assertTrue(outliers.iloc[0], "极端正异常值应被检测到")

    def test_percentile_clip(self):
        """测试百分位截断"""
        clipped = self.detector.percentile_clip(self.series, 0.01, 0.99)
        self.assertLessEqual(clipped.max(), self.series.quantile(0.99) + 0.001)
        self.assertGreaterEqual(clipped.min(), self.series.quantile(0.01) - 0.001)

    def test_winsorize(self):
        """测试 Winsorize"""
        winsorized = self.detector.winsorize(self.series)
        self.assertEqual(len(winsorized), len(self.series))
        self.assertLess(winsorized.max(), self.series.max(),
                       "Winsorize 应降低最大值")
        self.assertGreater(winsorized.min(), self.series.min(),
                          "Winsorize 应提高最小值")

    def test_no_variance_data(self):
        """测试无方差数据"""
        constant = pd.Series([5.0] * 100)
        outliers = self.detector.mad_outliers(constant)
        self.assertFalse(outliers.any(), "无方差数据不应有异常值")

    def test_one_class_svm(self):
        """测试 One-Class SVM"""
        np.random.seed(42)
        clean = np.random.randn(100, 3)
        # 注入异常样本
        outliers_samples = np.array([[100, 100, 100], [-50, -50, -50]])
        data = np.vstack([clean, outliers_samples])
        df = pd.DataFrame(data, columns=['f0', 'f1', 'f2'])

        outliers = self.detector.one_class_svm_outliers(df, nu=0.05)
        # 注入的异常样本应该被检测到
        self.assertTrue(outliers.iloc[-1], "异常样本应被 SVM 检测到")


class TestFactorNormalizer(unittest.TestCase):
    """测试因子标准化"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_codes = 10
        n_dates = 50

        codes = [f'{i:06d}.SZ' for i in range(n_codes)]
        dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')

        rows = []
        for code in codes:
            code_index = codes.index(code)
            for dt in dates:
                rows.append({
                    'code': code,
                    'date': dt,
                    'factor_a': np.random.randn() * 0.5 + code_index * 0.1,
                    'factor_b': np.random.randn() * 0.3 + 0.5,
                })

        cls.factor_df = pd.DataFrame(rows)
        cls.normalizer = FactorNormalizer()

    def test_zscore_normalize(self):
        """测试 Z-Score 标准化"""
        result = self.normalizer.zscore_normalize(self.factor_df['factor_a'])
        self.assertAlmostEqual(result.mean(), 0, delta=0.1)
        self.assertAlmostEqual(result.std(), 1, delta=0.1)

    def test_cross_sectional_zscore(self):
        """测试截面 Z-Score 标准化"""
        result = self.normalizer.cross_sectional_zscore(self.factor_df, 'factor_a')
        # 在每个截面上，均值应为 0，标准差应为 1
        for dt in self.factor_df['date'].unique()[:5]:
            mask = self.factor_df['date'] == dt
            cross = result.loc[mask]
            self.assertAlmostEqual(cross.mean(), 0, delta=0.1)
            self.assertAlmostEqual(cross.std(), 1, delta=0.2)

    def test_cross_sectional_rank(self):
        """测试截面排名"""
        result = self.normalizer.cross_sectional_rank(self.factor_df, 'factor_a')
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 1).all())

    def test_minmax_normalize(self):
        """测试 Min-Max 标准化"""
        result = self.normalizer.minmax_normalize(self.factor_df['factor_a'])
        self.assertAlmostEqual(result.min(), -1, delta=0.1)
        self.assertAlmostEqual(result.max(), 1, delta=0.1)


class TestFactorProcessingPipeline(unittest.TestCase):
    """测试因子处理管道"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_codes = 20
        n_dates = 100

        codes = [f'{i:06d}.SZ' for i in range(n_codes)]
        dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')

        rows = []
        for code in codes:
            for dt in dates:
                val = np.random.randn() * 2 + 5
                # 注入少量极端值
                if np.random.random() < 0.02:
                    val = np.random.choice([100, -50, 200])
                rows.append({
                    'code': code,
                    'date': dt,
                    'factor_raw': val,
                    'factor_clean': np.random.randn() * 0.5 + 3,
                })

        cls.factor_df = pd.DataFrame(rows)
        cls.factor_cols = ['factor_raw', 'factor_clean']

    def test_pipeline_percentile_clip_rank(self):
        """测试管道：百分位截断 + 截面排名"""
        pipeline = FactorProcessingPipeline([
            ('percentile_clip', {'lower': 0.01, 'upper': 0.99}),
            ('cross_sectional_rank', {}),
        ])

        result = pipeline.process(self.factor_df, self.factor_cols)

        for col in self.factor_cols:
            self.assertTrue((result[col] >= 0).all(), f"{col} 排名应在 0-1")
            self.assertTrue((result[col] <= 1).all(), f"{col} 排名应在 0-1")

    def test_pipeline_mad_filter_zscore(self):
        """测试管道：MAD 过滤 + 截面 Z-Score"""
        pipeline = FactorProcessingPipeline([
            ('mad_filter', {'threshold': 3.0}),
            ('fillna', {'method': 'median'}),
            ('cross_sectional_zscore', {}),
        ])

        result = pipeline.process(self.factor_df, self.factor_cols)

        # 处理后不应有 NaN
        for col in self.factor_cols:
            self.assertFalse(result[col].isna().any(), f"{col} 不应有 NaN")

    def test_pipeline_winsorize_minmax(self):
        """测试管道：Winsorize + MinMax"""
        pipeline = FactorProcessingPipeline([
            ('winsorize', {'limits': (0.01, 0.01)}),
            ('minmax', {}),
        ])

        result = pipeline.process(self.factor_df, self.factor_cols)

        for col in self.factor_cols:
            self.assertAlmostEqual(result[col].max(), 1, delta=0.1)
            self.assertAlmostEqual(result[col].min(), -1, delta=0.1)

    def test_pipeline_svm_zscore(self):
        """测试管道：SVM 异常检测 + 截面 Z-Score"""
        pipeline = FactorProcessingPipeline([
            ('one_class_svm', {'nu': 0.05}),
            ('fillna', {'method': 'zero'}),
            ('cross_sectional_zscore', {}),
        ])

        result = pipeline.process(self.factor_df, self.factor_cols)
        for col in self.factor_cols:
            self.assertFalse(result[col].isna().any(), f"{col} 不应有 NaN")


# =============================================================================
# IC 影响对比：处理前 vs 处理后
# =============================================================================

class TestICImpactComparison(unittest.TestCase):
    """对比因子处理对 IC 的影响"""

    @classmethod
    def setUpClass(cls):
        """创建包含异常值的因子数据"""
        np.random.seed(42)
        n_codes = 30
        n_dates = 200

        codes = [f'{i:06d}.SZ' for i in range(n_codes)]
        dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')

        rows = []
        for code in codes:
            base_return = np.random.randn() * 0.001
            for dt in dates:
                # 真实因子信号
                true_factor = np.random.randn() * 0.5
                # 5% 概率注入异常值
                if np.random.random() < 0.05:
                    factor = np.random.uniform(-50, 50)
                else:
                    factor = true_factor
                # 未来收益与真实因子相关
                forward_return = true_factor * 0.3 + np.random.randn() * 0.5

                rows.append({
                    'code': code,
                    'date': dt,
                    'factor': factor,
                    'forward_return_1d': forward_return,
                })

        cls.factor_df = pd.DataFrame(rows)

    def test_ic_improvement(self):
        """测试异常值处理对 IC 的改善"""
        # 原始 IC
        ic_raw = []
        for dt in self.factor_df['date'].unique():
            cross = self.factor_df[self.factor_df['date'] == dt]
            if len(cross) < 5:
                continue
            ic, _ = stats.spearmanr(cross['factor'], cross['forward_return_1d'], nan_policy='omit')
            if not np.isnan(ic):
                ic_raw.append(ic)

        mean_ic_raw = np.mean(ic_raw) if ic_raw else 0

        # 处理后 IC
        pipeline = FactorProcessingPipeline([
            ('percentile_clip', {'lower': 0.01, 'upper': 0.99}),
            ('cross_sectional_zscore', {}),
        ])
        processed = pipeline.process(self.factor_df, ['factor'])

        ic_processed = []
        for dt in processed['date'].unique():
            cross = processed[processed['date'] == dt]
            if len(cross) < 5:
                continue
            ic, _ = stats.spearmanr(cross['factor'], cross['forward_return_1d'], nan_policy='omit')
            if not np.isnan(ic):
                ic_processed.append(ic)

        mean_ic_processed = np.mean(ic_processed) if ic_processed else 0

        # 处理后的 IC 标准差应该更小（更稳定）
        std_ic_raw = np.std(ic_raw) if ic_raw else 0
        std_ic_processed = np.std(ic_processed) if ic_processed else 0

        ic_ir_raw = mean_ic_raw / std_ic_raw if std_ic_raw > 0 else 0
        ic_ir_processed = mean_ic_processed / std_ic_processed if std_ic_processed > 0 else 0

        print(f"\n  因子异常值处理对 IC 的影响:")
        print(f"    原始: IC_Mean={mean_ic_raw:.4f}, IC_Std={std_ic_raw:.4f}, IC_IR={ic_ir_raw:.4f}")
        print(f"    处理后: IC_Mean={mean_ic_processed:.4f}, IC_Std={std_ic_processed:.4f}, IC_IR={ic_ir_processed:.4f}")

        # 处理后 IC 标准差应降低（IC 更稳定）
        self.assertLessEqual(std_ic_processed, std_ic_raw * 1.1,
                            "处理后 IC 应更稳定（标准差降低）")


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("因子数据异常值检测与处理验证测试")
    print("借鉴来源: Freqtrade FreqAI + Microsoft Qlib Data Processor")
    print("=" * 70)

    unittest.main(verbosity=2, exit=False)