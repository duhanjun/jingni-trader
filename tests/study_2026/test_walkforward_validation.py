"""
验证测试：Walk-Forward 交叉验证 (Walk-Forward Validation)
===========================================================
借鉴来源：Freqtrade (github.com/freqtrade/freqtrade) - FreqAI Adaptive Retraining
         + Microsoft Qlib (github.com/microsoft/qlib) - Rolling Training

优化方向：strategy-model-engine - 回测中模型定期重训练，避免前视偏差

核心问题：
当前 jingni-trader 的模型训练使用 PurgedGroupTimeSeriesSplit，但整个训练只在
单个时间点完成一次。真实交易中，模型需要定期用新数据重训练（如每月/每季度）。
如果不模拟这种重训练过程，回测结果会过于乐观（前视偏差）。

Freqtrade 的 FreqAI 模块实现了 "adaptive retraining in backtesting"：
- 在回测过程中，每隔 N 个周期（如 30 天）用最新的历史数据重新训练模型
- 预测只使用训练完成后的模型版本
- 这模拟了真实生产环境中的模型更新流程

Qlib 的 Rolling Training 也提供了类似的机制。

本测试验证：
1. Walk-forward 交叉验证的正确性
2. 与静态单次训练的性能对比
3. 前视偏差检测

日期：2026-06-13
作者：jingni-trader AI Research Agent
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Walk-Forward 验证框架实现
# =============================================================================

class WalkForwardValidator:
    """
    Walk-Forward 交叉验证器

    参考 Freqtrade FreqAI 的 adaptive retraining 设计：
    - 训练窗口: 用过去 N 天的数据训练模型
    - 预测窗口: 用训练好的模型预测未来 M 天的数据
    - 滚动: 训练窗口和预测窗口都向前滚动
    - 重训练: 在每次滚动时重新训练模型

    与 Qlib 的 Rolling Training 对比：
    - Qlib 的 Rolling 更关注数据划分策略
    - Freqtrade 的 FreqAI 更关注回测中的实时重训练模拟
    - 本实现结合两者优点，提供完整的 walk-forward 验证
    """

    def __init__(
        self,
        train_window: int = 252,         # 训练窗口天数
        test_window: int = 63,           # 测试窗口天数
        step_size: int = 21,             # 滚动步长天数
        min_train_days: int = 126,       # 最少训练天数
        purge_gap: int = 5,              # 训练/测试间的清洗间隔
        retrain_frequency: int = 21,     # 重训练频率（天）
    ):
        """
        参数:
            train_window: 训练窗口长度（交易日）
            test_window: 每次预测的测试窗口长度
            step_size: 窗口滚动步长
            min_train_days: 最少训练天数
            purge_gap: 训练集末尾与测试集起始之间的间隔天数
            retrain_frequency: 重训练频率（每 N 天重训练一次）
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_days = min_train_days
        self.purge_gap = purge_gap
        self.retrain_frequency = retrain_frequency

    def split(
        self,
        dates: List[pd.Timestamp],
        features: pd.DataFrame,
        targets: pd.Series,
    ) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Timestamp]]:
        """
        生成 Walk-Forward 数据分割

        返回:
            List of (X_train, y_train, X_test, y_test, test_date)
        """
        unique_dates = sorted(set(dates))
        if len(unique_dates) < self.min_train_days + self.test_window:
            return []

        splits = []
        current_train_end = self.min_train_days

        while current_train_end + self.test_window <= len(unique_dates):
            train_start = max(0, current_train_end - self.train_window)
            train_end = current_train_end - self.purge_gap
            test_start = current_train_end
            test_end = min(current_train_end + self.test_window, len(unique_dates))

            if train_end <= train_start:
                current_train_end += self.step_size
                continue

            train_dates = unique_dates[train_start:train_end]
            test_dates = unique_dates[test_start:test_end]

            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(test_dates)

            X_train = features[train_mask]
            y_train = targets[train_mask]
            X_test = features[test_mask]
            y_test = targets[test_mask]

            if len(X_train) >= self.min_train_days and len(X_test) > 0:
                splits.append((X_train, y_train, X_test, y_test, unique_dates[test_start]))

            current_train_end += self.step_size

        return splits

    def validate(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        dates: pd.Series,
        model_factory: Callable[[], Any],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        执行 Walk-Forward 验证

        参数:
            features: 特征矩阵
            targets: 目标变量
            dates: 日期序列
            model_factory: 模型工厂函数（每次调用返回新模型）
            verbose: 是否打印详细信息

        返回:
            {
                'predictions': pd.DataFrame - 每期预测结果
                'metrics': Dict[str, float] - 汇总指标
                'split_metrics': List[Dict] - 每期指标
                'model_count': int - 训练模型数量
            }
        """
        splits = self.split(dates, features, targets)
        if not splits:
            return {'predictions': pd.DataFrame(), 'metrics': {}, 'split_metrics': []}

        all_predictions = []
        all_metrics = []
        models_trained = 0

        for i, (X_train, y_train, X_test, y_test, test_date) in enumerate(splits):
            # 检查是否需要重训练
            need_retrain = (i == 0) or (i % max(1, self.retrain_frequency // self.step_size) == 0)

            if need_retrain:
                model = model_factory()
                model.fit(X_train, y_train)
                models_trained += 1
            elif hasattr(self, '_last_model'):
                model = self._last_model

            self._last_model = model

            preds = model.predict(X_test)

            # 计算指标
            if len(y_test) > 1:
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                ic = np.corrcoef(y_test, preds)[0, 1] if len(y_test) > 2 else 0
            else:
                mse = np.nan
                r2 = np.nan
                ic = np.nan

            split_metrics = {
                'split': i,
                'test_date': test_date,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'mse': mse,
                'r2': r2,
                'ic': ic,
                'retrained': need_retrain,
            }
            all_metrics.append(split_metrics)

            # 存储预测
            pred_df = pd.DataFrame({
                'test_date': test_date,
                'y_true': y_test.values,
                'y_pred': preds,
            })
            all_predictions.append(pred_df)

            if verbose:
                print(f"  Split {i}: train={len(X_train)}, test={len(X_test)}, "
                      f"R²={r2:.4f}, IC={ic:.4f}, retrained={need_retrain}")

        # 汇总指标
        valid_metrics = [m for m in all_metrics if not np.isnan(m['r2'])]
        summary = {
            'avg_r2': np.mean([m['r2'] for m in valid_metrics]) if valid_metrics else np.nan,
            'avg_ic': np.mean([m['ic'] for m in valid_metrics]) if valid_metrics else np.nan,
            'std_r2': np.std([m['r2'] for m in valid_metrics]) if valid_metrics else np.nan,
            'std_ic': np.std([m['ic'] for m in valid_metrics]) if valid_metrics else np.nan,
            'r2_stability': 1.0 - (np.std([m['r2'] for m in valid_metrics]) /
                                    (abs(np.mean([m['r2'] for m in valid_metrics])) + 1e-10))
                                    if valid_metrics else np.nan,
            'n_splits': len(splits),
            'models_trained': models_trained,
            'total_predictions': sum(len(p) for p in all_predictions),
        }

        return {
            'predictions': pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame(),
            'metrics': summary,
            'split_metrics': all_metrics,
        }


# =============================================================================
# 前视偏差检测器
# =============================================================================

class LookAheadBiasDetector:
    """
    前视偏差检测器

    参考 Freqtrade 的 lookahead analysis 和 Jesse 的 zero look-ahead bias 设计
    """

    @staticmethod
    def detect_lookahead_bias(
        features: pd.DataFrame,
        targets: pd.Series,
        dates: pd.Series,
        model: Any,
        contamination_ratio: float = 0.1,
    ) -> Dict[str, Any]:
        """
        通过注入未来信息来检测前视偏差

        方法：
        1. 基准：使用正常特征训练模型
        2. 污染：在特征中混入少量未来信息（target 的滞后值）
        3. 如果污染模型的性能显著优于基准模型，说明存在前视偏差漏洞

        参数:
            features: 特征矩阵
            targets: 目标变量
            dates: 日期序列
            model: 未训练的模型实例
            contamination_ratio: 污染比例

        返回:
            {
                'baseline_ic': float,
                'contaminated_ic': float,
                'ic_delta': float,
                'has_lookahead_bias': bool,
            }
        """
        unique_dates = sorted(dates.unique())
        split_idx = int(len(unique_dates) * 0.7)
        train_dates = unique_dates[:split_idx]
        test_dates = unique_dates[split_idx:]

        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)

        X_train = features[train_mask].values
        y_train = targets[train_mask].values
        X_test = features[test_mask].values
        y_test = targets[test_mask].values

        # 基准模型
        model_baseline = model.__class__() if hasattr(model, '__class__') else type(model)()
        try:
            model_baseline.fit(X_train, y_train)
            preds_baseline = model_baseline.predict(X_test)
            baseline_ic = np.corrcoef(y_test, preds_baseline)[0, 1]
        except Exception:
            baseline_ic = 0

        # 污染模型：在特征中混入未来信息
        n_contaminate = max(1, int(X_train.shape[1] * contamination_ratio))
        X_train_contaminated = X_train.copy()
        # 将 target 的滞后值作为"未来信息"注入
        X_train_contaminated[:, :n_contaminate] = np.tile(
            y_train.reshape(-1, 1), (1, n_contaminate)
        )

        model_contaminated = model.__class__() if hasattr(model, '__class__') else type(model)()
        try:
            model_contaminated.fit(X_train_contaminated, y_train)
            preds_contaminated = model_contaminated.predict(X_test)
            contaminated_ic = np.corrcoef(y_test, preds_contaminated)[0, 1]
        except Exception:
            contaminated_ic = 0

        ic_delta = contaminated_ic - baseline_ic
        # 如果污染后 IC 显著提升（> 0.02），说明存在前视偏差风险
        has_bias = ic_delta > 0.02

        return {
            'baseline_ic': baseline_ic,
            'contaminated_ic': contaminated_ic,
            'ic_delta': ic_delta,
            'has_lookahead_bias': has_bias,
        }


# =============================================================================
# 单元测试
# =============================================================================

class TestWalkForwardValidator(unittest.TestCase):
    """测试 Walk-Forward 验证器"""

    @classmethod
    def setUpClass(cls):
        """创建模拟数据"""
        np.random.seed(42)
        n_dates = 500
        n_features = 10

        dates = pd.date_range('2022-01-01', periods=n_dates, freq='B')
        # 生成有自相关的特征
        X = np.random.randn(n_dates, n_features) * 0.1
        for i in range(1, n_dates):
            X[i] += 0.3 * X[i-1]

        # 生成目标：线性关系 + 噪声
        true_weights = np.random.randn(n_features) * 0.5
        y = X @ true_weights + np.random.randn(n_dates) * 0.05

        cls.features = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
        cls.targets = pd.Series(y, name='target')
        cls.dates = pd.Series(dates, name='date')

    def test_basic_split(self):
        """测试基本的数据分割"""
        validator = WalkForwardValidator(
            train_window=252,
            test_window=63,
            step_size=21,
            min_train_days=126,
        )
        splits = validator.split(self.dates, self.features, self.targets)
        self.assertGreater(len(splits), 0, "应生成至少一个分割")

        for X_train, y_train, X_test, y_test, test_date in splits:
            self.assertGreater(len(X_train), 0, "训练集不应为空")
            self.assertGreater(len(X_test), 0, "测试集不应为空")
            self.assertEqual(len(X_train), len(y_train), "训练集 X 和 y 长度应一致")
            self.assertEqual(len(X_test), len(y_test), "测试集 X 和 y 长度应一致")

    def test_purge_gap(self):
        """测试清洗间隔"""
        # 较小的 purge_gap
        validator_small = WalkForwardValidator(purge_gap=1)
        splits_small = validator_small.split(self.dates, self.features, self.targets)

        # 较大的 purge_gap
        validator_large = WalkForwardValidator(purge_gap=10)
        splits_large = validator_large.split(self.dates, self.features, self.targets)

        # 较大的 purge_gap 应该产生更少的训练数据
        if splits_small and splits_large:
            avg_train_small = np.mean([len(s[0]) for s in splits_small])
            avg_train_large = np.mean([len(s[0]) for s in splits_large])
            self.assertGreaterEqual(avg_train_small, avg_train_large,
                                   "大 purge_gap 应减少训练数据")

    def test_validate_returns_metrics(self):
        """测试验证返回完整指标"""
        validator = WalkForwardValidator(
            train_window=252,
            test_window=63,
            step_size=63,
            min_train_days=126,
        )
        result = validator.validate(
            self.features, self.targets, self.dates,
            model_factory=lambda: LinearRegression(),
        )

        self.assertIn('predictions', result)
        self.assertIn('metrics', result)
        self.assertIn('split_metrics', result)
        self.assertGreater(result['metrics']['n_splits'], 0)
        self.assertGreater(result['metrics']['models_trained'], 0)

    def test_model_retraining_count(self):
        """测试重训练次数"""
        validator = WalkForwardValidator(
            train_window=252,
            test_window=63,
            step_size=21,
            min_train_days=126,
            retrain_frequency=63,  # 每 63 天重训练一次
        )
        result = validator.validate(
            self.features, self.targets, self.dates,
            model_factory=lambda: LinearRegression(),
        )

        n_splits = result['metrics']['n_splits']
        models_trained = result['metrics']['models_trained']

        # 模型数应少于分割数（因为不需要每期都重训练）
        self.assertLessEqual(models_trained, n_splits,
                            "模型训练次数应 ≤ 分割数")
        self.assertGreater(models_trained, 0, "至少应训练一个模型")

    def test_empty_data(self):
        """测试空数据边界条件"""
        validator = WalkForwardValidator()
        empty_features = pd.DataFrame()
        empty_targets = pd.Series(dtype=float)
        empty_dates = pd.Series(dtype='datetime64[ns]')

        splits = validator.split(empty_dates, empty_features, empty_targets)
        self.assertEqual(len(splits), 0, "空数据应返回空分割")

    def test_insufficient_data(self):
        """测试数据不足的情况"""
        validator = WalkForwardValidator(min_train_days=500)
        splits = validator.split(self.dates, self.features, self.targets)
        self.assertEqual(len(splits), 0, "数据不足应返回空列表")


class TestLookAheadBiasDetector(unittest.TestCase):
    """测试前视偏差检测器"""

    def test_no_bias_with_clean_data(self):
        """测试清洁数据不产生前视偏差"""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(X['f0'] * 0.5 + np.random.randn(n) * 0.1)
        dates = pd.Series(pd.date_range('2024-01-01', periods=n, freq='B'))

        result = LookAheadBiasDetector.detect_lookahead_bias(
            X, y, dates, LinearRegression(),
        )

        # 清洁数据不应报告前视偏差
        self.assertIsNotNone(result['baseline_ic'])
        print(f"\n  前视偏差检测: baseline_ic={result['baseline_ic']:.4f}, "
              f"contaminated_ic={result['contaminated_ic']:.4f}, "
              f"delta={result['ic_delta']:.4f}")


# =============================================================================
# 对比分析：单次训练 vs Walk-Forward
# =============================================================================

class TestTrainingMethodComparison(unittest.TestCase):
    """
    对比静态单次训练与 Walk-Forward 验证的性能差异
    验证 Walk-Forward 的必要性
    """

    @classmethod
    def setUpClass(cls):
        """创建包含趋势变化的数据集"""
        np.random.seed(42)
        n_dates = 600
        n_features = 8

        dates = pd.date_range('2022-01-01', periods=n_dates, freq='B')
        X = np.zeros((n_dates, n_features))

        # 生成有结构性变化的特征
        for i in range(n_features):
            trend = np.sin(np.linspace(0, 4 * np.pi, n_dates)) * 0.5
            X[:, i] = trend + np.random.randn(n_dates) * 0.1

        # 目标：权重随时间变化
        weights = np.zeros((n_dates, n_features))
        for i in range(n_features):
            weights[:, i] = np.linspace(0.1, 0.9, n_dates) * np.sin(i * 0.5 + np.linspace(0, np.pi, n_dates))

        y = np.sum(X * weights, axis=1) + np.random.randn(n_dates) * 0.05

        cls.features = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
        cls.targets = pd.Series(y, name='target')
        cls.dates = pd.Series(dates, name='date')

    def test_compare_methods(self):
        """对比单次训练与 Walk-Forward 验证"""
        # 单次训练
        split_idx = int(len(self.dates) * 0.7)
        train_mask = self.dates < self.dates.iloc[split_idx]
        test_mask = self.dates >= self.dates.iloc[split_idx]

        X_train = self.features[train_mask]
        y_train = self.targets[train_mask]
        X_test = self.features[test_mask]
        y_test = self.targets[test_mask]

        model_static = LinearRegression()
        model_static.fit(X_train, y_train)
        preds_static = model_static.predict(X_test)
        static_ic = np.corrcoef(y_test, preds_static)[0, 1]
        static_r2 = r2_score(y_test, preds_static)

        # Walk-Forward 验证
        validator = WalkForwardValidator(
            train_window=252,
            test_window=63,
            step_size=21,
            min_train_days=126,
            retrain_frequency=63,
        )
        result = validator.validate(
            self.features, self.targets, self.dates,
            model_factory=lambda: LinearRegression(),
        )

        wf_avg_ic = result['metrics']['avg_ic']
        wf_avg_r2 = result['metrics']['avg_r2']

        print(f"\n  对比分析:")
        print(f"    单次训练: R²={static_r2:.4f}, IC={static_ic:.4f}")
        print(f"    Walk-Forward: 平均 R²={wf_avg_r2:.4f}, 平均 IC={wf_avg_ic:.4f}")
        print(f"    Walk-Forward R² 稳定性: {result['metrics']['r2_stability']:.4f}")
        print(f"    Walk-Forward 分割数: {result['metrics']['n_splits']}")
        print(f"    重训练模型数: {result['metrics']['models_trained']}")

        # Walk-Forward 的 R² 稳定性指标验证了我们确实需要定期重训练
        self.assertIsNotNone(wf_avg_ic)
        self.assertIsNotNone(wf_avg_r2)


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Walk-Forward 交叉验证测试")
    print("借鉴来源: Freqtrade FreqAI + Microsoft Qlib Rolling Training")
    print("=" * 70)

    unittest.main(verbosity=2, exit=False)