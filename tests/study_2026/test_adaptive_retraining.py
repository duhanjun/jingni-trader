"""
优化方向: 自适应 ML 模型重训练 — 滑动窗口与概念漂移处理
借鉴来源: Freqtrade FreqAI (https://github.com/freqtrade/freqtrade)
         - adaptive ML module with auto-retraining
         - sliding window continual learning
         - feature parameter configuration
         - model expiration mechanism

优化目标:
  jingni-trader 的 strategy-model-engine 目前使用一次性训练 + Optuna 超参搜索，
  缺乏市场变化适应能力。借鉴 FreqAI 的持续学习设计，实现：
  1. 滑动窗口重训练机制
  2. 模型过期 (性能衰减时自动重训)
  3. 重训练触发条件 (时间窗口、性能阈值)

验证内容:
  1. 滑动窗口 vs 固定窗口的预测性能对比
  2. 模型过期检测及重训练效果
  3. 性能衰减模式识别
  4. 重训练开销分析
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Adaptive ML Retraining Engine (inspired by FreqAI)
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    train_window: int = 60          # 训练窗口 (交易日)
    retrain_interval: int = 10      # 重训练间隔 (交易日)
    min_train_samples: int = 30     # 最少训练样本
    performance_threshold: float = 0.3  # 性能衰减阈值 (MSE 增长超过此比例触发重训)
    validation_window: int = 20     # 验证窗口
    warmup_period: int = 50         # 预热期 (先积累数据)


class ModelCheckpoint:
    """模型检查点"""

    def __init__(self, model, scaler, trained_at: pd.Timestamp,
                 train_mse: float, val_mse: float):
        self.model = model
        self.scaler = scaler
        self.trained_at = trained_at
        self.train_mse = train_mse
        self.val_mse = val_mse


class AdaptiveModel:
    """
    自适应 ML 模型 (滑动窗口 + 模型过期)

    工作机制:
      1. 预热期: 积累数据，首次训练
      2. 正常运行期: 使用模型预测
      3. 重训练触发条件:
         a. 距上次训练超过 retrain_interval
         b. 近期预测性能较训练时衰减超过 performance_threshold
      4. 滑动窗口: 每次重训练只用最近 train_window 的数据
    """

    def __init__(self, config: TrainingConfig, model_cls=Ridge,
                 model_kwargs: Optional[Dict] = None):
        self.config = config
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {'alpha': 1.0}
        self.checkpoint: Optional[ModelCheckpoint] = None
        self.scaler = StandardScaler()

        self._feature_buffer: List[np.ndarray] = []
        self._target_buffer: List[float] = []
        self._timestamp_buffer: List[pd.Timestamp] = []
        self._prediction_history: List[Dict] = []
        self._training_log: List[Dict] = []
        self._n_retrains = 0

    def feed(self, features: np.ndarray, target: float, timestamp: pd.Timestamp):
        """输入新数据点"""
        self._feature_buffer.append(features)
        self._target_buffer.append(target)
        self._timestamp_buffer.append(timestamp)

        # 保持缓冲区大小
        max_buffer = max(self.config.train_window * 3, self.config.warmup_period)
        if len(self._feature_buffer) > max_buffer:
            self._feature_buffer = self._feature_buffer[-max_buffer:]
            self._target_buffer = self._target_buffer[-max_buffer:]
            self._timestamp_buffer = self._timestamp_buffer[-max_buffer:]

    def should_retrain(self) -> bool:
        """判断是否需要重训练"""
        if len(self._feature_buffer) < self.config.train_window:
            return False

        if self.checkpoint is None:
            # 首次训练: 需要积累足够数据
            if len(self._feature_buffer) >= self.config.warmup_period:
                return True
            return False

        # 条件1: 距上次训练超过间隔
        last_train = self.checkpoint.trained_at
        current = self._timestamp_buffer[-1]
        days_since_train = (current - last_train).days
        if days_since_train >= self.config.retrain_interval:
            return True

        # 条件2: 性能衰减
        if len(self._prediction_history) >= self.config.validation_window:
            recent_actual = np.array(self._target_buffer[-self.config.validation_window:])
            recent_pred = np.array([p['prediction'] for p in
                                    self._prediction_history[-self.config.validation_window:]])
            if len(recent_actual) == len(recent_pred):
                recent_mse = mean_squared_error(recent_actual, recent_pred)
                baseline_mse = self.checkpoint.val_mse
                if baseline_mse > 0 and recent_mse > baseline_mse * (1 + self.config.performance_threshold):
                    return True

        return False

    def retrain(self) -> ModelCheckpoint:
        """重训练模型"""
        # 使用滑动窗口
        X = np.array(self._feature_buffer[-self.config.train_window:])
        y = np.array(self._target_buffer[-self.config.train_window:])

        # 分割训练/验证
        split = len(X) - self.config.validation_window
        if split < self.config.min_train_samples:
            split = max(self.config.min_train_samples, len(X) // 2)

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 训练
        model = self.model_cls(**self.model_kwargs)
        model.fit(X_train_scaled, y_train)

        # 评估
        train_pred = model.predict(X_train_scaled)
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, model.predict(X_val_scaled))

        self.checkpoint = ModelCheckpoint(
            model=model, scaler=scaler,
            trained_at=self._timestamp_buffer[-1],
            train_mse=train_mse, val_mse=val_mse
        )
        self.scaler = scaler
        self._n_retrains += 1

        self._training_log.append({
            'trained_at': self._timestamp_buffer[-1],
            'train_mse': train_mse,
            'val_mse': val_mse,
            'retrain_count': self._n_retrains,
            'train_samples': len(X_train),
        })

        return self.checkpoint

    def predict(self, features: np.ndarray) -> float:
        """预测"""
        if self.checkpoint is None:
            return 0.0
        X = np.array(features).reshape(1, -1)
        X_scaled = self.checkpoint.scaler.transform(X)
        return float(self.checkpoint.model.predict(X_scaled)[0])

    def predict_and_record(self, features: np.ndarray, actual: float,
                           timestamp: pd.Timestamp) -> float:
        """预测并记录"""
        pred = self.predict(features)
        self._prediction_history.append({
            'timestamp': timestamp,
            'prediction': pred,
            'actual': actual,
        })
        return pred


# ============================================================
# 对比基准: 固定窗口模型 (现有方式)
# ============================================================

class FixedWindowModel:
    """
    固定窗口模型 (模拟 jingni-trader 现有方式)
    一次性训练，不重训练
    """

    def __init__(self, train_window: int = 60):
        self.train_window = train_window
        self.model: Optional[Ridge] = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, features: np.ndarray) -> float:
        if not self.is_trained:
            return 0.0
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])


# ============================================================
# Test Suite
# ============================================================

class TestAdaptiveRetraining(unittest.TestCase):
    """自适应 ML 重训练测试"""

    @classmethod
    def setUpClass(cls):
        """生成带概念漂移的合成数据"""
        np.random.seed(42)
        n_samples = 500

        # 生成多因子特征
        X_raw = np.random.randn(n_samples, 5)
        # 生成带漂移的系数
        betas = np.zeros((n_samples, 5))
        for i in range(5):
            # 每个因子的系数随时间变化 (概念漂移)
            t = np.linspace(0, 4 * np.pi, n_samples)
            betas[:, i] = 0.5 + 0.3 * np.sin(t + i * np.pi / 5)

        # 生成目标
        y_noise = np.random.randn(n_samples) * 0.1
        y = np.sum(X_raw * betas, axis=1) + y_noise

        cls.X = X_raw
        cls.y = y
        cls.betas = betas
        cls.timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='B')
        cls.n_samples = n_samples

    def test_sliding_window_vs_fixed(self):
        """测试滑动窗口 vs 固定窗口的预测性能"""
        config = TrainingConfig(
            train_window=60,
            retrain_interval=10,
            min_train_samples=30,
            performance_threshold=0.3,
            validation_window=20,
            warmup_period=50,
        )

        adaptive = AdaptiveModel(config)
        fixed = FixedWindowModel(train_window=60)

        adaptive_preds = []
        adaptive_actuals = []
        fixed_preds = []
        fixed_actuals = []

        # 固定窗口: 用前60数据训练，后面不再更新
        fixed.train(self.X[:60], self.y[:60])

        for i in range(self.n_samples):
            features = self.X[i]
            actual = self.y[i]
            timestamp = self.timestamps[i]

            # 自适应模型
            adaptive.feed(features, actual, timestamp)

            if i >= config.warmup_period:
                if i == config.warmup_period:
                    adaptive.retrain()

                if adaptive.should_retrain():
                    adaptive.retrain()

                pred = adaptive.predict_and_record(features, actual, timestamp)
                adaptive_preds.append(pred)
                adaptive_actuals.append(actual)

                # 固定模型
                fixed_pred = fixed.predict(features)
                fixed_preds.append(fixed_pred)
                fixed_actuals.append(actual)

        # 计算 MSE
        adaptive_mse = mean_squared_error(adaptive_actuals, adaptive_preds)
        fixed_mse = mean_squared_error(fixed_actuals, fixed_preds)

        print(f"\n=== 滑动窗口 vs 固定窗口 ===")
        print(f"自适应模型 MSE: {adaptive_mse:.6f}")
        print(f"固定窗口模型 MSE: {fixed_mse:.6f}")
        print(f"性能提升: {(fixed_mse - adaptive_mse) / fixed_mse * 100:.1f}%")
        print(f"自适应模型重训练次数: {adaptive._n_retrains}")

        # 自适应模型应优于固定窗口模型
        self.assertLess(adaptive_mse, fixed_mse,
                       f"自适应模型 MSE ({adaptive_mse:.6f}) 应低于固定窗口 ({fixed_mse:.6f})")

    def test_model_expiration_detection(self):
        """测试模型过期检测"""
        config = TrainingConfig(
            train_window=40,
            retrain_interval=999,  # 不按时间触发
            performance_threshold=0.5,  # 放大阈值以便触发
            validation_window=10,
            warmup_period=30,
        )

        adaptive = AdaptiveModel(config)

        # 正常数据段训练
        normal_X = np.random.randn(60, 5)
        normal_y = normal_X @ np.array([0.5, 0.5, 0.5, 0.5, 0.5]) + np.random.randn(60) * 0.2

        for i in range(60):
            adaptive.feed(normal_X[i], normal_y[i], self.timestamps[i])

        adaptive.retrain()
        initial_val_mse = adaptive.checkpoint.val_mse
        print(f"\n=== 模型过期检测 ===")
        print(f"初始训练验证 MSE: {initial_val_mse:.6f}")

        # 注入概念漂移 (系数反转)
        drift_X = np.random.randn(30, 5)
        drift_y = drift_X @ np.array([-0.5, -0.5, -0.5, -0.5, -0.5]) + np.random.randn(30) * 0.2

        for i in range(30):
            adaptive.feed(drift_X[i], drift_y[i], self.timestamps[60 + i])
            adaptive.predict_and_record(drift_X[i], drift_y[i], self.timestamps[60 + i])

        # 检查是否触发重训练
        should_retrain = adaptive.should_retrain()
        print(f"漂移后是否触发重训练: {should_retrain}")

        if should_retrain:
            adaptive.retrain()
            print(f"重训练后验证 MSE: {adaptive.checkpoint.val_mse:.6f}")
            # 重训练被正确触发
            self.assertTrue(should_retrain, "漂移后应触发重训练")
        else:
            self.fail("漂移后未触发重训练，检测逻辑可能有问题")

    def test_rolling_performance(self):
        """测试滚动预测性能 (分段评估)"""
        config = TrainingConfig(
            train_window=60,
            retrain_interval=10,
            performance_threshold=0.3,
            validation_window=20,
            warmup_period=50,
        )

        adaptive = AdaptiveModel(config)

        # 分段评估
        segment_size = 100
        n_segments = self.n_samples // segment_size
        segment_mses = []

        for seg in range(n_segments):
            start = seg * segment_size
            end = min(start + segment_size, self.n_samples)

            seg_preds = []
            seg_actuals = []

            for i in range(start, end):
                features = self.X[i]
                actual = self.y[i]
                timestamp = self.timestamps[i]

                adaptive.feed(features, actual, timestamp)

                if i >= config.warmup_period:
                    if adaptive.checkpoint is None:
                        adaptive.retrain()

                    if adaptive.should_retrain():
                        adaptive.retrain()

                    pred = adaptive.predict_and_record(features, actual, timestamp)
                    seg_preds.append(pred)
                    seg_actuals.append(actual)

            if seg_preds:
                seg_mse = mean_squared_error(seg_actuals, seg_preds)
                segment_mses.append(seg_mse)

        print(f"\n=== 滚动性能评估 ===")
        for i, mse in enumerate(segment_mses):
            print(f"段 {i+1}: MSE = {mse:.6f}")

        # MSE 不应在后期剧烈恶化 (说明自适应在起作用)
        if len(segment_mses) >= 3:
            first_half = np.mean(segment_mses[:len(segment_mses)//2])
            second_half = np.mean(segment_mses[len(segment_mses)//2:])
            ratio = second_half / first_half if first_half > 0 else 1
            print(f"后半段/前半段 MSE 比: {ratio:.2f}")
            # 由于概念漂移，自适应模型应该保持较好性能
            self.assertLess(ratio, 3.0,
                           f"后半段 MSE ({second_half:.6f}) 不应过度恶化于前半段 ({first_half:.6f})")

    def test_retraining_overhead(self):
        """测试重训练开销"""
        config = TrainingConfig(
            train_window=60,
            retrain_interval=10,
            min_train_samples=30,
            performance_threshold=0.3,
            validation_window=20,
            warmup_period=50,
        )

        adaptive = AdaptiveModel(config)

        # 预热
        for i in range(60):
            adaptive.feed(self.X[i], self.y[i], self.timestamps[i])

        # 首次训练
        start = time.time()
        adaptive.retrain()
        first_train_time = time.time() - start

        # 后续重训练
        for i in range(60, self.n_samples):
            adaptive.feed(self.X[i], self.y[i], self.timestamps[i])

        start = time.time()
        adaptive.retrain()
        retrain_time = time.time() - start

        print(f"\n=== 重训练开销 ===")
        print(f"首次训练耗时: {first_train_time:.4f}s")
        print(f"重训练耗时:   {retrain_time:.4f}s")
        print(f"总重训练次数: {adaptive._n_retrains}")
        print(f"单次预测耗时: 通常 < 1ms (sklearn predict)")

        # 重训练应在合理时间完成
        self.assertLess(retrain_time, 1.0,
                       f"单次重训练应在 1s 内完成，实际 {retrain_time:.3f}s")

    def test_warmup_period(self):
        """测试预热期逻辑"""
        config = TrainingConfig(
            warmup_period=50,
            train_window=30,
            retrain_interval=10,
        )

        adaptive = AdaptiveModel(config)

        # 预热期内不应训练
        for i in range(30):
            adaptive.feed(self.X[i], self.y[i], self.timestamps[i])
            self.assertFalse(adaptive.should_retrain(),
                           f"预热期第{i}步不应触发训练")

        # 预热期后应可训练
        for i in range(30, 60):
            adaptive.feed(self.X[i], self.y[i], self.timestamps[i])

        self.assertTrue(adaptive.should_retrain(),
                       "预热期后应可训练")

        print(f"\n=== 预热期验证 ===")
        print(f"预热期 {config.warmup_period} 步内不触发训练: OK")
        print(f"预热期后可触发训练: OK")


if __name__ == '__main__':
    unittest.main(verbosity=2)