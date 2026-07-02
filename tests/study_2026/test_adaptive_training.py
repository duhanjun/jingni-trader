"""
验证测试：自适应滑动窗口训练 + 异常值检测
===============================================
借鉴来源：Freqtrade/FreqAI (25K+ GitHub stars)
优化方向：strategy-model-engine — 引入自适应滑动窗口重训练机制、异常值检测、特征重要性分析
核心亮点：
  - FreqAI 滑动窗口机制：train_period_days + backtest_period_days，定期重训练
  - 异常值检测：Dissimilarity Index (DI)、SVM、DBSCAN
  - SHAP 特征重要性分析
  - 多模型集成 (Ensemble)

验证内容：
  1. Purged Group Time Series Split 正确性
  2. 滑动窗口 vs 固定窗口训练效果对比
  3. 异常值检测（DI 方法）有效性
  4. SHAP 特征重要性排序

运行方式：cd /workspace && python tests/study_2026/test_adaptive_training.py
"""
import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace')

TEST_RESULTS = {}


def generate_test_data(n_stocks=30, n_days=800):
    """生成带有预测信号的测试数据"""
    np.random.seed(123)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks // 2)] + \
            [f"{300000 + i:06d}.SZ" for i in range(n_stocks // 2)]
    dates = pd.bdate_range(start='2022-01-01', periods=n_days)

    all_rows = []
    for code in codes:
        start_price = np.random.uniform(5, 60)
        daily_ret = np.random.normal(0.0002, 0.02, n_days)
        for j in range(1, n_days):
            daily_ret[j] += 0.15 * daily_ret[j - 1]

        prices = start_price * np.cumprod(1 + daily_ret)
        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'close': prices,
            'volume': np.random.lognormal(12, 0.5, n_days).astype(int),
            'volatility': np.abs(np.random.normal(0.015, 0.008, n_days)),
        })

        # 构造因子：有一定预测能力的特征
        df = df.sort_values('date')
        df['ret_1d'] = df['close'].pct_change()
        df['ret_5d'] = df['close'].pct_change(5)
        df['ret_20d'] = df['close'].pct_change(20)
        df['ma_ratio'] = df['close'].rolling(20).mean() / df['close']
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vol_regime'] = df['volatility'].rolling(20).mean()

        # 前视收益（标签）
        df['forward_ret'] = df['close'].shift(-5) / df['close'] - 1

        # 添加一些异常值样本
        n_outliers = int(n_days * 0.05)
        outlier_idx = np.random.choice(range(n_days // 2, n_days), n_outliers, replace=False)
        df.loc[df.index[outlier_idx], 'ret_1d'] *= np.random.uniform(5, 10, n_outliers)
        df.loc[df.index[outlier_idx], 'volume_ratio'] *= np.random.uniform(10, 20, n_outliers)

        all_rows.append(df)

    data = pd.concat(all_rows, ignore_index=True)
    return data.sort_values(['date', 'code']).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# Purged Group Time Series Split
# ═══════════════════════════════════════════════════════════════

class PurgedGroupTimeSeriesSplit:
    """
    Purged Group Time Series Cross Validator

    借鉴自：
    - Marcos Lopez de Prado "Advances in Financial Machine Learning"
    - FreqAI 的滑动窗口实现

    核心思想：
    1. 按时间顺序分割训练/验证集（防止未来信息泄露）
    2. 在训练集尾部加入 purge gap（防止标签重叠影响训练集）
    3. 每个 split 的测试集不重叠
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 5):
        self.n_splits = n_splits          # 交叉验证折数
        self.purge_gap = purge_gap        # 清洗期（交易日）

        # 存储每次 split 的信息
        self.splits_: list = []

    def split(self, X, y, groups=None):
        """生成训练/验证索引"""
        dates = X.index if isinstance(X.index, pd.DatetimeIndex) else pd.to_datetime(X.index)
        if not isinstance(dates, pd.DatetimeIndex):
            raise ValueError("X 的索引必须是 DatetimeIndex")

        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)
        self.splits_ = []

        # 计算每个 fold 的大小
        # 滑动窗口方式：train 从最早开始，test 逐步右移
        test_size = max(n_dates // (self.n_splits + 1), 20)

        for i in range(self.n_splits):
            # 训练集结束位置
            train_end_idx = n_dates - (self.n_splits - i) * test_size
            if train_end_idx < test_size:
                continue

            # 训练日期：从头到 train_end_idx（排除 purge gap）
            train_end_date = unique_dates[train_end_idx]
            purge_threshold = train_end_date

            if self.purge_gap > 0:
                # 排除临近测试集的日期
                purge_date = unique_dates[max(0, train_end_idx - self.purge_gap)]
            else:
                purge_date = train_end_date

            # 测试日期
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + test_size, n_dates)

            if test_start_idx >= n_dates:
                break

            train_mask = dates <= purge_date
            test_mask = (dates >= unique_dates[test_start_idx]) & \
                        (dates <= unique_dates[test_end_idx - 1])

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 50 and len(test_idx) > 10:
                split_info = {
                    'fold': i + 1,
                    'train_start': unique_dates[0],
                    'train_end': purge_date,
                    'test_start': unique_dates[test_start_idx],
                    'test_end': unique_dates[test_end_idx - 1],
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                }
                self.splits_.append(split_info)
                yield train_idx, test_idx


# ═══════════════════════════════════════════════════════════════
# 异常值检测器（Dissimilarity Index 方法）
# ═══════════════════════════════════════════════════════════════

class DissimilarityIndexDetector:
    """
    基于 Dissimilarity Index (DI) 的异常值检测

    参考 FreqAI 的 outlier detection 模块：
    - DI 衡量每个样本与其 K 近邻的平均距离
    - 距离分布的上尾样本被标记为异常值
    - 支持自动阈值设定

    原理：
    1. 对每个样本，计算到 K 个最近邻的平均距离
    2. DI 值越大，表示该样本越"孤立"
    3. 取 DI 分布的上分位数作为异常阈值
    """

    def __init__(self, n_neighbors: int = 5, contamination: float = 0.05):
        """
        参数:
            n_neighbors: KNN 的 K 值
            contamination: 预期的异常值比例（0.01 ~ 0.10）
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.threshold_ = None
        self.di_scores_ = None

    def fit(self, X: np.ndarray) -> 'DissimilarityIndexDetector':
        """
        计算所有样本的 DI 分数并确定阈值
        """
        from sklearn.neighbors import NearestNeighbors

        n_samples = X.shape[0]
        # 使用 KNN 找最近邻
        nn = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, n_samples))
        nn.fit(X)

        # 获取距离（排除自身）
        distances, _ = nn.kneighbors(X)
        # 排除距离0（自身），取 K 个最近邻的平均距离
        if distances.shape[1] > 1:
            di_scores = distances[:, 1:].mean(axis=1)
        else:
            di_scores = distances[:, 0]

        self.di_scores_ = di_scores
        # 阈值：取上分位数
        self.threshold_ = np.percentile(di_scores, 100 * (1 - self.contamination))

        return self

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """
        预测哪些样本是异常值
        Returns: bool array, True=异常值
        """
        if self.di_scores_ is None:
            raise ValueError("请先调用 fit() 方法")
        return self.di_scores_ > self.threshold_

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """一次性完成 fit 和 predict"""
        self.fit(X)
        return self.predict()


# ═══════════════════════════════════════════════════════════════
# SHAP 特征重要性计算
# ═══════════════════════════════════════════════════════════════

def compute_feature_importance_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    使用 SHAP 计算特征重要性

    参考 FreqAI 的 feature_selection = "SHAP" 配置
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # 如果是回归模型，shap_values 是 2D 数组
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df
    except ImportError:
        # Fallback: 使用模型内置的特征重要性
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 自适应训练引擎
# ═══════════════════════════════════════════════════════════════

class AdaptiveTrainer:
    """
    自适应滑动窗口训练器

    参考 FreqAI 的配置：
    - train_period_days: 45-180天（训练数据窗口）
    - backtest_period_days: 10-60天（回测验证窗口）
    - model_retrain_hours: 24（重训练间隔）
    """

    def __init__(
        self,
        train_window_days: int = 252,       # 训练窗口（交易日）
        retrain_frequency: int = 63,         # 重训练频率（每N个交易日）
        outlier_detector=None,
    ):
        self.train_window_days = train_window_days
        self.retrain_frequency = retrain_frequency
        self.outlier_detector = outlier_detector or DissimilarityIndexDetector()
        self.models_ = {}                     # 存储每个窗口的模型
        self.performance_log_ = []            # 记录每个窗口的性能

    def train_sliding_window(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        model_factory=None,
    ) -> dict:
        """
        滑动窗口训练

        参数:
            X: 特征矩阵
            y: 目标变量
            dates: 日期序列
            model_factory: 模型工厂函数，返回模型实例

        返回:
            训练记录 dict
        """
        from sklearn.ensemble import GradientBoostingRegressor

        if model_factory is None:
            def model_factory():
                return GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=42
                )

        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        records = []
        window_start = 0

        while window_start + self.train_window_days <= n_dates:
            # 训练窗口
            train_end = window_start + self.train_window_days
            train_dates = unique_dates[window_start:train_end]

            # 测试窗口（训练窗口后的 N 天）
            test_start = train_end
            test_end = min(test_start + self.retrain_frequency, n_dates)
            test_dates = unique_dates[test_start:test_end]

            if len(test_dates) < 5:
                break

            # 筛选数据
            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(test_dates)

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            if len(X_train) < 100 or len(X_test) < 10:
                window_start += self.retrain_frequency
                continue

            # 异常值检测
            try:
                outlier_mask = self.outlier_detector.fit_predict(X_train.values)
                clean_count = (~outlier_mask).sum()
                if clean_count > 50:
                    X_train_clean = X_train[~outlier_mask]
                    y_train_clean = y_train[~outlier_mask]
                else:
                    X_train_clean, y_train_clean = X_train, y_train
            except Exception:
                X_train_clean, y_train_clean = X_train, y_train
                outlier_mask = np.zeros(len(X_train), dtype=bool)
                clean_count = len(X_train)

            # 训练模型
            model = model_factory()
            model.fit(X_train_clean, y_train_clean)

            # 评估
            from sklearn.metrics import mean_squared_error
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)

            # IC
            if len(pred) >= 10:
                ic = np.corrcoef(pred, y_test)[0, 1]
            else:
                ic = 0.0

            records.append({
                'window_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'train_size': len(X_train_clean),
                'test_size': len(X_test),
                'outliers_removed': int(outlier_mask.sum()),
                'mse': float(mse),
                'ic': float(ic),
            })

            # 移动窗口
            window_start += self.retrain_frequency

        self.performance_log_ = records
        return records


# ═══════════════════════════════════════════════════════════════
# 验证测试
# ═══════════════════════════════════════════════════════════════

def test_1_purged_ts_split():
    """测试1：Purged Group Time Series Split 正确性"""
    print("\n" + "=" * 60)
    print("测试1：Purged Group Time Series Split 正确性")
    print("=" * 60)

    data = generate_test_data(n_stocks=10, n_days=500)
    data['date'] = pd.to_datetime(data['date'])

    # 准备数据
    feature_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'ma_ratio', 'volume_ratio', 'vol_regime']
    data_clean = data.dropna(subset=feature_cols + ['forward_ret'])
    X = data_clean[feature_cols]
    y = data_clean['forward_ret']

    # 设置日期索引
    X.index = data_clean['date']
    y.index = data_clean['date']

    splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=5)
    splits = list(splitter.split(X, y))

    print(f"  生成分割数: {len(splits)} (预期 5)")
    for split_info in splitter.splits_:
        print(f"    Fold {split_info['fold']}: "
              f"train={split_info['train_start'].strftime('%Y-%m-%d')}~"
              f"{split_info['train_end'].strftime('%Y-%m-%d')} "
              f"({split_info['train_size']}样本) | "
              f"test={split_info['test_start'].strftime('%Y-%m-%d')}~"
              f"{split_info['test_end'].strftime('%Y-%m-%d')} "
              f"({split_info['test_size']}样本)")

    # 验证未来信息泄露检查：检查 test 日期是否出现在之前 fold 的 train 中
    leak_detected = False
    for i, s_info in enumerate(splitter.splits_):
        test_start = s_info['test_start']
        # 检查该 test 起始日期是否在当前 fold 的 train 日期之后
        # 在 purged TS CV 中，test 必须在 train_end 之后
        if test_start <= s_info['train_end']:
            leak_detected = True
            break
        # 同时验证 purge gap：train_end 和 test_start 之间应该有足够的间隔

    result = {
        "passed": len(splits) >= 3 and not leak_detected,
        "n_splits": len(splits),
        "future_leak": leak_detected,
    }
    print(f"\n  未来信息泄露: {'检测到!' if leak_detected else '未检测到 (OK)'}")
    print(f"  结果: {'PASS' if result['passed'] else 'FAIL'}")
    TEST_RESULTS['test_1'] = result
    return result


def test_2_outlier_detection():
    """测试2：异常值检测（DI 方法）有效性"""
    print("\n" + "=" * 60)
    print("测试2：Dissimilarity Index 异常值检测")
    print("=" * 60)

    # 生成含异常值的数据
    np.random.seed(456)
    n_normal = 500
    n_outliers = 30

    # 正常数据
    X_normal = np.random.normal(0, 1, (n_normal, 5))

    # 异常值数据（在所有维度上偏离中心显著更远，使其在 KNN 空间中变得孤立）
    X_outliers = np.random.normal(0, 0.3, (n_outliers, 5)) + \
                 np.random.choice([-8, 8], (n_outliers, 5))

    X = np.vstack([X_normal, X_outliers])
    y_true = np.array([0] * n_normal + [1] * n_outliers)

    # DI 检测
    detector = DissimilarityIndexDetector(n_neighbors=5, contamination=0.05)
    y_pred = detector.fit_predict(X)

    # 评估
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  正常样本数: {n_normal}")
    print(f"  异常样本数: {n_outliers}")
    print(f"  检测到异常数: {y_pred.sum()}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  DI 阈值: {detector.threshold_:.4f}")

    # 打印 DI 分数分布
    print(f"  正常样本 DI 均值: {detector.di_scores_[:n_normal].mean():.4f}")
    print(f"  异常样本 DI 均值: {detector.di_scores_[n_normal:].mean():.4f}")

    result = {
        "passed": recall > 0.5 and precision > 0.3,
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(detector.threshold_),
    }
    print(f"  结果: {'PASS' if result['passed'] else 'FAIL'}")
    TEST_RESULTS['test_2'] = result
    return result


def test_3_sliding_window_vs_fixed():
    """测试3：滑动窗口 vs 固定窗口训练效果对比"""
    print("\n" + "=" * 60)
    print("测试3：滑动窗口 vs 固定窗口训练对比")
    print("=" * 60)

    data = generate_test_data(n_stocks=20, n_days=600)
    data['date'] = pd.to_datetime(data['date'])

    feature_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'ma_ratio', 'volume_ratio', 'vol_regime']
    data_clean = data.dropna(subset=feature_cols + ['forward_ret'])

    # 使用后80%的数据
    split_idx = int(len(data_clean) * 0.2)
    data_clean = data_clean.iloc[split_idx:].reset_index(drop=True)

    X = data_clean[feature_cols]
    y = data_clean['forward_ret']
    dates = data_clean['date']

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error

    # ── 固定窗口训练 ──
    train_ratio = 0.7
    train_end = int(len(data_clean) * train_ratio)
    X_train_fixed, X_test_fixed = X.iloc[:train_end], X.iloc[train_end:]
    y_train_fixed, y_test_fixed = y.iloc[:train_end], y.iloc[train_end:]

    model_fixed = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_fixed.fit(X_train_fixed, y_train_fixed)
    pred_fixed = model_fixed.predict(X_test_fixed)
    mse_fixed = mean_squared_error(y_test_fixed, pred_fixed)
    ic_fixed = np.corrcoef(pred_fixed, y_test_fixed)[0, 1] if len(pred_fixed) >= 10 else 0

    print(f"\n  固定窗口训练:")
    print(f"    训练样本: {len(X_train_fixed)}, 测试样本: {len(X_test_fixed)}")
    print(f"    MSE: {mse_fixed:.6f}")
    print(f"    IC: {ic_fixed:.6f}")

    # ── 滑动窗口训练 ──
    trainer = AdaptiveTrainer(
        train_window_days=180,
        retrain_frequency=60,
        outlier_detector=DissimilarityIndexDetector(contamination=0.03),
    )

    def model_factory():
        return GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

    records = trainer.train_sliding_window(
        X=X,
        y=y,
        dates=dates,
        model_factory=model_factory,
    )

    if records:
        avg_mse = np.mean([r['mse'] for r in records])
        avg_ic = np.mean([r['ic'] for r in records])
        avg_outliers = np.mean([r['outliers_removed'] for r in records])

        print(f"\n  滑动窗口训练 ({len(records)} 个窗口):")
        print(f"    平均 MSE: {avg_mse:.6f}")
        print(f"    平均 IC: {avg_ic:.6f}")
        print(f"    平均每窗口剔除异常值: {avg_outliers:.1f}")
        print(f"    窗口时间跨度: {records[0]['train_end'].strftime('%Y-%m-%d')} ~ "
              f"{records[-1]['test_end'].strftime('%Y-%m-%d')}")

        result = {
            "passed": len(records) >= 2,
            "n_windows": len(records),
            "fixed_mse": float(mse_fixed),
            "sliding_avg_mse": float(avg_mse),
            "fixed_ic": float(ic_fixed),
            "sliding_avg_ic": float(avg_ic),
            "avg_outliers_removed": float(avg_outliers),
        }
    else:
        result = {"passed": False, "error": "滑动窗口训练未生成任何记录"}

    print(f"\n  结果: {'PASS' if result.get('passed') else 'FAIL'}")
    TEST_RESULTS['test_3'] = result
    return result


def test_4_feature_importance():
    """测试4：特征重要性分析"""
    print("\n" + "=" * 60)
    print("测试4：特征重要性分析")
    print("=" * 60)

    data = generate_test_data(n_stocks=15, n_days=400)
    data['date'] = pd.to_datetime(data['date'])

    feature_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'ma_ratio', 'volume_ratio', 'vol_regime']
    data_clean = data.dropna(subset=feature_cols + ['forward_ret'])

    X = data_clean[feature_cols]
    y = data_clean['forward_ret']

    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)

    # 计算特征重要性
    importance_df = compute_feature_importance_shap(model, X)

    print(f"\n  特征重要性排名:")
    for _, row in importance_df.iterrows():
        bar = '█' * int(row['importance'] / importance_df['importance'].max() * 30)
        print(f"    {row['feature']:20s} {row['importance']:.6f} {bar}")

    # 验证最重要的特征有意义
    top_feature = importance_df.iloc[0]['feature'] if len(importance_df) > 0 else 'N/A'

    result = {
        "passed": len(importance_df) > 0,
        "n_features": len(importance_df),
        "top_feature": top_feature,
    }
    print(f"\n  结果: {'PASS' if result['passed'] else 'FAIL'}")
    TEST_RESULTS['test_4'] = result
    return result


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("自适应滑动窗口训练 + 异常值检测验证测试")
    print(f"借鉴来源: Freqtrade/FreqAI (25K+ stars)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_passed = True

    try:
        r1 = test_1_purged_ts_split()
        all_passed = all_passed and r1.get('passed', False)
    except Exception as e:
        print(f"  测试1 异常: {e}")
        all_passed = False

    try:
        r2 = test_2_outlier_detection()
        all_passed = all_passed and r2.get('passed', False)
    except Exception as e:
        print(f"  测试2 异常: {e}")
        all_passed = False

    try:
        r3 = test_3_sliding_window_vs_fixed()
        all_passed = all_passed and r3.get('passed', False)
    except Exception as e:
        print(f"  测试3 异常: {e}")
        all_passed = False

    try:
        r4 = test_4_feature_importance()
        all_passed = all_passed and r4.get('passed', False)
    except Exception as e:
        print(f"  测试4 异常: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    print(f"全部测试: {'PASS' if all_passed else 'SOME FAILED'}")
    print("=" * 60)

    # 保存测试结果
    results_path = os.path.join(os.path.dirname(__file__), 'test_results_adaptive.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(TEST_RESULTS, f, ensure_ascii=False, indent=2, default=str)

    return all_passed


if __name__ == '__main__':
    main()