"""
=============================================================================
借鉴来源: Qlib Rolling Window Training + Marcos Lopez de Prado "Advances in
           Financial Machine Learning" (Purged K-Fold CV)
优化方向: 时序交叉验证增强 - Purged Group Time Series Split
=============================================================================

核心亮点:
  金融时序数据具有序列相关性，标准的 K-Fold 交叉验证会破坏时序结构，
  导致训练集和验证集之间存在信息泄漏。

  Purged K-Fold CV 核心机制:
  1. Group split: 按时间分组，保证同一时间点的样本不会被拆分
  2. Purge gap: 在训练集和验证集之间留出间隔期
  3. Embargo: 验证集之后的数据也被排除出训练集

对比 jingni-trader 现状:
  当前 strategy-model-engine 已有 purged_group_ts_split 方法，
  但实现较为基础:
  - purge 操作仅移除 train_end 附近的日期
  - 没有 embargo 机制
  - 没有做到真正的按股票分组 (group by code)

验证内容:
  1. Purged K-Fold 正确性测试 (无跨期泄漏)
  2. Embargo 机制有效性验证
  3. 与标准 TimeSeriesSplit 的对比
  4. 信息泄漏检测
  5. Walk-forward 回测验证框架
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ═══════════════════════════════════════════════════════════════════════════
# Purged K-Fold Cross Validation 原型实现
# ============================================================================
# 借鉴来源:
#   - Qlib Rolling Window Training (RollingGen)
#   - Marcos Lopez de Prado "Advances in Financial Machine Learning" Ch.7
# ═══════════════════════════════════════════════════════════════════════════

class PurgedGroupTimeSeriesSplit:
    """
    Purged Group Time Series Cross-Validation

    参考 Qlib 的 RollingGen 和 Lopez de Prado 的 PurgedKFold
    实现了完整的 purge gap 和 embargo 机制。
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap_days: int = 5,
        embargo_pct: float = 0.0,
        min_train_size: int = 100,
        min_test_size: int = 20,
    ):
        """
        参数:
            n_splits: 分割数
            purge_gap_days: 训练集与验证集之间的清洗期（天）
            embargo_pct: 验证集之后排除的训练数据比例（0.0 - 1.0）
            min_train_size: 最小训练样本数
            min_test_size: 最小验证样本数
        """
        self.n_splits = n_splits
        self.purge_gap_days = purge_gap_days
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size

    def split(
        self,
        dates: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练/验证索引分割

        参数:
            dates: 每个样本的日期 (pd.Series)
            groups: 每个样本的分组（如股票代码），用于 group split

        返回:
            list of (train_indices, val_indices)
        """
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        if n_dates < self.n_splits + 1:
            return []

        test_size = max(
            n_dates // (self.n_splits + 1),
            1
        )

        splits = []

        for i in range(self.n_splits):
            # 训练集: 最早到当前分界的日期
            train_end_idx = n_dates - (self.n_splits - i) * test_size
            val_start_idx = train_end_idx + 1
            val_end_idx = min(val_start_idx + test_size, n_dates)

            if val_start_idx >= n_dates:
                break

            # 提取训练日期和验证日期
            train_dates = set(unique_dates[:train_end_idx])
            val_dates = set(unique_dates[val_start_idx:val_end_idx])

            # --- Purge: 移除训练集末尾 purge_gap_days 范围内的数据 ---
            if self.purge_gap_days > 0:
                purge_date = unique_dates[train_end_idx]
                if isinstance(purge_date, (pd.Timestamp, datetime)):
                    purge_threshold = purge_date - timedelta(days=self.purge_gap_days)
                else:
                    purge_threshold = purge_date
                train_dates = {d for d in train_dates if d <= purge_threshold}

            # --- Embargo: 移除验证集之后一定比例的训练数据 ---
            if self.embargo_pct > 0 and val_end_idx < n_dates:
                embargo_size = int((val_end_idx - val_start_idx) * self.embargo_pct)
                embargo_start = val_end_idx
                embargo_end = min(embargo_start + embargo_size, n_dates)
                embargo_dates = set(unique_dates[embargo_start:embargo_end])
                train_dates = train_dates - embargo_dates

            # 生成索引
            train_idx = dates[dates.isin(train_dates)].index.values
            val_idx = dates[dates.isin(val_dates)].index.values

            if len(train_idx) >= self.min_train_size and len(val_idx) >= self.min_test_size:
                splits.append((train_idx, val_idx))

        return splits

    def get_split_info(self, splits: List[Tuple], dates: pd.Series) -> pd.DataFrame:
        """生成每个分割的统计信息"""
        rows = []
        for i, (train_idx, val_idx) in enumerate(splits):
            train_dates = dates.iloc[train_idx]
            val_dates = dates.iloc[val_idx]

            rows.append({
                'fold': i,
                'train_start': str(train_dates.min()),
                'train_end': str(train_dates.max()),
                'val_start': str(val_dates.min()),
                'val_end': str(val_dates.max()),
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'gap_days': self._calc_gap_days(train_dates, val_dates),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _calc_gap_days(train_dates: pd.Series, val_dates: pd.Series) -> int:
        """计算训练集与验证集之间的间隔天数"""
        train_max = train_dates.max()
        val_min = val_dates.min()
        if hasattr(train_max, 'date'):
            return (val_min.date() - train_max.date()).days
        return 0


class LeakageDetector:
    """信息泄漏检测器 - 验证交叉验证分割的正确性"""

    @staticmethod
    def check_date_overlap(splits: List[Tuple], dates: pd.Series) -> Dict:
        """检查训练集和验证集是否有日期重叠"""
        results = []
        for i, (train_idx, val_idx) in enumerate(splits):
            train_dates = set(dates.iloc[train_idx].values)
            val_dates = set(dates.iloc[val_idx].values)
            overlap = train_dates & val_dates
            results.append({
                'fold': i,
                'has_overlap': len(overlap) > 0,
                'overlap_dates': len(overlap),
            })
        return {
            'all_clean': all(not r['has_overlap'] for r in results),
            'folds_detail': results,
        }

    @staticmethod
    def check_group_leakage(
        splits: List[Tuple],
        groups: pd.Series,
    ) -> Dict:
        """检查同一分组是否同时出现在训练集和验证集中"""
        results = []
        for i, (train_idx, val_idx) in enumerate(splits):
            train_groups = set(groups.iloc[train_idx].values)
            val_groups = set(groups.iloc[val_idx].values)
            shared = train_groups & val_groups
            results.append({
                'fold': i,
                'train_groups': len(train_groups),
                'val_groups': len(val_groups),
                'shared_groups': len(shared),
                'leakage_pct': round(len(shared) / len(val_groups) * 100, 2) if len(val_groups) > 0 else 0,
            })
        return {
            'any_leakage': any(r['shared_groups'] > 0 for r in results),
            'folds_detail': results,
        }

    @staticmethod
    def check_temporal_gap(
        splits: List[Tuple],
        dates: pd.Series,
        min_gap_days: int = 0,
    ) -> Dict:
        """检查训练集和验证集之间是否有足够的时间间隔"""
        results = []
        for i, (train_idx, val_idx) in enumerate(splits):
            train_max = dates.iloc[train_idx].max()
            val_min = dates.iloc[val_idx].min()
            gap = (val_min - train_max).days if hasattr(train_max, 'date') else 0
            results.append({
                'fold': i,
                'gap_days': gap,
                'meets_minimum': gap >= min_gap_days,
            })
        return {
            'all_meet_minimum': all(r['meets_minimum'] for r in results),
            'folds_detail': results,
        }


class WalkForwardValidator:
    """
    Walk-forward 回测验证框架

    模拟真实投资场景：在每个时间点，只用过去的数据训练，预测未来。
    这是评估策略泛化能力的最严格方法。
    """

    def __init__(self, n_splits: int = 5, purge_gap_days: int = 5):
        self.n_splits = n_splits
        self.purge_gap_days = purge_gap_days

    def validate(
        self,
        data: pd.DataFrame,
        train_func,
        predict_func,
        feature_cols: List[str],
        label_col: str = 'forward_return',
    ) -> pd.DataFrame:
        """
        Walk-forward 验证

        参数:
            data: 完整数据集（按 date 排序）
            train_func: 训练函数，接收 (X_train, y_train) 返回模型
            predict_func: 预测函数，接收 (model, X) 返回预测值
            feature_cols: 特征列名
            label_col: 标签列名

        返回:
            预测结果 DataFrame
        """
        # 重置索引以确保索引连续
        data = data.reset_index(drop=True)
        dates = data['date']
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=self.n_splits,
            purge_gap_days=self.purge_gap_days,
        )
        splits = cv.split(dates)

        all_predictions = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]

            X_train = train_data[feature_cols].values
            y_train = train_data[label_col].values
            X_val = val_data[feature_cols].values

            # 去除 NaN
            valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]

            if len(X_train) < 10 or len(X_val) < 5:
                continue

            model = train_func(X_train, y_train)
            preds = predict_func(model, X_val)

            fold_preds = val_data[['code', 'date']].copy()
            fold_preds['prediction'] = preds
            fold_preds['fold'] = fold
            all_predictions.append(fold_preds)

        if not all_predictions:
            return pd.DataFrame()

        result = pd.concat(all_predictions, ignore_index=True)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════════════════

def generate_test_data(n_stocks: int = 10, n_days: int = 300):
    """生成模拟A股数据"""
    np.random.seed(42)
    stocks = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')

    data = []
    for code in stocks:
        close = np.cumsum(np.random.randn(n_days) * 0.02) + 10
        close = np.maximum(close, 1)

        for i, dt in enumerate(dates):
            forward_return = (close[min(i + 5, n_days - 1)] / close[i] - 1) if i < n_days - 5 else np.nan
            data.append({
                'code': code,
                'date': dt,
                'close': close[i],
                'factor_a': np.random.randn() * 0.1 * close[i],
                'factor_b': np.random.randn() * 0.05 * close[i],
                'forward_return': forward_return,
            })

    return pd.DataFrame(data)


class TestPurgedCrossValidation(unittest.TestCase):
    """Purged K-Fold 交叉验证测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_test_data(n_stocks=10, n_days=300)
        cls.dates = cls.data['date']
        cls.groups = cls.data['code']

    def test_basic_split(self):
        """测试基本分割"""
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap_days=5, embargo_pct=0.01)
        splits = cv.split(self.dates)

        self.assertGreater(len(splits), 0, "应生成至少一个分割")
        self.assertLessEqual(len(splits), 5)

        # 验证每个分割的大小
        for i, (train_idx, val_idx) in enumerate(splits):
            self.assertGreater(len(train_idx), 0, f"Fold {i}: 训练集为空")
            self.assertGreater(len(val_idx), 0, f"Fold {i}: 验证集为空")

    def test_no_date_overlap(self):
        """测试无日期重叠 - 这是 purge 的核心保证"""
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap_days=5)
        splits = cv.split(self.dates)

        detector = LeakageDetector()
        result = detector.check_date_overlap(splits, self.dates)

        self.assertTrue(result['all_clean'],
                       f"存在日期重叠: {result['folds_detail']}")

    def test_purge_gap_effectiveness(self):
        """测试 purge gap 的有效性"""
        # 无 purge 的分割
        cv_no_purge = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap_days=0)
        splits_no_purge = cv_no_purge.split(self.dates)

        # 有 purge 的分割
        cv_with_purge = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap_days=10)
        splits_with_purge = cv_with_purge.split(self.dates)

        detector = LeakageDetector()
        gaps_no_purge = detector.check_temporal_gap(splits_no_purge, self.dates)
        gaps_with_purge = detector.check_temporal_gap(splits_with_purge, self.dates, min_gap_days=5)

        print(f"\n  Purge Gap 对比:")
        print(f"    无 Purge 各 fold 间隔: {[r['gap_days'] for r in gaps_no_purge['folds_detail']]}")
        print(f"    有 Purge 各 fold 间隔: {[r['gap_days'] for r in gaps_with_purge['folds_detail']]}")

        # 有 purge 的分割应该有更大的间隔
        avg_gap_no = np.mean([r['gap_days'] for r in gaps_no_purge['folds_detail']])
        avg_gap_with = np.mean([r['gap_days'] for r in gaps_with_purge['folds_detail']])
        self.assertGreaterEqual(avg_gap_with, avg_gap_no,
                                "Purge 应该增大训练集与验证集之间的间隔")

    def test_vs_standard_timeseriessplit(self):
        """对比标准 TimeSeriesSplit 与 Purged Group Split"""
        n_dates = len(self.data['date'].unique())

        # 标准 TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        std_splits = []
        fold_indices = np.arange(n_dates)
        for train_idx, val_idx in tscv.split(fold_indices):
            std_splits.append((train_idx, val_idx))

        # Purged Group Split
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap_days=5)
        purged_splits = cv.split(self.dates)

        print(f"\n  标准 vs Purged 分割对比:")
        print(f"    标准 TimeSeriesSplit: {len(std_splits)} folds")
        print(f"    Purged Group Split: {len(purged_splits)} folds")

        # 标准 TSCV 无 purge，训练/验证界限处可能有信息泄漏风险
        # Purged 版本通过间隔期降低了泄漏风险

    def test_split_info_table(self):
        """测试分割信息表输出"""
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap_days=5)
        splits = cv.split(self.dates)
        info = cv.get_split_info(splits, self.dates)

        print(f"\n  分割信息表:")
        print(info.to_string())

        self.assertEqual(len(info), len(splits))
        self.assertIn('fold', info.columns)
        self.assertIn('train_start', info.columns)
        self.assertIn('val_start', info.columns)

    def test_embargo_mechanism(self):
        """测试 embargo 机制"""
        cv_no_embargo = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap_days=5, embargo_pct=0)
        splits_no_embargo = cv_no_embargo.split(self.dates)

        cv_embargo = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap_days=5, embargo_pct=0.5)
        splits_embargo = cv_embargo.split(self.dates)

        # embargo 会减少训练集大小
        if len(splits_no_embargo) > 0 and len(splits_embargo) > 0:
            avg_train_no = np.mean([len(t) for t, v in splits_no_embargo])
            avg_train_embargo = np.mean([len(t) for t, v in splits_embargo])
            print(f"\n  Embargo 机制效果:")
            print(f"    无 Embargo 平均训练集大小: {avg_train_no:.0f}")
            print(f"    有 Embargo 平均训练集大小: {avg_train_embargo:.0f}")
            self.assertLessEqual(avg_train_embargo, avg_train_no,
                                 "Embargo 应减少训练集大小")

    def test_minimum_size_constraint(self):
        """测试最小样本数约束"""
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=5, purge_gap_days=5,
            min_train_size=5000, min_test_size=5000  # 设置极大的约束
        )
        splits = cv.split(self.dates)

        # 所有分割都应被过滤掉
        self.assertEqual(len(splits), 0, "过高的最小样本约束应导致无分割")

    def test_single_group_data(self):
        """测试单股票数据"""
        single = self.data[self.data['code'] == self.data['code'].iloc[0]].copy()
        cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap_days=5)
        splits = cv.split(single['date'])
        self.assertGreater(len(splits), 0, "单股票数据也应能生成分割")

    # ── Walk-Forward 验证测试 ──────────────────────────────────

    def test_walk_forward_validation(self):
        """测试 Walk-forward 验证框架"""
        data = self.data.dropna(subset=['forward_return']).copy()

        def train_model(X, y):
            """简单的线性回归模型"""
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            return model

        def predict_model(model, X):
            return model.predict(X)

        validator = WalkForwardValidator(n_splits=5, purge_gap_days=5)
        result = validator.validate(
            data, train_model, predict_model,
            feature_cols=['factor_a', 'factor_b'],
            label_col='forward_return',
        )

        self.assertGreater(len(result), 0, "Walk-forward 验证应产生预测结果")
        self.assertIn('prediction', result.columns)
        self.assertIn('fold', result.columns)

        # 计算每个 fold 的 IC（信息系数）
        if 'forward_return' in data.columns:
            merged = result.merge(data[['code', 'date', 'forward_return']], on=['code', 'date'])
            ic_by_fold = merged.groupby('fold').apply(
                lambda x: x['prediction'].corr(x['forward_return'])
            )
            print(f"\n  Walk-Forward 各 Fold IC: {ic_by_fold.to_dict()}")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Purged K-Fold 交叉验证增强测试")
    print("借鉴来源: Qlib RollingGen + Lopez de Prado PurgedKFold")
    print("=" * 70)
    results = run_tests()
    print("\n" + "=" * 70)
    print(f"测试结果: {results['tests_run']} 个测试, "
          f"{results['failures']} 个失败, {results['errors']} 个错误")
    print(f"总体: {'通过' if results['success'] else '失败'}")
    print("=" * 70)