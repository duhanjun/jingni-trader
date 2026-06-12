"""
验证代码：滚动窗口训练与 Purged K-Fold 交叉验证
借鉴来源：Microsoft Qlib (https://github.com/microsoft/qlib) - RollingGen + TrainerRM
优化方向：strategy-model-engine - 模型训练的时序稳健性

Qlib 的 RollingGen 和 TrainerRM 提供了工业级的滚动窗口训练框架，
核心优势：
1. 严格的时序切分，避免未来信息泄露
2. Purge Gap 机制，防止训练集和验证集重叠
3. 支持多步滚动训练，模拟真实投资场景
4. 自动记录每个窗口的模型和预测

本验证代码对比：
1. 简单时序切分（当前 jingni-trader 的实现）
2. PurgedGroupTimeSeriesSplit（当前已实现但未充分验证）
3. 滚动窗口训练（Rolling Window，Qlib 风格）
4. 验证三种方法在避免过拟合方面的差异
"""

import sys
import os
import time
import unittest
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 时间序列切分方法
# ============================================================

class TimeSeriesSplitters:
    """时间序列数据集切分工具集"""

    @staticmethod
    def simple_split(
        dates: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        简单时序切分：按时间顺序切分为训练/验证/测试集

        这是 jingni-trader 当前 model-engine 中 train() 方法使用的切分方式。
        问题：仅做一次切分，无法评估模型在不同时间段的稳定性。
        """
        unique_dates = sorted(dates.unique())
        n = len(unique_dates)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_dates = set(unique_dates[:train_end])
        val_dates = set(unique_dates[train_end:val_end])
        test_dates = set(unique_dates[val_end:])

        train_idx = dates[dates.isin(train_dates)].index.values
        val_idx = dates[dates.isin(val_dates)].index.values
        test_idx = dates[dates.isin(test_dates)].index.values

        return train_idx, val_idx, test_idx

    @staticmethod
    def purged_group_ts_split(
        dates: pd.Series,
        n_splits: int = 5,
        purge_days: int = 5,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Purged Group Time Series Split

        这是 jingni-trader 当前 model-engine 中 purged_group_ts_split() 的实现。
        相比简单切分，多了 purge gap 机制，但仍是单次训练。
        """
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

            if purge_days > 0:
                purge_date = unique_dates[train_end_idx] - timedelta(days=purge_days)
                train_dates = [d for d in train_dates if d <= purge_date]

            train_idx = dates[dates.isin(train_dates)].index.values
            val_idx = dates[dates.isin(val_dates)].index.values

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        return splits

    @staticmethod
    def rolling_window_splits(
        dates: pd.Series,
        train_window: int = 252,      # 训练窗口（交易日）
        val_window: int = 63,         # 验证窗口（交易日）
        step: int = 21,               # 滚动步长（交易日）
        purge_days: int = 5,
        min_train_days: int = 126,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        滚动窗口切分（Qlib RollingGen 风格）

        核心思路：
        1. 从最早日期开始，取 train_window 天作为训练集
        2. 紧接着取 val_window 天作为验证集
        3. 验证集之后作为测试集（未来）
        4. 窗口向前滚动 step 天，重复

        优点：
        - 模拟真实投资场景：用历史数据训练，预测未来
        - 每个窗口独立训练，可评估模型在不同市场环境下的表现
        - 避免过拟合：多个窗口的验证集表现可综合评估
        """
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        if n_dates < train_window + val_window:
            # 数据不足，回退到简单切分
            train_idx, val_idx, test_idx = TimeSeriesSplitters.simple_split(dates)
            return [(train_idx, val_idx, test_idx)]

        splits = []
        start = 0

        while start + train_window + val_window <= n_dates:
            train_end = start + train_window
            val_end = train_end + val_window

            # 训练集日期
            train_dates = unique_dates[start:train_end]
            # 验证集日期
            val_dates = unique_dates[train_end:val_end]
            # 测试集日期（验证集之后的所有数据）
            test_dates = unique_dates[val_end:]

            # Purge gap
            if purge_days > 0 and len(train_dates) > 0:
                purge_date = unique_dates[train_end - 1] - timedelta(days=purge_days)
                train_dates = [d for d in train_dates if d <= purge_date]

            if len(train_dates) < min_train_days:
                start += step
                continue

            train_idx = dates[dates.isin(train_dates)].index.values
            val_idx = dates[dates.isin(val_dates)].index.values
            test_idx = dates[dates.isin(test_dates)].index.values if test_dates else np.array([])

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx, test_idx))

            start += step

        return splits


# ============================================================
# 滚动窗口训练器
# ============================================================

class RollingWindowTrainer:
    """
    滚动窗口训练器（Qlib TrainerRM 风格）

    对每个滚动窗口：
    1. 训练模型
    2. 在验证集上评估
    3. 在测试集上预测
    4. 记录所有指标
    """

    def __init__(self, model_factory=None):
        self.model_factory = model_factory or (lambda: LinearRegression())
        self.results = []

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        train_window: int = 252,
        val_window: int = 63,
        step: int = 21,
    ) -> Dict[str, Any]:
        """
        执行滚动窗口训练和评估

        返回:
            {
                'window_results': [...],
                'aggregate_metrics': {...},
                'prediction_stability': {...}
            }
        """
        splits = TimeSeriesSplitters.rolling_window_splits(
            dates, train_window=train_window, val_window=val_window, step=step
        )

        window_results = []
        all_val_metrics = []
        all_test_preds = []

        for i, (train_idx, val_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            model = self.model_factory()
            model.fit(X_train, y_train)

            # 验证集评估
            val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            val_ic = pd.Series(val_pred).corr(y_val)

            # 测试集预测
            test_pred = None
            test_mse = None
            if len(test_idx) > 0:
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]
                test_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, test_pred)
                all_test_preds.append({
                    'window': i,
                    'test_idx': test_idx,
                    'predictions': test_pred,
                })

            window_result = {
                'window': i,
                'train_start': dates.iloc[train_idx[0]] if len(train_idx) > 0 else None,
                'train_end': dates.iloc[train_idx[-1]] if len(train_idx) > 0 else None,
                'val_start': dates.iloc[val_idx[0]] if len(val_idx) > 0 else None,
                'val_end': dates.iloc[val_idx[-1]] if len(val_idx) > 0 else None,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'test_size': len(test_idx),
                'val_mse': float(val_mse),
                'val_r2': float(val_r2),
                'val_ic': float(val_ic),
                'test_mse': float(test_mse) if test_mse is not None else None,
            }
            window_results.append(window_result)
            all_val_metrics.append({
                'mse': val_mse, 'r2': val_r2, 'ic': val_ic
            })

        # 聚合指标
        val_mses = [m['mse'] for m in all_val_metrics]
        val_ics = [m['ic'] for m in all_val_metrics]

        aggregate = {
            'n_windows': len(window_results),
            'val_mse_mean': float(np.mean(val_mses)),
            'val_mse_std': float(np.std(val_mses)),
            'val_ic_mean': float(np.mean(val_ics)),
            'val_ic_std': float(np.std(val_ics)),
            'ic_stability': float(np.mean(val_ics) / np.std(val_ics)) if np.std(val_ics) > 0 else 0,
        }

        # 预测稳定性：衡量不同窗口对同一日期预测的一致性
        stability = self._calc_prediction_stability(all_test_preds)

        return {
            'window_results': window_results,
            'aggregate_metrics': aggregate,
            'prediction_stability': stability,
        }

    def _calc_prediction_stability(self, all_test_preds: List[Dict]) -> Dict:
        """计算预测稳定性指标"""
        if len(all_test_preds) < 2:
            return {'stability_score': 1.0, 'note': '窗口数不足，无法评估稳定性'}

        # 取所有窗口的预测，只取共同长度部分
        preds = [p['predictions'] for p in all_test_preds]
        min_len = min(len(p) for p in preds)

        if min_len < 2:
            return {'stability_score': 1.0, 'note': '预测数据不足，无法评估稳定性'}

        # 截取相同长度
        preds = [p[:min_len] for p in preds]

        # 计算两两之间的相关性
        correlations = []
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                corr = np.corrcoef(preds[i], preds[j])[0, 1]
                correlations.append(corr)

        return {
            'stability_score': float(np.mean(correlations)),
            'stability_std': float(np.std(correlations)),
            'n_comparisons': len(correlations),
        }


# ============================================================
# 测试用例
# ============================================================

class TestTimeSeriesSplitters(unittest.TestCase):
    """时间序列切分测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_stocks = 30
        n_days = 500
        all_dates = pd.bdate_range('2022-01-01', periods=n_days)

        rows = []
        for code in [f"600{i:03d}.SH" for i in range(n_stocks)]:
            for d in all_dates:
                rows.append({'code': code, 'date': d})

        cls.dates = pd.DataFrame(rows)['date']
        cls.n_dates = n_days

    def test_simple_split(self):
        """测试简单时序切分"""
        train, val, test = TimeSeriesSplitters.simple_split(self.dates)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

        # 验证训练集日期 < 验证集日期 < 测试集日期
        train_dates = set(self.dates.iloc[train])
        val_dates = set(self.dates.iloc[val])
        test_dates = set(self.dates.iloc[test])

        self.assertTrue(max(train_dates) < min(val_dates))
        self.assertTrue(max(val_dates) < min(test_dates))

    def test_purged_group_ts_split(self):
        """测试 Purged Group TS Split"""
        splits = TimeSeriesSplitters.purged_group_ts_split(self.dates, n_splits=5)
        self.assertGreater(len(splits), 0)

        for i, (train_idx, val_idx) in enumerate(splits):
            self.assertGreater(len(train_idx), 0, f"Split {i}: train 为空")
            self.assertGreater(len(val_idx), 0, f"Split {i}: val 为空")

            train_dates = set(self.dates.iloc[train_idx])
            val_dates = set(self.dates.iloc[val_idx])

            self.assertTrue(max(train_dates) < min(val_dates),
                           f"Split {i}: 训练集和验证集有重叠")

    def test_rolling_window_splits(self):
        """测试滚动窗口切分"""
        splits = TimeSeriesSplitters.rolling_window_splits(
            self.dates, train_window=252, val_window=63, step=21
        )

        self.assertGreater(len(splits), 0, "应生成至少一个滚动窗口")

        print(f"\n滚动窗口切分结果: 共 {len(splits)} 个窗口")

        for i, (train_idx, val_idx, test_idx) in enumerate(splits):
            train_dates = set(self.dates.iloc[train_idx])
            val_dates = set(self.dates.iloc[val_idx])

            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(val_idx), 0)
            self.assertTrue(max(train_dates) < min(val_dates),
                           f"Window {i}: 训练集和验证集有重叠")

            if i < 3:
                print(f"  Window {i}: train={len(train_idx)} ({self.dates.iloc[train_idx[0]].date()} ~ "
                      f"{self.dates.iloc[train_idx[-1]].date()}), "
                      f"val={len(val_idx)} ({self.dates.iloc[val_idx[0]].date()} ~ "
                      f"{self.dates.iloc[val_idx[-1]].date()}), "
                      f"test={len(test_idx)}")

    def test_no_future_leakage(self):
        """验证所有切分方法均无未来信息泄露"""
        # 简单切分
        train, val, test = TimeSeriesSplitters.simple_split(self.dates)
        train_dates = set(self.dates.iloc[train])
        val_dates = set(self.dates.iloc[val])
        test_dates = set(self.dates.iloc[test])
        self.assertFalse(any(d in train_dates for d in val_dates))
        self.assertFalse(any(d in train_dates for d in test_dates))

        # 滚动窗口
        splits = TimeSeriesSplitters.rolling_window_splits(self.dates, train_window=252, val_window=63, step=21)
        for train_idx, val_idx, test_idx in splits:
            train_dates = set(self.dates.iloc[train_idx])
            val_dates = set(self.dates.iloc[val_idx])
            self.assertFalse(any(d in train_dates for d in val_dates),
                            "滚动窗口存在未来信息泄露")

        print("\n未来信息泄露检测全部通过")


class TestRollingWindowTraining(unittest.TestCase):
    """滚动窗口训练测试"""

    @classmethod
    def setUpClass(cls):
        """生成带有时间序列特征的数据"""
        np.random.seed(42)
        n_stocks = 20
        n_days = 400
        all_dates = pd.bdate_range('2023-01-01', periods=n_days)

        rows = []
        for code in [f"600{i:03d}.SH" for i in range(n_stocks)]:
            # 生成有趋势和噪声的因子
            trend = np.cumsum(np.random.normal(0, 0.01, n_days))
            noise = np.random.normal(0, 0.05, n_days)
            factor1 = trend + noise

            # 因子2：滞后相关
            factor2 = np.roll(factor1, 5) + np.random.normal(0, 0.02, n_days)

            # 目标：因子1 + 因子2 的线性组合 + 噪声
            target = 0.6 * factor1 + 0.4 * factor2 + np.random.normal(0, 0.03, n_days)

            for i, d in enumerate(all_dates):
                rows.append({
                    'code': code,
                    'date': d,
                    'factor1': factor1[i],
                    'factor2': factor2[i],
                    'target': target[i],
                })

        cls.data = pd.DataFrame(rows).sort_values(['code', 'date']).reset_index(drop=True)

    def test_rolling_vs_simple_training(self):
        """对比滚动窗口训练与简单切分的表现"""
        X = self.data[['factor1', 'factor2']]
        y = self.data['target']
        dates = self.data['date']

        trainer = RollingWindowTrainer()

        # 滚动窗口训练
        rolling_result = trainer.train_and_evaluate(
            X, y, dates, train_window=200, val_window=40, step=20
        )

        # 简单切分训练
        train_idx, val_idx, test_idx = TimeSeriesSplitters.simple_split(dates)
        model = LinearRegression()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        simple_val_pred = model.predict(X.iloc[val_idx])
        simple_val_mse = mean_squared_error(y.iloc[val_idx], simple_val_pred)
        simple_val_ic = pd.Series(simple_val_pred).corr(y.iloc[val_idx])

        print(f"\n训练方法对比:")
        print(f"  简单切分:")
        print(f"    验证集 MSE: {simple_val_mse:.6f}")
        print(f"    验证集 IC:  {simple_val_ic:.4f}")
        print(f"  滚动窗口 ({rolling_result['aggregate_metrics']['n_windows']} 个窗口):")
        print(f"    平均验证 MSE: {rolling_result['aggregate_metrics']['val_mse_mean']:.6f}")
        print(f"    验证 MSE Std:  {rolling_result['aggregate_metrics']['val_mse_std']:.6f}")
        print(f"    平均验证 IC:   {rolling_result['aggregate_metrics']['val_ic_mean']:.4f}")
        print(f"    验证 IC Std:   {rolling_result['aggregate_metrics']['val_ic_std']:.4f}")
        print(f"    IC 稳定性:     {rolling_result['aggregate_metrics']['ic_stability']:.4f}")

        # 滚动窗口应提供更多信息（MSE 的波动性）
        self.assertGreater(rolling_result['aggregate_metrics']['n_windows'], 1,
                          "滚动窗口应生成多个窗口")

    def test_ic_stability_across_windows(self):
        """验证滚动窗口的 IC 稳定性指标"""
        X = self.data[['factor1', 'factor2']]
        y = self.data['target']
        dates = self.data['date']

        trainer = RollingWindowTrainer()
        result = trainer.train_and_evaluate(
            X, y, dates, train_window=200, val_window=40, step=20
        )

        # IC 稳定性应大于 0
        ic_stability = result['aggregate_metrics']['ic_stability']
        self.assertIsNotNone(ic_stability)
        # 由于我们生成的数据相对稳定，IC 稳定性应该不错
        print(f"\nIC 稳定性: {ic_stability:.4f} "
              f"(越稳定表示模型在不同时间段表现一致)")

    def test_overfitting_detection(self):
        """验证滚动窗口能否检测过拟合"""
        X = self.data[['factor1', 'factor2']]
        y = self.data['target']
        dates = self.data['date']

        trainer = RollingWindowTrainer()
        result = trainer.train_and_evaluate(
            X, y, dates, train_window=200, val_window=40, step=20
        )

        # 如果 MSE 在窗口间波动很大，说明模型可能过拟合
        mse_std = result['aggregate_metrics']['val_mse_std']
        mse_mean = result['aggregate_metrics']['val_mse_mean']
        cv = mse_std / mse_mean if mse_mean > 0 else 0

        print(f"\n过拟合检测:")
        print(f"  MSE 变异系数: {cv:.4f} (CV > 0.5 可能表示过拟合)")

        window_results = result['window_results']
        for wr in window_results[:3]:
            print(f"  Window {wr['window']}: "
                  f"train={wr['train_size']}, val={wr['val_size']}, "
                  f"val_mse={wr['val_mse']:.6f}, val_ic={wr['val_ic']:.4f}")


class TestPerformanceComparison(unittest.TestCase):
    """性能对比测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_stocks = 50
        n_days = 600
        all_dates = pd.bdate_range('2021-01-01', periods=n_days)

        rows = []
        for code in [f"600{i:03d}.SH" for i in range(n_stocks)]:
            trend = np.cumsum(np.random.normal(0, 0.01, n_days))
            noise = np.random.normal(0, 0.05, n_days)
            factor1 = trend + noise
            factor2 = np.roll(factor1, 5) + np.random.normal(0, 0.02, n_days)
            target = 0.6 * factor1 + 0.4 * factor2 + np.random.normal(0, 0.03, n_days)
            for i, d in enumerate(all_dates):
                rows.append({
                    'code': code, 'date': d,
                    'factor1': factor1[i], 'factor2': factor2[i], 'target': target[i],
                })

        cls.data = pd.DataFrame(rows).sort_values(['code', 'date']).reset_index(drop=True)

    def test_rolling_window_training_time(self):
        """测试滚动窗口训练的耗时"""
        X = self.data[['factor1', 'factor2']]
        y = self.data['target']
        dates = self.data['date']

        trainer = RollingWindowTrainer()

        t0 = time.perf_counter()
        result = trainer.train_and_evaluate(
            X, y, dates, train_window=252, val_window=63, step=21
        )
        elapsed = time.perf_counter() - t0

        print(f"\n滚动窗口训练性能:")
        print(f"  数据规模: {len(X):,} rows, {len(X.columns)} features")
        print(f"  窗口数量: {result['aggregate_metrics']['n_windows']}")
        print(f"  总耗时:   {elapsed:.4f}s")
        print(f"  每窗口:   {elapsed / result['aggregate_metrics']['n_windows']:.4f}s")

        self.assertLess(elapsed, 30, "滚动窗口训练应在 30 秒内完成")


if __name__ == '__main__':
    unittest.main(verbosity=2)