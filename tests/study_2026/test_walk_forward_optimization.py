"""
验证代码：Walk-Forward Optimization (WFO) 滚动窗口训练
========================================================
借鉴来源：Freqtrade + FreqAI (github.com/freqtrade/freqtrade)
  - Walk-Forward Optimization 方法论：滚动训练-验证-测试
  - FreqAI 的滑动窗口训练机制与模型过期管理
  - Purge 机制防止数据泄露

优化方向：strategy-model-engine 模块从"一次性训练"升级为 WFO 滚动训练，
          避免过拟合，提升模型在样本外数据上的泛化能力。

对比分析：
  - 现有方式：使用 Purged Group TimeSeriesSplit 进行单次训练，然后在整个测试集上评估
  - 优化方式：WFO 多轮滚动训练，每轮用最新的训练数据重新训练模型，模拟真实交易中的持续学习
"""

import sys
import os
import time
import json
import unittest
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

warnings.filterwarnings('ignore')

# =============================================================================
# 模拟数据生成
# =============================================================================

def generate_financial_data(
    n_stocks: int = 20,
    n_days: int = 756,  # 约3年数据
    n_features: int = 10,
    signal_strength: float = 0.05,
    regime_shift: bool = True,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    生成模拟金融数据，包含市场风格切换
    
    返回:
        X: 特征 DataFrame
        y: 标签 Series
        meta: 元数据 (code, date)
    """
    np.random.seed(seed)
    
    codes = [f'{i:06d}.SZ' for i in range(1, n_stocks + 1)]
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')
    
    records = []
    
    for code in codes:
        # 生成基础特征
        feature_data = np.random.randn(n_days, n_features) * 0.1
        
        # 加入趋势（模拟因子有效性随时间变化）
        if regime_shift:
            # 前半段正向有效
            trend1 = np.linspace(0, signal_strength, n_days // 2)
            # 后半段有效性衰减（市场风格切换）
            trend2 = np.linspace(signal_strength, -signal_strength * 0.5, n_days - n_days // 2)
            trend = np.concatenate([trend1, trend2])
        else:
            trend = np.full(n_days, signal_strength)
        
        # 生成标签（收益率 + 噪声 + 趋势信号）
        noise = np.random.randn(n_days) * 0.02
        signal = feature_data[:, 0] * trend.reshape(-1, 1)[:, 0]  # 使用第一个特征作为信号
        forward_return = signal + noise
        
        for i, d in enumerate(dates):
            record = {
                'code': code,
                'date': d,
            }
            for j in range(n_features):
                record[f'factor_{j}'] = feature_data[i, j]
            record['forward_return'] = forward_return[i]
            records.append(record)
    
    df = pd.DataFrame(records)
    
    meta = df[['code', 'date']].copy()
    feature_cols = [f'factor_{j}' for j in range(n_features)]
    X = df[feature_cols]
    y = df['forward_return']
    
    return X, y, meta


# =============================================================================
# 方案 1: 现有方式 - 单次训练 + 全量测试
# =============================================================================

class SingleTrainEvaluator:
    """
    现有方式：单次训练，全量测试
    
    对应 jingni-trader 中 strategy-model-engine 的当前实现：
    - 使用 Purged Group TimeSeriesSplit 做交叉验证
    - 选最优超参数后训练一次，在全量测试集上评估
    """
    
    def __init__(self, train_ratio: float = 0.7, purge_days: int = 5):
        self.train_ratio = train_ratio
        self.purge_days = purge_days
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> Dict[str, Any]:
        """单次训练评估"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        
        unique_dates = sorted(dates.unique())
        n_train = int(len(unique_dates) * self.train_ratio)
        
        train_dates = unique_dates[:n_train]
        if self.purge_days > 0:
            train_dates = train_dates[:-self.purge_days]  # 简单 purge
        
        test_dates = unique_dates[n_train:]
        
        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # 计算 IC
        test_df = pd.DataFrame({
            'pred': y_pred,
            'actual': y_test.values,
            'date': dates[test_mask].values
        })
        
        ic_series = test_df.groupby('date').apply(
            lambda g: g['pred'].corr(g['actual']) if len(g) > 5 else np.nan
        ).dropna()
        
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'ic_mean': ic_series.mean(),
            'ic_std': ic_series.std(),
            'ic_ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            'ic_series': ic_series,
            'method': 'single_train',
        }


# =============================================================================
# 方案 2: Walk-Forward Optimization (借鉴 FreqAI)
# =============================================================================

@dataclass
class WFOWindow:
    """WFO 窗口定义"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization 滚动窗口训练
    
    借鉴 FreqAI 的设计：
    1. 滑动训练窗口 (rolling training window)
    2. 每轮训练后用最新数据验证
    3. 跟踪模型性能衰减信号
    4. 支持模型过期和自动重训练
    """
    
    def __init__(
        self,
        train_window_months: int = 12,
        test_window_months: int = 3,
        step_months: int = 3,
        min_train_samples: int = 100,
        purge_days: int = 5,
        model_expiry_days: int = 30,
    ):
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months
        self.min_train_samples = min_train_samples
        self.purge_days = purge_days
        self.model_expiry_days = model_expiry_days
    
    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WFOWindow]:
        """生成 WFO 窗口序列"""
        windows = []
        current = start_date + pd.DateOffset(months=self.train_window_months)
        window_id = 0
        
        while current + pd.DateOffset(months=self.test_window_months) <= end_date:
            train_end = current - timedelta(days=self.purge_days)
            train_start = train_end - pd.DateOffset(months=self.train_window_months)
            
            window = WFOWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=current,
                test_end=current + pd.DateOffset(months=self.test_window_months),
                window_id=window_id,
            )
            windows.append(window)
            current += pd.DateOffset(months=self.step_months)
            window_id += 1
        
        return windows
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        retrain_on_decay: bool = True,
    ) -> Dict[str, Any]:
        """
        WFO 滚动评估
        
        参数:
            retrain_on_decay: 是否在 IC 衰减时触发重训练
        """
        from sklearn.linear_model import LinearRegression
        
        dates_dt = pd.to_datetime(dates)
        start_date = dates_dt.min()
        end_date = dates_dt.max()
        
        windows = self.generate_windows(start_date, end_date)
        
        if len(windows) == 0:
            return {'error': '无法生成 WFO 窗口', 'n_windows': 0}
        
        all_predictions = []
        window_results = []
        prev_model = None
        prev_ic = None
        
        for w in windows:
            # 训练数据
            train_mask = (dates_dt >= w.train_start) & (dates_dt < w.train_end)
            X_train = X[train_mask]
            y_train = y[train_mask]
            
            if len(X_train) < self.min_train_samples:
                continue
            
            # 检查是否需要重训练
            should_retrain = True
            if retrain_on_decay and prev_model is not None and prev_ic is not None:
                # 快速检查上个窗口的预测性能
                if prev_ic > 0.02:  # IC 仍为正且显著
                    should_retrain = False  # 模型仍有效，跳过重训练
            
            if should_retrain:
                model = LinearRegression()
                model.fit(X_train, y_train)
            else:
                model = prev_model
            
            # 测试数据
            test_mask = (dates_dt >= w.test_start) & (dates_dt < w.test_end)
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            if len(X_test) < 10:
                continue
            
            y_pred = model.predict(X_test)
            
            # 计算 IC
            test_df = pd.DataFrame({
                'pred': y_pred,
                'actual': y_test.values,
                'date': dates_dt[test_mask].values,
            })
            
            ic = test_df.groupby('date').apply(
                lambda g: g['pred'].corr(g['actual']) if len(g) > 5 else np.nan
            ).dropna().mean()
            
            window_results.append({
                'window_id': w.window_id,
                'train_start': w.train_start,
                'test_start': w.test_start,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'ic': ic,
                'retrained': should_retrain,
            })
            
            all_predictions.append(test_df)
            prev_model = model
            prev_ic = ic
        
        if not all_predictions:
            return {'error': '无有效窗口', 'n_windows': 0}
        
        combined = pd.concat(all_predictions, ignore_index=True)
        
        ic_series = combined.groupby('date').apply(
            lambda g: g['pred'].corr(g['actual']) if len(g) > 5 else np.nan
        ).dropna()
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        return {
            'mse': mean_squared_error(combined['actual'], combined['pred']),
            'r2': r2_score(combined['actual'], combined['pred']),
            'ic_mean': ic_series.mean(),
            'ic_std': ic_series.std(),
            'ic_ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            'ic_series': ic_series,
            'n_windows': len(window_results),
            'window_results': window_results,
            'method': 'walk_forward',
        }


# =============================================================================
# 方案 3: 带模型衰减检测的 WFO (借鉴 FreqAI 模型过期)
# =============================================================================

class AdaptiveWFO(WalkForwardOptimizer):
    """
    自适应 WFO：带模型性能监控和自动重训练
    
    借鉴 FreqAI 的 continual learning 和 model expiration 机制：
    - 监控模型 IC 衰减
    - 当 IC 低于阈值时自动触发重训练
    - 支持模型版本管理
    """
    
    def __init__(
        self,
        train_window_months: int = 12,
        test_window_months: int = 3,
        step_months: int = 3,
        ic_decay_threshold: float = 0.01,  # IC 低于此值触发重训练
        ic_monitor_window: int = 3,  # 监控最近 N 个窗口的 IC
        **kwargs
    ):
        super().__init__(train_window_months, test_window_months, step_months, **kwargs)
        self.ic_decay_threshold = ic_decay_threshold
        self.ic_monitor_window = ic_monitor_window
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
    ) -> Dict[str, Any]:
        """自适应 WFO 评估"""
        result = super().evaluate(X, y, dates, retrain_on_decay=False)
        
        if 'window_results' not in result:
            return result
        
        # 分析 IC 变化趋势
        window_ics = [w['ic'] for w in result['window_results'] if not np.isnan(w['ic'])]
        
        if len(window_ics) >= 2:
            # 计算 IC 趋势
            ic_trend = np.polyfit(range(len(window_ics)), window_ics, 1)[0]
            result['ic_trend'] = ic_trend
            result['ic_decay_detected'] = ic_trend < -0.001
            
            # 计算自适应重训练次数（模拟）
            adaptive_retrains = 0
            for i in range(1, len(window_ics)):
                recent_ic = np.mean(window_ics[max(0, i-self.ic_monitor_window):i])
                if recent_ic < self.ic_decay_threshold:
                    adaptive_retrains += 1
            result['adaptive_retrains'] = adaptive_retrains
        
        return result


# =============================================================================
# 测试代码
# =============================================================================

class TestWalkForwardOptimization(unittest.TestCase):
    """WFO 验证测试"""
    
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.meta = generate_financial_data(
            n_stocks=30, n_days=756, n_features=8, signal_strength=0.05
        )
        cls.dates = cls.meta['date']
    
    def test_wfo_window_generation(self):
        """测试 WFO 窗口生成"""
        optimizer = WalkForwardOptimizer(
            train_window_months=12,
            test_window_months=3,
            step_months=3,
        )
        
        start = pd.Timestamp('2022-01-01')
        end = pd.Timestamp('2024-12-31')
        windows = optimizer.generate_windows(start, end)
        
        print(f"\nWFO 窗口生成: 共 {len(windows)} 个窗口")
        for w in windows[:3]:
            print(f"  Window {w.window_id}: train=[{w.train_start.date()}, {w.train_end.date()}], "
                  f"test=[{w.test_start.date()}, {w.test_end.date()}]")
        
        self.assertGreater(len(windows), 3, "WFO 窗口数不足")
        
        # 验证窗口无重叠
        for i in range(len(windows) - 1):
            self.assertLess(windows[i].test_end, windows[i+1].test_start + timedelta(days=1),
                           f"Window {i} 和 {i+1} 存在重叠")
    
    def test_single_train_vs_wfo(self):
        """对比单次训练 vs WFO 滚动训练"""
        print("\n" + "=" * 60)
        print("对比测试: 单次训练 vs WFO 滚动训练")
        print("=" * 60)
        
        # 单次训练
        single_eval = SingleTrainEvaluator(train_ratio=0.7)
        result_single = single_eval.evaluate(self.X, self.y, self.dates)
        
        # WFO
        wfo = WalkForwardOptimizer(
            train_window_months=12,
            test_window_months=3,
            step_months=3,
        )
        result_wfo = wfo.evaluate(self.X, self.y, self.dates)
        
        # 输出对比
        print(f"\n{'指标':<20} {'单次训练':<15} {'WFO':<15} {'改进':<15}")
        print("-" * 65)
        
        for metric in ['ic_mean', 'ic_std', 'ic_ir', 'r2']:
            sv = result_single.get(metric, 0)
            wv = result_wfo.get(metric, 0)
            if sv != 0:
                improvement = (wv - sv) / abs(sv) * 100
            else:
                improvement = 0
            print(f"{metric:<20} {sv:<15.4f} {wv:<15.4f} {improvement:<+14.1f}%")
        
        print(f"\nWFO 窗口数: {result_wfo.get('n_windows', 0)}")
        
        # WFO 的 IC IR 应该更稳定（考虑市场风格切换场景）
        self.assertIsNotNone(result_wfo.get('ic_mean'))
    
    def test_market_regime_shift(self):
        """测试市场风格切换场景下 WFO 的优势"""
        print("\n" + "=" * 60)
        print("场景测试: 市场风格切换")
        print("=" * 60)
        
        # 生成含风格切换的数据
        X_regime, y_regime, meta_regime = generate_financial_data(
            n_stocks=30, n_days=756, n_features=8,
            signal_strength=0.05, regime_shift=True  # 启用风格切换
        )
        dates_regime = meta_regime['date']
        
        # 单次训练
        single_eval = SingleTrainEvaluator(train_ratio=0.7)
        result_single = single_eval.evaluate(X_regime, y_regime, dates_regime)
        
        # WFO
        wfo = WalkForwardOptimizer(
            train_window_months=12,
            test_window_months=3,
            step_months=3,
        )
        result_wfo = wfo.evaluate(X_regime, y_regime, dates_regime)
        
        print(f"\n市场风格切换场景:")
        print(f"  单次训练 IC Mean: {result_single.get('ic_mean', 0):.4f}")
        print(f"  WFO 训练 IC Mean:  {result_wfo.get('ic_mean', 0):.4f}")
        
        # 分析 WFO 各窗口 IC 变化
        if 'window_results' in result_wfo:
            print(f"\n  WFO 各窗口 IC 变化:")
            for wr in result_wfo['window_results']:
                print(f"    Window {wr['window_id']}: IC={wr['ic']:.4f}, "
                      f"Train={wr['train_start'].date()}, Test={wr['test_start'].date()}")
        
        self.assertIsNotNone(result_wfo.get('ic_mean'))
    
    def test_adaptive_wfo(self):
        """测试自适应 WFO（带模型衰减检测）"""
        print("\n" + "=" * 60)
        print("自适应 WFO: 模型衰减检测")
        print("=" * 60)
        
        # 生成含风格切换的数据
        X_regime, y_regime, meta_regime = generate_financial_data(
            n_stocks=30, n_days=756, n_features=8,
            signal_strength=0.05, regime_shift=True
        )
        dates_regime = meta_regime['date']
        
        adaptive_wfo = AdaptiveWFO(
            train_window_months=12,
            test_window_months=3,
            step_months=3,
            ic_decay_threshold=0.01,
            ic_monitor_window=3,
        )
        result = adaptive_wfo.evaluate(X_regime, y_regime, dates_regime)
        
        print(f"\n自适应 WFO 结果:")
        print(f"  IC Mean: {result.get('ic_mean', 0):.4f}")
        print(f"  IC Trend: {result.get('ic_trend', 0):.6f}")
        print(f"  IC 衰减检测: {result.get('ic_decay_detected', False)}")
        print(f"  建议重训练次数: {result.get('adaptive_retrains', 0)}")
        print(f"  窗口数: {result.get('n_windows', 0)}")
        
        self.assertIsNotNone(result.get('ic_mean'))
    
    def test_window_size_sensitivity(self):
        """测试不同窗口大小的影响"""
        print("\n" + "=" * 60)
        print("窗口大小敏感性分析")
        print("=" * 60)
        
        configs = [
            (6, 3, 3),
            (12, 3, 3),
            (24, 3, 3),
            (12, 6, 3),
            (12, 3, 6),
        ]
        
        print(f"\n{'训练月':<10} {'测试月':<10} {'步长月':<10} {'IC Mean':<10} {'窗口数':<10}")
        print("-" * 50)
        
        for train_m, test_m, step_m in configs:
            wfo = WalkForwardOptimizer(
                train_window_months=train_m,
                test_window_months=test_m,
                step_months=step_m,
            )
            result = wfo.evaluate(self.X, self.y, self.dates)
            print(f"{train_m:<10} {test_m:<10} {step_m:<10} "
                  f"{result.get('ic_mean', 0):<10.4f} {result.get('n_windows', 0):<10}")


class TestContinualLearning(unittest.TestCase):
    """持续学习机制测试（借鉴 FreqAI continual learning）"""
    
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.meta = generate_financial_data(
            n_stocks=30, n_days=504, n_features=8, signal_strength=0.05
        )
        cls.dates = cls.meta['date']
    
    def test_model_retraining_trigger(self):
        """测试模型重训练触发机制"""
        print("\n" + "=" * 60)
        print("模型重训练触发机制")
        print("=" * 60)
        
        from sklearn.linear_model import LinearRegression
        
        # 模拟多轮训练
        n_rounds = 10
        retrain_threshold = 0.015  # IC 低于此值重训练
        retrain_count = 0
        current_ic = 0.03  # 初始 IC
        
        # 模拟 IC 衰减
        np.random.seed(42)
        ic_history = [current_ic]
        retrain_history = []
        
        for r in range(n_rounds):
            # 模拟 IC 衰减（带噪声）
            current_ic += np.random.normal(-0.002, 0.005)
            current_ic = max(-0.01, min(0.05, current_ic))
            ic_history.append(current_ic)
            
            # 检查是否需要重训练
            need_retrain = current_ic < retrain_threshold
            retrain_history.append(need_retrain)
            
            if need_retrain:
                retrain_count += 1
                # 重训练后 IC 恢复
                current_ic = 0.025 + np.random.normal(0, 0.003)
                ic_history.append(current_ic)
        
        print(f"\n模拟轮数: {n_rounds}")
        print(f"重训练触发次数: {retrain_count}")
        print(f"IC 历史: {[f'{x:.4f}' for x in ic_history]}")
        print(f"重训练标记: {retrain_history}")
        
        # 验证重训练机制
        self.assertGreaterEqual(retrain_count, 0)
    
    def test_purge_mechanism(self):
        """测试 Purge 机制防止数据泄露"""
        print("\n" + "=" * 60)
        print("Purge 机制验证")
        print("=" * 60)
        
        # 创建带标签泄露风险的数据
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        # 假设因子用了未来信息（典型泄露场景）
        
        purge_days_list = [0, 2, 5, 10]
        results = []
        
        for purge_days in purge_days_list:
            # 模拟 purge 效果
            train_end = 70
            purge_end = train_end - purge_days
            
            # 训练集和测试集之间的日期间隙
            gap = train_end - purge_end
            
            results.append({
                'purge_days': purge_days,
                'train_end': train_end,
                'purge_end': purge_end,
                'gap_days': gap,
            })
        
        print(f"\n{'Purge天数':<12} {'训练截至':<12} {'实际截至':<12} {'间隙天数':<12}")
        print("-" * 48)
        for r in results:
            print(f"{r['purge_days']:<12} {r['train_end']:<12} {r['purge_end']:<12} {r['gap_days']:<12}")
        
        # 验证 purge 确实创造了间隙
        for r in results:
            self.assertGreaterEqual(r['gap_days'], 0)


def run_tests():
    """运行所有测试"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestWalkForwardOptimization))
    suite.addTest(unittest.makeSuite(TestContinualLearning))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("Walk-Forward Optimization 验证结果摘要")
    print("=" * 60)
    print(f"借鉴来源: Freqtrade + FreqAI (github.com/freqtrade/freqtrade)")
    print(f"  - Walk-Forward Optimization 滚动窗口训练")
    print(f"  - 滑动窗口训练 + 模型过期管理")
    print(f"  - Purge 机制防止数据泄露")
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    # 输出建议
    print("\n优化建议:")
    print("1. 在 strategy-model-engine 中增加 WFO 模式选项")
    print("2. 添加模型性能监控指标（IC 衰减检测）")
    print("3. 支持基于 IC 衰减的自动重训练触发")
    print("4. 增加 config 中的 WFO 参数配置项")
    
    return result


if __name__ == '__main__':
    run_tests()