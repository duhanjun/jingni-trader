"""
验证测试：滚动窗口回测
借鉴来源：Qlib (https://github.com/microsoft/qlib) - 严格回测框架
        具体借鉴 Qlib 的 rolling window backtest 机制和 purged grouping 方法

优化方向：为 jingni-trader 的 backtest-engine 增加滚动窗口回测能力，
        支持样本外（OOS）滚动测试，避免过拟合评估。

设计思路：
  - Qlib 的 TrainerRM (Rolling Model) 支持按时间滚动的训练/验证/测试分割
  - 支持 Purged Group Time Series Cross-Validation (防止数据泄露)
  - 每个窗口独立训练和回测，最终合并所有窗口的绩效
  - 本测试验证滚动窗口回测的正确性、过拟合检测能力和性能
"""
import sys
import os
import unittest
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ============================================================
# 滚动窗口回测核心实现
# ============================================================

@dataclass
class RollingWindowConfig:
    """滚动窗口配置 - 借鉴 Qlib 的 TrainerRM 配置"""
    train_months: int = 36           # 训练窗口（月）
    valid_months: int = 12           # 验证窗口（月）
    test_months: int = 6             # 向前测试窗口（月）
    step_months: int = 3             # 滚动步长（月）
    purge_days: int = 5              # 清洗期（天）- 防止数据泄露
    min_train_samples: int = 252     # 最小训练样本数
    embargo_days: int = 0            # 禁运期（天）


@dataclass 
class WindowResult:
    """单个窗口的回测结果"""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_samples: int
    test_samples: int
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None


class RollingWindowSplitter:
    """
    滚动窗口分割器
    
    借鉴 Qlib 的 PurgedGroupTimeSeriesSplit，但使用固定窗口滚动而非交叉验证。
    每个窗口 = 训练期 + 清洗期 + 测试期，然后向前滚动 step_months。
    """
    
    def __init__(self, config: RollingWindowConfig):
        self.config = config
    
    def generate_windows(self, data: pd.DataFrame, date_col: str = 'date') -> List[Tuple]:
        """
        生成滚动窗口列表
        
        参数:
            data: 含 date 列的 DataFrame
        
        返回:
            List of (train_start, train_end, test_start, test_end) 元组
        """
        dates = pd.to_datetime(sorted(data[date_col].unique()))
        if len(dates) == 0:
            return []
        
        min_date = dates.min()
        max_date = dates.max()
        
        windows = []
        current_start = min_date
        
        while True:
            train_end = current_start + pd.DateOffset(months=self.config.train_months)
            
            # 清洗期
            purge_end = train_end + pd.DateOffset(days=self.config.purge_days)
            
            test_start = purge_end
            test_end = test_start + pd.DateOffset(months=self.config.test_months)
            
            if test_end > max_date:
                break
            
            # 计算实际样本数
            train_dates = dates[(dates >= current_start) & (dates <= train_end)]
            test_dates = dates[(dates >= test_start) & (dates <= test_end)]
            
            if len(train_dates) >= self.config.min_train_samples and len(test_dates) > 0:
                windows.append((
                    pd.Timestamp(current_start),
                    pd.Timestamp(train_end),
                    pd.Timestamp(test_start),
                    pd.Timestamp(test_end)
                ))
            
            current_start += pd.DateOffset(months=self.config.step_months)
        
        return windows


class RollingWindowBacktest:
    """
    滚动窗口回测引擎
    
    借鉴 Qlib 的回测设计：
    1. 将数据按时间分成多个滚动窗口
    2. 每个窗口内：训练模型 → 预测 → 评估
    3. 合并所有窗口的预测结果，计算整体绩效指标
    4. 支持 purged grouping 防止数据泄露
    """
    
    def __init__(self, config: RollingWindowConfig = None):
        self.config = config or RollingWindowConfig()
        self.splitter = RollingWindowSplitter(self.config)
        self.results: List[WindowResult] = []
    
    def run(self, 
            data: pd.DataFrame,
            feature_cols: List[str],
            label_col: str,
            model_factory: callable = None,
            date_col: str = 'date',
            code_col: str = 'code') -> Dict[str, Any]:
        """
        运行滚动窗口回测
        
        参数:
            data: 包含特征和标签的 DataFrame
            feature_cols: 特征列名列表
            label_col: 标签列名
            model_factory: 返回新模型实例的可调用对象 (默认 LinearRegression)
            date_col: 日期列名
            code_col: 股票代码列名
        
        返回:
            {
                'window_results': [WindowResult, ...],
                'overall_metrics': dict,
                'merged_predictions': DataFrame,
            }
        """
        if model_factory is None:
            model_factory = lambda: LinearRegression()
        
        windows = self.splitter.generate_windows(data, date_col)
        self.results = []
        
        all_predictions = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # 分割数据
            train_mask = (
                (data[date_col] >= train_start) & 
                (data[date_col] <= train_end)
            )
            test_mask = (
                (data[date_col] >= test_start) & 
                (data[date_col] <= test_end)
            )
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            if train_data.empty or test_data.empty:
                continue
            
            # 准备特征和标签
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data[label_col].fillna(0)
            X_test = test_data[feature_cols].fillna(0)
            y_test = test_data[label_col].fillna(0)
            
            # 训练模型
            model = model_factory()
            model.fit(X_train, y_train)
            
            # 预测
            predictions = model.predict(X_test)
            
            # 计算指标
            metrics = self._calc_window_metrics(y_test, predictions)
            
            result = WindowResult(
                window_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_samples=len(train_data),
                test_samples=len(test_data),
                metrics=metrics,
                predictions=predictions,
                actuals=y_test.values
            )
            self.results.append(result)
            
            # 保存预测结果
            pred_df = test_data[[date_col, code_col]].copy()
            pred_df['prediction'] = predictions
            pred_df['actual'] = y_test.values
            pred_df['window_id'] = i
            all_predictions.append(pred_df)
        
        # 合并所有预测
        if all_predictions:
            merged_predictions = pd.concat(all_predictions, ignore_index=True)
        else:
            merged_predictions = pd.DataFrame()
        
        # 计算整体指标
        overall_metrics = self._calc_overall_metrics(merged_predictions)
        
        return {
            'window_results': self.results,
            'overall_metrics': overall_metrics,
            'merged_predictions': merged_predictions,
            'n_windows': len(self.results),
        }
    
    def _calc_window_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """计算单个窗口的绩效指标"""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        
        if len(y_true) < 2:
            return {}
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # IC (Information Coefficient - Pearson)
        ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0
        
        # Rank IC (Spearman)
        from scipy import stats
        rank_ic, _ = stats.spearmanr(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'ic': float(ic),
            'rank_ic': float(rank_ic),
            'n_samples': len(y_true),
        }
    
    def _calc_overall_metrics(self, merged_preds: pd.DataFrame) -> Dict[str, float]:
        """计算整体绩效指标（所有窗口合并）"""
        if merged_preds.empty:
            return {}
        
        # 按日期聚合
        daily_metrics = []
        for date, group in merged_preds.groupby('date'):
            if len(group) < 2:
                continue
            
            y_true = group['actual'].values
            y_pred = group['prediction'].values
            
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid.sum() < 2:
                continue
            
            ic = np.corrcoef(y_true[valid], y_pred[valid])[0, 1]
            daily_metrics.append({
                'date': date,
                'ic': ic,
                'mse': mean_squared_error(y_true[valid], y_pred[valid]),
            })
        
        if not daily_metrics:
            return {}
        
        df = pd.DataFrame(daily_metrics)
        ic_series = df['ic'].dropna()
        
        return {
            'mean_ic': float(ic_series.mean()) if len(ic_series) > 0 else 0,
            'ic_std': float(ic_series.std()) if len(ic_series) > 0 else 0,
            'ic_ir': float(ic_series.mean() / ic_series.std()) if len(ic_series) > 0 and ic_series.std() > 0 else 0,
            'ic_positive_ratio': float((ic_series > 0).mean()) if len(ic_series) > 0 else 0,
            'total_windows': len(self.results),
            'total_dates': len(df),
        }
    
    def check_overfitting(self, threshold_ratio: float = 0.5) -> Dict[str, Any]:
        """
        过拟合检测
        
        借鉴 Qlib 的过拟合检测思路：
        - 对比各窗口的 IC 稳定性
        - IC 波动过大 / IC 衰减过高 → 疑似过拟合
        """
        if len(self.results) < 2:
            return {'is_overfit': False, 'reason': '窗口数不足'}
        
        window_ics = [r.metrics.get('ic', 0) for r in self.results if r.metrics]
        
        if not window_ics:
            return {'is_overfit': False, 'reason': '无有效IC数据'}
        
        ics = np.array(window_ics)
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        
        # 检查 IC 衰减（前半段 vs 后半段）
        mid = len(ics) // 2
        first_half_mean = np.mean(ics[:mid])
        second_half_mean = np.mean(ics[mid:])
        ic_decay = (first_half_mean - second_half_mean) / (abs(first_half_mean) + 1e-8)
        
        # 检查 IC 不稳定度
        ic_cv = abs(std_ic / (abs(mean_ic) + 1e-8))  # 变异系数
        
        is_overfit = ic_decay > threshold_ratio or ic_cv > 2.0
        
        return {
            'is_overfit': is_overfit,
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            'ic_cv': float(ic_cv),
            'ic_decay': float(ic_decay),
            'first_half_ic': float(first_half_mean),
            'second_half_ic': float(second_half_mean),
            'window_ics': [float(x) for x in ics],
        }


# ============================================================
# Purged Group 交叉验证辅助函数
# ============================================================

def purged_group_train_test_split(
    dates: pd.Series,
    train_end_date: pd.Timestamp,
    test_start_date: pd.Timestamp,
    purge_days: int = 5,
    embargo_days: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Purged Group K-Fold 辅助 - 借鉴 Qlib 的 PurgedGroupTimeSeriesSplit
    
    在时序数据上做分组分割，通过 purge（清洗期）和 embargo（禁运期）
    防止训练集和测试集之间的数据泄露。
    """
    dates = pd.to_datetime(dates)
    
    purge_date = train_end_date - timedelta(days=purge_days)
    embargo_start = test_start_date + timedelta(days=embargo_days)
    
    train_mask = dates <= purge_date
    test_mask = (dates >= embargo_start) & (dates <= test_start_date + pd.DateOffset(months=6))
    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    return train_idx, test_idx


# ============================================================
# 回测指标计算
# ============================================================

def calc_backtest_metrics(
    predictions: pd.DataFrame,
    top_k: int = 50,
    freq: str = 'daily',
    benchmark_returns: pd.Series = None
) -> Dict[str, float]:
    """
    计算模拟回测绩效指标
    
    借鉴 Qlib 的回测评估方式：
    - 每日选预测值最高的 top_k 个股票
    - 计算等权组合收益
    - 计算基准对冲后的超额收益
    """
    if predictions.empty:
        return {}
    
    # 每日选 Top-K 股票
    daily_returns = []
    
    for date, group in predictions.groupby('date'):
        if len(group) < top_k:
            continue
        
        top_stocks = group.nlargest(top_k, 'prediction')
        avg_return = top_stocks['actual'].mean()
        daily_returns.append({'date': date, 'portfolio_return': avg_return})
    
    if not daily_returns:
        return {}
    
    ret_df = pd.DataFrame(daily_returns).set_index('date')
    daily_ret = ret_df['portfolio_return']
    
    # 计算指标
    total_return = (1 + daily_ret).prod() - 1
    n_days = len(daily_ret)
    annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
    volatility = daily_ret.std() * np.sqrt(252)
    sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
    
    cumulative = (1 + daily_ret).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    win_rate = (daily_ret > 0).mean()
    
    return {
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar),
        'win_rate': float(win_rate),
        'n_days': n_days,
    }


# ============================================================
# 测试用例
# ============================================================

class TestRollingWindowBacktest(unittest.TestCase):
    """滚动窗口回测测试"""
    
    @classmethod
    def setUpClass(cls):
        """生成模拟测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
        codes = [f'{i:06d}.SH' for i in range(600001, 600051)]  # 50 stocks
        
        rows = []
        for date in dates:
            for code in codes:
                code_idx = codes.index(code)
                
                # 模拟有持续预测能力的特征
                feature_1 = np.sin(date.dayofyear * 0.05 + code_idx * 0.1) + np.random.randn() * 0.1
                feature_2 = np.cos(date.dayofyear * 0.03 + code_idx * 0.07) + np.random.randn() * 0.1
                feature_3 = np.random.randn() * 0.2
                
                # 收益率 = 特征1*0.5 + 特征2*0.3 + 噪声
                future_return = feature_1 * 0.5 + feature_2 * 0.3 + np.random.randn() * 0.1
                
                rows.append({
                    'code': code,
                    'date': date,
                    'feature_1': feature_1,
                    'feature_2': feature_2,
                    'feature_3': feature_3,
                    'future_return': future_return,
                })
        
        cls.test_data = pd.DataFrame(rows)
    
    def test_window_generation(self):
        """测试窗口生成"""
        config = RollingWindowConfig(
            train_months=12,
            test_months=3,
            step_months=3,
            purge_days=5
        )
        splitter = RollingWindowSplitter(config)
        windows = splitter.generate_windows(self.test_data)
        
        self.assertGreater(len(windows), 0)
        
        # 验证窗口顺序
        for i in range(1, len(windows)):
            # 每个后续窗口的起始日期应在前面窗口之后
            self.assertGreater(windows[i][0], windows[i - 1][0])
            
            # train_end 应早于 test_start
            self.assertLess(windows[i][1], windows[i][2])
        
        print(f"\n[窗口生成] 共生成 {len(windows)} 个窗口")
        for i, w in enumerate(windows[:3]):
            print(f"  窗口{i}: train=[{w[0].date()}, {w[1].date()}], test=[{w[2].date()}, {w[3].date()}]")
        if len(windows) > 3:
            print(f"  ... 共 {len(windows)} 个窗口")
    
    def test_rolling_window_backtest_full(self):
        """测试滚动窗口回测完整流程"""
        config = RollingWindowConfig(
            train_months=12,
            test_months=3,
            step_months=6,
            purge_days=5,
            min_train_samples=100
        )
        
        engine = RollingWindowBacktest(config)
        result = engine.run(
            data=self.test_data,
            feature_cols=['feature_1', 'feature_2', 'feature_3'],
            label_col='future_return',
        )
        
        self.assertGreater(result['n_windows'], 0)
        self.assertFalse(result['merged_predictions'].empty)
        self.assertIn('overall_metrics', result)
        
        metrics = result['overall_metrics']
        print(f"\n[滚动窗口回测] 窗口数: {result['n_windows']}")
        print(f"  平均 IC: {metrics.get('mean_ic', 0):.4f}")
        print(f"  IC_IR: {metrics.get('ic_ir', 0):.4f}")
        print(f"  IC 胜率: {metrics.get('ic_positive_ratio', 0):.2%}")
        
        # 验证每个窗口都有合理的结果
        for wr in result['window_results']:
            self.assertGreater(wr.train_samples, 0)
            self.assertGreater(wr.test_samples, 0)
            self.assertIn('ic', wr.metrics)
    
    def test_overfitting_detection(self):
        """测试过拟合检测"""
        config = RollingWindowConfig(
            train_months=12,
            test_months=3,
            step_months=3,
            purge_days=5,
            min_train_samples=100
        )
        
        engine = RollingWindowBacktest(config)
        engine.run(
            data=self.test_data,
            feature_cols=['feature_1', 'feature_2', 'feature_3'],
            label_col='future_return',
        )
        
        of_result = engine.check_overfitting()
        
        self.assertIn('is_overfit', of_result)
        self.assertIn('ic_cv', of_result)
        self.assertIn('ic_decay', of_result)
        
        print(f"\n[过拟合检测]")
        print(f"  是否过拟合: {of_result['is_overfit']}")
        print(f"  平均IC: {of_result.get('mean_ic', 0):.4f}")
        print(f"  IC标准差: {of_result.get('std_ic', 0):.4f}")
        print(f"  IC变异系数: {of_result.get('ic_cv', 0):.4f}")
        print(f"  IC衰减率: {of_result.get('ic_decay', 0):.4f}")
        print(f"  前半段IC: {of_result.get('first_half_ic', 0):.4f}")
        print(f"  后半段IC: {of_result.get('second_half_ic', 0):.4f}")
        
        # 由于使用了真实特征关系，不应过拟合
        # (但如果 CV 过大可能是数据量不足)
        print(f"  窗口IC序列: {[f'{x:.4f}' for x in of_result.get('window_ics', [])]}")
    
    def test_purged_split(self):
        """测试 Purged Group 分割"""
        dates = pd.to_datetime(self.test_data['date'])
        
        train_end = pd.Timestamp('2023-06-30')
        test_start = pd.Timestamp('2023-07-01')
        
        train_idx, test_idx = purged_group_train_test_split(
            dates, train_end, test_start, purge_days=5
        )
        
        # 验证分割正确性
        train_dates = dates.iloc[train_idx]
        test_dates = dates.iloc[test_idx]
        
        # 训练集最大日期 <= train_end - 5天
        self.assertLessEqual(train_dates.max(), train_end - timedelta(days=5))
        
        # 测试集所有日期 >= test_start
        self.assertGreaterEqual(test_dates.min(), test_start)
        
        # 无重叠
        self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
        
        print(f"\n[Purged分割] 训练集: {len(train_idx)} 样本, 日期范围 [{train_dates.min().date()}, {train_dates.max().date()}]")
        print(f"  测试集: {len(test_idx)} 样本, 日期范围 [{test_dates.min().date()}, {test_dates.max().date()}]")
    
    def test_rolling_vs_single_backtest(self):
        """对比滚动窗口与单次回测"""
        config = RollingWindowConfig(
            train_months=12,
            test_months=3,
            step_months=6,
            purge_days=5,
            min_train_samples=100
        )
        
        engine = RollingWindowBacktest(config)
        result_rolling = engine.run(
            data=self.test_data,
            feature_cols=['feature_1', 'feature_2', 'feature_3'],
            label_col='future_return',
        )
        
        # 单次训练（用前36个月训练，后12个月测试）
        data = self.test_data
        all_dates = sorted(data['date'].unique())
        split_idx = int(len(all_dates) * 0.7)
        split_date = all_dates[split_idx]
        
        train_data = data[data['date'] <= split_date]
        test_data = data[data['date'] > split_date]
        
        X_train = train_data[['feature_1', 'feature_2', 'feature_3']].fillna(0)
        y_train = train_data['future_return'].fillna(0)
        X_test = test_data[['feature_1', 'feature_2', 'feature_3']].fillna(0)
        y_test = test_data['future_return'].fillna(0)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        single_preds = model.predict(X_test)
        
        single_ic = np.corrcoef(y_test, single_preds)[0, 1]
        rolling_ic = result_rolling['overall_metrics'].get('mean_ic', 0)
        
        print(f"\n[滚动 vs 单次]")
        print(f"  滚动窗口 IC: {rolling_ic:.4f}")
        print(f"  单次回测 IC: {single_ic:.4f}")
        print(f"  滚动窗口数: {result_rolling['n_windows']}")
        print(f"  单次训练样本: {len(train_data)}, 测试样本: {len(test_data)}")
        
        # 滚动窗口回测的优势：
        # 1. 更真实模拟实际投资中的模型更新过程
        # 2. 提供 IC 时间序列，可评估信号稳定性
        # 3. 支持过拟合检测
        self.assertGreaterEqual(result_rolling['n_windows'], 1,
                              "滚动窗口应至少产生1个窗口")
    
    def test_metric_calculation(self):
        """测试回测指标计算"""
        np.random.seed(42)
        
        # 生成模拟预测结果
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
        codes = [f'{i:06d}.SH' for i in range(600001, 600051)]
        
        rows = []
        for date in dates:
            for code in codes:
                pred = np.random.randn()
                actual = pred * 0.3 + np.random.randn() * 0.1  # 部分预测能力
                rows.append({
                    'date': date,
                    'code': code,
                    'prediction': pred,
                    'actual': actual,
                })
        
        predictions = pd.DataFrame(rows)
        metrics = calc_backtest_metrics(predictions, top_k=10)
        
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        
        print(f"\n[指标计算] Top-10 选股:")
        print(f"  年化收益: {metrics['annual_return']:.2%}")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  胜率: {metrics['win_rate']:.2%}")
        print(f"  总收益率: {metrics['total_return']:.2%}")


if __name__ == '__main__':
    unittest.main(verbosity=2)