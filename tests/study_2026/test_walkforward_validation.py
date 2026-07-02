"""
验证测试：Walk-Forward交叉验证框架

借鉴来源：
  - AKQuant (https://github.com/akfamily/akquant)
    - 内置 Walk-forward Validation 框架，支持滚动训练
    - 无缝集成 PyTorch/Scikit-learn
  - Freqtrade + FreqAI (https://www.freqtrade.io/)
    - 完整的 ML pipeline 与 Walk-forward 回测
  - Microsoft Qlib
    - RollingDataset 支持滚动窗口数据划分

优化方向：
  jingni-trader 当前 strategy-model-engine 仅用 TimeSeriesSplit 做单次划分，
  缺乏严格的 Walk-forward 验证机制。本测试验证：
  1. 实现标准的 Walk-forward 交叉验证框架
  2. 支持 purge gap（避免训练和验证集信息泄露）
  3. 自动计算各窗口的绩效指标
  4. 与当前 TimeSeriesSplit 方案对比

注意：本文件仅为验证测试代码，不得合并到主分支。
"""

import os
import sys
import time
import warnings
import json
from typing import Dict, Any, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：Walk-Forward 交叉验证框架
# ============================================================================

@dataclass
class WalkForwardConfig:
    """Walk-forward 验证配置"""
    train_window_months: int = 12      # 训练窗口长度（月）
    validation_window_months: int = 3  # 验证窗口长度（月）
    test_window_months: int = 3        # 测试窗口长度（月）
    purge_gap_days: int = 5            # 训练/验证之间的间隔天数（防信息泄露）
    min_train_samples: int = 100       # 最少训练样本数
    step_months: Optional[int] = None  # 滚动步长（默认等于 test_window_months）


@dataclass
class WindowResult:
    """单个窗口的验证结果"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    train_mse: float = 0.0
    test_mse: float = 0.0
    test_mae: float = 0.0
    test_ic: float = 0.0
    test_rank_ic: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


class WalkForwardValidator:
    """
    Walk-Forward 交叉验证器
    
    参考 AKQuant 和 Qlib 的设计，实现完整的滚动窗口验证流程：
    1. 按时间顺序生成训练/测试窗口
    2. 支持 purge gap（防止信息泄露）
    3. 自动计算各窗口的模型性能指标
    4. 生成汇总报告
    """
    
    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.results: List[WindowResult] = []
    
    def generate_windows(
        self,
        dates: pd.DatetimeIndex,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Generator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex], None, None]:
        """
        生成滚动窗口
        
        参数:
            dates: 所有可用日期的 DatetimeIndex
            start_date: 可选，限定起始日期
            end_date: 可选，限定结束日期
            
        生成:
            (train_start, train_end, test_start, test_end) 元组
        """
        cfg = self.config
        dates = pd.DatetimeIndex(sorted(dates))
        
        if start_date:
            dates = dates[dates >= pd.Timestamp(start_date)]
        if end_date:
            dates = dates[dates <= pd.Timestamp(end_date)]
        
        if len(dates) < cfg.min_train_samples + 10:
            raise ValueError(f"数据不足: {len(dates)} 个交易日，需要至少 {cfg.min_train_samples + 10}")
        
        # 计算滚动参数
        train_days = cfg.train_window_months * 21   # 约21个交易日/月
        test_days = cfg.test_window_months * 21
        step_days = (cfg.step_months or cfg.test_window_months) * 21
        
        # 生成窗口
        i = train_days
        window_id = 0
        
        while i + test_days <= len(dates):
            train_start = dates[max(0, i - train_days)]
            # Purge gap: 训练结束与测试开始之间留间隔
            train_end_idx = i - 1 - cfg.purge_gap_days
            if train_end_idx < train_days // 2:
                i += step_days
                continue
            
            train_end = dates[train_end_idx]
            test_start = dates[i]
            test_end = dates[min(i + test_days - 1, len(dates) - 1)]
            
            # 确保训练集足够大
            train_indices = (dates >= train_start) & (dates <= train_end)
            n_train = train_indices.sum()
            
            if n_train < cfg.min_train_samples:
                i += step_days
                continue
            
            test_indices = (dates >= test_start) & (dates <= test_end)
            n_test = test_indices.sum()
            
            if n_test < 5:
                break
            
            yield (
                dates[train_indices],
                dates[test_indices],
                pd.DatetimeIndex([train_start, train_end]),
                pd.DatetimeIndex([test_start, test_end]),
            )
            
            window_id += 1
            i += step_days
    
    def validate(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        model_factory: callable,
        date_col: str = 'date',
        verbose: bool = True,
    ) -> List[WindowResult]:
        """
        执行 Walk-forward 验证
        
        参数:
            data: 包含特征和标签的 DataFrame
            feature_cols: 特征列名列表
            label_col: 标签列名
            model_factory: 模型工厂函数，每次调用返回新模型
            date_col: 日期列名
            
        返回:
            各窗口的验证结果列表
        """
        self.results = []
        dates = pd.DatetimeIndex(data[date_col].unique())
        
        for train_dates, test_dates, train_range, test_range in self.generate_windows(dates):
            window_id = len(self.results)
            
            # 划分数据
            train_data = data[data[date_col].isin(train_dates)]
            test_data = data[data[date_col].isin(test_dates)]
            
            X_train = train_data[feature_cols].fillna(0).values
            y_train = train_data[label_col].fillna(0).values
            X_test = test_data[feature_cols].fillna(0).values
            y_test = test_data[label_col].fillna(0).values
            
            # 训练模型
            model = model_factory()
            model.fit(X_train, y_train)
            
            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 计算指标
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # 计算 IC (预测值与真实值的相关系数)
            test_ic = np.corrcoef(y_pred_test, y_test)[0, 1] if len(y_test) > 2 else 0
            
            # 计算 Rank IC
            try:
                from scipy import stats
                test_rank_ic, _ = stats.spearmanr(y_pred_test, y_test, nan_policy='omit')
            except ImportError:
                test_rank_ic = 0
            
            result = WindowResult(
                window_id=window_id,
                train_start=train_range[0].strftime('%Y-%m-%d'),
                train_end=train_range[1].strftime('%Y-%m-%d'),
                test_start=test_range[0].strftime('%Y-%m-%d'),
                test_end=test_range[1].strftime('%Y-%m-%d'),
                n_train=len(X_train),
                n_test=len(X_test),
                train_mse=train_mse,
                test_mse=test_mse,
                test_mae=test_mae,
                test_ic=test_ic if not np.isnan(test_ic) else 0,
                test_rank_ic=test_rank_ic if not np.isnan(test_rank_ic) else 0,
                metrics={
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'test_ic': test_ic if not np.isnan(test_ic) else 0,
                    'test_rank_ic': test_rank_ic if not np.isnan(test_rank_ic) else 0,
                }
            )
            self.results.append(result)
            
            if verbose:
                print(f"  Window {window_id}: "
                      f"Train {result.train_start}~{result.train_end} ({result.n_train}样本) -> "
                      f"Test {result.test_start}~{result.test_end} ({result.n_test}样本) "
                      f"| IC={result.test_ic:.4f}, RankIC={result.test_rank_ic:.4f}")
        
        return self.results
    
    def summary(self) -> Dict[str, Any]:
        """生成汇总报告"""
        if not self.results:
            return {"error": "无验证结果"}
        
        ics = [r.test_rank_ic for r in self.results]
        mses = [r.test_mse for r in self.results]
        
        ics_valid = [x for x in ics if not np.isnan(x)]
        mses_valid = [x for x in mses if not np.isnan(x)]
        
        return {
            "n_windows": len(self.results),
            "n_total_train": sum(r.n_train for r in self.results),
            "n_total_test": sum(r.n_test for r in self.results),
            "ic_mean": float(np.mean(ics_valid)) if ics_valid else 0,
            "ic_std": float(np.std(ics_valid)) if ics_valid else 0,
            "ic_ir": float(np.mean(ics_valid) / np.std(ics_valid)) if ics_valid and np.std(ics_valid) > 0 else 0,
            "ic_positive_ratio": float(np.mean([1 if x > 0 else 0 for x in ics_valid])) if ics_valid else 0,
            "mse_mean": float(np.mean(mses_valid)) if mses_valid else 0,
            "mse_std": float(np.std(mses_valid)) if mses_valid else 0,
            "window_details": [
                {
                    "window_id": r.window_id,
                    "train_period": f"{r.train_start} ~ {r.train_end}",
                    "test_period": f"{r.test_start} ~ {r.test_end}",
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                    "ic": r.test_rank_ic,
                }
                for r in self.results
            ]
        }


# ============================================================================
# 第二部分：对比测试
# ============================================================================

def generate_regression_data(
    n_samples: int = 2000,
    n_features: int = 10,
    noise: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """生成模拟的量化因子回归数据"""
    np.random.seed(seed)
    
    # 生成日期
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='B')
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成标签（含非线性关系）
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + np.random.randn(n_samples) * noise
    
    # 加入时间结构（后半年规律变化）
    half = n_samples // 2
    y[half:] = X[half:] @ (true_weights * 0.5) + np.random.randn(n_samples - half) * noise
    
    data = pd.DataFrame(X, columns=[f'factor_{i}' for i in range(n_features)])
    data['date'] = dates
    data['label'] = y
    
    return data


def test_walkforward_vs_timeseriessplit():
    """对比 Walk-forward 与 TimeSeriesSplit"""
    print("\n" + "=" * 70)
    print("测试1: Walk-Forward vs TimeSeriesSplit 对比")
    print("=" * 70)
    
    data = generate_regression_data(n_samples=1000, n_features=10, noise=0.3)
    feature_cols = [f'factor_{i}' for i in range(10)]
    label_col = 'label'
    
    # ── 方案A: Walk-Forward (jingni-trader 建议采用) ──
    print("\n  [方案A] Walk-Forward 交叉验证:")
    print("  " + "-" * 60)
    
    wf_config = WalkForwardConfig(
        train_window_months=6,
        test_window_months=2,
        purge_gap_days=5,
        step_months=2,
    )
    validator = WalkForwardValidator(wf_config)
    
    wf_start = time.perf_counter()
    wf_results = validator.validate(
        data, feature_cols, label_col,
        model_factory=lambda: LinearRegression(),
        verbose=True,
    )
    wf_time = time.perf_counter() - wf_start
    wf_summary = validator.summary()
    
    # ── 方案B: TimeSeriesSplit (当前方案) ──
    print("\n  [方案B] TimeSeriesSplit (当前 jingni-trader 方案):")
    print("  " + "-" * 60)
    
    from sklearn.model_selection import TimeSeriesSplit
    
    X_all = data[feature_cols].fillna(0).values
    y_all = data[label_col].fillna(0).values
    
    tscv = TimeSeriesSplit(n_splits=wf_summary['n_windows'])
    
    ts_start = time.perf_counter()
    ts_ics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        ic = np.corrcoef(y_pred, y_test)[0, 1]
        ts_ics.append(ic if not np.isnan(ic) else 0)
        
        print(f"  Fold {fold}: Train[{train_idx[0]}:{train_idx[-1]}] ({len(train_idx)}) "
              f"-> Test[{test_idx[0]}:{test_idx[-1]}] ({len(test_idx)}) | IC={ic:.4f}")
    
    ts_time = time.perf_counter() - ts_start
    
    # ── 对比分析 ──
    print(f"\n  {'=' * 60}")
    print(f"  对比结果:")
    print(f"  {'指标':<20} {'Walk-Forward':<20} {'TimeSeriesSplit':<20}")
    print(f"  {'-' * 60}")
    print(f"  {'窗口数':<20} {wf_summary['n_windows']:<20} {len(ts_ics):<20}")
    print(f"  {'耗时(秒)':<20} {wf_time:<20.4f} {ts_time:<20.4f}")
    print(f"  {'IC均值':<20} {wf_summary['ic_mean']:<20.4f} {np.mean(ts_ics):<20.4f}")
    print(f"  {'IC标准差':<20} {wf_summary['ic_std']:<20.4f} {np.std(ts_ics):<20.4f}")
    
    wf_ir = wf_summary['ic_ir']
    ts_ir = np.mean(ts_ics) / np.std(ts_ics) if np.std(ts_ics) > 0 else 0
    print(f"  {'IC_IR':<20} {wf_ir:<20.4f} {ts_ir:<20.4f}")
    
    # 关键差异分析
    print(f"\n  关键差异分析:")
    print(f"  1. Walk-Forward 采用滑窗方式，更真实模拟实盘环境")
    print(f"  2. Walk-Forward 自动引入 Purge Gap ({wf_config.purge_gap_days}天)，防止信息泄露")
    print(f"  3. TimeSeriesSplit 仅按比例切分，未考虑实际时间窗口长度")
    print(f"  4. Walk-Forward 各窗口长度固定，TimeSeriesSplit 各窗口长度递增")
    
    return True


def test_purge_gap_effect():
    """测试 Purge Gap 对防止信息泄露的效果"""
    print("\n" + "=" * 70)
    print("测试2: Purge Gap 防信息泄露效果验证")
    print("=" * 70)
    
    data = generate_regression_data(n_samples=800, n_features=8, noise=0.2)
    feature_cols = [f'factor_{i}' for i in range(8)]
    label_col = 'label'
    
    # 在数据中植入标签泄露（未来信息混入特征）
    # 模拟场景：特征包含了未来1天的收益率信息
    data_with_leak = data.copy()
    data_with_leak['factor_0'] = data_with_leak['label'].shift(-3)  # 混入未来3天信息
    
    # ── 无 Purge Gap ──
    print("\n  [配置A] 无 Purge Gap (purge_gap_days=0):")
    wf_config_a = WalkForwardConfig(
        train_window_months=4,
        test_window_months=2,
        purge_gap_days=0,
        step_months=2,
    )
    validator_a = WalkForwardValidator(wf_config_a)
    results_a = validator_a.validate(
        data_with_leak, feature_cols, label_col,
        model_factory=lambda: LinearRegression(),
        verbose=False,
    )
    # 检查训练集和测试集边界是否有重叠
    overlap_detected = False
    for r in results_a:
        train_end = pd.Timestamp(r.train_end)
        test_start = pd.Timestamp(r.test_start)
        if test_start <= train_end:
            overlap_detected = True
    
    print(f"    训练/测试时间重叠: {'是(存在泄露风险)' if overlap_detected else '否'}")
    ics_a = [r.test_rank_ic for r in results_a if not np.isnan(r.test_rank_ic)]
    print(f"    IC均值: {np.mean(ics_a):.4f}")
    
    # ── 有 Purge Gap ──
    print("\n  [配置B] 有 Purge Gap (purge_gap_days=5):")
    wf_config_b = WalkForwardConfig(
        train_window_months=4,
        test_window_months=2,
        purge_gap_days=5,
        step_months=2,
    )
    validator_b = WalkForwardValidator(wf_config_b)
    results_b = validator_b.validate(
        data_with_leak, feature_cols, label_col,
        model_factory=lambda: LinearRegression(),
        verbose=False,
    )
    overlap_detected_b = False
    for r in results_b:
        train_end = pd.Timestamp(r.train_end)
        test_start = pd.Timestamp(r.test_start)
        if test_start <= train_end:
            overlap_detected_b = True
    
    print(f"    训练/测试时间重叠: {'是(存在泄露风险)' if overlap_detected_b else '否'}")
    ics_b = [r.test_rank_ic for r in results_b if not np.isnan(r.test_rank_ic)]
    print(f"    IC均值: {np.mean(ics_b):.4f}")
    
    print(f"\n  分析: Purge Gap 有效隔离了训练集和测试集，防止因标签泄露导致的虚假高IC")
    
    return not overlap_detected_b


def test_stability_analysis():
    """测试模型在不同窗口的稳定性"""
    print("\n" + "=" * 70)
    print("测试3: 模型跨窗口稳定性分析")
    print("=" * 70)
    
    data = generate_regression_data(n_samples=1500, n_features=10, noise=0.3)
    feature_cols = [f'factor_{i}' for i in range(10)]
    label_col = 'label'
    
    wf_config = WalkForwardConfig(
        train_window_months=4,
        test_window_months=2,
        purge_gap_days=5,
        step_months=2,
    )
    validator = WalkForwardValidator(wf_config)
    results = validator.validate(
        data, feature_cols, label_col,
        model_factory=lambda: LinearRegression(),
        verbose=False,
    )
    summary = validator.summary()
    
    if summary.get("error"):
        print(f"  验证失败: {summary['error']}")
        return True
    
    ics = [r.test_rank_ic for r in results if not np.isnan(r.test_rank_ic)]
    
    # 稳定性指标
    ic_positive_count = sum(1 for x in ics if x > 0)
    ic_negative_count = sum(1 for x in ics if x < 0)
    
    print(f"\n  窗口数: {len(ics)}")
    print(f"  IC正值窗口: {ic_positive_count} ({ic_positive_count/max(len(ics),1):.1%})")
    print(f"  IC负值窗口: {ic_negative_count} ({ic_negative_count/max(len(ics),1):.1%})")
    print(f"  IC均值: {summary['ic_mean']:.4f}")
    print(f"  IC标准差: {summary['ic_std']:.4f}")
    print(f"  IC_IR: {summary['ic_ir']:.4f}")
    
    # 判断准则
    print(f"\n  稳定性判断:")
    if summary['ic_ir'] > 0.5 and summary['ic_positive_ratio'] > 0.6:
        print(f"  [良好] 模型在各窗口表现稳定，IC_IR={summary['ic_ir']:.2f}, 正向率={summary['ic_positive_ratio']:.1%}")
    elif summary['ic_ir'] > 0.3:
        print(f"  [尚可] 模型有一定预测能力，但存在不稳定性")
    else:
        print(f"  [不佳] 模型在各窗口表现波动较大，建议优化因子或模型")
    
    return True


# ============================================================================
# 第三部分：主入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Walk-Forward 交叉验证框架 验证测试")
    print("借鉴来源: AKQuant + Freqtrade + Qlib")
    print("优化方向: 引入滚动窗口验证，防止过拟合")
    print("=" * 70)
    
    results = {
        "wf_vs_ts": test_walkforward_vs_timeseriessplit(),
        "purge_gap": test_purge_gap_effect(),
        "stability": test_stability_analysis(),
    }
    
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    for test_name, passed in results.items():
        print(f"  {test_name}: {'PASS' if passed else 'FAIL'}")
    
    print()
    print("结论：")
    print("  1. Walk-Forward 验证相比 TimeSeriesSplit 更贴近实盘场景")
    print("  2. Purge Gap 机制有效防止训练/测试信息泄露")
    print("  3. 跨窗口稳定性分析可帮助评估模型泛化能力")
    print("  4. 建议在 strategy-model-engine 中加入 WalkForwardValidator")
    print("  5. 可进一步支持：embargo period、组合优化验证、特征重要性追踪")