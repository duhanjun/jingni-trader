"""
验证测试：增强因子IC分析（Enhanced IC Analysis）

借鉴来源：
  - Microsoft Qlib (https://github.com/microsoft/qlib)
    - 模型评估模块中包含完整的 IC 分析流水线
    - 支持 IC decay、分组 IC、月度 IC 热力图
    - qlib/contrib/evaluate.py 提供多种评估工具

优化方向：
  jingni-trader 当前仅有基础的 IC 计算（均值、标准差、IC_IR、正向率），
  本测试验证扩展 IC 分析维度：
  1. IC Decay 分析（因子预测能力随时间衰减曲线）
  2. 分组 IC 分析（按行业/市值分组）
  3. IC 热力图/稳定性矩阵
  4. 因子换手率分析

注意：本文件仅为验证测试代码，不得合并到主分支。
"""

import os
import sys
import time
import warnings
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：增强 IC 分析工具
# ============================================================================

@dataclass
class ICDecayResult:
    """IC 衰减分析结果"""
    periods: List[int]
    ic_means: List[float]
    ic_stds: List[float]
    ic_irs: List[float]
    ic_positive_ratios: List[float]
    half_life: Optional[int] = None  # IC半衰期（IC均值降到峰值一半的期数）


@dataclass 
class GroupICResult:
    """分组 IC 结果"""
    group_name: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    n_samples: int
    group_values: List[float] = field(default_factory=list)


class EnhancedICAnalyzer:
    """
    增强 IC 分析器
    
    参考 Qlib 的评估模块，提供比基础 IC 计算更丰富的分析维度。
    """
    
    def __init__(self):
        pass
    
    def ic_decay(
        self,
        factor_values: pd.Series,
        returns: pd.DataFrame,
        max_periods: int = 20,
        ic_type: str = 'spearman',
    ) -> ICDecayResult:
        """
        IC 衰减分析
        
        计算因子在不同预测期限下的 IC，绘制衰减曲线。
        
        参数:
            factor_values: 因子值 Series (index 为 (date, code))
            returns: 收益率 DataFrame，包含各期限收益列
            max_periods: 最大预测期限
            ic_type: 'spearman' 或 'pearson'
            
        返回:
            ICDecayResult 对象
        """
        periods = list(range(1, max_periods + 1))
        ic_means, ic_stds, ic_irs, ic_positive_ratios = [], [], [], []
        
        for period in periods:
            ret_col = f'ret_forward_{period}d'
            if ret_col not in returns.columns:
                continue
            
            # 对齐数据
            aligned = pd.DataFrame({
                'factor': factor_values,
                'ret': returns[ret_col]
            }).dropna()
            
            if len(aligned) < 10:
                ic_means.append(0)
                ic_stds.append(0)
                ic_irs.append(0)
                ic_positive_ratios.append(0)
                continue
            
            if ic_type == 'spearman':
                ic, _ = stats.spearmanr(aligned['factor'], aligned['ret'], nan_policy='omit')
            else:
                ic, _ = stats.pearsonr(aligned['factor'].fillna(0), aligned['ret'].fillna(0))
            
            ic_means.append(float(ic) if not np.isnan(ic) else 0)
            ic_stds.append(0)  # 单期 IC 无法计算 std
            ic_irs.append(0)
            ic_positive_ratios.append(1 if ic > 0 else 0)
        
        # 估计半衰期
        half_life = self._estimate_half_life(ic_means)
        
        return ICDecayResult(
            periods=periods[:len(ic_means)],
            ic_means=ic_means,
            ic_stds=ic_stds,
            ic_irs=ic_irs,
            ic_positive_ratios=ic_positive_ratios,
            half_life=half_life,
        )
    
    def _estimate_half_life(self, ic_means: List[float]) -> Optional[int]:
        """估计 IC 半衰期"""
        max_ic = max(abs(x) for x in ic_means)
        if max_ic == 0:
            return None
        
        half_threshold = max_ic / 2
        for i, ic in enumerate(ic_means):
            if abs(ic) < half_threshold:
                return i + 1
        
        return None
    
    def group_ic(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        group_labels: pd.Series,
        ic_type: str = 'spearman',
    ) -> List[GroupICResult]:
        """
        分组 IC 分析
        
        按行业、市值等维度分组计算 IC，发现因子在不同子集中的表现差异。
        
        参数:
            factor_values: 因子值
            forward_returns: 未来收益率
            group_labels: 分组标签（如行业、市值分档）
            ic_type: IC 类型
            
        返回:
            GroupICResult 列表
        """
        aligned = pd.DataFrame({
            'factor': factor_values,
            'ret': forward_returns,
            'group': group_labels,
        }).dropna()
        
        results = []
        for group_name, group_data in aligned.groupby('group'):
            if len(group_data) < 10:
                continue
            
            if ic_type == 'spearman':
                ic, _ = stats.spearmanr(group_data['factor'], group_data['ret'], nan_policy='omit')
            else:
                ic, _ = stats.pearsonr(group_data['factor'].fillna(0), group_data['ret'].fillna(0))
            
            results.append(GroupICResult(
                group_name=str(group_name),
                ic_mean=float(ic) if not np.isnan(ic) else 0,
                ic_std=0,  # 单截面无法计算
                ic_ir=0,
                n_samples=len(group_data),
                group_values=group_data['factor'].tolist(),
            ))
        
        # 按 IC 排序
        results.sort(key=lambda x: abs(x.ic_mean), reverse=True)
        return results
    
    def rolling_ic(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        window: int = 60,
        min_periods: int = 30,
    ) -> pd.Series:
        """
        滚动 IC 分析
        
        计算因子 IC 的滚动窗口序列，评估因子在不同市场环境中的稳定性。
        
        参数:
            factor_df: 包含因子和收益的 DataFrame
            factor_col: 因子列名
            forward_col: 未来收益列名
            window: 滚动窗口大小
            min_periods: 最小样本数
            
        返回:
            滚动 IC 序列
        """
        ic_series = self._calc_daily_ic(factor_df, factor_col, forward_col)
        
        if ic_series is None or ic_series.empty:
            return pd.Series(dtype=float)
        
        rolling_ic = ic_series.rolling(window=window, min_periods=min_periods).mean()
        return rolling_ic
    
    def _calc_daily_ic(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_col: str,
    ) -> Optional[pd.Series]:
        """计算日度 IC 序列"""
        dates = sorted(factor_df['date'].unique())
        
        ic_list = []
        for dt in dates:
            cross = factor_df[factor_df['date'] == dt].dropna(subset=[factor_col, forward_col])
            if len(cross) < 10:
                continue
            
            ic, _ = stats.spearmanr(cross[factor_col], cross[forward_col], nan_policy='omit')
            if not np.isnan(ic):
                ic_list.append({"date": dt, "ic": ic})
        
        if not ic_list:
            return None
        
        ic_df = pd.DataFrame(ic_list)
        ic_df['date'] = pd.to_datetime(ic_df['date'])
        return ic_df.set_index('date')['ic']
    
    def factor_turnover(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        n_groups: int = 5,
    ) -> Dict[str, float]:
        """
        因子换手率分析
        
        计算因子的截面换手率（相邻两期分组的股票变动比例）。
        高换手率因子可能带来高交易成本。
        
        参数:
            factor_df: 因子数据
            factor_cols: 因子列名列表
            n_groups: 分组数量
            
        返回:
            各因子的平均换手率
        """
        turnover_rates = {}
        
        for factor_col in factor_cols:
            if factor_col not in factor_df.columns:
                continue
            
            # 按日期排序
            sorted_df = factor_df.sort_values(['date', 'code'])
            
            # 计算每日分组
            sorted_df['group'] = sorted_df.groupby('date')[factor_col].transform(
                lambda x: pd.qcut(x.rank(method='first'), n_groups, labels=False, duplicates='drop')
                if len(x.dropna()) >= n_groups else np.nan
            )
            
            # 计算相邻日期的分组变动
            sorted_df['group_prev'] = sorted_df.groupby('code')['group'].shift(1)
            sorted_df['changed'] = (sorted_df['group'] != sorted_df['group_prev']).astype(float)
            
            avg_turnover = sorted_df.groupby('date')['changed'].mean().mean()
            turnover_rates[factor_col] = float(avg_turnover) if not np.isnan(avg_turnover) else 0
        
        return turnover_rates
    
    def ic_heatmap_matrix(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        forward_col: str,
        n_periods: int = 12,
    ) -> Dict[str, Any]:
        """
        IC 月度热力图数据
        
        计算各因子在各月份的 IC，用于热力图可视化。
        
        返回:
            包含月度 IC 矩阵的字典
        """
        heatmap_data = {}
        
        for factor_col in factor_cols:
            if factor_col not in factor_df.columns:
                continue
            
            ic_series = self._calc_daily_ic(factor_df, factor_col, forward_col)
            if ic_series is None or ic_series.empty:
                continue
            
            # 按月聚合
            monthly_ic = ic_series.resample('ME').apply(
                lambda x: x.mean() if len(x) > 0 else np.nan
            )
            
            heatmap_data[factor_col] = {
                str(d.date()): float(v) if not (isinstance(v, float) and np.isnan(v)) else None
                for d, v in monthly_ic.dropna().items()
            }
        
        return heatmap_data


# ============================================================================
# 第二部分：测试验证
# ============================================================================

def generate_factor_data(n_stocks: int = 100, n_days: int = 500) -> pd.DataFrame:
    """生成模拟因子数据"""
    np.random.seed(42)
    
    codes = [f"{i:06d}.{'SH' if i % 2 == 0 else 'SZ'}" for i in range(1, n_stocks + 1)]
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    
    rows = []
    for code in codes:
        base_price = np.random.uniform(5, 100)
        price = base_price
        
        # 各因子的基准值
        momentum_base = np.random.uniform(-0.5, 0.5)
        value_base = np.random.uniform(-0.3, 0.3)
        
        for i, date in enumerate(dates):
            daily_return = np.random.normal(0.0005, 0.02)
            price *= (1 + daily_return)
            
            row = {
                'code': code,
                'date': date,
                'close': price,
                'momentum': momentum_base + np.random.normal(0, 0.1),
                'value_factor': value_base + np.random.normal(0, 0.05),
                'volatility_factor': np.random.normal(0, 0.02),
                'quality_factor': np.random.normal(0, 0.03),
                'size_factor': np.random.lognormal(10, 1),
                # 行业（用于分组IC测试）
                'industry': np.random.choice(['金融', '科技', '消费', '制造', '医药', '能源'], 1)[0],
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 生成未来收益
    for period in [1, 3, 5, 10, 20]:
        df[f'ret_forward_{period}d'] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-period) / x - 1
        )
    
    return df


def test_ic_decay():
    """测试：IC 衰减分析"""
    print("\n" + "=" * 70)
    print("测试1: IC 衰减分析")
    print("=" * 70)
    
    data = generate_factor_data(n_stocks=50, n_days=300)
    analyzer = EnhancedICAnalyzer()
    
    # 用 momentum 因子做衰减分析
    factor_vals = data.set_index(['date', 'code'])['momentum']
    returns = data.set_index(['date', 'code'])[[f'ret_forward_{p}d' for p in [1, 3, 5, 10, 20]]]
    
    result = analyzer.ic_decay(factor_vals, returns, max_periods=20)
    
    print(f"\n  Momentum 因子 IC 衰减:")
    print(f"  {'期限':<10} {'IC均值':<12} {'方向':<8}")
    print(f"  {'-' * 30}")
    
    for p, ic in zip(result.periods, result.ic_means):
        direction = "正向" if ic > 0 else "负向"
        bar = "█" * min(int(abs(ic) * 500), 20)
        print(f"  {f'{p}日':<10} {ic:<12.4f} {direction:<8} {bar}")
    
    if result.half_life:
        print(f"\n  IC 半衰期: {result.half_life} 日 (IC降到峰值的50%)")
    else:
        print(f"\n  IC 半衰期: 未衰减到50%以内")
    
    return True


def test_group_ic():
    """测试：分组 IC 分析"""
    print("\n" + "=" * 70)
    print("测试2: 分组 IC 分析（按行业）")
    print("=" * 70)
    
    data = generate_factor_data(n_stocks=100, n_days=200)
    analyzer = EnhancedICAnalyzer()
    
    # 取最新截面
    latest_date = data['date'].max()
    cross_section = data[data['date'] == latest_date].copy()
    
    results = analyzer.group_ic(
        factor_values=cross_section['momentum'],
        forward_returns=cross_section['ret_forward_5d'],
        group_labels=cross_section['industry'],
    )
    
    print(f"\n  截面日期: {latest_date.date()}")
    print(f"  {'行业':<10} {'IC':<12} {'样本数':<10} {'评级':<8}")
    print(f"  {'-' * 40}")
    
    for r in results:
        rating = "强" if abs(r.ic_mean) > 0.1 else ("中" if abs(r.ic_mean) > 0.05 else "弱")
        print(f"  {r.group_name:<10} {r.ic_mean:<12.4f} {r.n_samples:<10} {rating:<8}")
    
    print(f"\n  分组 IC 分析可帮助识别因子在特定行业/板块的有效性")
    return True


def test_rolling_ic():
    """测试：滚动 IC 稳定性"""
    print("\n" + "=" * 70)
    print("测试3: 滚动 IC 稳定性分析")
    print("=" * 70)
    
    data = generate_factor_data(n_stocks=80, n_days=400)
    analyzer = EnhancedICAnalyzer()
    
    rolling = analyzer.rolling_ic(
        data, 'momentum', 'ret_forward_5d',
        window=60, min_periods=30,
    )
    
    if rolling is not None and not rolling.empty:
        # 计算稳定性指标
        positive_ratio = (rolling > 0).mean()
        mean_ic = rolling.mean()
        std_ic = rolling.std()
        
        # 划分市场阶段
        rolling_df = rolling.reset_index()
        rolling_df.columns = ['date', 'rolling_ic']
        
        mid_point = len(rolling_df) // 2
        early_ic = rolling_df['rolling_ic'].iloc[:mid_point].mean()
        late_ic = rolling_df['rolling_ic'].iloc[mid_point:].mean()
        
        print(f"\n  滚动IC统计:")
        print(f"  {'均值:':<15} {mean_ic:.4f}")
        print(f"  {'标准差:':<15} {std_ic:.4f}")
        print(f"  {'正向率:':<15} {positive_ratio:.1%}")
        print(f"  {'IC_IR:':<15} {mean_ic/std_ic if std_ic > 0 else 0:.4f}")
        
        print(f"\n  市场阶段对比:")
        print(f"  {'前期IC均值:':<15} {early_ic:.4f}")
        print(f"  {'后期IC均值:':<15} {late_ic:.4f}")
        print(f"  {'变化:':<15} {(late_ic - early_ic):.4f} {'(衰减)' if late_ic < early_ic else '(改善)'}")
        
        # 稳定性判断
        if positive_ratio > 0.6 and abs(mean_ic / std_ic) > 0.3 if std_ic > 0 else False:
            print(f"\n  [稳定] 因子IC在不同市场环境下保持正向")
        elif positive_ratio > 0.5:
            print(f"\n  [一般] 因子IC存在一定的市场依赖")
        else:
            print(f"\n  [不稳定] 因子IC波动较大，建议谨慎使用")
    
    return True


def test_factor_turnover():
    """测试：因子换手率分析"""
    print("\n" + "=" * 70)
    print("测试4: 因子换手率分析")
    print("=" * 70)
    
    data = generate_factor_data(n_stocks=60, n_days=200)
    analyzer = EnhancedICAnalyzer()
    
    factor_cols = ['momentum', 'value_factor', 'volatility_factor', 'quality_factor']
    turnover_rates = analyzer.factor_turnover(data, factor_cols, n_groups=5)
    
    print(f"\n  {'因子':<20} {'平均换手率':<15} {'交易成本影响':<15}")
    print(f"  {'-' * 50}")
    
    for factor, rate in sorted(turnover_rates.items(), key=lambda x: x[1], reverse=True):
        if rate > 0.5:
            impact = "高"
        elif rate > 0.3:
            impact = "中"
        else:
            impact = "低"
        print(f"  {factor:<20} {rate:<15.2%} {impact:<15}")
    
    print(f"\n  注: 换手率越高，因子的交易成本越大，实际收益可能低于回测结果")
    return True


# ============================================================================
# 第三部分：主入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("增强因子IC分析 验证测试")
    print("借鉴来源: Microsoft Qlib")
    print("优化方向: 扩展IC分析维度（衰减/分组/滚动/换手率）")
    print("=" * 70)
    
    results = {
        "ic_decay": test_ic_decay(),
        "group_ic": test_group_ic(),
        "rolling_ic": test_rolling_ic(),
        "factor_turnover": test_factor_turnover(),
    }
    
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    for test_name, passed in results.items():
        print(f"  {test_name}: {'PASS' if passed else 'FAIL'}")
    
    print()
    print("结论：")
    print("  1. IC 衰减分析可评估因子预测能力的持续性，辅助选择预测期限")
    print("  2. 分组 IC 分析可识别因子在不同行业/市值段的差异化表现")
    print("  3. 滚动 IC 分析可评估因子在不同市场环境下的稳定性")
    print("  4. 因子换手率分析可估算因子的实际交易成本")
    print("  5. 建议将 EnhancedICAnalyzer 集成到 factor-engine 的 ic_analysis 中")
    print("  6. 可进一步支持：IC 衰减曲线拟合、因子拥挤度分析、多因子 IC 相关性矩阵")