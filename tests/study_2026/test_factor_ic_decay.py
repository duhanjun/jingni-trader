"""
优化方向: 因子 IC 衰减分析与分层单调性检验 (IC Decay & Monotonicity)
借鉴来源: FactorHub (github.com/cn-vhql/FactorHub) - 因子评估体系
借鉴亮点: FactorHub 提供完整的因子评估体系，包括 IC 衰减分析、分层单调性检验、
         换手率分析等。当前 jingni-trader 仅止于 IC 均值和 IR 计算。

优化目标: 在 jingni-trader 的 factor-engine 中增加 IC 衰减分析和分层单调性检验，
         提升因子评估的深度和科学性。
"""

import sys
import os
sys.path.insert(0, '/workspace')

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import time
import json


# ============================================================================
# 1. IC 衰减分析 (IC Decay)
# ============================================================================

def ic_decay_analysis(
    factor_data: pd.DataFrame,
    price_data: pd.DataFrame,
    factor_col: str,
    forward_periods: List[int] = None,
    ic_type: str = "spearman",
) -> Dict[str, Any]:
    """
    IC 衰减分析：计算因子在不同前瞻期上的 IC 均值变化

    借鉴 FactorHub 的因子评估体系，IC 衰减曲线能揭示因子预测能力的持续时间。
    理想因子应具有缓慢衰减的 IC 曲线。

    参数:
        factor_data: 因子数据，需包含 code, date, factor_col
        price_data: 价格数据，需包含 code, date, close
        factor_col: 因子列名
        forward_periods: 前瞻期列表，默认 [1, 5, 10, 20, 40, 60]
        ic_type: "spearman" 或 "pearson"

    返回:
        {
            "ic_decay_curve": {period: ic_mean, ...},
            "ic_decay_ir": {period: ic_ir, ...},
            "ic_decay_t_stat": {period: t_stat, ...},
            "half_life_days": int,  # IC 半衰期
            "decay_rate": float,     # 衰减率
        }
    """
    if forward_periods is None:
        forward_periods = [1, 5, 10, 20, 40, 60]

    # 计算各前瞻期的未来收益
    merged = factor_data[['code', 'date', factor_col]].merge(
        price_data[['code', 'date', 'close']],
        on=['code', 'date'],
        how='inner'
    )
    merged = merged.sort_values(['code', 'date'])

    for period in forward_periods:
        merged[f'forward_{period}d'] = merged.groupby('code')['close'].transform(
            lambda x: x.shift(-period) / x - 1
        )

    results = {
        "ic_decay_curve": {},
        "ic_decay_ir": {},
        "ic_decay_t_stat": {},
        "ic_decay_positive_ratio": {},
    }

    for period in forward_periods:
        fwd_col = f'forward_{period}d'
        ic_series = []
        dates = sorted(merged['date'].unique())

        for dt in dates:
            cross = merged[merged['date'] == dt].dropna(subset=[factor_col, fwd_col])
            if len(cross) < 10:
                continue

            if ic_type == "spearman":
                ic, _ = stats.spearmanr(cross[factor_col], cross[fwd_col], nan_policy='omit')
            else:
                ic, _ = stats.pearsonr(cross[factor_col].fillna(0), cross[fwd_col].fillna(0))

            if not np.isnan(ic):
                ic_series.append(ic)

        if ic_series:
            ic_arr = np.array(ic_series)
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr, ddof=1))
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
            pos_ratio = float((ic_arr > 0).mean())

            results["ic_decay_curve"][str(period)] = round(ic_mean, 6)
            results["ic_decay_ir"][str(period)] = round(ic_ir, 4)
            results["ic_decay_t_stat"][str(period)] = round(t_stat, 4)
            results["ic_decay_positive_ratio"][str(period)] = round(pos_ratio, 4)

    # 计算 IC 半衰期
    ic_values = [results["ic_decay_curve"].get(str(p), 0) for p in forward_periods]
    half_life = _estimate_half_life(forward_periods, ic_values)

    # 计算衰减率 (指数拟合)
    decay_rate = _estimate_decay_rate(forward_periods, ic_values)

    results["half_life_days"] = half_life
    results["decay_rate"] = round(decay_rate, 6)

    return results


def _estimate_half_life(periods: List[int], ic_values: List[float]) -> int:
    """估计 IC 半衰期：IC 衰减到初始值一半所需的天数"""
    if len(ic_values) < 2 or abs(ic_values[0]) < 1e-10:
        return None

    half_target = abs(ic_values[0]) / 2
    for i, (p, ic) in enumerate(zip(periods, ic_values)):
        if i > 0 and abs(ic) <= half_target:
            return p
    return periods[-1] * 2  # 未衰减到一半


def _estimate_decay_rate(periods: List[int], ic_values: List[float]) -> float:
    """用指数衰减模型估计衰减率"""
    valid = [(p, v) for p, v in zip(periods, ic_values) if not np.isnan(v) and v != 0]
    if len(valid) < 2:
        return 0.0

    x = np.array([p for p, _ in valid])
    y = np.log(np.abs([v for _, v in valid]))

    try:
        slope, _, _, _, _ = stats.linregress(x, y)
        return -slope  # 正数表示衰减速度
    except Exception:
        return 0.0


# ============================================================================
# 2. 分层单调性检验 (Group Monotonicity Test)
# ============================================================================

def group_monotonicity_test(
    factor_data: pd.DataFrame,
    price_data: pd.DataFrame,
    factor_col: str,
    n_groups: int = 10,
    forward_period: int = 5,
) -> Dict[str, Any]:
    """
    分层单调性检验

    借鉴 FactorHub 的因子评估体系，验证因子值与未来收益之间是否存在单调关系。
    将股票按因子值分成 N 组，检验各组未来收益是否单调递增/递减。

    参数:
        factor_data: 因子数据
        price_data: 价格数据
        factor_col: 因子列名
        n_groups: 分组数量
        forward_period: 前瞻期

    返回:
        {
            "group_returns": {group: mean_return, ...},
            "top_bottom_spread": float,  # 顶组 - 底组收益差
            "is_monotonic": bool,         # 是否单调
            "monotonic_score": float,     # 单调性得分 (0-1)
            "group_ic": float,           # 分组 IC
            "top_group_win_rate": float,  # 顶组胜率
        }
    """
    merged = factor_data[['code', 'date', factor_col]].merge(
        price_data[['code', 'date', 'close']],
        on=['code', 'date'],
        how='inner'
    )
    merged = merged.sort_values(['code', 'date'])
    merged[f'forward_{forward_period}d'] = merged.groupby('code')['close'].transform(
        lambda x: x.shift(-forward_period) / x - 1
    )

    fwd_col = f'forward_{forward_period}d'

    group_returns = {f"G{i+1}": [] for i in range(n_groups)}
    dates = sorted(merged['date'].unique())

    for dt in dates:
        cross = merged[merged['date'] == dt].dropna(subset=[factor_col, fwd_col])
        if len(cross) < n_groups * 3:
            continue

        cross['group'] = pd.qcut(
            cross[factor_col].rank(method='first'),
            q=n_groups,
            labels=[f"G{i+1}" for i in range(n_groups)]
        )

        for group_name in group_returns:
            group_data = cross[cross['group'] == group_name]
            if len(group_data) > 0:
                group_returns[group_name].append(group_data[fwd_col].mean())

    # 计算各组平均收益
    avg_returns = {}
    for group_name, returns_list in group_returns.items():
        if returns_list:
            avg_returns[group_name] = round(float(np.mean(returns_list)), 6)
        else:
            avg_returns[group_name] = 0.0

    # 计算顶底收益差
    top_bottom_spread = avg_returns.get(f"G{n_groups}", 0) - avg_returns.get("G1", 0)

    # 计算单调性得分
    ret_values = [avg_returns.get(f"G{i+1}", 0) for i in range(n_groups)]
    monotonic_score = _calc_monotonic_score(ret_values)

    # 计算分组 IC
    group_ic = _calc_group_ic(ret_values, n_groups)

    # 顶组胜率
    top_returns = group_returns.get(f"G{n_groups}", [])
    top_win_rate = float((np.array(top_returns) > 0).mean()) if top_returns else 0

    return {
        "group_returns": avg_returns,
        "top_bottom_spread": round(top_bottom_spread, 6),
        "is_monotonic": monotonic_score > 0.7,
        "monotonic_score": round(monotonic_score, 4),
        "group_ic": round(group_ic, 4),
        "top_group_win_rate": round(top_win_rate, 4),
    }


def _calc_monotonic_score(values: List[float]) -> float:
    """计算单调性得分：基于 Spearman 秩相关系数"""
    if len(values) < 2:
        return 0.0
    ranks = np.arange(1, len(values) + 1)
    try:
        rho, _ = stats.spearmanr(ranks, values)
        if np.isnan(rho):
            return 0.0
        return abs(rho)
    except Exception:
        return 0.0


def _calc_group_ic(values: List[float], n_groups: int) -> float:
    """计算分组 IC (组序号与组收益的相关性)"""
    if len(values) < 2:
        return 0.0
    group_ranks = np.arange(1, n_groups + 1)[:len(values)]
    try:
        ic, _ = stats.spearmanr(group_ranks, values)
        return float(ic) if not np.isnan(ic) else 0.0
    except Exception:
        return 0.0


# ============================================================================
# 3. 因子换手率分析 (Turnover Analysis)
# ============================================================================

def turnover_analysis(
    factor_data: pd.DataFrame,
    factor_col: str,
    top_pct: float = 0.2,
    n_groups: int = 10,
) -> Dict[str, Any]:
    """
    因子换手率分析

    借鉴 FactorHub 的因子评估体系，分析因子的换手率。
    高换手率意味着因子信号变化频繁，可能导致高交易成本。

    返回:
        {
            "avg_turnover": float,       # 平均换手率
            "avg_spearman_rank_corr": float,  # 平均秩相关性
            "autocorrelation_1d": float,  # 1日自相关
            "autocorrelation_5d": float,  # 5日自相关
            "stability_score": float,     # 稳定性得分 (0-1)
        }
    """
    df = factor_data[['code', 'date', factor_col]].dropna(subset=[factor_col])

    # 计算每日截面排名
    df['rank'] = df.groupby('date')[factor_col].rank(pct=True)

    # 按 top_pct 选股（前 N% 的股票为多头组合）
    df['in_top'] = (df['rank'] >= (1 - top_pct)).astype(int)

    # 计算每日换手率
    dates = sorted(df['date'].unique())
    turnovers = []
    rank_corrs = []

    for i in range(1, len(dates)):
        prev = df[df['date'] == dates[i - 1]][['code', 'in_top', 'rank']]
        curr = df[df['date'] == dates[i]][['code', 'in_top', 'rank']]

        merged = prev.merge(curr, on='code', suffixes=('_prev', '_curr'))
        if len(merged) < 10:
            continue

        # 换手率: 多头组合中买入/卖出比例
        prev_top = merged['in_top_prev'] == 1
        curr_top = merged['in_top_curr'] == 1
        new_entries = (~prev_top & curr_top).sum()
        exits = (prev_top & ~curr_top).sum()
        total_top = max(prev_top.sum(), curr_top.sum())
        if total_top > 0:
            turnover = (new_entries + exits) / (2 * total_top)
            turnovers.append(turnover)

        # 秩相关性
        if len(merged) > 20:
            try:
                rho, _ = stats.spearmanr(merged['rank_prev'], merged['rank_curr'])
                if not np.isnan(rho):
                    rank_corrs.append(rho)
            except Exception:
                pass

    avg_turnover = float(np.mean(turnovers)) if turnovers else 0
    avg_rank_corr = float(np.mean(rank_corrs)) if rank_corrs else 0

    # 自相关：因子值的时间序列自相关
    pivot = df.pivot(index='date', columns='code', values=factor_col)
    pivot_mean = pivot.mean(axis=1).dropna()

    autocorr_1d = float(pivot_mean.autocorr(lag=1)) if len(pivot_mean) > 1 else 0
    autocorr_5d = float(pivot_mean.autocorr(lag=5)) if len(pivot_mean) > 5 else 0

    # 稳定性得分
    stability_score = (avg_rank_corr + autocorr_1d) / 2 if avg_rank_corr and autocorr_1d else 0

    return {
        "avg_turnover": round(avg_turnover, 4),
        "avg_spearman_rank_corr": round(avg_rank_corr, 4),
        "autocorrelation_1d": round(autocorr_1d, 4),
        "autocorrelation_5d": round(autocorr_5d, 4),
        "stability_score": round(stability_score, 4),
    }


# ============================================================================
# 4. 测试代码
# ============================================================================

def generate_test_data(n_stocks: int = 100, n_days: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成模拟因子数据和价格数据"""
    np.random.seed(42)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.bdate_range('2023-01-01', periods=n_days)

    factor_rows = []
    price_rows = []

    # 生成具有真实预测能力的因子
    for code in codes:
        start_price = np.random.uniform(5, 100)
        daily_returns = np.random.normal(0.0005, 0.02, n_days)
        # 模拟因子：与未来收益正相关
        true_alpha = np.random.normal(0, 0.01, n_days)
        prices = start_price * np.exp(np.cumsum(daily_returns + true_alpha * 0.5))

        for i, (date, price) in enumerate(zip(dates, prices)):
            price_rows.append({
                'code': code,
                'date': date,
                'close': price,
                'volume': int(np.random.lognormal(10, 0.5)),
            })
            factor_rows.append({
                'code': code,
                'date': date,
                'alpha_factor': true_alpha[i] + np.random.normal(0, 0.005),
                'noise_factor': np.random.normal(0, 0.01),  # 无预测能力的噪声因子
            })

    factor_df = pd.DataFrame(factor_rows).sort_values(['code', 'date']).reset_index(drop=True)
    price_df = pd.DataFrame(price_rows).sort_values(['code', 'date']).reset_index(drop=True)

    return factor_df, price_df


def test_ic_decay():
    """测试 IC 衰减分析"""
    print("\n" + "=" * 60)
    print("测试 1: IC 衰减分析 (IC Decay)")
    print("=" * 60)

    factor_df, price_df = generate_test_data()

    # 测试有预测能力的因子
    print("\n  [有效因子] alpha_factor:")
    result = ic_decay_analysis(factor_df, price_df, "alpha_factor")
    print(f"    IC 衰减曲线: {result['ic_decay_curve']}")
    print(f"    IC-IR 衰减: {result['ic_decay_ir']}")
    print(f"    半衰期: {result['half_life_days']} 天")
    print(f"    衰减率: {result['decay_rate']}")

    # 测试无预测能力的因子
    print("\n  [噪声因子] noise_factor:")
    result_noise = ic_decay_analysis(factor_df, price_df, "noise_factor")
    print(f"    IC 衰减曲线: {result_noise['ic_decay_curve']}")
    print(f"    半衰期: {result_noise['half_life_days']} 天")

    # 验证：有效因子应有更长的半衰期和更高的 IC
    valid_half_life = result['half_life_days']
    noise_half_life = result_noise['half_life_days']
    valid_ic1 = abs(result['ic_decay_curve'].get('1', 0))
    noise_ic1 = abs(result_noise['ic_decay_curve'].get('1', 0))

    print(f"\n  对比: 有效因子 IC(t+1)={valid_ic1:.4f}, 噪声因子 IC(t+1)={noise_ic1:.4f}")
    if valid_ic1 > noise_ic1:
        print("  ✓ 有效因子 IC 显著高于噪声因子")
    else:
        print("  ✗ 异常：噪声因子 IC 不应高于有效因子")

    return result


def test_group_monotonicity():
    """测试分层单调性检验"""
    print("\n" + "=" * 60)
    print("测试 2: 分层单调性检验 (Group Monotonicity)")
    print("=" * 60)

    factor_df, price_df = generate_test_data()

    # 测试有效因子
    print("\n  [有效因子] alpha_factor:")
    result = group_monotonicity_test(factor_df, price_df, "alpha_factor", n_groups=10)
    print(f"    各组收益: {result['group_returns']}")
    print(f"    顶底收益差: {result['top_bottom_spread']:.4%}")
    print(f"    单调性得分: {result['monotonic_score']:.4f}")
    print(f"    是否单调: {result['is_monotonic']}")
    print(f"    分组 IC: {result['group_ic']:.4f}")
    print(f"    顶组胜率: {result['top_group_win_rate']:.4f}")

    # 测试噪声因子
    print("\n  [噪声因子] noise_factor:")
    result_noise = group_monotonicity_test(factor_df, price_df, "noise_factor", n_groups=10)
    print(f"    顶底收益差: {result_noise['top_bottom_spread']:.4%}")
    print(f"    单调性得分: {result_noise['monotonic_score']:.4f}")
    print(f"    是否单调: {result_noise['is_monotonic']}")

    # 验证：有效因子应有更高的单调性得分
    if result['monotonic_score'] > result_noise['monotonic_score']:
        print("  ✓ 有效因子单调性得分高于噪声因子")
    else:
        print("  ✗ 异常：噪声因子单调性得分不应高于有效因子")

    return result


def test_turnover():
    """测试因子换手率分析"""
    print("\n" + "=" * 60)
    print("测试 3: 因子换手率分析 (Turnover Analysis)")
    print("=" * 60)

    factor_df, price_df = generate_test_data()

    print("\n  [有效因子] alpha_factor:")
    result = turnover_analysis(factor_df, "alpha_factor")
    print(f"    平均换手率: {result['avg_turnover']:.4f}")
    print(f"    平均秩相关性: {result['avg_spearman_rank_corr']:.4f}")
    print(f"    1日自相关: {result['autocorrelation_1d']:.4f}")
    print(f"    5日自相关: {result['autocorrelation_5d']:.4f}")
    print(f"    稳定性得分: {result['stability_score']:.4f}")

    # 验证：换手率应在合理范围
    assert 0 <= result['avg_turnover'] <= 1, "换手率应在 0-1 之间"
    assert -1 <= result['autocorrelation_1d'] <= 1, "自相关应在 -1 到 1 之间"
    print("  ✓ 所有指标在合理范围内")

    return result


def test_full_workflow():
    """测试完整因子评估流程"""
    print("\n" + "=" * 60)
    print("测试 4: 完整因子评估流程")
    print("=" * 60)

    factor_df, price_df = generate_test_data(n_stocks=200, n_days=500)

    factors = ["alpha_factor", "noise_factor"]
    results = {}

    for factor in factors:
        print(f"\n  [{factor}]")
        ic_result = ic_decay_analysis(factor_df, price_df, factor)
        mono_result = group_monotonicity_test(factor_df, price_df, factor)
        turnover_result = turnover_analysis(factor_df, factor)

        # 综合评分
        ic_ir = abs(ic_result['ic_decay_ir'].get('5', 0))
        mono_score = mono_result['monotonic_score']
        stability = turnover_result['stability_score']

        composite_score = (ic_ir * 0.4 + mono_score * 0.4 + stability * 0.2)
        composite_score = round(composite_score, 4)

        results[factor] = {
            "ic_ir_5d": ic_ir,
            "monotonic_score": mono_score,
            "stability_score": stability,
            "composite_score": composite_score,
            "half_life_days": ic_result['half_life_days'],
            "top_bottom_spread": mono_result['top_bottom_spread'],
            "avg_turnover": turnover_result['avg_turnover'],
        }

        print(f"    IC-IR(5d): {ic_ir:.4f}")
        print(f"    单调性得分: {mono_score:.4f}")
        print(f"    稳定性得分: {stability:.4f}")
        print(f"    综合评分: {composite_score:.4f}")

    # 验证：有效因子综合评分应高于噪声因子
    if results['alpha_factor']['composite_score'] > results['noise_factor']['composite_score']:
        print("\n  ✓ 有效因子综合评分高于噪声因子")
    else:
        print("\n  ✗ 异常：噪声因子综合评分不应高于有效因子")

    print(f"\n  完整评估结果 (JSON):")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    return results


if __name__ == "__main__":
    test_ic_decay()
    test_group_monotonicity()
    test_turnover()
    test_full_workflow()
    print("\n" + "=" * 60)
    print("测试完成: IC 衰减分析与分层单调性检验")
    print("=" * 60)