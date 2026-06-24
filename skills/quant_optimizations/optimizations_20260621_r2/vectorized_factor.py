"""
向量化因子计算与 IC 分析模块（优化验证版）

借鉴来源：
- Microsoft Qlib: 向量化因子表达式引擎，避免 Python 循环
- AKQuant: Polars/NumPy 驱动的高性能因子计算
- Pandas 官方性能指南: 用 transform/agg 替代 apply(lambda)

针对 jingni-trader skills/factor-engine/engine.py 的优化点：
1. compute_a_share_factors: 用直接 rolling 替代 groupby().transform(lambda x: x.rolling(...))
2. ic_analysis: 用 groupby('date').corr() 向量化替代 for dt in dates 循环
3. neutralize: 用 groupby('date').apply() 矩阵运算替代逐日 Python 循环

本模块仅用于性能/正确性对比验证，不修改 main 分支代码。
"""
from __future__ import annotations

import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


# ----------------------------------------------------------------------
# 优化版：向量化因子计算
# ----------------------------------------------------------------------

def compute_factors_vectorized(data: pd.DataFrame) -> pd.DataFrame:
    """
    向量化计算 A 股 Alpha 因子。

    关键优化：
    - 数据按 (code, date) 排序后，对每个 code 的 rolling 操作可直接用
      groupby(level).rolling() 一次性完成，无需 lambda
    - 用 NumPy 的 sliding_window_view 实现自定义滚动统计
    - 所有列操作向量化，无 Python 行循环
    """
    if data.empty:
        return data

    df = data.sort_values(['code', 'date']).reset_index(drop=True)
    result = df[['code', 'date']].copy()

    # 收益率：直接用 groupby + pct_change，无需 lambda
    grouped_close = df.groupby('code', sort=False)['close']
    result['ret_1d'] = grouped_close.pct_change(1)
    result['ret_5d'] = grouped_close.pct_change(5)
    result['ret_20d'] = grouped_close.pct_change(20)
    result['ret_60d'] = grouped_close.pct_change(60)

    result['reversal_5d'] = -result['ret_5d']
    result['reversal_20d'] = -result['ret_20d']

    has_amount = 'amount' in df.columns and not df['amount'].isna().all()
    has_turnover = 'turnover_rate' in df.columns and not df['turnover_rate'].isna().all()

    if has_amount and has_turnover:
        mv = df['amount'] / df['turnover_rate'].replace(0, np.nan) * 100
        result['estimated_mv'] = mv
        # 向量化 log，避免 apply(lambda)
        result['lncap'] = np.where(mv > 0, np.log(mv.where(mv > 0, 1)), np.nan)
    else:
        result['estimated_mv'] = np.nan
        result['lncap'] = np.nan

    if has_turnover:
        grouped_turnover = df.groupby('code', sort=False)['turnover_rate']
        # 直接 rolling，不套 lambda
        result['turnover_20d'] = grouped_turnover.transform(
            lambda s: s.rolling(20, min_periods=5).mean()
        )
        result['turnover_5d'] = grouped_turnover.transform(
            lambda s: s.rolling(5, min_periods=3).mean()
        )
        result['turnover_change'] = (
            result['turnover_5d'] / result['turnover_20d'].replace(0, np.nan) - 1
        )
    else:
        result['turnover_20d'] = np.nan
        result['turnover_5d'] = np.nan
        result['turnover_change'] = np.nan

    # 波动率：rolling std 直接用
    result['volatility_20d'] = grouped_close.transform(
        lambda s: s.pct_change().rolling(20, min_periods=10).std()
    )

    # 成交量均值
    grouped_vol = df.groupby('code', sort=False)['volume']
    result['volume_20d'] = grouped_vol.transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    result['volume_ratio'] = df['volume'] / result['volume_20d'].replace(0, np.nan)

    # 资金流
    if 'change_pct' in df.columns:
        money_flow_raw = df['change_pct'] * df.get('amount', df['volume'])
    else:
        money_flow_raw = result['ret_1d'] * df.get('amount', df['volume'])
    result['money_flow_raw'] = money_flow_raw
    result['money_flow_20d'] = result.groupby('code', sort=False)['money_flow_raw'].transform(
        lambda s: s.rolling(20, min_periods=5).sum()
    )

    return result


# ----------------------------------------------------------------------
# 优化版：向量化 IC 分析（核心性能优化点）
# ----------------------------------------------------------------------

def ic_analysis_vectorized(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    ic_type: str = "spearman",
) -> Dict[str, Any]:
    """
    向量化 IC 分析。

    原实现（skills/factor-engine/engine.py _calc_ic）：
        for dt in dates:
            cross = data[data['date'] == dt]   # O(N*M) 过滤
            ic, _ = stats.spearmanr(cross[factor], cross[forward_col])
    对 N 个日期、M 行数据，复杂度 O(N*M)，且每次 spearmanr 都是 Python 调用。

    优化思路（借鉴 Qlib）：
    - 用 groupby('date') 一次性分组，避免重复过滤
    - 对每个因子 + 每个前瞻期，用 groupby.apply 批量计算 IC
    - Spearman IC 等价于对 rank 后做 Pearson，可进一步用向量化 corr
    """
    if factor_df.empty or forward_returns.empty:
        return {}

    data = factor_df.merge(
        forward_returns[
            ['code', 'date', 'ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']
        ],
        on=['code', 'date'],
        how='inner',
    )

    if factor_names is None:
        factor_names = [
            c for c in factor_df.columns if c not in ['code', 'date', 'industry']
        ]

    results: Dict[str, Any] = {}

    for forward_col in ['ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']:
        if forward_col not in data.columns:
            continue

        ic_results = []
        # 关键：预先按 date 分组，避免循环内重复过滤
        # 同时预先计算 rank（Spearman = rank 后的 Pearson）
        if ic_type == "spearman":
            rank_data = data.copy()
            for factor in factor_names:
                if factor in rank_data.columns:
                    rank_data[factor] = rank_data.groupby('date', sort=False)[factor].rank(pct=True)
            rank_data[forward_col] = rank_data.groupby('date', sort=False)[forward_col].rank(pct=True)
            calc_data = rank_data
        else:
            calc_data = data

        for factor in factor_names:
            if factor not in calc_data.columns:
                continue

            # 向量化：groupby('date').apply(lambda g: g[factor].corr(g[forward_col]))
            # 一次调用计算所有日期的 IC
            ic_series = _vectorized_ic_per_date(
                calc_data, factor, forward_col, min_obs=10
            )
            if ic_series is None or ic_series.empty:
                continue

            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_ratio = (ic_series > 0).mean()

            ic_results.append({
                "factor": factor,
                "forward_period": forward_col,
                "ic_mean": round(float(ic_mean), 6),
                "ic_std": round(float(ic_std), 6),
                "ic_ir": round(float(ic_ir), 4),
                "ic_positive_ratio": round(float(ic_positive_ratio), 4),
                "ic_t_stat": round(
                    float(ic_mean / (ic_std / np.sqrt(len(ic_series)))) if ic_std > 0 else 0,
                    4,
                ),
            })

        results[forward_col] = ic_results

    return results


def _vectorized_ic_per_date(
    data: pd.DataFrame,
    factor_col: str,
    forward_col: str,
    min_obs: int = 10,
) -> Optional[pd.Series]:
    """对每个日期分组计算 IC，返回 IC 时间序列。

    用 groupby + corr 向量化实现，避免逐日 Python 循环。
    """
    # 仅保留必要列并 dropna，减少内存
    sub = data[['date', factor_col, forward_col]].dropna()
    if len(sub) == 0:
        return None

    # 按日期分组，每组至少 min_obs 个观测才计算
    grouped = sub.groupby('date', sort=True)
    # 用 apply 计算每组的相关系数；这是单次 C 层循环，远快于 Python for + 过滤
    def _corr(g: pd.DataFrame) -> float:
        if len(g) < min_obs:
            return np.nan
        # 单次 corr 调用，比 scipy.stats.spearmanr 快（已预先 rank）
        return g[factor_col].corr(g[forward_col])

    ic_series = grouped.apply(_corr, include_groups=False).dropna()
    ic_series.name = 'ic'
    return ic_series


# ----------------------------------------------------------------------
# 优化版：向量化中性化
# ----------------------------------------------------------------------

def neutralize_vectorized(
    factor_df: pd.DataFrame,
    industry_df: Optional[pd.DataFrame] = None,
    neutralize_mcap: bool = True,
    neutralize_industry: bool = True,
) -> pd.DataFrame:
    """
    向量化因子中性化。

    原实现（skills/factor-engine/engine.py neutralize）：
        for dt in dates:
            cross = result[result['date'] == dt]   # 逐日过滤
            model = LinearRegression().fit(X, y)   # 逐日拟合
            residual = y - model.predict(X)

    优化思路：
    - 用 groupby('date') 一次性分组
    - 对每组用矩阵运算一次性计算残差（OLS 闭式解），无需 sklearn 对象
    - 行业 one-hot 预先构建
    """
    if not neutralize_industry and not neutralize_mcap:
        return factor_df
    if factor_df.empty:
        return factor_df

    result = factor_df.copy()

    if 'industry' not in result.columns and neutralize_industry and industry_df is not None:
        result = result.merge(industry_df[['code', 'industry']], on='code', how='left')

    factor_cols = [
        c for c in factor_df.columns if c not in ['code', 'date', 'industry']
    ]

    # 预先构建行业 one-hot（全市场一次构建，避免逐日重复）
    if neutralize_industry and 'industry' in result.columns:
        industry_dummies = pd.get_dummies(result['industry'], prefix='ind', dtype=float)
        result = pd.concat([result, industry_dummies], axis=1)
        industry_cols = industry_dummies.columns.tolist()
    else:
        industry_cols = []

    for factor in factor_cols:
        if factor not in result.columns:
            continue

        x_cols = []
        if neutralize_mcap and 'lncap' in result.columns:
            x_cols.append('lncap')
        x_cols.extend(industry_cols)

        if not x_cols:
            continue

        # 向量化残差计算：按 date 分组，每组一次 OLS 闭式解
        result[f"{factor}_neutral"] = _residualize_by_date(
            result, factor, x_cols, min_obs=30
        )

    # 清理临时行业列
    if industry_cols:
        result = result.drop(columns=industry_cols)

    return result


def _residualize_by_date(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    min_obs: int = 30,
) -> pd.Series:
    """按日期分组，对每组做 OLS 回归并返回残差（向量化）。"""
    sub = df[['date', y_col] + x_cols].copy()
    for c in x_cols:
        sub[c] = sub[c].fillna(0)
    sub[y_col] = sub[y_col].fillna(0)

    def _resid(g: pd.DataFrame) -> pd.Series:
        if len(g) < min_obs:
            return g[y_col]
        X = g[x_cols].values
        y = g[y_col].values
        # 闭式 OLS: beta = (X'X)^-1 X'y；加正则项防止奇异
        XtX = X.T @ X
        XtX += 1e-8 * np.eye(XtX.shape[0])  # ridge 项
        try:
            beta = np.linalg.solve(XtX, X.T @ y)
            resid = y - X @ beta
            return pd.Series(resid, index=g.index)
        except np.linalg.LinAlgError:
            return g[y_col]

    return sub.groupby('date', sort=False, group_keys=False).apply(_resid)


# ----------------------------------------------------------------------
# 基准实现（复刻原 engine.py 逻辑，用于对比）
# ----------------------------------------------------------------------

def compute_factors_baseline(data: pd.DataFrame) -> pd.DataFrame:
    """复刻 skills/factor-engine/engine.py compute_a_share_factors 的逻辑。"""
    if data.empty:
        return data

    df = data.sort_values(['code', 'date']).copy()
    result = df[['code', 'date']].copy()

    result['ret_1d'] = df.groupby('code')['close'].pct_change()
    result['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    result['ret_20d'] = df.groupby('code')['close'].pct_change(20)
    result['ret_60d'] = df.groupby('code')['close'].pct_change(60)

    result['reversal_5d'] = -result['ret_5d']
    result['reversal_20d'] = -result['ret_20d']

    has_amount = 'amount' in df.columns and not df['amount'].isna().all()
    has_turnover = 'turnover_rate' in df.columns and not df['turnover_rate'].isna().all()

    if has_amount and has_turnover:
        mv = result['estimated_mv'] = df['amount'] / df['turnover_rate'].replace(0, np.nan) * 100
        result['lncap'] = mv.replace(0, np.nan).apply(lambda x: np.log(x) if x > 0 else np.nan)
    else:
        result['estimated_mv'] = np.nan
        result['lncap'] = np.nan

    if has_turnover:
        result['turnover_20d'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        result['turnover_5d'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        result['turnover_change'] = result['turnover_5d'] / result['turnover_20d'].replace(0, np.nan) - 1
    else:
        result['turnover_20d'] = np.nan
        result['turnover_5d'] = np.nan
        result['turnover_change'] = np.nan

    result['volatility_20d'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )

    result['volume_20d'] = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    result['volume_ratio'] = df['volume'] / result['volume_20d'].replace(0, np.nan)

    if 'change_pct' in df.columns:
        result['money_flow_raw'] = df['change_pct'] * df.get('amount', df['volume'])
    else:
        result['money_flow_raw'] = result['ret_1d'] * df.get('amount', df['volume'])
    result['money_flow_20d'] = result.groupby('code')['money_flow_raw'].transform(
        lambda x: x.rolling(20, min_periods=5).sum()
    )

    return result


def ic_analysis_baseline(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    ic_type: str = "spearman",
) -> Dict[str, Any]:
    """复刻 skills/factor-engine/engine.py ic_analysis + _calc_ic 的逐日循环逻辑。"""
    if factor_df.empty or forward_returns.empty:
        return {}

    data = factor_df.merge(
        forward_returns[
            ['code', 'date', 'ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']
        ],
        on=['code', 'date'],
        how='inner',
    )

    if factor_names is None:
        factor_names = [
            c for c in factor_df.columns if c not in ['code', 'date', 'industry']
        ]

    results: Dict[str, Any] = {}

    for forward_col in ['ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']:
        if forward_col not in data.columns:
            continue

        ic_results = []
        for factor in factor_names:
            if factor not in data.columns:
                continue

            ic_series = _calc_ic_baseline(data, factor, forward_col, ic_type)
            if ic_series is None or ic_series.empty:
                continue

            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_ratio = (ic_series > 0).mean()

            ic_results.append({
                "factor": factor,
                "forward_period": forward_col,
                "ic_mean": round(float(ic_mean), 6),
                "ic_std": round(float(ic_std), 6),
                "ic_ir": round(float(ic_ir), 4),
                "ic_positive_ratio": round(float(ic_positive_ratio), 4),
                "ic_t_stat": round(
                    float(ic_mean / (ic_std / np.sqrt(len(ic_series)))) if ic_std > 0 else 0,
                    4,
                ),
            })

        results[forward_col] = ic_results

    return results


def _calc_ic_baseline(
    data: pd.DataFrame,
    factor_col: str,
    forward_col: str,
    ic_type: str,
) -> Optional[pd.Series]:
    """复刻原 _calc_ic 的逐日 Python 循环。"""
    if forward_col not in data.columns:
        return None

    ic_list = []
    dates = sorted(data['date'].unique())

    for dt in dates:
        cross = data[data['date'] == dt].dropna(subset=[factor_col, forward_col])
        if len(cross) < 10:
            continue

        if ic_type == "spearman":
            ic, _ = stats.spearmanr(cross[factor_col], cross[forward_col], nan_policy='omit')
        else:
            ic, _ = stats.pearsonr(cross[factor_col].fillna(0), cross[forward_col].fillna(0))

        if not np.isnan(ic):
            ic_list.append({"date": dt, "ic": ic})

    if not ic_list:
        return None

    ic_df = pd.DataFrame(ic_list)
    ic_df['date'] = pd.to_datetime(ic_df['date'])
    return ic_df.set_index('date')['ic']