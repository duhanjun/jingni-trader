"""
向量化因子分析引擎 (Vectorized Factor Analysis)

借鉴来源:
  - Qlib: 因子 IC 分析的批量化处理
  - Alphalens: 分层回测（quantile returns）与 IC 统计的标准方法
  - VectorBT: 向量化思想替代逐日 Python 循环

优化点:
  1. 原 _calc_ic 对每个日期循环调用 scipy.stats.spearmanr，O(D) 次 Python 调用
     → 改为 groupby + rank 向量化，一次性计算全部日期的 Rank IC
  2. 原 neutralize 对每个日期循环拟合 LinearRegression
     → 改为按日期分组的最小二乘向量化求解（numpy lstsq）
  3. 新增分层回测（quantile returns）能力，借鉴 Alphalens

该模块为独立实现，不修改 main 分支的 factor-engine/engine.py。
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class VectorizedFactorAnalysis:
    """向量化因子分析引擎"""

    @staticmethod
    def calc_ic_series(
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        ic_type: str = "spearman",
        min_samples: int = 10,
    ) -> pd.Series:
        """
        向量化计算单因子的 IC 时间序列

        原实现: for dt in dates: scipy.stats.spearmanr(...)  → O(D) 次 Python 调用
        本实现: groupby('date') 一次性 rank + 相关，全部向量化

        参数:
            factor_df: 含 code, date, factor_col
            forward_returns: 含 code, date, forward_col
            factor_col: 因子列名
            forward_col: 远期收益列名
            ic_type: "spearman" (Rank IC) 或 "pearson"
            min_samples: 每个截面最少样本数

        返回:
            pd.Series, index=date, values=ic
        """
        merged = factor_df[['code', 'date', factor_col]].merge(
            forward_returns[['code', 'date', forward_col]],
            on=['code', 'date'], how='inner'
        ).dropna(subset=[factor_col, forward_col])

        if merged.empty:
            return pd.Series(dtype=float)

        # 按日期分组
        grouped = merged.groupby('date')

        # 过滤样本数不足的日期
        counts = grouped.size()
        valid_dates = counts[counts >= min_samples].index
        merged = merged[merged['date'].isin(valid_dates)]
        if merged.empty:
            return pd.Series(dtype=float)

        if ic_type == "spearman":
            # Rank IC: 对每个日期截面做 rank，再算 Pearson 相关
            merged['_f_rank'] = merged.groupby('date')[factor_col].rank()
            merged['_r_rank'] = merged.groupby('date')[forward_col].rank()
            x_col, y_col = '_f_rank', '_r_rank'
        else:
            x_col, y_col = factor_col, forward_col

        # 向量化计算每个日期的相关系数
        # 公式: corr = cov(x,y) / (std(x)*std(y))
        # 用 groupby + transform 计算 demean 后的乘积
        g = merged.groupby('date')
        x = merged[x_col]
        y = merged[y_col]
        x_mean = g[x_col].transform('mean')
        y_mean = g[y_col].transform('mean')
        dx = x - x_mean
        dy = y - y_mean

        merged['_dxdy'] = dx * dy
        merged['_dx2'] = dx ** 2
        merged['_dy2'] = dy ** 2

        agg = merged.groupby('date').agg(
            cov=('_dxdy', 'mean'),
            std_x=('_dx2', 'mean'),
            std_y=('_dy2', 'mean'),
            n=(x_col, 'size'),
        )
        # 注意：这里用 mean 而非 sum，因为 corr 对常数因子不敏感
        # cov = mean(dxdy), var = mean(dx^2)
        denom = np.sqrt(agg['std_x'] * agg['std_y'])
        ic_series = agg['cov'] / denom.replace(0, np.nan)
        ic_series = ic_series.dropna()
        ic_series.name = 'ic'
        return ic_series

    @staticmethod
    def calc_ic_stats(
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_names: List[str],
        forward_periods: Optional[List[str]] = None,
        ic_type: str = "spearman",
    ) -> Dict[str, Any]:
        """
        批量计算多因子 × 多周期的 IC 统计

        返回结构同原 engine.ic_analysis:
            { forward_col: [ {factor, ic_mean, ic_std, ic_ir, ...}, ... ] }
        """
        if forward_periods is None:
            forward_periods = [c for c in forward_returns.columns
                               if c.startswith('ret_forward_')]

        results: Dict[str, List[Dict]] = {}
        for forward_col in forward_periods:
            if forward_col not in forward_returns.columns:
                continue
            ic_list = []
            for factor in factor_names:
                if factor not in factor_df.columns:
                    continue
                ic_series = VectorizedFactorAnalysis.calc_ic_series(
                    factor_df, forward_returns, factor, forward_col, ic_type
                )
                if ic_series.empty:
                    continue
                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
                n = len(ic_series)
                ic_t = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0.0
                ic_list.append({
                    "factor": factor,
                    "forward_period": forward_col,
                    "ic_mean": round(float(ic_mean), 6),
                    "ic_std": round(float(ic_std), 6),
                    "ic_ir": round(float(ic_ir), 4),
                    "ic_positive_ratio": round(float((ic_series > 0).mean()), 4),
                    "ic_t_stat": round(float(ic_t), 4),
                })
            results[forward_col] = ic_list
        return results

    @staticmethod
    def neutralize_vectorized(
        factor_df: pd.DataFrame,
        neutralize_mcap: bool = True,
        neutralize_industry: bool = True,
        min_samples: int = 30,
    ) -> pd.DataFrame:
        """
        向量化因子中性化（行业 + 市值）

        原实现: for dt in dates: LinearRegression().fit(X, y)  → O(D) 次 sklearn 调用
        本实现: 按日期分组，用 numpy 最小二乘一次性求解残差

        要求 factor_df 含: code, date, lncap(可选), industry(可选), [因子列]
        """
        if not neutralize_industry and not neutralize_mcap:
            return factor_df

        if factor_df.empty:
            return factor_df

        result = factor_df.copy()
        factor_cols = [c for c in factor_df.columns
                       if c not in ['code', 'date', 'industry', 'lncap', 'estimated_mv']]

        # 构建设计矩阵的列
        # 行业哑变量需要全样本构建（保持列对齐）
        if neutralize_industry and 'industry' in result.columns:
            industry_dummies = pd.get_dummies(result['industry'], prefix='ind', dtype=float)
            dummy_cols = list(industry_dummies.columns)
            for col in dummy_cols:
                result[col] = industry_dummies[col].values
        else:
            dummy_cols = []

        x_base_cols = []
        if neutralize_mcap and 'lncap' in result.columns:
            x_base_cols.append('lncap')
        x_base_cols += dummy_cols

        if not x_base_cols:
            return factor_df

        # 按日期分组，向量化求解残差
        for factor in factor_cols:
            if factor not in result.columns:
                continue
            neutralized = pd.Series(index=result.index, dtype=float)

            for dt, idx in result.groupby('date').groups.items():
                if len(idx) < min_samples:
                    neutralized.loc[idx] = result.loc[idx, factor]
                    continue

                X = result.loc[idx, x_base_cols].fillna(0).values
                y = result.loc[idx, factor].fillna(0).values

                # 加截距项
                X_with_const = np.column_stack([np.ones(len(X)), X])
                # 最小二乘求解: beta = (X'X)^-1 X'y
                try:
                    beta, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
                    y_pred = X_with_const @ beta
                    residual = y - y_pred
                    neutralized.loc[idx] = residual
                except np.linalg.LinAlgError:
                    neutralized.loc[idx] = y

            result[f"{factor}_neutral"] = neutralized

        # 清理临时哑变量列
        result = result.drop(columns=dummy_cols)
        return result

    @staticmethod
    def quantile_returns(
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        分层回测：按因子值分位组计算各组远期收益

        借鉴 Alphalens 的分层分析。
        返回: DataFrame, index=date, columns=quantile_1..quantile_n, values=组内平均收益
        """
        merged = factor_df[['code', 'date', factor_col]].merge(
            forward_returns[['code', 'date', forward_col]],
            on=['code', 'date'], how='inner'
        ).dropna(subset=[factor_col, forward_col])

        if merged.empty:
            return pd.DataFrame()

        # 按日期截面分位
        merged['quantile'] = merged.groupby('date')[factor_col].transform(
            lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop') + 1
        )
        merged = merged.dropna(subset=['quantile'])
        merged['quantile'] = merged['quantile'].astype(int)

        # 各组平均收益
        quantile_ret = merged.groupby(['date', 'quantile'])[forward_col].mean().unstack('quantile')
        quantile_ret.columns = [f'q{int(c)}' for c in quantile_ret.columns]
        return quantile_ret
