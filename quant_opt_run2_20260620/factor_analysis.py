"""
因子分析与预处理模块

借鉴来源：
  - Alphalens (Quantopian): IC 分析、因子分组收益
    https://github.com/quantopian/alphalens
  - Microsoft Qlib: 因子预处理 (winsorize/standardize/neutralize)
    https://qlib.readthedocs.io/en/latest/component/data.html

与 jingni-trader 现有实现的对比：
  - 现有 factor-engine 仅计算因子值，无预处理、无 IC 分析
  - 本模块提供：
      1. Winsorize 缩尾（去极值）
      2. Standardize 标准化
      3. Neutralize 行业/市值中性化
      4. IC / Rank IC / IC 衰减 / ICIR
      5. 因子分组收益分析

注意：本文件位于 feat/quant-opt-20260620 分支，不修改 main 分支代码。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ============================================================
# 因子预处理
# ============================================================

class FactorPreprocessor:
    """因子预处理流水线"""

    @staticmethod
    def winsorize(
        factor: pd.Series,
        quantile: float = 0.025,
        groupby: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        缩尾处理：将超出 [q, 1-q] 分位数的值截断到边界

        参数:
            factor: 因子值 Series
            quantile: 截尾分位数（默认 2.5%）
            groupby: 若提供，则按组（如日期）分别缩尾
        """
        if groupby is None:
            lo, hi = factor.quantile(quantile), factor.quantile(1 - quantile)
            return factor.clip(lo, hi)
        tmp = pd.DataFrame({"v": factor, "g": groupby})
        bounds = tmp.groupby("g")["v"].quantile([quantile, 1 - quantile]).unstack()
        bounds.columns = ["lo", "hi"]
        merged = tmp.join(bounds, on="g")
        return pd.Series(
            np.where(merged["v"] < merged["lo"], merged["lo"],
                     np.where(merged["v"] > merged["hi"], merged["hi"], merged["v"])),
            index=factor.index,
            name=factor.name,
        )

    @staticmethod
    def standardize(
        factor: pd.Series,
        groupby: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        标准化：z-score (x - mean) / std

        参数:
            factor: 因子值 Series
            groupby: 若提供，则按组分别标准化（截面标准化）
        """
        if groupby is None:
            mu, sigma = factor.mean(), factor.std()
            return (factor - mu) / sigma if sigma > 0 else factor - mu
        tmp = pd.DataFrame({"v": factor, "g": groupby})
        g = tmp.groupby("g")["v"]
        mu = g.transform("mean")
        sigma = g.transform("std").replace(0, np.nan)
        return pd.Series((tmp["v"] - mu) / sigma, index=factor.index, name=factor.name)

    @staticmethod
    def neutralize(
        factor: pd.DataFrame,
        factor_col: str,
        date_col: str = "date",
        industry_col: Optional[str] = "industry",
        market_cap_col: Optional[str] = "market_cap",
    ) -> pd.Series:
        """
        中性化：对行业哑变量 + log(市值) 做线性回归，取残差

        参数:
            factor: 含因子列、日期列、可选行业/市值列的 DataFrame
            factor_col: 因子列名
            industry_col: 行业列名（None 则不中性化行业）
            market_cap_col: 市值列名（None 则不中性化市值）
        返回:
            中性化后的因子 Series（与原 factor 同索引）
        """
        out = pd.Series(index=factor.index, dtype=float, name=factor_col)
        for dt, grp in factor.groupby(date_col):
            y = grp[factor_col].astype(float)
            valid = y.notna()
            if valid.sum() < 3:
                out.loc[grp.index] = y
                continue
            X_cols = []
            if industry_col and industry_col in grp.columns:
                dummies = pd.get_dummies(grp[industry_col], prefix="ind", drop_first=True)
                X_cols.append(dummies.loc[valid])
            if market_cap_col and market_cap_col in grp.columns:
                mc = np.log(grp[market_cap_col].astype(float).clip(lower=1))
                X_cols.append(mc.loc[valid].to_frame("log_mc"))
            if not X_cols:
                out.loc[grp.index] = y
                continue
            X = pd.concat(X_cols, axis=1).fillna(0.0).astype(float)
            yv = y.loc[valid].astype(float)
            # 最小二乘: beta = (X'X)^-1 X'y
            try:
                beta = np.linalg.lstsq(X.values, yv.values, rcond=None)[0]
                resid = yv.values - X.values @ beta
                out.loc[yv.index] = resid
            except np.linalg.LinAlgError:
                out.loc[grp.index] = y
        return out

    def pipeline(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        date_col: str = "date",
        winsorize_q: float = 0.025,
        standardize_flag: bool = True,
        industry_col: Optional[str] = None,
        market_cap_col: Optional[str] = None,
    ) -> pd.Series:
        """标准预处理流水线：缩尾 -> 标准化 -> 中性化"""
        s = factor_df[factor_col].astype(float)
        g = factor_df[date_col]
        s = self.winsorize(s, winsorize_q, g)
        if standardize_flag:
            s = self.standardize(s, g)
        if industry_col or market_cap_col:
            tmp = factor_df.copy()
            tmp[factor_col] = s
            s = self.neutralize(tmp, factor_col, date_col, industry_col, market_cap_col)
            if standardize_flag:
                s = self.standardize(s, g)
        return s


# ============================================================
# 因子 IC 分析
# ============================================================

class FactorICAnalyzer:
    """
    因子 IC (Information Coefficient) 分析

    IC = corr(factor_t, forward_return_{t->t+n})
    Rank IC = spearman(factor_t, forward_return_{t->t+n})
    """

    def __init__(self, n_forward: int = 5):
        self.n_forward = n_forward

    def compute_forward_returns(
        self,
        data: pd.DataFrame,
        price_col: str = "close",
        code_col: str = "code",
        date_col: str = "date",
        periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        计算前向收益率

        参数:
            data: 含 code, date, price 的 DataFrame
            periods: 前向周期列表，默认 [1, 5, 10, 20]

        返回:
            DataFrame: code, date, fwd_ret_1, fwd_ret_5, ...
        """
        if periods is None:
            periods = [1, 5, 10, 20]
        df = data[[code_col, date_col, price_col]].copy()
        df = df.sort_values([code_col, date_col]).reset_index(drop=True)
        for n in periods:
            col = f"fwd_ret_{n}"
            df[col] = df.groupby(code_col)[price_col].transform(
                lambda s: s.shift(-n) / s - 1
            )
        return df

    def compute_ic(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_ret_col: str = "fwd_ret_5",
        date_col: str = "date",
        method: str = "pearson",
    ) -> pd.Series:
        """
        计算每日 IC 序列

        参数:
            factor_df: 含 date, factor_col, forward_ret_col 的 DataFrame
            method: 'pearson' (IC) 或 'spearman' (Rank IC)

        返回:
            每日 IC Series (index=date)
        """
        def _ic(g: pd.DataFrame) -> float:
            valid = g[[factor_col, forward_ret_col]].dropna()
            if len(valid) < 5:
                return np.nan
            if method == "spearman":
                # 手动实现 spearman：rank 后做 pearson，避免依赖 scipy
                x = valid[factor_col].rank()
                y = valid[forward_ret_col].rank()
                return x.corr(y)
            return valid[factor_col].corr(valid[forward_ret_col])

        return factor_df.groupby(date_col).apply(_ic)

    def analyze(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        date_col: str = "date",
        forward_ret_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        完整 IC 分析

        返回:
            {
                "ic_mean": float,          # IC 均值
                "ic_std": float,
                "icir": float,             # IC / IC_std (IC 信息比率)
                "ic_positive_ratio": float,# IC > 0 占比
                "rank_ic_mean": float,
                "rank_icir": float,
                "ic_decay": dict,          # 不同前向周期的 IC 衰减
                "ic_series": pd.Series,    # 每日 IC 序列
            }
        """
        if forward_ret_cols is None:
            forward_ret_cols = [c for c in factor_df.columns if c.startswith("fwd_ret_")]
        if not forward_ret_cols:
            raise ValueError("未找到前向收益率列 (fwd_ret_*)")

        result: Dict[str, Any] = {}
        primary = forward_ret_cols[0]

        ic_series = self.compute_ic(factor_df, factor_col, primary, date_col, "pearson")
        rank_ic_series = self.compute_ic(factor_df, factor_col, primary, date_col, "spearman")

        ic_clean = ic_series.dropna()
        rank_ic_clean = rank_ic_series.dropna()

        result["ic_mean"] = float(ic_clean.mean()) if len(ic_clean) > 0 else float("nan")
        result["ic_std"] = float(ic_clean.std()) if len(ic_clean) > 1 else 0.0
        result["icir"] = float(ic_clean.mean() / ic_clean.std()) if result["ic_std"] > 0 else 0.0
        result["ic_positive_ratio"] = float((ic_clean > 0).mean()) if len(ic_clean) > 0 else 0.0
        result["rank_ic_mean"] = float(rank_ic_clean.mean()) if len(rank_ic_clean) > 0 else float("nan")
        rank_ic_std = float(rank_ic_clean.std()) if len(rank_ic_clean) > 1 else 0.0
        result["rank_icir"] = float(rank_ic_clean.mean() / rank_ic_std) if rank_ic_std > 0 else 0.0

        # IC 衰减
        decay = {}
        for col in forward_ret_cols:
            s = self.compute_ic(factor_df, factor_col, col, date_col, "pearson").dropna()
            decay[col] = float(s.mean()) if len(s) > 0 else float("nan")
        result["ic_decay"] = decay
        result["ic_series"] = ic_series
        result["rank_ic_series"] = rank_ic_series
        return result

    # ------------------------------------------------------------
    # 因子分组收益分析
    # ------------------------------------------------------------
    def quantile_returns(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_ret_col: str = "fwd_ret_5",
        date_col: str = "date",
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        按因子值分位数分组，计算各组前向收益均值

        返回:
            DataFrame: quantile, mean_return, count
        """
        tmp = factor_df[[date_col, factor_col, forward_ret_col]].copy()
        tmp["q"] = tmp.groupby(date_col)[factor_col].transform(
            lambda s: pd.qcut(s, n_quantiles, labels=False, duplicates="drop")
        )
        valid = tmp.dropna(subset=["q"])
        if valid.empty:
            return pd.DataFrame(columns=["quantile", "mean_return", "count"])
        grp = valid.groupby("q")[forward_ret_col]
        return pd.DataFrame({
            "quantile": [f"Q{int(q)+1}" for q in grp.mean().index],
            "mean_return": grp.mean().values,
            "count": grp.count().values,
        })
