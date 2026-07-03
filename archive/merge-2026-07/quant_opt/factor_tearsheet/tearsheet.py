"""
Factor Tear Sheet Module (Alphalens-inspired)
============================================

借鉴 Alphalens (Quantopian) 的因子分析框架设计，引入以下能力：

1. ``get_clean_factor_and_forward_returns`` 风格的因子数据预处理
   - 多周期 forward returns 计算 (默认 1/5/10 日)
   - 因子值分箱 (quantile 或等宽 bin)
   - z-score 异常值过滤
   - 按组 (industry) 中性化分箱
2. 分位数组合收益分析 (mean return by quantile)
3. 因子换手率 (turnover) 分析
4. 完整 tear sheet 输出 (dict, 可被 reports-engine 复用)

设计原则：
- 纯函数 + DataFrame 接口, 不引入额外依赖 (除 numpy/pandas/scipy)
- 与 jingni-trader 现有数据 schema 兼容 (code/date/... 字段)
- 不修改 main 分支代码, 作为 opt-in 模块存在

References
----------
- Alphalens utils.get_clean_factor_and_forward_returns
  https://quantopian.github.io/alphalens/alphalens.html
- Alphalens tears.create_returns_tear_sheet / create_information_tear_sheet
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------------------
# 1. 因子数据预处理
# --------------------------------------------------------------------------------------

def compute_forward_returns(
    prices: pd.DataFrame,
    periods: Tuple[int, ...] = (1, 5, 10),
    cumulative: bool = True,
) -> pd.DataFrame:
    """
    计算多周期 forward returns.

    Parameters
    ----------
    prices : pd.DataFrame
        必须包含 ``date`` 和 ``code`` 列, 以及 ``close`` 列.
    periods : tuple of int
        前瞻周期, 单位为交易日.  默认 (1, 5, 10).
    cumulative : bool
        True 计算累计收益, False 计算单期收益.

    Returns
    -------
    pd.DataFrame
        包含 ``ret_forward_{period}D`` 列的 DataFrame.
    """
    if "close" not in prices.columns:
        raise ValueError("prices must contain 'close' column")

    df = prices[["code", "date", "close"]].copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    for p in periods:
        if cumulative:
            # 累计收益: close_{t+p} / close_t - 1
            df[f"ret_forward_{p}D"] = (
                df.groupby("code")["close"].transform(lambda x: x.shift(-p) / x - 1)
            )
        else:
            # 单期收益: close_{t+p}/close_{t+p-1} - 1
            df[f"ret_forward_{p}D"] = (
                df.groupby("code")["close"].transform(
                    lambda x: x.pct_change().shift(-p)
                )
            )

    return df


def quantize_factor(
    factor: pd.Series,
    quantiles: int = 5,
    bins: Optional[int] = None,
    zero_aware: bool = False,
    groupby: Optional[pd.Series] = None,
) -> pd.Series:
    """
    将因子值按分位数 (或等宽 bin) 离散化.

    Parameters
    ----------
    factor : pd.Series
        因子原始值.
    quantiles : int
        分位桶数. 0 或 None 表示按 bins 划分.
    bins : int
        等宽分箱数.
    zero_aware : bool
        True 时正负值分别分箱 (借鉴 Alphalens zero_aware).
    groupby : pd.Series
        按组 (行业) 独立分箱, index 与 factor 对齐.

    Returns
    -------
    pd.Series
        每个因子值对应的分桶标签, dtype 为 int.
    """
    s = factor.astype(float)

    def _qcut(x: pd.Series) -> pd.Series:
        if bins is not None and bins > 0:
            return pd.cut(x, bins=bins, labels=False, include_lowest=True)
        # 默认按分位数
        if zero_aware:
            pos = x[x > 0]
            neg = x[x < 0]
            pos_lbl = pd.qcut(pos.rank(method="first"), quantiles, labels=False, duplicates="drop") if len(pos) > quantiles else pd.Series(0, index=pos.index)
            neg_lbl = pd.qcut(neg.rank(method="first"), quantiles, labels=False, duplicates="drop") if len(neg) > quantiles else pd.Series(0, index=neg.index)
            return pd.concat([pos_lbl, neg_lbl]).reindex(x.index).fillna(-1).astype(int)
        return pd.qcut(x.rank(method="first"), quantiles, labels=False, duplicates="drop").fillna(-1).astype(int)

    if groupby is not None:
        out = pd.Series(index=s.index, dtype=int)
        for _, idx in groupby.groupby(groupby).groups.items():
            out.loc[idx] = _qcut(s.loc[idx]).values
        return out
    return _qcut(s)


def get_clean_factor_and_forward_returns(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    quantiles: int = 5,
    bins: Optional[int] = None,
    periods: Tuple[int, ...] = (1, 5, 10),
    groupby: Optional[pd.Series] = None,
    filter_zscore: Optional[float] = 20.0,
    zero_aware: bool = False,
) -> pd.DataFrame:
    """
    一步完成: 合并因子与价格 -> 计算 forward returns -> 分桶 -> 异常值过滤.

    Parameters
    ----------
    factor : pd.DataFrame
        必须包含 ``date`` 和 ``code`` 列, 至少一个因子列.
    prices : pd.DataFrame
        必须包含 ``date`` 和 ``code`` 与 ``close`` 列.
    quantiles : int
        分桶数.
    bins : int
        等宽分箱数 (优先于 quantiles).
    periods : tuple
        forward returns 周期.
    groupby : pd.Series
        行业分组 (index 对齐 factor).
    filter_zscore : float
        过滤超过 |z|>filter_zscore 的极端值. None 关闭.
    zero_aware : bool
        因子正负分别分桶.

    Returns
    -------
    pd.DataFrame
        MultiIndex [date, code], 列为 factor/factor_quantile/ret_forward_*D.
    """
    if factor.empty or prices.empty:
        return pd.DataFrame()

    # 1) 计算 forward returns
    fwd = compute_forward_returns(prices, periods=periods)

    # 2) merge
    factor_col = [c for c in factor.columns if c not in ("code", "date")][0]
    merged = factor.merge(
        fwd[["code", "date"] + [f"ret_forward_{p}D" for p in periods]],
        on=["code", "date"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    # 3) z-score 过滤
    if filter_zscore is not None and filter_zscore > 0:
        vals = merged[factor_col].astype(float)
        z = (vals - vals.mean()) / (vals.std() + 1e-12)
        merged = merged[(z.abs() < filter_zscore)].copy()

    # 4) 分桶
    grp = None
    if groupby is not None and len(groupby) == len(merged):
        grp = groupby.values
    merged["factor_quantile"] = quantize_factor(
        merged[factor_col],
        quantiles=quantiles,
        bins=bins,
        zero_aware=zero_aware,
        groupby=grp,
    )

    # 5) 输出 MultiIndex [date, code]
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.set_index(["date", "code"]).sort_index()
    merged = merged.rename(columns={factor_col: "factor"})

    keep_cols = ["factor", "factor_quantile"] + [f"ret_forward_{p}D" for p in periods]
    return merged[keep_cols]


# --------------------------------------------------------------------------------------
# 2. 分位组合收益分析
# --------------------------------------------------------------------------------------

def mean_return_by_quantile(
    factor_data: pd.DataFrame,
    by_date: bool = True,
) -> pd.DataFrame:
    """
    各分位组合的 forward return 均值.

    借鉴 Alphalens performance.mean_return_by_quantile.
    """
    fwd_cols = [c for c in factor_data.columns if c.startswith("ret_forward_")]
    if not fwd_cols or "factor_quantile" not in factor_data.columns:
        return pd.DataFrame()

    grp_cols = ["factor_quantile"]
    if by_date:
        grp_cols = ["date", "factor_quantile"]

    res = (
        factor_data.groupby(grp_cols)[fwd_cols]
        .mean()
        .reset_index()
    )
    return res


def compute_mean_return_spread(
    mean_returns: pd.DataFrame,
    upper_q: int = 5,
    lower_q: int = 1,
) -> pd.Series:
    """
    long-short 收益差: top quantile - bottom quantile.
    """
    fwd_cols = [c for c in mean_returns.columns if c.startswith("ret_forward_")]
    if not fwd_cols:
        return pd.Series(dtype=float)
    upper = mean_returns[mean_returns["factor_quantile"] == upper_q].set_index("date")[fwd_cols]
    lower = mean_returns[mean_returns["factor_quantile"] == lower_q].set_index("date")[fwd_cols]
    common = upper.index.intersection(lower.index)
    return (upper.loc[common] - lower.loc[common]).mean()


# --------------------------------------------------------------------------------------
# 3. IC (Information Coefficient) 分析
# --------------------------------------------------------------------------------------

def factor_information_coefficient(
    factor_data: pd.DataFrame,
    groupby: Optional[pd.Series] = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    计算每日因子-收益 cross-sectional IC.
    与 jingni-trader 现有 ic_analysis 等价, 但遵循 Alphalens 的返回结构.
    """
    fwd_cols = [c for c in factor_data.columns if c.startswith("ret_forward_")]
    if "factor" not in factor_data.columns or not fwd_cols:
        return pd.DataFrame()

    rows = []
    for dt, grp in factor_data.groupby(level="date"):
        for fwd in fwd_cols:
            sub = grp[["factor", fwd]].dropna()
            if len(sub) < 10:
                continue
            x = sub["factor"].values
            y = sub[fwd].values
            if method == "spearman":
                ic, _ = stats.spearmanr(x, y, nan_policy="omit")
            else:
                ic, _ = stats.pearsonr(x, y)
            if not np.isnan(ic):
                rows.append({"date": dt, "period": fwd, "ic": ic})
    return pd.DataFrame(rows)


def ic_summary(ic_ts: pd.DataFrame) -> Dict[str, float]:
    """
    IC 时间序列汇总: mean / std / IR / t-stat / positive ratio.
    """
    out: Dict[str, float] = {}
    if ic_ts.empty:
        return out
    for period, sub in ic_ts.groupby("period"):
        s = sub["ic"]
        std = float(s.std())
        mean = float(s.mean())
        ir = mean / std if std > 0 else 0.0
        t = mean / (std / np.sqrt(len(s))) if std > 0 else 0.0
        out[period] = {
            "ic_mean": round(mean, 6),
            "ic_std": round(std, 6),
            "ic_ir": round(ir, 4),
            "ic_t_stat": round(t, 4),
            "ic_positive_ratio": round(float((s > 0).mean()), 4),
        }
    return out


# --------------------------------------------------------------------------------------
# 4. Turnover 分析
# --------------------------------------------------------------------------------------

def factor_turnover(
    factor_data: pd.DataFrame,
    quantile: int = 5,
) -> pd.DataFrame:
    """
    计算每个再平衡日 top/bottom quantile 持仓的换手率.
    借鉴 Alphalens performance.factor_turnover.
    """
    if "factor_quantile" not in factor_data.columns:
        return pd.DataFrame()

    df = factor_data.reset_index()
    df = df.sort_values(["date", "code"])

    # 每个 date 选取 top / bottom quantile
    top = df[df["factor_quantile"] == quantile].groupby("date")["code"].apply(set)
    bottom = df[df["factor_quantile"] == 1].groupby("date")["code"].apply(set)

    dates = sorted(set(top.index) | set(bottom.index))
    if len(dates) < 2:
        return pd.DataFrame(columns=["date", "top_turnover", "bottom_turnover"])

    rows = []
    prev_top: set = set()
    prev_bottom: set = set()
    for dt in dates:
        t = top.get(dt, set())
        b = bottom.get(dt, set())
        top_to = (len(t - prev_top) + len(prev_top - t)) / max(len(t | prev_top), 1) if (t or prev_top) else 0.0
        bot_to = (len(b - prev_bottom) + len(prev_bottom - b)) / max(len(b | prev_bottom), 1) if (b or prev_bottom) else 0.0
        rows.append({"date": dt, "top_turnover": top_to, "bottom_turnover": bot_to})
        prev_top = t
        prev_bottom = b
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. 完整 tear sheet 整合
# --------------------------------------------------------------------------------------

def create_full_tear_sheet(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    quantiles: int = 5,
    periods: Tuple[int, ...] = (1, 5, 10),
    groupby: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """
    一站式因子 tear sheet, 返回 dict (JSON 可序列化),
    可被 reports-engine 直接 render.

    Returns
    -------
    dict with keys:
        - clean_factor_summary: 行/列/覆盖率
        - mean_return_by_quantile: dict[period -> DataFrame]
        - mean_return_spread: dict[period -> dict]
        - ic_timeseries: DataFrame (date, period, ic)
        - ic_summary: dict[period -> dict]
        - turnover: DataFrame
        - periods: list[int]
        - quantiles: int
    """
    out: Dict[str, object] = {
        "clean_factor_summary": {},
        "mean_return_by_quantile": {},
        "mean_return_spread": {},
        "ic_timeseries": None,
        "ic_summary": {},
        "turnover": None,
        "periods": list(periods),
        "quantiles": quantiles,
    }

    clean = get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        quantiles=quantiles,
        periods=periods,
        groupby=groupby,
    )
    if clean.empty:
        return out

    out["clean_factor_summary"] = {
        "rows": int(len(clean)),
        "dates": int(clean.index.get_level_values("date").nunique()),
        "assets": int(clean.index.get_level_values("code").nunique()),
    }

    # mean return by quantile
    mrq = mean_return_by_quantile(clean, by_date=True)
    if not mrq.empty:
        for p in periods:
            col = f"ret_forward_{p}D"
            sub = mrq[["date", "factor_quantile", col]].dropna()
            out["mean_return_by_quantile"][col] = sub
        # quantiles=N -> 桶 0..N-1
        spread = compute_mean_return_spread(mrq, upper_q=quantiles - 1, lower_q=0)
        for p in periods:
            col = f"ret_forward_{p}D"
            if col in spread.index:
                out["mean_return_spread"][col] = {
                    "long_short_mean": round(float(spread[col]), 6),
                }

    # IC
    ic_ts = factor_information_coefficient(clean, method="spearman")
    out["ic_timeseries"] = ic_ts
    out["ic_summary"] = ic_summary(ic_ts)

    # turnover: top=N-1, bottom=0
    out["turnover"] = factor_turnover(clean, quantile=quantiles - 1)

    return out
