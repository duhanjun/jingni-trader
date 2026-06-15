"""
方向 3：因子 IC 稳定性 + Walk-Forward 滚动评估

借鉴：
- qlib.contrib.evaluate.risk_analysis 的 IC 序列计算
- 米筐 rqfactor 的因子生命周期管理
- 经典多因子研究流程：IC、Rank IC、IC Decay、ICIR

目标：
- 在 panel 上对每个因子输出 IC 序列、IC mean/std/IR
- 支持 rank IC (Spearman) 和 normal IC (Pearson)
- Walk-Forward 滚动评估：在 train 段训练因子权重，在 test 段验证
- 自动判断因子是否"过拟合"
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1) IC 分析
# ---------------------------------------------------------------------------
@dataclass
class FactorIC:
    factor: str
    forward: int
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_positive_ratio: float
    ic_t_stat: float
    ic_series: Optional[pd.Series] = None
    rank_ic_series: Optional[pd.Series] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Series 不能直接 JSON 化
        d["ic_series"] = self.ic_series.to_list() if self.ic_series is not None else []
        d["rank_ic_series"] = self.rank_ic_series.to_list() if self.rank_ic_series is not None else []
        d["dates"] = self.ic_series.index.strftime("%Y-%m-%d").tolist() if self.ic_series is not None else []
        return d


def calc_forward_returns(close: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """
    close: date x asset 矩阵
    返回长表 [date, code, ret_forward_Nd] 多个未来收益
    """
    out = []
    for p in periods:
        fut = close.shift(-p) / close - 1
        long = fut.stack(future_stack=True).reset_index()
        long.columns = ["date", "code", f"ret_forward_{p}d"]
        out.append(long.set_index(["date", "code"]))
    return pd.concat(out, axis=1).reset_index()


def cross_sectional_ic(
    factor_panel: pd.DataFrame,   # date x factor 列已经按 code 拉平 → 长表 [date, code, factor]
    fwd_returns_long: pd.DataFrame,  # 长表 [date, code, ret_forward_*d]
    factor_col: str,
    forward_col: str = "ret_forward_5d",
    method: str = "spearman",
) -> pd.Series:
    """
    计算因子的横截面 IC 时间序列
    """
    merged = factor_panel[["date", "code", factor_col]].merge(
        fwd_returns_long[["date", "code", forward_col]], on=["date", "code"], how="inner"
    ).dropna(subset=[factor_col, forward_col])

    ic_by_date = {}
    for dt, sub in merged.groupby("date"):
        if len(sub) < 5:  # 至少要 5 只股票才能算 IC
            continue
        x = sub[factor_col].values
        y = sub[forward_col].values
        if method == "spearman":
            r, _ = stats.spearmanr(x, y)
        else:
            r, _ = stats.pearsonr(x, y)
        if not np.isnan(r):
            ic_by_date[dt] = r
    return pd.Series(ic_by_date).sort_index()


def analyze_factor(
    factor_panel: pd.DataFrame,
    close: pd.DataFrame,
    factor_col: str,
    forward_periods: Tuple[int, ...] = (1, 5, 20),
) -> Dict[str, FactorIC]:
    """对单个因子做完整 IC 分析"""
    fwd = calc_forward_returns(close, list(forward_periods))
    results: Dict[str, FactorIC] = {}
    for p in forward_periods:
        fwd_col = f"ret_forward_{p}d"
        ic = cross_sectional_ic(factor_panel, fwd, factor_col, fwd_col, "spearman")
        if ic.empty:
            results[fwd_col] = FactorIC(factor_col, p, 0, 0, 0, 0, 0)
            continue
        ic_mean = float(ic.mean())
        ic_std = float(ic.std())
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos = float((ic > 0).mean())
        t_stat = float(ic_mean / (ic_std / np.sqrt(len(ic)))) if ic_std > 0 else 0
        results[fwd_col] = FactorIC(
            factor=factor_col,
            forward=p,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_positive_ratio=ic_pos,
            ic_t_stat=t_stat,
            ic_series=ic,
        )
    return results


def analyze_all_factors(
    factor_long: pd.DataFrame,      # [date, code, f1, f2, ...]
    close: pd.DataFrame,
    factor_cols: Optional[List[str]] = None,
    forward_periods: Tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """批量 IC 分析，返回透视表"""
    if factor_cols is None:
        factor_cols = [c for c in factor_long.columns
                       if c not in ("date", "code", "industry")]
    rows = []
    for f in factor_cols:
        res = analyze_factor(factor_long, close, f, forward_periods)
        for fwd, ic in res.items():
            rows.append({
                "factor": f,
                "forward": ic.forward,
                "ic_mean": ic.ic_mean,
                "ic_std": ic.ic_std,
                "ic_ir": ic.ic_ir,
                "ic_pos_ratio": ic.ic_positive_ratio,
                "ic_t_stat": ic.ic_t_stat,
                "n_periods": len(ic.ic_series) if ic.ic_series is not None else 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2) Walk-Forward 滚动评估
# ---------------------------------------------------------------------------
@dataclass
class WFoldResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_ic: float
    test_ic: float
    train_sharpe: float
    test_sharpe: float


def walk_forward(
    factor_long: pd.DataFrame,
    close: pd.DataFrame,
    factor_col: str,
    train_months: int = 12,
    test_months: int = 3,
    forward: int = 5,
) -> List[WFoldResult]:
    """
    滚动 walk-forward 评估：
    1. 用 train_months 月数据计算 IC、选 topk 信号
    2. 在 test_months 月上检验 IC、Sharpe
    """
    fwd = calc_forward_returns(close, [forward])[["date", "code", f"ret_forward_{forward}d"]]
    merged = factor_long[["date", "code", factor_col]].merge(
        fwd, on=["date", "code"], how="inner"
    ).dropna()

    dates = pd.DatetimeIndex(sorted(merged["date"].unique()))
    if len(dates) == 0:
        return []

    start = dates[0]
    end = dates[-1]
    results: List[WFoldResult] = []

    cur = start
    while True:
        train_end = cur + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break

        train_df = merged[(merged["date"] >= cur) & (merged["date"] < train_end)]
        test_df = merged[(merged["date"] >= test_start) & (merged["date"] < test_end)]
        if train_df.empty or test_df.empty:
            cur = test_start
            continue

        train_ic = float(_cross_sectional_corr(train_df, factor_col, f"ret_forward_{forward}d"))
        test_ic = float(_cross_sectional_corr(test_df, factor_col, f"ret_forward_{forward}d"))

        # Sharpe: 用因子构造简单 long-short，每日 (top-1/3) - (bottom-1/3)
        train_sharpe = _long_short_sharpe(train_df, factor_col, f"ret_forward_{forward}d")
        test_sharpe = _long_short_sharpe(test_df, factor_col, f"ret_forward_{forward}d")

        results.append(WFoldResult(
            train_start=str(cur.date()),
            train_end=str(train_end.date()),
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            train_ic=train_ic,
            test_ic=test_ic,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
        ))
        cur = test_start

    return results


def _cross_sectional_corr(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    if df.empty:
        return 0.0
    corrs = []
    for _, sub in df.groupby("date"):
        if len(sub) >= 5:  # 至少 5 只股票
            r, _ = stats.spearmanr(sub[x_col], sub[y_col])
            if not np.isnan(r):
                corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0


def _long_short_sharpe(df: pd.DataFrame, factor_col: str, ret_col: str) -> float:
    """long top 1/3, short bottom 1/3  →  daily L-S return series"""
    if df.empty:
        return 0.0
    rets = []
    for _, sub in df.groupby("date"):
        if len(sub) < 5:
            continue
        sub = sub.sort_values(factor_col)
        n = len(sub) // 3
        if n == 0:
            continue
        long_ret = sub.iloc[-n:][ret_col].mean()
        short_ret = sub.iloc[:n][ret_col].mean()
        rets.append(long_ret - short_ret)
    if not rets:
        return 0.0
    arr = np.array(rets)
    if arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))
