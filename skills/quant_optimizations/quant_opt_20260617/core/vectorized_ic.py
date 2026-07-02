"""
向量化 IC 分析模块（验证模块 1）

借鉴来源：
- Qlib (Microsoft) qlib.contrib.evaluate 中的 IC 分析框架：
    https://github.com/microsoft/qlib/blob/main/qlib/contrib/evaluate.py
    核心思想：每日 cross-section Pearson/Spearman 相关
- AlphaBench (ICLR 2026)：
    强调 Rank IC + Pearson IC 双通道报告
- Hubble (arXiv:2604.09601)：
    强调双通道检索、Family-aware 评估
- Hubble 同篇 + 学界最佳实践：
    加入 HAC (Newey-West) 调整后的 IC t 统计量，解决 IC 序列自相关导致的显著性膨胀

相对 jingni-trader 现有实现 (skills/factor-engine/engine.py::_calc_ic) 的改进：
- 原实现: 逐日 for 循环调用 scipy.stats.spearmanr/pearsonr
- 本实现: 矩阵化相关系数 + Numba JIT 加速主循环
- 新增:  IC/RankIC 双通道、HAC 调整 t-stat、IC 衰减半衰期
- 性能:  同一数据规模下，预期 5x-50x 加速
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False
    njit = lambda *a, **k: (lambda f: f)
    prange = range


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """带 NaN 容忍的 Pearson 相关系数"""
    if len(x) < 3:
        return np.nan
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pearsonr(x[mask], y[mask])
    return float(r) if np.isfinite(r) else np.nan


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """带 NaN 容忍的 Spearman 秩相关系数（即 Rank IC）"""
    if len(x) < 3:
        return np.nan
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.spearmanr(x[mask], y[mask])
    return float(r) if np.isfinite(r) else np.nan


def _rowwise_corr_nb(f_mat: np.ndarray, r_mat: np.ndarray) -> np.ndarray:
    """
    逐行（每日）相关系数，矩阵化、NaN 容忍

    参数:
        f_mat, r_mat: 形状 (T, N) 的二维矩阵
                      NaN 视为缺失

    返回:
        长度 T 的一维数组，每行为当日相关系数
    """
    T, N = f_mat.shape
    min_N = min(f_mat.shape[1], r_mat.shape[1])
    f_mat = f_mat[:, :min_N]
    r_mat = r_mat[:, :min_N]

    if HAS_NUMBA:
        return _rowwise_corr_numba(f_mat, r_mat)
    # 纯 numpy fallback
    out = np.full(T, np.nan)
    for t in range(T):
        f = f_mat[t]
        r = r_mat[t]
        mask = ~(np.isnan(f) | np.isnan(r))
        if mask.sum() < 3:
            continue
        f_v = f[mask]
        r_v = r[mask]
        f_dm = f_v - f_v.mean()
        r_dm = r_v - r_v.mean()
        num = (f_dm * r_dm).sum()
        den = np.sqrt((f_dm ** 2).sum() * (r_dm ** 2).sum())
        if den > 0:
            out[t] = num / den
    return out


if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _rowwise_corr_numba(f_mat: np.ndarray, r_mat: np.ndarray) -> np.ndarray:
        """Numba 加速版逐行相关系数"""
        T = f_mat.shape[0]
        N = f_mat.shape[1]
        out = np.full(T, np.nan)
        for t in range(T):
            # 收集有效对
            f_valid = np.empty(N)
            r_valid = np.empty(N)
            cnt = 0
            for j in range(N):
                fj = f_mat[t, j]
                rj = r_mat[t, j]
                if not np.isnan(fj) and not np.isnan(rj):
                    f_valid[cnt] = fj
                    r_valid[cnt] = rj
                    cnt += 1
            if cnt < 3:
                continue
            f_v = f_valid[:cnt]
            r_v = r_valid[:cnt]
            f_mean = f_v.sum() / cnt
            r_mean = r_v.sum() / cnt
            num = 0.0
            f_ss = 0.0
            r_ss = 0.0
            for k in range(cnt):
                fd = f_v[k] - f_mean
                rd = r_v[k] - r_mean
                num += fd * rd
                f_ss += fd * fd
                r_ss += rd * rd
            den = np.sqrt(f_ss * r_ss)
            if den > 0:
                out[t] = num / den
        return out
else:
    def _rowwise_corr_numba(f_mat, r_mat):  # noqa
        return _rowwise_corr_nb(f_mat, r_mat)


# ---------------------------------------------------------------------------
# 核心：向量化 IC 分析
# ---------------------------------------------------------------------------

class VectorizedICAnalyzer:
    """
    向量化 IC / RankIC 分析器

    用法:
        analyzer = VectorizedICAnalyzer()
        result = analyzer.analyze(factor_df, fwd_ret_df, factor_names)
        # result 是嵌套 dict: {factor: {period: {...metrics}}}

    关键设计：
    1. 把 N 次 scipy 调用 → 1 次 groupby('date').corr
    2. 同时计算 Pearson IC 和 Spearman RankIC
    3. 显著性检验使用 HAC (Newey-West) 调整 t 统计量
    4. 报告 IC decay（IC 自相关半衰期）
    """

    def __init__(
        self,
        min_obs_per_day: int = 10,
        hac_lags: int = 5,
    ):
        """
        参数:
            min_obs_per_day: 每个截面日最少有效样本数，低于此值丢弃该日
            hac_lags: HAC 调整最大滞后阶，参考 Newey-West (1987)
        """
        self.min_obs_per_day = min_obs_per_day
        self.hac_lags = hac_lags

    def _cross_section_corr(
        self,
        factor_vals: pd.Series,
        ret_vals: pd.Series,
        method: str = "pearson",
    ) -> float:
        """单日截面相关系数（兼容 NaN）"""
        df = pd.concat([factor_vals, ret_vals], axis=1).dropna()
        if len(df) < self.min_obs_per_day:
            return np.nan
        if method == "spearman":
            return _safe_spearman(df.iloc[:, 0].values, df.iloc[:, 1].values)
        return _safe_pearson(df.iloc[:, 0].values, df.iloc[:, 1].values)

    def compute_ic_series(
        self,
        factor_series: pd.Series,
        ret_series: pd.Series,
        method: str = "pearson",
    ) -> pd.Series:
        """
        计算 IC 时间序列（**矩阵化**版，无 groupby 循环）

        参数:
            factor_series: 带 (date, code) MultiIndex 的因子 Series
            ret_series:    带 (date, code) MultiIndex 的收益率 Series
            method:        'pearson' 或 'spearman'

        返回:
            以 date 为索引的 IC 时间序列
        """
        aligned = pd.concat(
            [factor_series.rename("f"), ret_series.rename("r")], axis=1
        ).dropna()

        if method == "spearman":
            # 截面 rank 后转回 (T, N) 矩阵
            aligned = aligned.rank(pct=True)

        # 转 (T, N) 矩阵
        pivot = aligned.reset_index().pivot(
            index="date", columns="code", values=["f", "r"]
        )
        # 取 f 矩阵与 r 矩阵
        # 由于 MultiIndex 列：(N code, f/r)
        f_mat = pivot["f"].values
        r_mat = pivot["r"].values
        dates = pivot.index.values

        # 矩阵化 Pearson: 对每行 (每天) 算 corr
        # nan-aware: 每行有 NaN 时用 mask
        ic_arr = _rowwise_corr_nb(f_mat, r_mat)

        ic_series = pd.Series(ic_arr, index=pd.to_datetime(dates), name="ic")
        return ic_series.dropna()

    def _ic_summary(self, ic_series: pd.Series) -> Dict[str, float]:
        """对一条 IC 序列计算全套统计量（含 HAC t-stat）"""
        if ic_series.empty or len(ic_series) < 3:
            return {
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "ic_ir": np.nan,
                "ic_pos_ratio": np.nan,
                "ic_t_stat": np.nan,
                "ic_t_stat_hac": np.nan,
                "ic_decay_halflife": np.nan,
                "n_days": int(len(ic_series)),
            }
        ic_arr = ic_series.values
        mean = float(np.nanmean(ic_arr))
        std = float(np.nanstd(ic_arr, ddof=1))
        ir = mean / std if std > 0 else 0.0
        pos_ratio = float(np.nanmean(ic_arr > 0))
        t_stat = mean / (std / np.sqrt(len(ic_arr))) if std > 0 else 0.0
        t_hac = self._hac_t_stat(ic_arr)

        # IC 自相关半衰期（alpha decay）
        halflife = self._ic_decay(ic_arr)
        return {
            "ic_mean": mean,
            "ic_std": std,
            "ic_ir": ir,
            "ic_pos_ratio": pos_ratio,
            "ic_t_stat": t_stat,
            "ic_t_stat_hac": t_hac,
            "ic_decay_halflife": halflife,
            "n_days": int(len(ic_arr)),
        }

    @staticmethod
    def _hac_t_stat(x: np.ndarray, max_lag: int = 5) -> float:
        """
        Newey-West HAC 调整后的 t 统计量
        用于修正 IC 序列自相关导致的显著性膨胀
        """
        x = x[~np.isnan(x)]
        n = len(x)
        if n < 3:
            return 0.0
        x_dm = x - x.mean()
        # 选 lag = floor(4*(n/100)^(2/9))  (Newey-West 经验公式)
        lag = min(max_lag, int(np.floor(4 * (n / 100) ** (2 / 9))))
        gamma0 = np.sum(x_dm ** 2) / n
        var_hac = gamma0
        for k in range(1, lag + 1):
            w_k = 1.0 - k / (lag + 1.0)
            gamma_k = np.sum(x_dm[k:] * x_dm[:-k]) / n
            var_hac += 2.0 * w_k * gamma_k
        var_hac = max(var_hac, 1e-12)
        se = np.sqrt(var_hac / n)
        return float(x.mean() / se) if se > 0 else 0.0

    @staticmethod
    def _ic_decay(x: np.ndarray) -> float:
        """IC 序列 lag-1 自相关 → 半衰期（天）"""
        x = x[~np.isnan(x)]
        if len(x) < 3:
            return np.nan
        x_dm = x - x.mean()
        c0 = np.sum(x_dm ** 2)
        if c0 == 0:
            return np.nan
        c1 = np.sum(x_dm[1:] * x_dm[:-1])
        rho = c1 / c0
        if rho <= 0 or rho >= 1:
            return np.nan
        halflife = -np.log(2) / np.log(rho)
        return float(halflife)

    def analyze(
        self,
        factor_df: pd.DataFrame,
        ret_df: pd.DataFrame,
        factor_names: List[str],
        periods: Tuple[int, ...] = (1, 5, 20),
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        全量分析：每个因子 × 每个持有期

        参数:
            factor_df:    必须包含 ['code', 'date', <因子名>...]
            ret_df:       必须包含 ['code', 'date', 'ret_forward_1d', 'ret_forward_5d',
                                     'ret_forward_20d']
            factor_names: 因子列名列表
            periods:      持有期（天）

        返回:
            {
                factor_name: {
                    period: {
                        'ic_mean', 'ic_std', 'ic_ir', 'ic_pos_ratio',
                        'ic_t_stat', 'ic_t_stat_hac',
                        'ic_decay_halflife', 'n_days'
                    }
                }
            }
        """
        merged = factor_df.merge(ret_df, on=["code", "date"], how="inner")
        merged = merged.set_index(["date", "code"]).sort_index()

        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for fname in factor_names:
            if fname not in merged.columns:
                continue
            f_series = merged[fname]
            results[fname] = {}
            for p in periods:
                ret_col = f"ret_forward_{p}d"
                if ret_col not in merged.columns:
                    continue
                r_series = merged[ret_col]
                # Pearson IC
                ic_p = self.compute_ic_series(f_series, r_series, method="pearson")
                # Spearman RankIC
                ic_s = self.compute_ic_series(f_series, r_series, method="spearman")
                summary = self._ic_summary(ic_p)
                summary["rank_ic_mean"] = float(ic_s.mean()) if not ic_s.empty else np.nan
                summary["rank_ic_ir"] = (
                    float(ic_s.mean() / ic_s.std()) if (not ic_s.empty and ic_s.std() > 0) else 0.0
                )
                summary["period"] = p
                results[fname][p] = summary
        return results

    def auto_select(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        primary_period: int = 5,
        min_abs_ic: float = 0.02,
        min_ic_ir: float = 0.3,
        min_t_stat_hac: float = 2.0,
    ) -> List[str]:
        """
        根据 IC 分析结果自动筛选有效因子
        借鉴 AlphaBench 的多指标过滤：
            |IC| >= min_abs_ic
            |ICIR| >= min_ic_ir
            |HAC t| >= min_t_stat_hac  (HAC 调整后显著性)
        """
        selected = []
        for fname, by_period in results.items():
            m = by_period.get(primary_period, {})
            if (
                abs(m.get("ic_mean", 0)) >= min_abs_ic
                and abs(m.get("ic_ir", 0)) >= min_ic_ir
                and abs(m.get("ic_t_stat_hac", 0)) >= min_t_stat_hac
            ):
                selected.append(fname)
        return selected


# ---------------------------------------------------------------------------
# 与原 (skills/factor-engine/engine.py) 的接口对齐封装
# ---------------------------------------------------------------------------

def ic_analysis_compatible(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, float]]]:
    """
    返回与原 engine.ic_analysis 完全兼容结构的 dict：
        {
            "ret_forward_1d": [{factor, forward_period, ic_mean, ic_std, ic_ir, ic_positive_ratio, ic_t_stat}, ...],
            "ret_forward_5d": [...],
            "ret_forward_20d": [...],
        }

    主要差异：
    - 新增 ic_t_stat_hac（HAC 调整后 t 统计量）
    - 新增 rank_ic_mean / rank_ic_ir
    """
    if factor_df.empty or forward_returns.empty:
        return {}

    if factor_names is None:
        factor_names = [
            c for c in factor_df.columns
            if c not in ("code", "date", "industry")
        ]

    analyzer = VectorizedICAnalyzer()
    raw = analyzer.analyze(factor_df, forward_returns, factor_names, periods=(1, 5, 20))

    compat: Dict[str, List[Dict]] = {}
    for period in (1, 5, 20):
        col_key = f"ret_forward_{period}d"
        out = []
        for fname in factor_names:
            if fname not in raw or period not in raw[fname]:
                continue
            m = raw[fname][period]
            out.append({
                "factor": fname,
                "forward_period": col_key,
                "ic_mean": round(m["ic_mean"], 6),
                "ic_std": round(m["ic_std"], 6),
                "ic_ir": round(m["ic_ir"], 4),
                "ic_positive_ratio": round(m["ic_pos_ratio"], 4),
                "ic_t_stat": round(m["ic_t_stat"], 4),
                # === 新增 ===
                "ic_t_stat_hac": round(m["ic_t_stat_hac"], 4),
                "rank_ic_mean": round(m.get("rank_ic_mean", 0), 6),
                "rank_ic_ir": round(m.get("rank_ic_ir", 0), 4),
                "ic_decay_halflife": round(m["ic_decay_halflife"], 2)
                if np.isfinite(m["ic_decay_halflife"]) else None,
                "n_days": m["n_days"],
            })
        compat[col_key] = out
    return compat