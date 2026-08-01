"""
估值分位计算模块
计算个股PE/PB/PS等估值指标在历史区间中的分位数
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger("valuation")


# 估值指标方向配置：是否要求正值才参与分位计算
# PE/PB/PS 为负或零时分位无意义（亏损公司），需剔除
# dv_ratio（股息率）可以为零（不分红），但不应为负
_POSITIVE_REQUIRED: Dict[str, bool] = {
    "pe_ttm": True,
    "pb": True,
    "ps_ttm": True,
    "dv_ratio": False,
}

# 越高越被低估的指标（股息率：越高越好）
# 其余估值指标（PE/PB/PS）越低越被低估
_HIGHER_IS_BETTER = {"dv_ratio"}

# 默认综合估值指标
_DEFAULT_METRICS: List[str] = ["pe_ttm", "pb", "ps_ttm", "dv_ratio"]


def _percentile_of_score(scores: np.ndarray, value: float) -> float:
    """返回 value 在 scores 中的百分位（0-100），midrank 法

    百分位 = (小于 value 的个数 + 0.5 * 等于 value 的个数) / 总数 * 100
    """
    arr = np.asarray(scores, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return 0.0
    below = float(np.sum(arr < value))
    equal = float(np.sum(arr == value))
    return (below + 0.5 * equal) / n * 100.0


def _verdict_from_percentile(percentile: float) -> str:
    """根据分位生成判定：<25% 偏低，25-75% 适中，>75% 偏高"""
    if percentile < 25:
        return "偏低"
    elif percentile <= 75:
        return "适中"
    else:
        return "偏高"


class ValuationAnalyzer:
    """个股估值分位分析"""

    def calculate_percentile(self, stock_code: str, metric: str,
                             historical_data: pd.DataFrame,
                             years: int = 5) -> Dict:
        """
        计算当前估值指标在历史N年中的分位数

        参数:
            stock_code: 股票代码
            metric: 估值指标 (pe_ttm, pb, ps_ttm, dv_ratio)
            historical_data: 含 code, date, metric列的DataFrame
            years: 回溯年数

        返回:
            {
                "stock": "000001.SZ",
                "metric": "pe_ttm",
                "current_value": 12.5,
                "percentile": 35.0,  # 当前值在历史中的百分位
                "median": 15.2,
                "max": 25.0,
                "min": 8.0,
                "verdict": "偏低",  # 偏低/适中/偏高
                "history_years": 5
            }
        """
        empty = {
            "stock": stock_code,
            "metric": metric,
            "current_value": None,
            "percentile": None,
            "median": None,
            "max": None,
            "min": None,
            "verdict": "无数据",
            "history_years": years,
            "samples": 0,
        }

        if historical_data is None or len(historical_data) == 0:
            logger.warning("historical_data 为空，无法计算分位: %s/%s", stock_code, metric)
            return empty

        required = {"code", "date", metric}
        missing = required - set(historical_data.columns)
        if missing:
            logger.error("historical_data 缺少必要列: %s", missing)
            return empty

        df = historical_data.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["code"] == stock_code]
        if df.empty:
            logger.warning("historical_data 中未找到股票 %s", stock_code)
            return empty

        df = df.sort_values("date", ascending=False)
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)

        # 估值类指标为负或零时分位无意义，剔除
        if _POSITIVE_REQUIRED.get(metric, False):
            df.loc[df[metric] <= 0, metric] = np.nan

        df_valid = df.dropna(subset=[metric])
        if df_valid.empty:
            logger.warning("过滤后无有效 %s 值: %s", metric, stock_code)
            return empty

        # 以最新有效观测日为基准，回溯 N 年
        reference_date = df_valid["date"].iloc[0]
        cutoff = reference_date - pd.DateOffset(years=years)
        window = df_valid[df_valid["date"] >= cutoff]
        if window.empty:
            return empty

        values = window[metric].astype(float).values
        current_value = float(values[0])  # 降序排列，首个为最新值
        percentile = _percentile_of_score(values, current_value)
        median = float(np.median(values))
        max_v = float(np.max(values))
        min_v = float(np.min(values))
        verdict = _verdict_from_percentile(percentile)

        return {
            "stock": stock_code,
            "metric": metric,
            "current_value": round(current_value, 4),
            "percentile": round(percentile, 2),
            "median": round(median, 4),
            "max": round(max_v, 4),
            "min": round(min_v, 4),
            "verdict": verdict,
            "history_years": years,
            "samples": int(values.size),
        }

    def batch_percentile(self, stock_codes: List[str], metric: str,
                         historical_data: pd.DataFrame,
                         years: int = 5) -> pd.DataFrame:
        """批量计算多只股票的估值分位

        返回按分位升序排列的 DataFrame（最被低估的在前）。
        """
        rows: List[Dict] = []
        for code in stock_codes:
            rows.append(self.calculate_percentile(code, metric, historical_data, years))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "percentile" in df.columns:
            df = df.sort_values("percentile", ascending=True, na_position="last")
        return df.reset_index(drop=True)

    def compare_valuation(self, stock_code: str,
                          historical_data: pd.DataFrame,
                          metrics: List[str] = None) -> Dict:
        """
        综合估值分析：同时计算PE/PB/PS/股息率分位

        返回:
            {
                "stock": "000001.SZ",
                "metrics": {
                    "pe_ttm": {...},
                    "pb": {...},
                    "ps_ttm": {...},
                    "dv_ratio": {...}
                },
                "overall_verdict": "估值合理偏低",
                "score": 65  # 0-100, 越高越被低估
            }
        """
        if metrics is None:
            metrics = list(_DEFAULT_METRICS)

        no_data = {
            "stock": stock_code,
            "metrics": {},
            "overall_verdict": "无数据",
            "score": None,
        }

        if historical_data is None or len(historical_data) == 0:
            logger.warning("historical_data 为空: %s", stock_code)
            return no_data

        available = [m for m in metrics if m in historical_data.columns]
        if not available:
            logger.error("historical_data 中未找到任何估值指标列: %s", metrics)
            return no_data

        metrics_result: Dict[str, Dict] = {}
        scores: List[float] = []
        for m in available:
            r = self.calculate_percentile(stock_code, m, historical_data)
            metrics_result[m] = r
            if r.get("percentile") is None:
                continue
            # 估值得分：越高越被低估
            #   PE/PB/PS 越低越被低估 => 得分 = 100 - 分位
            #   股息率 越高越被低估 => 得分 = 分位
            if m in _HIGHER_IS_BETTER:
                score_m = r["percentile"]
            else:
                score_m = 100.0 - r["percentile"]
            scores.append(score_m)

        if not scores:
            return {
                "stock": stock_code,
                "metrics": metrics_result,
                "overall_verdict": "无数据",
                "score": None,
            }

        overall_score = float(np.mean(scores))
        if overall_score >= 75:
            overall_verdict = "估值偏低"
        elif overall_score >= 50:
            overall_verdict = "估值合理偏低"
        elif overall_score >= 25:
            overall_verdict = "估值合理偏高"
        else:
            overall_verdict = "估值偏高"

        return {
            "stock": stock_code,
            "metrics": metrics_result,
            "overall_verdict": overall_verdict,
            "score": round(overall_score, 2),
        }
