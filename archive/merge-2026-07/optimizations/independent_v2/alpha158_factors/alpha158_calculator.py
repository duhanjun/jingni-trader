"""
Alpha158 风格因子计算器

借鉴来源: Microsoft Qlib Alpha158 因子库
(https://qlib.readthedocs.io/en/latest/component/data.html)

设计思路:
- jingni-trader 现有 pandas_ta_calculator.py 仅提供 ~20 个基础技术指标
  (rsi, macd, ma, bollinger, atr 等)，缺少 K线形态、价格位置、
  价量统计等系统性 Alpha 因子。
- 本模块实现 Qlib Alpha158 的核心子集（约 60 个因子），覆盖:
  1. K线形态因子 (9): KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
  2. 趋势因子 (10): ROC5/10/20/30/60, MA5/10/20/30/60 (归一化)
  3. 波动因子 (15): STD5/10/20/30/60, MAX5/10/20/30/60, MIN5/10/20/30/60 (归一化)
  4. 位置因子 (10): IMAX5/10/20/30/60, IMIN5/10/20/30/60
  5. 价量统计 (10): CORR5/10/20/30/60, CORD5/10/20/30/60
  6. 成交量比 (5): VSTD5/10/20/30/60, VMA5/10/20/30/60 (归一化)

实现原则:
- 纯 pandas/numpy，无 qlib 依赖
- 用 groupby + transform 实现多股票并行计算
- 因子值经 close 归一化，避免量纲问题
- 接口与 BaseFactorCalculator 兼容
"""
from __future__ import annotations

from typing import List, Dict
import numpy as np
import pandas as pd


class Alpha158Calculator:
    """Alpha158 风格因子计算器（子集实现）"""

    # 因子分类，便于按类别批量计算
    FACTOR_GROUPS = {
        "kline": [
            "KMID", "KLEN", "KMID2", "KUP", "KUP2",
            "KLOW", "KLOW2", "KSFT", "KSFT2",
        ],
        "trend": [
            "ROC5", "ROC10", "ROC20", "ROC30", "ROC60",
            "MA5", "MA10", "MA20", "MA30", "MA60",
        ],
        "volatility": [
            "STD5", "STD10", "STD20", "STD30", "STD60",
            "MAX5", "MAX10", "MAX20", "MAX30", "MAX60",
            "MIN5", "MIN10", "MIN20", "MIN30", "MIN60",
        ],
        "position": [
            "IMAX5", "IMAX10", "IMAX20", "IMAX30", "IMAX60",
            "IMIN5", "IMIN10", "IMIN20", "IMIN30", "IMIN60",
        ],
        "price_volume": [
            "CORR5", "CORR10", "CORR20", "CORR30", "CORR60",
            "CORD5", "CORD10", "CORD20", "CORD30", "CORD60",
        ],
        "volume": [
            "VSTD5", "VSTD10", "VSTD20", "VSTD30", "VSTD60",
            "VMA5", "VMA10", "VMA20", "VMA30", "VMA60",
        ],
    }

    def get_available_factors(self) -> List[str]:
        """返回所有支持的因子名称。"""
        all_factors = []
        for group in self.FACTOR_GROUPS.values():
            all_factors.extend(group)
        return all_factors

    def get_factor_info(self, factor_name: str) -> Dict:
        """返回因子元信息。"""
        for group, factors in self.FACTOR_GROUPS.items():
            if factor_name in factors:
                return {
                    "name": factor_name,
                    "group": group,
                    "direction": 0,
                    "normalized_by": "close",
                }
        return {}

    def calculate(self, data: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """批量计算因子。

        参数:
            data: OHLCV 数据，必须列: code, date, open, high, low, close, volume
            factor_names: 需要计算的因子名列表

        返回:
            DataFrame，列为 code, date, [各因子列]
        """
        if data.empty:
            return data

        df = data.sort_values(["code", "date"]).reset_index(drop=True)
        result = df[["code", "date"]].copy()

        for factor_name in factor_names:
            result[factor_name] = self._calc_factor(df, factor_name)

        return result

    def _calc_factor(self, df: pd.DataFrame, factor_name: str) -> pd.Series:
        """按股票分组计算单个因子（向量化 groupby + transform）。"""
        g = df.groupby("code")

        # ---------- K线形态因子 ----------
        if factor_name == "KMID":
            return (df["close"] - df["open"]) / df["open"]
        elif factor_name == "KLEN":
            return (df["high"] - df["low"]) / df["open"]
        elif factor_name == "KMID2":
            return (df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, np.nan)
        elif factor_name == "KUP":
            return (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
        elif factor_name == "KUP2":
            return (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"]).replace(0, np.nan)
        elif factor_name == "KLOW":
            return (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
        elif factor_name == "KLOW2":
            return (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
        elif factor_name == "KSFT":
            return (2 * df["close"] - df["high"] - df["low"]) / df["open"]
        elif factor_name == "KSFT2":
            return (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)

        # ---------- 趋势因子 ----------
        elif factor_name.startswith("ROC"):
            n = int(factor_name[3:])
            return g["close"].transform(lambda s: s.shift(n) / s)
        elif factor_name.startswith("MA") and not factor_name.startswith("MAX"):
            n = int(factor_name[2:])
            return g["close"].transform(lambda s: s.rolling(n).mean()) / df["close"]

        # ---------- 波动因子 ----------
        elif factor_name.startswith("STD"):
            n = int(factor_name[3:])
            return g["close"].transform(lambda s: s.rolling(n).std()) / df["close"]
        elif factor_name.startswith("MAX"):
            n = int(factor_name[3:])
            return g["high"].transform(lambda s: s.rolling(n).max()) / df["close"]
        elif factor_name.startswith("MIN"):
            n = int(factor_name[3:])
            return g["low"].transform(lambda s: s.rolling(n).min()) / df["close"]

        # ---------- 位置因子 ----------
        elif factor_name.startswith("IMAX"):
            n = int(factor_name[4:])
            # IdxMax: 过去 n 日内最高价出现的位置（归一化到 [0,1]）
            return g["high"].transform(
                lambda s: s.rolling(n).apply(
                    lambda x: float(np.argmax(x)) / n if len(x) == n else np.nan,
                    raw=True,
                )
            )
        elif factor_name.startswith("IMIN"):
            n = int(factor_name[4:])
            return g["low"].transform(
                lambda s: s.rolling(n).apply(
                    lambda x: float(np.argmin(x)) / n if len(x) == n else np.nan,
                    raw=True,
                )
            )

        # ---------- 价量统计因子 ----------
        elif factor_name.startswith("CORR") and not factor_name.startswith("CORD"):
            n = int(factor_name[4:])
            log_vol = np.log(df["volume"] + 1)
            return g.apply(
                lambda grp: grp["close"].rolling(n).corr(log_vol.loc[grp.index])
            ).reset_index(level=0, drop=True).reindex(df.index)
        elif factor_name.startswith("CORD"):
            n = int(factor_name[4:])
            close_ret = g["close"].transform(lambda s: s.pct_change())
            vol_ret = g["volume"].transform(lambda s: s.pct_change())
            log_vol_ret = np.log(vol_ret + 1).replace([np.inf, -np.inf], np.nan)
            return g.apply(
                lambda grp: close_ret.loc[grp.index].rolling(n).corr(
                    log_vol_ret.loc[grp.index]
                )
            ).reset_index(level=0, drop=True).reindex(df.index)

        # ---------- 成交量因子 ----------
        elif factor_name.startswith("VSTD"):
            n = int(factor_name[4:])
            return g["volume"].transform(lambda s: s.rolling(n).std()) / (
                g["volume"].transform(lambda s: s.rolling(n).mean()) + 1
            )
        elif factor_name.startswith("VMA"):
            n = int(factor_name[3:])
            return g["volume"].transform(lambda s: s.rolling(n).mean()) / (
                df["volume"] + 1
            )

        else:
            raise ValueError(f"不支持的因子: {factor_name}")
