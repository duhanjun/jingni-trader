"""
向量化因子引擎：将 vectorbt 风格的 2D 矩阵运算整合到 jingni-trader

兼容设计：
- 接受与现有 PandasTaCalculator 相同的输入 (long-format DataFrame)
- 返回相同的输出格式
- 但内部完全用 numpy 2D 矩阵 + numba JIT 运行

性能对比目标：
- 在 100 只股票 × 1000 个交易日规模下，MA20 计算应 < 50ms
- 在 1000 只股票 × 2400 个交易日规模下，< 1s
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .vector_ops import (
    numba_ma, numba_std, numba_ema, numba_rsi,
    numba_rolling_corr, numba_cross_section_rank,
)


class VectorizedFactorCalculator:
    """
    向量化因子计算器（Numba 加速版）

    用法:
        calc = VectorizedFactorCalculator()
        result = calc.calculate(data, ['ma_20', 'rsi_14', 'rank_volume'])
    """

    SUPPORTED = {
        "ma_5":   ("ma", 5),
        "ma_10":  ("ma", 10),
        "ma_20":  ("ma", 20),
        "ma_60":  ("ma", 60),
        "std_20": ("std", 20),
        "ema_10": ("ema", 10),
        "ema_20": ("ema", 20),
        "rsi_14": ("rsi", 14),
        "rank_volume": ("rank", "volume"),
        "rank_amount": ("rank", "amount"),
        "rank_close":   ("rank", "close"),
        "corr_close_volume_20": ("corr", ("close", "volume", 20)),
    }

    @staticmethod
    def get_available_factors() -> List[str]:
        return list(VectorizedFactorCalculator.SUPPORTED.keys())

    @staticmethod
    def get_factor_info(name: str) -> Dict[str, Any]:
        info_map = {
            "ma_5":   {"name": "5日均线",   "direction": 0, "params": {"window": 5}},
            "ma_20":  {"name": "20日均线",  "direction": 0, "params": {"window": 20}},
            "std_20": {"name": "20日标准差", "direction": 0, "params": {"window": 20}},
            "ema_20": {"name": "20日EMA",   "direction": 0, "params": {"window": 20}},
            "rsi_14": {"name": "14日RSI",   "direction": -1, "params": {"window": 14}},
            "rank_volume": {"name": "成交量截面排名", "direction": 0, "params": {}},
        }
        return info_map.get(name, {})

    # ---------------------------------------------------------------------
    # 主接口
    # ---------------------------------------------------------------------
    def calculate(
        self,
        data: pd.DataFrame,
        factor_names: List[str],
        timing: bool = False,
    ) -> pd.DataFrame:
        """
        批量计算因子
        """
        if data.empty:
            return data
        # 强制拷贝，避免 SettingWithCopyWarning
        data = data.sort_values(["code", "date"]).reset_index(drop=True)
        codes = data["code"].unique()
        dates = pd.Index(sorted(data["date"].unique()))
        T, N = len(dates), len(codes)
        if T == 0 or N == 0:
            return data[["code", "date"]].copy()

        # 构造 2D 矩阵 (T, N)，列序与 codes 对齐
        panels = self._build_2d_panels(data, dates, codes, ["open", "high", "low", "close", "volume", "amount"])
        result = data[["code", "date"]].copy()
        timings: Dict[str, float] = {}

        for fname in factor_names:
            t0 = time.perf_counter()
            spec = self.SUPPORTED.get(fname)
            if spec is None:
                raise ValueError(f"不支持的因子: {fname}")
            kind = spec[0]
            if kind == "ma":
                arr = numba_ma(panels["close"], spec[1])
            elif kind == "std":
                arr = numba_std(panels["close"], spec[1])
            elif kind == "ema":
                arr = numba_ema(panels["close"], spec[1])
            elif kind == "rsi":
                arr = numba_rsi(panels["close"], spec[1])
            elif kind == "rank":
                arr = numba_cross_section_rank(panels[spec[1]])
            elif kind == "corr":
                f1, f2, win = spec[1]
                arr = numba_rolling_corr(panels[f1], panels[f2], win)
            else:
                raise ValueError(f"未知算子: {kind}")
            timings[fname] = time.perf_counter() - t0
            result[fname] = self._unpack_2d(arr, data, codes)

        if timing:
            print(f"[VectorizedFactorCalculator] timings: {timings}")
        return result

    # ---------------------------------------------------------------------
    # 内部工具
    # ---------------------------------------------------------------------
    def _build_2d_panels(
        self,
        data: pd.DataFrame,
        dates: pd.Index,
        codes: np.ndarray,
        fields: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        构造 (T, N) 的 2D 矩阵，缺失值填 NaN
        """
        # 用 pivot 实现
        panels = {}
        date_to_t = {d: i for i, d in enumerate(dates)}
        code_to_n = {c: i for i, c in enumerate(codes)}
        T, N = len(dates), len(codes)
        for f in fields:
            if f not in data.columns:
                panels[f] = np.full((T, N), np.nan)
                continue
            arr = np.full((T, N), np.nan)
            sub = data[["date", "code", f]].dropna(subset=[f])
            for d, c, v in zip(sub["date"], sub["code"], sub[f]):
                t = date_to_t.get(d)
                n = code_to_n.get(c)
                if t is not None and n is not None:
                    arr[t, n] = v
            panels[f] = arr
        return panels

    def _unpack_2d(
        self, arr: np.ndarray, data: pd.DataFrame, codes: np.ndarray
    ) -> np.ndarray:
        """把 (T, N) 矩阵按 (code, date) 顺序摊平成 1D 数组"""
        # 给 data 标 index 用作排序键
        dates = pd.Index(sorted(data["date"].unique()))
        # 对每行: (date, code) 索引
        T, N = arr.shape
        date_to_t = {d: i for i, d in enumerate(dates)}
        code_to_n = {c: i for i, c in enumerate(codes)}
        out = np.full(len(data), np.nan)
        # 用 vectorize 加速查找（实际还是 Python 循环，但是单层）
        d_arr = data["date"].values
        c_arr = data["code"].values
        for i in range(len(data)):
            t = date_to_t.get(d_arr[i])
            n = code_to_n.get(c_arr[i])
            if t is not None and n is not None:
                out[i] = arr[t, n]
        return out


class VectorizedFactorEngine:
    """
    高级封装：保留 jingni-trader 的 BaseFactorCalculator 接口风格
    """
    def __init__(self):
        self._calc = VectorizedFactorCalculator()

    def calc(self, data: pd.DataFrame, factor_names: List[str], **kwargs) -> pd.DataFrame:
        return self._calc.calculate(data, factor_names, **kwargs)

    def get_available_factors(self) -> List[str]:
        return self._calc.get_available_factors()
