"""合成测试数据构造器。

不依赖任何外部数据源/网络，提供确定性可复现的 OHLCV DataFrame。
所有需要"DATA 阶段产物"的测试都应使用本模块的构造器，避免重复代码。

来源：从 tests/test_integration_e2e.py 与 tests/test_jingni_datafeed_integration.py
中抽出共用部分，统一管理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_daily(
    codes: list[str] | None = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    seed: int = 20240101,
) -> pd.DataFrame:
    """构造最小可用 OHLCV DataFrame。

    Args:
        codes: 股票代码列表，默认 ["000001.SZ", "600000.SH"]
        start: 起始日期
        end: 结束日期
        seed: 随机种子，保证可复现

    Returns:
        DataFrame，列：code/date/open/high/low/close/volume
        覆盖 data-engine 外部数据校验所需的全部字段。
    """
    if codes is None:
        codes = ["000001.SZ", "600000.SH"]

    frames = []
    rng = np.random.RandomState(seed)
    for code in codes:
        dates = pd.bdate_range(start, end)
        n = len(dates)
        base = rng.uniform(8, 20)
        closes = base * (1 + np.cumsum(rng.normal(0, 0.01, n)))
        opens = closes * (1 + rng.normal(0, 0.002, n))
        highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n)))
        vol = rng.randint(1_000_000, 10_000_000, n)
        frames.append(pd.DataFrame({
            "code": code,
            "date": dates,
            "open": opens.round(2),
            "high": highs.round(2),
            "low": lows.round(2),
            "close": closes.round(2),
            "volume": vol,
        }))
    return pd.concat(frames, ignore_index=True)


def make_data_parquet(tmp_path, codes=None, start="2024-01-01", end="2024-03-31") -> str:
    """构造最小可用 DATA parquet 产物（OHLCV），供下游 skill 测试使用。

    与 make_synthetic_daily 的区别：
    - 本函数直接落盘到 <tmp_path>/cleaned_data.parquet 并返回路径
    - 默认时间范围更短（3 个月），加速测试

    来源：从 tests/test_jingni_datafeed_integration.py::_make_data_parquet 抽出。
    """
    codes = codes or ["000001.SZ", "600000.SH"]
    rng = np.random.RandomState(42)
    frames = []
    for code in codes:
        dates = pd.bdate_range(start, end)
        n = len(dates)
        closes = 10 * (1 + np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({
            "code": code,
            "date": dates,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": rng.randint(1_000_000, 5_000_000, n),
        }))
    df = pd.concat(frames, ignore_index=True)
    path = str(tmp_path / "cleaned_data.parquet")
    df.to_parquet(path, index=False)
    return path


def make_factor_dataframe(
    codes: list[str] | None = None,
    factor_name: str = "momentum_20d",
) -> pd.DataFrame:
    """构造 mock datafeed 返回的因子 DataFrame。

    用于 factor-engine 的 _try_load_factor_from_datafeed 测试。

    来源：从 tests/test_jingni_datafeed_integration.py 抽出。
    """
    if codes is None:
        codes = ["000001.SZ", "600000.SH"]
    return pd.DataFrame({
        "code": codes,
        "date": pd.to_datetime(["2024-01-02"] * len(codes)),
        "factor_name": [factor_name] * len(codes),
        "factor_value": [0.05, -0.03][:len(codes)],
    })
