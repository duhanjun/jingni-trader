"""
数据生成器 (synthetic_data)
============================

为了在不依赖外部 API（如 tushare）的情况下完成回测验证，
生成与 A 股日线统计特征近似的模拟数据。

设计原则（与 jingni-trader/skills/data-engine/engine.py 的 _generate_synthetic_data 一致）:
  - 几何布朗运动
  - 个股之间相关性可控
  - 包含涨跌停、ST 标记
  - 包含 industry 字段
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


INDUSTRIES = [
    "银行", "地产", "医药", "科技", "消费", "能源",
    "材料", "工业", "公用事业", "金融",
]


def generate_panel(
    symbols: List[str],
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    n_factors: int = 2,
    factor_strength: float = 0.3,
    seed: int = 42,
    inject_limit: bool = True,
) -> pd.DataFrame:
    """
    生成模拟 A 股日线 panel 数据

    Returns:
        DataFrame(每行 = 1 只股票 1 个交易日):
            code, date, open, high, low, close, volume,
            is_st, is_limit_up, is_limit_down, industry, lncap
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_stocks = len(symbols)

    # 行业与市值（每只股票固定）
    stock_meta = {}
    for i, sym in enumerate(symbols):
        industry = INDUSTRIES[i % len(INDUSTRIES)]
        lncap = rng.normal(15.0, 1.5)  # 30~50 亿
        stock_meta[sym] = {"industry": industry, "lncap": lncap}

    # 个股基础动量（决定个股 alpha 强弱）
    base_drift = rng.normal(0.0003, 0.001, n_stocks)
    base_vol = rng.uniform(0.012, 0.025, n_stocks)

    # 公共市场因子（模拟大盘）
    market = np.cumsum(rng.normal(0.0003, 0.010, n_days))
    market -= market.mean()
    market *= 0.5  # 控制幅度

    # 隐藏因子信号（让因子有效）
    n_factors = max(1, n_factors)
    factor_loadings = rng.normal(0, 1, (n_stocks, n_factors))
    factor_returns = np.zeros((n_days, n_factors))
    for k in range(n_factors):
        factor_returns[:, k] = np.cumsum(rng.normal(0, 0.005, n_days))

    rows = []
    for i, sym in enumerate(symbols):
        drift = base_drift[i]
        vol = base_vol[i]
        # 收益率 = 个股漂移 + 市场暴露 + 因子暴露 + 噪声
        returns = (
            drift
            + 0.7 * np.diff(np.concatenate([[0.0], market]))  # 市场 beta ~ 0.7
            + factor_strength * (factor_loadings[i, :] @ factor_returns.T)
            + rng.normal(0, vol, n_days)
        )

        # 价格
        price = 10.0 * np.exp(np.cumsum(returns))
        # 限制范围
        price = np.clip(price, 2.0, 200.0)

        # 重建 OHLC
        daily_amp = np.abs(rng.normal(0.005, 0.002, n_days))
        high = price * (1 + daily_amp)
        low = price * (1 - daily_amp)
        open_ = np.concatenate([[price[0]], price[:-1]]) * (1 + rng.normal(0, 0.002, n_days))
        volume = rng.integers(1_000_000, 30_000_000, n_days)

        df = pd.DataFrame({
            "date": dates,
            "code": sym,
            "open": np.round(open_, 2),
            "high": np.round(np.maximum(high, price), 2),
            "low": np.round(np.minimum(low, price), 2),
            "close": np.round(price, 2),
            "volume": volume,
        })
        df["pre_close"] = df["close"].shift(1).fillna(df["close"].iloc[0])
        df["change_pct"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100

        # 涨跌停
        df["is_limit_up"] = df["change_pct"] >= 9.8
        df["is_limit_down"] = df["change_pct"] <= -9.8

        # 注入确定性的涨跌停样本（测试用）
        if inject_limit and i == 0 and len(dates) > 50:
            # 第一个股票，在第 50、100、150 个交易日强制涨停
            for idx in [50, 100, 150]:
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc("is_limit_up")] = True
                    df.iloc[idx, df.columns.get_loc("close")] = df.iloc[idx]["pre_close"] * 1.10
        if inject_limit and i == 1 and len(dates) > 80:
            # 第二个股票，在第 80、130 个交易日强制跌停
            for idx in [80, 130]:
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc("is_limit_down")] = True
                    df.iloc[idx, df.columns.get_loc("close")] = df.iloc[idx]["pre_close"] * 0.90

        # ST
        df["is_st"] = False
        df.loc[df.sample(frac=0.03, random_state=i).index, "is_st"] = True

        df["industry"] = stock_meta[sym]["industry"]
        df["lncap"] = stock_meta[sym]["lncap"]

        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)
    return panel


def generate_forward_returns(panel: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """生成前瞻收益"""
    df = panel[["code", "date", "close"]].copy()
    df = df.sort_values(["code", "date"])
    df[f"ret_{periods}d_fwd"] = df.groupby("code")["close"].shift(-periods) / df["close"] - 1
    return df


def generate_alpha_factor(panel: pd.DataFrame, factor_loading: np.ndarray) -> pd.DataFrame:
    """
    根据给定的因子载荷，生成因子值（用于单元测试）

    Args:
        factor_loading: shape=(n_stocks,)，每只股票的因子载荷
    """
    rows = []
    codes = panel["code"].unique()
    code_to_load = dict(zip(codes, factor_loading))
    for dt, grp in panel.groupby("date"):
        for code in grp["code"].unique():
            base = code_to_load[code]
            # 因子值 = 基础载荷 + 噪声
            rows.append({
                "date": dt,
                "code": code,
                "factor": base + np.random.normal(0, 0.5),
            })
    return pd.DataFrame(rows)


__all__ = ["generate_panel", "generate_forward_returns", "generate_alpha_factor", "INDUSTRIES"]
