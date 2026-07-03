"""
A股面板数据 schema 与工具函数

约定输入数据格式（与 jingni-trader 一致）：
    DataFrame columns = [code, date, open, high, low, close, volume, ...]
    每只股票按 code 分组、按 date 升序
"""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED_COLS = ["code", "date", "open", "high", "low", "close", "volume"]


def ensure_panel(df: pd.DataFrame) -> pd.DataFrame:
    """检查 panel 是否满足最小列要求，并按 (code, date) 排序"""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"panel 缺少必需列: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def pivot_field(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """将长表 pivot 为 (date x code) 矩阵，便于矢量化计算"""
    return df.pivot(index="date", columns="code", values=field).sort_index()


def panel_from_pivot(close_pivot: pd.DataFrame) -> pd.DataFrame:
    """把 (date x code) 的 pivot 表转回长表"""
    long_df = close_pivot.stack(future_stack=True).reset_index()
    long_df.columns = ["date", "code", "value"]
    return long_df
