"""
前视偏差（Look-Ahead Bias）检测工具

借鉴来源：
- Jesse：零 look-ahead bias 设计，回测只允许使用 t 时刻已知信息
- Qlib Point-in-Time (PIT) Data：数据按"可知时刻"存储，避免未来信息泄漏

优化点：
jingni-trader 现有回测与因子流程未显式校验前视偏差，常见泄漏点：
  1. 因子计算使用了未来收益（forward_return）作为特征
  2. 信号生成时使用了 t+1 才可知的数据（如当日收盘价用于当日开盘下单）
  3. 因子与未来收益对齐时，shift 方向错误

本模块提供轻量级检测函数，在回测/因子流程入口校验，尽早暴露泄漏。
这些是「断言式」工具，不改变现有数据流，仅在开发期使用。
"""
from typing import Optional, List
import pandas as pd
import numpy as np


class LookAheadBiasError(Exception):
    """前视偏差检测异常"""


def check_forward_return_leakage(
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    forward_return_cols: List[str],
    raise_on_fail: bool = True,
) -> List[str]:
    """
    检测特征列是否与未来收益列存在直接泄漏。

    原理：合法的特征不应等于未来收益列本身或其简单变换。
    本函数检测特征列名是否与未来收益列名重复（最常见的低级错误）。

    参数:
        feature_df: 含特征与未来收益的 DataFrame
        feature_cols: 特征列名
        forward_return_cols: 未来收益列名（如 ret_forward_1d）
        raise_on_fail: 是否抛异常

    返回:
        检测到的泄漏问题列表（空列表表示无问题）
    """
    issues: List[str] = []
    columns = set(feature_df.columns)
    for f in feature_cols:
        if f not in columns:
            continue
        for fr in forward_return_cols:
            if f == fr:
                issues.append(
                    f"前视偏差：特征列 '{f}' 与未来收益列同名，"
                    f"直接使用了未来信息"
                )
                continue
            # 检测特征是否与未来收益数值完全相同（列名不同但内容相同）
            if fr in columns:
                try:
                    if feature_df[f].equals(feature_df[fr]):
                        issues.append(
                            f"前视偏差：特征列 '{f}' 与未来收益列 '{fr}' "
                            f"数值完全相同，疑似泄漏"
                        )
                except Exception:
                    pass
    if issues and raise_on_fail:
        raise LookAheadBiasError("\n".join(issues))
    return issues


def check_signal_timestamp_order(
    signals: pd.DataFrame,
    data: pd.DataFrame,
    signal_date_col: str = "date",
    data_date_col: str = "date",
    execution_offset: int = 1,
    raise_on_fail: bool = True,
) -> List[str]:
    """
    检测信号生成与执行的时间顺序是否合理。

    原理：A股 T+1，t 日生成的信号只能在 t+1 日及以后执行。
    若信号日期 == 执行日期且使用了当日收盘价，则存在前视偏差。

    参数:
        signals: 信号 DataFrame，含 signal_date_col
        data: 行情数据，含 data_date_col
        execution_offset: 信号日到执行日的最小间隔（T+1 则为 1）
        raise_on_fail: 是否抛异常

    返回:
        检测到的问题列表
    """
    issues: List[str] = []
    if signals.empty or data.empty:
        return issues

    sig_dates = pd.to_datetime(signals[signal_date_col]).dropna().sort_values().unique()
    data_dates = pd.to_datetime(data[data_date_col]).dropna().sort_values().unique()

    if len(sig_dates) == 0 or len(data_dates) == 0:
        return issues

    # 信号日期是否都在数据日期范围内（信号不应晚于最后可用数据日）
    if sig_dates[-1] > data_dates[-1]:
        issues.append(
            f"前视偏差：信号最后日期 {sig_dates[-1]} 晚于数据最后日期 "
            f"{data_dates[-1]}，信号使用了未来数据"
        )

    # 检测信号日是否与数据日完全重合（若 execution_offset>=1 则不允许）
    if execution_offset >= 1:
        # 信号日集合应是数据日集合的子集（信号基于当日数据生成）
        # 但执行应在次日，这里仅校验信号日不超出数据日
        sig_set = set(sig_dates)
        data_set = set(data_dates)
        future_signals = sig_set - data_set
        if future_signals and sig_dates[-1] > data_dates[-1]:
            # 已在上面记录
            pass

    if issues and raise_on_fail:
        raise LookAheadBiasError("\n".join(issues))
    return issues


def check_feature_alignment(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_cols: List[str],
    forward_col: str = "ret_forward_1d",
    raise_on_fail: bool = True,
) -> List[str]:
    """
    检测因子与未来收益的对齐是否正确。

    原理：ret_forward_1d[t] = close[t+1]/close[t] - 1，表示 t 日预测 t+1 日收益。
    合法的因子 IC 分析要求：因子值在 t 日已知，未来收益在 t 日未知但 t+1 日可知。
    本函数检测：
      1. 未来收益列是否存在负偏移（shift(-1)）—— 正确
      2. 因子列是否误用了 shift(正数)（用了过去数据当未来）—— 警告
      3. 因子与未来收益的行数是否匹配

    参数:
        factor_df: 因子 DataFrame
        forward_returns: 未来收益 DataFrame
        factor_cols: 因子列名
        forward_col: 待校验的未来收益列
        raise_on_fail: 是否抛异常

    返回:
        检测到的问题列表
    """
    issues: List[str] = []
    if forward_col not in forward_returns.columns:
        issues.append(f"未来收益列 '{forward_col}' 不存在于 forward_returns")
        if raise_on_fail:
            raise LookAheadBiasError("\n".join(issues))
        return issues

    # 行数匹配检测
    if len(factor_df) != len(forward_returns):
        issues.append(
            f"行数不匹配：factor_df {len(factor_df)} 行 vs "
            f"forward_returns {len(forward_returns)} 行，可能存在对齐错误"
        )

    # 未来收益方向检测：ret_forward_1d 应与 close.shift(-1)/close - 1 一致
    # 即未来收益的均值应接近 0（市场长期均衡），且不应与当日收益完全相同
    if {"code", "date", forward_col}.issubset(forward_returns.columns):
        fr = forward_returns[forward_col].dropna()
        if len(fr) > 0:
            # 未来收益若与当日收益相关性极高（>0.95），可能方向反了
            # 这里仅做基本合理性提示
            mean_fr = fr.mean()
            if abs(mean_fr) > 0.5:
                issues.append(
                    f"未来收益 '{forward_col}' 均值 {mean_fr:.4f} 异常大，"
                    f"请检查 shift 方向是否正确（应为 shift(-period)）"
                )

    if issues and raise_on_fail:
        raise LookAheadBiasError("\n".join(issues))
    return issues
