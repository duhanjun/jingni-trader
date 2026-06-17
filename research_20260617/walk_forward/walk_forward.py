"""
Walk-Forward 滚动验证框架（验证版）
==================================

借鉴来源：
- QuantConnect / Zipline-Reloaded：TimeSeriesSplit + purge 间隔的标准做法
- 微软 Qlib 的 `segments = {train, valid, test}` + 滚动 retrain 机制
- 《Advances in Financial Machine Learning》by Marcos López de Prado,
  第 7/11 章：CPCV (Combinatorial Purged Cross-Validation) 与 embargo
- 现有 jingni-trader 的 `TRAIN_WINDOW_MONTHS=36, VALIDATION_WINDOW_MONTHS=12,
  TEST_WINDOW_MONTHS=12, PURGE_GAP_DAYS=5` 配置（scripts/config.py）但未提供
  对应的执行框架

设计目标：
1. 把现有 Config 中的 train/valid/test 三段式时间窗口
   转化为可执行的 walk-forward 框架。
2. 在每段中执行「训练 -> 验证 -> 测试」三步，并 purge gap 避免标签泄露。
3. 聚合所有 out-of-sample 段的结果，给出综合绩效指标。
4. 避免在 main 分支上直接修改 engine.py，仅作为参考实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd


@dataclass
class WalkForwardSegment:
    """单个滚动窗口"""

    segment_id: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str


@dataclass
class WalkForwardConfig:
    """滚动窗口配置（与 jingni-trader scripts/config.py 字段对齐）"""

    train_window_months: int = 36
    valid_window_months: int = 12
    test_window_months: int = 12
    purge_gap_days: int = 5
    step_months: int = 12  # 窗口向前滑动的步长
    anchored: bool = False  # True=扩展窗口（train 起点固定），False=滚动窗口
    min_train_months: int = 12


class WalkForwardSplitter:
    """时间序列滚动划分器"""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def split(self, dates: pd.DatetimeIndex) -> List[WalkForwardSegment]:
        """根据日期索引生成多段 (train, valid, test) 划分

        参数:
            dates: 全样本日期索引（按时间升序）

        说明：
            - anchored=False（滚动窗口）：train 起点每次向前推 step_months，
              train 长度保持 train_window_months
            - anchored=True（扩展窗口）：train 起点固定为 start，每次只把
              验证/测试窗口向前推；train 长度随时间增加
        """
        if len(dates) == 0:
            return []
        start, end = dates[0], dates[-1]
        segments: List[WalkForwardSegment] = []
        sid = 0

        # test 窗口每次向前推 step_months
        test_start_cursor = start + pd.DateOffset(
            months=self.config.train_window_months
        ) + pd.DateOffset(months=self.config.valid_window_months) + pd.DateOffset(
            days=self.config.purge_gap_days * 2
        )

        max_iters = 200
        for _ in range(max_iters):
            test_start = test_start_cursor + pd.DateOffset(days=0)
            test_end = test_start + pd.DateOffset(months=self.config.test_window_months)
            if test_end > end:
                break
            valid_end = test_start - pd.DateOffset(days=self.config.purge_gap_days)
            valid_start = valid_end - pd.DateOffset(months=self.config.valid_window_months)
            if self.config.anchored:
                train_start = start
            else:
                # 滚动模式：train_end = valid_start - purge_gap
                train_end = valid_start - pd.DateOffset(days=self.config.purge_gap_days)
                train_start = train_end - pd.DateOffset(
                    months=self.config.train_window_months
                )
            train_end = valid_start - pd.DateOffset(days=self.config.purge_gap_days)

            segments.append(
                WalkForwardSegment(
                    segment_id=sid,
                    train_start=train_start.strftime("%Y-%m-%d"),
                    train_end=train_end.strftime("%Y-%m-%d"),
                    valid_start=valid_start.strftime("%Y-%m-%d"),
                    valid_end=valid_end.strftime("%Y-%m-%d"),
                    test_start=test_start.strftime("%Y-%m-%d"),
                    test_end=test_end.strftime("%Y-%m-%d"),
                )
            )
            sid += 1
            test_start_cursor = test_start_cursor + pd.DateOffset(
                months=self.config.step_months
            )
        return segments


def filter_by_range(
    df: pd.DataFrame, start: str, end: str, date_col: str = "date"
) -> pd.DataFrame:
    """按日期范围过滤"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df[(df[date_col] >= pd.to_datetime(start)) & (df[date_col] < pd.to_datetime(end))]


@dataclass
class WalkForwardResult:
    """滚动验证结果聚合"""

    segments: List[Dict] = field(default_factory=list)
    aggregated_metrics: Dict = field(default_factory=dict)
    out_of_sample_equity: Optional[pd.Series] = None

    def to_dict(self) -> Dict:
        return {
            "segments": self.segments,
            "aggregated_metrics": self.aggregated_metrics,
            "out_of_sample_equity": (
                None if self.out_of_sample_equity is None
                else [
                    {"date": str(d), "equity": float(v)}
                    for d, v in self.out_of_sample_equity.items()
                ]
            ),
        }


def aggregate_metrics(per_segment_metrics: List[Dict]) -> Dict:
    """聚合多段绩效指标

    关键指标：
    - avg_sharpe: 平均夏普（稳定性）
    - sharpe_std: 夏普标准差
    - sharpe_ir: 夏普均值/标准差（meta-IR）
    - worst_drawdown: 最差段最大回撤
    - best_return / worst_return: 收益分布
    - consistency: 正收益段占比
    """
    if not per_segment_metrics:
        return {}
    sharpes = [m.get("sharpe_ratio", 0.0) for m in per_segment_metrics]
    returns = [m.get("total_return", 0.0) for m in per_segment_metrics]
    drawdowns = [m.get("max_drawdown", 0.0) for m in per_segment_metrics]
    return {
        "n_segments": len(per_segment_metrics),
        "avg_sharpe": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
        "sharpe_ir": (
            float(np.mean(sharpes) / np.std(sharpes, ddof=1))
            if len(sharpes) > 1 and np.std(sharpes, ddof=1) > 0
            else 0.0
        ),
        "consistency": float(np.mean([1 if r > 0 else 0 for r in returns])),
        "avg_return": float(np.mean(returns)),
        "best_return": float(np.max(returns)),
        "worst_return": float(np.min(returns)),
        "worst_drawdown": float(np.min(drawdowns)),
        "avg_drawdown": float(np.mean(drawdowns)),
    }


def run_walk_forward(
    data: pd.DataFrame,
    splitter: WalkForwardSplitter,
    train_fn: Callable[[pd.DataFrame], Dict],
    valid_fn: Callable[[pd.DataFrame, Dict], Dict],
    test_fn: Callable[[pd.DataFrame, Dict, Dict], Dict],
    date_col: str = "date",
) -> WalkForwardResult:
    """执行 Walk-Forward 滚动验证

    参数:
        data: 全样本数据（date/code/...）
        splitter: 时间窗口划分器
        train_fn: 训练函数，输入 train 段数据，返回模型参数字典
        valid_fn: 验证函数，输入 valid 段数据 + 模型参数，返回调优后参数
        test_fn: 测试函数，输入 test 段数据 + 调优后参数 + 训练信息，返回绩效 dict
    """
    raw_dates = pd.to_datetime(data[date_col]).drop_duplicates().sort_values()
    dates = pd.DatetimeIndex(raw_dates.values)
    segments = splitter.split(dates)

    per_seg_metrics: List[Dict] = []
    seg_records: List[Dict] = []
    oos_equities: List[pd.Series] = []

    for seg in segments:
        train_df = filter_by_range(data, seg.train_start, seg.train_end, date_col)
        valid_df = filter_by_range(data, seg.valid_start, seg.valid_end, date_col)
        test_df = filter_by_range(data, seg.test_start, seg.test_end, date_col)

        if train_df.empty or valid_df.empty or test_df.empty:
            continue
        try:
            train_info = train_fn(train_df)
            tuned_params = valid_fn(valid_df, train_info)
            test_result = test_fn(test_df, tuned_params, train_info)
        except Exception as exc:  # noqa: BLE001
            seg_records.append(
                {
                    "segment_id": seg.segment_id,
                    "error": str(exc),
                }
            )
            continue

        per_seg_metrics.append(test_result.get("metrics", {}))
        seg_records.append(
            {
                "segment_id": seg.segment_id,
                "train_range": [seg.train_start, seg.train_end],
                "valid_range": [seg.valid_start, seg.valid_end],
                "test_range": [seg.test_start, seg.test_end],
                "metrics": test_result.get("metrics", {}),
            }
        )
        eq = test_result.get("equity_curve")
        if isinstance(eq, pd.DataFrame) and not eq.empty:
            oos_equities.append(eq.set_index(date_col)["equity"])

    # 拼接所有 OOS 段
    oos_eq: Optional[pd.Series] = None
    if oos_equities:
        oos_eq = pd.concat(oos_equities)
        oos_eq = oos_eq[~oos_eq.index.duplicated(keep="last")].sort_index()
    agg = aggregate_metrics(per_seg_metrics)
    return WalkForwardResult(
        segments=seg_records, aggregated_metrics=agg, out_of_sample_equity=oos_eq
    )
