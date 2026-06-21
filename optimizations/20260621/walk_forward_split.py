"""
Optimization C: Walk-Forward 滚动训练验证分割器
================================================================================

借鉴来源:
  - Microsoft Qlib 的 RollingGen (qlib/workflow/task/gen.py): 滚动窗口生成
    train/valid/test 任务, 支持自定义 step、窗口长度, 是 Qlib 在线训练与
    防过拟合的核心机制。
  - akquant 的 Walk-forward Validation 框架: 原生集成 PyTorch/sklearn, 训练在
    当前 bar 完成、新模型在下一 bar 生效, ``test_window`` 定义 OOS 范围。

jingni-trader 现状痛点:
  - skills/strategy-model-engine/engine.py 的 ``purged_group_ts_split`` (第 109 行)
    是扩展式 (expanding) CV: 训练集始终从最早开始, 越往后越大。它没有真正的
    "滚动 + 重训" 语义, 也无法表达 "固定窗口训练 → 预测未来 N 天 → 平移 → 再训"。
  - 缺少 embargo (测试集后的隔离带), 在有标签前移 (forward return) 时容易泄露。
  - 没有把分割结果序列化, 难以复现实验。

本模块提供 ``RollingWindowSplit``, 生成 (train_start, train_end, test_start, test_end)
元组列表, 支持:
  - 滚动窗口 (固定 train 长度) 与扩展窗口 (expanding) 两种模式;
  - purge gap (训练集尾部切除, 防止 forward-return 泄露到测试集);
  - embargo (测试集后隔离带, 防止测试集的 forward return 泄露到下一折训练集);
  - 可选 valid 窗口。

本文件为独立验证实现, 不修改 main 分支任何代码。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List, Literal, Optional, Tuple

import pandas as pd

logger = logging.getLogger("walk-forward-split")


@dataclass(frozen=True)
class WindowSegment:
    """单个滚动折的区间定义 (闭区间, 日期为交易日)。"""
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    valid_start: Optional[pd.Timestamp] = None
    valid_end: Optional[pd.Timestamp] = None

    def describe(self) -> str:
        v = ""
        if self.valid_start is not None:
            v = f" valid=[{self.valid_start.date()} ~ {self.valid_end.date()}]"
        return (
            f"fold#{self.fold} train=[{self.train_start.date()} ~ "
            f"{self.train_end.date()}]{v} test=[{self.test_start.date()} ~ "
            f"{self.test_end.date()}]"
        )


class RollingWindowSplit:
    """Walk-forward 滚动窗口分割器。

    参数:
        train_window: 训练集交易日数 (滚动模式) 或最小训练集长度 (扩展模式)。
        test_window: 每折测试集交易日数。
        step: 滚动步长 (交易日数)。默认 = test_window。
        valid_window: 验证集交易日数, 0/None 表示不用验证集。
        purge_days: 从训练集尾部切除的交易日数, 防止 forward-return 标签泄露
                    到测试集 (标签通常前移 FORWARD_PERIOD 天)。
        embargo_days: 测试集之后隔离的交易日数, 防止测试集标签泄露到下一折
                      训练集。下一折 train_start 会跳过这段。
        mode: 'rolling' (固定窗口) 或 'expanding' (训练集从最早开始不断增长)。

    用法:
        >>> splitter = RollingWindowSplit(
        ...     train_window=252, test_window=63, step=63,
        ...     purge_days=21, embargo_days=10, mode='rolling')
        >>> folds = splitter.split(trading_dates)
        >>> for seg in folds:
        ...     train_mask = (dates >= seg.train_start) & (dates <= seg.train_end)
        ...     test_mask  = (dates >= seg.test_start)  & (dates <= seg.test_end)
    """

    def __init__(
        self,
        train_window: int,
        test_window: int,
        step: Optional[int] = None,
        valid_window: Optional[int] = 0,
        purge_days: int = 0,
        embargo_days: int = 0,
        mode: Literal["rolling", "expanding"] = "rolling",
    ):
        if train_window <= 0 or test_window <= 0:
            raise ValueError("train_window 和 test_window 必须为正")
        if purge_days < 0 or embargo_days < 0:
            raise ValueError("purge_days / embargo_days 不能为负")
        if mode not in ("rolling", "expanding"):
            raise ValueError("mode 必须是 'rolling' 或 'expanding'")
        self.train_window = train_window
        self.test_window = test_window
        self.step = step or test_window
        self.valid_window = valid_window or 0
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.mode = mode

    def split(
        self,
        trading_dates: List[pd.Timestamp] | pd.DatetimeIndex,
    ) -> List[WindowSegment]:
        """对交易日序列生成滚动折。

        trading_dates 必须是已排序、去重的交易日列表 (或 DatetimeIndex)。

        区间语义 (闭区间, 索引基于传入的交易日序列):
            train = [train_start_idx, train_end_idx]   (train_end 已扣除 purge)
            purge 区 = (train_end_idx, test_start_idx)  这段不用于训练
            test  = [test_start_idx, test_end_idx]
            embargo 区 = (本折 test_end_idx, 下一折 train_start_idx)  这段不用于下一折训练
        """
        dates = pd.DatetimeIndex(sorted(set(pd.Timestamp(d) for d in trading_dates)))
        n = len(dates)
        # 第一折测试集起始索引 = train_window + purge_days
        # (前面 train_window 天做训练, 再空出 purge_days 防标签泄露)
        min_required = self.train_window + self.purge_days + self.test_window
        if n < min_required:
            logger.warning(
                "交易日数 %d 不足以生成任何折 (需要 >= %d)", n, min_required,
            )
            return []

        folds: List[WindowSegment] = []
        fold_idx = 0
        test_start_idx = self.train_window + self.purge_days
        prev_test_end_idx = -(self.embargo_days + 1)  # 第一折无 embargo 约束

        while test_start_idx + self.test_window <= n:
            test_end_idx = test_start_idx + self.test_window - 1

            if self.mode == "rolling":
                train_start_idx = test_start_idx - self.purge_days - self.train_window
                train_end_idx = test_start_idx - self.purge_days - 1
            else:  # expanding
                train_start_idx = 0
                train_end_idx = test_start_idx - self.purge_days - 1

            # embargo: 下一折训练集起点必须 > 上一折测试集终点 + embargo
            # (rolling 模式下 train_start_idx 受此约束; 不满足则推进 test_start)
            if self.mode == "rolling":
                min_train_start = prev_test_end_idx + self.embargo_days + 1
                if train_start_idx < min_train_start:
                    test_start_idx += 1
                    continue

            if train_start_idx < 0 or train_end_idx < train_start_idx:
                break

            train_start = dates[train_start_idx]
            train_end = dates[train_end_idx]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]

            # 可选 valid 窗口: 从训练集末尾切出
            valid_start = valid_end = None
            if self.valid_window > 0:
                v_end_idx = train_end_idx
                v_start_idx = v_end_idx - self.valid_window + 1
                if v_start_idx > train_start_idx:
                    valid_start = dates[v_start_idx]
                    valid_end = dates[v_end_idx]
                    train_end = dates[v_start_idx - 1]  # 训练集截短到 valid 之前

            folds.append(WindowSegment(
                fold=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                valid_start=valid_start,
                valid_end=valid_end,
            ))
            prev_test_end_idx = test_end_idx
            test_start_idx += self.step
            fold_idx += 1

        logger.info("RollingWindowSplit(mode=%s) 生成 %d 折", self.mode, len(folds))
        return folds

    def iter_masks(
        self,
        dates: pd.Series,
    ) -> Iterator[Tuple[int, pd.Series, pd.Series, Optional[pd.Series]]]:
        """便捷迭代: 直接返回 (fold, train_mask, test_mask, valid_mask)。

        dates: 含日期的 Series (与训练数据行对齐)。
        """
        unique_dates = sorted(dates.unique())
        folds = self.split(unique_dates)
        for seg in folds:
            d = pd.to_datetime(dates)
            train_mask = (d >= seg.train_start) & (d <= seg.train_end)
            test_mask = (d >= seg.test_start) & (d <= seg.test_end)
            valid_mask = None
            if seg.valid_start is not None:
                valid_mask = (d >= seg.valid_start) & (d <= seg.valid_end)
            yield seg.fold, train_mask, test_mask, valid_mask
