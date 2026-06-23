"""
滚动训练框架 (Walk-Forward Rolling Framework)

借鉴来源: Microsoft Qlib RollingGen + TaskManager
核心思想: 将 walk-forward 训练解耦为"任务生成"与"任务执行"，
         滚动只是任务生成关注点，模型本身不感知滚动逻辑。

与 jingni-trader 现有 strategy-model-engine 的对比:
  - 原实现: 模型训练是单次 fit/predict，无内置 walk-forward 支持
  - 本实现: 生成自包含的滚动任务配置列表，每个任务有独立的 train/valid/test 段，
            可独立执行 (甚至分布式)，最后用 RecorderCollector 合并预测结果

Qlib 的关键设计:
  1. RollingGen: 生成滚动任务配置 (rolling / expanding 两种模式)
  2. TaskManager: 管理任务生命周期，支持分布式执行
  3. RecorderCollector: 合并各滚动任务的预测为统一时间序列

本文件为验证性实现，不修改 main 分支的任何代码。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger("walk-forward-rolling")


class RollingType(Enum):
    """滚动类型 (借鉴 Qlib RollingGen)"""
    ROLLING = "rolling"      # 滑动窗口: train 窗口固定长度，向前滑动
    EXPANDING = "expanding"  # 扩展窗口: train 起点固定，终点向前扩展


@dataclass
class Segment:
    """时间段定义 (train / valid / test)"""
    start: str
    end: str

    def contains(self, date: str) -> bool:
        return self.start <= date <= self.end

    def __repr__(self):
        return f"[{self.start} ~ {self.end}]"


@dataclass
class RollingTask:
    """
    单个滚动任务配置 (自包含，可独立执行)

    借鉴 Qlib: 每个任务是一个完整的配置，包含 train/valid/test 段，
    可独立传给模型训练器执行。
    """
    task_id: str
    rolling_type: RollingType
    train: Segment
    valid: Optional[Segment]
    test: Segment
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'rolling_type': self.rolling_type.value,
            'train': {'start': self.train.start, 'end': self.train.end},
            'valid': {'start': self.valid.start, 'end': self.valid.end} if self.valid else None,
            'test': {'start': self.test.start, 'end': self.test.end},
            'metadata': self.metadata,
        }


class RollingGen:
    """
    滚动任务生成器 (借鉴 Qlib qlib.workflow.task.gen.RollingGen)

    生成 walk-forward 训练的任务序列:
      Task 1: train[2018-01-01 ~ 2020-12-31] test[2021-01-01 ~ 2021-06-30]
      Task 2: train[2018-07-01 ~ 2021-06-30] test[2021-07-01 ~ 2021-12-31]
      Task 3: train[2019-01-01 ~ 2021-12-31] test[2022-01-01 ~ 2022-06-30]
      ...

    两种模式:
      - ROLLING: train 窗口固定长度，整体前移 (适合应对市场风格变化)
      - EXPANDING: train 起点固定，终点前移 (适合数据量逐步增多的场景)
    """

    def __init__(
        self,
        rolling_type: RollingType = RollingType.ROLLING,
        train_window: int = 730,    # 训练窗口天数 (约2年)
        valid_window: int = 180,    # 验证窗口天数 (约半年)
        test_window: int = 180,     # 测试窗口天数 (约半年)
        step: int = 180,            # 滚动步长天数 (约半年)
    ):
        self.rolling_type = rolling_type
        self.train_window = train_window
        self.valid_window = valid_window
        self.test_window = test_window
        self.step = step

    def generate(
        self,
        start_date: str,
        end_date: str,
        trading_dates: Optional[pd.DatetimeIndex] = None,
    ) -> List[RollingTask]:
        """
        生成滚动任务列表

        参数:
            start_date: 数据起始日期
            end_date: 数据结束日期
            trading_dates: 交易日历 (用于精确按交易日滚动)

        返回:
            List[RollingTask]
        """
        if trading_dates is not None and len(trading_dates) > 0:
            return self._generate_by_calendar(trading_dates)
        else:
            return self._generate_by_date(start_date, end_date)

    def _generate_by_calendar(self, trading_dates: pd.DatetimeIndex) -> List[RollingTask]:
        """按交易日历生成 (更精确)"""
        dates = trading_dates.sort_values()
        n = len(dates)
        train_n = self._days_to_trading_days(self.train_window)
        valid_n = self._days_to_trading_days(self.valid_window)
        test_n = self._days_to_trading_days(self.test_window)
        step_n = self._days_to_trading_days(self.step)

        tasks = []
        task_idx = 0
        # 第一个 test 段的起点
        test_start_idx = train_n + valid_n

        while test_start_idx + test_n <= n:
            if self.rolling_type == RollingType.ROLLING:
                train_start_idx = test_start_idx - train_n - valid_n
                train_end_idx = train_start_idx + train_n - 1
            else:  # EXPANDING
                train_start_idx = 0
                train_end_idx = test_start_idx - valid_n - 1

            if train_start_idx < 0:
                break

            valid_start_idx = train_end_idx + 1
            valid_end_idx = valid_start_idx + valid_n - 1
            test_end_idx = test_start_idx + test_n - 1

            if test_end_idx >= n:
                test_end_idx = n - 1

            train_seg = Segment(
                start=dates[train_start_idx].strftime('%Y-%m-%d'),
                end=dates[train_end_idx].strftime('%Y-%m-%d'),
            )
            valid_seg = Segment(
                start=dates[valid_start_idx].strftime('%Y-%m-%d'),
                end=dates[valid_end_idx].strftime('%Y-%m-%d'),
            )
            test_seg = Segment(
                start=dates[test_start_idx].strftime('%Y-%m-%d'),
                end=dates[test_end_idx].strftime('%Y-%m-%d'),
            )

            tasks.append(RollingTask(
                task_id=f"rolling_{task_idx:03d}",
                rolling_type=self.rolling_type,
                train=train_seg,
                valid=valid_seg,
                test=test_seg,
                metadata={
                    'train_days': train_end_idx - train_start_idx + 1,
                    'test_days': test_end_idx - test_start_idx + 1,
                },
            ))
            task_idx += 1
            test_start_idx += step_n

        logger.info(f"生成 {len(tasks)} 个滚动任务 (模式: {self.rolling_type.value})")
        return tasks

    def _generate_by_date(self, start_date: str, end_date: str) -> List[RollingTask]:
        """按自然日生成 (近似)"""
        dates = pd.date_range(start_date, end_date, freq='D')
        return self._generate_by_calendar(dates)

    @staticmethod
    def _days_to_trading_days(days: int) -> int:
        """自然日转交易日 (近似: 交易日约占自然日的 5/7)"""
        return max(1, int(days * 5 / 7))


class TaskExecutor:
    """
    任务执行器 (借鉴 Qlib TaskManager + Trainer)

    接收滚动任务列表，逐个执行用户提供的训练函数，收集结果。
    支持简单的错误隔离: 单个任务失败不影响其他任务。
    """

    def __init__(self, train_func: Callable[[RollingTask], Dict[str, Any]]):
        """
        参数:
            train_func: 用户提供的训练函数，接收 RollingTask，返回结果字典
                        结果字典应包含 'predictions' (pd.Series) 和 'metrics' (dict)
        """
        self.train_func = train_func

    def run(self, tasks: List[RollingTask]) -> List[Dict[str, Any]]:
        """执行所有任务"""
        results = []
        for i, task in enumerate(tasks):
            logger.info(f"执行任务 {i+1}/{len(tasks)}: {task.task_id} test={task.test}")
            try:
                result = self.train_func(task)
                result['task_id'] = task.task_id
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                logger.error(f"任务 {task.task_id} 失败: {e}")
                results.append({
                    'task_id': task.task_id,
                    'status': 'failed',
                    'error': str(e),
                })
        return results


class RecorderCollector:
    """
    记录收集器 (借鉴 Qlib RecorderCollector)

    将各滚动任务在各自 test 段的预测结果合并为统一的时间序列，
    用于后续的回测和绩效评估。
    """

    @staticmethod
    def collect_predictions(
        task_results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        合并各任务的预测结果

        参数:
            task_results: TaskExecutor.run() 的返回值

        返回:
            DataFrame, 列: date, code, prediction, task_id
        """
        all_preds = []
        for r in task_results:
            if r.get('status') != 'success':
                continue
            preds = r.get('predictions')
            if preds is None or len(preds) == 0:
                continue
            if isinstance(preds, pd.DataFrame):
                preds = preds.copy()
            else:
                preds = pd.DataFrame(preds)
            preds['task_id'] = r['task_id']
            all_preds.append(preds)

        if not all_preds:
            return pd.DataFrame(columns=['date', 'code', 'prediction', 'task_id'])

        combined = pd.concat(all_preds, ignore_index=True)
        # 去重: 同一 (date, code) 只保留最后一个任务的预测
        combined = combined.sort_values('task_id').drop_duplicates(
            subset=['date', 'code'], keep='last'
        )
        logger.info(f"合并预测: {len(combined)} 条记录, 来自 {len(all_preds)} 个任务")
        return combined

    @staticmethod
    def collect_metrics(task_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """合并各任务的指标"""
        rows = []
        for r in task_results:
            if r.get('status') != 'success':
                continue
            row = {'task_id': r['task_id']}
            row.update(r.get('metrics', {}))
            rows.append(row)
        return pd.DataFrame(rows)


# ============================================================
# 验证用: 简单的 walk-forward 回测
# ============================================================

def walk_forward_backtest(
    data: pd.DataFrame,
    factor_col: str,
    return_col: str = 'forward_return_5d',
    train_start: str = '2019-01-01',
    train_end: str = '2024-12-31',
    top_n: int = 20,
    train_window: int = 730,
    test_window: int = 180,
    step: int = 180,
) -> Dict[str, Any]:
    """
    简单的 walk-forward 因子回测验证

    流程:
    1. 用 RollingGen 生成滚动任务
    2. 每个任务: 在 train 段计算因子 IC，在 test 段按因子选股
    3. 合并各 test 段的选股结果，计算累计收益
    """
    if 'date' not in data.columns or factor_col not in data.columns:
        return {'success': False, 'error': '数据缺少必要列'}

    data = data.sort_values(['code', 'date']).copy()
    trading_dates = pd.DatetimeIndex(sorted(data['date'].unique()))

    gen = RollingGen(
        rolling_type=RollingType.ROLLING,
        train_window=train_window,
        valid_window=0,
        test_window=test_window,
        step=step,
    )
    tasks = gen.generate(train_start, train_end, trading_dates)

    def _train_func(task: RollingTask) -> Dict[str, Any]:
        # train 段: 计算 IC
        train_mask = (data['date'] >= task.train.start) & (data['date'] <= task.train.end)
        train_data = data[train_mask]
        if len(train_data) == 0 or return_col not in train_data.columns:
            ic = 0.0
        else:
            valid = train_data[[factor_col, return_col]].dropna()
            if len(valid) > 10:
                ic = valid[factor_col].corr(valid[return_col])
            else:
                ic = 0.0

        # test 段: 选股
        test_mask = (data['date'] >= task.test.start) & (data['date'] <= task.test.end)
        test_data = data[test_mask].copy()
        if len(test_data) == 0:
            return {'predictions': pd.DataFrame(), 'metrics': {'ic': ic}}

        # 每日选 top_n
        test_data['rank'] = test_data.groupby('date')[factor_col].rank(ascending=False)
        selected = test_data[test_data['rank'] <= top_n][['date', 'code', 'rank']].copy()
        selected.rename(columns={'rank': 'prediction'}, inplace=True)

        return {
            'predictions': selected,
            'metrics': {'train_ic': float(ic), 'n_test_days': test_data['date'].nunique()},
        }

    executor = TaskExecutor(_train_func)
    results = executor.run(tasks)

    combined_preds = RecorderCollector.collect_predictions(results)
    metrics_df = RecorderCollector.collect_metrics(results)

    return {
        'success': True,
        'n_tasks': len(tasks),
        'n_successful': sum(1 for r in results if r.get('status') == 'success'),
        'combined_predictions': combined_preds,
        'task_metrics': metrics_df,
        'tasks': [t.to_dict() for t in tasks],
    }
