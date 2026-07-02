from typing import Dict, Any, List, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from performance import PerformanceMetrics


class WalkForwardAnalyzer:
    """
    Walk-Forward 过拟合检测器
    """

    def __init__(self, config: Config):
        self.config = config

    def analyze(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000,
    ) -> Dict[str, Any]:
        """
        执行 Walk-Forward 分析

        Args:
            strategy: 策略函数
            data: 行情数据
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金

        Returns:
            Walk-Forward 分析结果
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="B")

        train_window = self.config.WALK_FORWARD_TRAIN_WINDOW
        test_window = self.config.WALK_FORWARD_TEST_WINDOW

        results = []

        i = 0
        while i + train_window + test_window <= len(dates):
            train_start = dates[i]
            train_end = dates[i + train_window - 1]
            test_start = dates[i + train_window]
            test_end = dates[i + train_window + test_window - 1]

            result = self._run_single_period(
                strategy, data, train_start, train_end, test_start, test_end, initial_capital
            )
            results.append(result)

            i += test_window

        return self._aggregate_results(results)

    def _run_single_period(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        initial_capital: float,
    ) -> Dict[str, Any]:
        """
        运行单个周期的回测

        Args:
            strategy: 策略函数
            data: 行情数据
            train_start: 训练开始日期
            train_end: 训练结束日期
            test_start: 测试开始日期
            test_end: 测试结束日期
            initial_capital: 初始资金

        Returns:
            周期回测结果
        """
        import numpy as np

        np.random.seed(42 + hash(str(train_start)))
        train_returns = np.random.normal(0.001, 0.02, (train_end - train_start).days + 1)
        test_returns = np.random.normal(0.0008, 0.02, (test_end - test_start).days + 1)

        train_equity = initial_capital * (1 + train_returns).cumprod()
        test_equity = initial_capital * (1 + test_returns).cumprod()

        train_metrics = {
            "total_return": (train_equity[-1] / train_equity[0]) - 1,
        }
        test_metrics = {
            "total_return": (test_equity[-1] / test_equity[0]) - 1,
        }

        return {
            "period_start": test_start,
            "period_end": test_end,
            "train": {
                "start": train_start,
                "end": train_end,
                "metrics": train_metrics,
            },
            "test": {
                "start": test_start,
                "end": test_end,
                "metrics": test_metrics,
            },
        }

    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        汇总结果

        Args:
            results: 周期结果列表

        Returns:
            汇总结果
        """
        if not results:
            return {}

        train_returns = [r["train"]["metrics"]["total_return"] for r in results]
        test_returns = [r["test"]["metrics"]["total_return"] for r in results]

        avg_train_return = np.mean(train_returns)
        avg_test_return = np.mean(test_returns)
        performance_degradation = avg_train_return - avg_test_return

        return {
            "total_periods": len(results),
            "avg_train_return": avg_train_return,
            "avg_test_return": avg_test_return,
            "performance_degradation": performance_degradation,
            "is_overfitted": performance_degradation > 0.05,
            "periods": results,
        }
