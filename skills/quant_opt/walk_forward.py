"""
Walk-forward 滚动训练与回测框架
================================

**借鉴来源**：
- AKQuant (akfamily/akquant) 的 ``ml.walk_forward`` 框架
- FinRL-X (AI4Finance-Foundation/FinRL-Trading) 的滚动训练与样本外再验证
- Qlib (microsoft/qlib) 的 ``qrun`` 滚动窗口

**当前 jingni-trader 的痛点**：
- ``strategy-model-engine`` 只有单次训练 + 单次回测，无法做时序交叉验证
- 用户难以判断模型的样本外表现
- 缺乏统一的 walk-forward 接口

**优化目标**：
- 提供 ``WalkForwardRunner`` 类，封装"训练-验证-测试"三阶段滚动
- 支持时间序列拆分（带 purge gap 防数据泄漏）
- 输出每个窗口的性能指标及聚合统计

**关键设计**：
1. 滑动窗口：``[train_start, train_end) → [valid_start, valid_end) → [test_start, test_end)``
2. purge gap：在训练/测试之间留 N 天空窗，防止前视偏差
3. 任意 ``train_fn`` / ``backtest_fn`` 都可以注入
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class WalkForwardWindow:
    """单个 walk-forward 窗口的配置"""
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    window_id: int = 0


@dataclass
class WindowResult:
    """单个窗口的回测结果"""
    window: WalkForwardWindow
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


def generate_windows(
    start_date: str,
    end_date: str,
    train_months: int = 36,
    valid_months: int = 6,
    test_months: int = 6,
    purge_gap_days: int = 5,
    step_months: int = 6,
) -> List[WalkForwardWindow]:
    """
    生成 walk-forward 窗口序列

    Args:
        start_date: 数据起始日 ``YYYY-MM-DD``
        end_date: 数据结束日 ``YYYY-MM-DD``
        train_months: 训练窗口长度（月）
        valid_months: 验证窗口长度（月）
        test_months: 测试窗口长度（月）
        purge_gap_days: 训练/测试之间清洗期（天）
        step_months: 每次向前推进的步长（月）

    Returns:
        ``WalkForwardWindow`` 列表
    """
    from dateutil.relativedelta import relativedelta

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    windows = []
    wid = 0
    cursor = start
    while True:
        train_s = cursor
        train_e = train_s + relativedelta(months=train_months)
        valid_s = train_e + pd.Timedelta(days=purge_gap_days)
        valid_e = valid_s + relativedelta(months=valid_months)
        test_s = valid_e + pd.Timedelta(days=purge_gap_days)
        test_e = test_s + relativedelta(months=test_months)

        if test_e > end:
            break

        windows.append(WalkForwardWindow(
            train_start=train_s.strftime("%Y-%m-%d"),
            train_end=train_e.strftime("%Y-%m-%d"),
            valid_start=valid_s.strftime("%Y-%m-%d"),
            valid_end=valid_e.strftime("%Y-%m-%d"),
            test_start=test_s.strftime("%Y-%m-%d"),
            test_end=test_e.strftime("%Y-%m-%d"),
            window_id=wid,
        ))
        wid += 1
        cursor = cursor + relativedelta(months=step_months)
    return windows


class WalkForwardRunner:
    """
    Walk-forward 滚动训练与回测

    使用示例::

        runner = WalkForwardRunner(
            train_fn=my_train,
            backtest_fn=my_backtest,
        )
        windows = generate_windows("2018-01-01", "2024-01-01", ...)
        report = runner.run(data, windows)
    """

    def __init__(
        self,
        train_fn: Callable[[pd.DataFrame, WalkForwardWindow], Any],
        backtest_fn: Callable[[Any, pd.DataFrame, WalkForwardWindow], Dict[str, float]],
    ):
        """
        Args:
            train_fn: 训练函数, 接收 (data, window) -> model
            backtest_fn: 回测函数, 接收 (model, data, window) -> metrics dict
        """
        self.train_fn = train_fn
        self.backtest_fn = backtest_fn

    def run(
        self,
        data: pd.DataFrame,
        windows: List[WalkForwardWindow],
    ) -> Dict[str, Any]:
        """
        执行所有窗口的滚动训练/回测

        Returns:
            {
                "window_results": [WindowResult, ...],
                "summary": {metric: {mean, std, min, max}}
            }
        """
        results: List[WindowResult] = []
        for w in windows:
            try:
                train_data = self._slice(data, w.train_start, w.train_end)
                model = self.train_fn(train_data, w)
                test_data = self._slice(data, w.test_start, w.test_end)
                metrics = self.backtest_fn(model, test_data, w)
                results.append(WindowResult(
                    window=w,
                    metrics=metrics,
                    error=None,
                ))
            except Exception as e:
                results.append(WindowResult(
                    window=w,
                    metrics={},
                    error=str(e),
                ))

        summary = self._summarize(results)
        return {
            "window_results": results,
            "summary": summary,
        }

    @staticmethod
    def _slice(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """按日期切片"""
        if "date" not in data.columns:
            raise ValueError("data must contain 'date' column")
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        mask = (data["date"] >= pd.Timestamp(start)) & (data["date"] < pd.Timestamp(end))
        return data.loc[mask].reset_index(drop=True)

    @staticmethod
    def _summarize(results: List[WindowResult]) -> Dict[str, Dict[str, float]]:
        """对所有窗口结果做汇总统计"""
        import numpy as np
        all_metrics: Dict[str, List[float]] = {}
        for r in results:
            if r.error:
                continue
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)) and not pd.isna(v):
                    all_metrics.setdefault(k, []).append(float(v))
        summary = {}
        for k, vs in all_metrics.items():
            arr = np.array(vs)
            summary[k] = {
                "mean": float(arr.mean()) if len(arr) else 0.0,
                "std": float(arr.std()) if len(arr) else 0.0,
                "min": float(arr.min()) if len(arr) else 0.0,
                "max": float(arr.max()) if len(arr) else 0.0,
                "n_windows": len(arr),
            }
        return summary


__all__ = [
    "WalkForwardRunner",
    "WalkForwardWindow",
    "WindowResult",
    "generate_windows",
]
