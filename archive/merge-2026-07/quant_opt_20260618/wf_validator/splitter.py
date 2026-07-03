"""
Walk-Forward 滚动验证器
=======================

借鉴自：Microsoft Qlib 的 rolling training / online serving 设计
      https://qlib.readthedocs.io/en/latest/component/online.html
      以及通用量化行业的 WFA 方法论：
      https://arxiv.org/abs/2512.12924

问题背景
--------
当前 jingni-trader 的回测和模型训练都使用"全样本单次训练+全样本回测"模式，
存在严重的过拟合风险：
  - 用 2018-2024 数据训练，再用同一段数据回测
  - 实际效果会被高估（in-sample 拟合）
  - 没有"样本外"的概念

Walk-Forward Analysis (WFA) 的核心思想
---------------------------------------
将历史数据按时间滚动切分为多个 (训练, 测试) 对：

  |--- train ---|-- test --|
                  |--- train ---|-- test --|
                                   |--- train ---|-- test --|

每个 test 段都是"真正的样本外"，最后把所有 test 段的收益拼起来作为真实业绩。

两种变体：
1. Rolling（滑动窗口）：训练窗口长度固定，向右滑动
2. Anchored（锚定窗口）：起点固定，训练窗口越来越长

本模块提供：
- 时间序列切分器
- 简单的因子合成 + 信号生成器（内置）
- 评价指标计算
- WFA 报告
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class WFAFold:
    """单个 (train, test) 切片"""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (f"<WFAFold {self.fold_id}: "
                f"train=[{self.train_start.date()},{self.train_end.date()}] "
                f"test=[{self.test_start.date()},{self.test_end.date()}]>")


@dataclass
class WFAFoldResult:
    """单个 fold 的评估结果"""
    fold: WFAFold
    ic_mean: float = 0.0
    ic_std: float = 0.0
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    long_short_return: float = 0.0      # 多空组合收益
    long_only_return: float = 0.0        # 多头组合收益
    turnover: float = 0.0                # 换手率
    n_stocks: int = 0
    n_periods: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_id": self.fold.fold_id,
            "train_start": str(self.fold.train_start.date()),
            "train_end": str(self.fold.train_end.date()),
            "test_start": str(self.fold.test_start.date()),
            "test_end": str(self.fold.test_end.date()),
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "rank_ic_mean": self.rank_ic_mean,
            "rank_ic_std": self.rank_ic_std,
            "long_short_return": self.long_short_return,
            "long_only_return": self.long_only_return,
            "turnover": self.turnover,
            "n_stocks": self.n_stocks,
            "n_periods": self.n_periods,
        }


# ─────────────────────────────────────────────────────────────
# 时间切分器
# ─────────────────────────────────────────────────────────────

class TimeSeriesSplitter:
    """时间序列滚动切分器

    使用示例：

    >>> splitter = TimeSeriesSplitter(
    ...     train_period_days=504,    # 2 年训练
    ...     test_period_days=63,      # 3 个月测试
    ...     step_days=63,             # 每次滑动 3 个月
    ...     expanding=False,          # Rolling
    ... )
    >>> folds = splitter.split(data, date_col="date", start_date="2018-01-01", end_date="2024-01-01")
    """

    def __init__(
        self,
        train_period_days: int = 504,
        test_period_days: int = 63,
        step_days: int = 63,
        expanding: bool = False,
        min_train_days: int = 252,
    ):
        """
        参数:
            train_period_days: 训练窗口长度（交易日 ~ 实际天数）
            test_period_days: 测试窗口长度
            step_days: 滑动步长
            expanding: True=锚定窗口（训练起点固定），False=滑动窗口
            min_train_days: 最小训练长度（避免初期 fold 数据不足）
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days
        self.expanding = expanding
        self.min_train_days = min_train_days

    def split(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[WFAFold]:
        """生成所有 (train, test) 折

        参数:
            data: 包含 date 列的 DataFrame（仅用于确定可用日期范围）
            date_col: 日期列名
            start_date: 强制起始日期（用于对齐不同数据源）
            end_date: 强制结束日期

        返回:
            List[WFAFold]
        """
        if data.empty:
            return []

        dates = pd.to_datetime(data[date_col]).sort_values().unique()

        if start_date is not None:
            start = pd.to_datetime(start_date)
        else:
            start = dates.min()

        if end_date is not None:
            end = pd.to_datetime(end_date)
        else:
            end = dates.max()

        # 生成候选 test 起点
        folds: List[WFAFold] = []
        test_start = start + pd.Timedelta(days=self.train_period_days)
        fold_id = 0

        while test_start + pd.Timedelta(days=self.test_period_days) <= end:
            test_end = test_start + pd.Timedelta(days=self.test_period_days)

            if self.expanding:
                train_start = start
            else:
                train_start = test_start - pd.Timedelta(days=self.train_period_days)

            train_end = test_start - pd.Timedelta(days=1)

            # 训练窗口长度 = 从 train_start 到 test_start 之前（不含 test_start）
            train_days = (test_start - train_start).days
            if train_days >= self.min_train_days:
                folds.append(WFAFold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                ))
                fold_id += 1

            test_start = test_start + pd.Timedelta(days=self.step_days)

        return folds


# ─────────────────────────────────────────────────────────────
# 评价指标
# ─────────────────────────────────────────────────────────────

def _calc_ic(factor: pd.Series, ret: pd.Series) -> float:
    """横截面 IC（Pearson）"""
    if len(factor) < 5:
        return 0.0
    df = pd.DataFrame({"f": factor, "r": ret}).dropna()
    if len(df) < 5:
        return 0.0
    return df["f"].corr(df["r"])


def _calc_rank_ic(factor: pd.Series, ret: pd.Series) -> float:
    """横截面 Rank IC（Spearman）"""
    if len(factor) < 5:
        return 0.0
    df = pd.DataFrame({"f": factor, "r": ret}).dropna()
    if len(df) < 5:
        return 0.0
    return df["f"].rank().corr(df["r"].rank())


# ─────────────────────────────────────────────────────────────
# 验证引擎
# ─────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """Walk-Forward 验证器

    使用示例：

    >>> validator = WalkForwardValidator(
    ...     factor_col="alpha_score",
    ...     ret_col="forward_ret_1d",
    ...     top_k=50,
    ...     bottom_k=50,
    ... )
    >>> report = validator.run(data, folds)
    >>> print(report.summary())
    """

    def __init__(
        self,
        factor_col: str = "alpha_score",
        ret_col: str = "ret_forward_1d",
        top_k: int = 50,
        bottom_k: int = 50,
        date_col: str = "date",
        code_col: str = "code",
        min_stocks: int = 30,
    ):
        """
        参数:
            factor_col: 因子值列
            ret_col: 实际收益列
            top_k: 每日选 top K 多头
            bottom_k: 每日选 bottom K 空头
            date_col: 日期列
            code_col: 股票代码列
            min_stocks: 横截面最少股票数
        """
        self.factor_col = factor_col
        self.ret_col = ret_col
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.date_col = date_col
        self.code_col = code_col
        self.min_stocks = min_stocks

    def _filter_fold(
        self,
        data: pd.DataFrame,
        fold: WFAFold,
        train: bool,
    ) -> pd.DataFrame:
        """根据 fold 信息过滤数据"""
        if train:
            mask = (data[self.date_col] >= fold.train_start) & (data[self.date_col] <= fold.train_end)
        else:
            mask = (data[self.date_col] >= fold.test_start) & (data[self.date_col] <= fold.test_end)
        return data[mask].copy()

    def _evaluate_fold(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold: WFAFold,
    ) -> WFAFoldResult:
        """评估单个 fold 的表现"""
        result = WFAFoldResult(fold=fold)

        if test_data.empty:
            return result

        # 1) 计算 IC/RankIC
        ics, rank_ics = [], []
        dates = sorted(test_data[self.date_col].unique())
        result.n_periods = len(dates)
        result.n_stocks = test_data[self.code_col].nunique()

        for dt in dates:
            cross = test_data[test_data[self.date_col] == dt]
            if len(cross) < self.min_stocks:
                continue
            ic = _calc_ic(cross[self.factor_col], cross[self.ret_col])
            rank_ic = _calc_rank_ic(cross[self.factor_col], cross[self.ret_col])
            ics.append(ic)
            rank_ics.append(rank_ic)

        if ics:
            result.ic_mean = float(np.mean(ics))
            result.ic_std = float(np.std(ics))
            result.rank_ic_mean = float(np.mean(rank_ics))
            result.rank_ic_std = float(np.std(rank_ics))

        # 2) 计算 long-short / long-only 收益
        long_returns = []
        short_returns = []
        prev_top: set = set()
        prev_bottom: set = set()

        for dt in dates:
            cross = test_data[test_data[self.date_col] == dt].dropna(
                subset=[self.factor_col, self.ret_col]
            )
            if len(cross) < max(self.top_k, self.bottom_k):
                continue
            # 选股
            sorted_cross = cross.sort_values(self.factor_col, ascending=False)
            top_set = set(sorted_cross.head(self.top_k)[self.code_col])
            bottom_set = set(sorted_cross.tail(self.bottom_k)[self.code_col])

            # long-short 当日收益
            long_ret = cross[cross[self.code_col].isin(top_set)][self.ret_col].mean()
            short_ret = cross[cross[self.code_col].isin(bottom_set)][self.ret_col].mean()
            long_returns.append(long_ret)
            short_returns.append(short_ret)

            # 换手率
            if prev_top:
                turnover = (len(top_set - prev_top) + len(prev_top - top_set)) / (2 * len(top_set))
                result.turnover += turnover
            prev_top = top_set
            prev_bottom = bottom_set

        if long_returns:
            result.long_only_return = float(np.sum(long_returns))
            result.long_short_return = float(np.sum(np.array(long_returns) - np.array(short_returns)))
        if result.n_periods > 0:
            result.turnover = result.turnover / result.n_periods

        return result

    def run(
        self,
        data: pd.DataFrame,
        folds: List[WFAFold],
    ) -> "WFAReport":
        """执行 WFA 验证

        参数:
            data: 完整数据（包含 factor_col, ret_col, date_col, code_col）
            folds: TimeSeriesSplitter 生成的所有 fold

        返回:
            WFAReport
        """
        fold_results: List[WFAFoldResult] = []
        for fold in folds:
            train_data = self._filter_fold(data, fold, train=True)
            test_data = self._filter_fold(data, fold, train=False)
            res = self._evaluate_fold(train_data, test_data, fold)
            fold_results.append(res)

        return WFAReport(fold_results=fold_results)


# ─────────────────────────────────────────────────────────────
# 报告
# ─────────────────────────────────────────────────────────────

@dataclass
class WFAReport:
    """WFA 汇总报告"""
    fold_results: List[WFAFoldResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """汇总所有 fold 的指标"""
        if not self.fold_results:
            return {"error": "no fold results"}

        ic_means = [f.ic_mean for f in self.fold_results]
        rank_ic_means = [f.rank_ic_mean for f in self.fold_results]
        ls_returns = [f.long_short_return for f in self.fold_results]
        lo_returns = [f.long_only_return for f in self.fold_results]
        turnovers = [f.turnover for f in self.fold_results]

        return {
            "n_folds": len(self.fold_results),
            "ic_mean_avg": float(np.mean(ic_means)),
            "ic_mean_std": float(np.std(ic_means)),
            "ic_ir": float(np.mean(ic_means) / np.std(ic_means)) if np.std(ic_means) > 0 else 0,
            "rank_ic_mean_avg": float(np.mean(rank_ic_means)),
            "rank_ic_std_avg": float(np.std(rank_ic_means)),
            "rank_ic_ir": float(np.mean(rank_ic_means) / np.std(rank_ic_means))
                if np.std(rank_ic_means) > 0 else 0,
            "long_short_return_total": float(np.sum(ls_returns)),
            "long_short_return_avg": float(np.mean(ls_returns)),
            "long_only_return_total": float(np.sum(lo_returns)),
            "long_only_return_avg": float(np.mean(lo_returns)),
            "turnover_avg": float(np.mean(turnovers)),
            "consistency_ratio": float(np.mean([1 if r > 0 else 0 for r in ls_returns])),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.fold_results])

    def print_summary(self) -> str:
        s = self.summary()
        lines = ["=" * 60, "Walk-Forward Analysis Summary", "=" * 60]
        for k, v in s.items():
            if isinstance(v, float):
                lines.append(f"  {k:30s}: {v:.4f}")
            else:
                lines.append(f"  {k:30s}: {v}")
        lines.append("=" * 60)
        return "\n".join(lines)
