"""
walk_forward.py
===============

借鉴 Qlib ``RollingGen`` (https://github.com/microsoft/qlib) 与
vectorbt 的 robustness testing 思想, 实现 jingni-trader 缺失的稳健性验证。

设计目标
--------
1. ``WalkForwardValidator`` 类提供滚动窗口 / 拓展窗口训练-测试能力
2. 支持任意策略参数网格扫描
3. 输出每个 fold 的指标, 聚合得到稳健性分数 (out-of-sample Sharpe 均值与标准差)
4. 与 jingni-trader 现有 ``backtest-engine`` 兼容, 可直接传入
   ``Context`` 风格的数据 / 信号

参考
----
- Bailey, Borwein, López de Prado (2014): "Pseudo-Mathematics and Financial Charlatanism"
- Qlib ``RollingGen`` 实现: https://github.com/microsoft/qlib/blob/main/qlib/model/rollinggen.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .performance_metrics import compute_metrics

logger = logging.getLogger("quant_opt_20260616.wf")


@dataclass
class WalkForwardConfig:
    """Walk-forward 配置"""
    train_months: int = 36       # 训练窗口长度
    test_months: int = 12        # 测试窗口长度
    expanding: bool = False      # True: 拓展窗口, False: 滚动窗口
    min_train_months: int = 12   # 最小训练窗口
    step_months: Optional[int] = None  # 步长 (None=与 test_months 相同)


class WalkForwardValidator:
    """Walk-forward 验证器"""

    def __init__(self, cfg: Optional[WalkForwardConfig] = None):
        self.cfg = cfg or WalkForwardConfig()
        self.folds_: List[Dict] = []

    def _generate_splits(self, dates: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """生成 (train_start, train_end, test_start, test_end) 列表"""
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)
        dates = dates.sort_values().unique()
        if len(dates) < 30:
            raise ValueError("时间序列太短, 至少需要 30 个交易日")

        start, end = dates[0], dates[-1]
        cfg = self.cfg
        step = cfg.step_months or cfg.test_months

        splits = []
        # 训练起点固定为 start; 训练终点推进
        train_end = start + pd.DateOffset(months=cfg.train_months)
        while train_end + pd.DateOffset(months=cfg.test_months) <= end:
            test_start = train_end
            test_end = train_end + pd.DateOffset(months=cfg.test_months)
            if cfg.expanding:
                train_start = start
            else:
                train_start = train_end - pd.DateOffset(months=cfg.train_months)
            # 找到训练/测试期内的实际交易日期
            train_mask = (dates >= train_start) & (dates < train_end)
            test_mask = (dates >= test_start) & (dates < test_end)
            if train_mask.sum() < cfg.min_train_months * 20:  # 至少 ~20个交易日/月
                train_end += pd.DateOffset(months=step)
                continue
            if test_mask.sum() < 30:
                train_end += pd.DateOffset(months=step)
                continue
            actual_train_start = dates[train_mask].min()
            actual_train_end = dates[train_mask].max()
            actual_test_start = dates[test_mask].min()
            actual_test_end = dates[test_mask].max()
            splits.append((actual_train_start, actual_train_end,
                           actual_test_start, actual_test_end))
            train_end += pd.DateOffset(months=step)
        if not splits:
            raise ValueError(
                f"无法生成 walk-forward splits: 数据跨度 {start} - {end} 太短 "
                f"或 train/test 月数太大"
            )
        return splits

    def run(
        self,
        data: pd.DataFrame,                  # 必须含 date 列
        signal_factory: Callable,            # f(train_data, test_data, params) -> signal_df
        param_grid: List[Dict],              # 参数组合
        risk_free: float = 0.03,
    ) -> pd.DataFrame:
        """
        执行 walk-forward 验证

        参数:
            data: 含 'date' 与 OHLCV 列
            signal_factory: 函数 f(train_data, test_data, params) -> signal DataFrame
                            (列: code, date, signal) 或 DataFrame (index=date, columns=code, values=signal)
            param_grid: 参数组合
            risk_free: 年化无风险利率

        返回:
            DataFrame, 每行 = (params + fold_info + metrics)
        """
        if "date" not in data.columns:
            raise KeyError("data 必须包含 'date' 列")
        if not isinstance(data["date"].iloc[0], pd.Timestamp):
            data = data.copy()
            data["date"] = pd.to_datetime(data["date"])
        dates = pd.DatetimeIndex(data["date"].unique())
        splits = self._generate_splits(dates)
        rows: List[Dict] = []
        for i, (ts, te, vs, ve) in enumerate(splits, 1):
            train = data[(data["date"] >= ts) & (data["date"] <= te)].copy()
            test = data[(data["date"] >= vs) & (data["date"] <= ve)].copy()
            for params in param_grid:
                try:
                    sig = signal_factory(train, test, params)
                except TypeError:
                    # 兼容旧签名 f(train, params)
                    try:
                        sig = signal_factory(train, params)
                        # 把信号按 (date, code) 在 test 区间应用
                        if sig is None or sig.empty:
                            continue
                        if "date" in sig.columns:
                            sig = sig[sig["date"].isin(test["date"].unique())]
                    except Exception as e:
                        logger.warning(f"Fold {i} params={params} signal_factory 失败: {e}")
                        continue
                except Exception as e:
                    logger.warning(f"Fold {i} params={params} signal_factory 失败: {e}")
                    continue
                if sig is None or sig.empty:
                    continue
                # 标准化信号为宽表
                if "date" in sig.columns and "code" in sig.columns:
                    sig_wide = sig.pivot_table(index="date", columns="code", values="signal", aggfunc="first")
                else:
                    sig_wide = sig
                close_wide = test.pivot_table(index="date", columns="code", values="close", aggfunc="first")
                common = sig_wide.index.intersection(close_wide.index)
                if len(common) < 5:
                    logger.debug(f"Fold {i} params={params} 公共日期过少: {len(common)}")
                    continue
                sig_wide = sig_wide.loc[common].fillna(0)
                close_wide = close_wide.loc[common].ffill()
                from .vectorized_backtest import vectorized_backtest, VectorBTConfig
                result = vectorized_backtest(close_wide, sig_wide, VectorBTConfig())
                eq = result["equity_curve"]
                if eq.empty or len(eq) < 5:
                    continue
                eq = eq.set_index("date")
                metrics = compute_metrics(eq["equity"], eq["ret"], risk_free=risk_free)
                rows.append({
                    "fold": i,
                    "train_start": ts, "train_end": te,
                    "test_start": vs, "test_end": ve,
                    **{f"p_{k}": v for k, v in params.items()},
                    **{f"m_{k}": v for k, v in metrics.items()},
                })
        self.folds_ = rows
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """汇总每个参数组合在所有 fold 上的表现"""
        if not self.folds_:
            raise ValueError("请先调用 run() 生成 folds")
        df = pd.DataFrame(self.folds_)
        if df.empty:
            return df
        # 找参数列
        param_cols = [c for c in df.columns if c.startswith("p_")]
        metric_cols = [c for c in df.columns if c.startswith("m_")]
        if not metric_cols:
            return df
        agg = df.groupby(param_cols)[metric_cols].agg(["mean", "std", "min", "max"])
        # 展平
        agg.columns = [f"{c[2:]}_{c[1]}" for c in agg.columns]
        return agg.reset_index()


__all__ = ["WalkForwardConfig", "WalkForwardValidator"]