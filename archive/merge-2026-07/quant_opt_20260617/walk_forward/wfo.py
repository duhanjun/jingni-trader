"""
Walk-Forward Optimization (WFO) 框架
====================================

借鉴来源:
- Qlib (microsoft/qlib, 36K+ stars) 的 segments/train_valid_test 时间分割 + 滚动重训
- Freqtrade 的 walk-forward 验证流程（rolling/anchored）
- jingni-trader 现有 strategy-model-engine.py 的 purged_group_ts_split

核心思想:
- 把数据按时间切分成 N 段：每段包含 [train, valid, test] 三个窗口
- 训练时只用 train，valid 用于选超参/早停，test 用于 OOS 评估
- 多段拼接 test，得到一个无未来信息泄露的累积 OOS 收益曲线
- 这是量化研究中防止「过拟合」的关键方法论

设计要点:
1. Anchored (锚定) vs Rolling (滚动) 两种模式：
   - anchored: train 起点固定，向未来扩展 (适应 regime 长期变化)
   - rolling: train 窗口长度固定 (更稳定的样本量)
2. Purge gap：train 末尾与 valid/test 起点之间留出空隙，避免标签穿越
3. Embargo：test 末尾后留 embargo 天不参与训练
4. 输出：每段 OOS 收益、拼接 OOS 收益、综合绩效指标
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
import numpy as np
import pandas as pd


@dataclass
class WFOSplit:
    """单段切分"""
    segment_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp        # 训练集 [start, end] (含)
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp        # 验证集 (可选)
    test_start: pd.Timestamp
    test_end: pd.Timestamp         # 测试集
    purge_days: int
    embargo_days: int


@dataclass
class WFOConfig:
    """WFO 配置"""
    n_splits: int = 5
    train_days: int = 252 * 2     # 训练窗口 2 年
    valid_days: int = 63          # 验证窗口 3 个月
    test_days: int = 63           # 测试窗口 3 个月
    purge_days: int = 5           # purge gap
    embargo_days: int = 5         # embargo
    anchored: bool = False        # anchored vs rolling
    min_train_days: int = 126     # 最小训练长度


class WalkForwardOptimizer:
    """Walk-Forward 优化器"""

    def __init__(self, config: Optional[WFOConfig] = None):
        self.config = config or WFOConfig()

    def split(self, dates: pd.DatetimeIndex) -> List[WFOSplit]:
        """
        生成 walk-forward 切分。

        参数:
            dates: 全部交易日的 DatetimeIndex（已排序）
        """
        cfg = self.config
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)
        dates = dates.sort_values().drop_duplicates()

        splits: List[WFOSplit] = []
        total_needed = cfg.train_days + cfg.valid_days + cfg.test_days
        if len(dates) < total_needed:
            return splits

        # 测试段起点列表（从后向前排）
        # test_end = dates[-1], test_start = dates[-test_days]
        # valid_start = test_start - valid_days, valid_end = test_start - 1
        # train_end = valid_start - purge_days - 1
        # train_start: rolling 时 = train_end - train_days + 1
        #              anchored 时 = dates[0]
        step = cfg.test_days  # 段间步长 = test_days，使段与段不重叠
        for i in range(cfg.n_splits):
            test_end_idx = len(dates) - 1 - i * step
            test_start_idx = test_end_idx - cfg.test_days + 1
            if test_start_idx < 0:
                break
            valid_end_idx = test_start_idx - cfg.purge_days - 1
            valid_start_idx = valid_end_idx - cfg.valid_days + 1
            train_end_idx = valid_start_idx - cfg.purge_days - 1
            if cfg.anchored:
                train_start_idx = 0
            else:
                train_start_idx = train_end_idx - cfg.train_days + 1

            if train_start_idx < 0 or (train_end_idx - train_start_idx + 1) < cfg.min_train_days:
                break
            if valid_start_idx < 0 or test_start_idx < 0:
                break

            split = WFOSplit(
                segment_id=i,
                train_start=dates[train_start_idx],
                train_end=dates[train_end_idx],
                valid_start=dates[valid_start_idx],
                valid_end=dates[valid_end_idx],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx],
                purge_days=cfg.purge_days,
                embargo_days=cfg.embargo_days,
            )
            splits.append(split)

        # 按 segment_id 升序返回（更早的在前）
        splits.reverse()
        for i, s in enumerate(splits):
            s.segment_id = i
        return splits

    def run(
        self,
        data: pd.DataFrame,
        train_fn: Callable[[pd.DataFrame, pd.DataFrame], Any],
        predict_fn: Callable[[Any, pd.DataFrame], pd.DataFrame],
        backtest_fn: Optional[Callable[[pd.DataFrame, pd.DataFrame], Dict[str, float]]] = None,
        date_col: str = "date",
    ) -> Dict[str, Any]:
        """
        执行 WFO。

        参数:
            data: 包含 [date_col, code, ...] 的全量数据
            train_fn(X_train, y_train) -> model: 训练函数
            predict_fn(model, X) -> DataFrame[code, date, signal/pred]: 预测函数
            backtest_fn(data, signals) -> dict:  可选回测函数（输出 metrics）
            date_col: 时间列名
        """
        dates = pd.DatetimeIndex(sorted(data[date_col].unique()))
        splits = self.split(dates)
        if not splits:
            return {"success": False, "error": "数据不足以生成任何切分"}

        segments_results: List[Dict[str, Any]] = []
        oos_returns_list: List[pd.Series] = []
        all_test_signals: List[pd.DataFrame] = []

        for sp in splits:
            train_mask = (data[date_col] >= sp.train_start) & (data[date_col] <= sp.train_end)
            valid_mask = (data[date_col] >= sp.valid_start) & (data[date_col] <= sp.valid_end)
            test_mask = (data[date_col] >= sp.test_start) & (data[date_col] <= sp.test_end)

            train_data = data[train_mask].copy()
            valid_data = data[valid_mask].copy()
            test_data = data[test_mask].copy()
            if train_data.empty or test_data.empty:
                continue

            try:
                # 训练阶段：可使用 valid_data 做超参选择
                model = train_fn(train_data, valid_data)
                # 预测 OOS
                test_signals = predict_fn(model, test_data)
                if test_signals is None or test_signals.empty:
                    continue
                all_test_signals.append(test_signals)

                # 回测 OOS
                seg_metrics: Dict[str, float] = {}
                if backtest_fn is not None:
                    seg_metrics = backtest_fn(test_data, test_signals)
                segments_results.append({
                    "segment_id": sp.segment_id,
                    "train_range": f"{sp.train_start.date()} ~ {sp.train_end.date()}",
                    "test_range": f"{sp.test_start.date()} ~ {sp.test_end.date()}",
                    "n_train": int(len(train_data)),
                    "n_test": int(len(test_data)),
                    "metrics": seg_metrics,
                })
                # 收集 OOS 收益（若有）
                if "equity_curve" in seg_metrics and not seg_metrics["equity_curve"].empty:
                    eq = seg_metrics["equity_curve"]
                    if isinstance(eq, pd.DataFrame) and "equity" in eq.columns:
                        ret = eq.set_index(date_col)["equity"].pct_change().dropna()
                        ret.index = pd.DatetimeIndex(ret.index)
                        oos_returns_list.append(ret)
            except Exception as e:
                segments_results.append({
                    "segment_id": sp.segment_id,
                    "error": str(e),
                })

        # 拼接 OOS 收益
        oos_summary: Dict[str, Any] = {}
        if oos_returns_list:
            combined = pd.concat(oos_returns_list).sort_index()
            # 去重（防止段间边界重复）
            combined = combined[~combined.index.duplicated(keep="last")]
            oos_summary = self._calc_oos_metrics(combined)

        return {
            "success": True,
            "n_segments": len(segments_results),
            "segments": segments_results,
            "oos_summary": oos_summary,
            "config": {
                "n_splits": self.config.n_splits,
                "train_days": self.config.train_days,
                "valid_days": self.config.valid_days,
                "test_days": self.config.test_days,
                "purge_days": self.config.purge_days,
                "embargo_days": self.config.embargo_days,
                "anchored": self.config.anchored,
            },
        }

    @staticmethod
    def _calc_oos_metrics(returns: pd.Series) -> Dict[str, float]:
        if returns.empty:
            return {}
        equity = (1 + returns).cumprod()
        total = equity.iloc[-1] - 1
        annual = (1 + total) ** (252 / max(len(returns), 1)) - 1
        vol = returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        running_max = equity.cummax()
        mdd = (equity / running_max - 1).min()
        sharpe = (annual - 0.03) / vol if vol > 0 else 0
        win_rate = (returns > 0).mean()
        return {
            "n_periods": int(len(returns)),
            "total_return": float(total),
            "annual_return": float(annual),
            "volatility": float(vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(mdd),
            "win_rate": float(win_rate),
            "first_date": str(returns.index[0].date()),
            "last_date": str(returns.index[-1].date()),
        }
