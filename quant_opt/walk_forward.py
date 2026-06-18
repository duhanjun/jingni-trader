"""
Walk-Forward Validation (WFV) 框架
=================================

借鉴来源
--------
1. **AKQuant**  (akfamily/akquant) - 内置 Walk-forward Validation 框架，
   https://akquant.akfamily.xyz/en/advanced/ml/   重点参考其
   "防止未来函数" 与 "Pipeline 防止数据泄漏" 的设计哲学。
2. **Microsoft Qlib** (microsoft/qlib) - DataHandler 的时间片划分，
   https://github.com/microsoft/qlib  重点参考其时间安全的切分原则。

核心目标
--------
jingni-trader 当前 `strategy-model-engine` 中
``purged_group_ts_split`` 只生成 K-fold CV 切分（用于超参搜索），
**没有** 真正执行 Walk-forward Validation。

WFV 把"训练-验证-测试"三段时间严格按时间滚动，每一段都使用
历史信息训练并在未来窗口评估，更接近真实部署逻辑，能更稳定地
诊断过拟合。

设计要点
--------
- 锚点 (anchor) 滚动：每次以"截止日 d_i" 切出
  ``[d_i - train_window, d_i - purge_gap)`` 训练，
  ``[d_i, d_i + eval_window)`` 评估。
- purge gap：训练末与评估起点之间留出 ``PURGE_GAP_DAYS`` 天的"空窗"，
  避免标签泄漏（label leakage）。
- embargo：相邻评估窗口之间也设置 embargo，防止同一只票的
  相邻期样本重复出现在多个 fold 中。
- 聚合指标：返回 ``eval_metrics_per_fold`` + ``mean/std``，方便
  上层调度器判断是否"过拟合触发样本外再验证"（与 jingni-trader
  SKILL.md 的状态机分支一致）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_opt.walk_forward")


@dataclass
class WFVConfig:
    """Walk-Forward 验证配置"""

    train_window_days: int = 252 * 2          # 训练窗口 (交易日)
    eval_window_days: int = 63                # 评估窗口 (约一个季度)
    step_days: int = 63                       # 滚动步长
    purge_gap_days: int = 5                   # 训练/评估间的 purge gap
    embargo_days: int = 1                     # 评估窗口之间的 embargo
    min_train_samples: int = 1000             # 最少训练样本
    n_splits: Optional[int] = None            # 显式限制切分数量


@dataclass
class WFVFoldResult:
    """单个 fold 的验证结果"""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    n_train: int
    n_eval: int
    metrics: Dict[str, float] = field(default_factory=dict)
    oos_predictions: Optional[pd.Series] = None   # 样本外预测 (eval 区间)

    def to_dict(self) -> Dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, pd.Timestamp):
                d[k] = v.strftime("%Y-%m-%d")
        if d.get("oos_predictions") is not None:
            d.pop("oos_predictions")
        return d


@dataclass
class WFVResult:
    """Walk-Forward 整体结果"""

    config: WFVConfig
    folds: List[WFVFoldResult]
    aggregate_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def summary(self) -> Dict:
        return {
            "n_folds": self.n_folds,
            "config": asdict(self.config),
            "aggregate_metrics": self.aggregate_metrics,
            "fold_metrics": [f.to_dict() for f in self.folds],
        }


class WalkForwardValidator:
    """Walk-Forward 验证器

    Parameters
    ----------
    config : WFVConfig
        验证配置
    score_fn : callable, optional
        ``score_fn(y_true, y_pred) -> dict[metric_name, value]``
        用于在 eval 区间计算指标。如果为 None，会使用 IC (pearson) 作为兜底。
    """

    def __init__(
        self,
        config: Optional[WFVConfig] = None,
        score_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None,
    ) -> None:
        self.config = config or WFVConfig()
        self.score_fn = score_fn or self._default_score

    @staticmethod
    def _default_score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """默认评分: 皮尔森 IC + 方向准确率"""
        if len(y_true) < 2:
            return {"ic": 0.0, "direction_acc": 0.0}
        df = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
        if len(df) < 2:
            return {"ic": 0.0, "direction_acc": 0.0}
        ic = float(df["y"].corr(df["p"]))
        direction_acc = float((np.sign(df["y"]) == np.sign(df["p"])).mean())
        return {"ic": ic, "direction_acc": direction_acc}

    def _build_anchors(
        self,
        dates: pd.Series,
    ) -> List[pd.Timestamp]:
        """根据交易日序列生成评估窗口起点 (anchors)"""
        unique_dates = pd.Series(sorted(dates.unique()))
        if unique_dates.empty:
            return []

        first_date = unique_dates.iloc[0]
        last_date = unique_dates.iloc[-1]

        # 用 trading-day index 锚定，避免按自然日漂移
        idx_last = len(unique_dates) - 1
        idx_first_eval = self.config.train_window_days + self.config.purge_gap_days
        if idx_first_eval >= idx_last:
            return []

        anchors_idx: List[int] = []
        step = max(1, self.config.step_days)
        i = idx_first_eval
        while i + self.config.eval_window_days <= idx_last:
            anchors_idx.append(i)
            i += step
            if self.config.n_splits and len(anchors_idx) >= self.config.n_splits:
                break
        return [unique_dates.iloc[k] for k in anchors_idx]

    def split(
        self,
        dates: pd.Series,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生成 (train_idx, eval_idx) 列表"""
        df = pd.DataFrame({"date": pd.to_datetime(dates).values})
        df["date"] = pd.to_datetime(df["date"])
        anchors = self._build_anchors(df["date"])

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for anchor in anchors:
            train_end = anchor - pd.Timedelta(days=self.config.purge_gap_days)
            train_start = train_end - pd.Timedelta(days=self.config.train_window_days)
            eval_start = anchor
            eval_end = anchor + pd.Timedelta(days=self.config.eval_window_days)

            train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
            eval_mask = (df["date"] >= eval_start) & (df["date"] < eval_end)
            train_idx = df.index[train_mask].to_numpy()
            eval_idx = df.index[eval_mask].to_numpy()

            if (
                len(train_idx) >= self.config.min_train_samples
                and len(eval_idx) >= 30
            ):
                splits.append((train_idx, eval_idx))
        return splits

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        fit_predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    ) -> WFVResult:
        """执行 Walk-Forward 验证

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵 (index 任意)
        y : pd.Series
            标签
        dates : pd.Series
            与 X/y 对齐的日期
        fit_predict_fn : callable
            ``fit_predict_fn(X_train, y_train, X_eval) -> y_pred``，
            由调用方注入具体模型训练逻辑 (LightGBM, Linear, ...)。
        """
        dates = pd.to_datetime(dates)
        splits = self.split(dates)
        if not splits:
            logger.warning("Walk-Forward 切分结果为空，请检查 WFVConfig 或数据长度")
            return WFVResult(config=self.config, folds=[], aggregate_metrics={})

        folds: List[WFVFoldResult] = []
        for fold_id, (train_idx, eval_idx) in enumerate(splits, 1):
            train_dates = dates.iloc[train_idx]
            eval_dates = dates.iloc[eval_idx]

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_eval, y_eval = X.iloc[eval_idx], y.iloc[eval_idx]

            try:
                y_pred = fit_predict_fn(X_train, y_train, X_eval)
            except Exception as e:  # pragma: no cover
                logger.error("Fold %d 训练失败: %s", fold_id, e)
                continue

            metrics = self.score_fn(y_eval.to_numpy(), np.asarray(y_pred))
            oos = pd.Series(np.asarray(y_pred), index=eval_idx, name="pred")

            folds.append(
                WFVFoldResult(
                    fold_id=fold_id,
                    train_start=train_dates.min(),
                    train_end=train_dates.max(),
                    eval_start=eval_dates.min(),
                    eval_end=eval_dates.max(),
                    n_train=len(train_idx),
                    n_eval=len(eval_idx),
                    metrics=metrics,
                    oos_predictions=oos,
                )
            )

        aggregate = self._aggregate(folds)
        return WFVResult(config=self.config, folds=folds, aggregate_metrics=aggregate)

    @staticmethod
    def _aggregate(folds: List[WFVFoldResult]) -> Dict[str, Dict[str, float]]:
        if not folds:
            return {}
        keys = list(folds[0].metrics.keys())
        agg: Dict[str, Dict[str, float]] = {}
        for k in keys:
            vals = [f.metrics[k] for f in folds if k in f.metrics]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            agg[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": int(len(arr)),
            }
        return agg

    @staticmethod
    def detect_overfit(
        result: WFVResult,
        ic_ir_threshold: float = 0.3,
        std_ratio_threshold: float = 1.5,
    ) -> Dict[str, object]:
        """简易过拟合检测:
        - mean IC / std(IC) < ic_ir_threshold  -> 弱信号
        - |mean| / |std| 太小 或 std 过大 -> 高波动
        """
        if not result.folds:
            return {"overfit": None, "reason": "no folds"}

        ic_stats = result.aggregate_metrics.get("ic", {})
        mean = ic_stats.get("mean", 0.0)
        std = ic_stats.get("std", 0.0)
        ic_ir = abs(mean) / std if std > 0 else 0.0

        flags = []
        if ic_ir < ic_ir_threshold:
            flags.append(f"IC/IR={ic_ir:.3f} < {ic_ir_threshold}")
        if std != 0 and abs(std / (mean + 1e-9)) > std_ratio_threshold:
            flags.append(f"std/mean={std/(mean+1e-9):.2f} > {std_ratio_threshold}")

        return {
            "overfit": bool(flags),
            "ic_ir": ic_ir,
            "flags": flags,
            "mean_ic": mean,
            "std_ic": std,
        }
