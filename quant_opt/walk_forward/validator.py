"""
借鉴来源: Microsoft Qlib Trainer / TrainerRM + Qlib RollingWindowExp
- 官方仓库: https://github.com/microsoft/qlib
- 核心模块: qlib/model/trainer.py, qlib/contrib/evaluate.py
- 论文: "Qlib: An AI-oriented Quantitative Investment Platform" (arXiv 2009.11189)

jingni-trader 现状:
  strategy-model-engine/engine.py 中只有
  `purged_group_ts_split()` 一次性切分 train/val,缺少:
    1) 滚动训练 (Walk-Forward): 训练集每次前移,而不是固定
    2) 多步长预测 (Ridge Regression multi-step, Qlib 特色)
    3) 综合评估: out-of-sample IC 序列、IC decay、IC vs turnover 等
    4) 跨折训练样本的 concat,避免 O(N^2) 内存

借鉴方案:
  提供一个 WalkForwardValidator,模拟"每月末滚动训练 + 未来 20 日预测"
  的真实投研流程,并返回多维度评估指标。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_opt.walk_forward")


# ===========================================================================
# 数据类
# ===========================================================================
@dataclass
class WalkForwardFold:
    """一个 fold 的所有元信息。Qlib 风格的 Fold 抽象。"""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_dates: List[pd.Timestamp] = field(default_factory=list)
    test_dates: List[pd.Timestamp] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """滚动验证的完整结果,Qlib 风格的 Result 聚合。"""
    folds: List[WalkForwardFold]
    predictions: pd.DataFrame              # 包含 fold_id, code, date, pred, label
    fold_metrics: List[Dict[str, float]]
    summary: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "n_folds": len(self.folds),
            "summary": self.summary,
            "fold_metrics": self.fold_metrics,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train_range": f"{f.train_start.date()} ~ {f.train_end.date()}",
                    "test_range":  f"{f.test_start.date()} ~ {f.test_end.date()}",
                    "n_train": len(f.train_dates),
                    "n_test":  len(f.test_dates),
                }
                for f in self.folds
            ],
        }


# ===========================================================================
# 主类
# ===========================================================================
class WalkForwardValidator:
    """
    滚动前向验证器 (Qlib TrainerRM 风格)。

    参数:
        train_window_months: 训练窗口长度 (月),Qlib 论文推荐 36
        test_window_months:  预测窗口长度 (月),Qlib 论文推荐 12
        step_months:         每次前进步长 (月),Qlib 默认 = test_window
        purge_days:          训练-测试间隔 (天),防止 label 泄漏
        min_train_samples:   单 fold 最少训练样本数
    """

    def __init__(
        self,
        train_window_months: int = 36,
        test_window_months: int = 12,
        step_months: Optional[int] = None,
        purge_days: int = 5,
        min_train_samples: int = 1000,
    ):
        self.train_window_months = train_window_months
        self.test_window_months = test_window_months
        self.step_months = step_months or test_window_months
        self.purge_days = purge_days
        self.min_train_samples = min_train_samples

    def split(self, dates: pd.Series) -> List[WalkForwardFold]:
        """把全部日期切成多个 fold,Qlib 风格的滚动切分。"""
        unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates.unique())))
        n = len(unique_dates)
        if n < 2:
            return []

        # 把月数转换成"交易日索引数"
        # Qlib: 用 month_lbound/月切分,这里用近似 (按日历月分)
        def date_to_idx(d):
            return unique_dates.searchsorted(d, side="left")

        train_idx = self.train_window_months * 21      # 一个月约 21 个交易日
        test_idx = self.test_window_months * 21
        step_idx = self.step_months * 21

        folds: List[WalkForwardFold] = []
        fid = 0
        # 第一个 fold: train 从头开始
        train_end_idx = train_idx
        while train_end_idx < n:
            train_start = unique_dates[0]
            train_end = unique_dates[train_end_idx - 1]
            test_start_idx = train_end_idx + self.purge_days
            test_end_idx = min(test_start_idx + test_idx, n)
            if test_end_idx > n:
                test_end_idx = n
            if test_start_idx >= test_end_idx:
                break
            test_start = unique_dates[test_start_idx]
            test_end = unique_dates[test_end_idx - 1]
            train_dates = [d for d in unique_dates[:train_end_idx]
                           if d <= train_end - pd.Timedelta(days=self.purge_days)]
            test_dates = list(unique_dates[test_start_idx:test_end_idx])
            if len(train_dates) < self.min_train_samples // 10:
                break
            folds.append(WalkForwardFold(
                fold_id=fid,
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                train_dates=train_dates, test_dates=test_dates,
            ))
            fid += 1
            train_end_idx += step_idx
        return folds

    # ----- 主流程 -----
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        model_factory: Callable,
    ) -> WalkForwardResult:
        """
        参数:
            X:             特征 DataFrame
            y:             标签 Series
            dates:         与 X 同长度的日期 Series
            model_factory: callable (无参) -> 一个 sklearn 风格 model,
                           每次 fold 都会重新调用以拿到"新模型"

        返回:
            WalkForwardResult
        """
        # 切分
        folds = self.split(dates)
        if not folds:
            raise ValueError("无法生成有效的 fold,数据是否够长?")
        logger.info(f"生成 {len(folds)} 个 fold")

        all_preds = []
        fold_metrics: List[Dict[str, float]] = []
        dates_arr = pd.to_datetime(dates)

        for f in folds:
            train_mask = dates_arr.isin(f.train_dates)
            test_mask = dates_arr.isin(f.test_dates)
            X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
            X_te, y_te = X.loc[test_mask], y.loc[test_mask]
            if len(X_tr) < self.min_train_samples or len(X_te) == 0:
                logger.warning(f"fold {f.fold_id} 样本过少,跳过")
                continue

            # 训练
            model = model_factory()
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)

            # 评估
            from scipy.stats import spearmanr, pearsonr
            ic_p, _ = pearsonr(pred, y_te)
            ic_s, _ = spearmanr(pred, y_te)
            metrics = {
                "fold_id": f.fold_id,
                "n_train": int(len(X_tr)),
                "n_test":  int(len(X_te)),
                "ic_pearson":  float(ic_p),
                "ic_spearman": float(ic_s),
            }
            fold_metrics.append(metrics)

            # 收集预测
            test_idx = X_te.index
            fold_pred = pd.DataFrame({
                "fold_id":  f.fold_id,
                "date":     dates_arr.loc[test_mask].values,
                "y_true":   y_te.values,
                "y_pred":   pred,
            }, index=test_idx)
            if "code" in X.columns or isinstance(X.index, pd.MultiIndex):
                if isinstance(X.index, pd.MultiIndex) and "code" in X.index.names:
                    fold_pred["code"] = X.index.get_level_values("code").values[test_idx]
                elif "code" in X.columns:
                    fold_pred["code"] = X["code"].values[test_idx]
            all_preds.append(fold_pred)
            logger.info(
                f"fold {f.fold_id} | train={f.train_start.date()}~{f.train_end.date()} "
                f"test={f.test_start.date()}~{f.test_end.date()} | "
                f"IC={ic_p:.4f} RankIC={ic_s:.4f}"
            )

        preds_df = pd.concat(all_preds, axis=0) if all_preds else pd.DataFrame()

        # 综合指标
        summary = self._aggregate_summary(preds_df, fold_metrics)
        return WalkForwardResult(
            folds=folds,
            predictions=preds_df,
            fold_metrics=fold_metrics,
            summary=summary,
        )

    @staticmethod
    def _aggregate_summary(preds: pd.DataFrame, fold_metrics: List[Dict]) -> Dict[str, float]:
        if preds.empty:
            return {}
        from scipy.stats import spearmanr, pearsonr
        # 整体 IC
        ic_p, _ = pearsonr(preds["y_pred"], preds["y_true"])
        ic_s, _ = spearmanr(preds["y_pred"], preds["y_true"])
        # IC 序列 (按 fold)
        ic_by_fold = [m["ic_pearson"] for m in fold_metrics]
        # IC IR (mean / std)
        ic_arr = np.array(ic_by_fold)
        ic_ir = float(ic_arr.mean() / ic_arr.std()) if ic_arr.std() > 0 else 0.0
        # 胜率 (IC > 0 的 fold 占比)
        win_rate = float((ic_arr > 0).mean())
        return {
            "overall_ic_pearson":  float(ic_p),
            "overall_ic_spearman": float(ic_s),
            "ic_mean":   float(ic_arr.mean()),
            "ic_std":    float(ic_arr.std()),
            "ic_ir":     ic_ir,
            "ic_win_rate": win_rate,
            "n_folds":   len(fold_metrics),
        }
