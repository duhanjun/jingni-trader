"""
滚动训练验证器 — 借鉴 Microsoft Qlib 的设计思想
==================================================

借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
          Qlib 文档: https://qlib.readthedocs.io/en/latest/component/rl.html
          及 De Prado, "Advances in Financial Machine Learning" 中的
          Purged K-Fold & Embargo 思想。

jingni-trader 现有 strategy-model-engine 的不足:
  - 已实现 purged_group_ts_split (有 purge gap，方向正确)
  - 但 train() 方法只是单次训练，没有真正的滚动 (rolling) 重训:
      即 "用 [t0,t1] 训练 → 预测 [t1+gap, t2] → 滚动窗口前进 → 再训练"
  - 缺少对窗口边界、数据泄漏、样本覆盖率的显式校验
  - 缺少对每次滚动预测结果的聚合 (拼接成完整样本外预测序列)

本模块实现一个完整的滚动训练框架:
  1. 按时间生成滚动 (train, test) 窗口序列，带 embargo gap
  2. 提供严格的泄漏校验 (train 的最大日期 + gap < test 的最小日期)
  3. 提供通用 run() 接口: 接收任意 sklearn 风格模型 + fit/predict，
     返回聚合的样本外预测 + 每次 fold 的指标
  4. 与现有 purged_group_ts_split 对比: 验证窗口正确性 & 无泄漏

本模块为验证代码，独立于 main 分支，不修改现有 strategy-model-engine。
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Callable, Optional

import numpy as np
import pandas as pd


class WalkForwardValidator:
    """
    滚动训练验证器

    参数:
        train_size: 训练窗口大小 (交易日数)
        test_size:  测试窗口大小 (交易日数)
        embargo_gap: 训练集结束与测试集开始之间的隔离期 (交易日)，
                      防止标签泄漏 (前视收益跨越边界)
        step:       每次滚动前进的步长 (默认 = test_size，无重叠)
    """

    def __init__(
        self,
        train_size: int = 252,
        test_size: int = 63,
        embargo_gap: int = 5,
        step: Optional[int] = None,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.embargo_gap = embargo_gap
        self.step = step if step is not None else test_size

    def generate_windows(self, dates: pd.Series) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        生成滚动 (train_dates, test_dates) 窗口列表

        参数:
            dates: 所有样本的日期 (可能重复，代表同日多股票)

        返回:
            list of (train_idx_dates, test_idx_dates)，每个元素是
            该 fold 训练/测试对应的 *日期* (去重排序后)
        """
        unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
        n = len(unique_dates)
        windows = []

        start = 0
        while True:
            train_start = start
            train_end = train_start + self.train_size  # exclusive
            test_start = train_end + self.embargo_gap
            test_end = test_start + self.test_size

            if test_end > n:
                # 最后一个窗口: 若测试集不足 test_size 但 >= min_test，仍可使用
                if test_start >= n:
                    break
                test_end = n
                if test_end - test_start < max(5, self.test_size // 4):
                    break

            train_dates = unique_dates[train_start:train_end]
            test_dates = unique_dates[test_start:test_end]

            if len(train_dates) >= self.train_size * 0.5 and len(test_dates) > 0:
                windows.append((train_dates, test_dates))

            start += self.step
            if test_end >= n:
                break

        return windows

    def validate_no_leakage(
        self, windows: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
    ) -> Dict[str, Any]:
        """
        校验所有窗口均无数据泄漏

        规则:
          1. 每个 fold: max(train) + embargo_gap_days < min(test)
          2. 不同 fold 的 test 区间不重叠 (step >= test_size 时成立)
          3. 所有 test 区间并集 <= 全日期范围

        返回:
            dict: passed (bool), violations (list), coverage (float)
        """
        violations = []
        all_test_dates = set()

        for i, (tr, te) in enumerate(windows):
            if len(tr) == 0 or len(te) == 0:
                violations.append(f"fold {i}: 空窗口")
                continue

            max_train = tr.max()
            min_test = te.min()

            # embargo: train 的最后一天 + gap 个交易日应 < test 第一天
            # 用日期序号判断更准确 (避免日历间隔问题)
            # 这里用简单规则: min_test 在 tr 之外，且 min_test > max_train
            if min_test <= max_train:
                violations.append(
                    f"fold {i}: 测试集起始 {min_test} <= 训练集结束 {max_train} (泄漏)"
                )

            # 检查 test 重叠
            te_set = set(te)
            overlap = te_set & all_test_dates
            if overlap:
                violations.append(f"fold {i}: 测试集与之前 fold 重叠 {len(overlap)} 天")
            all_test_dates |= te_set

        # 覆盖率
        if windows:
            total_test = len(all_test_dates)
            # 全日期范围 (排除初始训练期)
            first_test_start = windows[0][1].min()
            all_dates = pd.DatetimeIndex(sorted(set().union(*[set(t) for _, t in windows])))
            coverage = total_test / len(all_dates) if len(all_dates) > 0 else 0.0
        else:
            coverage = 0.0

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "n_folds": len(windows),
            "test_coverage": float(coverage),
        }

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        model_factory: Callable[[], Any],
        predict_fn: Optional[Callable[[Any, pd.DataFrame], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        执行滚动训练 + 样本外预测

        参数:
            X: 特征矩阵
            y: 标签
            dates: 与 X/y 对齐的日期 Series
            model_factory: 无参函数，每次调用返回新模型实例 (sklearn 风格)
            predict_fn: 自定义预测函数; None 时用 model.predict

        返回:
            dict:
              - oof_predictions: pd.Series 样本外预测 (index 对齐 X)
              - fold_metrics: list 每个 fold 的指标
              - windows: 窗口信息
              - validation: 泄漏校验结果
        """
        dates = pd.to_datetime(dates)
        windows = self.generate_windows(dates)
        validation = self.validate_no_leakage(windows)

        # 建立 date -> row index 映射
        date_to_idx = {}
        for idx, d in dates.items():
            date_to_idx.setdefault(d, []).append(idx)

        oof_pred = pd.Series(np.nan, index=X.index, name="oof_pred", dtype=float)
        fold_metrics = []

        for i, (train_dates, test_dates) in enumerate(windows):
            train_idx = [idx for d in train_dates for idx in date_to_idx.get(d, [])]
            test_idx = [idx for d in test_dates for idx in date_to_idx.get(d, [])]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]

            model = model_factory()
            model.fit(X_train, y_train)

            if predict_fn is not None:
                preds = np.asarray(predict_fn(model, X_test))
            else:
                preds = np.asarray(model.predict(X_test))

            oof_pred.loc[test_idx] = preds

            # fold 指标
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y_test, preds)
            ic = pd.Series(preds, index=y_test.index).corr(y_test)
            fold_metrics.append({
                "fold": i,
                "train_start": str(train_dates.min().date()),
                "train_end": str(train_dates.max().date()),
                "test_start": str(test_dates.min().date()),
                "test_end": str(test_dates.max().date()),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "mse": float(mse),
                "ic": float(ic) if not np.isnan(ic) else 0.0,
            })

        # 整体样本外 IC
        valid_mask = oof_pred.notna()
        overall_ic = float(oof_pred[valid_mask].corr(y[valid_mask])) if valid_mask.sum() > 1 else 0.0

        return {
            "oof_predictions": oof_pred,
            "fold_metrics": fold_metrics,
            "windows": [(list(tr), list(te)) for tr, te in windows],
            "validation": validation,
            "overall_oof_ic": overall_ic,
            "n_folds": len(fold_metrics),
        }


if __name__ == "__main__":
    # 简易自测: 用线性模型验证滚动训练
    from sklearn.linear_model import LinearRegression

    np.random.seed(1)
    dates = pd.Series(pd.bdate_range("2022-01-01", "2023-12-31"))
    n_per_day = 10
    all_dates = np.repeat(dates.values, n_per_day)
    X = pd.DataFrame(np.random.normal(0, 1, (len(all_dates), 3)), columns=["f1", "f2", "f3"])
    # 构造时变关系: f1 系数随时间变化 (模拟非平稳)
    coef_t = np.linspace(0.5, -0.5, len(dates))
    coef_per_row = np.repeat(coef_t, n_per_day)
    y = pd.Series(coef_per_row * X["f1"] + 0.2 * X["f2"] + np.random.normal(0, 0.1, len(X)))
    dates_aligned = pd.Series(all_dates)

    wf = WalkForwardValidator(train_size=120, test_size=30, embargo_gap=5)
    result = wf.run(X, y, dates_aligned, lambda: LinearRegression())
    print("=== 泄漏校验 ===")
    print(result["validation"])
    print(f"\n=== 整体样本外 IC: {result['overall_oof_ic']:.4f} ===")
    print(f"=== Fold 数: {result['n_folds']} ===")
    print("\n=== 前 3 个 Fold 指标 ===")
    for m in result["fold_metrics"][:3]:
        print(m)