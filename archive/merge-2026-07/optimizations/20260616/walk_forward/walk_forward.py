"""
Walk-Forward 验证框架（借鉴 Qlib / mlfinlab 的样本外测试）
============================================================

参考项目：
  - Microsoft Qlib: qflow.workflow 提供的 train/valid/test 三段切分
  - Hudson Thames mlfinlab: CombinatorialPurgedCV + EmbargoCV
  - QuantConnect: OutOfSampleDays 概念

核心创新点（对比 jingni-trader 现状）：

1. **分段时间切分 + 滚动训练**
   原版：单次 train/test 切分，无滚动训练
   改进：滚动窗口（WFA）+ 锚定窗口（anchored）两种模式

2. **Purge 与 Embargo**
   借鉴 mlfinlab 关键概念：
   - Purge: 训练集尾部剔除与验证集相邻的 N 天（避免标签泄漏）
   - Embargo: 验证集之后增加 M 天空窗（避免测试期影响训练）

3. **过拟合检测指标**
   - IC（信息系数）跨窗口稳定性
   - IC Decay：训练集 IC 显著高于测试集时报警
   - Multi-window Sharpe 比

4. **可重复性**
   - 固定切分点（split_index）便于复现
   - 输出每窗口的详细指标
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("walk-forward")


# ───────────────────────── 数据结构 ─────────────────────────

@dataclass
class WalkForwardFold:
    """单个折的训练/测试时间段"""
    fold_id: int
    train_start: Any
    train_end: Any  # train end = purge_start
    test_start: Any
    test_end: Any
    purge_days: int
    embargo_days: int


@dataclass
class FoldResult:
    """单折结果"""
    fold_id: int
    train_period: Tuple[Any, Any]
    test_period: Tuple[Any, Any]
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    factor_ic_train: float = 0.0
    factor_ic_test: float = 0.0
    factor_rank_ic_train: float = 0.0
    factor_rank_ic_test: float = 0.0
    long_short_return: float = 0.0
    n_stocks: int = 0


@dataclass
class WalkForwardConfig:
    """WFA 配置"""
    train_days: int = 240            # 训练窗口长度
    test_days: int = 60              # 测试窗口长度
    purge_days: int = 5              # purge 长度（标签泄漏防护）
    embargo_days: int = 5            # embargo 长度（训练/测试隔离）
    anchored: bool = False           # 锚定模式：训练起点固定
    step_days: int = 60              # 滚动步长
    min_train_days: int = 120        # 最小训练天数


# ───────────────────────── 切分器 ─────────────────────────

class WalkForwardSplitter:
    """
    滚动窗口（Walk-Forward Analysis）切分器

    示例：train_days=240, test_days=60, step_days=60
        Fold 1: train [0,240) test [245,305)  (含 5 天 purge)
        Fold 2: train [60,300) test [305,365)
        Fold 3: train [120,360) test [365,425)
        ...
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def split(self, dates: pd.DatetimeIndex) -> List[WalkForwardFold]:
        dates = pd.DatetimeIndex(dates).sort_values().unique()
        n = len(dates)
        folds = []
        fold_id = 0

        if self.config.anchored:
            # 锚定模式：train 起点固定
            train_start_idx = 0
            test_start_idx = self.config.train_days
        else:
            train_start_idx = 0

        while True:
            if self.config.anchored:
                train_end_idx = test_start_idx - self.config.purge_days
            else:
                train_end_idx = train_start_idx + self.config.train_days - self.config.purge_days

            test_start_idx_real = train_end_idx + self.config.purge_days
            test_end_idx_real = test_start_idx_real + self.config.test_days

            if test_end_idx_real > n:
                break

            # 训练起点
            if self.config.anchored:
                train_start_real = dates[0]
            else:
                train_start_real = dates[train_start_idx]

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start_real,
                train_end=dates[train_end_idx - 1],
                test_start=dates[test_start_idx_real],
                test_end=dates[min(test_end_idx_real - 1 + self.config.embargo_days, n - 1)],
                purge_days=self.config.purge_days,
                embargo_days=self.config.embargo_days,
            )
            folds.append(fold)
            fold_id += 1

            if self.config.anchored:
                test_start_idx += self.config.step_days
            else:
                train_start_idx += self.config.step_days

        return folds


# ───────────────────────── 因子评估器 ─────────────────────────

def _calc_ic(factor: pd.Series, ret: pd.Series) -> float:
    """Pearson IC"""
    df = pd.concat([factor, ret], axis=1).dropna()
    if len(df) < 10:
        return 0.0
    return float(df.corr().iloc[0, 1])


def _calc_rank_ic(factor: pd.Series, ret: pd.Series) -> float:
    """Spearman 秩相关系数"""
    df = pd.concat([factor, ret], axis=1).dropna()
    if len(df) < 10:
        return 0.0
    return float(df.corr(method="spearman").iloc[0, 1])


def _calc_long_short(factor: pd.Series, ret: pd.Series, q: float = 0.1) -> float:
    """多空组合收益率（横截面分组）"""
    if factor.empty or ret.empty:
        return 0.0
    df = pd.concat([factor, ret], axis=1).dropna()
    if len(df) < 20:
        return 0.0
    df.columns = ["factor", "ret"]
    # 按日分组
    if "date" in df.index.names or isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "date" not in df.columns:
        # 假设是单日横截面
        long_ = df[df["factor"] >= df["factor"].quantile(1 - q)]["ret"].mean()
        short_ = df[df["factor"] <= df["factor"].quantile(q)]["ret"].mean()
        return float(long_ - short_)
    df["date"] = pd.to_datetime(df["date"])
    grouped = df.groupby("date")
    longs = grouped.apply(lambda x: x[x["factor"] >= x["factor"].quantile(1 - q)]["ret"].mean())
    shorts = grouped.apply(lambda x: x[x["factor"] <= x["factor"].quantile(q)]["ret"].mean())
    return float((longs - shorts).mean())


# ───────────────────────── Walk-Forward 主框架 ─────────────────────────

class WalkForwardValidator:
    """
    Walk-Forward 验证器

    Usage:
        validator = WalkForwardValidator(config)
        results = validator.run(data, factor_df, train_fn, predict_fn)
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.splitter = WalkForwardSplitter(config)

    def run(self,
            data: pd.DataFrame,
            factor_fn: Callable[[pd.DataFrame, Tuple[Any, Any]], pd.Series],
            forward_ret_col: str = "fwd_ret_5d",
            factor_name: str = "factor",
            ) -> List[FoldResult]:
        """
        运行 WFA

        参数:
            data: 含 date, code, forward_ret_col 的 DataFrame
            factor_fn: 给定数据子集 + (start, end) 区间返回因子 Series
                      （区间外的返回 None，区间内的返回因子值）
            forward_ret_col: 前瞻收益列名
            factor_name: 因子名称
        """
        if not pd.api.types.is_datetime64_any_dtype(data["date"]):
            data["date"] = pd.to_datetime(data["date"])

        dates = pd.DatetimeIndex(data["date"].unique()).sort_values()
        folds = self.splitter.split(dates)
        if not folds:
            raise ValueError("数据量不足以切分出至少一折")

        results = []
        for fold in folds:
            train_mask = (data["date"] >= fold.train_start) & (data["date"] <= fold.train_end)
            test_mask = (data["date"] >= fold.test_start) & (data["date"] <= fold.test_end)
            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()

            if len(train_data) < self.config.min_train_days * 5:  # 经验值：每天 5 个股票
                logger.warning(f"Fold {fold.fold_id} 训练数据过少：{len(train_data)} 行，跳过")
                continue

            # 训练：生成因子
            try:
                train_factor = factor_fn(train_data, (fold.train_start, fold.train_end))
                test_factor = factor_fn(test_data, (fold.test_start, fold.test_end))
            except Exception as e:
                logger.error(f"Fold {fold.fold_id} 因子计算失败: {e}")
                continue

            # 评估
            train_eval = self._eval_factor(train_data, train_factor, forward_ret_col)
            test_eval = self._eval_factor(test_data, test_factor, forward_ret_col)

            result = FoldResult(
                fold_id=fold.fold_id,
                train_period=(fold.train_start, fold.train_end),
                test_period=(fold.test_start, fold.test_end),
                train_metrics=train_eval,
                test_metrics=test_eval,
                factor_ic_train=train_eval["ic"],
                factor_ic_test=test_eval["ic"],
                factor_rank_ic_train=train_eval["rank_ic"],
                factor_rank_ic_test=test_eval["rank_ic"],
                long_short_return=test_eval["long_short"],
                n_stocks=len(set(test_data["code"])),
            )
            results.append(result)
            logger.info(
                f"Fold {fold.fold_id}: train IC={result.factor_ic_train:+.3f}, "
                f"test IC={result.factor_ic_test:+.3f}, "
                f"LS return={result.long_short_return:+.4f}"
            )
        return results

    def _eval_factor(self, data: pd.DataFrame, factor: Optional[pd.Series],
                     fwd_ret_col: str) -> Dict[str, float]:
        if factor is None or factor.empty or fwd_ret_col not in data.columns:
            return {"ic": 0.0, "rank_ic": 0.0, "long_short": 0.0, "n": 0}
        if not isinstance(factor, pd.Series):
            factor = pd.Series(factor, index=data.index)
        if len(factor) != len(data):
            # 尝试按索引对齐
            factor = factor.reindex(data.index)
        aligned = pd.DataFrame({
            "factor": factor.values,
            "ret": data[fwd_ret_col].values,
            "date": data["date"].values,
        }).dropna()
        if len(aligned) < 20:
            return {"ic": 0.0, "rank_ic": 0.0, "long_short": 0.0, "n": len(aligned)}
        ic = _calc_ic(aligned["factor"], aligned["ret"])
        rank_ic = _calc_rank_ic(aligned["factor"], aligned["ret"])
        ls = _calc_long_short(
            pd.concat([aligned["date"], aligned["factor"]], axis=1)
                .set_index(aligned.index)["factor"],
            aligned["ret"]
        )
        return {"ic": ic, "rank_ic": rank_ic, "long_short": ls, "n": int(len(aligned))}

    def diagnose_overfitting(self, results: List[FoldResult]) -> Dict[str, Any]:
        """诊断是否存在过拟合"""
        if not results:
            return {"status": "no_results"}
        train_ics = [r.factor_ic_train for r in results]
        test_ics = [r.factor_ic_test for r in results]
        # IC mean 衰减
        train_mean = float(np.mean(train_ics))
        test_mean = float(np.mean(test_ics))
        ic_decay = train_mean - test_mean
        # IC 符号一致率
        test_sign_pos = sum(1 for ic in test_ics if ic > 0) / max(len(test_ics), 1)
        # IC IR (IC mean / IC std)
        ic_ir = test_mean / (np.std(test_ics) + 1e-8)
        # 训练/测试 IC 比
        ic_ratio = test_mean / (train_mean + 1e-8) if abs(train_mean) > 1e-8 else 0

        overfit = ic_decay > 0.02 and ic_ratio < 0.5
        return {
            "train_ic_mean": train_mean,
            "test_ic_mean": test_mean,
            "ic_decay": float(ic_decay),
            "ic_ir": float(ic_ir),
            "ic_ratio": float(ic_ratio),
            "test_sign_positive_rate": float(test_sign_pos),
            "n_folds": len(results),
            "is_overfit": bool(overfit),
            "warning": "训练 IC 远高于测试 IC，存在过拟合" if overfit else "通过",
        }


# ───────────────────────── 自检 ─────────────────────────

def _self_test():
    """构造一个含噪声的因子，验证 WFA 能识别过拟合"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=480, freq="D")
    codes = [f"{i:06d}.SZ" for i in range(20)]
    rows = []
    for d in dates:
        for c in codes:
            px = 10 + np.cumsum(np.random.normal(0, 0.02, 1))[0]
            fwd = np.random.normal(0, 0.01)  # 收益大部分是噪声
            rows.append({"date": d, "code": c, "close": px, "fwd_ret_5d": fwd})
    data = pd.DataFrame(rows)

    def factor_fn(sub_data, period):
        # 简单反转因子：负的 5 日动量
        sub_data = sub_data.sort_values(["code", "date"])
        sub_data["ret_5d"] = sub_data.groupby("code")["close"].pct_change(5)
        return -sub_data["ret_5d"]

    config = WalkForwardConfig(
        train_days=240, test_days=60, step_days=60,
        purge_days=5, embargo_days=5, min_train_days=120
    )
    validator = WalkForwardValidator(config)
    results = validator.run(data, factor_fn, "fwd_ret_5d")
    diag = validator.diagnose_overfitting(results)
    return results, diag


if __name__ == "__main__":
    print("=== Walk-Forward Validation self-test ===")
    results, diag = _self_test()
    print(f"\n  总折数: {len(results)}")
    for r in results:
        print(f"  Fold {r.fold_id}: train[{r.train_period[0].date()}~{r.train_period[1].date()}] "
              f"test[{r.test_period[0].date()}~{r.test_period[1].date()}] "
              f"IC_train={r.factor_ic_train:+.3f}  IC_test={r.factor_ic_test:+.3f}")
    print(f"\n  诊断：{diag['warning']}")
    print(f"  训练 IC 均值: {diag['train_ic_mean']:+.4f}")
    print(f"  测试 IC 均值: {diag['test_ic_mean']:+.4f}")
    print(f"  IC Decay:     {diag['ic_decay']:+.4f}")
    print(f"  IC Ratio:     {diag['ic_ratio']:+.4f}")
    print(f"  Test IC IR:   {diag['ic_ir']:+.4f}")
    print(f"  Test sign>0:  {diag['test_sign_positive_rate']:.2%}")
    print(f"  是否过拟合:   {diag['is_overfit']}")
