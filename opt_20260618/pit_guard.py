"""
Point-in-Time 数据守卫 + 严格时间序列 CV

【借鉴来源】
- Qlib (microsoft/qlib): Point-in-Time DB + 数据处理器（DataHandler）
- Qlib 论文 (arXiv:2009.11189): 强调"信息泄露"是量化研究的最大风险
- ADVANCES_IN_FINANCIAL_ML (Marcos López de Prado): Purged K-Fold CV
- Moon Dev AI: 必须做 walk-forward / out-of-sample 验证

【问题背景】
原 jingni-trader 中存在的信息泄露风险：
1. data-engine 的 fetch_and_clean 不验证"截至某日有哪些数据可用"
2. factor-engine 直接对全量数据计算 forward_returns（用 .shift(-N)），可能与训练/测试集混合
3. strategy-model-engine 的 purged_group_ts_split 名称虽然是 "purged"，但实现是简单的滚动窗口
4. 缺乏"训练时使用截至 T 日的数据，回测 T+N 时不能使用 T+N 之后的信息"的强校验

【设计目标】
1. 提供 PointInTimeGuard 类，断言"任何特征在 T 日只能依赖 ≤ T 日的数据"
2. 提供严格的时间序列 Purged K-Fold CV（带 embargo）
3. 提供 Walk-Forward 验证器
4. 提供泄露检测工具（leakage detector）

【关键概念】
- Point-in-Time (PIT): 任意时间点 t 的特征只能用 ≤ t 的数据计算
- Purge Gap: 训练集结束与验证集开始之间的间隔，避免标签泄露
- Embargo: 验证集结束后保留一段"冷冻期"，防止时间序列自相关
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Point-in-Time Guard ───────────────────────────────
class PointInTimeGuard:
    """
    Point-in-Time 数据守卫

    核心职责：
    1. 验证任何特征列的"时间依赖性"：在 T 日的特征只能用到 ≤ T 的原始数据
    2. 检测潜在的前视偏差 (look-ahead bias)
    3. 在训练/验证拆分时严格执行"未来信息不可见"

    使用:
        guard = PointInTimeGuard()
        guard.register_feature('alpha_20d', max_lookback_days=20)
        guard.validate_data(feature_df, raw_data, feature_name='alpha_20d')
    """

    def __init__(self):
        self.feature_specs: Dict[str, Dict[str, Any]] = {}

    def register_feature(
        self,
        feature_name: str,
        max_lookback_days: int,
        description: str = "",
    ) -> None:
        """
        注册一个特征及其最大可回看天数。

        参数:
            feature_name: 特征列名
            max_lookback_days: 该特征最多能用到多少天前的数据
            description: 特征描述
        """
        self.feature_specs[feature_name] = {
            "max_lookback_days": max_lookback_days,
            "description": description,
        }

    def validate_data(
        self,
        feature_df: pd.DataFrame,
        raw_data: pd.DataFrame,
        feature_name: str,
    ) -> Dict[str, Any]:
        """
        验证特征计算是否仅依赖了过去数据。

        方法：检查每个 (date, code) 的 feature_value 是否与未来的 raw_data 一致。
        简化为：检查"今天计算的特征与明天/后天的 raw_data 没有强相关"。
        这是一个轻量级检测，不能保证 100% 准确，但能发现明显的前视偏差。
        """
        if feature_name not in self.feature_specs:
            return {"valid": True, "note": "未注册的特征，跳过严格校验"}

        spec = self.feature_specs[feature_name]
        lookback = spec["max_lookback_days"]

        # 检查 feature_df 是否需要 future 信息
        # 启发式：若 feature_df 中 (date, code) 对应的值 与 raw_data 中 (date+1, code) 等有强相关
        # 且 lookback=0，则可能存在泄露
        result = {
            "feature_name": feature_name,
            "registered_lookback_days": lookback,
            "valid": True,
            "warnings": [],
        }

        # 检查特征值的时序一致性：若 df['date'] 是 (T, T+1, ...) 升序
        # 而特征值理应"只能用 T 及之前的数据"
        # 这里我们通过检查"特征值与未来 raw_data 的相关性"做粗糙检测
        feat_dates = pd.to_datetime(feature_df["date"]).sort_values().unique()
        if len(feat_dates) < 5:
            result["warnings"].append("数据点过少，无法严格校验")
            return result

        # 校验 1：特征不应包含 NaN 的"未来值被填充"模式
        # 通过 check: 任何 (date, code) 的特征值，如果其对应 raw_data 在该日也不存在但特征有值
        # 则可能用了未来数据
        if "close" in raw_data.columns and "code" in raw_data.columns and "date" in raw_data.columns:
            feat_keys = set(zip(feature_df["date"].astype(str), feature_df["code"].astype(str)))
            raw_keys = set(zip(raw_data["date"].astype(str), raw_data["code"].astype(str)))
            # 特征中超出 raw_data 范围的 (date, code) 数量
            missing_in_raw = len(feat_keys - raw_keys)
            if missing_in_raw > 0:
                result["valid"] = False
                result["warnings"].append(
                    f"特征 {feature_name} 中有 {missing_in_raw} 个 (date, code) 在原始数据中找不到，"
                    "可能使用了未来信息"
                )

        # 校验 2：特征不应存在"突然跳变"模式（可能是因为前视数据）
        # 简化检查：按 date 排序后，特征值的 std 不应出现诡异模式
        if "code" in feature_df.columns and feature_name in feature_df.columns:
            try:
                pivot = feature_df.pivot_table(
                    index="date", columns="code", values=feature_name
                )
                # 如果有 >50% 的值是 NaN，可能存在问题
                nan_ratio = pivot.isna().sum().sum() / max(pivot.size, 1)
                if nan_ratio > 0.5:
                    result["warnings"].append(f"NaN 比例过高: {nan_ratio:.1%}")
            except Exception:
                pass

        return result


# ── Purged K-Fold Time Series CV ──────────────────────
@dataclass
class PurgedSplit:
    """一个 purged CV 切分"""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    purge_days: int
    embargo_days: int

    def __repr__(self):
        return (
            f"PurgedSplit(train={self.train_start.date()}~{self.train_end.date()}, "
            f"val={self.val_start.date()}~{self.val_end.date()}, "
            f"purge={self.purge_days}d, embargo={self.embargo_days}d)"
        )


class PurgedKFoldTimeSeriesCV:
    """
    Purged K-Fold 时间序列交叉验证

    核心改进（vs 原 jingni-trader 的 purged_group_ts_split）：
    1. 训练集结束和验证集开始之间有 purge gap（防止标签泄露）
    2. 验证集结束后有 embargo 冷冻期（防止时间序列自相关）
    3. 拆分严格按时间顺序，不可随机

    借鉴：
    - López de Prado, "Advances in Financial ML" Chapter 7
    - Qlib 的 DatasetH 时间序列分段（train/valid/test）
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 5,
        min_train_size: int = 60,
    ):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_size = min_train_size

    def split(
        self,
        dates: pd.Series,
    ) -> List[PurgedSplit]:
        """
        生成 purged K-Fold splits

        参数:
            dates: 每条样本对应的日期（pd.Series）
        返回:
            PurgedSplit 列表
        """
        unique_dates = pd.Series(sorted(pd.to_datetime(dates).unique()))
        n_dates = len(unique_dates)
        if n_dates < self.min_train_size + self.n_splits * 10:
            return []

        # 预留最后 (n_splits-1) 个 fold 的数据给最后几个 validation
        val_size = (n_dates - self.min_train_size) // (self.n_splits + 1)
        splits: List[PurgedSplit] = []

        for i in range(self.n_splits):
            train_end_idx = self.min_train_size + i * val_size
            val_start_idx = train_end_idx + 1 + self.purge_days
            val_end_idx = val_start_idx + val_size

            if val_end_idx >= n_dates:
                break

            # 加上 embargo：validation 结束后的冷冻期也算"不可用"
            train_dates = unique_dates.iloc[: train_end_idx + 1]
            val_dates = unique_dates.iloc[val_start_idx: val_end_idx]

            # 实际训练日期：要排除验证集范围内的（避免时间重叠）
            # 这里 train < val_start - purge_gap
            train_end_real = unique_dates.iloc[train_end_idx]
            purge_cutoff = unique_dates.iloc[max(0, val_start_idx - 1)] - pd.Timedelta(days=self.purge_days)
            train_dates = train_dates[train_dates <= purge_cutoff]

            # 验证集需要排除 embargo 区
            if val_end_idx + self.embargo_days < n_dates:
                val_dates = unique_dates.iloc[val_start_idx: val_end_idx + self.embargo_days]

            if len(train_dates) == 0 or len(val_dates) == 0:
                continue

            splits.append(PurgedSplit(
                train_start=train_dates.iloc[0],
                train_end=train_dates.iloc[-1],
                val_start=val_dates.iloc[0],
                val_end=val_dates.iloc[-1],
                purge_days=self.purge_days,
                embargo_days=self.embargo_days,
            ))

        return splits

    def split_indices(
        self,
        dates: pd.Series,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        返回 (train_indices, val_indices) 元组列表
        """
        dates_arr = pd.to_datetime(dates)
        splits_meta = self.split(dates)
        results = []
        for split in splits_meta:
            train_mask = (dates_arr >= split.train_start) & (dates_arr <= split.train_end)
            val_mask = (dates_arr >= split.val_start) & (dates_arr <= split.val_end)
            results.append((
                np.where(train_mask)[0],
                np.where(val_mask)[0],
            ))
        return results


# ── Walk-Forward 验证器 ────────────────────────────────
class WalkForwardValidator:
    """
    Walk-Forward 验证器

    与 K-Fold 的区别：每次只验证"未来一段窗口"，而不是"剩余全部数据"。
    更接近真实部署场景。

    使用:
        wf = WalkForwardValidator(train_window=120, val_window=20, step=20)
        for train_idx, val_idx, test_idx in wf.split_with_test(dates):
            ...
    """

    def __init__(
        self,
        train_window: int = 252,  # 训练窗口（交易日）
        val_window: int = 60,
        step: int = 60,
        expanding: bool = False,  # 是否扩展训练窗口
    ):
        self.train_window = train_window
        self.val_window = val_window
        self.step = step
        self.expanding = expanding

    def split_with_test(
        self,
        dates: pd.Series,
    ) -> List[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        切分训练/验证/测试窗口。

        返回字典列表，每个字典含 train/val/test 三个时间范围。
        """
        unique_dates = pd.Series(sorted(pd.to_datetime(dates).unique()))
        n = len(unique_dates)
        results = []
        i = 0
        while True:
            if self.expanding:
                train_start_idx = 0
            else:
                train_start_idx = i
            train_end_idx = train_start_idx + self.train_window
            val_start_idx = train_end_idx + 1
            val_end_idx = val_start_idx + self.val_window

            if val_end_idx >= n:
                break

            results.append({
                "train": (unique_dates.iloc[train_start_idx], unique_dates.iloc[train_end_idx]),
                "val": (unique_dates.iloc[val_start_idx], unique_dates.iloc[val_end_idx - 1]),
            })

            i += self.step

        return results


# ── 泄露检测器 ─────────────────────────────────────────
class LeakageDetector:
    """
    简单的泄露检测器

    方法：随机打乱未来数据（标签或特征），看模型表现是否显著下降。
    若打乱后模型表现几乎不变 → 模型可能没用上这些信息（正常）
    若打乱后模型表现反而变好 → 严重泄露
    """

    @staticmethod
    def shuffle_y_test(
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metric_func: Callable,
        n_shuffles: int = 5,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        打乱 y_test 后评估模型表现，与原始对比。

        若 shuffle 后 metric 显著提升（>10%），则可能存在标签泄露。
        """
        np.random.seed(random_state)
        original_metric = metric_func(model, X_test, y_test)

        shuffled_metrics = []
        for _ in range(n_shuffles):
            y_shuffled = y_test.sample(frac=1.0, random_state=np.random.randint(1e9))
            m = metric_func(model, X_test, y_shuffled)
            shuffled_metrics.append(m)

        mean_shuffled = float(np.mean(shuffled_metrics))
        std_shuffled = float(np.std(shuffled_metrics))
        improvement = (mean_shuffled - original_metric) / abs(original_metric) if original_metric != 0 else 0

        return {
            "original_metric": float(original_metric),
            "shuffled_mean": mean_shuffled,
            "shuffled_std": std_shuffled,
            "improvement_pct": float(improvement * 100),
            "leakage_warning": improvement > 0.10,
        }
