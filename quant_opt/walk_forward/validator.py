"""
Walk-Forward Validation Framework (AKQuant-inspired)
====================================================

借鉴 AKQuant (akfamily/akquant) 高级特性中的 Walk-forward Validation 设计:

1. **Signal vs Action 分离** (Core Design Philosophy #1)
   - 模型只输出连续信号, 不直接产生 buy/sell 指令
   - 信号通过 threshold 映射到 action, 避免模型过拟合到离散决策
2. **Rolling Window**
   - ``train_window`` + ``test_window`` 滚动划分
   - 每个 fold 独立 fit + predict, 严防 look-ahead
3. **Pipeline 防泄露** (Design Philosophy #5)
   - 特征计算放在 prepare_features, 训练前对齐索引
4. **Model.clone() 接口** (Design Philosophy #2)
   - 自定义模型需实现 clone, 避免 deepcopy 副作用

References
----------
- AKQuant ML Guide: https://akquant.akfamily.xyz/en/advanced/ml/
- QuantConnect Lean: walk-forward optimization in Engine/Optimizer
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. Walk-Forward Splitter
# --------------------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """Walk-Forward 验证配置."""

    train_window: int = 252              # 训练窗口大小 (交易日)
    test_window: int = 63                # 测试窗口大小 (约一个季度)
    rolling_step: Optional[int] = None   # 滚动步长.  None = test_window
    min_train_size: int = 120            # 最小训练样本数
    expanding: bool = False              # True 时训练窗口累积扩展 (expanding window)

    def __post_init__(self) -> None:
        if self.rolling_step is None:
            self.rolling_step = self.test_window
        if self.train_window < self.min_train_size:
            raise ValueError("train_window must be >= min_train_size")


@dataclass
class WalkForwardFold:
    """单次 fold 的索引信息."""

    fold_id: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    train_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    test_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def walk_forward_splits(
    n_samples: int,
    cfg: WalkForwardConfig,
    dates: Optional[pd.Series] = None,
) -> List[WalkForwardFold]:
    """
    生成 walk-forward folds.

    Parameters
    ----------
    n_samples : int
        总样本数.
    cfg : WalkForwardConfig
        滚动配置.
    dates : pd.Series, optional
        长度为 n_samples 的日期序列. 提供后 fold 元数据中保存真实日期.

    Returns
    -------
    list[WalkForwardFold]
    """
    if n_samples < cfg.train_window + cfg.test_window:
        return []

    folds: List[WalkForwardFold] = []
    fold_id = 0
    train_start = 0
    while True:
        if cfg.expanding:
            train_end = train_start + cfg.train_window
        else:
            train_end = train_start + cfg.train_window

        test_start = train_end
        test_end = test_start + cfg.test_window
        if test_end > n_samples:
            break

        if dates is not None and len(dates) == n_samples:
            ts = dates.iloc[train_start]
            te = dates.iloc[train_end - 1]
            vs = dates.iloc[test_start]
            ve = dates.iloc[test_end - 1]
        else:
            ts = train_start
            te = train_end - 1
            vs = test_start
            ve = test_end - 1

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=ts,
                train_end=te,
                test_start=vs,
                test_end=ve,
                train_index=np.arange(train_start, train_end),
                test_index=np.arange(test_start, test_end),
            )
        )
        fold_id += 1
        # 下一次训练的起点 = 当前 test 末尾, 严防 test -> 下一次 train 的泄露
        # 若 rolling_step > test_window, 则额外跳过 embargo 区间
        train_start = test_end - (cfg.rolling_step - cfg.test_window if cfg.rolling_step > cfg.test_window else 0)
        # 简化: 直接以 test_end 为下一轮起点
        train_start = test_end

    return folds


# --------------------------------------------------------------------------------------
# 2. 通用模型包装 (Signal vs Action 分离)
# --------------------------------------------------------------------------------------

class SignalModel:
    """
    抽象基类: 只产出连续信号, 不直接产生 buy/sell.
    借鉴 AKQuant "Signal vs Action Separation" 原则.

    自定义模型需实现:
        - fit(X, y) -> self
        - predict(X) -> np.ndarray (连续值)
        - clone() -> SignalModel   (避免 deepcopy 副作用)
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SignalModel":  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def clone(self) -> "SignalModel":  # pragma: no cover
        return copy.deepcopy(self)


class MeanReversionSignal(SignalModel):
    """
    一个最小可用的样例模型: 用 z-score 做均值回复信号.
    用于验证 walk-forward 流程的端到端正确性.
    """

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self._mu: Optional[float] = None
        self._sigma: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanReversionSignal":
        # 简单实现: 把目标序列 (y) 的均值/方差作为 baseline
        self._mu = float(y.mean())
        self._sigma = float(y.std()) + 1e-12
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 期望 X 至少有一列 'close' ;  用 z-score 作为 signal
        if "close" not in X.columns:
            raise ValueError("MeanReversionSignal requires 'close' column")
        close = X["close"].astype(float).values
        rolling_mu = pd.Series(close).rolling(self.lookback, min_periods=1).mean().values
        rolling_sd = pd.Series(close).rolling(self.lookback, min_periods=1).std().fillna(0).values + 1e-12
        z = (close - rolling_mu) / rolling_sd
        return -z  # 负 z-score -> 反转 -> 多头信号

    def clone(self) -> "MeanReversionSignal":
        new = MeanReversionSignal(self.lookback)
        new._mu = self._mu
        new._sigma = self._sigma
        return new


# --------------------------------------------------------------------------------------
# 3. 验证执行器
# --------------------------------------------------------------------------------------

def run_walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], SignalModel],
    cfg: WalkForwardConfig,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    执行 walk-forward 验证.

    Parameters
    ----------
    X : pd.DataFrame
        特征. 必须与 y 同长度, index 对齐.
    y : pd.Series
        目标变量 (例如未来 1 日收益).
    model_factory : callable
        返回一个**未训练**的 SignalModel 实例.  每次 fold 重新调用, 严防状态泄露.
    cfg : WalkForwardConfig
    threshold : float
        将连续信号映射为 1/0/-1 的阈值. 借鉴 AKQuant Signal vs Action 分离.

    Returns
    -------
    dict:
        - folds: list[WalkForwardFold]
        - oos_signals: np.ndarray (concat of all OOS predictions)
        - oos_index: pd.Index (对应的行 index)
        - oos_y: np.ndarray (对应的 y)
        - actions: np.ndarray (1=long, 0=cash, -1=short)
        - hit_ratio: 命中率
        - mean_oos_signal: float
        - per_fold_metrics: list[dict]
    """
    n = len(X)
    folds = walk_forward_splits(
        n_samples=n,
        cfg=cfg,
        dates=X.index.to_series() if hasattr(X.index, "to_series") else None,
    )

    if not folds:
        return {
            "folds": [],
            "oos_signals": np.array([]),
            "oos_index": pd.Index([]),
            "oos_y": np.array([]),
            "actions": np.array([]),
            "hit_ratio": 0.0,
            "mean_oos_signal": 0.0,
            "per_fold_metrics": [],
        }

    Xv = X.reset_index(drop=True)
    yv = y.reset_index(drop=True).astype(float)
    original_index = X.index

    all_signals: List[float] = []
    all_y: List[float] = []
    all_idx: List[Any] = []
    all_actions: List[int] = []
    per_fold: List[Dict[str, float]] = []

    for fold in folds:
        # 1) 重新创建模型 (避免状态泄露)
        model = model_factory()

        # 2) fit
        tr_idx = fold.train_index
        te_idx = fold.test_index
        try:
            model.fit(Xv.iloc[tr_idx], yv.iloc[tr_idx])
        except Exception as e:  # pragma: no cover
            per_fold.append({"fold_id": fold.fold_id, "error": str(e)})
            continue

        # 3) predict OOS
        sig = model.predict(Xv.iloc[te_idx])
        oos_y = yv.iloc[te_idx].values

        # 4) 信号 -> 动作 (Signal vs Action 分离)
        act = np.zeros_like(sig, dtype=int)
        act[sig > threshold] = 1
        act[sig < -threshold] = -1

        # 5) 命中率 (动作方向 与 真实收益符号 一致)
        hit = float(((np.sign(oos_y) == np.sign(act)) & (act != 0)).mean())

        per_fold.append({
            "fold_id": fold.fold_id,
            "train_size": int(len(tr_idx)),
            "test_size": int(len(te_idx)),
            "train_start": str(fold.train_start),
            "train_end": str(fold.train_end),
            "test_start": str(fold.test_start),
            "test_end": str(fold.test_end),
            "mean_signal": float(np.mean(sig)),
            "hit_ratio": hit,
        })

        all_signals.extend(sig.tolist())
        all_y.extend(oos_y.tolist())
        all_idx.extend(original_index[te_idx].tolist())
        all_actions.extend(act.tolist())

    sig_arr = np.asarray(all_signals, dtype=float)
    y_arr = np.asarray(all_y, dtype=float)
    act_arr = np.asarray(all_actions, dtype=int)

    # 总命中率 (仅对有动作的样本)
    active = act_arr != 0
    overall_hit = float(((np.sign(y_arr) == np.sign(act_arr)) & active).sum() / max(active.sum(), 1))

    return {
        "folds": folds,
        "oos_signals": sig_arr,
        "oos_index": pd.Index(all_idx),
        "oos_y": y_arr,
        "actions": act_arr,
        "hit_ratio": overall_hit,
        "mean_oos_signal": float(sig_arr.mean()) if sig_arr.size else 0.0,
        "per_fold_metrics": per_fold,
    }
