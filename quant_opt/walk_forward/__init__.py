"""
Walk-Forward Validation
=======================

A walk-forward framework inspired by **AKQuant**'s ``akquant.ml``
module (``akquant.akfamily.xyz/en/advanced/ml``) and by the canonical
"Advances in Financial Machine Learning" chapter 7 (Lopez de Prado).

The framework is **not** a generic CV splitter.  It models the actual
ML-for-quant workflow:

```
        ┌──────────────────┐
        │  fit on T_train  │──── predict on T_val ────┐
        └──────────────────┘                          │
                                                     ▼
                          train ─► val ─► test  (one window)
                                                     │
        ┌──────────────────┐                          │
        │  fit on T_train  │──── predict on T_val ────┘
        └──────────────────┘
                ...
```

Design
------

* **Signal vs. action separation** (AKQuant) — the model only emits a
  *signal* (a real-valued score).  Mapping ``signal → action`` is done
  in the backtest layer, not in the model.  This avoids leaking
  threshold-fitted decision rules into the training set.
* **Purged gap** between train and validation to prevent look-ahead
  bias when the label is forward-looking (e.g. 5-day forward return).
* **Embargo** after the validation window to mitigate leakage from
  autocorrelation in features.
* **Rolling** (fixed-size training window) and **expanding** (growing
  training window) modes.
* **Model cloning** so the user can plug any scikit-learn-compatible
  estimator and the framework will copy it before each ``fit``.

The output of :py:meth:`WalkForward.run` is a ``WalkForwardResult``
containing the fitted model per window, the predicted signals on the
held-out slice, and a summary metric (default: information coefficient,
IC).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone


# ── Configuration objects ────────────────────────────────────────────


@dataclass
class WalkForwardConfig:
    """All knobs in one place.

    Parameters
    ----------
    train_period : int
        Number of trading days used for training.
    val_period : int
        Number of trading days used for validation (and out-of-sample
        prediction).
    step : int
        Step size (in days) to roll the window forward.
    expanding : bool
        If ``True`` the training window grows from ``min_train_period``
        to the start of each validation window.  ``min_train_period``
        must be set in that case.
    min_train_period : int
        Initial size of the training window when ``expanding=True``.
    purge_gap : int
        Days dropped between train and validation to prevent
        look-ahead bias.
    embargo : int
        Days dropped after each validation window before the next
        window starts.
    label_col, feature_cols, date_col, code_col : str
        Column names in the input frame.
    """

    train_period: int = 252 * 2
    val_period: int = 63
    step: int = 63
    expanding: bool = False
    min_train_period: int = 252
    purge_gap: int = 5
    embargo: int = 5
    label_col: str = "label"
    feature_cols: Sequence[str] = field(default_factory=tuple)
    date_col: str = "date"
    code_col: str = "code"

    def __post_init__(self) -> None:
        if self.train_period <= 0:
            raise ValueError("train_period must be > 0")
        if self.val_period <= 0:
            raise ValueError("val_period must be > 0")
        if self.step <= 0:
            raise ValueError("step must be > 0")
        if self.expanding and self.min_train_period <= 0:
            raise ValueError("min_train_period must be > 0 when expanding=True")


# ── Result objects ───────────────────────────────────────────────────


@dataclass
class WindowResult:
    """Per-window artifacts."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive
    val_start: pd.Timestamp
    val_end: pd.Timestamp  # inclusive
    model: Any
    metrics: Dict[str, float]
    predictions: pd.DataFrame  # columns = code, date, signal, label


@dataclass
class WalkForwardResult:
    """Aggregated output of :py:meth:`WalkForward.run`."""

    config: WalkForwardConfig
    windows: List[WindowResult]
    summary: Dict[str, float]
    signals: pd.DataFrame  # all windows concatenated, columns: code, date, signal, label, window_id

    def aggregate_ic(self) -> Dict[str, float]:
        """Return mean / std / IR of the per-window IC series."""
        ics = [w.metrics.get("ic", np.nan) for w in self.windows]
        ics = pd.Series(ics).dropna()
        if ics.empty:
            return {"mean_ic": 0.0, "std_ic": 0.0, "ic_ir": 0.0, "n_windows": 0}
        return {
            "mean_ic": float(ics.mean()),
            "std_ic": float(ics.std()),
            "ic_ir": float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0,
            "n_windows": int(len(ics)),
        }


# ── Main class ───────────────────────────────────────────────────────


class WalkForward:
    """Walk-forward validator for cross-sectional / panel ML models.

    Example
    -------

    >>> from sklearn.linear_model import Ridge
    >>> cfg = WalkForwardConfig(
    ...     train_period=504, val_period=63, step=63,
    ...     feature_cols=["mom_20", "turnover_5d", "vol_20d"],
    ...     label_col="forward_5d",
    ... )
    >>> wf = WalkForward(model_factory=lambda: Ridge(alpha=1.0))
    >>> result = wf.run(df, cfg)
    >>> result.summary
    """

    def __init__(self, model_factory: Callable[[], Any]) -> None:
        self.model_factory = model_factory

    # ── Window generation ───────────────────────────────────────────

    def generate_windows(
        self, dates: Iterable[pd.Timestamp], cfg: WalkForwardConfig
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Return ``(train_start, train_end, val_start, val_end)`` tuples."""
        unique_dates = pd.Series(pd.unique(pd.Series(list(dates))))
        unique_dates = unique_dates.sort_values().reset_index(drop=True)
        n = len(unique_dates)
        if n < cfg.train_period + cfg.val_period + cfg.purge_gap:
            return []

        windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        # start position of the *training* window
        if cfg.expanding:
            starts = list(range(cfg.min_train_period, n - cfg.val_period + 1, cfg.step))
            train_starts = [0] * len(starts)
            train_ends = [s - cfg.purge_gap - 1 for s in starts]
        else:
            starts = list(range(cfg.train_period, n - cfg.val_period + 1, cfg.step))
            train_starts = [s - cfg.train_period for s in starts]
            train_ends = [s - cfg.purge_gap - 1 for s in starts]

        val_starts = [s for s in starts]
        val_ends = [min(s + cfg.val_period - 1, n - 1) for s in starts]

        # embargo: skip windows that overlap the embargo of the previous
        # validation window.  The check is on the *val* start index, not
        # the train start index (train can start well before val without
        # conflict — only the val window overlap matters).
        prev_val_end: Optional[int] = None
        for ts, te, vs, ve in zip(train_starts, train_ends, val_starts, val_ends):
            if prev_val_end is not None and vs <= prev_val_end + cfg.embargo:
                continue
            windows.append(
                (
                    unique_dates.iloc[ts],
                    unique_dates.iloc[te],
                    unique_dates.iloc[vs],
                    unique_dates.iloc[ve],
                )
            )
            prev_val_end = ve

        return windows

    # ── Entry point ─────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        cfg: WalkForwardConfig,
        progress: bool = False,
    ) -> WalkForwardResult:
        """Execute the walk-forward validation.

        Parameters
        ----------
        df : pd.DataFrame
            Panel frame containing ``code``, ``date``, the configured
            ``label_col`` and every column in ``cfg.feature_cols``.
        cfg : WalkForwardConfig
            Window configuration.
        progress : bool
            If ``True``, print a one-line progress message per window.
        """
        if df.empty:
            raise ValueError("input DataFrame is empty")
        for col in [cfg.code_col, cfg.date_col, cfg.label_col, *cfg.feature_cols]:
            if col not in df.columns:
                raise ValueError(f"column {col!r} missing in input frame")

        # ensure date is sortable
        df = df.copy()
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
        df = df.sort_values([cfg.date_col, cfg.code_col]).reset_index(drop=True)

        windows = self.generate_windows(df[cfg.date_col], cfg)
        if not windows:
            return WalkForwardResult(config=cfg, windows=[], summary={"n_windows": 0},
                                      signals=pd.DataFrame())

        results: List[WindowResult] = []
        signals: List[pd.DataFrame] = []

        for wid, (tr_s, tr_e, va_s, va_e) in enumerate(windows):
            train_mask = (df[cfg.date_col] >= tr_s) & (df[cfg.date_col] <= tr_e)
            val_mask = (df[cfg.date_col] >= va_s) & (df[cfg.date_col] <= va_e)

            train_df = df.loc[train_mask]
            val_df = df.loc[val_mask]

            X_train = train_df[list(cfg.feature_cols)].values
            y_train = train_df[cfg.label_col].values
            X_val = val_df[list(cfg.feature_cols)].values
            y_val = val_df[cfg.label_col].values

            model = clone(self.model_factory())
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            # metrics
            metrics = self._compute_metrics(y_val, preds, val_df, cfg)
            pred_df = val_df[[cfg.code_col, cfg.date_col]].copy()
            pred_df["signal"] = preds
            pred_df["label"] = y_val
            pred_df["window_id"] = wid
            signals.append(pred_df)

            results.append(
                WindowResult(
                    window_id=wid,
                    train_start=tr_s,
                    train_end=tr_e,
                    val_start=va_s,
                    val_end=va_e,
                    model=model,
                    metrics=metrics,
                    predictions=pred_df,
                )
            )
            if progress:
                print(f"[WF] window {wid}: train={tr_s.date()}..{tr_e.date()} "
                      f"val={va_s.date()}..{va_e.date()} ic={metrics.get('ic', float('nan')):.4f}")

        summary = self._summary_metrics(results)
        return WalkForwardResult(
            config=cfg,
            windows=results,
            summary=summary,
            signals=pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(),
        )

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        val_df: pd.DataFrame,
        cfg: WalkForwardConfig,
    ) -> Dict[str, float]:
        from scipy.stats import spearmanr, pearsonr

        if len(y_true) < 2:
            return {"ic": 0.0, "rank_ic": 0.0, "mse": float("nan")}

        ic, _ = pearsonr(y_pred, y_true)
        rank_ic, _ = spearmanr(y_pred, y_true)
        mse = float(np.mean((y_pred - y_true) ** 2))

        # cross-sectional daily IC
        daily_ics = []
        if "__preds__" not in val_df.columns:
            val_df = val_df.copy()
            val_df["__preds__"] = y_pred
        for dt, grp in val_df.groupby(cfg.date_col):
            if len(grp) < 5:
                continue
            d_ic, _ = spearmanr(grp["__preds__"], grp[cfg.label_col])
            if not np.isnan(d_ic):
                daily_ics.append(d_ic)
        mean_daily_rank_ic = float(np.mean(daily_ics)) if daily_ics else 0.0

        return {
            "ic": float(ic) if not np.isnan(ic) else 0.0,
            "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
            "mse": mse,
            "mean_daily_rank_ic": mean_daily_rank_ic,
            "n_samples": int(len(y_true)),
        }

    @staticmethod
    def _summary_metrics(results: Sequence[WindowResult]) -> Dict[str, float]:
        ics = [r.metrics.get("ic", 0.0) for r in results]
        rank_ics = [r.metrics.get("rank_ic", 0.0) for r in results]
        mses = [r.metrics.get("mse", 0.0) for r in results]
        return {
            "n_windows": len(results),
            "mean_ic": float(np.mean(ics)) if ics else 0.0,
            "std_ic": float(np.std(ics)) if ics else 0.0,
            "ic_ir": float(np.mean(ics) / (np.std(ics) + 1e-12)) if ics else 0.0,
            "mean_rank_ic": float(np.mean(rank_ics)) if rank_ics else 0.0,
            "mean_mse": float(np.mean(mses)) if mses else 0.0,
        }
