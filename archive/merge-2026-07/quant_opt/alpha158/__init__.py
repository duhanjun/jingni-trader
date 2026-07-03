"""
Alpha158-style factor library
=============================

A curated set of well-known quant factors expressed in the
``quant_opt.factor_expression`` DSL.  The names and formulas are
borrowed from the public Alpha101 (Kakushadze 2016) / Alpha158 (Qlib)
releases, and from the AKQuant / VeighNa vnpy.alpha tutorials.

Why
---

``jingni-trader`` ships with ``compute_a_share_factors`` that hard-codes
~15 ad-hoc factors.  This module is a drop-in upgrade: it returns a
*list of factor expression strings* the user can pipe straight into
``FactorEngine.calc_many``.  It also exposes a registry so users can
register their own factor families with documentation metadata.

The library is **declarative** (no imperative pandas code in this
file) which means the same factors can later be re-evaluated on
Polars / DuckDB / cuDF by swapping the engine backend — the way Qlib
and AKQuant separate the operator semantics from the execution engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class FactorSpec:
    name: str
    expression: str
    family: str
    description: str
    direction: int  # +1 = higher is better, -1 = lower is better, 0 = unknown


# ── Curated library ─────────────────────────────────────────────────


# The library covers the most common A股 factors.  Expressions are
# written against the canonical column names Close / Open / High / Low
# / Volume / Amount / TurnoverRate / Industry, which the engine
# resolves case-insensitively.
ALPHA158_LIKE: List[FactorSpec] = [
    # ── Price/return momentum ─────────────────────────────────────
    FactorSpec("ret_1d", "$close / Ref($close, 1) - 1", "momentum",
               "1-day return", 1),
    FactorSpec("ret_5d", "$close / Ref($close, 5) - 1", "momentum",
               "5-day return", 1),
    FactorSpec("ret_10d", "$close / Ref($close, 10) - 1", "momentum",
               "10-day return", 1),
    FactorSpec("ret_20d", "$close / Ref($close, 20) - 1", "momentum",
               "20-day return (medium-term momentum)", 1),
    FactorSpec("ret_60d", "$close / Ref($close, 60) - 1", "momentum",
               "60-day return (long-term momentum)", 1),
    FactorSpec("reversal_5d", "-($close / Ref($close, 5) - 1)", "momentum",
               "5-day reversal (short-term mean reversion)", 1),

    # ── Moving-average based ─────────────────────────────────────
    FactorSpec("ma_bias_5", "$close / MA($close, 5) - 1", "ma",
               "Bias of close to 5-day MA", 0),
    FactorSpec("ma_bias_10", "$close / MA($close, 10) - 1", "ma",
               "Bias of close to 10-day MA", 0),
    FactorSpec("ma_bias_20", "$close / MA($close, 20) - 1", "ma",
               "Bias of close to 20-day MA", 0),
    FactorSpec("ma_cross_5_20", "MA($close, 5) - MA($close, 20)", "ma",
               "5/20-day MA spread (MACD-like)", 1),
    FactorSpec("ma_cross_5_10", "MA($close, 5) - MA($close, 10)", "ma",
               "5/10-day MA spread", 1),

    # ── Volatility ───────────────────────────────────────────────
    FactorSpec("std_5", "Std($close, 5)", "volatility",
               "5-day close std (raw)", -1),
    FactorSpec("std_20", "Std($close, 20)", "volatility",
               "20-day close std (raw)", -1),
    FactorSpec("std_60", "Std($close, 60)", "volatility",
               "60-day close std (raw)", -1),
    FactorSpec("range_5", "(MA($high, 5) - MA($low, 5)) / MA($close, 5)",
               "volatility", "5-day high-low range / close", -1),
    FactorSpec("range_20", "(MA($high, 20) - MA($low, 20)) / MA($close, 20)",
               "volatility", "20-day high-low range / close", -1),

    # ── Volume / liquidity ───────────────────────────────────────
    FactorSpec("volume_ma_5", "MA($volume, 5)", "volume",
               "5-day average volume", 0),
    FactorSpec("volume_ma_20", "MA($volume, 20)", "volume",
               "20-day average volume", 0),
    FactorSpec("volume_ratio_5_20", "MA($volume, 5) / MA($volume, 20)",
               "volume", "Volume 5d / 20d ratio", 1),
    FactorSpec("amount_ma_20", "MA($amount, 20)", "volume",
               "20-day average amount", 0),

    # ── Cross-sectional ranks ───────────────────────────────────
    FactorSpec("rank_close", "Rank($close)", "cross_section",
               "Cross-sectional rank of close (raw level)", 0),
    FactorSpec("rank_volume", "Rank($volume)", "cross_section",
               "Cross-sectional rank of volume", 0),
    FactorSpec("rank_amount", "Rank($amount)", "cross_section",
               "Cross-sectional rank of amount", 0),
    FactorSpec("rank_ret_20d", "Rank($close / Ref($close, 20) - 1)",
               "cross_section", "Cross-sectional rank of 20d return", 1),
    FactorSpec("rank_std_20", "Rank(Std($close, 20))", "cross_section",
               "Cross-sectional rank of 20d std", -1),

    # ── Alpha101-flavored composites ─────────────────────────────
    FactorSpec("alpha_001",
               "Rank($close - Ref($close, 1)) - Rank($close - Ref($close, 5))",
               "alpha101", "Momentum acceleration", 0),
    FactorSpec("alpha_004",
               "-1 * TsRank($volume, 5)",
               "alpha101", "Inverse 5-day volume rank (mean reversion on vol)", 1),
    FactorSpec("alpha_006",
               "-1 * (Rank(Sign(Delta($close, 1)) * Sign(Delta($volume, 1))))",
               "alpha101", "Cross product of price and volume change direction",
               0),
    FactorSpec("alpha_012",
               "Sign(Delta($volume, 1)) * -1 * Sign(Delta($close, 1))",
               "alpha101", "Volume/price sign reversal", 0),
    FactorSpec("alpha_021",
               "MA(($close - MA($close, 8)) / MA($close, 8), 5) - "
               "MA(($close - MA($close, 8)) / MA($close, 8), 3)",
               "alpha101", "Multi-horizon bias difference", 0),
    FactorSpec("alpha_023",
               "MA(($high - $low) / Ref($close, 1) - "
               "MA(($high - $low) / Ref($close, 1), 20), 20)",
               "alpha101", "Range breakout vs trailing 20d mean range", 0),
    FactorSpec("alpha_038",
               "-1 * Rank(MA($close, 10)) * Rank(MA($volume, 10))",
               "alpha101", "Joint rank of price and volume MA", 0),

    # ── A股本土化扩展 ───────────────────────────────────────────
    FactorSpec("turnover_ma_5", "MA($turnoverRate, 5)", "a_share",
               "5-day avg turnover rate (换手率)", 0),
    FactorSpec("turnover_ma_20", "MA($turnoverRate, 20)", "a_share",
               "20-day avg turnover rate", 0),
    FactorSpec("turnover_change_5_20",
               "MA($turnoverRate, 5) / MA($turnoverRate, 20) - 1",
               "a_share", "Turnover acceleration", 1),
    FactorSpec("ln_amount_20",
               "Log(MA($amount, 20))",
               "a_share", "Log of 20d average amount (size proxy)", 0),
    FactorSpec("hl_range_1",
               "($high - $low) / Ref($close, 1)",
               "a_share", "Daily high-low range", -1),
    FactorSpec("intraday_ret",
               "($close - $open) / Ref($close, 1)",
               "a_share", "Intraday return", 0),
]


# ── Registry ────────────────────────────────────────────────────────


class FactorLibrary:
    """In-memory factor library the user can extend."""

    def __init__(self) -> None:
        self._specs: Dict[str, FactorSpec] = {}
        for spec in ALPHA158_LIKE:
            self._specs[spec.name] = spec

    def register(self, spec: FactorSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"factor {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def names(self) -> List[str]:
        return list(self._specs.keys())

    def by_family(self, family: str) -> List[FactorSpec]:
        return [s for s in self._specs.values() if s.family == family]

    def expressions(self, names: Sequence[str]) -> List[str]:
        return [self._specs[n].expression for n in names]

    def spec(self, name: str) -> FactorSpec:
        return self._specs[name]

    def as_dict(self) -> Dict[str, str]:
        return {n: s.expression for n, s in self._specs.items()}
