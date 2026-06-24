"""
集成测试: 串联 factor_expr_engine + dynamic_weighting + vectorized_backtest
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_market(n_stocks: int = 20, n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    rows = []
    for code in codes:
        close = 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n_days)))
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "open": close[i] * (1 + rng.normal(0, 0.005)),
                "high": close[i] * (1 + abs(rng.normal(0, 0.01))),
                "low": close[i] * (1 - abs(rng.normal(0, 0.01))),
                "close": close[i],
                "volume": int(rng.integers(1_000_000, 5_000_000)),
                "amount": float(close[i] * rng.integers(1_000_000, 5_000_000)),
            })
    return pd.DataFrame(rows)


def test_end_to_end_factor_to_backtest():
    """端到端: 表达式引擎产出因子 -> 截面排名生成信号 -> 向量化回测"""
    from quant_opt_20260616_core.factor_expr_engine import FactorExprEngine
    from quant_opt_20260616_core.vectorized_backtest import VectorizedBacktester

    market = _make_market()
    eng = FactorExprEngine()
    factors = eng.compute_batch(market, {
        "mom_20": "$close / Ref($close, 20) - 1",
        "vol_20": "Std($close, 20) / Mean($close, 20)",
        "amt_ma5": "Mean($amount, 5)",
    })
    # 合并, 截面排名打分
    merged = market.merge(factors, on=["code", "date"], how="left")
    # 简单等权 alpha = 排名
    merged["score"] = (
        merged.groupby("date")["mom_20"].rank(pct=True) * 0.5
        - merged.groupby("date")["vol_20"].rank(pct=True) * 0.3
        + merged.groupby("date")["amt_ma5"].rank(pct=True) * 0.2
    )
    # 缺失剔除
    merged = merged.dropna(subset=["score"])
    # 简单信号: 每日 score 前 20% 买入, 其余不持有
    merged["rank"] = merged.groupby("date")["score"].rank(pct=True, ascending=False)
    sig = merged[merged["rank"] <= 0.2][["code", "date"]].copy()
    sig["signal"] = 1
    # 下一日重置: 用 0 表示不持有
    all_dates = market["date"].drop_duplicates().sort_values()
    full = pd.DataFrame(
        [(c, d) for c in market["code"].unique() for d in all_dates],
        columns=["code", "date"],
    )
    sig = full.merge(sig, on=["code", "date"], how="left")
    sig["signal"] = sig["signal"].fillna(0).astype(int)

    bt = VectorizedBacktester()
    res = bt.run(market, sig)
    m = res["metrics"]
    print(f"  [OK] end-to-end: total_return={m['total_return']:.3f} sharpe={m['sharpe_ratio']:.3f} mdd={m['max_drawdown']:.3f}")
    assert m["n_days"] > 0


def test_dynamic_weighting_with_factor_engine():
    """用表达式引擎产 IC, 动态加权产出 alpha_score, 与等权对比"""
    from quant_opt_20260616_core.factor_expr_engine import FactorExprEngine
    from quant_opt_20260616_core.dynamic_weighting import DynamicFactorWeighting

    market = _make_market(n_stocks=15, n_days=300, seed=3)
    eng = FactorExprEngine()
    factors = eng.compute_batch(market, {
        "mom_5": "$close / Ref($close, 5) - 1",
        "mom_20": "$close / Ref($close, 20) - 1",
        "vol_20": "Std($close, 20) / Mean($close, 20)",
    })
    merged = market[["code", "date", "close"]].merge(factors, on=["code", "date"], how="left")
    # 构造 forward return
    merged = merged.sort_values(["code", "date"])
    merged["fwd_ret_5d"] = merged.groupby("code")["close"].transform(
        lambda s: s.shift(-5) / s - 1
    )
    # 计算每日 IC
    fac_names = ["mom_5", "mom_20", "vol_20"]
    rows = []
    for dt, g in merged.groupby("date"):
        row = {"date": dt}
        for f in fac_names:
            sub = g[[f, "fwd_ret_5d"]].dropna()
            if len(sub) >= 5:
                row[f] = sub[f].rank().corr(sub["fwd_ret_5d"].rank())
        rows.append(row)
    ic_history = pd.DataFrame(rows).set_index("date")
    weighting = DynamicFactorWeighting(method="icir_decay", halflife=60)
    w = weighting.compute(ic_history)
    print(f"  [OK] dynamic weighting: {w}")
    assert abs(sum(w.values()) - 1.0) < 1e-6


def run() -> dict:
    test_end_to_end_factor_to_backtest()
    test_dynamic_weighting_with_factor_engine()
    return {"status": "passed", "cases": 2}


if __name__ == "__main__":
    run()