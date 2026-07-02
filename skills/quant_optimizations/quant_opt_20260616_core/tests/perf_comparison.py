"""
性能对比: 向量化版本 vs 原生 (逐日循环) 版本
============================================

针对同样的输入数据, 测量:
1. 耗时 (秒)
2. 总收益/年化收益差异
3. 资金曲线终值差异
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

# 路径
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def _make_market(n_stocks: int = 100, n_days: int = 500, seed: int = 7) -> pd.DataFrame:
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


def _make_signals(market: pd.DataFrame, top_frac: float = 0.2) -> pd.DataFrame:
    df = market.sort_values(["code", "date"]).copy()
    df["ret_20"] = df.groupby("code")["close"].pct_change(20)
    df["rank"] = df.groupby("date")["ret_20"].rank(pct=True, ascending=False)
    sig = df[["code", "date", "rank"]].copy()
    sig["signal"] = (sig["rank"] <= top_frac).astype(int)
    return sig[["code", "date", "signal"]]


def _load_existing_native():
    """加载项目原有的 native_adapter"""
    bt_engine_path = os.path.join(PROJECT_ROOT, "skills", "backtest-engine")
    if bt_engine_path not in sys.path:
        sys.path.insert(0, bt_engine_path)
    try:
        from scripts.adapters.native_adapter import NativeAdapter
        return NativeAdapter
    except Exception as e:
        print(f"[warn] cannot import existing native_adapter: {e}")
        return None


def main():
    market = _make_market(n_stocks=100, n_days=500)
    signals = _make_signals(market)

    from skills.quant_optimizations.quant_opt_20260616_core.vectorized_backtest import VectorizedBacktester
    vec = VectorizedBacktester()

    t0 = time.time()
    res_v = vec.run(market, signals)
    t_v = time.time() - t0

    # 现有原生
    NativeAdapter = _load_existing_native()
    if NativeAdapter is not None:
        adapter = NativeAdapter()
        # 原生接口签名略不同: 需要 signals 列名包含 'signal'
        t0 = time.time()
        result = adapter.run_backtest(
            data=market,
            signals=signals,
            init_capital=1_000_000.0,
        )
        t_n = time.time() - t0
    else:
        result = None
        t_n = None

    print("\n=== 性能对比 (100 标的 x 500 交易日) ===")
    print(f"向量化回测: {t_v:.3f}s  "
          f"total_return={res_v['metrics']['total_return']:.4f}  "
          f"max_dd={res_v['metrics']['max_drawdown']:.4f}")
    if result is not None and t_n is not None:
        eq_v = res_v["equity_curve"].set_index("date")["equity"].iloc[-1]
        eq_n = result["equity_curve"].set_index("date")["equity"].iloc[-1]
        print(f"原生回测:   {t_n:.3f}s  "
              f"total_return={result['metrics']['total_return']:.4f}  "
              f"max_dd={result['metrics']['max_drawdown']:.4f}")
        print(f"加速比:    {t_n / t_v:.1f}x")
        print(f"资金曲线终值: 向量化={eq_v:.2f}, 原生={eq_n:.2f}, 差={(eq_v - eq_n) / eq_n * 100:.2f}%")
    else:
        print("原生 adapter 不可用, 跳过对比")


if __name__ == "__main__":
    main()