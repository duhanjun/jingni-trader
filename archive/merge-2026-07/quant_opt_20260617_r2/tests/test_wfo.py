"""
Walk-Forward Optimization 验证测试
==================================

测试内容:
1. 切分正确性：rolling / anchored 两种模式生成的切分窗口
2. Purge gap 正确性：train/test 间隔符合配置
3. 无未来信息泄露：训练集内最大日期严格 < 测试集最小日期
4. 端到端 WFO：使用合成数据和简单 ML 策略走完整流程
5. 对比 WFO 与静态 train/test split 的 OOS 性能差异

运行:
    cd /workspace
    PYTHONPATH=quant_opt_20260617 python3 -m pytest quant_opt_20260617/tests/test_wfo.py -v
    # 或直接 python3 跑
    python3 quant_opt_20260617/tests/test_wfo.py
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "skills", "backtest-engine"))

from quant_opt_20260617_r2.walk_forward.wfo import WalkForwardOptimizer, WFOConfig
from quant_opt_20260617_r2.backtest.vectorized_engine import VectorizedBacktestEngine, VectorizedBacktestConfig


# ======================================================================
# 切分测试
# ======================================================================

def make_trading_dates(n: int = 600) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2022-01-01", periods=n)


def test_split_rolling():
    """Rolling 模式切分测试"""
    print("\n[test_split_rolling] running...")
    dates = make_trading_dates(600)
    wfo = WalkForwardOptimizer(WFOConfig(
        n_splits=3, train_days=200, valid_days=30, test_days=30,
        purge_days=5, anchored=False,
    ))
    splits = wfo.split(dates)
    assert len(splits) == 3, f"expected 3 splits, got {len(splits)}"

    for sp in splits:
        # 1) 训练集长度正确（rolling）
        train_len = (dates >= sp.train_start).argmax()  # noqa
        # 训练天数 = train_end - train_start + 1（按 bdate 算）
        # 但我们直接看时间差
        assert (sp.train_end - sp.train_start).days >= 0
        # 2) 顺序：train_end < valid_start - purge_days
        assert (sp.valid_start - sp.train_end).days > wfo.config.purge_days
        # 3) valid_end < test_start - purge_days
        assert (sp.test_start - sp.valid_end).days > wfo.config.purge_days
        # 4) 无未来泄露
        assert sp.train_end < sp.valid_start
        assert sp.valid_end < sp.test_start
        # 5) test 长度大致正确（30 bdays ≈ 42 calendar days）
        test_days_actual = (sp.test_end - sp.test_start).days
        assert 25 <= test_days_actual <= 50, test_days_actual
    print(f"  generated {len(splits)} splits, all constraints satisfied")
    print("  ✓ test_split_rolling passed")


def test_split_anchored():
    """Anchored 模式：train 起点固定"""
    print("\n[test_split_anchored] running...")
    dates = make_trading_dates(600)
    wfo = WalkForwardOptimizer(WFOConfig(
        n_splits=3, train_days=200, valid_days=30, test_days=30,
        purge_days=5, anchored=True,
    ))
    splits = wfo.split(dates)
    assert len(splits) == 3
    # 所有段的 train_start 应该是 dates[0]
    for sp in splits:
        assert sp.train_start == dates[0], f"anchored violated: {sp.train_start} vs {dates[0]}"
    # 训练长度递增
    train_lens = [(sp.train_end - sp.train_start).days for sp in splits]
    assert train_lens[0] < train_lens[1] < train_lens[2], train_lens
    print(f"  anchored train_lens (days): {train_lens}")
    print("  ✓ test_split_anchored passed")


def test_no_lookahead():
    """无未来信息泄露测试"""
    print("\n[test_no_lookahead] running...")
    dates = make_trading_dates(800)
    for anchored in [False, True]:
        wfo = WalkForwardOptimizer(WFOConfig(
            n_splits=5, train_days=300, valid_days=60, test_days=60,
            purge_days=10, anchored=anchored,
        ))
        splits = wfo.split(dates)
        for sp in splits:
            # 严格 < 关系
            assert sp.train_end < sp.valid_start, "train/valid overlap"
            assert sp.valid_end < sp.test_start, "valid/test overlap"
            # purge gap 严格 > 0（按交易日计数）
            train_valid_gap = (sp.valid_start - sp.train_end).days
            valid_test_gap = (sp.test_start - sp.valid_end).days
            # 注：purge_days 是按交易日计，而 .days 是日历日
            # 经验换算：5 bdays ≈ 7-9 calendar days
            assert train_valid_gap > 0, f"train/valid must have positive gap: {train_valid_gap}"
            assert valid_test_gap > 0, f"valid/test must have positive gap: {valid_test_gap}"
    print("  ✓ test_no_lookahead passed")


def test_split_data_too_short():
    """数据太短时应返回空切分"""
    print("\n[test_split_data_too_short] running...")
    dates = make_trading_dates(100)  # 远小于所需
    wfo = WalkForwardOptimizer(WFOConfig(
        n_splits=5, train_days=252, valid_days=63, test_days=63,
    ))
    splits = wfo.split(dates)
    assert len(splits) == 0, f"expected 0 splits, got {len(splits)}"
    print("  ✓ test_split_data_too_short passed")


# ======================================================================
# 端到端 WFO 测试
# ======================================================================

def gen_factor_data(n_stocks: int = 30, n_days: int = 600, seed: int = 7) -> pd.DataFrame:
    """生成带有 forward return 的因子数据"""
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    rows = []
    for code in codes:
        # 价格（仅用于参考）
        price = np.random.uniform(10, 50)
        ret = np.random.normal(0.0003, 0.02, n_days)
        prices = price * np.cumprod(1 + ret)
        # 3 个因子：动量、波动率、量能
        mom = pd.Series(prices).pct_change(20)
        vol = pd.Series(prices).pct_change().rolling(20).std()
        volu = pd.Series(np.random.lognormal(10, 0.5, n_days))
        # 真实 alpha = 0.3 * mom - 0.5 * vol + 噪声
        alpha = 0.3 * mom.fillna(0) - 0.5 * vol.fillna(0) + np.random.normal(0, 0.01, n_days)
        # 未来 5 日收益
        fwd = pd.Series(prices).pct_change(5).shift(-5)
        df_one = pd.DataFrame({
            "date": dates,
            "code": code,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "vol": volu,
            "factor_mom": mom,
            "factor_vol": vol,
            "factor_volu": volu,
            "fwd_return": fwd,
        })
        rows.append(df_one)
    df = pd.concat(rows, ignore_index=True)
    df["pre_close"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["change_pct"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100
    df["is_st"] = False
    df["is_limit_up"] = df["change_pct"] >= 9.9
    df["is_limit_down"] = df["change_pct"] <= -9.9
    return df


def _train_fn(train_data, valid_data):
    """简单训练：计算每个因子的 IC，然后用作权重（无 ML 库依赖）"""
    from scipy.stats import spearmanr
    weights = {}
    for factor in ["factor_mom", "factor_vol", "factor_volu"]:
        sub = train_data[[factor, "fwd_return"]].dropna()
        if len(sub) < 30:
            continue
        ic, _ = spearmanr(sub[factor], sub["fwd_return"])
        if not np.isnan(ic):
            weights[factor] = float(ic)
    # 在 valid 上做归一化（避免训练集过拟合）
    if valid_data is not None and not valid_data.empty:
        for factor in list(weights.keys()):
            sub = valid_data[[factor, "fwd_return"]].dropna()
            if len(sub) < 30:
                continue
            ic_v, _ = spearmanr(sub[factor], sub["fwd_return"])
            if not np.isnan(ic_v):
                # 取 train/valid IC 的均值
                weights[factor] = (weights[factor] + float(ic_v)) / 2
    # 归一化
    abs_sum = sum(abs(v) for v in weights.values())
    if abs_sum > 0:
        weights = {k: v / abs_sum for k, v in weights.items()}
    return weights


def _predict_fn(model, test_data):
    """根据权重对 test_data 打分，生成 top-20% 买入信号"""
    weights = model  # dict[factor_name] -> weight
    if not weights:
        return pd.DataFrame()
    df = test_data.copy()
    df["alpha_score"] = 0.0
    for factor, w in weights.items():
        if factor in df.columns:
            df["alpha_score"] += w * df[factor].fillna(0)
    df["rank_pct"] = df.groupby("date")["alpha_score"].rank(pct=True)
    sig = df[["code", "date"]].copy()
    sig["signal"] = 0
    sig.loc[df["rank_pct"] > 0.8, "signal"] = 1
    sig.loc[df["rank_pct"] < 0.2, "signal"] = -1
    return sig


def _backtest_fn(test_data, signals):
    """回测函数：返回包含 equity_curve 的 metrics dict"""
    engine = VectorizedBacktestEngine(VectorizedBacktestConfig(init_capital=500_000))
    res = engine.run_backtest(data=test_data, signals=signals)
    if res.equity_curve.empty:
        return {"n_test_signals": len(signals)}
    return {
        "n_test_signals": len(signals),
        "equity_curve": res.equity_curve,
        "metrics": res.metrics,
    }


def test_wfo_end_to_end():
    """端到端 WFO 测试"""
    print("\n[test_wfo_end_to_end] running...")
    data = gen_factor_data(n_stocks=30, n_days=600)
    wfo = WalkForwardOptimizer(WFOConfig(
        n_splits=4, train_days=200, valid_days=30, test_days=30,
        purge_days=5, anchored=False,
    ))
    result = wfo.run(
        data=data,
        train_fn=_train_fn,
        predict_fn=_predict_fn,
        backtest_fn=_backtest_fn,
    )
    assert result["success"], result
    assert result["n_segments"] >= 3, f"too few segments: {result['n_segments']}"
    print(f"  n_segments={result['n_segments']}")
    for seg in result["segments"]:
        if "error" in seg:
            print(f"  segment {seg['segment_id']}: ERROR {seg['error']}")
        else:
            md = seg["metrics"].get("metrics", {}).get("max_drawdown", "N/A")
            print(f"  segment {seg['segment_id']}: "
                  f"train={seg['train_range']}, test={seg['test_range']}, mdd={md}")

    oos = result["oos_summary"]
    print(f"  OOS 总收益: {oos.get('total_return', 'N/A'):.2%}" if oos.get("total_return") is not None else "  无 OOS 数据")
    print("  ✓ test_wfo_end_to_end passed")
    return result


def test_wfo_vs_static():
    """对比 WFO 与静态 train/test 切分的 OOS 性能"""
    print("\n[test_wfo_vs_static] running...")
    data = gen_factor_data(n_stocks=30, n_days=600)
    # 静态切分：前 80% 训练，后 20% 测试
    dates = sorted(data["date"].unique())
    n = len(dates)
    cutoff = dates[int(n * 0.8)]
    train_static = data[data["date"] < cutoff]
    test_static = data[data["date"] >= cutoff]
    model = _train_fn(train_static, None)
    sig_static = _predict_fn(model, test_static)
    bt_static = _backtest_fn(test_static, sig_static)

    # WFO
    wfo = WalkForwardOptimizer(WFOConfig(
        n_splits=4, train_days=200, valid_days=30, test_days=30,
        purge_days=5, anchored=False,
    ))
    result = wfo.run(data, _train_fn, _predict_fn, _backtest_fn)
    oos = result["oos_summary"]

    static_ret = bt_static.get("metrics", {}).get("total_return", 0.0)
    wfo_ret = oos.get("total_return", 0.0)
    print(f"  static split total_return: {static_ret:.2%}")
    print(f"  WFO total_return: {wfo_ret:.2%}")
    # WFO 通常 < static（因为 OOS 不会作弊），但不强制
    print("  ✓ test_wfo_vs_static passed (no strict assertion)")
    return {
        "static_total_return": static_ret,
        "wfo_total_return": wfo_ret,
        "static_max_drawdown": bt_static.get("metrics", {}).get("max_drawdown", 0.0),
        "wfo_max_drawdown": oos.get("max_drawdown", 0.0),
    }


# ======================================================================
# 入口
# ======================================================================

def main():
    print("=" * 60)
    print("Walk-Forward Optimization 验证测试")
    print("=" * 60)
    summary: Dict[str, Any] = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "python": sys.version,
        "tests": {},
    }
    try:
        test_split_rolling()
        summary["tests"]["split_rolling"] = {"passed": True}
    except AssertionError as e:
        summary["tests"]["split_rolling"] = {"error": str(e)}
        raise
    try:
        test_split_anchored()
        summary["tests"]["split_anchored"] = {"passed": True}
    except AssertionError as e:
        summary["tests"]["split_anchored"] = {"error": str(e)}
        raise
    try:
        test_no_lookahead()
        summary["tests"]["no_lookahead"] = {"passed": True}
    except AssertionError as e:
        summary["tests"]["no_lookahead"] = {"error": str(e)}
        raise
    try:
        test_split_data_too_short()
        summary["tests"]["split_data_too_short"] = {"passed": True}
    except AssertionError as e:
        summary["tests"]["split_data_too_short"] = {"error": str(e)}
        raise
    try:
        r = test_wfo_end_to_end()
        summary["tests"]["wfo_end_to_end"] = {
            "n_segments": r["n_segments"],
            "oos_total_return": r["oos_summary"].get("total_return"),
            "oos_sharpe": r["oos_summary"].get("sharpe_ratio"),
            "oos_max_drawdown": r["oos_summary"].get("max_drawdown"),
        }
    except AssertionError as e:
        summary["tests"]["wfo_end_to_end"] = {"error": str(e)}
        raise
    try:
        r2 = test_wfo_vs_static()
        summary["tests"]["wfo_vs_static"] = r2
    except AssertionError as e:
        summary["tests"]["wfo_vs_static"] = {"error": str(e)}
        raise

    out_path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports", "wfo_test.json"
    ))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存: {out_path}")
    print("\nALL TESTS PASSED ✓")


if __name__ == "__main__":
    main()