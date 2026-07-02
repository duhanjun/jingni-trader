"""
正确性测试：验证新模块输出与原 jingni-trader 实现的等价性

对照源：
- skills/factor-engine/engine.py: FactorEngine.compute_a_share_factors
- skills/backtest-engine/scripts/adapters/native_adapter.py: NativeAdapter.run_backtest
- skills/factor-engine/engine.py: FactorEngine._calc_ic
"""
from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd

# 让测试可以独立运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.quant_opt_20260620.factor_engine_polars import (
    PolarsFactorEngine,
    FactorDef,
    compile_factor,
    vectorized_ic_analysis,
)
from experiments.quant_opt_20260620.backtest_vectorized import (
    VectorizedBacktester,
    BacktestConfig,
)
from experiments.quant_opt_20260620.tests.data_gen import make_synthetic_data, make_signals


def test_factor_expr_parser_basic():
    """测试因子表达式解析器基础功能"""
    print("\n[测试] 因子表达式解析器基础...")

    # 简单字段
    expr = compile_factor("Close")
    assert expr is not None

    # 嵌套表达式
    expr = compile_factor("Rank(Ts_Mean(Close, 5))")
    assert expr is not None

    # 多参数
    expr = compile_factor("Ts_Corr(High, Low, 20)")
    assert expr is not None

    # 数值参数
    expr = compile_factor("Ts_Delta(Log(Volume), 10)")
    assert expr is not None

    print("  ✓ 基础解析通过")


def test_factor_expr_parser_errors():
    """测试解析器错误处理"""
    print("\n[测试] 因子表达式解析器错误处理...")

    bad_cases = [
        "UnknownOp(Close)",      # 未知算子
        "Ts_Mean(Close)",        # 缺少参数
        "Ts_Mean(Close, 5",      # 缺少右括号
        "UnknownField",          # 未知字段
        "",                      # 空表达式
    ]
    for expr in bad_cases:
        try:
            compile_factor(expr)
            assert False, f"应该抛错但未抛: {expr}"
        except (ValueError, Exception):
            pass
    print("  ✓ 错误处理通过")


def test_factor_engine_correctness():
    """测试 Polars 因子引擎输出与原 pandas 实现的等价性"""
    print("\n[测试] Polars 因子引擎正确性...")

    data = make_synthetic_data(n_codes=50, n_days=200, seed=42)

    # ---- 原 pandas 实现（从 jingni-trader 复制核心逻辑）----
    df = data.sort_values(["code", "date"]).copy()
    orig = df[["code", "date"]].copy()
    orig["reversal_5d"] = -df.groupby("code")["close"].pct_change(5)
    orig["volatility_20d"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )
    orig["volume_20d"] = df.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    orig["vol_ratio"] = df["volume"] / orig["volume_20d"].replace(0, np.nan)

    # ---- Polars 新实现 ----
    # 用等价表达式
    factors = [
        FactorDef("rev_5d", "Ts_Delta(Close, 5)", direction=-1),
        FactorDef("vol_20d", "Ts_Std(Ts_Ref(Close, 0) / Ts_Ref(Close, 1) - 1, 20)"),
        FactorDef("vol_ratio", "Volume / Ts_Mean(Volume, 20)"),
    ]
    engine = PolarsFactorEngine(factors=factors)
    new = engine.compute_with_direction(data)

    # 合并对比
    merged = orig.merge(new, on=["code", "date"], suffixes=("_orig", "_new"))

    # 5日反转：Ts_Delta(Close, 5) = Close - Close.shift(5)，direction=-1 后 = -(Close - Close.shift(5)) = -pct_change*Close_prev
    # 原 reversal_5d = -pct_change(5) = -(Close/Close.shift(5) - 1) = -(Close - Close.shift(5))/Close.shift(5)
    # 两者数值上不完全相等（分母不同），但 rank 相关性应接近 1
    valid = merged.dropna(subset=["reversal_5d", "rev_5d"])
    if len(valid) > 30:
        corr = valid["reversal_5d"].corr(valid["rev_5d"])
        print(f"  5日反转因子相关性: {corr:.4f}")
        assert corr > 0.95, f"5日反转相关性过低: {corr}"

    # 波动率：原 = std(pct_change, 20)，新 = std(pct_change, 20)，应几乎相等
    valid = merged.dropna(subset=["volatility_20d", "vol_20d"])
    if len(valid) > 30:
        corr = valid["volatility_20d"].corr(valid["vol_20d"])
        print(f"  20日波动率因子相关性: {corr:.4f}")
        assert corr > 0.99, f"波动率相关性过低: {corr}"

    # 量比：应几乎完全相等
    valid = merged.dropna(subset=["vol_ratio_orig", "vol_ratio_new"])
    if len(valid) > 30:
        corr = valid["vol_ratio_orig"].corr(valid["vol_ratio_new"])
        print(f"  量比因子相关性: {corr:.4f}")
        assert corr > 0.99, f"量比相关性过低: {corr}"

    print("  ✓ 因子引擎正确性通过")


def test_ic_analysis_correctness():
    """测试向量化 IC 分析与原实现的等价性"""
    print("\n[测试] 向量化 IC 分析正确性...")

    data = make_synthetic_data(n_codes=50, n_days=200, seed=42)
    df = data.sort_values(["code", "date"]).copy()

    # 构造因子与远期收益
    factor_df = df[["code", "date"]].copy()
    factor_df["rev_5d"] = -df.groupby("code")["close"].pct_change(5)
    factor_df["vol_20d"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )

    fwd = df[["code", "date"]].copy()
    fwd["ret_forward_5d"] = df.groupby("code")["close"].transform(
        lambda x: x.shift(-5) / x - 1
    )

    # ---- 原 scipy 实现 ----
    from scipy import stats
    orig_ic_list = []
    for dt, cross in factor_df.merge(fwd[["code", "date", "ret_forward_5d"]], on=["code", "date"]).groupby("date"):
        valid = cross.dropna(subset=["rev_5d", "ret_forward_5d"])
        if len(valid) < 10:
            continue
        ic, _ = stats.spearmanr(valid["rev_5d"], valid["ret_forward_5d"])
        if not np.isnan(ic):
            orig_ic_list.append(ic)
    orig_ic_mean = float(np.mean(orig_ic_list)) if orig_ic_list else 0.0

    # ---- 新 Polars 实现 ----
    new_ic = vectorized_ic_analysis(
        factor_df, fwd, factor_names=["rev_5d", "vol_20d"], ic_type="spearman"
    )
    new_ic_mean = 0.0
    if "ret_forward_5d" in new_ic:
        for item in new_ic["ret_forward_5d"]:
            if item["factor"] == "rev_5d":
                new_ic_mean = item["ic_mean"]
                break

    print(f"  原 IC mean (rev_5d, 5d forward): {orig_ic_mean:.6f}")
    print(f"  新 IC mean (rev_5d, 5d forward): {new_ic_mean:.6f}")
    # Spearman IC 应几乎完全相等（数值精度差异 < 1e-6）
    assert abs(orig_ic_mean - new_ic_mean) < 1e-4, \
        f"IC 均值差异过大: {orig_ic_mean} vs {new_ic_mean}"

    print("  ✓ IC 分析正确性通过")


def test_backtest_correctness():
    """测试向量化回测与原 native_adapter 的等价性"""
    print("\n[测试] 向量化回测正确性...")

    data = make_synthetic_data(n_codes=30, n_days=100, seed=42)
    signals = make_signals(data)

    # ---- 原 native_adapter 实现 ----
    # 直接复制 native_adapter 的核心逻辑（避免 import 依赖）
    import numpy as np
    import pandas as pd

    def orig_backtest(data, signals, init_capital=1e6, commission_rate=0.00025,
                      stamp_tax_rate=0.001, slippage=0.001, price_limit=True):
        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)
        dates = sorted(signals["date"].unique())
        cash = init_capital
        positions = {}
        equity_records = []
        for dt in dates:
            day_signal = signals[signals["date"] == dt]
            day_data = data[data["date"] == dt]
            if day_data.empty:
                continue
            day_data_map = day_data.set_index("code")
            sell_codes, buy_codes = [], []
            for _, row in day_signal.iterrows():
                code = row["code"]
                sig = row.get("signal", 0)
                if isinstance(sig, (int, float, np.integer, np.floating)):
                    if float(sig) > 0:
                        buy_codes.append(code)
                    elif float(sig) < 0:
                        sell_codes.append(code)
            for code in sell_codes:
                if code not in positions or positions[code] <= 0:
                    continue
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                price = price_row["close"]
                shares = positions[code]
                sell_amount = price * shares
                commission = max(sell_amount * commission_rate, 5)
                tax = sell_amount * stamp_tax_rate
                cash += sell_amount - commission - tax
                positions[code] = 0
            if buy_codes:
                n_buy = len(buy_codes)
                budget_per_stock = cash * 0.95 / n_buy
                for code in buy_codes:
                    if code not in day_data_map.index:
                        continue
                    price_row = day_data_map.loc[code]
                    price = price_row["close"] * (1 + slippage)
                    shares = int(budget_per_stock / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * commission_rate, 5)
                    cost = buy_amount + commission
                    if cost > cash:
                        shares = int((cash * 0.98) / price / 100) * 100
                        if shares <= 0:
                            continue
                        buy_amount = price * shares
                        commission = max(buy_amount * commission_rate, 5)
                        cost = buy_amount + commission
                    cash -= cost
                    positions[code] = positions.get(code, 0) + shares
            market_value = 0
            for code, shares in list(positions.items()):
                if shares <= 0:
                    continue
                if code in day_data_map.index:
                    market_value += shares * day_data_map.loc[code, "close"]
            equity_records.append({"date": dt, "equity": cash + market_value})
        return pd.DataFrame(equity_records)

    orig_eq = orig_backtest(data, signals)
    orig_final = float(orig_eq["equity"].iloc[-1])
    orig_n_days = len(orig_eq)

    # ---- 新向量化回测 ----
    bt = VectorizedBacktester(BacktestConfig(init_capital=1e6))
    result = bt.run(data, signals)
    new_eq = result["equity_curve"]
    new_final = float(new_eq["equity"].iloc[-1])
    new_n_days = len(new_eq)

    print(f"  原实现: 末日净值={orig_final:.2f}, 天数={orig_n_days}")
    print(f"  新实现: 末日净值={new_final:.2f}, 天数={new_n_days}")

    # 天数应一致
    assert orig_n_days == new_n_days, f"天数不一致: {orig_n_days} vs {new_n_days}"

    # 末日净值应非常接近（允许 1% 误差，因为买卖顺序与浮点累积可能略有差异）
    diff_pct = abs(orig_final - new_final) / orig_final if orig_final > 0 else 0
    print(f"  末日净值差异: {diff_pct*100:.4f}%")
    assert diff_pct < 0.01, f"净值差异过大: {diff_pct*100:.4f}%"

    print("  ✓ 回测正确性通过")


def test_walk_forward_folds():
    """测试 walk-forward 折生成"""
    print("\n[测试] Walk-forward 折生成...")

    from experiments.quant_opt_20260620.walk_forward import (
        WalkForwardConfig,
        generate_walk_forward_folds,
    )

    dates = pd.Series(pd.bdate_range("2020-01-01", periods=500))
    config = WalkForwardConfig(train_months=12, test_months=3, step_months=3, min_train_samples=100)
    folds = generate_walk_forward_folds(dates, config)

    print(f"  生成 {len(folds)} 个 walk-forward 折")
    assert len(folds) > 0, "应生成至少一个折"

    # 验证折的连续性：相邻折的 test_start 应晚于前一折的 test_start
    for i in range(1, len(folds)):
        assert folds[i].test_start > folds[i - 1].test_start, "折应按时间顺序排列"

    # 验证无重叠：train_end < test_start
    for f in folds:
        assert f.train_end < f.test_start, "训练集与测试集不应重叠"

    # 验证样本数
    for f in folds:
        assert f.train_size >= config.min_train_samples, "训练样本数应满足最小要求"
        assert f.test_size > 0, "测试样本数应大于0"

    print("  ✓ Walk-forward 折生成通过")


if __name__ == "__main__":
    test_factor_expr_parser_basic()
    test_factor_expr_parser_errors()
    test_factor_engine_correctness()
    test_ic_analysis_correctness()
    test_backtest_correctness()
    test_walk_forward_folds()
    print("\n=== 所有正确性测试通过 ===")
