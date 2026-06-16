"""
对比测试：新方案 vs jingni-trader 原生实现
============================================

验证 3 个优化方向的提升：
1. Factor Expression Engine vs 现有 compute_a_share_factors 硬编码实现
2. Event-Driven Backtest vs native_adapter 朴素回测
3. Walk-Forward Validation vs 无 WFA
"""

from __future__ import annotations
import sys
import time
import importlib.util
import types
import json
from pathlib import Path
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

# ─── 路径设置 ───
WORKSPACE = Path("/workspace")
sys.path.insert(0, str(WORKSPACE / "optimizations" / "20260616" / "factor_expression_engine"))
sys.path.insert(0, str(WORKSPACE / "optimizations" / "20260616" / "event_driven_backtest"))
sys.path.insert(0, str(WORKSPACE / "optimizations" / "20260616" / "walk_forward"))

from expression_engine import parse_and_eval, compute_alpha, ALPHA101_FORMULAS
from event_engine import EventDrivenBacktest, RiskLimits
from walk_forward import WalkForwardConfig, WalkForwardValidator, FoldResult

# ─── 加载 jingni-trader 原生代码 ───
# 构造一个层级包结构，使相对导入工作
_BACKTEST_PKG = types.ModuleType("backtest_pkg")
_BACKTEST_PKG.__path__ = [str(WORKSPACE / "skills" / "backtest-engine" / "scripts")]
sys.modules["backtest_pkg"] = _BACKTEST_PKG
_BASE_PKG = types.ModuleType("backtest_pkg.base")
_BASE_PKG.__path__ = [str(WORKSPACE / "skills" / "backtest-engine" / "scripts" / "base")]
sys.modules["backtest_pkg.base"] = _BASE_PKG
_ADAPTERS_PKG = types.ModuleType("backtest_pkg.adapters")
_ADAPTERS_PKG.__path__ = [str(WORKSPACE / "skills" / "backtest-engine" / "scripts" / "adapters")]
sys.modules["backtest_pkg.adapters"] = _ADAPTERS_PKG


def _load(full_name: str, path: str):
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    _load("backtest_pkg.base.base_backtest_engine",
          str(WORKSPACE / "skills" / "backtest-engine" / "scripts" / "base" / "base_backtest_engine.py"))
    _load("backtest_pkg.base.base_backtest",
          str(WORKSPACE / "skills" / "backtest-engine" / "scripts" / "base" / "base_backtest.py"))
    native_mod = _load("backtest_pkg.adapters.native_adapter",
                       str(WORKSPACE / "skills" / "backtest-engine" / "scripts" / "adapters" / "native_adapter.py"))
    NativeBacktestAdapter = native_mod.NativeAdapter
    HAS_ORIGINAL_BACKTEST = True
except Exception as e:
    print(f"无法加载原生 backtest: {e}")
    HAS_ORIGINAL_BACKTEST = False
    NativeBacktestAdapter = None

HAS_ORIGINAL_FACTOR = False  # 原 base_factor.py 仅有抽象类，无具体实现


# ───────────────────────── 共享测试数据生成 ─────────────────────────

def make_synthetic_data(n_days: int = 250, n_stocks: int = 20, seed: int = 42) -> pd.DataFrame:
    """生成合成 A 股日 K 线数据"""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        np.random.seed(seed + int(c.split(".")[0]))
        rets = np.random.normal(0.0005, 0.02, n_days)
        rets = pd.Series(rets)
        rets.iloc[::30] += np.random.normal(0, 0.04, sum(1 for _ in range(0, n_days, 30)))
        prices = 10 * (1 + rets).cumprod()
        volumes = np.random.randint(500_000, 5_000_000, n_days)
        for i, d in enumerate(dates):
            px = prices.iloc[i]
            rows.append({
                "date": d,
                "code": c,
                "open": px * (1 + np.random.normal(0, 0.002)),
                "high": px * (1 + abs(np.random.normal(0, 0.005))),
                "low": px * (1 - abs(np.random.normal(0, 0.005))),
                "close": px,
                "volume": int(volumes[i]),
                "amount": int(volumes[i] * px),
            })
    return pd.DataFrame(rows)


# ───────────────────────── 测试 1: 因子引擎对比 ─────────────────────────

def test_factor_engine_comparison():
    print("\n" + "=" * 60)
    print("测试 1: 因子引擎对比")
    print("=" * 60)

    data = make_synthetic_data(n_days=200, n_stocks=10)

    # 新方案
    t0 = time.time()
    new_results = {}
    for name in ALPHA101_FORMULAS.keys():
        s = compute_alpha(name, data)
        new_results[name] = s
    new_time = time.time() - t0

    print(f"\n  新引擎（Expression Engine）：")
    print(f"    因子数: {len(new_results)}")
    print(f"    总耗时: {new_time:.3f}s")
    print(f"    平均每个: {new_time/max(len(new_results),1)*1000:.1f}ms")
    print(f"    因子清单: {', '.join(new_results.keys())}")

    # 抽样校验：Reversal_5d 应该等于 -(close - close.shift(5)) per code
    sample = data.sort_values(["code", "date"]).reset_index(drop=True)
    delta_expected = sample.groupby("code")["close"].transform(lambda v: v - v.shift(5))
    expected = -delta_expected
    got = new_results["Reversal_5d"].reindex(sample.index).fillna(-9999).reset_index(drop=True)
    exp = expected.fillna(-9999).reset_index(drop=True)
    diff = float((got - exp).abs().max())
    print(f"    Reversal_5d 校验：与手动实现 max diff = {diff:.6f}")

    print(f"\n  原引擎（base_factor.py）：")
    print(f"    状态: 仅含抽象基类 BaseFactorCalculator，缺少具体因子实现")
    print(f"    无可对比的具体性能数据")

    print(f"\n  新引擎核心优势：")
    print(f"    ✓ 声明式：公式字符串定义因子，无需修改引擎代码")
    print(f"    ✓ 零成本扩展：用户可即时新增任意公式")
    print(f"    ✓ 7 个预置因子：Alpha006/012/033 + 自定义 4 个")
    print(f"    ✓ 横截面/时序算子统一 API")
    print(f"    ✓ 解析器基于 Python AST，无黑盒")

    return {
        "new_time": new_time,
        "n_factors": len(new_results),
        "factor_names": list(new_results.keys()),
        "reversal_5d_diff": diff,
    }


# ───────────────────────── 测试 2: 回测引擎对比 ─────────────────────────

def test_backtest_engine_comparison():
    print("\n" + "=" * 60)
    print("测试 2: 回测引擎对比")
    print("=" * 60)

    data = make_synthetic_data(n_days=200, n_stocks=20)
    data = data.sort_values(["code", "date"]).reset_index(drop=True)
    data["ret_5d"] = data.groupby("code")["close"].pct_change(5)
    data["fwd_ret_5d"] = data.groupby("code")["close"].shift(-5) / data["close"] - 1
    signals = []
    for d in sorted(data["date"].unique())[:195]:
        day = data[(data["date"] == d) & data["ret_5d"].notna()].nlargest(3, "ret_5d")
        for _, r in day.iterrows():
            signals.append({"date": r["date"], "code": r["code"], "signal": 1})
    signals_df = pd.DataFrame(signals)

    # 新引擎
    print(f"\n  新引擎 (Event-Driven)：")
    t0 = time.time()
    engine_new = EventDrivenBacktest(
        init_capital=1_000_000,
        risk_limits=RiskLimits(t_plus_1=True, price_limit_check=True, slippage=0.0001),
        signal_delay_days=1,
    )
    result_new = engine_new.run(data, signals_df)
    new_time = time.time() - t0
    print(f"    耗时: {new_time:.3f}s")
    print(f"    成交笔数: {len(result_new['trades'])}")
    print(f"    订单数: {len(result_new['orders'])}")
    print(f"    总收益: {result_new['metrics'].get('total_return', 0):+.4f}")
    print(f"    Sharpe:  {result_new['metrics'].get('sharpe_ratio', 0):+.3f}")
    print(f"    最大回撤: {result_new['metrics'].get('max_drawdown', 0):.4f}")

    # 原引擎
    orig_time = float("nan")
    orig_trades = 0
    orig_metrics = {}
    if HAS_ORIGINAL_BACKTEST and NativeBacktestAdapter is not None:
        print(f"\n  原引擎 (Native Adapter)：")
        t0 = time.time()
        try:
            adapter = NativeBacktestAdapter()
            result_orig = adapter.run_backtest(data, signals_df, init_capital=1_000_000)
            orig_time = time.time() - t0
            print(f"    耗时: {orig_time:.3f}s")
            if isinstance(result_orig, dict):
                trades = result_orig.get("trades", [])
                if hasattr(trades, "__len__"):
                    orig_trades = len(trades)
                metrics = result_orig.get("metrics", {})
                if metrics:
                    orig_metrics = metrics
                    print(f"    成交笔数: {orig_trades}")
                    if "total_return" in metrics:
                        print(f"    总收益: {metrics['total_return']:+.4f}")
            elif isinstance(result_orig, pd.DataFrame):
                orig_trades = len(result_orig)
        except Exception as e:
            import traceback
            print(f"    原引擎运行失败: {e}")
            print(traceback.format_exc())

    print(f"\n  关键差异（结构性 vs 实现性）：")
    print(f"    ┌─ 原引擎 ─────────────────────────────────────────┐")
    print(f"    │ 信号生成后立即按当日 close 成交（look-ahead bias）│")
    print(f"    │ 仅记录已成交，无订单状态机                       │")
    print(f"    │ 风控：仅涨跌停检查（price_limit）                │")
    print(f"    └──────────────────────────────────────────────────┘")
    print(f"    ┌─ 新引擎 ─────────────────────────────────────────┐")
    print(f"    │ 信号 → 次日 open 价成交（严格 T+1）              │")
    print(f"    │ 完整订单状态机 PENDING→FILLED/CANCELED/REJECTED  │")
    print(f"    │ 7 维风控：单票权重/单日亏损/单笔金额/涨跌停/...  │")
    print(f"    └──────────────────────────────────────────────────┘")

    return {
        "new_time": new_time,
        "orig_time": orig_time,
        "new_trades": len(result_new["trades"]),
        "orig_trades": orig_trades,
        "new_metrics": result_new["metrics"],
    }


# ───────────────────────── 测试 3: Walk-Forward 对比 ─────────────────────────

def test_walk_forward_value():
    print("\n" + "=" * 60)
    print("测试 3: Walk-Forward 验证（过拟合检测）")
    print("=" * 60)

    data = make_synthetic_data(n_days=480, n_stocks=20)
    data["fwd_ret_5d"] = data.groupby("code")["close"].shift(-5) / data["close"] - 1
    data = data.sort_values(["code", "date"]).reset_index(drop=True)

    def factor_fn(sub_data, period):
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

    print(f"\n  WFA 折数: {len(results)}")
    print(f"  训练 IC 均值: {diag['train_ic_mean']:+.4f}")
    print(f"  测试 IC 均值: {diag['test_ic_mean']:+.4f}")
    print(f"  IC Decay:     {diag['ic_decay']:+.4f}")
    print(f"  IC Ratio:     {diag['ic_ratio']:+.4f}")
    print(f"  Test IC IR:   {diag['ic_ir']:+.4f}")
    print(f"  Test sign>0:  {diag['test_sign_positive_rate']:.2%}")
    print(f"  诊断结论:     {diag['warning']}")

    t0 = time.time()
    _ = validator._eval_factor(
        data.iloc[:1000], pd.Series(data["fwd_ret_5d"].iloc[:1000]), "fwd_ret_5d"
    )
    eval_time = time.time() - t0
    print(f"\n  单折评估耗时: {eval_time*1000:.1f}ms")

    print(f"\n  框架价值：")
    print(f"    ┌─ 原项目（main 分支）─────────────────────────────┐")
    print(f"    │ README 提及 '样本外再验证' 但无具体实现          │")
    print(f"    │ 因子评估仅做单次 train/test 切分                 │")
    print(f"    │ 无 purge/embargo 机制                             │")
    print(f"    └──────────────────────────────────────────────────┘")
    print(f"    ┌─ 新方案 ─────────────────────────────────────────┐")
    print(f"    │ 滚动 WFA + 锚定 WFA 两种模式                     │")
    print(f"    │ Purge/Embargo 概念（参考 mlfinlab）               │")
    print(f"    │ 自动过拟合诊断（IC Decay、IC Ratio、IC IR）       │")
    print(f"    └──────────────────────────────────────────────────┘")

    return {
        "n_folds": len(results),
        "diagnosis": diag,
    }


# ───────────────────────── 主入口 ─────────────────────────

def main():
    print("=" * 60)
    print("jingni-trader 优化验证套件（2026-06-16）")
    print("=" * 60)

    results = {
        "timestamp": "2026-06-16",
        "branch": "feat/quant-opt-20260616",
    }
    results["factor"] = test_factor_engine_comparison()
    results["backtest"] = test_backtest_engine_comparison()
    results["walk_forward"] = test_walk_forward_value()

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            if isinstance(obj, (pd.Series,)):
                return obj.tolist()
            return super().default(obj)

    out_path = WORKSPACE / "optimizations" / "20260616" / "reports" / "comparison_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, cls=NpEncoder))
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
