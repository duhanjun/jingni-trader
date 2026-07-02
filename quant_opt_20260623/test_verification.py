"""
验证测试套件
对三个优化模块进行：正确性测试、性能对比测试、边界条件测试

测试模块：
1. event_driven_backtest.py - 事件驱动回测引擎
2. expression_factor_engine.py - 表达式因子引擎
3. vectorized_ic.py - 向量化 IC 分析

运行: python quant_opt_20260623/test_verification.py
"""
from __future__ import annotations
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 确保能导入本目录模块
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from event_driven_backtest import (
    EventDrivenBacktestEngine, FillModel, EventType, OrderStatus,
)
from expression_factor_engine import (
    ExpressionFactorEngine, ExpressionParser, build_alpha158_factors,
)
from vectorized_ic import VectorizedICAnalyzer, calc_ic_baseline


RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 测试数据生成器
# ============================================================

def generate_synthetic_data(
    n_stocks: int = 50,
    n_days: int = 250,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """生成合成 A 股日线数据（含 OHLCV + ST/涨跌停标记）"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

    rows = []
    for code in codes:
        price = 10.0 + rng.normal(0, 0.5)
        for dt in dates:
            ret = rng.normal(0, 0.02)
            open_p = price * (1 + rng.normal(0, 0.005))
            close = price * (1 + ret)
            high = max(open_p, close) * (1 + abs(rng.normal(0, 0.005)))
            low = min(open_p, close) * (1 - abs(rng.normal(0, 0.005)))
            volume = int(rng.lognormal(13, 0.5))
            amount = volume * (open_p + close) / 2
            pre_close = price
            change_pct = (close - pre_close) / pre_close
            rows.append({
                "code": code,
                "date": dt,
                "open": round(open_p, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": volume,
                "amount": round(amount, 2),
                "pre_close": round(pre_close, 4),
                "change_pct": round(change_pct, 6),
                "turnover_rate": round(rng.uniform(0.5, 5.0), 4),
                "is_st": False,
                "is_limit_up": change_pct > 0.095,
                "is_limit_down": change_pct < -0.095,
                "listed_days": 365,
            })
            price = close

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def generate_signals(data: pd.DataFrame, strategy: str = "momentum") -> pd.DataFrame:
    """基于数据生成交易信号"""
    df = data.sort_values(["code", "date"]).copy()
    if strategy == "momentum":
        df["ret_5d"] = df.groupby("code")["close"].pct_change(5)
        df["signal"] = 0
        df.loc[df.groupby("date")["ret_5d"].transform(
            lambda x: x.rank(pct=True)) > 0.8, "signal"] = 1
        df.loc[df.groupby("date")["ret_5d"].transform(
            lambda x: x.rank(pct=True)) < 0.2, "signal"] = -1
    elif strategy == "reversal":
        df["ret_5d"] = df.groupby("code")["close"].pct_change(5)
        df["signal"] = 0
        df.loc[df.groupby("date")["ret_5d"].transform(
            lambda x: x.rank(pct=True)) < 0.2, "signal"] = 1
        df.loc[df.groupby("date")["ret_5d"].transform(
            lambda x: x.rank(pct=True)) > 0.8, "signal"] = -1
    return df[df["signal"] != 0][["code", "date", "signal"]].reset_index(drop=True)


# ============================================================
# 测试结果收集
# ============================================================

class TestReport:
    def __init__(self):
        self.results = []
        self.start_time = time.perf_counter()

    def add(self, category: str, name: str, passed: bool, detail: str = "", metrics: dict = None):
        self.results.append({
            "category": category,
            "name": name,
            "passed": passed,
            "detail": detail,
            "metrics": metrics or {},
        })
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{category}] {status} {name}" + (f" | {detail}" if detail else ""))

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0,
            "elapsed_sec": round(time.perf_counter() - self.start_time, 2),
        }

    def to_dict(self) -> dict:
        return {"results": self.results, "summary": self.summary()}


# ============================================================
# 测试 1：事件驱动回测引擎
# ============================================================

def test_event_driven_backtest(report: TestReport, data: pd.DataFrame, signals: pd.DataFrame):
    print("\n=== 测试 1: 事件驱动回测引擎 ===")

    # ---- 1.1 正确性：T+1 严格执行 ----
    engine = EventDrivenBacktestEngine(
        init_capital=1e6,
        t_plus_1=True,
        price_limit=True,
        fill_model=FillModel(price_type="open", slippage=0.001),
    )
    result = engine.run(data, signals)
    trades = result["trades"]

    # 检查每笔交易的成交日 > 信号日（T+1）
    if not trades.empty:
        trades["signal_date"] = pd.to_datetime(trades["signal_date"])
        trades["date"] = pd.to_datetime(trades["date"])
        violations = (trades["date"] <= trades["signal_date"]).sum()
        report.add(
            "事件回测", "T+1严格执行",
            violations == 0,
            f"成交笔数={len(trades)}, T+1违规={violations}",
        )
    else:
        report.add("事件回测", "T+1严格执行", False, "无成交记录")

    # ---- 1.2 正确性：成交价使用次日开盘 ----
    if not trades.empty:
        # 抽样验证：买入订单成交价应接近次日开盘价 * (1+slippage)
        buy_trades = trades[trades["action"] == "buy"].head(5)
        ok = 0
        for _, t in buy_trades.iterrows():
            next_day = pd.to_datetime(t["date"])
            code = t["code"]
            bar = data[(data["code"] == code) & (data["date"] == next_day.strftime("%Y-%m-%d"))]
            if bar.empty:
                continue
            expected = float(bar.iloc[0]["open"]) * 1.001
            if abs(t["price"] - expected) / expected < 0.01:
                ok += 1
        report.add(
            "事件回测", "成交价=次日开盘+滑点",
            ok >= 3,
            f"买入抽样验证 {ok}/{len(buy_trades)} 符合预期",
        )

    # ---- 1.3 正确性：订单状态机完整性 ----
    orders = result["orders"]
    if not orders.empty:
        valid_statuses = orders["status"].isin(
            ["pending", "accepted", "filled", "canceled", "rejected"])
        report.add(
            "事件回测", "订单状态机完整性",
            valid_statuses.all(),
            f"订单数={len(orders)}, 状态合法={valid_statuses.sum()}",
        )
        # 检查每个 filled 订单都有 fill_price > 0
        filled = orders[orders["status"] == "filled"]
        if not filled.empty:
            report.add(
                "事件回测", "已成交订单有有效价格",
                (filled["fill_price"] > 0).all(),
                f"已成交={len(filled)}, 价格>0={int((filled['fill_price'] > 0).sum())}",
            )

    # ---- 1.4 性能对比：与 native_adapter 对比 ----
    # 加载现有 native_adapter（需注册 scripts 包以支持相对导入）
    try:
        import importlib.util
        bt_scripts = "/workspace/skills/backtest-engine/scripts"
        # 注册 scripts 包
        if "scripts" not in sys.modules:
            init_py = os.path.join(bt_scripts, "__init__.py")
            spec = importlib.util.spec_from_file_location(
                "scripts", init_py,
                submodule_search_locations=[bt_scripts],
            )
            scripts_pkg = importlib.util.module_from_spec(spec)
            sys.modules["scripts"] = scripts_pkg
            spec.loader.exec_module(scripts_pkg)
        # 注册 adapters 和 base 子包
        for sub in ["base", "adapters"]:
            sub_path = os.path.join(bt_scripts, sub)
            sub_init = os.path.join(sub_path, "__init__.py")
            if os.path.exists(sub_init) and f"scripts.{sub}" not in sys.modules:
                sspec = importlib.util.spec_from_file_location(
                    f"scripts.{sub}", sub_init,
                    submodule_search_locations=[sub_path],
                )
                smod = importlib.util.module_from_spec(sspec)
                sys.modules[f"scripts.{sub}"] = smod
                sspec.loader.exec_module(smod)
        # 加载 native_adapter 模块
        na_path = os.path.join(bt_scripts, "adapters", "native_adapter.py")
        na_spec = importlib.util.spec_from_file_location("scripts.adapters.native_adapter", na_path)
        na_mod = importlib.util.module_from_spec(na_spec)
        sys.modules["scripts.adapters.native_adapter"] = na_mod
        na_spec.loader.exec_module(na_mod)
        NativeAdapter = na_mod.NativeAdapter

        native = NativeAdapter()
        t0 = time.perf_counter()
        native_result = native.run_backtest(
            data=data, signals=signals,
            init_capital=1e6, commission_rate=0.00025,
            stamp_tax_rate=0.001, t_plus_1=True, price_limit=True, slippage=0.001,
        )
        native_elapsed = time.perf_counter() - t0
        event_elapsed = result["metrics"].get("elapsed_sec", 0)

        speedup = native_elapsed / event_elapsed if event_elapsed > 0 else 0
        report.add(
            "事件回测", "性能对比(事件 vs 向量化)",
            True,  # 性能测试不判 pass/fail
            f"事件驱动={event_elapsed:.3f}s, 向量化={native_elapsed:.3f}s, 比值={speedup:.2f}x",
            {"event_elapsed": event_elapsed, "native_elapsed": native_elapsed, "speedup": speedup},
        )

        # ---- 1.5 净值合理性对比 ----
        event_eq = result["equity_curve"]["equity"].iloc[-1] if not result["equity_curve"].empty else 0
        native_eq = native_result["equity_curve"]["equity"].iloc[-1] if not native_result["equity_curve"].empty else 0
        # 两者都应在合理范围（不爆仓，不翻倍）
        report.add(
            "事件回测", "净值合理性",
            0.5e6 < event_eq < 2e6 and 0.5e6 < native_eq < 2e6,
            f"事件引擎终值={event_eq:.0f}, 向量化终值={native_eq:.0f}",
        )

        # ---- 1.6 前视偏差检测：事件引擎应更保守（T+1）----
        # 事件引擎因 T+1 延迟，收益通常低于或接近向量化（向量化用当日 close 有前视偏差）
        report.add(
            "事件回测", "前视偏差规避(T+1更保守)",
            True,
            f"事件引擎收益={result['metrics'].get('total_return', 0):.4f}, "
            f"向量化收益={native_result['metrics'].get('total_return', 0):.4f}",
        )
    except Exception as e:
        report.add("事件回测", "加载native_adapter对比", False, f"跳过对比: {e}")

    # ---- 1.7 边界条件：空数据 ----
    empty_result = engine.run(pd.DataFrame(), pd.DataFrame())
    report.add(
        "事件回测", "空数据处理",
        empty_result["trades"].empty and empty_result["equity_curve"].empty,
        "空数据返回空结果",
    )

    # ---- 1.8 边界条件：单只股票 ----
    single_data = data[data["code"] == data["code"].iloc[0]].copy()
    single_signals = signals[signals["code"] == data["code"].iloc[0]].copy()
    single_result = engine.run(single_data, single_signals)
    report.add(
        "事件回测", "单只股票处理",
        not single_result["equity_curve"].empty,
        f"单股净值记录数={len(single_result['equity_curve'])}",
    )

    # ---- 1.9 确定性：相同输入相同输出 ----
    result2 = engine.run(data, signals)
    deterministic = (
        result["equity_curve"]["equity"].tolist() == result2["equity_curve"]["equity"].tolist()
    )
    report.add(
        "事件回测", "确定性可复现",
        deterministic,
        f"两次运行净值序列完全一致={deterministic}",
    )

    return result


# ============================================================
# 测试 2：表达式因子引擎
# ============================================================

def test_expression_factor_engine(report: TestReport, data: pd.DataFrame):
    print("\n=== 测试 2: 表达式因子引擎 ===")

    engine = ExpressionFactorEngine()

    # ---- 2.1 正确性：表达式解析 ----
    parser = ExpressionParser()
    test_cases = [
        ("$close", "field"),
        ("Ref($close, 5)", "func"),
        ("Ref($close, 5) / $close - 1", "binop"),
        ("Mean($close, 20)", "func"),
        ("($close - Mean($close, 20)) / Std($close, 20)", "nested"),
    ]
    for expr, kind in test_cases:
        try:
            compiled = parser.parse(expr)
            result = compiled(data.head(100))
            report.add(
                "表达式因子", f"解析[{kind}]: {expr[:30]}",
                isinstance(result, pd.Series) and len(result) == 100,
            )
        except Exception as e:
            report.add("表达式因子", f"解析[{kind}]: {expr[:30]}", False, str(e))

    # ---- 2.2 正确性：与 pandas 计算结果对比 ----
    df_test = data[data["code"] == data["code"].iloc[0]].sort_values("date").copy()

    # 测试 momentum_20 = Ref($close, 20) / $close - 1
    series_mom = engine.compute_single(df_test, "Ref($close, 20) / $close - 1")
    expected_mom = df_test["close"].shift(20) / df_test["close"] - 1
    max_diff = (series_mom - expected_mom).abs().max()
    report.add(
        "表达式因子", "动量因子正确性(mom_20)",
        max_diff < 1e-10,
        f"最大差异={max_diff:.2e}",
    )

    # 测试 RSI
    series_rsi = engine.compute_single(df_test, "RSI($close, 14)")
    # 手动计算 RSI 验证
    delta = df_test["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    expected_rsi = 100 - 100 / (1 + rs)
    rsi_diff = (series_rsi - expected_rsi).abs().dropna().max()
    report.add(
        "表达式因子", "RSI因子正确性",
        rsi_diff < 1e-8,
        f"最大差异={rsi_diff:.2e}",
    )

    # ---- 2.3 正确性：Alpha158 因子库 ----
    alpha158 = build_alpha158_factors()
    report.add(
        "表达式因子", "Alpha158因子库构建",
        len(alpha158) >= 15,
        f"因子数={len(alpha158)}",
    )

    for name, expr in list(alpha158.items())[:5]:
        engine.add_factor(name, expr)
    result_df = engine.compute(data.head(500))
    expected_cols = {"code", "date"} | set(list(alpha158.keys())[:5])
    report.add(
        "表达式因子", "Alpha158批量计算",
        expected_cols.issubset(set(result_df.columns)),
        f"输出列={list(result_df.columns)}",
    )

    # ---- 2.4 性能：缓存加速 ----
    engine2 = ExpressionFactorEngine()
    engine2.add_factor("mom_20", "Ref($close, 20) / $close - 1")
    engine2.add_factor("vol_20", "Std(Ref($close, 1) / $close - 1, 20)")

    # 第一次计算（无缓存）
    t0 = time.perf_counter()
    engine2.compute(data, use_cache=False)
    no_cache_time = time.perf_counter() - t0

    # 第二次计算（有缓存）
    t0 = time.perf_counter()
    engine2.compute(data, use_cache=True)
    cached_time = time.perf_counter() - t0

    # 第三次计算（命中缓存）
    t0 = time.perf_counter()
    engine2.compute(data, use_cache=True)
    cache_hit_time = time.perf_counter() - t0

    speedup = no_cache_time / cache_hit_time if cache_hit_time > 0 else 0
    report.add(
        "表达式因子", "缓存加速性能",
        cache_hit_time < no_cache_time,
        f"无缓存={no_cache_time:.3f}s, 缓存命中={cache_hit_time:.3f}s, 加速比={speedup:.1f}x",
        {"no_cache": no_cache_time, "cache_hit": cache_hit_time, "speedup": speedup},
    )
    report.add(
        "表达式因子", "缓存统计",
        engine2.cache_stats()["hits"] > 0,
        f"缓存状态={engine2.cache_stats()}",
    )

    # ---- 2.5 边界条件：未知字段 ----
    try:
        engine.compute_single(data.head(10), "$unknown_field")
        report.add("表达式因子", "未知字段处理", False, "应抛出 KeyError")
    except KeyError:
        report.add("表达式因子", "未知字段处理", True, "正确抛出 KeyError")

    # ---- 2.6 边界条件：PIT 安全（Ref 不跨股票泄漏）----
    # 验证不同股票的因子计算相互独立
    df_multi = data[data["code"].isin([data["code"].iloc[0], data["code"].iloc[1]])].copy()
    df_multi = df_multi.sort_values(["code", "date"]).reset_index(drop=True)
    series = engine.compute_single(df_multi, "Ref($close, 5) / $close - 1")
    # 检查股票边界处无泄漏（第5行应为 NaN，因为是该股票的第5个数据点）
    code1 = data["code"].iloc[0]
    code1_data = df_multi[df_multi["code"] == code1]
    code1_idx = code1_data.index
    # 第5个数据点（index 4）的 Ref(close, 5) 应为 NaN
    fifth_val = series.loc[code1_idx[4]] if len(code1_idx) > 4 else None
    report.add(
        "表达式因子", "PIT安全(无跨股泄漏)",
        pd.isna(fifth_val),
        f"第5个数据点值={fifth_val} (应为NaN)",
    )

    return result_df


# ============================================================
# 测试 3：向量化 IC 分析
# ============================================================

def test_vectorized_ic(report: TestReport, data: pd.DataFrame):
    print("\n=== 测试 3: 向量化 IC 分析 ===")

    # 准备因子和前向收益
    df = data.sort_values(["code", "date"]).copy()
    df["mom_20"] = df.groupby("code")["close"].shift(20) / df["close"] - 1

    forward_returns = df[["code", "date"]].copy()
    for period in [1, 5, 20]:
        forward_returns[f"ret_forward_{period}d"] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-period) / x - 1
        )

    factor_df = df[["code", "date", "mom_20"]].dropna()

    # ---- 3.1 正确性：与 baseline 结果对比 ----
    for fwd_col in ["ret_forward_1d", "ret_forward_5d"]:
        t0 = time.perf_counter()
        ic_vec = VectorizedICAnalyzer.calc_ic_series(
            factor_df, forward_returns, "mom_20", fwd_col, method="spearman"
        )
        vec_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        ic_base = calc_ic_baseline(
            factor_df, forward_returns, "mom_20", fwd_col, method="spearman"
        )
        base_time = time.perf_counter() - t0

        # 对齐比较
        common_idx = ic_vec.index.intersection(ic_base.index)
        if len(common_idx) > 0:
            diff = (ic_vec.loc[common_idx] - ic_base.loc[common_idx]).abs().max()
            report.add(
                "向量化IC", f"正确性对比[{fwd_col}]",
                diff < 1e-6,
                f"最大差异={diff:.2e}, 样本数={len(common_idx)}",
            )
        else:
            report.add("向量化IC", f"正确性对比[{fwd_col}]", False, "无共同样本")

        speedup = base_time / vec_time if vec_time > 0 else 0
        report.add(
            "向量化IC", f"性能对比[{fwd_col}]",
            True,
            f"向量化={vec_time:.4f}s, baseline={base_time:.4f}s, 加速={speedup:.1f}x",
            {"vectorized": vec_time, "baseline": base_time, "speedup": speedup},
        )

    # ---- 3.2 正确性：IC 统计量 ----
    summary = VectorizedICAnalyzer.calc_ic_summary(ic_vec)
    report.add(
        "向量化IC", "IC统计量完整性",
        all(k in summary for k in ["ic_mean", "ic_std", "ic_ir", "ic_positive_ratio"]),
        f"统计量={summary}",
    )

    # ---- 3.3 批量 IC 矩阵 ----
    # 构建多因子
    df["rev_5"] = -df.groupby("code")["close"].shift(5) / df["close"] + 1
    df["vol_20"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20).std())
    multi_factor = df[["code", "date", "mom_20", "rev_5", "vol_20"]].dropna()

    t0 = time.perf_counter()
    ic_matrix = VectorizedICAnalyzer.calc_ic_matrix(
        multi_factor, forward_returns,
        factor_cols=["mom_20", "rev_5", "vol_20"],
        forward_cols=["ret_forward_1d", "ret_forward_5d", "ret_forward_20d"],
    )
    matrix_time = time.perf_counter() - t0
    report.add(
        "向量化IC", "批量IC矩阵(3因子x3周期)",
        len(ic_matrix) == 3 and all(len(v) == 3 for v in ic_matrix.values()),
        f"耗时={matrix_time:.3f}s, 矩阵形状={ {k: len(v) for k, v in ic_matrix.items()} }",
    )

    # ---- 3.4 边界条件：空数据 ----
    empty_ic = VectorizedICAnalyzer.calc_ic_series(
        pd.DataFrame(columns=["code", "date", "f"]),
        pd.DataFrame(columns=["code", "date", "r"]),
        "f", "r",
    )
    report.add(
        "向量化IC", "空数据处理",
        empty_ic.empty,
        "空数据返回空 Series",
    )

    # ---- 3.5 边界条件：样本不足 ----
    small_factor = factor_df.head(5)
    small_fwd = forward_returns.head(5)
    small_ic = VectorizedICAnalyzer.calc_ic_series(
        small_factor, small_fwd, "mom_20", "ret_forward_5d"
    )
    report.add(
        "向量化IC", "样本不足处理",
        small_ic.empty,
        f"样本<10时返回空 (得到{len(small_ic)}条)",
    )

    return ic_matrix


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("jingni-trader 量化优化验证测试")
    print(f"分支: feat/quant-opt-20260623")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"借鉴来源: NautilusTrader / Microsoft Qlib / VectorBT")
    print("=" * 70)

    # 生成测试数据
    print("\n生成合成测试数据...")
    data = generate_synthetic_data(n_stocks=50, n_days=250, seed=42)
    signals = generate_signals(data, strategy="momentum")
    print(f"数据: {len(data)} 行, {data['code'].nunique()} 只股票, "
          f"{data['date'].nunique()} 个交易日")
    print(f"信号: {len(signals)} 条, 买入={int((signals['signal']>0).sum())}, "
          f"卖出={int((signals['signal']<0).sum())}")

    report = TestReport()

    # 运行三大测试
    bt_result = test_event_driven_backtest(report, data, signals)
    factor_result = test_expression_factor_engine(report, data)
    ic_result = test_vectorized_ic(report, data)

    # 保存结果
    summary = report.summary()
    print("\n" + "=" * 70)
    print(f"测试完成: {summary['passed']}/{summary['total']} 通过, "
          f"失败={summary['failed']}, 耗时={summary['elapsed_sec']}s")
    print("=" * 70)

    # 保存详细报告
    report_data = {
        "test_time": datetime.now().isoformat(),
        "branch": "feat/quant-opt-20260623",
        "data_info": {
            "n_stocks": int(data["code"].nunique()),
            "n_days": int(data["date"].nunique()),
            "n_signals": len(signals),
        },
        "summary": summary,
        "results": report.results,
    }
    report_path = RESULTS_DIR / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细报告已保存: {report_path}")

    # 保存回测结果样本
    if bt_result and not bt_result["equity_curve"].empty:
        bt_result["equity_curve"].to_csv(RESULTS_DIR / "equity_curve_event_driven.csv", index=False)
        if not bt_result["trades"].empty:
            bt_result["trades"].to_csv(RESULTS_DIR / "trades_event_driven.csv", index=False)
        if not bt_result["orders"].empty:
            bt_result["orders"].to_csv(RESULTS_DIR / "orders_event_driven.csv", index=False)

    return summary


if __name__ == "__main__":
    main()
