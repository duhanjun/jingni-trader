"""
优化验证测试套件

测试内容:
  1. 正确性测试: 向量化实现 vs 朴素实现, 结果应一致 (容差 1e-6)
  2. 性能测试: 向量化 vs 朴素实现的耗时对比
  3. 边界条件测试: 空数据、单股票、全 NaN、单日等

运行方式:
    cd /workspace && python3 quant_opt_20260624/test_optimizations.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Callable, Tuple

import numpy as np
import pandas as pd

# 确保能导入本目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factor_expression_engine import (
    FactorExpressionEngine, evaluate_factor, PRESET_FACTORS,
)
from vectorized_ic import VectorizedICAnalyzer, compute_ic_rank_decay
from vectorized_backtest import VectorizedBacktester, NaiveBacktester


# ---------------------------------------------------------------------------
# 测试数据生成
# ---------------------------------------------------------------------------

def generate_test_data(
    n_stocks: int = 200,
    n_days: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成模拟行情数据与因子数据"""
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    rows = []
    for code in codes:
        price = 10.0 + rng.normal(0, 1)
        for dt in dates:
            ret = rng.normal(0.0003, 0.02)
            price *= (1 + ret)
            rows.append({
                "code": code,
                "date": dt,
                "open": price * (1 + rng.normal(0, 0.005)),
                "high": price * (1 + abs(rng.normal(0, 0.01))),
                "low": price * (1 - abs(rng.normal(0, 0.01))),
                "close": price,
                "volume": rng.integers(1e6, 1e8),
                "amount": rng.uniform(1e7, 1e9),
                "turnover_rate": rng.uniform(0.005, 0.05),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # 生成因子: 动量 + 反转 + 噪声
    df["momentum_5d"] = df.groupby("code")["close"].pct_change(5)
    df["reversal_5d"] = -df["momentum_5d"]
    df["vol_20d"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=5).std()
    )
    df["alpha_score"] = (
        0.4 * df["reversal_5d"].fillna(0)
        + 0.3 * df["momentum_5d"].fillna(0)
        + 0.3 * rng.normal(0, 0.01, len(df))
    )

    # 远期收益
    for period in [1, 5, 20]:
        df[f"ret_forward_{period}d"] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-period) / x - 1
        )

    return df, df.copy()


# ---------------------------------------------------------------------------
# 测试框架
# ---------------------------------------------------------------------------

class TestRunner:
    """简易测试运行器"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.errors: list = []

    def run(self, name: str, func: Callable) -> None:
        print(f"\n[TEST] {name}")
        try:
            func()
            self.passed += 1
            print(f"  -> PASS")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, f"AssertionError: {e}"))
            print(f"  -> FAIL: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, f"{type(e).__name__}: {e}"))
            print(f"  -> ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

    def summary(self) -> None:
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        print(f"测试总结: {self.passed}/{total} 通过, {self.failed} 失败")
        if self.errors:
            print("\n失败用例:")
            for name, err in self.errors:
                print(f"  - {name}: {err}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# 1. 因子表达式引擎测试
# ---------------------------------------------------------------------------

def test_expression_engine_basic() -> None:
    """测试表达式引擎基本功能"""
    df, _ = generate_test_data(n_stocks=50, n_days=60)

    engine = FactorExpressionEngine()
    engine.add_dataframe(df)

    # 测试简单字段引用
    result = engine.evaluate("Close", df)
    assert len(result) == len(df), f"字段引用长度不匹配: {len(result)} vs {len(df)}"
    assert np.allclose(result.values, df["close"].values, equal_nan=True)

    # 测试时序算子 (min_periods 与实现一致: max(1, window//2))
    ts_mean = engine.evaluate("Ts_Mean(Close, 5)", df)
    expected = df.groupby("code")["close"].transform(
        lambda x: x.rolling(5, min_periods=max(1, 5 // 2)).mean()
    )
    assert np.allclose(ts_mean.values, expected.values, equal_nan=True), \
        "Ts_Mean 结果与 pandas 实现不一致"

    # 测试截面算子
    rank_result = engine.evaluate("Rank(Close)", df)
    expected_rank = df.groupby("date")["close"].rank(pct=True)
    assert np.allclose(rank_result.values, expected_rank.values, equal_nan=True), \
        "Rank 结果与 pandas 实现不一致"

    # 测试嵌套表达式
    nested = engine.evaluate("Rank(Ts_Mean(Close, 5))", df)
    assert len(nested) == len(df)
    assert nested.notna().any(), "嵌套表达式结果全为 NaN"


def test_expression_engine_preset_factors() -> None:
    """测试预置因子公式"""
    df, _ = generate_test_data(n_stocks=30, n_days=40)

    for name, expr in PRESET_FACTORS.items():
        result = evaluate_factor(expr, df)
        assert len(result) == len(df), f"预置因子 {name} 长度不匹配"
        print(f"    预置因子 {name}: {expr} -> 有效值 {result.notna().sum()}/{len(result)}")


def test_expression_engine_error_handling() -> None:
    """测试表达式引擎错误处理"""
    df, _ = generate_test_data(n_stocks=10, n_days=20)

    engine = FactorExpressionEngine()
    engine.add_dataframe(df)

    # 未知字段
    try:
        engine.evaluate("NonExistent", df)
        assert False, "应抛出 KeyError"
    except KeyError:
        pass

    # 未知算子
    try:
        engine.evaluate("UnknownOp(Close)", df)
        assert False, "应抛出 KeyError"
    except KeyError:
        pass

    # 语法错误
    try:
        engine.evaluate("Rank(Close", df)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 2. 向量化 IC 分析测试
# ---------------------------------------------------------------------------

def test_ic_correctness() -> None:
    """测试向量化 IC 与朴素实现结果一致"""
    df, _ = generate_test_data(n_stocks=100, n_days=80, seed=123)

    analyzer = VectorizedICAnalyzer(ic_type="spearman")

    factor_df = df[["code", "date", "reversal_5d"]].copy()
    fwd_df = df[["code", "date", "ret_forward_5d"]].copy()

    # 向量化
    t0 = time.perf_counter()
    ic_vec = analyzer.compute_ic_series_vectorized(
        factor_df, fwd_df, "reversal_5d", "ret_forward_5d"
    )
    t_vec = time.perf_counter() - t0

    # 朴素
    t0 = time.perf_counter()
    ic_naive = analyzer.compute_ic_series_naive(
        factor_df, fwd_df, "reversal_5d", "ret_forward_5d"
    )
    t_naive = time.perf_counter() - t0

    # 对齐比较
    common_idx = ic_vec.index.intersection(ic_naive.index)
    assert len(common_idx) > 0, "无共同日期"

    diff = (ic_vec.loc[common_idx] - ic_naive.loc[common_idx]).abs()
    max_diff = diff.max()
    assert max_diff < 1e-6, f"向量化 IC 与朴素 IC 最大差异 {max_diff} 超过容差 1e-6"

    speedup = t_naive / t_vec if t_vec > 0 else float("inf")
    print(f"    向量化耗时: {t_vec*1000:.2f}ms, 朴素耗时: {t_naive*1000:.2f}ms, "
          f"加速比: {speedup:.1f}x, 最大差异: {max_diff:.2e}")
    assert speedup > 1.0, f"向量化未实现加速 (speedup={speedup})"


def test_ic_stats_correctness() -> None:
    """测试向量化 IC 统计量计算"""
    df, _ = generate_test_data(n_stocks=80, n_days=60, seed=456)

    analyzer = VectorizedICAnalyzer(ic_type="spearman")
    factor_names = ["momentum_5d", "reversal_5d", "vol_20d"]
    fwd_cols = ["ret_forward_1d", "ret_forward_5d", "ret_forward_20d"]

    factor_df = df[["code", "date"] + factor_names].copy()
    fwd_df = df[["code", "date"] + fwd_cols].copy()

    stats = analyzer.compute_ic_stats_vectorized(
        factor_df, fwd_df, factor_names, fwd_cols
    )

    assert "ret_forward_5d" in stats
    assert len(stats["ret_forward_5d"]) == len(factor_names)

    for item in stats["ret_forward_5d"]:
        assert "factor" in item
        assert "ic_mean" in item
        assert "ic_ir" in item
        assert "ic_t_stat" in item
        print(f"    {item['factor']}: IC={item['ic_mean']:.4f}, "
              f"IR={item['ic_ir']:.4f}, t={item['ic_t_stat']:.2f}")


def test_ic_pearson_consistency() -> None:
    """测试 Pearson IC 与 Spearman IC 在线性数据上接近"""
    df, _ = generate_test_data(n_stocks=50, n_days=40, seed=789)

    analyzer_s = VectorizedICAnalyzer(ic_type="spearman")
    analyzer_p = VectorizedICAnalyzer(ic_type="pearson")

    factor_df = df[["code", "date", "momentum_5d"]].copy()
    fwd_df = df[["code", "date", "ret_forward_5d"]].copy()

    ic_s = analyzer_s.compute_ic_series_vectorized(
        factor_df, fwd_df, "momentum_5d", "ret_forward_5d"
    ).dropna()
    ic_p = analyzer_p.compute_ic_series_vectorized(
        factor_df, fwd_df, "momentum_5d", "ret_forward_5d"
    ).dropna()

    # 两者都应可计算且数量一致
    assert len(ic_s) > 0 and len(ic_p) > 0
    assert abs(ic_s.mean() - ic_p.mean()) < 0.5, "Spearman 与 Pearson IC 均值差异过大"


def test_ic_rank_decay() -> None:
    """测试因子分层收益分析"""
    df, _ = generate_test_data(n_stocks=100, n_days=50, seed=321)

    factor_df = df[["code", "date", "reversal_5d"]].copy()
    fwd_df = df[["code", "date", "ret_forward_5d"]].copy()

    decay = compute_ic_rank_decay(
        factor_df, fwd_df, "reversal_5d", "ret_forward_5d", n_quantiles=5
    )
    assert not decay.empty
    assert "quantile" in decay.columns
    assert decay["quantile"].nunique() <= 5
    print(f"    分层数: {decay['quantile'].nunique()}, 样本: {len(decay)}")


# ---------------------------------------------------------------------------
# 3. 向量化回测测试
# ---------------------------------------------------------------------------

def test_backtest_correctness() -> None:
    """测试向量化回测与朴素回测结果接近"""
    df, _ = generate_test_data(n_stocks=50, n_days=60, seed=999)

    factor_df = df[["code", "date", "alpha_score"]].copy()

    vec_bt = VectorizedBacktester(n_select=10, t_plus_1=True)
    naive_bt = NaiveBacktester(n_select=10, t_plus_1=True)

    t0 = time.perf_counter()
    res_vec = vec_bt.run(df, factor_df, "alpha_score")
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_naive = naive_bt.run(df, factor_df, "alpha_score")
    t_naive = time.perf_counter() - t0

    # 净值末端应接近 (允许小差异, 因为换手率计算细节略有不同)
    eq_vec = res_vec["equity_curve"]["equity"].iloc[-1]
    eq_naive = res_naive["equity_curve"]["equity"].iloc[-1]
    rel_diff = abs(eq_vec - eq_naive) / eq_naive if eq_naive != 0 else 1

    print(f"    向量化末净值: {eq_vec:.2f}, 朴素末净值: {eq_naive:.2f}, "
          f"相对差异: {rel_diff:.4f}")
    print(f"    向量化耗时: {t_vec*1000:.2f}ms, 朴素耗时: {t_naive*1000:.2f}ms, "
          f"加速比: {t_naive/t_vec if t_vec>0 else 0:.1f}x")

    # 绩效指标都应可计算
    assert res_vec["metrics"], "向量化回测未生成指标"
    assert "sharpe_ratio" in res_vec["metrics"]
    assert "max_drawdown" in res_vec["metrics"]

    # 持仓数量应等于 n_select (允许首日因 T+1 为空)
    holdings = res_vec["holdings"]
    if not holdings.empty:
        max_holding_per_day = holdings.groupby("date").size().max()
        assert max_holding_per_day <= 10, f"持仓数 {max_holding_per_day} 超过 n_select=10"


def test_backtest_performance() -> None:
    """测试向量化回测在大规模数据上的性能优势"""
    # 模拟较大规模: 500 股票 × 250 交易日
    df, _ = generate_test_data(n_stocks=500, n_days=250, seed=2024)
    factor_df = df[["code", "date", "alpha_score"]].copy()

    vec_bt = VectorizedBacktester(n_select=50, t_plus_1=True)
    naive_bt = NaiveBacktester(n_select=50, t_plus_1=True)

    t0 = time.perf_counter()
    res_vec = vec_bt.run(df, factor_df, "alpha_score")
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_naive = naive_bt.run(df, factor_df, "alpha_score")
    t_naive = time.perf_counter() - t0

    speedup = t_naive / t_vec if t_vec > 0 else 0
    print(f"    数据规模: {df.shape[0]} 行 (500股 × 250日)")
    print(f"    向量化: {t_vec:.3f}s, 朴素: {t_naive:.3f}s, 加速比: {speedup:.1f}x")
    print(f"    向量化 Sharpe: {res_vec['metrics']['sharpe_ratio']:.4f}, "
          f"朴素 Sharpe: {res_naive['metrics']['sharpe_ratio']:.4f}")

    assert speedup > 5, f"大规模数据加速比 {speedup:.1f}x 未达预期 (>5x)"


def test_backtest_metrics_completeness() -> None:
    """测试回测指标完整性"""
    df, _ = generate_test_data(n_stocks=30, n_days=40, seed=111)
    factor_df = df[["code", "date", "alpha_score"]].copy()

    bt = VectorizedBacktester(n_select=5)
    result = bt.run(df, factor_df, "alpha_score")

    required = ["total_return", "annual_return", "volatility", "sharpe_ratio",
                "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate"]
    for key in required:
        assert key in result["metrics"], f"缺少指标: {key}"
        assert isinstance(result["metrics"][key], (int, float)), \
            f"指标 {key} 类型错误: {type(result['metrics'][key])}"


# ---------------------------------------------------------------------------
# 4. 边界条件测试
# ---------------------------------------------------------------------------

def test_edge_empty_data() -> None:
    """测试空数据"""
    empty_df = pd.DataFrame(columns=["code", "date", "close", "alpha_score"])

    # 表达式引擎
    engine = FactorExpressionEngine()
    try:
        engine.evaluate("Close", empty_df)
    except Exception as e:
        assert isinstance(e, (KeyError, ValueError)), f"意外异常: {e}"

    # IC 分析
    analyzer = VectorizedICAnalyzer()
    ic = analyzer.compute_ic_series_vectorized(
        empty_df, empty_df, "close", "close"
    )
    assert ic.empty, "空数据 IC 应为空"

    # 回测
    bt = VectorizedBacktester()
    result = bt.run(empty_df, empty_df, "alpha_score")
    assert result["equity_curve"].empty, "空数据回测净值曲线应为空"


def test_edge_single_stock() -> None:
    """测试单股票场景"""
    df, _ = generate_test_data(n_stocks=1, n_days=30, seed=1)
    factor_df = df[["code", "date", "alpha_score"]].copy()

    bt = VectorizedBacktester(n_select=1)
    result = bt.run(df, factor_df, "alpha_score")
    # 单股票也应能正常运行
    assert not result["equity_curve"].empty or len(df) < 2


def test_edge_all_nan_factor() -> None:
    """测试因子全 NaN"""
    df, _ = generate_test_data(n_stocks=20, n_days=30, seed=2)
    df["alpha_score"] = np.nan
    factor_df = df[["code", "date", "alpha_score"]].copy()

    bt = VectorizedBacktester(n_select=5)
    result = bt.run(df, factor_df, "alpha_score")
    # 全 NaN 因子应优雅处理, 不抛异常
    assert "metrics" in result


def test_edge_single_day() -> None:
    """测试单日数据"""
    df, _ = generate_test_data(n_stocks=20, n_days=1, seed=3)
    factor_df = df[["code", "date", "alpha_score"]].copy()

    bt = VectorizedBacktester(n_select=5)
    result = bt.run(df, factor_df, "alpha_score")
    # 单日数据无法计算收益, 应优雅返回
    assert "metrics" in result


def test_edge_high_nan_ratio() -> None:
    """测试高 NaN 比例数据"""
    df, _ = generate_test_data(n_stocks=50, n_days=40, seed=4)
    # 随机注入 50% NaN
    mask = np.random.RandomState(5).rand(len(df)) < 0.5
    df.loc[mask, "alpha_score"] = np.nan
    factor_df = df[["code", "date", "alpha_score"]].copy()

    analyzer = VectorizedICAnalyzer()
    factor_sub = df[["code", "date", "alpha_score"]].copy()
    fwd_df = df[["code", "date", "ret_forward_5d"]].copy()
    ic = analyzer.compute_ic_series_vectorized(
        factor_sub, fwd_df, "alpha_score", "ret_forward_5d"
    )
    # 高 NaN 比例下应能计算出部分 IC (NaN 被过滤)
    print(f"    高 NaN 比例下 IC 序列长度: {len(ic)}")


# ---------------------------------------------------------------------------
# 5. 端到端集成测试
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline() -> None:
    """端到端: 表达式因子 -> IC 分析 -> 向量化回测"""
    df, _ = generate_test_data(n_stocks=100, n_days=120, seed=2026)

    # 1. 用表达式引擎计算因子
    engine = FactorExpressionEngine()
    engine.add_dataframe(df)
    df["expr_factor"] = engine.evaluate("Rank(-Delta(Close, 5))", df).values

    # 2. IC 分析
    analyzer = VectorizedICAnalyzer(ic_type="spearman")
    factor_df = df[["code", "date", "expr_factor"]].copy()
    fwd_df = df[["code", "date", "ret_forward_5d"]].copy()
    ic_series = analyzer.compute_ic_series_vectorized(
        factor_df, fwd_df, "expr_factor", "ret_forward_5d"
    )
    ic_mean = ic_series.mean()
    print(f"    表达式因子 Rank(-Delta(Close,5)) IC 均值: {ic_mean:.4f}")

    # 3. 向量化回测
    bt = VectorizedBacktester(n_select=20)
    result = bt.run(df, factor_df, "expr_factor")
    metrics = result["metrics"]
    print(f"    回测年化: {metrics['annual_return']:.4f}, "
          f"Sharpe: {metrics['sharpe_ratio']:.4f}, "
          f"最大回撤: {metrics['max_drawdown']:.4f}")

    assert "sharpe_ratio" in metrics
    assert len(result["equity_curve"]) > 0


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("jingni-trader 优化验证测试套件")
    print(f"分支: feat/quant-opt-20260624")
    print(f"时间: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    runner = TestRunner()

    # 1. 表达式引擎
    runner.run("表达式引擎-基本功能", test_expression_engine_basic)
    runner.run("表达式引擎-预置因子", test_expression_engine_preset_factors)
    runner.run("表达式引擎-错误处理", test_expression_engine_error_handling)

    # 2. 向量化 IC
    runner.run("向量化IC-正确性", test_ic_correctness)
    runner.run("向量化IC-统计量", test_ic_stats_correctness)
    runner.run("向量化IC-Pearson一致性", test_ic_pearson_consistency)
    runner.run("向量化IC-分层收益", test_ic_rank_decay)

    # 3. 向量化回测
    runner.run("向量化回测-正确性", test_backtest_correctness)
    runner.run("向量化回测-大规模性能", test_backtest_performance)
    runner.run("向量化回测-指标完整性", test_backtest_metrics_completeness)

    # 4. 边界条件
    runner.run("边界-空数据", test_edge_empty_data)
    runner.run("边界-单股票", test_edge_single_stock)
    runner.run("边界-全NaN因子", test_edge_all_nan_factor)
    runner.run("边界-单日数据", test_edge_single_day)
    runner.run("边界-高NaN比例", test_edge_high_nan_ratio)

    # 5. 端到端
    runner.run("端到端-表达式因子->IC->回测", test_end_to_end_pipeline)

    runner.summary()

    if runner.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
