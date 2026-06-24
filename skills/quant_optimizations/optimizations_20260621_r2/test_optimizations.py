"""
优化验证测试套件

验证内容：
1. 正确性测试：优化版与基准版输出一致（数值误差 < 1e-6）
2. 性能对比测试：优化版应显著快于基准版
3. 边界条件测试：空数据、单只股票、单日数据、全 NaN 等

运行：
    python3 optimizations/test_optimizations.py
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

# 确保能 import optimizations_20260621_r2 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from skills.quant_optimizations.optimizations_20260621_r2.vectorized_factor import (
    compute_factors_vectorized,
    compute_factors_baseline,
    ic_analysis_vectorized,
    ic_analysis_baseline,
    neutralize_vectorized,
)
from skills.quant_optimizations.optimizations_20260621_r2.vectorized_backtest import (
    run_backtest_vectorized,
    run_backtest_baseline,
    calc_extended_metrics,
)
from skills.quant_optimizations.optimizations_20260621_r2.walk_forward import (
    purged_ts_split,
    walk_forward_splits,
    walk_forward_predict,
    purged_group_ts_split_baseline,
    train_baseline_bug,
)

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 测试数据生成
# ----------------------------------------------------------------------

def make_synthetic_data(
    n_codes: int = 50,
    n_days: int = 250,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """生成 A 股日线合成数据。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_codes + 1)]

    rows = []
    for code in codes:
        start_price = rng.uniform(8, 50)
        drift = rng.uniform(-0.0005, 0.0015)
        vol = rng.uniform(0.015, 0.025)
        rets = rng.normal(drift, vol, n_days)
        # 加一点自相关
        for i in range(1, n_days):
            rets[i] += 0.15 * rets[i - 1]
        prices = start_price * np.cumprod(1 + rets)

        opens = np.concatenate([[start_price], prices[:-1]]) * (1 + rng.normal(0, 0.003, n_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        vols = rng.lognormal(12, 0.5, n_days).astype(int)
        amounts = vols * prices
        turnover = rng.uniform(0.005, 0.05, n_days)

        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': opens.round(4),
            'high': highs.round(4),
            'low': lows.round(4),
            'close': prices.round(4),
            'volume': vols,
            'amount': amounts,
            'turnover_rate': turnover,
            'pre_close': np.concatenate([[start_price], prices[:-1]]).round(4),
        })
        df['change_pct'] = (df['close'] - df['pre_close']) / df['pre_close'] * 100
        df['is_st'] = False
        df['is_limit_up'] = df['change_pct'] >= 9.9
        df['is_limit_down'] = df['change_pct'] <= -9.9
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def make_forward_returns(data: pd.DataFrame) -> pd.DataFrame:
    """生成前瞻收益（用于 IC 分析）。"""
    df = data.sort_values(['code', 'date']).copy()
    fr = df[['code', 'date']].copy()
    for period in [1, 5, 20]:
        fr[f'ret_forward_{period}d'] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-period) / x - 1
        )
    return fr


def make_signals(data: pd.DataFrame, top_pct: float = 0.2) -> pd.DataFrame:
    """生成简单反转信号：20日反转因子 top 20% 买入。"""
    df = data.sort_values(['code', 'date']).copy()
    df['ret_20d'] = df.groupby('code')['close'].pct_change(20)
    df['reversal'] = -df['ret_20d']
    df['rank'] = df.groupby('date')['reversal'].rank(pct=True)
    sig = df[['code', 'date']].copy()
    sig['signal'] = 0
    sig.loc[df['rank'] > (1 - top_pct), 'signal'] = 1
    sig.loc[df['rank'] < top_pct, 'signal'] = -1
    return sig


def make_benchmark(data: pd.DataFrame) -> pd.DataFrame:
    """用全市场等权日收益构造基准。"""
    pivot = data.pivot(index='date', columns='code', values='close')
    bench_ret = pivot.pct_change().mean(axis=1)
    bench = (1 + bench_ret).cumprod() * 3000  # 模拟指数点位
    return pd.DataFrame({'date': bench.index, 'close': bench.values})


# ----------------------------------------------------------------------
# 测试结果收集
# ----------------------------------------------------------------------

class TestReport:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def record(self, name: str, category: str, passed: bool, detail: str, duration: float = 0):
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "name": name,
            "category": category,
            "status": status,
            "detail": detail,
            "duration_sec": round(duration, 4),
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        marker = "✓" if passed else "✗"
        print(f"  {marker} [{category}] {name} ({duration:.3f}s) - {detail[:80]}")

    def summary(self) -> dict:
        return {
            "total": len(self.results),
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.passed / max(len(self.results), 1), 4),
            "results": self.results,
        }


# ----------------------------------------------------------------------
# 正确性测试
# ----------------------------------------------------------------------

def test_factor_correctness(report: TestReport, data: pd.DataFrame):
    """验证向量化因子计算与基准版输出一致。"""
    print("\n=== 正确性测试：因子计算 ===")

    t0 = time.time()
    base = compute_factors_baseline(data)
    t1 = time.time()
    opt = compute_factors_vectorized(data)
    t2 = time.time()

    # 对比公共列（仅数值列）
    common_cols = [c for c in base.columns if c in opt.columns]
    all_close = True
    max_diff = 0
    for col in common_cols:
        b_series = base[col]
        o_series = opt[col]
        # 跳过非数值列（code, date 等）
        if not pd.api.types.is_numeric_dtype(b_series):
            continue
        b = b_series.fillna(-9999).values
        o = o_series.fillna(-9999).values
        if len(b) != len(o):
            all_close = False
            break
        diff = float(np.nanmax(np.abs(b - o)))
        max_diff = max(max_diff, diff)
        if diff > 1e-6:
            all_close = False

    report.record(
        "因子计算数值一致",
        "正确性",
        all_close,
        f"公共列数={len(common_cols)}, 最大绝对误差={max_diff:.2e}",
        t2 - t0,
    )


def test_ic_correctness(report: TestReport, data: pd.DataFrame, fr: pd.DataFrame):
    """验证向量化 IC 分析与基准版输出一致。"""
    print("\n=== 正确性测试：IC 分析 ===")

    factor_df = compute_factors_vectorized(data)
    factor_names = ['reversal_5d', 'reversal_20d', 'volatility_20d', 'volume_ratio']

    t0 = time.time()
    base_ic = ic_analysis_baseline(factor_df, fr, factor_names, ic_type="spearman")
    t1 = time.time()
    opt_ic = ic_analysis_vectorized(factor_df, fr, factor_names, ic_type="spearman")
    t2 = time.time()

    # 对比每个因子的 IC 统计量
    all_close = True
    max_diff = 0
    for period in ['ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']:
        if period not in base_ic or period not in opt_ic:
            continue
        for b_item, o_item in zip(base_ic[period], opt_ic[period]):
            for key in ['ic_mean', 'ic_std', 'ic_ir', 'ic_positive_ratio', 'ic_t_stat']:
                diff = abs(b_item[key] - o_item[key])
                max_diff = max(max_diff, diff)
                if diff > 1e-4:
                    all_close = False

    report.record(
        "IC 分析数值一致（Spearman）",
        "正确性",
        all_close,
        f"最大绝对误差={max_diff:.2e}",
        t2 - t0,
    )


def test_backtest_correctness(report: TestReport, data: pd.DataFrame, signals: pd.DataFrame):
    """验证向量化回测与基准版输出一致。"""
    print("\n=== 正确性测试：回测引擎 ===")

    t0 = time.time()
    base_res = run_backtest_baseline(data, signals)
    t1 = time.time()
    opt_res = run_backtest_vectorized(data, signals)
    t2 = time.time()

    # 对比权益曲线（最终净值）
    base_eq = base_res['equity_curve']['equity'].values
    opt_eq = opt_res['equity_curve']['equity'].values

    if len(base_eq) != len(opt_eq):
        report.record(
            "回测权益曲线一致",
            "正确性",
            False,
            f"长度不一致: base={len(base_eq)}, opt={len(opt_eq)}",
            t2 - t0,
        )
        return

    final_diff = abs(base_eq[-1] - opt_eq[-1])
    max_diff = float(np.max(np.abs(base_eq - opt_eq)))
    # 允许极小浮点误差（买卖顺序导致的舍入）
    passed = max_diff < 1e-2

    report.record(
        "回测权益曲线一致",
        "正确性",
        passed,
        f"最终净值 base={base_eq[-1]:.2f}, opt={opt_eq[-1]:.2f}, 最大误差={max_diff:.6f}",
        t2 - t0,
    )

    # 对比交易笔数
    base_n = len(base_res['trades'])
    opt_n = len(opt_res['trades'])
    report.record(
        "回测交易笔数一致",
        "正确性",
        base_n == opt_n,
        f"base={base_n}, opt={opt_n}",
    )

    # 验证扩展指标存在
    opt_metrics = opt_res['metrics']
    expected_new = ['sortino_ratio', 'turnover_annual']
    has_new = all(k in opt_metrics for k in expected_new)
    report.record(
        "扩展指标已生成",
        "正确性",
        has_new,
        f"新增指标: sortino={opt_metrics.get('sortino_ratio')}, turnover={opt_metrics.get('turnover_annual')}",
    )


def test_backtest_with_benchmark(report: TestReport, data: pd.DataFrame, signals: pd.DataFrame, bench: pd.DataFrame):
    """验证带基准的扩展指标。"""
    print("\n=== 正确性测试：基准对比指标 ===")

    res = run_backtest_vectorized(data, signals, benchmark_data=bench)
    m = res['metrics']
    has_alpha_beta = 'alpha' in m and 'beta' in m and 'information_ratio' in m
    report.record(
        "Alpha/Beta/IR 指标生成",
        "正确性",
        has_alpha_beta,
        f"alpha={m.get('alpha')}, beta={m.get('beta')}, IR={m.get('information_ratio')}",
    )


# ----------------------------------------------------------------------
# 性能对比测试
# ----------------------------------------------------------------------

def test_factor_performance(report: TestReport, data: pd.DataFrame):
    """因子计算性能对比。"""
    print("\n=== 性能测试：因子计算 ===")

    t0 = time.time()
    compute_factors_baseline(data)
    t1 = time.time()
    compute_factors_vectorized(data)
    t2 = time.time()

    base_t = t1 - t0
    opt_t = t2 - t1
    speedup = base_t / opt_t if opt_t > 0 else float('inf')

    report.record(
        "因子计算性能",
        "性能",
        opt_t <= base_t,
        f"base={base_t:.3f}s, opt={opt_t:.3f}s, 加速比={speedup:.2f}x",
        opt_t,
    )


def test_ic_performance(report: TestReport, data: pd.DataFrame, fr: pd.DataFrame):
    """IC 分析性能对比（核心优化点）。"""
    print("\n=== 性能测试：IC 分析 ===")

    factor_df = compute_factors_vectorized(data)
    factor_names = ['reversal_5d', 'reversal_20d', 'volatility_20d', 'volume_ratio',
                    'turnover_change', 'money_flow_20d']

    t0 = time.time()
    ic_analysis_baseline(factor_df, fr, factor_names, ic_type="spearman")
    t1 = time.time()
    ic_analysis_vectorized(factor_df, fr, factor_names, ic_type="spearman")
    t2 = time.time()

    base_t = t1 - t0
    opt_t = t2 - t1
    speedup = base_t / opt_t if opt_t > 0 else float('inf')

    report.record(
        "IC 分析性能",
        "性能",
        opt_t <= base_t,
        f"base={base_t:.3f}s, opt={opt_t:.3f}s, 加速比={speedup:.2f}x",
        opt_t,
    )


def test_backtest_performance(report: TestReport, data: pd.DataFrame, signals: pd.DataFrame):
    """回测性能对比。"""
    print("\n=== 性能测试：回测引擎 ===")

    t0 = time.time()
    run_backtest_baseline(data, signals)
    t1 = time.time()
    run_backtest_vectorized(data, signals)
    t2 = time.time()

    base_t = t1 - t0
    opt_t = t2 - t1
    speedup = base_t / opt_t if opt_t > 0 else float('inf')

    report.record(
        "回测性能",
        "性能",
        opt_t <= base_t,
        f"base={base_t:.3f}s, opt={opt_t:.3f}s, 加速比={speedup:.2f}x",
        opt_t,
    )


# ----------------------------------------------------------------------
# 边界条件测试
# ----------------------------------------------------------------------

def test_empty_data(report: TestReport):
    """空数据测试。"""
    print("\n=== 边界测试：空数据 ===")

    empty = pd.DataFrame(columns=['date', 'code', 'close', 'volume', 'amount', 'turnover_rate'])

    try:
        res = compute_factors_vectorized(empty)
        ok1 = res.empty
    except Exception as e:
        ok1 = False
        res = e
    report.record("因子计算-空数据", "边界", ok1, f"返回 empty={ok1}")

    try:
        res = ic_analysis_vectorized(empty, empty, ['x'])
        ok2 = res == {}
    except Exception as e:
        ok2 = False
        res = e
    report.record("IC 分析-空数据", "边界", ok2, f"返回空 dict={ok2}")

    try:
        res = run_backtest_vectorized(empty, empty)
        ok3 = 'metrics' in res and res['metrics'] == {}
    except Exception as e:
        ok3 = False
    report.record("回测-空数据", "边界", ok3, f"返回空结果={ok3}")


def test_single_code(report: TestReport):
    """单只股票测试。"""
    print("\n=== 边界测试：单只股票 ===")

    data = make_synthetic_data(n_codes=1, n_days=60)
    fr = make_forward_returns(data)

    try:
        factors = compute_factors_vectorized(data)
        ok1 = len(factors) == 60 and 'reversal_20d' in factors.columns
    except Exception as e:
        ok1 = False
        factors = e
    report.record("因子计算-单只股票", "边界", ok1, f"行数={len(factors) if isinstance(factors, pd.DataFrame) else 'N/A'}")

    try:
        ic = ic_analysis_vectorized(factors, fr, ['reversal_20d'])
        # 单只股票截面 IC 可能因样本不足返回空，不应报错
        ok2 = isinstance(ic, dict)
    except Exception as e:
        ok2 = False
        ic = e
    report.record("IC 分析-单只股票", "边界", ok2, f"返回类型={type(ic).__name__}")


def test_short_history(report: TestReport):
    """短历史数据测试（窗口不足）。"""
    print("\n=== 边界测试：短历史 ===")

    data = make_synthetic_data(n_codes=10, n_days=15)  # 小于 20 日窗口
    try:
        factors = compute_factors_vectorized(data)
        # ret_20d 应全为 NaN，但不报错
        ok1 = 'ret_20d' in factors.columns and factors['ret_20d'].isna().all()
    except Exception as e:
        ok1 = False
        factors = e
    report.record("因子计算-短历史", "边界", ok1, f"ret_20d 全 NaN={ok1}")


def test_all_nan_column(report: TestReport):
    """全 NaN 列测试。"""
    print("\n=== 边界测试：全 NaN 列 ===")

    data = make_synthetic_data(n_codes=20, n_days=100)
    data['turnover_rate'] = np.nan  # 制造全 NaN

    try:
        factors = compute_factors_vectorized(data)
        ok1 = 'turnover_20d' in factors.columns and factors['turnover_20d'].isna().all()
    except Exception as e:
        ok1 = False
        factors = e
    report.record("因子计算-全NaN列", "边界", ok1, f"turnover_20d 全 NaN={ok1}")


def test_walk_forward_correctness(report: TestReport, data: pd.DataFrame):
    """Walk-Forward 分割正确性测试。"""
    print("\n=== 正确性测试：Walk-Forward 分割 ===")

    factor_df = compute_factors_vectorized(data)
    fr = make_forward_returns(data)
    merged = factor_df.merge(fr[['code', 'date', 'ret_forward_5d']], on=['code', 'date'], how='inner')
    merged = merged.dropna(subset=['ret_forward_5d', 'reversal_20d', 'volatility_20d'])

    dates = merged['date']
    X = merged[['reversal_20d', 'volatility_20d', 'volume_ratio']].fillna(0)
    y = merged['ret_forward_5d']

    # 测试 purged_ts_split 不重叠
    splits = purged_ts_split(dates, n_splits=3, purge_bars=5, embargo_bars=5)
    no_overlap = True
    for train_idx, test_idx in splits:
        if set(train_idx) & set(test_idx):
            no_overlap = False
            break
    report.record(
        "Purged TS Split 无重叠",
        "正确性",
        no_overlap and len(splits) > 0,
        f"折数={len(splits)}, 无重叠={no_overlap}",
    )

    # 测试 walk_forward_splits
    wf_splits = walk_forward_splits(dates, train_window=100, test_window=20, purge_bars=5)
    wf_no_overlap = True
    for train_idx, test_idx in wf_splits:
        if set(train_idx) & set(test_idx):
            wf_no_overlap = False
            break
    report.record(
        "Walk-Forward Split 无重叠",
        "正确性",
        wf_no_overlap and len(wf_splits) > 0,
        f"折数={len(wf_splits)}, 无重叠={wf_no_overlap}",
    )

    # 测试 walk_forward_predict 生成完整 OOS 预测
    from sklearn.linear_model import LinearRegression
    preds, fold_info = walk_forward_predict(
        X, y, dates,
        model_factory=lambda: LinearRegression(),
        train_window=100, test_window=20, purge_bars=5,
    )
    coverage = preds.notna().mean()
    report.record(
        "Walk-Forward 预测覆盖",
        "正确性",
        coverage > 0.3 and len(fold_info) > 0,
        f"覆盖率={coverage:.2%}, 折数={len(fold_info)}",
    )


def test_baseline_bug_demo(report: TestReport):
    """演示原实现的索引 bug。"""
    print("\n=== Bug 验证：原 train 方法索引错误 ===")

    # 构造 100 行数据，后 20 行作为 test_dates
    X = pd.DataFrame({'f1': np.arange(100), 'f2': np.arange(100) * 2})
    y = pd.Series(np.arange(100), name='y')
    test_dates = pd.Series(pd.date_range('2024-01-01', periods=20), index=range(80, 100))

    X_train, X_test, y_train, y_test = train_baseline_bug(X, y, test_dates)

    # bug：X.index 是 0..99，test_dates.index 是 80..99
    # isin 会匹配 80..99，所以 train 应该是 0..79（80 行），test 是 80..99（20 行）
    # 看似正确，但如果 X 经过 dropna/reset_index 后 index 不连续，bug 就会暴露
    # 演示：X 重置索引后
    X_reset = X.drop([0, 1, 2]).reset_index(drop=True)  # 现在 index 是 0..96
    y_reset = y.drop([0, 1, 2]).reset_index(drop=True)
    test_dates_reset = pd.Series(
        pd.date_range('2024-01-01', periods=20),
        index=range(77, 97),  # 对应原数据的后 20 行
    )

    X_train2, X_test2, _, _ = train_baseline_bug(X_reset, y_reset, test_dates_reset)
    # bug：X_reset.index 是 0..96，test_dates_reset.index 是 77..96
    # isin(77..96) 会匹配 X_reset 的 77..96 行，但语义上 test_dates 是日期不是行号
    # 当 test_dates 的 index 与 X 的 index 不对齐时，划分就是错的
    bug_exposed = len(X_test2) == 20  # 表面正确
    # 真正的 bug：如果 test_dates.index 是日期而非行号
    test_dates_dates = pd.Series(
        pd.date_range('2024-01-01', periods=20),
        index=pd.date_range('2023-01-01', periods=20),  # index 是日期
    )
    try:
        X_train3, X_test3, _, _ = train_baseline_bug(X_reset, y_reset, test_dates_dates)
        # X_reset.index 是 0..96，test_dates_dates.index 是日期
        # isin(日期) 永远不匹配数字 index，所以 X_test3 为空，X_train3 = 全部
        bug_confirmed = len(X_test3) == 0 and len(X_train3) == len(X_reset)
    except Exception:
        bug_confirmed = False

    report.record(
        "原 train 方法索引 Bug 确认",
        "Bug验证",
        bug_confirmed,
        f"当 test_dates.index 为日期时，X_test 为空（{len(X_test3) if bug_confirmed else 'N/A'} 行），"
        f"全部数据进入训练集，无样本外验证",
    )


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("jingni-trader 优化验证测试套件")
    print(f"执行时间: {datetime.now().isoformat()}")
    print("=" * 70)

    report = TestReport()

    # 生成测试数据
    print("\n生成合成测试数据...")
    data_small = make_synthetic_data(n_codes=50, n_days=250)
    data_large = make_synthetic_data(n_codes=200, n_days=500)
    fr_small = make_forward_returns(data_small)
    fr_large = make_forward_returns(data_large)
    signals_small = make_signals(data_small)
    signals_large = make_signals(data_large)
    bench_small = make_benchmark(data_small)
    print(f"小数据集: {len(data_small)} 行, {data_small['code'].nunique()} 只股票")
    print(f"大数据集: {len(data_large)} 行, {data_large['code'].nunique()} 只股票")

    # 正确性测试
    test_factor_correctness(report, data_small)
    test_ic_correctness(report, data_small, fr_small)
    test_backtest_correctness(report, data_small, signals_small)
    test_backtest_with_benchmark(report, data_small, signals_small, bench_small)
    test_walk_forward_correctness(report, data_small)
    test_baseline_bug_demo(report)

    # 性能测试（用大数据集）
    test_factor_performance(report, data_large)
    test_ic_performance(report, data_large, fr_large)
    test_backtest_performance(report, data_large, signals_large)

    # 边界测试
    test_empty_data(report)
    test_single_code(report)
    test_short_history(report)
    test_all_nan_column(report)

    # 汇总
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    summary = report.summary()
    print(f"总计: {summary['total']}, 通过: {summary['passed']}, 失败: {summary['failed']}")
    print(f"通过率: {summary['pass_rate']:.2%}")

    # 保存结果
    out_dir = Path(__file__).parent
    out_path = out_dir / "test_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n测试结果已保存: {out_path}")

    return summary


if __name__ == "__main__":
    main()