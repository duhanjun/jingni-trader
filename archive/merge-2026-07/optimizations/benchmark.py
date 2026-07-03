"""
性能基准对比测试

对比 jingni-trader 原生实现与优化后向量化实现的性能差异。
输出 JSON 格式结果，供验证报告引用。
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from optimizations.vectorized_backtest import VectorizedBacktestEngine
from optimizations.vectorized_factor import VectorizedFactorAnalysis
from optimizations.factor_expression import FactorExpressionEngine, ALPHA158_EXPRESSIONS

# 引入原 native_adapter 做对比（不修改它，仅读取）
sys.path.insert(0, os.path.join(ROOT, 'skills', 'backtest-engine'))
from scripts.adapters.native_adapter import NativeAdapter


def make_data(n_stocks, n_days, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_days)
    codes = [f'{600000+i:06d}.SH' for i in range(n_stocks)]
    rows = []
    for code in codes:
        price = 10.0 + rng.normal(0, 0.5)
        for dt in dates:
            ret = rng.normal(0, 0.02)
            price = max(price * (1 + ret), 1.0)
            vol = int(rng.integers(100000, 1000000))
            rows.append({
                'code': code, 'date': dt,
                'open': price * (1 + rng.normal(0, 0.005)),
                'high': price * (1 + abs(rng.normal(0, 0.01))),
                'low': price * (1 - abs(rng.normal(0, 0.01))),
                'close': price, 'volume': vol, 'amount': vol * price,
                'turnover_rate': round(rng.uniform(0.5, 5.0), 4),
                'change_pct': ret * 100,
                'is_limit_up': False, 'is_limit_down': False,
            })
    return pd.DataFrame(rows)


def make_signals(data):
    df = data.sort_values(['code', 'date']).copy()
    df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    sig = df[['code', 'date', 'ret_5d']].dropna().copy()
    sig['signal'] = 0
    sig['rank'] = sig.groupby('date')['ret_5d'].rank(pct=True)
    sig.loc[sig['rank'] > 0.8, 'signal'] = 1
    sig.loc[sig['rank'] < 0.2, 'signal'] = -1
    return sig[sig['signal'] != 0][['code', 'date', 'signal']]


def make_factor_data(n_stocks, n_days, seed=42):
    data = make_data(n_stocks, n_days, seed)
    df = data.sort_values(['code', 'date']).copy()
    df['factor_a'] = df.groupby('code')['close'].pct_change(5) * -1
    factor_df = df[['code', 'date', 'factor_a', 'close']].dropna()
    fr = df[['code', 'date', 'close']].copy()
    fr['ret_forward_5d'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1
    fr = fr.dropna()
    return factor_df, fr


# ════════════════════════════════════════════════════════
# 基准 1: 回测引擎性能对比
# ════════════════════════════════════════════════════════
def bench_backtest():
    print("\n" + "=" * 60)
    print("基准 1: 回测引擎性能对比 (native vs vectorized)")
    print("=" * 60)
    results = []

    for n_stocks, n_days in [(20, 60), (50, 120), (100, 120), (200, 120)]:
        data = make_data(n_stocks, n_days, seed=7)
        signals = make_signals(data)

        # 原生实现
        native = NativeAdapter()
        t0 = time.perf_counter()
        native_result = native.run_backtest(data, signals)
        native_time = time.perf_counter() - t0

        # 向量化实现
        vec = VectorizedBacktestEngine()
        t0 = time.perf_counter()
        vec_result = vec.run_backtest(data, signals)
        vec_time = time.perf_counter() - t0

        # 正确性：对比最终净值
        native_eq = native_result['equity_curve']['equity'].iloc[-1] if not native_result['equity_curve'].empty else 0
        vec_eq = vec_result['equity_curve']['equity'].iloc[-1] if not vec_result['equity_curve'].empty else 0
        speedup = native_time / vec_time if vec_time > 0 else float('inf')

        native_trades = len(native_result['trades'])
        vec_trades = len(vec_result['trades'])

        row = {
            "scale": f"{n_stocks}股×{n_days}日",
            "native_sec": round(native_time, 4),
            "vectorized_sec": round(vec_time, 4),
            "speedup": round(speedup, 2),
            "native_final_equity": round(native_eq, 2),
            "vec_final_equity": round(vec_eq, 2),
            "native_trades": native_trades,
            "vec_trades": vec_trades,
        }
        results.append(row)
        print(f"  {row['scale']:>16} | native={native_time:.3f}s vec={vec_time:.3f}s "
              f"| 加速比={speedup:.2f}x | 净值差={abs(native_eq-vec_eq):.2f}")

    return results


# ════════════════════════════════════════════════════════
# 基准 2: IC 分析性能对比
# ════════════════════════════════════════════════════════
def bench_ic_analysis():
    print("\n" + "=" * 60)
    print("基准 2: IC 分析性能对比 (scipy逐日 vs 向量化)")
    print("=" * 60)
    results = []

    for n_stocks, n_days in [(30, 80), (100, 120), (200, 200)]:
        factor_df, fr = make_factor_data(n_stocks, n_days, seed=7)

        # 原生 scipy 逐日实现（复刻原 engine._calc_ic 逻辑）
        merged = factor_df.merge(fr[['code', 'date', 'ret_forward_5d']], on=['code', 'date']).dropna(subset=['factor_a', 'ret_forward_5d'])
        t0 = time.perf_counter()
        ref_ic = {}
        for dt, grp in merged.groupby('date'):
            if len(grp) < 10:
                continue
            ic, _ = scipy_stats.spearmanr(grp['factor_a'], grp['ret_forward_5d'])
            if not np.isnan(ic):
                ref_ic[dt] = ic
        native_time = time.perf_counter() - t0

        # 向量化实现
        t0 = time.perf_counter()
        vec_ic = VectorizedFactorAnalysis.calc_ic_series(factor_df, fr, 'factor_a', 'ret_forward_5d')
        vec_time = time.perf_counter() - t0

        # 正确性：对比 IC 均值
        native_mean = np.mean(list(ref_ic.values())) if ref_ic else 0
        vec_mean = vec_ic.mean() if not vec_ic.empty else 0
        speedup = native_time / vec_time if vec_time > 0 else float('inf')
        ic_diff = abs(native_mean - vec_mean)

        row = {
            "scale": f"{n_stocks}股×{n_days}日",
            "scipy_sec": round(native_time, 4),
            "vectorized_sec": round(vec_time, 4),
            "speedup": round(speedup, 2),
            "scipy_ic_mean": round(native_mean, 6),
            "vec_ic_mean": round(float(vec_mean), 6),
            "ic_diff": round(ic_diff, 8),
            "n_dates": len(ref_ic),
        }
        results.append(row)
        print(f"  {row['scale']:>16} | scipy={native_time:.4f}s vec={vec_time:.4f}s "
              f"| 加速比={speedup:.2f}x | IC差={ic_diff:.2e}")

    return results


# ════════════════════════════════════════════════════════
# 基准 3: 因子表达式引擎
# ════════════════════════════════════════════════════════
def bench_expression_engine():
    print("\n" + "=" * 60)
    print("基准 3: 因子表达式引擎")
    print("=" * 60)
    results = []

    for n_stocks, n_days in [(20, 60), (50, 100), (100, 120)]:
        data = make_data(n_stocks, n_days, seed=7)
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        data['ret_1d'] = data.groupby('code')['close'].pct_change()
        engine = FactorExpressionEngine()

        t0 = time.perf_counter()
        out = engine.batch_evaluate(ALPHA158_EXPRESSIONS, data)
        elapsed = time.perf_counter() - t0

        row = {
            "scale": f"{n_stocks}股×{n_days}日",
            "n_factors": len(ALPHA158_EXPRESSIONS),
            "n_rows": len(data),
            "elapsed_sec": round(elapsed, 4),
            "factors_per_sec": round(len(ALPHA158_EXPRESSIONS) / elapsed, 2) if elapsed > 0 else 0,
        }
        results.append(row)
        print(f"  {row['scale']:>16} | {len(ALPHA158_EXPRESSIONS)}因子 × {len(data)}行 "
              f"| {elapsed:.3f}s | {row['factors_per_sec']:.1f} 因子/秒")

    # 展示部分因子表达式
    print("\n  预定义因子库 (Alpha158 风格):")
    for name, expr in list(ALPHA158_EXPRESSIONS.items())[:5]:
        print(f"    {name:14s} = {expr}")

    return results


if __name__ == '__main__':
    all_results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "backtest_benchmark": bench_backtest(),
        "ic_benchmark": bench_ic_analysis(),
        "expression_benchmark": bench_expression_engine(),
    }

    out_path = os.path.join(ROOT, 'optimizations', 'benchmark_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n基准结果已保存: {out_path}")
