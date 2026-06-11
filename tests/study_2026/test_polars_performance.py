"""
==============================================================================
借鉴来源: AKQuant (github.com/akfamily/akquant) - Rust+Polars 混合架构
         vnpy 4.0 AlphaLab - Polars 高性能数据处理
         Qlib columnar binary format - 列式存储专为金融时序优化
优化方向: 数据获取与处理效率 — Pandas vs Polars 性能对比,
         评估将数据处理层迁移到 Polars 的性能收益
==============================================================================

当前 jingni-trader 的数据处理和因子计算全部基于 pandas，当股票数量
超过 500 或时间跨度超过 5 年时，pandas groupby+rolling 组合会显著变慢。

AKQuant 使用 Polars Lazy API 做因子计算，性能提升 5-15x。
vnpy 4.0 的 AlphaLab 已全面转向 Polars。
Qlib 使用自研 columnar binary 格式替代 pandas，速度提升数十倍。

本验证代码:
  1. 在相同 A 股模拟数据上对比 Pandas vs Polars 的因子计算性能
  2. 覆盖常见操作: groupby_rolling, pct_change, rank, window aggregation
  3. 评估内存使用差异
  4. 给出数据规模-性能的 scalability 分析
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

# 尝试导入 Polars
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("[WARNING] Polars 未安装, 部分测试将跳过。安装: pip install polars")

# 尝试导入 memory_profiler
try:
    from memory_profiler import memory_usage as mem_usage
    HAS_MEM_PROFILE = True
except ImportError:
    HAS_MEM_PROFILE = False


# ==========================================================================
# 1. 测试数据生成
# ==========================================================================

def create_benchmark_data(
    n_stocks: int = 200,
    n_days: int = 252 * 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成模拟的 A 股日线数据
    模拟真实数据特征：
    - OHLCV + turnover_rate + amount
    - 含停牌、涨跌停
    - 含行业分类
    """
    np.random.seed(seed)
    dates = pd.bdate_range('2021-01-01', periods=n_days)
    stocks = [f"{i:06d}.{'SZ' if i % 2 == 1 else 'SH'}" for i in range(1, n_stocks + 1)]

    industries = ['银行', '医药', '电子', '计算机', '食品饮料', '机械', '化工', '地产']
    code_to_ind = {code: industries[i % len(industries)] for i, code in enumerate(stocks)}

    rows = []
    for code in stocks:
        price = np.random.uniform(5, 100)
        for dt in dates:
            daily_ret = np.random.normal(0.0003, 0.018)
            price *= (1 + daily_ret)
            # 模拟停牌 (1% 概率)
            if np.random.random() < 0.01:
                daily_vol = 0
            else:
                daily_vol = int(np.random.lognormal(14, 0.5))

            rows.append({
                'date': dt,
                'code': code,
                'open': price * (1 + np.random.normal(0, 0.003)),
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': daily_vol,
                'amount': price * daily_vol if daily_vol > 0 else 0,
                'turnover_rate': np.random.uniform(0.005, 0.08),
                'industry': code_to_ind[code],
            })

    df = pd.DataFrame(rows)

    # 保证 OHLC 一致性
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df.sort_values(['date', 'code']).reset_index(drop=True)


# ==========================================================================
# 2. Pandas 因子计算 (当前 jingni-trader 方式)
# ==========================================================================

def compute_factors_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """使用 Pandas 计算因子 — 等价于 factor-engine 的 compute_a_share_factors"""
    df = df.sort_values(['code', 'date']).copy()
    result = pd.DataFrame()

    # 收益率因子
    def pct_change_gb(col, period):
        return df.groupby('code')[col].transform(lambda x: x.pct_change(period))

    result['ret_1d'] = pct_change_gb('close', 1)
    result['ret_5d'] = pct_change_gb('close', 5)
    result['ret_20d'] = pct_change_gb('close', 20)
    result['reversal_5d'] = -result['ret_5d']
    result['reversal_20d'] = -result['ret_20d']

    # 成交量因子
    def rolling_mean_gb(col, window):
        return df.groupby('code')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    vol_20d = rolling_mean_gb('volume', 20)
    result['volume_ratio'] = df['volume'] / vol_20d.replace(0, np.nan)

    # 波动率因子
    df['ret_1d_calc'] = pct_change_gb('close', 1)
    result['volatility_20d'] = df.groupby('code')['ret_1d_calc'].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )

    # 截面排名因子 (借鉴 Qlib Rank)
    result['rank_ret_20d'] = result.groupby(df['date'])['ret_20d'].rank(pct=True)

    return result


# ==========================================================================
# 3. Polars 因子计算 (优化方案)
# ==========================================================================

def compute_factors_polars(pd_df: pd.DataFrame) -> pl.DataFrame:
    """使用 Polars 计算因子 — 对标 compute_factors_pandas"""
    df = pl.from_pandas(pd_df)

    # 收益率因子 (Polars 原生支持 group_by + rolling + pct_change)
    ret_cols = []
    for period in [1, 5, 20]:
        col_name = f'ret_{period}d'
        ret_expr = (
            (pl.col('close') / pl.col('close').shift(period) - 1)
            .over('code')
            .alias(col_name)
        )
        ret_cols.append(ret_expr)

    # 用 Polars Lazy 模式做优化 (借鉴 AKQuant 的 query plan optimization)
    result = df.lazy().with_columns([
        # 收益率因子
        (pl.col('close') / pl.col('close').shift(1) - 1).over('code').alias('ret_1d'),
        (pl.col('close') / pl.col('close').shift(5) - 1).over('code').alias('ret_5d'),
        (pl.col('close') / pl.col('close').shift(20) - 1).over('code').alias('ret_20d'),

        # 反转因子
        (-(pl.col('close') / pl.col('close').shift(5) - 1)).over('code').alias('reversal_5d'),
        (-(pl.col('close') / pl.col('close').shift(20) - 1)).over('code').alias('reversal_20d'),

        # 成交量因子 (rolling_mean via over + rolling)
        pl.col('volume').rolling_mean(20, min_periods=1).over('code').alias('vol_20d_ma'),
    ])

    # volume_ratio 需要多步计算, 没法一个 lazy 里完成
    result = result.collect()

    result = result.with_columns([
        (pl.col('volume') / pl.col('vol_20d_ma').replace(0, None))
            .alias('volume_ratio'),
        (pl.col('ret_1d').rolling_std(20, min_periods=10).over('code'))
            .alias('volatility_20d'),
    ])

    # 截面排名
    result = result.with_columns([
        pl.col('ret_20d')
            .rank('ordinal', descending=False)
            .over('date')
            .alias('rank_ret_20d')
            .cast(pl.Float64)
            / pl.col('ret_20d').count().over('date').cast(pl.Float64),
    ])

    return result


# ==========================================================================
# 4. 性能测试
# ==========================================================================

@dataclass if False else None  # dummy to avoid import error
# Use simple dict instead
class BenchmarkResult:
    pass


def benchmark_operation(name: str, fn, *args, n_runs: int = 3, **kwargs):
    """运行基准测试"""
    # 预热
    fn(*args, **kwargs)
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        times.append(time.time() - t0)
    return {
        'name': name,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'results': result,
    }


def run_benchmarks():
    """运行全面性能基准对比"""
    print("=" * 70)
    print("Pandas vs Polars 性能对比 - A股因子计算场景")
    print("=" * 70)

    # 不同数据规模
    configs = [
        {'n_stocks': 50, 'n_days': 252, 'label': '1年/50股'},
        {'n_stocks': 100, 'n_days': 252 * 3, 'label': '3年/100股'},
        {'n_stocks': 200, 'n_days': 252 * 3, 'label': '3年/200股'},
        {'n_stocks': 500, 'n_days': 252 * 3, 'label': '3年/500股'},
    ]

    results = []
    for cfg in configs:
        print(f"\n--- 数据规模: {cfg['label']} ---")
        df = create_benchmark_data(n_stocks=cfg['n_stocks'], n_days=cfg['n_days'])
        print(f"  数据量: {len(df):,} 行, {df['code'].nunique():,} 只股票")

        # Pandas 测试
        pd_result = benchmark_operation(
            f"Pandas_{cfg['label']}",
            compute_factors_pandas, df, n_runs=3
        )
        print(f"  Pandas: {pd_result['mean_time']:.3f}s (std={pd_result['std_time']:.3f}s)")

        # Polars 测试
        if HAS_POLARS:
            pl_result = benchmark_operation(
                f"Polars_{cfg['label']}",
                compute_factors_polars, df, n_runs=3
            )
            speedup = pd_result['mean_time'] / max(pl_result['mean_time'], 0.0001)
            print(f"  Polars: {pl_result['mean_time']:.3f}s (std={pl_result['std_time']:.3f}s)")
            print(f"  加速比: {speedup:.1f}x {'<<<' if speedup > 2 else ''}")

            results.append({
                'label': cfg['label'],
                'rows': len(df),
                'pandas_time': pd_result['mean_time'],
                'polars_time': pl_result['mean_time'],
                'speedup': speedup,
            })
        else:
            results.append({
                'label': cfg['label'],
                'rows': len(df),
                'pandas_time': pd_result['mean_time'],
                'polars_time': float('nan'),
                'speedup': 0,
            })

    # 汇总
    print("\n" + "=" * 70)
    print("性能对比汇总")
    print("=" * 70)
    print(f"  {'规模':<16} {'行数':>10} {'Pandas(s)':>10} {'Polars(s)':>10} {'加速比':>8}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['label']:<16} {r['rows']:>10,} {r['pandas_time']:>10.3f} "
              f"{r['polars_time']:>10.3f} {r['speedup']:>7.1f}x")

    return results


def test_correctness():
    """验证 Pandas 和 Polars 计算结果的一致性"""
    print("\n" + "=" * 70)
    print("正确性验证: Pandas vs Polars 结果一致性")
    print("=" * 70)

    df = create_benchmark_data(n_stocks=50, n_days=100)

    pd_factors = compute_factors_pandas(df)

    if HAS_POLARS:
        pl_factors = compute_factors_polars(df)
        pl_df = pl_factors.select(pd_factors.columns).to_pandas()
        # Align index
        pl_df.index = pd_factors.index

        common_cols = [c for c in pd_factors.columns if c in pl_df.columns]
        for col in common_cols:
            pd_vals = pd_factors[col].dropna()
            pl_vals = pl_df[col].dropna()
            common_idx = pd_vals.index.intersection(pl_vals.index)
            if len(common_idx) < 5:
                continue
            diff = (pd_vals.loc[common_idx] - pl_vals.loc[common_idx]).abs().max()
            status = "PASS" if diff < 0.01 else "FAIL"
            print(f"  {col:<20}: max_diff={diff:.6f}  [{status}]")
    else:
        print("  Polars 未安装，跳过正确性验证")


def test_memory_estimate():
    """估算内存使用差异"""
    print("\n" + "=" * 70)
    print("内存使用估算")
    print("=" * 70)

    df = create_benchmark_data(n_stocks=200, n_days=252 * 3)

    # Pandas 内存
    pd_mem = df.memory_usage(deep=True).sum() / 1024 / 1024

    if HAS_POLARS:
        pl_df = pl.from_pandas(df)
        pl_mem = pl_df.estimated_size() / 1024 / 1024
        print(f"  Pandas DataFrame: {pd_mem:.1f} MB")
        print(f"  Polars DataFrame: {pl_mem:.1f} MB")
        print(f"  内存节省: {(1 - pl_mem / pd_mem) * 100:.1f}%")
    else:
        print(f"  Pandas DataFrame: {pd_mem:.1f} MB")
        print(f"  Polars 未安装，无法对比")


if __name__ == "__main__":
    print("=" * 70)
    print("jingni-trader 优化验证: Polars 高性能数据处理")
    print("借鉴来源: AKQuant Rust+Polars 架构 / vnpy 4.0 AlphaLab / Qlib columnar")
    print("优化方向: 数据获取与处理效率")
    print("=" * 70)

    test_correctness()
    test_memory_estimate()
    results = run_benchmarks()

    print("\n" + "=" * 70)
    print("全部测试完成")
    print("=" * 70)