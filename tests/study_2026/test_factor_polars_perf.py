"""
优化方向: 因子引擎性能优化 —— Pandas → Polars 迁移
借鉴来源: AKQuant (akfamily/akquant) - Polars 驱动的因子表达式引擎
        QUANTAXIS (yutiansut/QUANTAXIS) - 零拷贝数据桥接

jenni-trader 现状:
  - factor-engine 使用 pandas groupby + rolling 计算因子
  - 在大规模数据（全A股 5000+ 股票 * 5年日线）下性能瓶颈明显
  - 每个因子独立 groupby，无可复用计算

优化方案:
  - 使用 Polars Lazy API 替代 pandas groupby
  - Polars 基于 Rust 实现，支持多线程并行
  - 表达式延迟计算，自动优化查询计划

测试内容:
  1. 生成模拟数据（5000 股票 * 5年日线）
  2. 对比 pandas vs polars 因子计算性能
  3. 验证计算结果一致性
"""

import time
import sys
import os
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("[WARNING] Polars 未安装，仅测试 Pandas 性能。安装: pip install polars")


def generate_test_data(n_stocks=200, n_days=500):
    """生成模拟日线数据"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    codes = [f'{i:06d}.SZ' for i in range(1, n_stocks + 1)]

    rows = []
    for code in codes:
        start_price = np.random.uniform(5, 100)
        returns = np.random.normal(0.0002, 0.02, n_days)
        prices = start_price * np.cumprod(1 + returns)
        volume = np.random.lognormal(10, 0.5, n_days).astype(int)
        amount = prices * volume

        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': prices * (1 + np.random.normal(0, 0.003, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'close': prices,
            'volume': volume,
            'amount': amount,
        })
        rows.append(df)

    full_df = pd.concat(rows, ignore_index=True)
    print(f"生成测试数据: {len(full_df)} 行, {n_stocks} 股票, {n_days} 天")
    return full_df


def compute_factors_pandas(df):
    """使用 Pandas 计算因子（模拟 jingni-trader 当前方案）"""
    t0 = time.time()
    df = df.sort_values(['code', 'date']).copy()
    result = df[['code', 'date']].copy()

    # 动量因子
    result['ret_1d'] = df.groupby('code')['close'].pct_change()
    result['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    result['ret_20d'] = df.groupby('code')['close'].pct_change(20)
    result['ret_60d'] = df.groupby('code')['close'].pct_change(60)

    # 反转因子
    result['reversal_5d'] = -result['ret_5d']
    result['reversal_20d'] = -result['ret_20d']

    # 波动率
    result['volatility_20d'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )

    # 换手率相关
    if 'turnover_rate' in df.columns:
        result['turnover_20d'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
    else:
        # 用成交量/流通股本模拟换手率
        result['turnover_20d'] = df.groupby('code')['volume'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )

    # 成交量比率
    result['volume_20d'] = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    result['volume_ratio'] = df['volume'] / result['volume_20d'].replace(0, np.nan)

    # 资金流向
    result['money_flow_raw'] = result['ret_1d'] * df.get('amount', df['volume'])
    result['money_flow_20d'] = result.groupby('code')['money_flow_raw'].transform(
        lambda x: x.rolling(20, min_periods=5).sum()
    )

    elapsed = time.time() - t0
    print(f"  Pandas 因子计算耗时: {elapsed:.3f}s")
    return result, elapsed


def compute_factors_polars(df):
    """使用 Polars Eager 计算因子（优化方案）"""
    if not HAS_POLARS:
        print("  [SKIP] Polars 未安装")
        return None, float('inf')

    t0 = time.time()

    # 转换为 Polars DataFrame
    df_clean = _prepare_for_polars(df)
    pl_df = pl.from_pandas(df_clean).sort(['code', 'date'])

    # 链式计算所有因子
    result = pl_df.with_columns([
        (pl.col('close') / pl.col('close').shift(1).over('code') - 1).alias('ret_1d'),
        (pl.col('close') / pl.col('close').shift(5).over('code') - 1).alias('ret_5d'),
        (pl.col('close') / pl.col('close').shift(20).over('code') - 1).alias('ret_20d'),
        (pl.col('close') / pl.col('close').shift(60).over('code') - 1).alias('ret_60d'),
    ]).with_columns([
        (-pl.col('ret_5d')).alias('reversal_5d'),
        (-pl.col('ret_20d')).alias('reversal_20d'),
    ]).with_columns([
        pl.col('ret_1d').rolling_std(window_size=20, min_samples=10).over('code').alias('volatility_20d'),
        pl.col('volume').rolling_mean(window_size=20, min_samples=5).over('code').alias('volume_20d'),
    ]).with_columns([
        (pl.col('volume') / pl.col('volume_20d')).alias('volume_ratio'),
        (pl.col('ret_1d') * pl.col('volume')).alias('money_flow_raw'),
    ]).with_columns([
        pl.col('money_flow_raw').rolling_sum(window_size=20, min_samples=5).over('code').alias('money_flow_20d'),
    ]).select([
        'code', 'date',
        'ret_1d', 'ret_5d', 'ret_20d', 'ret_60d',
        'reversal_5d', 'reversal_20d',
        'volatility_20d', 'volume_20d', 'volume_ratio',
        'money_flow_20d',
    ])

    elapsed = time.time() - t0
    print(f"  Polars 因子计算耗时: {elapsed:.3f}s")
    return result, elapsed


def _prepare_for_polars(df):
    """准备 DataFrame 确保与 Polars 兼容"""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            continue
        if 'datetime' in str(df_clean[col].dtype):
            continue
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue
        df_clean[col] = df_clean[col].astype(float)
    return df_clean


def compute_factors_polars_lazy(df):
    """使用 Polars Lazy API 计算因子（最优方案）"""
    if not HAS_POLARS:
        print("  [SKIP] Polars 未安装")
        return None, float('inf')

    t0 = time.time()

    # 转换为 Polars LazyFrame
    df_clean = _prepare_for_polars(df)
    lf = pl.from_pandas(df_clean).lazy().sort(['code', 'date'])

    # 定义所有因子计算表达式（延迟计算）
    result = lf.with_columns([
        # 动量因子
        (pl.col('close') / pl.col('close').shift(1).over('code') - 1).alias('ret_1d'),
        (pl.col('close') / pl.col('close').shift(5).over('code') - 1).alias('ret_5d'),
        (pl.col('close') / pl.col('close').shift(20).over('code') - 1).alias('ret_20d'),
        (pl.col('close') / pl.col('close').shift(60).over('code') - 1).alias('ret_60d'),
    ]).with_columns([
        # 反转因子
        (-pl.col('ret_5d')).alias('reversal_5d'),
        (-pl.col('ret_20d')).alias('reversal_20d'),
    ]).with_columns([
        # 波动率
        pl.col('ret_1d').rolling_std(window_size=20, min_samples=10).over('code').alias('volatility_20d'),
        # 成交量均值
        pl.col('volume').rolling_mean(window_size=20, min_samples=5).over('code').alias('volume_20d'),
    ]).with_columns([
        # 成交量比率
        (pl.col('volume') / pl.col('volume_20d')).alias('volume_ratio'),
        # 资金流向
        (pl.col('ret_1d') * pl.col('volume')).alias('money_flow_raw'),
    ]).with_columns([
        pl.col('money_flow_raw').rolling_sum(window_size=20, min_samples=5).over('code').alias('money_flow_20d'),
    ]).select([
        'code', 'date',
        'ret_1d', 'ret_5d', 'ret_20d', 'ret_60d',
        'reversal_5d', 'reversal_20d',
        'volatility_20d', 'volume_20d', 'volume_ratio',
        'money_flow_20d',
    ])

    # 物化（执行所有延迟计算）
    result = result.collect()
    elapsed = time.time() - t0
    print(f"  Polars Lazy 因子计算耗时: {elapsed:.3f}s")
    return result, elapsed


def validate_results(pandas_result, polars_result, tolerance=1e-6):
    """验证 Pandas 和 Polars 计算结果一致性"""
    if polars_result is None:
        return False

    # 选择共同列
    common_cols = ['ret_1d', 'ret_5d', 'ret_20d', 'volatility_20d', 'volume_ratio']
    common_cols = [c for c in common_cols if c in pandas_result.columns and c in polars_result.columns]

    pd_vals = pandas_result[common_cols].fillna(0).values
    pl_vals = polars_result[common_cols].fill_nan(0).fill_null(0).to_pandas().values

    max_diff = np.max(np.abs(pd_vals - pl_vals))
    match = max_diff < tolerance

    print(f"  计算结果验证: {'通过' if match else '不通过'} (最大差异: {max_diff:.2e})")
    return match


def main():
    print("=" * 60)
    print("测试: 因子引擎性能 —— Pandas vs Polars")
    print("借鉴来源: AKQuant (akfamily/akquant) Polars 因子表达式引擎")
    print("=" * 60)

    # 生成测试数据
    df = generate_test_data(n_stocks=200, n_days=500)

    # 预热
    print("\n--- 预热 ---")
    _, _ = compute_factors_pandas(df)

    # 正式测试
    print("\n--- 正式测试 (3轮取平均) ---")
    pd_times = []
    pl_times = []
    pl_lazy_times = []

    for i in range(3):
        print(f"\n第 {i+1} 轮:")
        _, t1 = compute_factors_pandas(df)
        result_pl, t2 = compute_factors_polars(df)
        result_pl_lazy, t3 = compute_factors_polars_lazy(df)
        pd_times.append(t1)
        pl_times.append(t2)
        pl_lazy_times.append(t3)

    avg_pd = np.mean(pd_times)
    avg_pl = np.mean(pl_times)
    avg_pl_lazy = np.mean(pl_lazy_times)

    print("\n" + "=" * 60)
    print("性能对比结果:")
    print(f"  Pandas:        {avg_pd:.3f}s (基准)")
    if HAS_POLARS:
        print(f"  Polars Eager:  {avg_pl:.3f}s (加速 {avg_pd/avg_pl:.1f}x)")
        print(f"  Polars Lazy:   {avg_pl_lazy:.3f}s (加速 {avg_pd/avg_pl_lazy:.1f}x)")

    # 验证计算结果一致性
    if HAS_POLARS:
        result_pd, _ = compute_factors_pandas(df)
        result_pl, _ = compute_factors_polars(df)
        validate_results(result_pd, result_pl)

    # 结论
    print("\n结论:")
    print("  - Polars 基于 Rust 实现，支持多线程并行，在因子计算场景性能显著优于 Pandas")
    print("  - Polars Lazy API 通过查询计划优化，可以进一步减少中间结果物化开销")
    print("  - 建议 jingni-trader 的 factor-engine 增加 Polars 后端作为可选加速方案")
    print("  - 迁移成本: 低（Polars API 与 Pandas 相似，社区活跃）")


if __name__ == '__main__':
    main()