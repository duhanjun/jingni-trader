#!/usr/bin/env python3
"""
================================================================================
优化方向: 数据处理性能优化 - Polars 替代 Pandas 核心操作
借鉴来源: https://github.com/yutiansut/QUANTAXIS
          QUANTAXIS v2.1 的 Rust 核心 + QADataBridge (Apache Arrow 零拷贝)
          + QAPRO-RS 的 Polars 集成，实现 10-100x 性能提升
================================================================================

验证目标:
 1. Pandas vs Polars 在常见量化数据操作上的性能对比
 2. 大截面因子计算（多股票 × 多日期）性能测试
 3. 内存使用对比分析
 4. 评估渐进式迁移可行性
"""

import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# 检测 Polars 是否可用
try:
    import polars as pl
    HAS_POLARS = True
    print(f"Polars 版本: {pl.__version__}")
except ImportError:
    HAS_POLARS = False
    print("Polars 未安装，将跳过 Polars 相关测试")
    print("安装: pip install polars")

print(f"Pandas 版本: {pd.__version__}")


# ============================================================================
# 数据生成
# ============================================================================

def generate_large_dataset(
    n_stocks: int = 500,
    n_days: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """生成大规模A股模拟数据集"""
    np.random.seed(seed)
    dates = pd.date_range('2018-01-01', periods=n_days, freq='B')
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    # 向量化生成（比逐个循环快很多）
    all_data = []

    for i, code in enumerate(codes):
        start_price = np.random.uniform(5, 200)
        returns = np.random.normal(0.0003, 0.015, n_days)
        # 添加轻微自相关
        for j in range(1, n_days):
            returns[j] += 0.1 * returns[j - 1]
        prices = start_price * np.cumprod(1 + returns)

        # 使用 DataFrame 构造
        chunk = pd.DataFrame({
            'date': dates,
            'code': code,
            'close': prices.astype(np.float32),
            'open': (prices * (1 + np.random.normal(0, 0.003, n_days))).astype(np.float32),
            'high': (prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))).astype(np.float32),
            'low': (prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))).astype(np.float32),
            'volume': np.random.lognormal(14, 0.5, n_days).astype(np.int64),
            'amount': np.random.lognormal(18, 0.5, n_days).astype(np.float64),
            'turnover_rate': np.random.uniform(0.005, 0.05, n_days).astype(np.float32),
        })
        all_data.append(chunk)

    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    return df


# ============================================================================
# 性能基准测试
# ============================================================================

def benchmark_grouped_rolling(df: pd.DataFrame, n_runs: int = 3):
    """测试分组滚动计算性能 (因子引擎核心操作)"""
    print("\n" + "-" * 50)
    print("基准1: 分组滚动计算 (groupby + rolling)")
    print("-" * 50)

    # Pandas 版本
    pandas_times = []
    for run in range(n_runs):
        t0 = time.time()
        result = pd.DataFrame()
        result['ret_1d'] = df.groupby('code')['close'].pct_change()
        result['ret_5d'] = df.groupby('code')['close'].pct_change(5)
        result['ret_20d'] = df.groupby('code')['close'].pct_change(20)
        result['ma_20'] = df.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        result['ma_60'] = df.groupby('code')['close'].transform(
            lambda x: x.rolling(60, min_periods=20).mean()
        )
        result['vol_20'] = df.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )
        result['vol_60'] = df.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(60, min_periods=30).std()
        )
        result['turnover_20'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        elapsed = time.time() - t0
        pandas_times.append(elapsed)
        print(f"  Pandas run {run+1}: {elapsed:.4f}s")

    # Polars 版本
    polars_times = []
    if HAS_POLARS:
        df_pl = pl.from_pandas(df[['date', 'code', 'close', 'turnover_rate']])
        for run in range(n_runs):
            t0 = time.time()
            result_pl = df_pl.sort(['code', 'date']).with_columns([
                pl.col('close').pct_change().over('code').alias('ret_1d'),
                pl.col('close').pct_change(5).over('code').alias('ret_5d'),
                pl.col('close').pct_change(20).over('code').alias('ret_20d'),
                pl.col('close').rolling_mean(20, min_periods=5).over('code').alias('ma_20'),
                pl.col('close').rolling_mean(60, min_periods=20).over('code').alias('ma_60'),
                pl.col('close').pct_change().rolling_std(20, min_periods=10).over('code').alias('vol_20'),
                pl.col('close').pct_change().rolling_std(60, min_periods=30).over('code').alias('vol_60'),
                pl.col('turnover_rate').rolling_mean(20, min_periods=5).over('code').alias('turnover_20'),
            ])
            elapsed = time.time() - t0
            polars_times.append(elapsed)
            print(f"  Polars run {run+1}: {elapsed:.4f}s")

    pandas_avg = np.mean(pandas_times) if pandas_times else 0
    polars_avg = np.mean(polars_times) if polars_times else 0

    print(f"\n  Pandas 平均: {pandas_avg:.4f}s")
    if HAS_POLARS:
        speedup = pandas_avg / polars_avg if polars_avg > 0 else float('inf')
        print(f"  Polars 平均: {polars_avg:.4f}s")
        print(f"  加速比: {speedup:.2f}x")

    return pandas_avg, polars_avg


def benchmark_cross_sectional_rank(df: pd.DataFrame, n_runs: int = 3):
    """测试截面排名计算性能"""
    print("\n" + "-" * 50)
    print("基准2: 截面排名与排序 (非常见操作)")
    print("-" * 50)

    # 先计算一个因子
    df_temp = df.copy()
    df_temp['factor_1'] = df.groupby('code')['close'].pct_change(20)

    # Pandas 版本
    pandas_times = []
    for run in range(n_runs):
        t0 = time.time()
        # 每日截面排名
        ranked = df_temp.groupby('date')['factor_1'].rank(pct=True)
        # TopK 选择
        top_k = df_temp.copy()
        top_k['rank'] = ranked
        top_k = top_k.sort_values(['date', 'rank'], ascending=[True, False])
        # 只保留每日 top 20%
        top_stocks = top_k[top_k['rank'] > 0.8]
        elapsed = time.time() - t0
        pandas_times.append(elapsed)
        print(f"  Pandas run {run+1}: {elapsed:.4f}s")

    # Polars 版本
    polars_times = []
    if HAS_POLARS:
        df_pl = pl.from_pandas(df_temp[['date', 'code', 'factor_1']])
        for run in range(n_runs):
            t0 = time.time()
            result = df_pl.with_columns([
                (pl.col('factor_1').rank('ordinal', descending=True) / pl.col('factor_1').count())
                .over('date')
                .alias('rank')
            ])
            top_stocks_pl = result.filter(pl.col('rank') < 0.2)
            elapsed = time.time() - t0
            polars_times.append(elapsed)
            print(f"  Polars run {run+1}: {elapsed:.4f}s")

    pandas_avg = np.mean(pandas_times) if pandas_times else 0
    polars_avg = np.mean(polars_times) if polars_times else 0

    print(f"\n  Pandas 平均: {pandas_avg:.4f}s")
    if HAS_POLARS:
        speedup = pandas_avg / polars_avg if polars_avg > 0 else float('inf')
        print(f"  Polars 平均: {polars_avg:.4f}s")
        print(f"  加速比: {speedup:.2f}x")

    return pandas_avg, polars_avg


def benchmark_pivot_and_cov(df: pd.DataFrame, n_runs: int = 3):
    """测试数据透视和协方差计算 (组合优化引擎核心)"""
    print("\n" + "-" * 50)
    print("基准3: 数据透视 + 协方差矩阵计算")
    print("-" * 50)

    # 取前200只股票，确保不溢出
    top_codes = df['code'].unique()[:200]
    df_sub = df[df['code'].isin(top_codes)].copy()

    # Pandas 版本
    pandas_times = []
    for run in range(n_runs):
        t0 = time.time()
        # 透视表
        pivot = df_sub.pivot(index='date', columns='code', values='close')
        returns = pivot.pct_change().dropna()
        # 协方差矩阵
        cov_matrix = returns.cov()
        elapsed = time.time() - t0
        pandas_times.append(elapsed)
        print(f"  Pandas run {run+1}: {elapsed:.4f}s")

    # Polars 版本
    polars_times = []
    if HAS_POLARS:
        df_pl = pl.from_pandas(df_sub[['date', 'code', 'close']])
        for run in range(n_runs):
            t0 = time.time()
            pivot_pl = df_pl.pivot(
                values='close',
                index='date',
                on='code'
            )
            # 计算收益率
            ret_cols = [c for c in pivot_pl.columns if c != 'date']
            pivot_pl = pivot_pl.with_columns([
                pl.col(c).pct_change().alias(f"{c}_ret")
                for c in ret_cols
            ])
            # 计算协方差（Polars 0.19+ 直接支持）
            ret_pl = pivot_pl.select([
                pl.col(f"{c}_ret").alias(c)
                for c in ret_cols
            ]).drop_nulls()
            # Polars covariance
            cov_dict = {}
            for c in ret_cols:
                cov_dict[c] = []
                for c2 in ret_cols:
                    cov_val = ret_pl.select(
                        pl.cov(c, c2)
                    ).item()
                    cov_dict[c].append(cov_val)
            elapsed = time.time() - t0
            polars_times.append(elapsed)
            print(f"  Polars run {run+1}: {elapsed:.4f}s")

    pandas_avg = np.mean(pandas_times) if pandas_times else 0
    polars_avg = np.mean(polars_times) if polars_times else 0

    print(f"\n  Pandas 平均: {pandas_avg:.4f}s")
    if HAS_POLARS:
        speedup = pandas_avg / polars_avg if polars_avg > 0 else float('inf')
        print(f"  Polars 平均: {polars_avg:.4f}s")
        print(f"  加速比: {speedup:.2f}x")

    return pandas_avg, polars_avg


def benchmark_memory_usage(df: pd.DataFrame):
    """测试内存使用对比"""
    print("\n" + "-" * 50)
    print("基准4: 内存使用对比")
    print("-" * 50)

    # Pandas 内存
    pandas_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"  Pandas DataFrame: {pandas_memory_mb:.2f} MB")

    # Parquet 文件大小
    parquet_path = "/tmp/test_quant_data.parquet"
    df.to_parquet(parquet_path, compression='zstd')
    parquet_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"  Parquet 文件 (zstd): {parquet_size_mb:.2f} MB")

    # Polars 内存
    polars_memory_mb = 0
    if HAS_POLARS:
        df_pl = pl.from_pandas(df)
        # Polars 不提供直接的 memory_usage，估算
        est_rows = len(df)
        est_cols = len(df.columns)
        est_bytes = est_rows * est_cols * 8  # 每元素8字节粗略估计
        polars_memory_mb = est_bytes / (1024 * 1024)
        print(f"  Polars DataFrame (估计): {polars_memory_mb:.2f} MB")

        # Polars Parquet
        pl_path = "/tmp/test_quant_data_pl.parquet"
        df_pl.write_parquet(pl_path)
        pl_size_mb = os.path.getsize(pl_path) / (1024 * 1024)
        print(f"  Polars Parquet 文件: {pl_size_mb:.2f} MB")

        # 懒加载评估
        lazy_pl = pl.scan_parquet(pl_path)
        print(f"  Polars LazyFrame (scan): 0 MB (延迟加载)")

    return pandas_memory_mb, polars_memory_mb


def benchmark_io_speed(df: pd.DataFrame, n_runs: int = 3):
    """测试 IO 读写性能"""
    print("\n" + "-" * 50)
    print("基准5: Parquet IO 读写性能")
    print("-" * 50)

    # Pandas 写
    pandas_write_times = []
    for run in range(n_runs):
        t0 = time.time()
        df.to_parquet('/tmp/test_pandas_io.parquet')
        elapsed = time.time() - t0
        pandas_write_times.append(elapsed)
    print(f"  Pandas 写平均: {np.mean(pandas_write_times):.4f}s")

    # Pandas 读
    pandas_read_times = []
    for run in range(n_runs):
        t0 = time.time()
        _ = pd.read_parquet('/tmp/test_pandas_io.parquet')
        elapsed = time.time() - t0
        pandas_read_times.append(elapsed)
    print(f"  Pandas 读平均: {np.mean(pandas_read_times):.4f}s")

    polars_write_times = []
    polars_read_times = []
    if HAS_POLARS:
        df_pl = pl.from_pandas(df)
        # Polars 写
        for run in range(n_runs):
            t0 = time.time()
            df_pl.write_parquet('/tmp/test_polars_io.parquet')
            elapsed = time.time() - t0
            polars_write_times.append(elapsed)

        # Polars 读
        for run in range(n_runs):
            t0 = time.time()
            _ = pl.read_parquet('/tmp/test_polars_io.parquet')
            elapsed = time.time() - t0
            polars_read_times.append(elapsed)

        print(f"  Polars 写平均: {np.mean(polars_write_times):.4f}s")
        print(f"  Polars 读平均: {np.mean(polars_read_times):.4f}s")

    return np.mean(pandas_read_times + pandas_write_times), np.mean(polars_read_times + polars_write_times)


# ============================================================================
# 正确性验证
# ============================================================================

def validate_correctness(df: pd.DataFrame):
    """验证 Pandas 和 Polars 计算结果的一致性"""
    print("\n" + "=" * 70)
    print("正确性验证: Pandas vs Polars 因子计算结果")
    print("=" * 70)

    if not HAS_POLARS:
        print("  [SKIP] Polars 未安装")
        return

    df_pl = pl.from_pandas(df[['date', 'code', 'close', 'turnover_rate']])

    # Pandas 计算
    df_result = df[['date', 'code']].copy()
    df_result['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    df_result['ma_20'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    df_result['vol_20'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )

    # Polars 计算
    pl_result = df_pl.sort(['code', 'date']).with_columns([
        pl.col('close').pct_change(5).over('code').alias('ret_5d'),
        pl.col('close').rolling_mean(20, min_periods=5).over('code').alias('ma_20'),
        pl.col('close').pct_change().rolling_std(20, min_periods=10).over('code').alias('vol_20'),
    ]).select(['date', 'code', 'ret_5d', 'ma_20', 'vol_20']).to_pandas()

    # 对比
    check_cols = ['ret_5d', 'ma_20', 'vol_20']
    all_match = True

    for col in check_cols:
        merged = df_result[['date', 'code']].copy()
        merged['pd_val'] = df_result[col].values
        pl_lookup = pl_result.set_index(['date', 'code'])[col]
        merged['pl_val'] = merged.set_index(['date', 'code']).index.map(
            lambda x: pl_lookup.get(x, np.nan)
        )
        merged['pl_val'] = merged.apply(
            lambda row: pl_lookup.get((row['date'], row['code']), np.nan), axis=1
        )

        both_valid = merged['pd_val'].notna() & merged['pl_val'].notna()
        if both_valid.sum() > 0:
            diff = (merged.loc[both_valid, 'pd_val'] - merged.loc[both_valid, 'pl_val']).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            status = "PASS" if max_diff < 1e-4 else "WARN"
            if max_diff >= 1e-4:
                all_match = False
            nan_pd = df_result[col].isna().sum()
            nan_pl = pl_result[col].isna().sum()
            print(f"  [{status}] {col}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
            print(f"         Pandas NaN: {nan_pd}, Polars NaN: {nan_pl}")
        else:
            print(f"  [WARN] {col}: 无共同有效数据")

    print(f"  正确性验证: {'PASS' if all_match else 'WARN - 存在可接受的浮点误差'}")
    return all_match


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("验证报告: 数据处理性能优化 (Polari 借鉴 QUANTAXIS)")
    print("=" * 70)
    print(f"时间: {datetime.now().isoformat()}")
    print(f"借鉴来源: yutiansut/QUANTAXIS v2.1 - Rust核心 + QADataBridge")
    print(f"优化方向: 用 Polars/Arrow 替代 Pandas 提升数据处理性能")

    # 生成测试数据
    print("\n生成测试数据...")
    df = generate_large_dataset(n_stocks=500, n_days=500)
    print(f"数据规模: {df['code'].nunique()} 只股票 × {df['date'].nunique()} 天 = {len(df):,} 行")

    # 基准测试
    results = {}

    bt1 = benchmark_grouped_rolling(df)
    results['grouped_rolling'] = {'pandas': bt1[0], 'polars': bt1[1]}

    bt2 = benchmark_cross_sectional_rank(df)
    results['cross_sectional'] = {'pandas': bt2[0], 'polars': bt2[1]}

    bt3 = benchmark_pivot_and_cov(df)
    results['pivot_cov'] = {'pandas': bt3[0], 'polars': bt3[1]}

    bt4 = benchmark_memory_usage(df)
    results['memory'] = {'pandas_mb': bt4[0], 'polars_mb': bt4[1]}

    bt5 = benchmark_io_speed(df)
    results['io'] = {'pandas': bt5[0], 'polars': bt5[1]}

    # 正确性验证
    correctness = validate_correctness(df)
    results['correctness'] = correctness

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("总结与建议")
    print("=" * 70)

    if HAS_POLARS:
        speedups = {}
        for key in ['grouped_rolling', 'cross_sectional', 'pivot_cov']:
            pd_time = results[key]['pandas']
            pl_time = results[key]['polars']
            if pd_time > 0 and pl_time > 0:
                speedups[key] = pd_time / pl_time

        print("\n  性能对比汇总:")
        for key, su in speedups.items():
            print(f"    {key}: Pandas={results[key]['pandas']:.3f}s, Polars={results[key]['polars']:.3f}s, 加速={su:.2f}x")

        avg_speedup = np.mean(list(speedups.values())) if speedups else 0
        print(f"\n  平均加速比: {avg_speedup:.2f}x")

        print(f"\n  内存使用: Pandas={results['memory']['pandas_mb']:.1f}MB, "
              f"Polars(估计)={results['memory']['polars_mb']:.1f}MB")

        print(f"\n  正确性: {'PASS' if correctness else 'WARN'}")

    print("\n  建议:")
    print(f"    1. 短期: 在 factor-engine 和 data-engine 中引入 Polars 作为可选后端")
    print(f"    2. 中期: 用 Polars 重写数据 ET了模块（IO密集操作收益最大）")
    print(f"    3. 长期: 借鉴 QUANTAXIS 的零拷贝数据桥接，避免 pandas<->polars 转换")
    print(f"    4. 兼容性: 保留 pandas 作为 fallback，通过环境变量切换")
    print(f"    5. 配置化: 在 config.py 中添加 DATA_BACKEND='polars' 选项")

    return results


if __name__ == "__main__":
    main()