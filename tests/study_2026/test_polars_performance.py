"""
验证方向: Polars 高性能数据处理 (Data Pipeline Optimization)
借鉴来源: akquant (Rust-based core + Polars engine) + Factor Engine paper (Polars-based computation)
日期: 2026-06-14

优化思路:
    当前 data-engine 和 factor-engine 使用 pandas 进行数据处理，pandas 的
    groupby + rolling 操作在大规模数据下存在性能瓶颈。借鉴 akquant 的
    Rust+Polars 混合架构和 Factor Engine 论文中的 Polars 实践，评估使用
    Polars 替代 pandas 的可行性和性能收益。

    验证内容:
    1. Pandas vs Polars eager vs Polars lazy 的性能对比
    2. 因子计算场景的加速比
    3. 数据清洗场景的加速比
    4. 内存占用对比
"""

import sys
import os
import time
import unittest
import tempfile

import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("警告: Polars 未安装，将跳过 Polars 对比测试")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================
# 1. 测试数据生成
# ============================================================

def generate_large_dataset(n_stocks=100, n_days=500):
    """
    生成大规模测试数据集
    模拟 A 股全市场约 500 只股票、3 年数据量
    """
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=n_days, freq='B')
    codes = [f'{i:06d}.SZ' for i in range(1, n_stocks + 1)]

    rows = []
    for code in codes:
        start_price = np.random.uniform(5, 200)
        price = start_price
        # 使用向量化生成以提高速度
        returns = np.random.normal(0.0003, 0.018, n_days)
        returns[0] = 0
        prices = start_price * np.cumprod(1 + returns)

        opens = prices * (1 + np.random.normal(0, 0.003, n_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        volumes = np.random.lognormal(15, 0.5, n_days)

        for j in range(n_days):
            rows.append({
                'code': code,
                'date': dates[j],
                'open': round(opens[j], 2),
                'high': round(highs[j], 2),
                'low': round(lows[j], 2),
                'close': round(prices[j], 2),
                'volume': int(volumes[j]),
            })

    return pd.DataFrame(rows)


# ============================================================
# 2. Pandas 实现 (对标现有代码)
# ============================================================

def pandas_compute_factors(df_pd: pd.DataFrame, verbose=False) -> pd.DataFrame:
    """使用 pandas 计算因子（模拟现有 factor-engine 的方式）"""
    df = df_pd.sort_values(['code', 'date']).copy()

    result = df[['code', 'date']].copy()

    if verbose:
        t0 = time.perf_counter()

    # 收益率因子
    result['ret_1d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change())
    result['ret_5d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(5))
    result['ret_20d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(20))
    result['ret_60d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(60))

    # 反转因子
    result['reversal_5d'] = -result['ret_5d']
    result['reversal_20d'] = -result['ret_20d']

    # 波动率
    result['volatility_20d'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )

    # 成交量
    result['volume_20d'] = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    result['volume_ratio'] = df['volume'] / result['volume_20d'].replace(0, np.nan)

    # 均线和价格位置
    result['ma5'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    result['ma20'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    result['ma60'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(60, min_periods=30).mean()
    )

    # 布林带位置
    result['bb_position'] = (result['close'] - result['ma20']) / (result['volatility_20d'] * result['close'])

    # 最高最低价相关
    result['high_20d'] = df.groupby('code')['high'].transform(
        lambda x: x.rolling(20, min_periods=10).max()
    )
    result['low_20d'] = df.groupby('code')['low'].transform(
        lambda x: x.rolling(20, min_periods=10).min()
    )
    result['hl_ratio'] = (result['close'] - result['low_20d']) / (
        result['high_20d'] - result['low_20d']
    ).replace(0, np.nan)

    # ATR
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df.groupby('code')['close'].shift(1)),
            abs(df['low'] - df.groupby('code')['close'].shift(1))
        )
    )
    result['atr_14'] = df[['code']].copy()
    result['atr_14']['_tr'] = tr
    result['atr_14'] = result.groupby('code')['_tr'].transform(
        lambda x: x.rolling(14, min_periods=7).mean()
    )

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  Pandas 因子计算: {elapsed:.3f}s")

    return result


# ============================================================
# 3. Polars 实现
# ============================================================

def polars_compute_factors(df_pd: pd.DataFrame, use_lazy: bool = False, verbose=False) -> pd.DataFrame:
    """使用 Polars 计算因子"""
    if verbose:
        t0 = time.perf_counter()

    # 转换
    df_pl = pl.from_pandas(df_pd)

    if use_lazy:
        df_pl = df_pl.lazy()

    # 构建表达式
    exprs = [
        pl.col('code'),
        pl.col('date'),
        pl.col('close'),
        pl.col('volume'),
        pl.col('high'),
        pl.col('low'),
    ]

    # 收益率因子
    exprs.append(pl.col('close').pct_change().over('code').alias('ret_1d'))
    exprs.append(pl.col('close').pct_change(5).over('code').alias('ret_5d'))
    exprs.append(pl.col('close').pct_change(20).over('code').alias('ret_20d'))
    exprs.append(pl.col('close').pct_change(60).over('code').alias('ret_60d'))

    # 反转因子
    exprs.append((-pl.col('close').pct_change(5).over('code')).alias('reversal_5d'))
    exprs.append((-pl.col('close').pct_change(20).over('code')).alias('reversal_20d'))

    # 波动率
    exprs.append(
        pl.col('close').pct_change().rolling_std(20, min_periods=10).over('code').alias('volatility_20d')
    )

    # 成交量
    exprs.append(
        pl.col('volume').rolling_mean(20, min_periods=5).over('code').alias('volume_20d')
    )
    exprs.append(
        (pl.col('volume') / pl.col('volume').rolling_mean(20, min_periods=5).over('code')).alias('volume_ratio')
    )

    # 均线
    exprs.append(
        pl.col('close').rolling_mean(5, min_periods=3).over('code').alias('ma5')
    )
    exprs.append(
        pl.col('close').rolling_mean(20, min_periods=10).over('code').alias('ma20')
    )
    exprs.append(
        pl.col('close').rolling_mean(60, min_periods=30).over('code').alias('ma60')
    )

    # 最高最低价
    exprs.append(
        pl.col('high').rolling_max(20, min_periods=10).over('code').alias('high_20d')
    )
    exprs.append(
        pl.col('low').rolling_min(20, min_periods=10).over('code').alias('low_20d')
    )

    if use_lazy:
        result_lazy = df_pl.with_columns(exprs[6:]).select(exprs[:6] + exprs[6:])
        result = result_lazy.collect()
    else:
        result = df_pl.with_columns(exprs[6:]).select(exprs[:6] + exprs[6:])

    if verbose:
        elapsed = time.perf_counter() - t0
        label = "Lazy" if use_lazy else "Eager"
        print(f"  Polars ({label}) 因子计算: {elapsed:.3f}s")

    return result.to_pandas()


# ============================================================
# 4. Polars 数据清洗
# ============================================================

def pandas_clean_data(df_pd: pd.DataFrame, verbose=False) -> pd.DataFrame:
    """Pandas 数据清洗"""
    if verbose:
        t0 = time.perf_counter()

    df = df_pd.copy()
    # 去重
    df = df.drop_duplicates(subset=['code', 'date'])
    # 排序
    df = df.sort_values(['code', 'date'])
    # 剔除停牌
    df = df[df['volume'] > 0]
    # 前向填充缺失
    df = df.set_index(['code', 'date'])
    df = df.groupby('code').apply(lambda x: x.ffill()).reset_index(level=0, drop=True)
    df = df.reset_index()
    # 过滤极端值
    df['close'] = df['close'].clip(lower=0.01)
    # 添加变化率列
    df['change_pct'] = df.groupby('code')['close'].transform(lambda x: x.pct_change() * 100)
    # 标记涨跌停
    df['is_limit_up'] = df['change_pct'] >= 9.9
    df['is_limit_down'] = df['change_pct'] <= -9.9

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  Pandas 数据清洗: {elapsed:.3f}s")

    return df


def polars_clean_data(df_pd: pd.DataFrame, use_lazy: bool = False, verbose=False) -> pd.DataFrame:
    """Polars 数据清洗"""
    if verbose:
        t0 = time.perf_counter()

    df_pl = pl.from_pandas(df_pd)

    if use_lazy:
        df_pl = df_pl.lazy()

    exprs = [
        pl.col('code'),
        pl.col('date'),
        pl.col('open'),
        pl.col('high'),
        pl.col('low'),
        pl.col('close').clip(0.01, None),
        pl.col('volume'),
    ]

    # 去重 + 排序
    if use_lazy:
        result = df_pl.unique(subset=['code', 'date']).sort(['code', 'date'])
    else:
        result = df_pl.unique(subset=['code', 'date']).sort(['code', 'date'])

    # 过滤停牌
    result = result.filter(pl.col('volume') > 0)

    # 前向填充
    result = result.with_columns(
        pl.col('close').forward_fill().over('code'),
        pl.col('open').forward_fill().over('code'),
        pl.col('high').forward_fill().over('code'),
        pl.col('low').forward_fill().over('code'),
        pl.col('volume').forward_fill().over('code'),
    )

    # 涨跌幅
    result = result.with_columns(
        (pl.col('close').pct_change().over('code') * 100).alias('change_pct')
    )

    # 涨跌停标记
    result = result.with_columns([
        (pl.col('change_pct') >= 9.9).alias('is_limit_up'),
        (pl.col('change_pct') <= -9.9).alias('is_limit_down'),
    ])

    if use_lazy:
        result = result.collect()

    if verbose:
        elapsed = time.perf_counter() - t0
        label = "Lazy" if use_lazy else "Eager"
        print(f"  Polars ({label}) 数据清洗: {elapsed:.3f}s")

    return result.to_pandas()


# ============================================================
# 5. I/O 性能对比
# ============================================================

def benchmark_io(df_pd: pd.DataFrame, verbose=False) -> dict:
    """对比 Parquet 读写性能"""
    results = {}

    with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
        # Pandas 写入
        t0 = time.perf_counter()
        df_pd.to_parquet(tmp.name, index=False)
        results['pandas_write'] = time.perf_counter() - t0

        # Pandas 读取
        t0 = time.perf_counter()
        _ = pd.read_parquet(tmp.name)
        results['pandas_read'] = time.perf_counter() - t0

    if HAS_POLARS:
        with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
            df_pl = pl.from_pandas(df_pd)

            # Polars 写入
            t0 = time.perf_counter()
            df_pl.write_parquet(tmp.name)
            results['polars_write'] = time.perf_counter() - t0

            # Polars 读取
            t0 = time.perf_counter()
            _ = pl.read_parquet(tmp.name)
            results['polars_read'] = time.perf_counter() - t0

    return results


# ============================================================
# 6. 测试用例
# ============================================================

@unittest.skipIf(not HAS_POLARS, "Polars 未安装")
class TestPolarsPerformance(unittest.TestCase):
    """Polars 性能验证测试"""

    @classmethod
    def setUpClass(cls):
        cls.data_small = generate_large_dataset(n_stocks=50, n_days=252)
        cls.data_large = generate_large_dataset(n_stocks=200, n_days=500)
        print(f"\n  小数据集: {len(cls.data_small)} 行 ({cls.data_small['code'].nunique()} 只股票)")
        print(f"  大数据集: {len(cls.data_large)} 行 ({cls.data_large['code'].nunique()} 只股票)")

    def test_factor_correctness(self):
        """测试: Polars 因子计算结果应与 Pandas 一致"""
        pd_result = pandas_compute_factors(self.data_small)
        pl_result = polars_compute_factors(self.data_small, use_lazy=False)
        pl_lazy_result = polars_compute_factors(self.data_small, use_lazy=True)

        # 比较关键列
        compare_cols = ['ret_1d', 'ret_5d', 'volatility_20d', 'volume_ratio', 'ma20']
        for col in compare_cols:
            pd_vals = pd_result[col].fillna(0).values
            pl_vals = pl_result[col].fillna(0).values
            pl_lazy_vals = pl_lazy_result[col].fillna(0).values

            # 数值精度容忍度
            diff_eager = np.max(np.abs(pd_vals - pl_vals))
            diff_lazy = np.max(np.abs(pd_vals - pl_lazy_vals))

            self.assertLess(diff_eager, 0.1, f"Polars Eager {col} 差异过大: {diff_eager}")
            self.assertLess(diff_lazy, 0.1, f"Polars Lazy {col} 差异过大: {diff_lazy}")

    def test_factor_performance_small(self):
        """测试: 小数据集因子计算性能对比"""
        n_runs = 5

        t0 = time.perf_counter()
        for _ in range(n_runs):
            pandas_compute_factors(self.data_small)
        pd_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            polars_compute_factors(self.data_small, use_lazy=False)
        pl_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            polars_compute_factors(self.data_small, use_lazy=True)
        pl_lazy_time = (time.perf_counter() - t0) / n_runs

        print(f"\n  因子计算性能 ({len(self.data_small)} 行, {n_runs} 次平均):")
        print(f"    Pandas:       {pd_time*1000:.1f} ms")
        print(f"    Polars Eager: {pl_time*1000:.1f} ms  ({pd_time/pl_time:.1f}x 加速)")
        print(f"    Polars Lazy:  {pl_lazy_time*1000:.1f} ms  ({pd_time/pl_lazy_time:.1f}x 加速)")

        # Polars 应该不慢于 Pandas
        self.assertLessEqual(pl_lazy_time, pd_time * 1.2,
                            f"Polars 性能不达预期: {pd_time/pl_lazy_time:.1f}x")

    def test_factor_performance_large(self):
        """测试: 大数据集因子计算性能对比"""
        n_runs = 3

        t0 = time.perf_counter()
        for _ in range(n_runs):
            pandas_compute_factors(self.data_large)
        pd_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            polars_compute_factors(self.data_large, use_lazy=True)
        pl_lazy_time = (time.perf_counter() - t0) / n_runs

        print(f"\n  因子计算性能 ({len(self.data_large)} 行, {n_runs} 次平均):")
        print(f"    Pandas:      {pd_time:.3f}s")
        print(f"    Polars Lazy: {pl_lazy_time:.3f}s  ({pd_time/pl_lazy_time:.1f}x 加速)")

    def test_data_cleaning_performance(self):
        """测试: 数据清洗性能对比"""
        n_runs = 5

        t0 = time.perf_counter()
        for _ in range(n_runs):
            pandas_clean_data(self.data_small)
        pd_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            polars_clean_data(self.data_small, use_lazy=True)
        pl_time = (time.perf_counter() - t0) / n_runs

        print(f"\n  数据清洗性能 ({len(self.data_small)} 行, {n_runs} 次平均):")
        print(f"    Pandas:      {pd_time*1000:.1f} ms")
        print(f"    Polars Lazy: {pl_time*1000:.1f} ms  ({pd_time/pl_time:.1f}x 加速)")

    def test_io_performance(self):
        """测试: I/O 性能对比"""
        results = benchmark_io(self.data_large, verbose=True)
        print(f"\n  I/O 性能 ({len(self.data_large)} 行):")
        print(f"    Pandas Write: {results.get('pandas_write', 0)*1000:.1f} ms")
        print(f"    Pandas Read:  {results.get('pandas_read', 0)*1000:.1f} ms")
        if 'polars_write' in results:
            print(f"    Polars Write: {results['polars_write']*1000:.1f} ms")
            print(f"    Polars Read:  {results['polars_read']*1000:.1f} ms")

    def test_memory_usage(self):
        """测试: 内存占用对比"""
        import sys as _sys

        pd_mem = self.data_large.memory_usage(deep=True).sum()
        pl_df = pl.from_pandas(self.data_large)
        pl_mem = pl_df.estimated_size()

        print(f"\n  内存占用 ({len(self.data_large)} 行):")
        print(f"    Pandas DataFrame: {pd_mem / 1024**2:.1f} MB")
        print(f"    Polars DataFrame: {pl_mem / 1024**2:.1f} MB")
        print(f"    节省: {(1 - pl_mem/pd_mem)*100:.1f}%")


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Polars 高性能数据处理验证测试")
    print("借鉴来源: akquant + Factor Engine (Polars-based)")
    print("=" * 60)

    if not HAS_POLARS:
        print("\n注意: Polars 未安装，对比测试将跳过。")
        print("如需安装 Polars: pip install polars")
        # 只运行 Pandas 基准测试
        data = generate_large_dataset(n_stocks=50, n_days=252)
        print(f"\n生成测试数据: {len(data)} 行")
        print(f"\nPandas 基准测试:")
        t0 = time.perf_counter()
        pandas_compute_factors(data, verbose=True)
        print(f"  总耗时: {time.perf_counter() - t0:.3f}s")
    else:
        unittest.main(verbosity=2)