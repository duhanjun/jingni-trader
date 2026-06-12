"""
验证测试：Polars vs Pandas 因子计算性能对比

借鉴来源：
  - Factor Engine (arXiv:2602.14138): "The library is built on Polars for its superior
    performance ... the design philosophy is centered on three core principles:
    modularity, compatibility, and extensibility."
  - AKQuant (GitHub akfamily/akquant, 1.4k+ stars): "Polars 驱动的高性能因子计算引擎，
    支持 Rank(Ts_Mean(Close, 5)) 等 Alpha101 风格公式，自动处理并行计算与数据对齐。
    Rust 核心引擎 + Zero-Copy 数据架构。"

优化方向：
  - jingni-trader 的 factor-engine 和 backtest-engine 均基于 pandas，
    评估切换到 Polars 的性能收益。
  - 验证 Polars 的零拷贝和惰性计算能力在大规模数据场景下的加速效果。

测试内容：
  1. 单因子计算性能对比（5日收益、20日波动率、换手率等）
  2. 批量因子计算性能对比
  3. 数据分组操作性能对比
  4. 内存占用对比
  5. 不同数据规模下的扩展性测试
"""

import sys
import os
import time
import logging
import unittest
import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("警告：polars 未安装，将跳过 Polars 相关测试")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test-polars-perf")


# ============================================================================
# 测试数据生成
# ============================================================================

def generate_test_data(n_codes: int = 50, n_days: int = 500, seed: int = 42):
    """生成模拟 A 股日线数据"""
    np.random.seed(seed)
    codes = [f"{600000 + i % 10000:06d}.{'SH' if i % 2 == 0 else 'SZ'}" for i in range(n_codes)]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    rows = []
    for code in codes:
        price = np.random.uniform(5, 100)
        for date in dates:
            ret = np.random.normal(0.0003, 0.02)
            price = max(price * (1 + ret), 1.0)
            vol = int(np.random.lognormal(12, 0.8))
            rows.append({
                'code': code,
                'date': date,
                'close': round(price, 2),
                'volume': vol,
                'amount': round(price * vol, 2),
                'turnover_rate': round(abs(np.random.normal(0.015, 0.01)), 4),
                'change_pct': round(ret * 100, 4),
            })

    df_pd = pd.DataFrame(rows).sort_values(['code', 'date']).reset_index(drop=True)
    df_pl = pl.from_pandas(df_pd) if HAS_POLARS else None
    return df_pd, df_pl, n_codes, n_days


# ============================================================================
# Pandas 因子计算
# ============================================================================

def pandas_compute_ret_5d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].pct_change(5).rename("ret_5d")


def pandas_compute_volatility_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    ).rename("volatility_20d")


def pandas_compute_turnover_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    ).rename("turnover_20d")


def pandas_compute_volume_ratio(df: pd.DataFrame) -> pd.Series:
    vol_20 = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    return (df['volume'] / vol_20.replace(0, np.nan)).rename("volume_ratio")


def pandas_compute_reversal_20d(df: pd.DataFrame) -> pd.Series:
    ret_20d = df.groupby('code')['close'].pct_change(20)
    return (-ret_20d).rename("reversal_20d")


def pandas_batch_factors(df: pd.DataFrame) -> pd.DataFrame:
    """批量计算所有因子（模拟 FactorEngine.compute_a_share_factors）"""
    result = pd.DataFrame()
    result['code'] = df['code']
    result['date'] = df['date']
    result['ret_5d'] = pandas_compute_ret_5d(df).values
    result['volatility_20d'] = pandas_compute_volatility_20d(df).values
    result['turnover_20d'] = pandas_compute_turnover_20d(df).values
    result['volume_ratio'] = pandas_compute_volume_ratio(df).values
    result['reversal_20d'] = pandas_compute_reversal_20d(df).values
    result['ret_1d'] = df.groupby('code')['close'].pct_change().values
    result['ret_20d'] = df.groupby('code')['close'].pct_change(20).values
    result['ret_60d'] = df.groupby('code')['close'].pct_change(60).values
    result['reversal_5d'] = (-df.groupby('code')['close'].pct_change(5)).values

    # 换手率变化
    t5 = df.groupby('code')['turnover_rate'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    result['turnover_change'] = (t5 / result['turnover_20d'].replace(0, np.nan) - 1).values

    # 资金流向
    ret = df.groupby('code')['close'].pct_change()
    mf_raw = ret * df['amount']
    result['money_flow_20d'] = mf_raw.groupby(df['code']).transform(
        lambda x: x.rolling(20, min_periods=5).sum()
    ).values

    return result


# ============================================================================
# Polars 因子计算
# ============================================================================

if HAS_POLARS:
    def polars_compute_ret_5d(df: pl.DataFrame) -> pl.Series:
        return df.with_columns(
            df.group_by("code", maintain_order=True).agg(
                pl.col("close").pct_change(5).alias("ret_5d")
            ).explode("ret_5d").get_column("ret_5d")
        ).get_column("ret_5d")


    def polars_compute_volatility_20d(df: pl.DataFrame) -> pl.Series:
        ret = df.with_columns(
            pl.col("close").pct_change().over("code").alias("ret")
        )
        return ret.with_columns(
            pl.col("ret").rolling_std(20, min_periods=10).over("code").alias("volatility_20d")
        ).get_column("volatility_20d")


    def polars_compute_turnover_20d(df: pl.DataFrame) -> pl.Series:
        return df.with_columns(
            pl.col("turnover_rate").rolling_mean(20, min_periods=5).over("code").alias("turnover_20d")
        ).get_column("turnover_20d")


    def polars_compute_volume_ratio(df: pl.DataFrame) -> pl.Series:
        vol_20 = df.with_columns(
            pl.col("volume").rolling_mean(20, min_periods=5).over("code").alias("vol_20d")
        ).get_column("vol_20d")
        return (df.get_column("volume") / vol_20).alias("volume_ratio")


    def polars_compute_reversal_20d(df: pl.DataFrame) -> pl.Series:
        ret_20 = df.with_columns(
            pl.col("close").pct_change(20).over("code").alias("ret_20d")
        ).get_column("ret_20d")
        return (-ret_20).alias("reversal_20d")


    def polars_batch_factors(df: pl.DataFrame) -> pl.DataFrame:
        """使用 Polars 惰性计算批量生成所有因子"""
        df = df.with_columns([
            pl.col("close").pct_change().over("code").alias("ret_1d"),
            pl.col("close").pct_change(5).over("code").alias("ret_5d"),
            pl.col("close").pct_change(20).over("code").alias("ret_20d"),
            pl.col("close").pct_change(60).over("code").alias("ret_60d"),
        ]).with_columns([
            (-pl.col("ret_5d")).alias("reversal_5d"),
            (-pl.col("ret_20d")).alias("reversal_20d"),
        ])

        df = df.with_columns(
            pl.col("close").pct_change().rolling_std(20, min_periods=10)
            .over("code").alias("volatility_20d")
        )

        df = df.with_columns(
            pl.col("turnover_rate").rolling_mean(20, min_periods=5)
            .over("code").alias("turnover_20d"),
            pl.col("turnover_rate").rolling_mean(5, min_periods=3)
            .over("code").alias("turnover_5d"),
        ).with_columns(
            (pl.col("turnover_5d") / pl.col("turnover_20d") - 1).alias("turnover_change")
        )

        df = df.with_columns(
            pl.col("volume").rolling_mean(20, min_periods=5)
            .over("code").alias("volume_20d"),
        ).with_columns(
            (pl.col("volume") / pl.col("volume_20d")).alias("volume_ratio")
        )

        df = df.with_columns(
            (pl.col("ret_1d") * pl.col("amount")).alias("mf_raw")
        ).with_columns(
            pl.col("mf_raw").rolling_sum(20, min_periods=5)
            .over("code").alias("money_flow_20d")
        )

        return df.select([
            "code", "date", "ret_1d", "ret_5d", "ret_20d", "ret_60d",
            "reversal_5d", "reversal_20d", "volatility_20d",
            "turnover_20d", "turnover_5d", "turnover_change",
            "volume_ratio", "money_flow_20d"
        ])


def measure_time(func, *args, **kwargs):
    """测量函数执行时间（毫秒）"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def measure_memory_mb():
    """粗略估计当前内存使用（MB）"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1


# ============================================================================
# 单元测试
# ============================================================================

class TestPerformanceComparison(unittest.TestCase):
    """性能对比测试"""

    SMALL_DATA_N_CODES = 10
    SMALL_DATA_N_DAYS = 252
    MEDIUM_DATA_N_CODES = 200
    MEDIUM_DATA_N_DAYS = 500

    @classmethod
    def setUpClass(cls):
        cls.small_pd, cls.small_pl, _, _ = generate_test_data(
            cls.SMALL_DATA_N_CODES, cls.SMALL_DATA_N_DAYS
        )
        cls.medium_pd, cls.medium_pl, _, _ = generate_test_data(
            cls.MEDIUM_DATA_N_CODES, cls.MEDIUM_DATA_N_DAYS
        )

    def test_pandas_correctness(self):
        """验证 Pandas 因子计算正确性"""
        df = self.small_pd.copy()
        result = pandas_batch_factors(df)
        self.assertEqual(len(result), len(df))
        self.assertIn('ret_5d', result.columns)
        self.assertIn('volatility_20d', result.columns)

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_polars_correctness(self):
        """验证 Polars 因子计算正确性"""
        df = self.small_pl.clone()
        result = polars_batch_factors(df)
        result_pd = result.to_pandas()
        self.assertEqual(len(result_pd), len(self.small_pd))
        self.assertIn('ret_5d', result_pd.columns)
        self.assertIn('volatility_20d', result_pd.columns)

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_pandas_vs_polars_consistency_small(self):
        """验证小数据集下 Pandas 与 Polars 计算结果一致"""
        pd_result = pandas_batch_factors(self.small_pd)
        pl_result = polars_batch_factors(self.small_pl).to_pandas()

        for col in ['ret_5d', 'reversal_20d', 'volatility_20d']:
            if col not in pd_result.columns or col not in pl_result.columns:
                continue
            pd_vals = pd_result[col].values
            pl_vals = pl_result[col].values
            mask = ~(np.isnan(pd_vals) & np.isnan(pl_vals))
            ratio = np.allclose(pd_vals[mask], pl_vals[mask], rtol=1e-4, equal_nan=True)
            self.assertTrue(ratio, f"列 {col} 不一致")

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_performance_ret_5d_small(self):
        """小数据集：5日收益率性能对比"""
        _, pd_time = measure_time(pandas_compute_ret_5d, self.small_pd)
        _, pl_time = measure_time(polars_compute_ret_5d, self.small_pl)
        logger.info(f"ret_5d 小数据集: Pandas={pd_time:.2f}ms, Polars={pl_time:.2f}ms, "
                     f"加速比={pd_time/pl_time:.2f}x")
        # 小数据集可能差距不大，仅记录

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_performance_volatility_20d_small(self):
        """小数据集：20日波动率性能对比"""
        _, pd_time = measure_time(pandas_compute_volatility_20d, self.small_pd)
        _, pl_time = measure_time(polars_compute_volatility_20d, self.small_pl)
        logger.info(f"volatility_20d 小数据集: Pandas={pd_time:.2f}ms, Polars={pl_time:.2f}ms, "
                     f"加速比={pd_time/pl_time:.2f}x")

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_performance_batch_small(self):
        """小数据集：批量因子计算性能对比"""
        _, pd_time = measure_time(pandas_batch_factors, self.small_pd)
        _, pl_time = measure_time(polars_batch_factors, self.small_pl)
        logger.info(f"批量因子 小数据集: Pandas={pd_time:.2f}ms, Polars={pl_time:.2f}ms, "
                     f"加速比={pd_time/pl_time:.2f}x")

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_performance_ret_5d_medium(self):
        """中等数据集：5日收益率性能对比 (200只股票 x 500天 = 10万行)"""
        _, pd_time = measure_time(pandas_compute_ret_5d, self.medium_pd)
        _, pl_time = measure_time(polars_compute_ret_5d, self.medium_pl)
        logger.info(f"ret_5d 中等数据集: Pandas={pd_time:.2f}ms, Polars={pl_time:.2f}ms, "
                     f"加速比={pd_time/pl_time:.2f}x")

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_performance_volatility_20d_medium(self):
        """中等数据集：20日波动率性能对比"""
        _, pd_time = measure_time(pandas_compute_volatility_20d, self.medium_pd)
        _, pl_time = measure_time(polars_compute_volatility_20d, self.medium_pl)
        logger.info(f"volatility_20d 中等数据集: Pandas={pd_time:.2f}ms, Polars={pl_time:.2f}ms, "
                     f"加速比={pd_time/pl_time:.2f}x")

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_performance_batch_medium(self):
        """中等数据集：批量因子计算性能对比"""
        _, pd_time = measure_time(pandas_batch_factors, self.medium_pd)
        _, pl_time = measure_time(polars_batch_factors, self.medium_pl)
        logger.info(f"批量因子 中等数据集: Pandas={pd_time:.2f}ms, Polars={pl_time:.2f}ms, "
                     f"加速比={pd_time/pl_time:.2f}x")


class TestScalability(unittest.TestCase):
    """扩展性测试"""

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_scale_1k_stocks(self):
        """1000只股票 x 252天 = 25万行"""
        pd_df, pl_df, _, _ = generate_test_data(n_codes=1000, n_days=252)
        _, pd_time = measure_time(pandas_batch_factors, pd_df)
        _, pl_time = measure_time(polars_batch_factors, pl_df)
        n_rows = len(pd_df)
        logger.info(f"扩展性 {n_rows}行: Pandas={pd_time:.2f}ms, Polars={pl_time:.2f}ms, "
                     f"加速比={pd_time/pl_time:.2f}x")

    @unittest.skipIf(not HAS_POLARS, "Polars 未安装")
    def test_groupby_performance(self):
        """测试 groupby 操作的性能差异"""
        pd_df, pl_df, _, _ = generate_test_data(n_codes=500, n_days=252)

        # Pandas groupby
        start = time.perf_counter()
        pd_mean = pd_df.groupby('code')['close'].transform('mean')
        pd_time = (time.perf_counter() - start) * 1000

        # Polars window function
        start = time.perf_counter()
        pl_mean = pl_df.with_columns(
            pl.col("close").mean().over("code").alias("close_mean")
        ).get_column("close_mean").to_pandas()
        pl_time = (time.perf_counter() - start) * 1000

        logger.info(f"groupby mean 500股: Pandas={pd_time:.2f}ms, Polars={pl_time:.2f}ms, "
                     f"加速比={pd_time/pl_time:.2f}x")


# ============================================================================
# 综合基准测试运行器
# ============================================================================

def run_benchmark_suite():
    """运行完整基准测试套件"""
    logger.info("=" * 70)
    logger.info("  Polars vs Pandas 因子计算性能对比基准测试")
    logger.info("=" * 70)

    scales = [
        ("小规模", 10, 252),       # ~2,520 行
        ("中规模", 200, 252),      # ~50,400 行
        ("中大规模", 500, 252),    # ~126,000 行
        ("大规模", 1000, 252),     # ~252,000 行
    ]

    results = []
    for label, n_codes, n_days in scales:
        logger.info(f"\n--- {label} ({n_codes}只股票 × {n_days}天 = {n_codes * n_days:,}行) ---")
        pd_df, pl_df, _, _ = generate_test_data(n_codes=n_codes, n_days=n_days)

        # 预热
        _ = pandas_batch_factors(pd_df)
        if HAS_POLARS:
            _ = polars_batch_factors(pl_df)

        # 测试
        _, pd_time = measure_time(pandas_batch_factors, pd_df)

        if HAS_POLARS:
            _, pl_time = measure_time(polars_batch_factors, pl_df)
            speedup = pd_time / pl_time
            logger.info(f"  Pandas: {pd_time:.1f}ms")
            logger.info(f"  Polars: {pl_time:.1f}ms")
            logger.info(f"  加速比: {speedup:.2f}x")
        else:
            pl_time = None
            speedup = None
            logger.info(f"  Pandas: {pd_time:.1f}ms")
            logger.info(f"  Polars: N/A (未安装)")

        results.append({
            "label": label,
            "n_rows": n_codes * n_days,
            "pandas_ms": pd_time,
            "polars_ms": pl_time,
            "speedup": speedup,
        })

    logger.info("\n" + "=" * 70)
    logger.info("  基准测试总结")
    logger.info("=" * 70)
    logger.info(f"{'规模':<12} {'行数':<12} {'Pandas(ms)':<12} {'Polars(ms)':<12} {'加速比':<10}")
    logger.info("-" * 58)
    for r in results:
        pl_str = f"{r['polars_ms']:.1f}" if r['polars_ms'] else "N/A"
        sp_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        logger.info(f"{r['label']:<12} {r['n_rows']:<12,} {r['pandas_ms']:<12.1f} {pl_str:<12} {sp_str:<10}")

    return results


if __name__ == "__main__":
    run_benchmark_suite()
    print("\n" + "=" * 70)
    print("  单元测试")
    print("=" * 70)
    unittest.main(verbosity=2)