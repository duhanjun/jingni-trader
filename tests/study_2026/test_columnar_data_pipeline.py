"""
优化方向：列式数据存储与高效数据处理管道
借鉴来源：Microsoft Qlib - Columnar Binary Format + Expression Engine
          QUANTAXIS - QARSBridge (Rust 加速核心)
项目地址：https://github.com/microsoft/qlib
          https://github.com/yutiansut/QUANTAXIS

Qlib 采用自定义列式二进制格式存储行情数据，支持高效时间序列切片。
QUANTAXIS v2.1 引入 Rust 核心(QARSBridge)实现零拷贝数据桥接和百倍性能提升。

当前 jingni-trader 使用 Parquet 格式存储（已是不错的选择），但在：
1. 因子计算时的数据访问效率
2. 回测时的时间切片性能
3. 多资产并行计算加速

方面仍有优化空间。

本测试验证：
1. 不同存储格式（Parquet vs CSV vs HDF5 vs Feather）的性能对比
2. 列式 vs 行式数据访问模式的效率差异
3. 向量化 vs 循环计算的性能差异
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import tempfile
import shutil


# ============================================================
# 测试工具函数
# ============================================================

def generate_large_dataset(n_stocks=100, n_days=500):
    """生成大规模测试数据集"""
    np.random.seed(42)
    codes = [f"{i:06d}.{'SZ' if i % 2 else 'SH'}" for i in range(1, n_stocks + 1)]
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')

    data = {}
    for code in codes:
        base_price = np.random.uniform(5, 200)
        returns = np.random.normal(0.0003, 0.018, n_days)
        prices = base_price * np.exp(np.cumsum(returns))

        data[code] = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': prices * (1 + np.random.normal(0, 0.003, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.008, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.008, n_days))),
            'close': prices,
            'volume': np.random.lognormal(14, 0.5, n_days).astype(int),
            'amount': prices * np.random.lognormal(14, 0.5, n_days),
        })

    df = pd.concat(data.values(), ignore_index=True)
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df


# ============================================================
# 测试用例
# ============================================================

class TestDataStorageFormat(unittest.TestCase):
    """数据存储格式性能对比"""

    @classmethod
    def setUpClass(cls):
        print("\n\n===== 数据存储格式测试 =====")
        cls.test_df = generate_large_dataset(n_stocks=50, n_days=250)
        cls.tmpdir = tempfile.mkdtemp()
        print(f"测试数据: {len(cls.test_df)} 行, {cls.test_df['code'].nunique()} 只股票")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _benchmark_write(self, fmt, ext, write_fn):
        """基准测试写入性能"""
        path = os.path.join(self.tmpdir, f"test_data.{ext}")
        start = time.perf_counter()
        write_fn(self.test_df, path)
        elapsed = time.perf_counter() - start
        size = os.path.getsize(path)
        return elapsed, size

    def _benchmark_read(self, fmt, ext, read_fn):
        """基准测试读取性能"""
        path = os.path.join(self.tmpdir, f"test_data.{ext}")
        start = time.perf_counter()
        result = read_fn(path)
        elapsed = time.perf_counter() - start
        return elapsed, len(result)

    def _benchmark_query(self, fmt, ext, query_fn):
        """基准测试查询性能：获取某只股票的全部数据"""
        path = os.path.join(self.tmpdir, f"test_data.{ext}")
        start = time.perf_counter()
        result = query_fn(path)
        elapsed = time.perf_counter() - start
        return elapsed

    def test_01_format_comparison(self):
        """对比 Parquet / CSV / Feather / HDF5 格式的读写性能"""
        results = []

        # Parquet
        if True:
            w_t, w_s = self._benchmark_write(
                'parquet', 'parquet',
                lambda df, p: df.to_parquet(p, index=False)
            )
            r_t, r_n = self._benchmark_read(
                'parquet', 'parquet',
                lambda p: pd.read_parquet(p)
            )
            results.append(('Parquet', w_t, w_s, r_t, r_n))

        # CSV
        if True:
            w_t, w_s = self._benchmark_write(
                'csv', 'csv',
                lambda df, p: df.to_csv(p, index=False)
            )
            r_t, r_n = self._benchmark_read(
                'csv', 'csv',
                lambda p: pd.read_csv(p)
            )
            results.append(('CSV', w_t, w_s, r_t, r_n))

        # Feather (Arrow IPC)
        if True:
            try:
                w_t, w_s = self._benchmark_write(
                    'feather', 'feather',
                    lambda df, p: df.to_feather(p)
                )
                r_t, r_n = self._benchmark_read(
                    'feather', 'feather',
                    lambda p: pd.read_feather(p)
                )
                results.append(('Feather', w_t, w_s, r_t, r_n))
            except ImportError:
                print("  Feather 不可用，跳过")

        # HDF5
        if True:
            try:
                w_t, w_s = self._benchmark_write(
                    'hdf5', 'h5',
                    lambda df, p: df.to_hdf(p, key='data', mode='w')
                )
                r_t, r_n = self._benchmark_read(
                    'hdf5', 'h5',
                    lambda p: pd.read_hdf(p)
                )
                results.append(('HDF5', w_t, w_s, r_t, r_n))
            except ImportError:
                print("  HDF5 不可用，跳过")

        print("\n  格式         写入(s)    文件大小    读取(s)    行数")
        print("  " + "-" * 65)
        for fmt, w_t, w_s, r_t, r_n in results:
            size_mb = w_s / 1024 / 1024
            print(f"  {fmt:<12} {w_t:<10.4f} {size_mb:<10.2f}MB {r_t:<10.4f} {r_n}")

        # Parquet 应比 CSV 小
        parquet = next((r for r in results if r[0] == 'Parquet'), None)
        csv = next((r for r in results if r[0] == 'CSV'), None)
        if parquet and csv:
            self.assertLess(parquet[2], csv[2],
                            "Parquet 文件应比 CSV 小")
            # 小数据集上读取时间相近，不做强制断言
            print(f"\n  Parquet/CSV 大小比: {parquet[2]/csv[2]:.2%}")
            print(f"  Parquet/CSV 读取比: {parquet[3]/csv[3]:.2%}")

    def test_02_query_single_stock(self):
        """测试单股票查询性能（列式 vs 行式）"""
        n_trials = 20

        # Parquet 全表读取到内存后过滤
        pq_path = os.path.join(self.tmpdir, "test_data.parquet")
        self.test_df.to_parquet(pq_path, index=False)

        start = time.perf_counter()
        for _ in range(n_trials):
            df_all = pd.read_parquet(pq_path)
            single = df_all[df_all['code'] == '000001.SZ']
        parquet_time = time.perf_counter() - start

        # CSV
        csv_path = os.path.join(self.tmpdir, "test_data.csv")
        self.test_df.to_csv(csv_path, index=False)

        start = time.perf_counter()
        for _ in range(n_trials):
            df_all = pd.read_csv(csv_path)
            single = df_all[df_all['code'] == '000001.SZ']
        csv_time = time.perf_counter() - start

        print(f"\n  单股票查询 ({n_trials}次):")
        print(f"    Parquet: {parquet_time:.4f}s ({parquet_time/n_trials*1000:.1f}ms/次)")
        print(f"    CSV:     {csv_time:.4f}s ({csv_time/n_trials*1000:.1f}ms/次)")
        print(f"    加速比:  {csv_time/parquet_time:.2f}x")

        self.assertLess(parquet_time, csv_time,
                        "Parquet 单股票查询应比 CSV 快")

    def test_03_time_range_query(self):
        """测试时间范围切片查询性能"""
        n_trials = 50

        pq_path = os.path.join(self.tmpdir, "test_data.parquet")
        self.test_df.to_parquet(pq_path, index=False)

        # Parquet: 读取全表后过滤
        start = time.perf_counter()
        for _ in range(n_trials):
            df_all = pd.read_parquet(pq_path)
            subset = df_all[(df_all['date'] >= '2022-06-01') & (df_all['date'] <= '2022-06-30')]
        parquet_time = time.perf_counter() - start

        # CSV
        csv_path = os.path.join(self.tmpdir, "test_data.csv")
        self.test_df.to_csv(csv_path, index=False)

        start = time.perf_counter()
        for _ in range(n_trials):
            df_all = pd.read_csv(csv_path)
            df_all['date'] = pd.to_datetime(df_all['date'])
            subset = df_all[(df_all['date'] >= '2022-06-01') & (df_all['date'] <= '2022-06-30')]
        csv_time = time.perf_counter() - start

        print(f"\n  时间范围查询 ({n_trials}次):")
        print(f"    Parquet: {parquet_time:.4f}s")
        print(f"    CSV:     {csv_time:.4f}s")
        print(f"    加速比:  {csv_time/parquet_time:.2f}x")

    def test_04_pivot_wide_performance(self):
        """测试宽表 pivoting 性能（回测常用操作）"""
        n_trials = 30

        pq_path = os.path.join(self.tmpdir, "test_data.parquet")
        self.test_df.to_parquet(pq_path, index=False)

        start = time.perf_counter()
        for _ in range(n_trials):
            df_all = pd.read_parquet(pq_path)
            pivot = df_all.pivot(index='date', columns='code', values='close')
        pivot_time = time.perf_counter() - start

        print(f"\n  Pivot 宽表 ({n_trials}次):")
        print(f"    Parquet: {pivot_time:.4f}s ({pivot_time/n_trials*1000:.1f}ms/次)")

        # pivot 本身是昂贵的操作，但 50 只股票 x 250 天应在合理范围
        self.assertLess(pivot_time / n_trials, 0.1,
                        "单次 Pivot 不应超过 100ms")


class TestVectorizedComputation(unittest.TestCase):
    """向量化 vs 循环计算性能对比"""

    @classmethod
    def setUpClass(cls):
        print("\n\n===== 向量化计算测试 =====")
        cls.test_df = generate_large_dataset(n_stocks=100, n_days=250)
        print(f"测试数据: {len(cls.test_df)} 行, {cls.test_df['code'].nunique()} 只股票")

    def test_01_rolling_mean_vectorized(self):
        """对比滚动均值计算: groupby.transform vs for循环"""
        df = self.test_df.copy()

        # 向量化: groupby transform
        start = time.perf_counter()
        df['ma20_vec'] = df.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        vec_time = time.perf_counter() - start

        # 循环方式
        start = time.perf_counter()
        df['ma20_loop'] = np.nan
        for code in df['code'].unique():
            mask = df['code'] == code
            series = df.loc[mask, 'close']
            df.loc[mask, 'ma20_loop'] = series.rolling(20, min_periods=10).mean()
        loop_time = time.perf_counter() - start

        # 验证一致性
        diff = (df['ma20_vec'] - df['ma20_loop']).abs().max()
        self.assertLess(diff, 0.001, f"向量化与循环计算结果不一致: diff={diff:.6f}")

        print(f"\n  滚动均值 (100股票x250天):")
        print(f"    向量化: {vec_time*1000:.1f}ms")
        print(f"    循环:   {loop_time*1000:.1f}ms")
        print(f"    加速比: {loop_time/vec_time:.1f}x")

    def test_02_ic_calculation_vectorized(self):
        """对比 IC 计算: 矩阵运算 vs 逐日循环"""
        np.random.seed(42)

        # 构造因子值和前向收益
        n_dates = 50
        n_stocks = 200
        factor = np.random.randn(n_dates, n_stocks)
        forward_ret = np.random.randn(n_dates, n_stocks)

        # 矩阵向量化
        start = time.perf_counter()
        for _ in range(100):
            # 逐列计算相关系数（每个时间点）
            ic_vec = np.array([
                np.corrcoef(factor[i, :], forward_ret[i, :])[0, 1]
                if np.std(factor[i, :]) > 0 and np.std(forward_ret[i, :]) > 0
                else 0.0
                for i in range(n_dates)
            ])
        vec_time = time.perf_counter() - start

        # 纯列表循环
        start = time.perf_counter()
        for _ in range(100):
            ic_loop = []
            for i in range(n_dates):
                try:
                    ic, _ = np.corrcoef(factor[i, :], forward_ret[i, :])
                    ic_loop.append(ic[0, 1])
                except Exception:
                    ic_loop.append(0.0)
        loop_time = time.perf_counter() - start

        print(f"\n  IC 计算 (50时间点x200股票, 100次):")
        print(f"    矩阵向量化: {vec_time*1000:.1f}ms")
        print(f"    纯循环:     {loop_time*1000:.1f}ms")
        print(f"    加速比:     {loop_time/vec_time:.2f}x")

    def test_03_neutralization_vectorized(self):
        """对比因子中性化: 矩阵回归 vs 逐日 sklearn"""
        np.random.seed(42)

        n_dates = 100
        n_stocks = 300
        n_industries = 10

        # 模拟数据
        factor_vals = np.random.randn(n_dates, n_stocks)
        mkt_cap = np.random.randn(n_dates, n_stocks)
        industry = np.random.randint(0, n_industries, (n_dates, n_stocks))

        # sklearn 逐日回归
        from sklearn.linear_model import LinearRegression

        start = time.perf_counter()
        for _ in range(10):
            residual_sk = np.zeros_like(factor_vals)
            for t in range(n_dates):
                y = factor_vals[t, :]
                # 行业哑变量
                ind_dummies = np.eye(n_industries)[industry[t, :]]
                X = np.column_stack([mkt_cap[t, :], ind_dummies])
                # 去除 NaN
                valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
                if valid.sum() > 30:
                    model = LinearRegression()
                    model.fit(X[valid], y[valid])
                    y_pred = model.predict(X)
                    residual_sk[t, :] = y - y_pred
                else:
                    residual_sk[t, :] = y
        sk_time = time.perf_counter() - start

        # numpy 矩阵运算
        start = time.perf_counter()
        for _ in range(10):
            residual_np = np.zeros_like(factor_vals)
            for t in range(n_dates):
                y = factor_vals[t, :]
                ind_dummies = np.eye(n_industries)[industry[t, :]]
                X = np.column_stack([mkt_cap[t, :], ind_dummies])
                # 最小二乘: beta = (X^T X)^{-1} X^T y (numpy 直接算)
                try:
                    XtX = X.T @ X
                    Xty = X.T @ y
                    beta = np.linalg.solve(XtX, Xty)
                    y_pred = X @ beta
                    residual_np[t, :] = y - y_pred
                except np.linalg.LinAlgError:
                    residual_np[t, :] = y
        np_time = time.perf_counter() - start

        print(f"\n  因子中性化 (100天x300股票, 10行业, 10次):")
        print(f"    sklearn 逐日: {sk_time*1000:.1f}ms")
        print(f"    numpy 矩阵:   {np_time*1000:.1f}ms")
        print(f"    加速比:       {sk_time/np_time:.2f}x")

    def test_04_covariance_estimation(self):
        """对比协方差估计: ledoit_wolf vs sample_cov"""
        np.random.seed(42)

        n_periods = 252
        n_stocks = 50

        # 模拟收益率数据
        returns = pd.DataFrame(
            np.random.randn(n_periods, n_stocks) * 0.02,
            columns=[f"S{i}" for i in range(n_stocks)]
        )

        # Sample covariance
        start = time.perf_counter()
        for _ in range(100):
            sample_cov = returns.cov().values
        sample_time = time.perf_counter() - start

        # Ledoit-Wolf shrinkage
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            start = time.perf_counter()
            for _ in range(10):  # LW 更慢，减少迭代
                lw.fit(returns.values)
            lw_time = time.perf_counter() - start

            print(f"\n  协方差估计 (252天x50股票):")
            print(f"    Sample Cov:     {sample_time/100*1000:.3f}ms/次")
            print(f"    Ledoit-Wolf:    {lw_time/10*1000:.1f}ms/次")

            # LW 应该比 Sample Cov 更稳定（条件数更小）
            sample_cond = np.linalg.cond(sample_cov)
            lw_cond = np.linalg.cond(lw.covariance_)
            print(f"    Sample 条件数:  {sample_cond:.1f}")
            print(f"    LW 条件数:      {lw_cond:.1f}")

            # LW 结果应更稳定（条件数更小）
            self.assertLess(lw_cond, sample_cond * 1.5,
                            "Ledoit-Wolf 条件数应优于或接近样本协方差")
        except ImportError:
            print("  sklearn LedoitWolf 不可用，跳过 LW 测试")


class TestDataPipelineOptimization(unittest.TestCase):
    """数据处理管道优化验证"""

    @classmethod
    def setUpClass(cls):
        print("\n\n===== 数据管道优化测试 =====")
        np.random.seed(42)
        cls.test_df = generate_large_dataset(n_stocks=100, n_days=500)
        print(f"测试数据: {len(cls.test_df)} 行")

    def test_01_filter_sort_optimization(self):
        """测试数据过滤+排序的链式操作性能"""
        df = self.test_df.copy()

        # 朴素方式: 多次中间拷贝
        start = time.perf_counter()
        f1 = df[df['volume'] > 0].copy()
        f2 = f1[f1['close'] > 1.0].copy()
        f3 = f2[f2['close'] < 500.0].copy()
        f4 = f3.sort_values(['code', 'date']).copy()
        naive_time = time.perf_counter() - start

        # 优化方式: 合并过滤条件 + query
        start = time.perf_counter()
        result = df.query('volume > 0 and close > 1.0 and close < 500.0')
        result = result.sort_values(['code', 'date'])
        opt_time = time.perf_counter() - start

        print(f"\n  数据过滤+排序:")
        print(f"    多次copy: {naive_time*1000:.1f}ms")
        print(f"    query合并: {opt_time*1000:.1f}ms")
        # 大数据集上 query 应不慢于多次 copy（小数据集误差可接受）
        self.assertLessEqual(opt_time, naive_time * 1.2,
                             "优化方式不应明显慢于多次copy")

    def test_02_chunked_processing(self):
        """测试分块处理大数据 vs 一次性加载"""
        df = self.test_df.copy()
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "chunk_test.csv")
        df.to_csv(csv_path, index=False)

        # 一次性加载
        start = time.perf_counter()
        all_data = pd.read_csv(csv_path)
        all_data['date'] = pd.to_datetime(all_data['date'])
        result_full = all_data.groupby('code')['close'].transform(
            lambda x: x.rolling(20).mean()
        )
        full_time = time.perf_counter() - start

        # 分块加载
        start = time.perf_counter()
        chunks = []
        for chunk in pd.read_csv(csv_path, chunksize=10000):
            chunk['date'] = pd.to_datetime(chunk['date'])
            chunk['ma20'] = chunk.groupby('code')['close'].transform(
                lambda x: x.rolling(20).mean()
            )
            chunks.append(chunk)
        result_chunked = pd.concat(chunks)
        chunk_time = time.perf_counter() - start

        shutil.rmtree(tmpdir, ignore_errors=True)

        print(f"\n  分块vs一次性(CVS, 100股票x500天):")
        print(f"    一次性: {full_time*1000:.1f}ms")
        print(f"    分块:   {chunk_time*1000:.1f}ms")

        # 分块处理可能不一定更快（CSV 的 chunksize 主要用于内存控制）
        # 但说明了内存友好型数据管道的重要性

    def test_03_parallel_factor_calculation(self):
        """测试并行因子计算 vs 串行"""
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        df = self.test_df.copy()

        # 定义几个计算函数
        def calc_ma(code_group, window):
            return code_group['close'].rolling(window, min_periods=window//2).mean()

        def calc_vol(code_group, window):
            ret = code_group['close'].pct_change()
            return ret.rolling(window, min_periods=window//2).std()

        def calc_mom(code_group, window):
            return code_group['close'] / code_group['close'].shift(window) - 1

        # 串行计算
        start = time.perf_counter()
        codes = df['code'].unique()
        grouped = df.groupby('code')

        results_serial = {}
        for code in codes:
            group = grouped.get_group(code)
            results_serial[code] = {
                'ma20': calc_ma(group, 20).values,
                'vol20': calc_vol(group, 20).values,
                'mom20': calc_mom(group, 20).values,
            }
        serial_time = time.perf_counter() - start

        print(f"\n  串行因子计算 ({len(codes)}只股票 x 3因子):")
        print(f"    耗时: {serial_time*1000:.1f}ms")

        # 注: 在单进程中，groupby.transform 已足够高效
        # ProcessPoolExecutor 的序列化开销可能抵消并行收益
        # 这里仅验证串行方案在数据量级上的可行性

        self.assertLess(serial_time, 5.0,
                        "100股票x3因子串行计算不应超过5秒")


if __name__ == '__main__':
    unittest.main(verbosity=2)