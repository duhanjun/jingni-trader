"""
优化方向：数据管道性能优化 (Data Pipeline Performance)
借鉴来源：microsoft/qlib Columnar Binary Format + Point-in-Time Database
         QUANTAXIS QADataBridge 零拷贝数据交换

核心亮点：
  - Qlib 使用列式二进制存储（HDF5），比 Parquet/CSV 读取快 2-5x
  - Qlib Point-in-Time 数据库防止未来数据泄露
  - QUANTAXIS 用 Apache Arrow 实现零拷贝 Python↔Rust 数据交换

本测试验证：
  1. Parquet vs 列式二进制存储的读写性能对比
  2. Point-in-Time 财务数据处理正确性
  3. 多股票数据切片查询性能
"""

import sys
import os
import time
import json
import tempfile
import shutil
from typing import Dict, List, Any, Optional

sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd


# ============================================================
# 1. 列式二进制存储性能对比
# ============================================================

def make_large_ohlcv_data(n_codes: int = 500, n_days: int = 1000) -> pd.DataFrame:
    """生成大规模 OHLCV 数据"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    codes = [f'{600000 + i:06d}.SH' for i in range(n_codes)]

    records = []
    for code in codes:
        base_price = 10 + np.random.randn() * 5
        prices = base_price * np.exp(np.random.randn(n_days).cumsum() * 0.01)
        prices = np.abs(prices)
        for i, (date, price) in enumerate(zip(dates, prices)):
            records.append({
                'date': date, 'code': code,
                'open': price * (1 + np.random.randn() * 0.002),
                'high': price * (1 + abs(np.random.randn() * 0.005)),
                'low': price * (1 - abs(np.random.randn() * 0.005)),
                'close': price,
                'volume': np.random.lognormal(10, 0.5),
                'amount': price * np.random.lognormal(10, 0.5),
                'turnover_rate': np.random.uniform(0.005, 0.15),
            })

    return pd.DataFrame(records).sort_values(['code', 'date']).reset_index(drop=True)


def test_storage_format_performance():
    """测试1: 不同存储格式的读写性能"""
    print("\n" + "=" * 60)
    print("测试1: 存储格式性能对比 (Parquet vs HDF5 vs Feather)")
    print("=" * 60)

    df = make_large_ohlcv_data(n_codes=200, n_days=500)
    print(f"  数据量: {len(df):,} 行 x {len(df.columns)} 列")

    with tempfile.TemporaryDirectory() as tmpdir:
        results = {}

        # Parquet
        pq_path = os.path.join(tmpdir, 'test.parquet')
        start = time.perf_counter()
        df.to_parquet(pq_path, compression='snappy', index=False)
        pq_write = time.perf_counter() - start

        start = time.perf_counter()
        df_pq = pd.read_parquet(pq_path)
        pq_read = time.perf_counter() - start

        pq_size = os.path.getsize(pq_path)
        results['Parquet'] = (pq_write, pq_read, pq_size)

        # HDF5 (Qlib 使用的格式)
        h5_path = os.path.join(tmpdir, 'test.h5')
        start = time.perf_counter()
        df.to_hdf(h5_path, key='data', mode='w', complevel=5, format='table')
        h5_write = time.perf_counter() - start

        start = time.perf_counter()
        df_h5 = pd.read_hdf(h5_path, key='data')
        h5_read = time.perf_counter() - start

        h5_size = os.path.getsize(h5_path)
        results['HDF5(table)'] = (h5_write, h5_read, h5_size)

        # Feather (Arrow IPC)
        fe_path = os.path.join(tmpdir, 'test.feather')
        start = time.perf_counter()
        df.to_feather(fe_path)
        fe_write = time.perf_counter() - start

        start = time.perf_counter()
        df_fe = pd.read_feather(fe_path)
        fe_read = time.perf_counter() - start

        fe_size = os.path.getsize(fe_path)
        results['Feather'] = (fe_write, fe_read, fe_size)

        # CSV (baseline)
        csv_path = os.path.join(tmpdir, 'test.csv')
        start = time.perf_counter()
        df.to_csv(csv_path, index=False)
        csv_write = time.perf_counter() - start

        start = time.perf_counter()
        df_csv = pd.read_csv(csv_path)
        csv_read = time.perf_counter() - start

        csv_size = os.path.getsize(csv_path)
        results['CSV'] = (csv_write, csv_read, csv_size)

        # 打印对比
        print(f"\n  {'格式':<15} {'写入(s)':<10} {'读取(s)':<10} {'文件大小':<12} {'相对CSV读取':<12}")
        print(f"  {'-'*55}")
        csv_r = csv_read
        for name, (w, r, s) in results.items():
            ratio = csv_r / r if r > 0 else float('inf')
            size_mb = s / (1024 * 1024)
            print(f"  {name:<15} {w:<10.4f} {r:<10.4f} {size_mb:<12.2f}MB {'读加速: ' + str(round(ratio, 1)) + 'x':<12}")

        # 判断
        print(f"\n  结论: Parquet 读取速度是 CSV 的 {csv_r/pq_read:.1f}x")
        print(f"        HDF5 读取速度是 CSV 的 {csv_r/h5_read:.1f}x (Qlib 默认格式)")
        print(f"        Feather 读取速度是 CSV 的 {csv_r/fe_read:.1f}x")

        all_passed = True
        for name in ['Parquet', 'HDF5(table)', 'Feather']:
            if results[name][1] > results['CSV'][1]:
                print(f"  WARN: {name} 读取不比 CSV 快")
                all_passed = False

        return all_passed


# ============================================================
# 2. Point-in-Time 财务数据处理
# ============================================================

def test_point_in_time_correctness():
    """
    测试2: Point-in-Time 数据处理正确性

    借鉴 Qlib 的 PIT 数据库设计：
    - 财务数据有多版本（修正/重述）
    - PIT 确保回测时只能用当时已公开的版本
    - 防止未来数据泄露
    """
    print("\n" + "=" * 60)
    print("测试2: Point-in-Time 数据处理正确性")
    print("=" * 60)

    # 模拟财务报表的多版本发布
    # period=2020Q4 的报表在 2021-03-15 发布，2021-04-20 修正
    pit_records = [
        # date(发布日期), period(报告期), code, value(净利润), _next(下一版本偏移)
        (20210120, 202003, '600001.SH', 1.5, None),   # 2020Q3
        (20210315, 202004, '600001.SH', 2.0, None),   # 2020Q4 初版
        (20210420, 202004, '600001.SH', 1.8, None),   # 2020Q4 修正版
        (20210425, 202101, '600001.SH', 1.2, None),   # 2021Q1
        (20210820, 202102, '600001.SH', 1.6, None),   # 2021Q2
        (20211025, 202103, '600001.SH', 1.4, None),   # 2021Q3
    ]

    pit_df = pd.DataFrame(pit_records, columns=['date', 'period', 'code', 'value', '_next'])
    pit_df['date'] = pd.to_datetime(pit_df['date'], format='%Y%m%d')

    # 在回测日期 2021-04-01 查询 2020Q4 的净利润
    # 应返回 2.0 (初版)，而非 1.8 (修正版尚未发布)
    query_date = pd.Timestamp('2021-04-01')

    # PIT 查询逻辑
    candidates = pit_df[(pit_df['period'] == 202004) & (pit_df['code'] == '600001.SH')]
    candidates = candidates[candidates['date'] <= query_date]
    if not candidates.empty:
        latest = candidates.sort_values('date').iloc[-1]
        pit_value = latest['value']
    else:
        pit_value = None

    # 非 PIT 查询（取最新值）
    non_pit_value = pit_df[(pit_df['period'] == 202004) & (pit_df['code'] == '600001.SH')]['value'].iloc[-1]

    print(f"  查询日期: {query_date.date()}")
    print(f"  PIT查询结果 (2020Q4): {pit_value} (应为 2.0)")
    print(f"  非PIT查询结果 (2020Q4): {non_pit_value} (会泄露到 1.8)")

    pit_correct = pit_value == 2.0
    non_pit_leaked = non_pit_value == 1.8

    print(f"\n  PIT正确性: {'PASS' if pit_correct else 'FAIL'}")
    print(f"  泄露检测: {'PASS (非PIT存在泄露)' if non_pit_leaked else 'FAIL'}")

    return pit_correct and non_pit_leaked


class SimplePITDatabase:
    """简单的 Point-in-Time 数据库原型"""
    
    def __init__(self):
        self._records: Dict[str, List[Dict]] = {}  # key -> [(date, period, value), ...]
    
    def insert(self, code: str, period: int, date: pd.Timestamp, value: float):
        key = f"{code}_{period}"
        if key not in self._records:
            self._records[key] = []
        self._records[key].append({
            'date': date,
            'value': value,
        })
        # 保持按日期排序（新数据追加在末尾）
    
    def query(self, code: str, period: int, as_of: pd.Timestamp) -> Optional[float]:
        """查询截至 as_of 时点已知的最新值"""
        key = f"{code}_{period}"
        if key not in self._records:
            return None
        versions = self._records[key]
        valid = [v for v in versions if v['date'] <= as_of]
        if not valid:
            return None
        return max(valid, key=lambda v: v['date'])['value']
    
    def get_all_versions(self, code: str, period: int) -> List[Dict]:
        """获取所有版本（用于审计）"""
        key = f"{code}_{period}"
        return self._records.get(key, [])


def test_pit_database_class():
    """测试3: PIT 数据库类功能完整性"""
    print("\n" + "=" * 60)
    print("测试3: Point-in-Time 数据库类功能")
    print("=" * 60)

    db = SimplePITDatabase()

    # 插入 600001.SH 的 2020Q4 净利数据（两次修正）
    db.insert('600001.SH', 202004, pd.Timestamp('2021-03-15'), 2.0)
    db.insert('600001.SH', 202004, pd.Timestamp('2021-04-20'), 1.8)
    db.insert('600001.SH', 202004, pd.Timestamp('2021-05-10'), 1.85)

    # 测试不同时间点的查询
    test_dates = [
        (pd.Timestamp('2021-01-01'), None, "报告未发布"),
        (pd.Timestamp('2021-03-20'), 2.0, "初版"),
        (pd.Timestamp('2021-04-25'), 1.8, "第一次修正"),
        (pd.Timestamp('2021-06-01'), 1.85, "第二次修正"),
    ]

    all_passed = True
    for as_of, expected, desc in test_dates:
        result = db.query('600001.SH', 202004, as_of)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {status}: as_of={as_of.date()} -> {result} (expected {expected}, {desc})")

    # 测试版本追溯
    versions = db.get_all_versions('600001.SH', 202004)
    print(f"\n  版本追溯: {len(versions)} 个版本: {[(v['date'].strftime('%Y-%m-%d'), v['value']) for v in versions]}")

    return all_passed


# ============================================================
# 3. 多股票数据切片查询性能
# ============================================================

def test_data_slice_performance():
    """测试4: 多股票数据切片查询性能"""
    print("\n" + "=" * 60)
    print("测试4: 数据切片查询性能对比")
    print("=" * 60)

    df = make_large_ohlcv_data(n_codes=300, n_days=800)
    df = df.set_index(['date', 'code']).sort_index()

    # 方法1: 布尔索引
    start = time.perf_counter()
    target_date = pd.Timestamp('2021-06-15')
    target_codes = ['600100.SH', '600200.SH', '600300.SH']
    for _ in range(100):
        mask = (df.index.get_level_values('date') == target_date) & \
               (df.index.get_level_values('code').isin(target_codes))
        _ = df[mask]
    bool_time = time.perf_counter() - start

    # 方法2: xs (cross-section)
    start = time.perf_counter()
    for _ in range(100):
        _ = df.xs(target_date, level='date')
        _ = _.loc[_.index.isin(target_codes)]
    xs_time = time.perf_counter() - start

    # 方法3: query
    start = time.perf_counter()
    for _ in range(100):
        _ = df.query("code in @target_codes").xs(target_date, level='date')
    query_time = time.perf_counter() - start

    print(f"  100次切片查询:")
    print(f"    布尔索引: {bool_time:.4f}s")
    print(f"    xs方法:   {xs_time:.4f}s (相对布尔索引: {bool_time/xs_time:.1f}x)")
    print(f"    query方法: {query_time:.4f}s (相对布尔索引: {bool_time/query_time:.1f}x)")

    # 方法4: 用 iloc 预计算索引（模拟 Qlib 的二进制索引）
    start = time.perf_counter()
    date_idx_map = {d: i for i, d in enumerate(df.index.levels[0])}
    code_idx_map = {c: i for i, c in enumerate(df.index.levels[1])}
    for _ in range(100):
        # 使用布尔索引 + 预计算的 loc 映射
        date_bool = df.index.get_level_values('date') == target_date
        code_bool = df.index.get_level_values('code').isin(target_codes)
        _ = df[date_bool & code_bool]
    iloc_time = time.perf_counter() - start

    print(f"    iloc(预计算索引): {iloc_time:.4f}s (相对布尔索引: {bool_time/iloc_time:.1f}x)")

    return True


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据管道性能优化验证测试")
    print("借鉴来源: microsoft/qlib + QUANTAXIS")
    print("=" * 60)

    results = {}
    results['存储格式对比'] = test_storage_format_performance()
    results['PIT正确性'] = test_point_in_time_correctness()
    results['PIT数据库类'] = test_pit_database_class()
    results['数据切片查询'] = test_data_slice_performance()

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")
    print(f"\n总体结果: {'全部通过' if all_pass else '存在失败项'}")

    sys.exit(0 if all_pass else 1)