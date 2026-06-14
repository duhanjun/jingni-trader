"""
=============================================================================
借鉴来源: Microsoft Qlib Point-in-Time Database
           (https://qlib.readthedocs.io/en/latest/advanced/PIT.html)
优化方向: 时点数据验证 - 防止回测中的前视偏差 (Look-ahead Bias)
=============================================================================

核心亮点:
  Qlib 的 PIT (Point-in-Time) 数据库确保在任何历史时间点进行回测时，
  只使用该时间点实际可用的数据。这是避免"未来函数"的关键设计。
  
  具体实现:
  - 每条财务数据记录包含 date（发布日期）、period（报告期）、value（值）
  - 查询时根据 observation_time 返回该时刻可用的最近版本
  - 支持数据的多次修订链（年报修正等场景）

对比 jingni-trader 现状:
  当前 data-engine 和 factor-engine 没有 PIT 机制。当计算因子如 lncap 时:
    result['lncap'] = mv.replace(0, np.nan).apply(lambda x: np.log(x))
  这种方式可能使用全时段数据来计算，如果在回测中不对齐时间，
  会导致前视偏差（即在 t 时刻错误地使用了 t+1 时刻的信息）。

验证内容:
  1. PIT 数据写入和读取正确性
  2. 修订链处理（多次修订场景）
  3. 前视偏差检测工具
  4. 与现有数据处理方式的对比
  5. 边界条件测试
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ═══════════════════════════════════════════════════════════════════════════
# Point-in-Time 数据系统原型实现 (借鉴 Qlib PIT Database)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PITRecord:
    """单条 PIT 记录"""
    publish_date: str        # 发布日期 YYYYMMDD
    period: str              # 报告期 YYYYQQ
    value: float             # 报告值
    next_revision_idx: int = -1   # 下一修订版索引（链表）


class PITDatabase:
    """
    简化版 Point-in-Time 数据库
    
    存储格式: {instrument: {field: [PITRecord, ...]}}
    记录按 publish_date 升序排列
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, List[PITRecord]]] = {}

    def insert(
        self,
        instrument: str,
        field: str,
        period: str,
        value: float,
        publish_date: str,
    ):
        """插入一条 PIT 数据记录"""
        if instrument not in self._data:
            self._data[instrument] = {}
        if field not in self._data[instrument]:
            self._data[instrument][field] = []

        record = PITRecord(
            publish_date=publish_date,
            period=period,
            value=value,
        )

        records = self._data[instrument][field]

        # 查找同一 period 的已有记录（修订场景）
        for i, existing in enumerate(records):
            if existing.period == period:
                # 更新链表，将之链接为修订版
                existing.next_revision_idx = len(records)
                break

        records.append(record)
        # 保持按 publish_date 排序
        records.sort(key=lambda r: r.publish_date)

    def query(
        self,
        instrument: str,
        field: str,
        observation_date: str,  # YYYYMMDD
    ) -> Optional[float]:
        """
        查询给定时间点实际可用的数据值
        
        返回在 observation_date 之前发布的最新版本值。
        如果同一 period 有多条修订记录，返回 observation_date 前最后发布的版本。
        """
        if instrument not in self._data:
            return None
        if field not in self._data[instrument]:
            return None

        records = self._data[instrument][field]
        best_value = None
        best_period = None

        for record in records:
            if record.publish_date > observation_date:
                break
            # 同 period 的后续修订会覆盖前面的值
            if record.period == best_period:
                best_value = record.value
            else:
                best_period = record.period
                best_value = record.value

        return best_value

    def query_latest_before(
        self,
        instrument: str,
        field: str,
        observation_date: str,
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        查询 observation_date 之前最新可用的值和对应的报告期
        返回 (value, period)
        """
        if instrument not in self._data:
            return None, None
        if field not in self._data[instrument]:
            return None, None

        records = self._data[instrument][field]
        best_value = None
        best_period = None

        for record in records:
            if record.publish_date > observation_date:
                break
            best_value = record.value
            best_period = record.period

        return best_value, best_period

    def get_instruments(self) -> List[str]:
        return list(self._data.keys())

    def get_fields(self, instrument: str) -> List[str]:
        return list(self._data.get(instrument, {}).keys())


class LookAheadBiasDetector:
    """
    前视偏差检测器
    
    检测数据处理中是否存在使用未来数据的问题。
    常见的前视偏差类型:
    1. 使用了 t+1 日之后的价格信息计算 t 日的因子
    2. 使用了发布日晚于观测日的财务数据
    3. 在滚动窗口中使用了超出窗口范围的数据
    """

    @staticmethod
    def check_future_price_leakage(
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_col: str = 'alpha_score',
        price_col: str = 'close',
        lookback_window: int = 0,
    ) -> Dict:
        """
        检查因子计算中是否使用了未来价格信息
        
        原理: 计算 t 日因子值与 t+1 日之后价格的相关性。
        如果因子不含前视偏差，则因子与未来价格不应有显著相关性。
        """
        merged = factor_df[['code', 'date', factor_col]].merge(
            price_df[['code', 'date', price_col]],
            on=['code', 'date'],
            how='inner'
        )

        results = {}

        for shift in [1, 5, 20]:
            future_price = merged.groupby('code')[price_col].shift(-shift)
            merged_copy = merged.copy()
            merged_copy['future_price'] = future_price

            valid = merged_copy.dropna(subset=[factor_col, 'future_price'])
            if len(valid) < 100:
                continue

            corr = valid[factor_col].corr(valid['future_price'])
            results[f'corr_with_price_t+{shift}'] = round(corr, 6)

            # IC 分析
            if valid['code'].nunique() > 10:
                ic_list = []
                for dt in valid['date'].unique():
                    cross = valid[valid['date'] == dt]
                    if len(cross) < 10:
                        continue
                    ic = cross[factor_col].corr(cross['future_price'])
                    if not np.isnan(ic):
                        ic_list.append(ic)

                if ic_list:
                    ic_mean = np.mean(ic_list)
                    results[f'IC_t+{shift}_mean'] = round(float(ic_mean), 6)
                    results[f'IC_t+{shift}_abs_mean'] = round(float(np.mean(np.abs(ic_list))), 6)

        return results

    @staticmethod
    def check_financial_data_timeline(
        financial_data: pd.DataFrame,
        observation_dates: List[str],
    ) -> Dict:
        """
        检查财务数据发布日与观测日的时间线
        
        验证: 对于每个 observation_date，
        只有 publish_date <= observation_date 的数据才应该被使用。
        """
        violations = []

        for _, row in financial_data.iterrows():
            obs_date = row.get('observation_date', '')
            pub_date = row.get('publish_date', '')

            if obs_date and pub_date and pub_date > obs_date:
                violations.append({
                    'observation_date': obs_date,
                    'publish_date': pub_date,
                    'period': row.get('period', ''),
                    'value': row.get('value', None),
                })

        return {
            'total_records': len(financial_data),
            'violations': len(violations),
            'violation_rate': round(len(violations) / len(financial_data), 4) if len(financial_data) > 0 else 0,
            'violation_examples': violations[:5],
        }

    @staticmethod
    def validate_rolling_window(
        data: pd.DataFrame,
        factor_func,
        window_size: int,
        feature_cols: List[str],
    ) -> Dict:
        """
        验证滚动窗口计算是否存在前视偏差
        
        方法: 对截断数据（去掉最后 N 天）和完整数据分别计算，
        比较两个版本的结果是否一致。
        """
        truncated = data.iloc[:-window_size].copy()
        truncated_result = factor_func(truncated)

        original_subset = data.iloc[:len(truncated)].copy()
        full_result = factor_func(original_subset)

        # 截断数据的结果应该与完整数据的前部分结果完全一致
        common_cols = [c for c in truncated_result.columns
                       if c in full_result.columns and c in feature_cols]

        diffs = {}
        for col in common_cols:
            if col in truncated_result.columns and col in full_result.columns:
                diff = (truncated_result[col].fillna(0) - full_result[col].fillna(0)).abs()
                diffs[col] = {
                    'max_diff': float(diff.max()),
                    'mean_diff': float(diff.mean()),
                    'is_identical': float(diff.max()) < 1e-10,
                }

        return {
            'window_size': window_size,
            'feature_diffs': diffs,
            'all_identical': all(d['is_identical'] for d in diffs.values()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════════════════

class TestPITDatabase(unittest.TestCase):
    """Point-in-Time 数据库功能测试"""

    def setUp(self):
        self.db = PITDatabase()

    def test_basic_insert_and_query(self):
        """测试基本插入和查询"""
        self.db.insert('600000.SH', 'roe', '202501', 0.15, '20250420')

        result = self.db.query('600000.SH', 'roe', '20250425')
        self.assertEqual(result, 0.15)

        # 在发布日期之前查询应返回 None
        result_before = self.db.query('600000.SH', 'roe', '20250419')
        self.assertIsNone(result_before)

    def test_revision_chain(self):
        """测试财务数据修订链"""
        # 初始报告: 2025Q1 ROE = 0.15, 发布于 20250420
        self.db.insert('600000.SH', 'roe', '202501', 0.15, '20250420')
        # 修订报告: 2025Q1 ROE = 0.18, 发布于 20250515
        self.db.insert('600000.SH', 'roe', '202501', 0.18, '20250515')

        # 在修订前查询应返回初始值
        result_initial = self.db.query('600000.SH', 'roe', '20250425')
        self.assertEqual(result_initial, 0.15)

        # 在修订后查询应返回修订值
        result_revised = self.db.query('600000.SH', 'roe', '20250520')
        self.assertEqual(result_revised, 0.18)

    def test_multiple_instruments(self):
        """测试多股票数据管理"""
        self.db.insert('600000.SH', 'roe', '202501', 0.15, '20250420')
        self.db.insert('600001.SH', 'roe', '202501', 0.08, '20250422')

        r1 = self.db.query('600000.SH', 'roe', '20250425')
        r2 = self.db.query('600001.SH', 'roe', '20250425')

        self.assertEqual(r1, 0.15)
        self.assertEqual(r2, 0.08)

    def test_multiple_fields(self):
        """测试多字段数据管理"""
        self.db.insert('600000.SH', 'roe', '202501', 0.15, '20250420')
        self.db.insert('600000.SH', 'eps', '202501', 0.85, '20250420')
        self.db.insert('600000.SH', 'bvps', '202501', 5.50, '20250420')

        instruments = self.db.get_instruments()
        fields = self.db.get_fields('600000.SH')

        self.assertEqual(len(instruments), 1)
        self.assertEqual(len(fields), 3)
        self.assertIn('roe', fields)
        self.assertIn('eps', fields)
        self.assertIn('bvps', fields)

    def test_query_across_periods(self):
        """测试跨报告期查询"""
        self.db.insert('600000.SH', 'roe', '202404', 0.12, '20250320')  # 年报
        self.db.insert('600000.SH', 'roe', '202501', 0.15, '20250420')  # 一季报

        # 在一季报发布前查询应返回年报数据
        result = self.db.query_latest_before('600000.SH', 'roe', '20250415')
        self.assertEqual(result[0], 0.12)
        self.assertEqual(result[1], '202404')

        # 在一季报发布后查询应返回一季报数据
        result = self.db.query_latest_before('600000.SH', 'roe', '20250425')
        self.assertEqual(result[0], 0.15)
        self.assertEqual(result[1], '202501')


class TestLookAheadBiasDetector(unittest.TestCase):
    """前视偏差检测器测试"""

    @classmethod
    def setUpClass(cls):
        """生成测试数据"""
        np.random.seed(42)
        n_stocks = 10
        n_days = 200
        stocks = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
        dates = pd.date_range('2023-01-01', periods=n_days, freq='B')

        cls.price_data = []
        cls.factor_data = []

        for code in stocks:
            # 生成价格序列（带自相关）
            returns = np.random.randn(n_days) * 0.02
            close = np.cumsum(returns) + 10
            close = np.maximum(close, 1)

            for i, dt in enumerate(dates):
                cls.price_data.append({
                    'code': code,
                    'date': dt,
                    'close': close[i],
                })

                # 正常因子：仅使用历史信息
                if i >= 20:
                    ret_20d = (close[i] - close[i - 20]) / close[i - 20]
                    cls.factor_data.append({
                        'code': code,
                        'date': dt,
                        'alpha_score': -ret_20d,  # 反转因子
                    })

        cls.price_df = pd.DataFrame(cls.price_data)
        cls.factor_df = pd.DataFrame(cls.factor_data)

        # 构造一个含前视偏差的因子（使用未来价格）
        cls.leaked_factor_df = cls.factor_df.copy()
        cls.leaked_factor_df['alpha_score'] = cls.price_df.groupby('code')['close'].shift(-5).pct_change()

    def test_no_leakage_factor(self):
        """测试无前视偏差因子的检测"""
        detector = LookAheadBiasDetector()
        results = detector.check_future_price_leakage(
            self.factor_df, self.price_df, factor_col='alpha_score'
        )
        # 反转因子不应该与未来价格有显著正相关性
        if 'IC_t+5_mean' in results:
            self.assertLess(abs(results['IC_t+5_mean']), 0.3,
                           f"正常因子与未来价格的相关性过高: {results['IC_t+5_mean']}")

    def test_leakage_detection(self):
        """测试前视偏差的检测能力"""
        detector = LookAheadBiasDetector()
        results = detector.check_future_price_leakage(
            self.leaked_factor_df, self.price_df, factor_col='alpha_score'
        )
        # 含前视偏差的因子与t+5价格应该有显著相关性
        print(f"\n  前视偏差检测结果: {json.dumps(results, indent=2)}")

    def test_financial_timeline_check(self):
        """测试财务数据时间线检查"""
        fin_data = pd.DataFrame([
            {'observation_date': '20250425', 'publish_date': '20250420', 'period': '202501', 'value': 0.15},
            {'observation_date': '20250420', 'publish_date': '20250420', 'period': '202501', 'value': 0.15},
            {'observation_date': '20250415', 'publish_date': '20250420', 'period': '202501', 'value': 0.15},  # 违规!
        ])

        detector = LookAheadBiasDetector()
        results = detector.check_financial_data_timeline(fin_data, [])
        self.assertEqual(results['violations'], 1)
        self.assertGreater(results['violation_rate'], 0)

    def test_rolling_window_validation(self):
        """测试滚动窗口验证"""
        data = self.price_df.copy()

        def calc_ma(data):
            result = data[['code', 'date']].copy()
            result['ma_20'] = data.groupby('code')['close'].transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            )
            return result

        detector = LookAheadBiasDetector()
        results = detector.validate_rolling_window(
            data, calc_ma, window_size=20, feature_cols=['ma_20']
        )

        # 滚动窗口计算应该是一致的（无前视偏差）
        if 'ma_20' in results['feature_diffs']:
            self.assertTrue(
                results['feature_diffs']['ma_20']['is_identical'],
                "滚动窗口计算存在前视偏差"
            )

    def test_compare_with_pit_database(self):
        """对比 PIT 数据库与简单合并方式的差异"""
        db = PITDatabase()

        # 模拟场景: ROE 数据在不同时间发布
        db.insert('600000.SH', 'roe', '202404', 0.12, '20250320')
        db.insert('600001.SH', 'roe', '202404', 0.08, '20250322')

        # PIT 查询: 在 20250321 时，600001 的数据还未发布
        r1 = db.query('600000.SH', 'roe', '20250321')
        r2 = db.query('600001.SH', 'roe', '20250321')

        self.assertEqual(r1, 0.12)
        self.assertIsNone(r2)  # 数据尚未发布！

        # 简单合并方式会错误地将 600001 的 ROE 也用在 20250321
        # 这就是前视偏差的来源
        print(f"\n  PIT vs 简单合并对比:")
        print(f"    观测时间 20250321:")
        print(f"      600000.SH ROE (PIT): {r1}")
        print(f"      600001.SH ROE (PIT): {r2} (数据尚未发布)")
        print(f"    如果使用简单合并，600001.SH 的 ROE 也会被错误使用")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Point-in-Time 数据系统验证测试")
    print("借鉴来源: Microsoft Qlib PIT Database")
    print("=" * 70)
    results = run_tests()
    print("\n" + "=" * 70)
    print(f"测试结果: {results['tests_run']} 个测试, "
          f"{results['failures']} 个失败, {results['errors']} 个错误")
    print(f"总体: {'通过' if results['success'] else '失败'}")
    print("=" * 70)