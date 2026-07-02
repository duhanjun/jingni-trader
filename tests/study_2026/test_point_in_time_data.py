"""
测试：Point-in-Time 数据处理与未来数据泄露防护
借鉴来源：Microsoft Qlib (https://github.com/microsoft/qlib) - PIT Data Provider
优化方向：data-engine - 加入 PitProvider 防止未来数据泄露

Qlib 引入了 Point-in-Time (PIT) 数据提供者，确保在任何回测时刻
只能获取当时已公开的信息，严格防止未来数据泄露（Look-ahead Bias）。
这对于财务数据尤其重要——年报可能在次年4月才发布，不能提前使用。

jingni-trader 当前的数据引擎缺少 PIT 保护机制，可能导致：
1. 财务报表数据的未来信息泄露
2. 股票列表的存活偏差（Survivorship Bias）
3. 复权价格的处理错误

本测试验证：
1. PIT 财务数据管理
2. 股票池的时点过滤（排除未上市/已退市股票）
3. 与 Qlib PITProvider 的兼容性设计
"""

import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# ============================================================================
# Point-in-Time 数据管理器
# ============================================================================

@dataclass
class PITFinancialRecord:
    """一份财务报告的 PIT 记录"""
    code: str
    report_period: str  # 报告期，如 '2023Q4'
    announce_date: str  # 实际公告日期 YYYY-MM-DD
    fields: Dict[str, float]  # 财务指标

@dataclass
class StockLifecycle:
    """股票生命周期"""
    code: str
    name: str
    list_date: str  # 上市日期
    delist_date: Optional[str] = None  # 退市日期
    st_periods: List[tuple] = field(default_factory=list)  # [(start, end), ...] ST 期间

class PointInTimeProvider:
    """
    Point-in-Time 数据提供者

    核心原则：
    - 在某个交易日，只能获取该日期之前已公开的数据
    - 财务数据按公告日期而非报告期来对齐
    - 股票池仅包含当天已上市且未退市的股票

    参考 Qlib 的 PITProvider 设计：
    https://qlib.readthedocs.io/en/latest/component/data.html#pit-data
    """

    def __init__(self):
        self._financial_data: List[PITFinancialRecord] = []
        self._stock_lifecycles: Dict[str, StockLifecycle] = {}
        self._trading_calendar: pd.DatetimeIndex = None

    def set_trading_calendar(self, dates: pd.DatetimeIndex):
        """设置交易日历"""
        self._trading_calendar = sorted(dates)

    def add_financial_record(self, record: PITFinancialRecord):
        """添加一条财务记录"""
        self._financial_data.append(record)

    def add_stock_lifecycle(self, lifecycle: StockLifecycle):
        """添加一只股票的生命周期"""
        self._stock_lifecycles[lifecycle.code] = lifecycle

    def get_active_stocks(self, trade_date: str) -> List[str]:
        """
        获取某一交易日活跃的股票列表

        过滤条件：
        1. 已上市（list_date <= trade_date）
        2. 未退市（delist_date is None or delist_date > trade_date）
        3. 非停牌（is_suspended 可选）

        参考 Qlib 的 InstrumentProvider.list_instruments()
        """
        trade_dt = pd.Timestamp(trade_date)
        active = []

        for code, lc in self._stock_lifecycles.items():
            list_dt = pd.Timestamp(lc.list_date)
            if list_dt > trade_dt:
                continue  # 尚未上市

            if lc.delist_date:
                delist_dt = pd.Timestamp(lc.delist_date)
                if delist_dt <= trade_dt:
                    continue  # 已退市

            active.append(code)

        return active

    def get_latest_financial(
        self,
        trade_date: str,
        codes: List[str],
        max_lag_days: int = 120
    ) -> pd.DataFrame:
        """
        获取某一交易日可用的最新财务数据（PIT 原则）

        规则：
        - 只使用 announce_date <= trade_date 的数据
        - 对每只股票，选择公告日期最新但不超过 trade_date 的记录
        - 如果最新公告距 trade_date 超过 max_lag_days 天，标记为过期

        返回:
            DataFrame，列为 code, [各财务指标], _age_days（距公告天数）
        """
        trade_dt = pd.Timestamp(trade_date)
        records = []

        for code in codes:
            # 找到该股票在 trade_date 之前已公告的所有记录
            available = [
                r for r in self._financial_data
                if r.code == code and pd.Timestamp(r.announce_date) <= trade_dt
            ]

            if not available:
                continue

            # 取最新公告的记录
            latest = max(available, key=lambda r: (r.announce_date, r.report_period))
            age_days = (trade_dt - pd.Timestamp(latest.announce_date)).days

            record = {
                'code': code,
                'report_period': latest.report_period,
                'announce_date': latest.announce_date,
                '_age_days': age_days,
                **latest.fields
            }
            records.append(record)

        return pd.DataFrame(records)

    def get_history_financial(
        self,
        trade_date: str,
        codes: List[str],
        periods: int = 4
    ) -> pd.DataFrame:
        """
        获取前 N 个报告期的历史财务数据

        用于构建时序财务特征（如季度环比增长率）。
        每期数据都遵循 PIT 原则。

        返回:
            DataFrame，列为 code, period_index(0=最新), [各财务指标]
        """
        trade_dt = pd.Timestamp(trade_date)
        all_records = []

        for code in codes:
            available = sorted(
                [r for r in self._financial_data
                 if r.code == code and pd.Timestamp(r.announce_date) <= trade_dt],
                key=lambda r: (r.announce_date, r.report_period),
                reverse=True
            )

            for i in range(min(periods, len(available))):
                rec = available[i]
                all_records.append({
                    'code': code,
                    'period_index': i,
                    'report_period': rec.report_period,
                    **rec.fields
                })

        return pd.DataFrame(all_records)


# ============================================================================
# 未来数据泄露检测器
# ============================================================================

class LookAheadDetector:
    """
    未来数据泄露检测器

    用于检查数据处理流程中是否存在未来数据泄露：
    1. 价格复权是否使用了未来日期的复权因子
    2. 财务报表是否在公告前被使用
    3. 股票池是否包含了尚未上市/已退市的股票
    """

    @staticmethod
    def check_adjustment_leak(
        price_data: pd.DataFrame,
        adjust_factor: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        检查复权价格是否存在未来数据泄露

        原理：
        - 前复权价格会修改历史数据，因此存在泄露风险
        - 后复权相对安全，但需要验证复权因子的时间戳

        返回:
            {'leak_count': N, 'leak_samples': [...]}
        """
        if price_data.empty or adjust_factor.empty:
            return {'leak_count': 0, 'leak_samples': [], 'status': 'no_data'}

        merged = price_data.merge(adjust_factor, on=['code', 'date'], how='inner')
        if 'adj_factor' not in merged.columns:
            return {'leak_count': 0, 'leak_samples': [], 'status': 'no_factor_col'}

        leaks = []
        for _, row in merged.iterrows():
            if pd.isna(row.get('adj_factor')):
                leaks.append({'code': row['code'], 'date': str(row['date'])})

        return {
            'leak_count': len(leaks),
            'leak_samples': leaks[:5],
            'status': 'ok' if len(leaks) == 0 else 'has_nan'
        }

    @staticmethod
    def check_financial_data_leak(
        financial_data: pd.DataFrame,
        announce_dates: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        检查财务报表数据是否在公告日期前被使用

        参数:
            financial_data: 含 code, date（使用日期）, report_period, 各财务指标
            announce_dates: 含 code, report_period, announce_date（实际公告日期）

        返回:
            {'leak_count': N, 'leak_samples': [...]}
        """
        if financial_data.empty or announce_dates.empty:
            return {'leak_count': 0, 'leak_samples': [], 'status': 'no_data'}

        merged = financial_data.merge(
            announce_dates, on=['code', 'report_period'], how='left'
        )

        # 找出使用日期早于公告日期的记录（数据泄露）
        merged['use_date'] = pd.to_datetime(merged['date'])
        merged['announce_dt'] = pd.to_datetime(merged['announce_date'])

        leaks = merged[merged['use_date'] < merged['announce_dt']]

        return {
            'leak_count': len(leaks),
            'leak_samples': leaks[['code', 'date', 'report_period', 'announce_date']].head(5).to_dict('records'),
            'status': 'ok' if len(leaks) == 0 else 'leak_detected'
        }

    @staticmethod
    def check_survivorship_bias(
        stock_pool_at_date: Dict[str, List[str]],
        stock_lifecycles: Dict[str, StockLifecycle]
    ) -> Dict[str, Any]:
        """
        检测存活偏差

        参数:
            stock_pool_at_date: {date: [codes]} 每个交易日使用的股票池
            stock_lifecycles: 包含所有股票的真实生命周期

        返回:
            每个交易日的偏差统计
        """
        results = {
            'dates_checked': 0,
            'dates_with_bias': 0,
            'unlisted_included': 0,
            'delisted_included': 0,
            'bias_samples': []
        }

        for trade_date, pool in stock_pool_at_date.items():
            results['dates_checked'] += 1
            trade_dt = pd.Timestamp(trade_date)
            date_has_bias = False

            for code in pool:
                lc = stock_lifecycles.get(code)
                if not lc:
                    continue

                list_dt = pd.Timestamp(lc.list_date)
                if list_dt > trade_dt:
                    results['unlisted_included'] += 1
                    date_has_bias = True
                    results['bias_samples'].append({
                        'date': trade_date,
                        'code': code,
                        'issue': 'unlisted',
                        'list_date': str(list_dt.date())
                    })

                if lc.delist_date:
                    delist_dt = pd.Timestamp(lc.delist_date)
                    if delist_dt <= trade_dt:
                        results['delisted_included'] += 1
                        date_has_bias = True
                        results['bias_samples'].append({
                            'date': trade_date,
                            'code': code,
                            'issue': 'delisted',
                            'delist_date': str(delist_dt.date())
                        })

            if date_has_bias:
                results['dates_with_bias'] += 1

        results['status'] = 'ok' if results['dates_with_bias'] == 0 else 'bias_detected'
        return results


# ============================================================================
# 测试用例
# ============================================================================

class TestPointInTimeProvider(unittest.TestCase):
    """Point-in-Time 数据提供者测试"""

    def setUp(self):
        self.pit = PointInTimeProvider()

        # 设置交易日历
        self.pit.set_trading_calendar(
            pd.date_range('2023-01-01', '2024-12-31', freq='B')
        )

        # 添加股票生命周期
        self.pit.add_stock_lifecycle(StockLifecycle(
            code='000001.SZ', name='平安银行',
            list_date='1991-04-03'
        ))
        self.pit.add_stock_lifecycle(StockLifecycle(
            code='000002.SZ', name='万科A',
            list_date='1991-01-29'
        ))
        self.pit.add_stock_lifecycle(StockLifecycle(
            code='688001.SH', name='华兴源创',
            list_date='2019-07-22'
        ))
        self.pit.add_stock_lifecycle(StockLifecycle(
            code='600000.SH', name='浦发银行',
            list_date='1999-11-10',
            delist_date='2024-06-01'  # 模拟退市
        ))

        # 添加财务数据（模拟 PIT）
        self.pit.add_financial_record(PITFinancialRecord(
            code='000001.SZ', report_period='2023Q4',
            announce_date='2024-04-20',
            fields={'roe': 0.15, 'eps': 1.5, 'bvps': 15.0}
        ))
        self.pit.add_financial_record(PITFinancialRecord(
            code='000001.SZ', report_period='2024Q1',
            announce_date='2024-04-30',
            fields={'roe': 0.04, 'eps': 0.4, 'bvps': 15.4}
        ))
        self.pit.add_financial_record(PITFinancialRecord(
            code='000001.SZ', report_period='2024Q2',
            announce_date='2024-08-28',
            fields={'roe': 0.08, 'eps': 0.8, 'bvps': 15.8}
        ))
        self.pit.add_financial_record(PITFinancialRecord(
            code='000002.SZ', report_period='2023Q4',
            announce_date='2024-03-29',
            fields={'roe': 0.10, 'eps': 1.2, 'bvps': 12.0}
        ))

    def test_active_stocks_filtering(self):
        """测试活跃股票过滤（防止存活偏差）"""
        # 在科创板第一只股票上市前（2019-07-22），不应包含 688001.SH
        before_listing = self.pit.get_active_stocks('2019-07-20')
        self.assertNotIn('688001.SH', before_listing)

        # 上市后应包含
        after_listing = self.pit.get_active_stocks('2019-07-23')
        self.assertIn('688001.SH', after_listing)

        # 退市后不应包含
        after_delist = self.pit.get_active_stocks('2024-06-02')
        self.assertNotIn('600000.SH', after_delist)

        # 退市前一天应包含
        before_delist = self.pit.get_active_stocks('2024-05-31')
        self.assertIn('600000.SH', before_delist)

        print(f"\n  上市前(2019-07-20)股票池: {before_listing}")
        print(f"  上市后(2019-07-23)股票池: {after_listing}")
        print(f"  退市后(2024-06-02)股票池: {after_delist}")

    def test_financial_data_pit_principle(self):
        """测试 PIT 财务数据原则"""
        # 在 2024Q1 公告前（2024-04-30），不应使用 2024Q1 数据
        before_announce = self.pit.get_latest_financial(
            '2024-04-29', ['000001.SZ']
        )
        if not before_announce.empty:
            # 此时最新可用的是 2023Q4（公告于 2024-04-20）
            self.assertEqual(
                before_announce.iloc[0]['report_period'], '2023Q4',
                "公告前应使用 2023Q4 数据"
            )

        # 公告后应使用 2024Q1
        after_announce = self.pit.get_latest_financial(
            '2024-05-01', ['000001.SZ']
        )
        self.assertEqual(
            after_announce.iloc[0]['report_period'], '2024Q1',
            "公告后应使用 2024Q1 数据"
        )

        print(f"\n  PIT 测试:")
        print(f"    2024-04-29 可用财报: {before_announce['report_period'].values if not before_announce.empty else '无'}")
        print(f"    2024-05-01 可用财报: {after_announce['report_period'].values}")

    def test_pit_vs_naive(self):
        """
        对比 PIT 方式与原始方式：模拟数据泄露

        假设回测在 2024 年 4 月进行：
        - PIT 方式：只能使用 4 月前已公告的财报（2023Q4）
        - 泄露方式：直接按报告期取最新（可能取到 2024Q2，但实际 8月才公告）
        """
        # PIT 方式
        pit_data = self.pit.get_latest_financial('2024-05-15', ['000001.SZ'])
        pit_period = pit_data.iloc[0]['report_period'] if not pit_data.empty else None

        # 模拟泄露：直接取所有记录中报告期最新的
        all_records = [
            r for r in self.pit._financial_data if r.code == '000001.SZ'
        ]
        naive_latest = max(all_records, key=lambda r: r.report_period)

        print(f"\n  PIT vs Naive 对比:")
        print(f"    交易日期: 2024-05-15")
        print(f"    PIT 方式取到财报: {pit_period}")
        print(f"    直接取最新财报: {naive_latest.report_period} (公告于 {naive_latest.announce_date})")
        print(f"    若直接取最新: {'数据泄露!' if naive_latest.report_period > pit_period else '无泄露'}")

        # pit 取到的报告期不应该晚于 naive 直接取的
        if pit_period:
            self.assertLessEqual(pit_period, naive_latest.report_period)

    def test_history_financial(self):
        """测试历史多期限财务数据获取"""
        history = self.pit.get_history_financial(
            '2024-05-15', ['000001.SZ'], periods=4
        )

        self.assertGreaterEqual(len(history), 1)
        self.assertIn('period_index', history.columns)
        self.assertIn('roe', history.columns)
        self.assertTrue((history['period_index'] >= 0).all())

        print(f"\n  多期限财务数据:")
        print(history[['code', 'period_index', 'report_period', 'roe', 'eps']].to_string())


class TestLookAheadDetector(unittest.TestCase):
    """未来数据泄露检测器测试"""

    def test_survivorship_bias_detection(self):
        """测试存活偏差检测"""
        # 模拟一个有偏差的场景：在股票未上市时将其纳入回测池
        stock_cycles = {
            '000001.SZ': StockLifecycle(code='000001.SZ', name='平安银行', list_date='1991-04-03'),
            '688001.SH': StockLifecycle(code='688001.SH', name='华兴源创', list_date='2019-07-22'),
        }

        # 模拟回测池：在 2019-07-01 错误地包含了 688001.SH（尚未上市）
        pool_at_date = {
            '2019-07-01': ['000001.SZ', '688001.SH'],
            '2019-07-23': ['000001.SZ', '688001.SH'],
        }

        result = LookAheadDetector.check_survivorship_bias(pool_at_date, stock_cycles)

        self.assertEqual(result['dates_with_bias'], 1)
        self.assertEqual(result['unlisted_included'], 1)
        self.assertEqual(result['status'], 'bias_detected')

        print(f"\n  存活偏差检测结果:")
        print(f"    检查交易日数: {result['dates_checked']}")
        print(f"    有偏差的交易日数: {result['dates_with_bias']}")
        print(f"    包含未上市股票次数: {result['unlisted_included']}")
        for s in result['bias_samples']:
            print(f"    {s}")

    def test_no_bias_detection(self):
        """测试无偏差场景"""
        stock_cycles = {
            '000001.SZ': StockLifecycle(code='000001.SZ', name='平安银行', list_date='1991-04-03'),
        }

        pool_at_date = {
            '2023-01-03': ['000001.SZ'],
            '2023-01-04': ['000001.SZ'],
        }

        result = LookAheadDetector.check_survivorship_bias(pool_at_date, stock_cycles)

        self.assertEqual(result['dates_with_bias'], 0)
        self.assertEqual(result['status'], 'ok')

    def test_financial_leak_detection(self):
        """测试财务数据泄露检测"""
        # 模拟：回测中使用财报数据的时间早于公告日期
        financial_data = pd.DataFrame([
            {'code': '000001.SZ', 'date': '2024-03-15', 'report_period': '2023Q4',
             'roe': 0.15, 'eps': 1.5},
            {'code': '000001.SZ', 'date': '2024-05-01', 'report_period': '2024Q1',
             'roe': 0.04, 'eps': 0.4},
        ])

        announce_dates = pd.DataFrame([
            {'code': '000001.SZ', 'report_period': '2023Q4', 'announce_date': '2024-04-20'},
            {'code': '000001.SZ', 'report_period': '2024Q1', 'announce_date': '2024-04-30'},
        ])

        result = LookAheadDetector.check_financial_data_leak(financial_data, announce_dates)

        # 2023Q4 在 2024-03-15 被使用，但 2024-04-20 才公告 → 泄露
        self.assertEqual(result['leak_count'], 1)
        self.assertEqual(result['status'], 'leak_detected')

        print(f"\n  财务数据泄露检测:")
        print(f"    泄露记录数: {result['leak_count']}")
        for s in result['leak_samples']:
            print(f"    {s}")


class TestPITIntegration(unittest.TestCase):
    """PIT 与现有 Factor Engine 的集成测试"""

    def test_pit_factor_alignment(self):
        """
        测试 PIT 数据与因子计算的时序对齐

        场景：在回测时，因子计算只能使用当时已知的财务数据
        """
        pit = PointInTimeProvider()
        pit.set_trading_calendar(pd.date_range('2023-01-01', '2024-12-31', freq='B'))

        pit.add_stock_lifecycle(StockLifecycle(
            code='000001.SZ', name='平安银行', list_date='1991-04-03'
        ))

        pit.add_financial_record(PITFinancialRecord(
            code='000001.SZ', report_period='2023Q4',
            announce_date='2024-04-20',
            fields={'roe': 0.15, 'eps': 1.5}
        ))
        pit.add_financial_record(PITFinancialRecord(
            code='000001.SZ', report_period='2024Q1',
            announce_date='2024-04-30',
            fields={'roe': 0.04, 'eps': 0.4}
        ))

        # 模拟逐日回测循环
        backtest_dates = [
            '2024-03-01',  # 此时只能用 2023Q3 或更早的财报（没有数据，应该跳过）
            '2024-04-25',  # 此时 2023Q4 已公告（4月20日），可用
            '2024-05-15',  # 此时 2024Q1 已公告（4月30日），可用
        ]

        available_values = {}
        for dt in backtest_dates:
            active = pit.get_active_stocks(dt)
            fin = pit.get_latest_financial(dt, active)
            if not fin.empty:
                available_values[dt] = fin.iloc[0].to_dict()

        # 2024-03-01 没有可用财报（2023Q4 在 2024-04-20 才公告）
        self.assertNotIn('2024-03-01', available_values)

        # 2024-04-25 可以用 2023Q4
        self.assertIn('2024-04-25', available_values)
        self.assertEqual(available_values['2024-04-25']['report_period'], '2023Q4')

        # 2024-05-15 可以用 2024Q1
        self.assertIn('2024-05-15', available_values)
        self.assertEqual(available_values['2024-05-15']['report_period'], '2024Q1')

        print(f"\n  PIT 因子对齐测试:")
        for dt, vals in available_values.items():
            print(f"    日期 {dt}: report_period={vals['report_period']}, roe={vals['roe']}")


if __name__ == '__main__':
    unittest.main(verbosity=2)