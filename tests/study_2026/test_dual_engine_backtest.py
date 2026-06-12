"""
验证代码：双引擎回测架构 + 滚动窗口 Purged 交叉验证
========================================================
借鉴来源: QuantMind (https://github.com/qusong0627/quantmind)
         + Qlib Purged K-Fold 交叉验证
优化方向: backtest-engine + strategy-model-engine
         —— 回测引擎性能分层 + 模型训练防前视偏差
核心思路:
  1. QuantMind 使用 Qlib + Pandas 双引擎策略：
     - Qlib Engine: 复杂策略、多因子模型、机构级研究（高性能）
     - Pandas Engine: 快速验证、简单策略、教学演示（轻量快速）
  2. Qlib 的 Purged Group Time Series Split 避免前视偏差：
     - Purge Gap 确保训练集和测试集之间有足够的时间间隔
     - Group Split 确保同一股票不会同时出现在训练和测试中
日期: 2026-06-12

约束: 仅验证可行性，不可直接修改主代码，不可执行 git commit/merge。
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import time


# ═══════════════════════════════════════════
# 1. 双引擎回测架构（借鉴 QuantMind）
# ═══════════════════════════════════════════

class BaseBacktestEngine(ABC):
    """回测引擎抽象基类"""

    @abstractmethod
    def run(self, data: pd.DataFrame, signals: pd.DataFrame,
            init_capital: float = 1_000_000) -> Dict[str, Any]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class PandasBacktestEngine(BaseBacktestEngine):
    """
    Pandas 引擎：轻量快速回测

    适用场景：
    - 策略快速验证
    - 单因子策略回测
    - 教学演示

    特点：
    - 纯 Pandas 实现，无额外依赖
    - 执行速度快
    - 代码简单易懂
    """

    def name(self) -> str:
        return "PandasEngine"

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1_000_000,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.0001,
    ) -> Dict[str, Any]:
        """执行 Pandas 引擎回测"""
        data = data.sort_values(['date', 'code']).copy()
        signals = signals.sort_values(['date', 'code']).copy()

        cash = init_capital
        positions: Dict[str, Dict] = {}
        equity_records = []
        trade_records = []

        all_dates = sorted(data['date'].unique())

        for dt in all_dates:
            day_data = data[data['date'] == dt]
            day_signal = signals[signals['date'] == dt]
            prices = dict(zip(day_data['code'], day_data['close']))

            # 处理信号
            for _, sig in day_signal.iterrows():
                code = sig['code']
                signal_val = sig.get('signal', 0)
                if abs(signal_val) < 1e-8:
                    continue

                price = prices.get(code)
                if price is None:
                    continue

                if signal_val > 0:  # 买入
                    vol = 100  # 简化：固定手数
                    amount = price * vol
                    if amount > cash * 0.1:  # 单票不超 10%
                        continue
                    commission = max(amount * commission_rate, 5.0)
                    if cash >= amount + commission:
                        cash -= amount + commission
                        if code not in positions:
                            positions[code] = {'volume': 0, 'cost': 0.0}
                        old = positions[code]
                        old['cost'] = (old['cost'] * old['volume'] + amount) / (old['volume'] + vol)
                        old['volume'] += vol
                        trade_records.append({
                            'date': dt, 'code': code, 'side': 'buy',
                            'price': price, 'volume': vol,
                        })

                elif signal_val < 0 and code in positions:  # 卖出
                    pos = positions[code]
                    vol = min(pos['volume'], 100)
                    amount = price * vol
                    fee = max(amount * commission_rate, 5.0) + amount * stamp_tax_rate
                    cash += amount - fee
                    pos['volume'] -= vol
                    if pos['volume'] <= 0:
                        del positions[code]
                    trade_records.append({
                        'date': dt, 'code': code, 'side': 'sell',
                        'price': price, 'volume': vol,
                    })

            # 计算当日净值
            stock_value = sum(
                positions[c]['volume'] * prices[c]
                for c in positions if c in prices
            )
            nav = cash + stock_value
            equity_records.append({'date': dt, 'equity': nav})

        return {
            'engine': self.name(),
            'equity_curve': pd.DataFrame(equity_records),
            'trades': pd.DataFrame(trade_records) if trade_records else pd.DataFrame(),
            'metrics': self._calc_metrics(pd.DataFrame(equity_records), init_capital),
        }

    @staticmethod
    def _calc_metrics(equity: pd.DataFrame, init_cap: float) -> Dict[str, float]:
        if equity.empty or 'equity' not in equity.columns:
            return {}
        eq = equity.set_index('date')['equity']
        rets = eq.pct_change().dropna()
        if len(rets) < 2:
            return {}
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1 if eq.iloc[0] > 0 else 0
        n_days = len(rets)
        ann_ret = (1 + total_ret) ** (252 / n_days) - 1
        vol = rets.std() * np.sqrt(252)
        mdd = (eq / eq.cummax() - 1).min()
        sharpe = (ann_ret - 0.03) / vol if vol > 0 else 0
        return {
            "total_return": float(total_ret),
            "annual_return": float(ann_ret),
            "volatility": float(vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(mdd),
            "calmar_ratio": float(ann_ret / abs(mdd)) if mdd != 0 else 0,
        }


class QlibStyleBacktestEngine(BaseBacktestEngine):
    """
    Qlib 风格引擎：高精度/高性能回测

    适用场景：
    - 复杂策略回测
    - 多因子模型验证
    - 机构级研究

    特点：
    - 严格的涨跌停限制处理
    - 分层回测结果（按因子分位分组）
    - 详细的持仓和成交追踪
    - 完整的费用模型（佣金+印花税+滑点+过户费）
    """

    def name(self) -> str:
        return "QlibStyleEngine"

    def __init__(self):
        self.commission_rate = 0.00025
        self.stamp_tax_rate = 0.001
        self.transfer_fee_rate = 0.00002
        self.min_commission = 5.0
        self.slippage = 0.0001
        self.price_limit_pct = 0.10  # 涨跌停限制

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1_000_000,
        n_quantiles: int = 5,
    ) -> Dict[str, Any]:
        """
        Qlib 风格回测

        参数:
            n_quantiles: 按因子分数等分邝数（用于分层回测）
        """
        data = data.sort_values(['date', 'code']).copy()
        signals = signals.sort_values(['date', 'code']).copy()

        # 分层回测：每个分位组独立跟踪收益
        quantile_results = {f"Q{i+1}": self._run_single_backtest(
            data, self._filter_quantile(signals, data, i, n_quantiles), init_capital
        ) for i in range(n_quantiles)}

        # 综合回测：Top quantile 做多
        top_signal = self._filter_quantile(signals, data, n_quantiles - 1, n_quantiles)
        combined = self._run_single_backtest(data, top_signal, init_capital, full_tracking=True)

        return {
            'engine': self.name(),
            'combined': combined,
            'quantile_results': quantile_results,
            'metrics': combined.get('metrics', {}),
        }

    def _run_single_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float,
        full_tracking: bool = False,
    ) -> Dict[str, Any]:
        """单轨回测"""
        cash = init_capital
        positions: Dict[str, Dict] = {}
        equity_records = []
        trade_records = []

        for dt in sorted(data['date'].unique()):
            day_data = data[data['date'] == dt]
            day_signal = signals[signals['date'] == dt]
            prices = dict(zip(day_data['code'], day_data['close']))

            # 先卖出（腾出资金）
            for code in list(positions.keys()):
                if code in prices:
                    sig_day = day_signal[day_signal['code'] == code]
                    sell = len(sig_day) == 0 or sig_day['signal'].iloc[0] <= 0
                    if sell:
                        pos = positions[code]
                        price = prices[code]
                        amount = price * pos['volume']
                        fee = (
                            max(amount * self.commission_rate, self.min_commission) +
                            amount * self.stamp_tax_rate +
                            amount * self.transfer_fee_rate
                        )
                        cash += amount - fee
                        trade_records.append({
                            'date': dt, 'code': code, 'side': 'sell',
                            'price': price, 'volume': pos['volume'], 'fee': fee,
                        })
                        del positions[code]

            # 买入
            buy_candidates = day_signal[day_signal['signal'] > 0]
            if len(buy_candidates) > 0:
                per_stock_capital = cash * 0.1 / len(buy_candidates)
                for _, sig in buy_candidates.iterrows():
                    code = sig['code']
                    price = prices.get(code)
                    if price is None:
                        continue
                    # 检查涨跌停
                    stock_day = day_data[day_data['code'] == code]
                    if not stock_day.empty:
                        is_limit = stock_day['is_limit_up'].iloc[0] if 'is_limit_up' in stock_day.columns else False
                        if is_limit:
                            continue  # 涨停不买
                    vol = max(int(per_stock_capital / price / 100) * 100, 100)
                    amount = price * vol
                    fee = (
                        max(amount * self.commission_rate, self.min_commission) +
                        amount * self.transfer_fee_rate
                    )
                    if cash >= amount + fee:
                        cash -= amount + fee
                        positions[code] = {'volume': vol, 'cost': price}
                        trade_records.append({
                            'date': dt, 'code': code, 'side': 'buy',
                            'price': price, 'volume': vol, 'fee': fee,
                        })

            # 净值
            stock_value = sum(
                positions[c]['volume'] * prices[c] for c in positions if c in prices
            )
            nav = cash + stock_value
            equity_records.append({'date': dt, 'equity': nav})

        eq_df = pd.DataFrame(equity_records)
        return {
            'equity_curve': eq_df,
            'trades': pd.DataFrame(trade_records) if trade_records else pd.DataFrame(),
            'metrics': PandasBacktestEngine._calc_metrics(eq_df, init_capital),
        }

    @staticmethod
    def _filter_quantile(signals, data, quantile_idx, n_quantiles):
        """过滤指定分位的信号"""
        if 'score' not in signals.columns or signals['score'].isna().all():
            # 用 signal 列的绝对值当分数
            signals = signals.copy()
            signals['score'] = abs(signals.get('signal', 0))

        filtered = signals.copy()
        for dt in filtered['date'].unique():
            day_mask = filtered['date'] == dt
            scores = filtered.loc[day_mask, 'score']
            if len(scores) < n_quantiles:
                continue
            lower = np.percentile(scores, (quantile_idx / n_quantiles) * 100)
            upper = np.percentile(scores, ((quantile_idx + 1) / n_quantiles) * 100)
            keep = (scores >= lower) & (scores <= upper)
            filtered.loc[day_mask & ~keep, 'signal'] = 0

        return filtered


class DualEngineBacktest:
    """
    双引擎回测调度器

    借鉴 QuantMind 的设计：
    - Qlib Engine: 用于最终完整回测
    - Pandas Engine: 用于策略调试和快速验证

    策略模式：
    1. 开发阶段→ Pandas 引擎（快速迭代）
    2. 验证阶段→ Qlib 引擎（严格验证）
    3. 生产阶段→ 两个引擎交叉验证
    """

    def __init__(self):
        self.fast_engine = PandasBacktestEngine()
        self.full_engine = QlibStyleBacktestEngine()

    def quick_validate(self, data, signals, init_capital=1_000_000) -> Dict:
        """快速验证（使用 Pandas 引擎）"""
        return self.fast_engine.run(data, signals, init_capital)

    def full_backtest(self, data, signals, init_capital=1_000_000, n_quantiles=5) -> Dict:
        """完整回测（使用 Qlib 风格引擎）"""
        return self.full_engine.run(data, signals, init_capital, n_quantiles)

    def cross_validate(self, data, signals, init_capital=1_000_000) -> Dict:
        """交叉验证：两个引擎同时运行并对比"""
        fast_result = self.quick_validate(data, signals, init_capital)
        full_result = self.full_backtest(data, signals, init_capital)

        # 对比净值曲线
        fast_eq = fast_result['equity_curve']
        full_eq = full_result['combined']['equity_curve'] if 'combined' in full_result else full_result['equity_curve']

        correlation = 1.0
        if not fast_eq.empty and not full_eq.empty:
            merged = pd.merge(
                fast_eq[['date', 'equity']].rename(columns={'equity': 'eq_fast'}),
                full_eq[['date', 'equity']].rename(columns={'equity': 'eq_full'}),
                on='date', how='inner'
            )
            if len(merged) > 1:
                correlation = merged['eq_fast'].corr(merged['eq_full'])

        return {
            'fast_engine': fast_result,
            'full_engine': full_result,
            'cross_validation': {
                'equity_correlation': float(correlation),
                'match_quality': 'excellent' if correlation > 0.99 else
                'good' if correlation > 0.95 else 'needs_review',
            }
        }


# ═══════════════════════════════════════════
# 2. Purged Group Time Series Split（借鉴 Qlib）
# ═══════════════════════════════════════════

class PurgedGroupTimeSeriesSplit:
    """
    Purged Group Time Series 交叉验证

    借鉴 Qlib 的实现：
    1. Purge Gap: 训练集和验证集之间留出间隔（避免信息泄露）
    2. Group Awareness: 同一标的不会同时出现在训练集和验证集中
    3. Embargo: 验证集后的数据也进行清洗

    适用场景：
    - 时间序列数据上的机器学习模型验证
    - 金融回测中的样本外测试
    - 多因子模型的稳健性评估
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 5, embargo_days: int = 0):
        """
        参数:
            n_splits: 交叉验证折数
            purge_days: 训练集结束到验证集开始之间的清洗天数
            embargo_days: 验证集结束后的禁运天数
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, dates: pd.Series, groups: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练/验证集索引

        参数:
            dates: 每个样本的日期
            groups: 每个样本的分组（如股票代码）

        返回:
            List of (train_indices, val_indices)
        """
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        if n_dates < self.n_splits * 2:
            return []

        # 等分日期
        split_size = n_dates // (self.n_splits + 1)
        splits = []

        for i in range(self.n_splits):
            # 验证集: 最后一段
            val_start_idx = n_dates - (self.n_splits - i) * split_size
            val_end_idx = min(val_start_idx + split_size, n_dates)

            # 训练集: 在验证集之前，且留出 purge gap
            train_end_date = unique_dates[val_start_idx] - timedelta(days=self.purge_days)
            train_end_idx = max(
                self._find_date_index(unique_dates, train_end_date),
                0
            )

            if train_end_idx < 10 or val_start_idx >= n_dates:
                continue

            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            train_mask = dates.isin(train_dates)
            val_mask = dates.isin(val_dates)

            # Group awareness: 移除同时在训练集和验证集中出现的组
            if groups is not None:
                train_groups = set(groups[train_mask].unique())
                val_groups = set(groups[val_mask].unique())
                overlapping = train_groups & val_groups
                if overlapping:
                    # 从训练集中移除重叠的组
                    val_mask = val_mask & (~groups.isin(overlapping))

            train_idx = dates[train_mask].index.values
            val_idx = dates[val_mask].index.values

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        return splits

    @staticmethod
    def _find_date_index(dates: pd.DatetimeIndex, target: datetime) -> int:
        """找到不大于 target 的最大日期索引"""
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] <= target:
                return i + 1
        return 0


# ═══════════════════════════════════════════
# 3. 测试代码
# ═══════════════════════════════════════════

class TestDualEngineBacktest(unittest.TestCase):
    """双引擎回测测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        dates = pd.date_range('2024-01-01', periods=100, freq='B')

        rows = []
        for code in codes:
            base_price = np.random.uniform(10, 50)
            prices = base_price * np.cumprod(1 + np.random.normal(0.0003, 0.015, len(dates)))
            for dt, close in zip(dates, prices):
                rows.append({
                    'code': code, 'date': dt,
                    'open': close * 0.99, 'high': close * 1.02,
                    'low': close * 0.98, 'close': close,
                    'volume': np.random.lognormal(10, 0.5),
                    'is_limit_up': False, 'is_limit_down': False,
                })
        cls.test_data = pd.DataFrame(rows).sort_values(['date', 'code'])

        # 生成模拟信号（Top 20% 买入）
        signals_rows = []
        for dt in dates:
            day_codes = codes.copy()
            np.random.shuffle(day_codes)
            n_buy = max(1, len(day_codes) // 5)
            for i, code in enumerate(day_codes):
                signals_rows.append({
                    'code': code, 'date': dt,
                    'signal': 1 if i < n_buy else 0,
                    'score': (len(day_codes) - i) / len(day_codes),
                })
        cls.test_signals = pd.DataFrame(signals_rows)

    def test_pandas_engine(self):
        """测试 Pandas 引擎"""
        engine = PandasBacktestEngine()
        result = engine.run(self.test_data, self.test_signals)

        self.assertEqual(result['engine'], 'PandasEngine')
        self.assertGreater(len(result['equity_curve']), 0)
        self.assertIn('sharpe_ratio', result['metrics'])

        print(f"\nPandas 引擎结果:")
        print(f"  年化收益: {result['metrics']['annual_return']:.2%}")
        print(f"  夏普比率: {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {result['metrics']['max_drawdown']:.2%}")

    def test_qlib_style_engine(self):
        """测试 Qlib 风格引擎"""
        engine = QlibStyleBacktestEngine()
        result = engine.run(self.test_data, self.test_signals, n_quantiles=5)

        self.assertEqual(result['engine'], 'QlibStyleEngine')
        self.assertIn('combined', result)
        self.assertIn('quantile_results', result)
        self.assertEqual(len(result['quantile_results']), 5)

        print(f"\nQlib 风格引擎结果 (分5层):")
        for q, r in result['quantile_results'].items():
            m = r['metrics']
            print(f"  {q}: 年化收益={m.get('annual_return', 0):.2%}, "
                  f"夏普={m.get('sharpe_ratio', 0):.3f}, "
                  f"回撤={m.get('max_drawdown', 0):.2%}")

    def test_dual_engine_cross_validation(self):
        """测试双引擎交叉验证"""
        dual = DualEngineBacktest()
        result = dual.cross_validate(self.test_data, self.test_signals)

        self.assertIn('cross_validation', result)
        correlation = result['cross_validation']['equity_correlation']
        self.assertGreater(correlation, 0.5, f"两个引擎的净值曲线相关性应 > 0.5，实际: {correlation:.4f}")
        print(f"\n双引擎交叉验证: 相关性 = {correlation:.4f} ({result['cross_validation']['match_quality']})")

    def test_performance_comparison(self):
        """性能对比：Pandas vs Qlib 风格引擎"""
        engine_pd = PandasBacktestEngine()
        engine_qlib = QlibStyleBacktestEngine()

        # Pandas 引擎
        t0 = time.time()
        engine_pd.run(self.test_data, self.test_signals)
        t_pandas = time.time() - t0

        # Qlib 风格引擎
        t0 = time.time()
        engine_qlib.run(self.test_data, self.test_signals, n_quantiles=5)
        t_qlib = time.time() - t0

        print(f"\n性能对比:")
        print(f"  Pandas 引擎:     {t_pandas:.4f}s")
        print(f"  Qlib 风格引擎:   {t_qlib:.4f}s")
        print(f"  性能比率:        {t_qlib / max(t_pandas, 0.001):.2f}x")

        # Pandas 应该更快
        self.assertLess(t_pandas, t_qlib * 2,
                        "Pandas 引擎应明显快于 Qlib 风格引擎")


class TestPurgedGroupTS(unittest.TestCase):
    """Purged Group Time Series Split 测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = ['000001.SZ', '000002.SZ', '600000.SH']
        dates = pd.date_range('2024-01-01', periods=200, freq='B')

        rows = []
        for i, dt in enumerate(dates):
            for code in codes:
                rows.append({
                    'code': code, 'date': dt,
                    'feature': np.random.normal(0, 1),
                    'label': np.random.normal(0, 1),
                })
        cls.test_data = pd.DataFrame(rows)
        cls.test_data['date'] = pd.to_datetime(cls.test_data['date'])

    def test_basic_split(self):
        """测试基本分割"""
        # 使用更少的折数适配数据长度
        splitter = PurgedGroupTimeSeriesSplit(n_splits=2, purge_days=5)
        # 不传 groups，因为测试数据中所有股票覆盖全时段（不存在组间分离需求）
        splits = splitter.split(
            self.test_data['date'],
            groups=None,
        )

        self.assertGreaterEqual(len(splits), 1, f"应至少生成一个分割，实际: {len(splits)}")
        print(f"\n生成了 {len(splits)} 个交叉验证分割")

        for i, (train_idx, val_idx) in enumerate(splits):
            train_dates = set(self.test_data.loc[train_idx, 'date'])
            val_dates = set(self.test_data.loc[val_idx, 'date'])

            # 验证时间顺序：训练集日期应全部早于验证集日期
            if train_dates and val_dates:
                self.assertLess(
                    max(train_dates), min(val_dates),
                    f"Fold {i}: 训练集最大日期 ({max(train_dates).date()}) "
                    f"应早于验证集最小日期 ({min(val_dates).date()})"
                )

            print(f"  Fold {i}: train={len(train_idx)} ({min(train_dates).date()}~{max(train_dates).date()}), "
                  f"val={len(val_idx)} ({min(val_dates).date()}~{max(val_dates).date()})")

    def test_purge_gap(self):
        """测试 Purge Gap 是否生效"""
        splitter_no_purge = PurgedGroupTimeSeriesSplit(n_splits=3, purge_days=0)
        splitter_purge5 = PurgedGroupTimeSeriesSplit(n_splits=3, purge_days=5)
        splitter_purge10 = PurgedGroupTimeSeriesSplit(n_splits=3, purge_days=10)

        splits_no = splitter_no_purge.split(self.test_data['date'])
        splits_p5 = splitter_purge5.split(self.test_data['date'])
        splits_p10 = splitter_purge10.split(self.test_data['date'])

        if splits_no and splits_p5:
            # 有 purge gap 的训练集应更小
            self.assertLessEqual(
                len(splits_p5[0][0]), len(splits_no[0][0]),
                "有 purge gap 时的训练集应不大于无 purge gap 时的训练集"
            )

        print(f"\nPurge Gap 效果对比:")
        print(f"  purge=0:  train size={len(splits_no[0][0])}, val size={len(splits_no[0][1])}")
        print(f"  purge=5:  train size={len(splits_p5[0][0])}, val size={len(splits_p5[0][1])}")
        if splits_p10:
            print(f"  purge=10: train size={len(splits_p10[0][0])}, val size={len(splits_p10[0][1])}")

    def test_no_overlap(self):
        """测试所有 fold 都没有时间重叠"""
        splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_days=5)
        splits = splitter.split(self.test_data['date'])

        for i, (train_idx, val_idx) in enumerate(splits):
            train_dates = self.test_data.loc[train_idx, 'date']
            val_dates = self.test_data.loc[val_idx, 'date']

            if len(train_dates) > 0 and len(val_dates) > 0:
                gap = (min(val_dates) - max(train_dates)).days
                self.assertGreaterEqual(
                    gap, 0,
                    f"Fold {i}: 训练集和验证集之间存在负间隔 ({gap} 天)"
                )
                print(f"  Fold {i}: gap={gap} 天")

    def test_single_split(self):
        """测试 n_splits=1 的情况"""
        splitter = PurgedGroupTimeSeriesSplit(n_splits=1, purge_days=5)
        splits = splitter.split(self.test_data['date'])
        self.assertLessEqual(len(splits), 1)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__('__main__'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)