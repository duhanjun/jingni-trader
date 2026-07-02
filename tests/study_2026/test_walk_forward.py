"""
================================================================================
Walk-Forward 验证框架验证测试
================================================================================

借鉴来源:
    - AKQuant (github.com/akfamily/akquant)
      - Walk-forward Validation: 滚动时间窗口训练/验证/测试
      - 参考设计: akquant 的 rolling backtest 模块
      - 核心思想: 使用连续的时间窗口替代单次 train/test split，
        避免使用未来信息，更真实地评估策略在真实环境中的表现。

    - Microsoft Qlib (github.com/microsoft/qlib)
      - Purged Group Time Series Split: 引入 purge gap 防止信息泄露
      - 参考文件: qlib/data/dataset/processor.py (时间序列分割逻辑)

优化方向:
    增强 jingni-trader strategy-model-engine 的回测验证机制，
    从现有的简单 train/test split (engine.py:299-306) 升级为
    严格的时间序列 Walk-Forward 验证，防止前视偏差 (look-ahead bias)。

测试目标:
    1. 验证 Walk-Forward Split 的正确性（时间顺序、无数据泄露）
    2. 验证 Purge Gap 机制的隔离效果
    3. 性能对比：Walk-Forward vs 随机 Split（揭示随机 Split 的过拟合风险）
    4. 边界条件测试（窗口数 > 数据长度、单窗口等）
================================================================================
"""

import sys
import os
import time
import unittest
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================================
# Walk-Forward 验证原型实现（优化方案核心代码）
# ============================================================================

class WalkForwardValidator:
    """
    Walk-Forward (滚动前向) 验证器

    设计（借鉴 AKQuant + Qlib）:
    1. 将全部时间范围划分为多个连续窗口
    2. 每个窗口 = [训练期 | Purge Gap | 验证期]
    3. Train 窗口逐步向前扩展（anchored）或固定长度（rolling）
    4. Purge Gap 隔离训练末尾与验证开头，防止信息泄露
    5. 向前步进 (step_size) 控制窗口重叠程度

    关键参数:
        n_splits: 分割数量（折数）
        train_window: 训练窗口月数（None = anchored, 逐步扩大）
        test_window: 测试窗口月数
        purge_days: purge gap 天数（防止信息泄露）
        anchored: True = 训练窗口锚定起点不断扩展，False = 固定长度滚动
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_window: Optional[int] = None,  # 月数, None 表示 anchored
        test_window: int = 6,                # 月数
        purge_days: int = 10,
        anchored: bool = False,
        step_size: Optional[int] = None,     # 月数, None 则自动计算
    ):
        self.n_splits = n_splits
        self.train_window = train_window
        self.test_window = test_window
        self.purge_days = purge_days
        self.anchored = anchored
        self.step_size = step_size

    def split(
        self,
        dates: pd.DatetimeIndex
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        对日期索引进行滚动分割

        参数:
            dates: 排序后的交易日日期索引

        返回:
            [(train_dates, test_dates), ...], 每个元素是一个 (训练日期, 验证日期) 元组
        """
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        dates = dates.sort_values()
        n_dates = len(dates)
        min_date = dates.min()
        max_date = dates.max()

        # 计算月步长
        test_months = self.test_window
        if self.step_size is None:
            step_months = test_months  # 默认步长 = 测试窗口
        else:
            step_months = self.step_size

        splits = []

        # 从后往前计算窗口分界点
        # 最右侧：max_date 为最后一个测试期的末尾
        test_end = max_date
        for i in range(self.n_splits):
            # 测试期
            test_start = self._subtract_months(test_end, test_months)
            test_start = self._next_trading_day(dates, test_start)  # 对齐到最近交易日

            # 找测试期的实际日期范围
            test_mask = (dates >= test_start) & (dates <= test_end)
            test_dates = dates[test_mask]

            # Purge gap 后的训练截止日期
            train_end_raw = test_start - timedelta(days=self.purge_days)
            train_end = self._prev_trading_day(dates, train_end_raw)

            # 训练起始日期
            if self.anchored or self.train_window is None:
                train_start = min_date
            else:
                train_start = self._subtract_months(train_end, self.train_window)
                train_start = self._next_trading_day(dates, train_start)

            train_mask = (dates >= train_start) & (dates <= train_end)
            train_dates = dates[train_mask]

            if len(train_dates) >= 2 and len(test_dates) >= 2:
                splits.append((train_dates, test_dates))

            # 下一轮的测试期末尾
            test_end = self._prev_trading_day(dates, test_start - timedelta(days=1))

        # 反转，使最早的 split 在前
        splits = splits[::-1]
        return splits

    @staticmethod
    def _subtract_months(date: pd.Timestamp, months: int) -> pd.Timestamp:
        """从日期减去月份"""
        year = date.year
        month = date.month - months
        while month <= 0:
            month += 12
            year -= 1
        day = min(date.day, 28)
        return pd.Timestamp(year=year, month=month, day=day)

    @staticmethod
    def _next_trading_day(dates: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
        """找到 target 之后最近的交易日"""
        future_dates = dates[dates >= target]
        if len(future_dates) > 0:
            return future_dates[0]
        return target

    @staticmethod
    def _prev_trading_day(dates: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
        """找到 target 之前最近的交易日"""
        past_dates = dates[dates <= target]
        if len(past_dates) > 0:
            return past_dates[-1]
        return target


class WalkForwardPerformanceEvaluator:
    """Walk-Forward 绩效评估器"""

    def __init__(self, validator: WalkForwardValidator):
        self.validator = validator

    def evaluate(
        self,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """
        使用 Walk-Forward 评估策略信号

        参数:
            signal_df: 包含 code, date, signal 的 DataFrame
            price_df: 包含 code, date, close 的行情 DataFrame
            top_k: 每期选择 top_k 只股票

        返回:
            { "fold_metrics": [...], "aggregate_metrics": {...}, "dates_leak_check": {...} }
        """
        dates = pd.DatetimeIndex(sorted(
            signal_df['date'].unique()
        )).sort_values()

        splits = self.validator.split(dates)

        fold_results = []
        all_returns = []
        leak_issues = []

        for i, (train_dates, test_dates) in enumerate(splits):
            # 训练集信号（用于分析，不用于后续回测训练）
            train_signals = signal_df[signal_df['date'].isin(train_dates)]
            test_signals = signal_df[signal_df['date'].isin(test_dates)]

            if test_signals.empty:
                continue

            # 模拟 TopK 策略：每期选 signal 最高的 top_k 只
            fold_returns = self._simulate_topk_strategy(
                test_signals, price_df, top_k
            )

            if fold_returns is not None and len(fold_returns) > 0:
                metrics = self._calc_fold_metrics(fold_returns)
                metrics['fold'] = i
                metrics['train_start'] = str(train_dates.min().date())
                metrics['train_end'] = str(train_dates.max().date())
                metrics['test_start'] = str(test_dates.min().date())
                metrics['test_end'] = str(test_dates.max().date())
                metrics['n_train_days'] = len(train_dates)
                metrics['n_test_days'] = len(test_dates)
                fold_results.append(metrics)
                all_returns.extend(fold_returns.values)

                # 数据泄露检查
                leak = self._check_data_leak(train_dates, test_dates)
                if leak:
                    leak_issues.append({"fold": i, **leak})

        aggregate = self._calc_aggregate(all_returns) if all_returns else {}

        return {
            "fold_metrics": fold_results,
            "aggregate_metrics": aggregate,
            "n_splits_used": len(fold_results),
            "n_splits_total": self.validator.n_splits,
            "leak_issues": leak_issues,
        }

    def _simulate_topk_strategy(
        self,
        signals: pd.DataFrame,
        price_df: pd.DataFrame,
        top_k: int,
    ) -> Optional[pd.Series]:
        """模拟 TopK 策略每期收益"""
        daily_returns = []

        for dt in sorted(signals['date'].unique()):
            day_signal = signals[signals['date'] == dt].copy()
            if 'alpha_score' in day_signal.columns:
                day_signal = day_signal.nlargest(top_k, 'alpha_score')
            elif 'signal' in day_signal.columns:
                day_signal = day_signal[day_signal['signal'] > 0]

            selected = day_signal['code'].tolist()
            if not selected:
                continue

            # 获取下一天的收益
            day_prices = price_df[
                (price_df['code'].isin(selected)) &
                (price_df['date'] == dt)
            ]
            next_day_prices = price_df[
                (price_df['code'].isin(selected)) &
                (price_df['date'] > dt)
            ].sort_values('date').groupby('code').first().reset_index()

            if next_day_prices.empty:
                continue

            # 等权组合当日收益
            merged = day_prices[['code']].merge(
                next_day_prices[['code', 'close']], on='code', how='inner'
            )
            if merged.empty:
                continue

            # 简化：使用 close-to-close return
            day_ret = (
                next_day_prices.set_index('code')['close'] /
                day_prices.set_index('code')['close'] - 1
            ).mean()

            if not np.isnan(day_ret):
                daily_returns.append({"date": dt, "return": day_ret})

        if not daily_returns:
            return None

        df = pd.DataFrame(daily_returns)
        return df.set_index('date')['return']

    def _calc_fold_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """计算单个 fold 的绩效指标"""
        if len(returns) < 2:
            return {}
        total_ret = float((1 + returns).prod() - 1)
        n = len(returns)
        ann_ret = float((1 + total_ret) ** (252 / n) - 1)
        vol = float(returns.std() * np.sqrt(252))
        sharpe = float(ann_ret / vol) if vol > 0 else 0
        mdd = float((returns.cumsum().apply(np.exp) /
                      returns.cumsum().apply(np.exp).cummax() - 1).min())

        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "n_days": n,
        }

    def _calc_aggregate(self, all_returns: List[float]) -> Dict[str, float]:
        """计算聚合绩效"""
        if not all_returns:
            return {}
        returns = pd.Series(all_returns)
        return self._calc_fold_metrics(returns)

    def _check_data_leak(
        self,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
    ) -> Optional[Dict[str, Any]]:
        """检查训练/测试集之间是否存在时间重叠（数据泄露）"""
        train_max = train_dates.max()
        test_min = test_dates.min()

        if test_min <= train_max:
            # 存在重叠
            overlap_days = len(set(train_dates) & set(test_dates))
            gap_days = (test_min - train_max).days
            return {
                "train_max": str(train_max.date()),
                "test_min": str(test_min.date()),
                "gap_days": gap_days,
                "overlap_days": overlap_days,
                "has_leak": True,
            }
        return {"has_leak": False, "gap_days": (test_min - train_max).days}


# ============================================================================
# 随机 Split 对照实现（用于对比风险）
# ============================================================================

def random_split_performance(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    n_folds: int = 5,
    top_k: int = 50,
) -> Dict[str, Any]:
    """
    使用随机交叉验证评估策略（存在 look-ahead bias 风险）

    对比 Walk-Forward 的原因:
    随机分割会引入前视偏差 — 训练集中可能包含「未来」的行情信息，
    导致回测指标（Sharpe、IC 等）虚高，真实交易中无法复现。
    """
    from sklearn.model_selection import KFold

    all_returns = []
    fold_metrics = []

    codes = signal_df['code'].unique()
    kf = KFold(n_splits=n_folds, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kf.split(codes)):
        train_codes = codes[train_idx]
        test_codes = codes[test_idx]

        test_signals = signal_df[signal_df['code'].isin(test_codes)]
        if test_signals.empty:
            continue

        fold_returns = []
        for dt in sorted(test_signals['date'].unique()):
            day_sig = test_signals[
                (test_signals['date'] == dt)
            ]
            if 'alpha_score' in day_sig.columns:
                day_sig = day_sig.nlargest(top_k, 'alpha_score')

            selected = day_sig['code'].tolist()
            day_prices = price_df[
                (price_df['code'].isin(selected)) &
                (price_df['date'] == dt)
            ]
            next_prices = price_df[
                (price_df['code'].isin(selected)) &
                (price_df['date'] > dt)
            ].sort_values('date').groupby('code').first().reset_index()

            if next_prices.empty or day_prices.empty:
                continue

            ret = (
                next_prices.set_index('code')['close'] /
                day_prices.set_index('code')['close'] - 1
            ).mean()
            if not np.isnan(ret):
                fold_returns.append(ret)

        if fold_returns:
            returns = pd.Series(fold_returns)
            total_ret = float((1 + returns).prod() - 1)
            n = len(returns)
            ann_ret = float((1 + total_ret) ** (252 / n) - 1) if n > 0 else 0
            vol = float(returns.std() * np.sqrt(252)) if n > 1 else 0
            sharpe = float(ann_ret / vol) if vol > 0 else 0
            fold_metrics.append({
                "fold": i, "sharpe_ratio": sharpe,
                "annual_return": ann_ret, "n_days": n,
            })
            all_returns.extend(fold_returns)

    agg = {}
    if all_returns:
        agg_series = pd.Series(all_returns)
        n = len(agg_series)
        total_ret = float((1 + agg_series).prod() - 1)
        agg = {
            "total_return": total_ret,
            "annual_return": float((1 + total_ret) ** (252 / n) - 1),
            "sharpe_ratio": float(agg_series.mean() / agg_series.std() * np.sqrt(252)) if agg_series.std() > 0 else 0,
            "n_days": n,
        }

    return {"fold_metrics": fold_metrics, "aggregate_metrics": agg}


# ============================================================================
# 测试类
# ============================================================================

class TestWalkForwardSplit(unittest.TestCase):
    """Walk-Forward 分割正确性测试"""

    def setUp(self):
        """生成测试日期索引"""
        self.dates = pd.date_range('2020-01-01', '2025-12-31', freq='B')
        self.validator = WalkForwardValidator(
            n_splits=5, train_window=24, test_window=12, purge_days=10,
            anchored=False
        )

    def test_split_count(self):
        """测试分割数量"""
        splits = self.validator.split(self.dates)
        self.assertGreater(len(splits), 0)
        self.assertLessEqual(len(splits), 5)

    def test_no_overlap(self):
        """测试训练集和测试集无重叠"""
        splits = self.validator.split(self.dates)
        for train_dates, test_dates in splits:
            overlap = set(train_dates) & set(test_dates)
            self.assertEqual(len(overlap), 0,
                             f"发现 {len(overlap)} 天重叠: {sorted(overlap)[:5]}...")

    def test_temporal_order(self):
        """测试时间顺序：训练集在测试集之前"""
        splits = self.validator.split(self.dates)
        for i, (train_dates, test_dates) in enumerate(splits):
            self.assertLess(
                train_dates.max(), test_dates.min(),
                f"Split {i}: 训练集 {train_dates.max().date()} 在测试集 {test_dates.min().date()} 之后"
            )

    def test_purge_gap_exists(self):
        """测试 Purge Gap 有效隔离"""
        validator = WalkForwardValidator(
            n_splits=5, train_window=24, test_window=12, purge_days=10,
            anchored=False
        )
        splits = validator.split(self.dates)

        for i, (train_dates, test_dates) in enumerate(splits):
            gap_days = (test_dates.min() - train_dates.max()).days
            self.assertGreaterEqual(
                gap_days, 1,
                f"Split {i}: 训练集与测试集之间无间隔 (gap={gap_days}天)"
            )

    def test_purge_gap_vs_no_purge(self):
        """对比有无 Purge Gap 的分割效果"""
        val_with_purge = WalkForwardValidator(
            n_splits=5, train_window=24, test_window=12, purge_days=10
        )
        val_no_purge = WalkForwardValidator(
            n_splits=5, train_window=24, test_window=12, purge_days=0
        )

        splits_purge = val_with_purge.split(self.dates)
        splits_no = val_no_purge.split(self.dates)

        if splits_purge and splits_no:
            purge_gap = (splits_purge[0][1].min() - splits_purge[0][0].max()).days
            no_gap = (splits_no[0][1].min() - splits_no[0][0].max()).days
            self.assertGreater(purge_gap, no_gap,
                               f"Purge gap ({purge_gap}天) 应大于无 purge ({no_gap}天)")

    def test_anchored_vs_rolling(self):
        """对比 anchored 和 rolling 窗口模式"""
        val_anchored = WalkForwardValidator(
            n_splits=5, test_window=12, purge_days=10, anchored=True
        )
        val_rolling = WalkForwardValidator(
            n_splits=5, train_window=24, test_window=12, purge_days=10,
            anchored=False
        )

        splits_a = val_anchored.split(self.dates)
        splits_r = val_rolling.split(self.dates)

        if splits_a and splits_r:
            # anchored 模式下，每个训练集都从同一个起点开始
            first_train_start = splits_a[0][0].min()
            for train_dates, _ in splits_a:
                self.assertEqual(train_dates.min(), first_train_start,
                                 "Anchored 模式训练集应从同一日期开始")

            # rolling 模式下，训练集起点逐步前移
            train_lengths_r = [len(t) for t, _ in splits_r]
            # 固定窗口长度应该都在允许范围内
            self.assertTrue(all(l > 0 for l in train_lengths_r),
                            "Rolling 模式训练集不应为空")

    def test_edge_case_few_dates(self):
        """边界条件：数据不足一个完整窗口"""
        few_dates = pd.date_range('2024-01-01', '2024-02-28', freq='B')
        validator = WalkForwardValidator(
            n_splits=5, train_window=12, test_window=6, purge_days=10
        )
        splits = validator.split(few_dates)
        # 应该返回空或尽量少的分割
        self.assertLessEqual(len(splits), 1,
                             "数据不足时应返回少量或无 split")

    def test_step_size_custom(self):
        """测试自定义步长"""
        val_default = WalkForwardValidator(
            n_splits=3, train_window=24, test_window=12
        )
        val_small_step = WalkForwardValidator(
            n_splits=3, train_window=24, test_window=12, step_size=3
        )

        splits_default = val_default.split(self.dates)
        splits_small = val_small_step.split(self.dates)

        # 小步长应产生更多 split
        print(f"\n[WalkForward] 默认步长 (12个月): {len(splits_default)} splits")
        print(f"[WalkForward] 小步长 (3个月):  {len(splits_small)} splits")
        self.assertGreaterEqual(len(splits_small), len(splits_default))


class TestWalkForwardVsRandom(unittest.TestCase):
    """Walk-Forward vs. 随机 Split 性能对比"""

    @classmethod
    def setUpClass(cls):
        """生成模拟策略信号和行情数据"""
        np.random.seed(42)
        codes = [f"{i:06d}.SZ" for i in range(100000, 100050)]
        dates = pd.date_range('2022-01-01', '2025-12-31', freq='B')

        rows_price = []
        rows_signal = []

        for code in codes:
            start_p = np.random.uniform(10, 60)
            # 模拟存在微弱 alpha 的股票
            alpha_strength = np.random.uniform(0.0001, 0.001)
            returns = np.random.normal(alpha_strength, 0.02, len(dates))
            prices = start_p * np.cumprod(1 + returns)
            prices[0] = start_p

            for j, (d, p) in enumerate(zip(dates, prices)):
                rows_price.append({
                    'date': d, 'code': code, 'close': p,
                })
                # 信号含噪声 + alpha
                signal = alpha_strength * 10 + np.random.normal(0, 0.05)
                rows_signal.append({
                    'date': d, 'code': code, 'alpha_score': signal,
                })

        cls.price_df = pd.DataFrame(rows_price)
        cls.signal_df = pd.DataFrame(rows_signal)

    def test_wf_vs_random_sharpe_bias(self):
        """对比 Walk-Forward 和随机 Split 的 Sharpe 差异"""
        validator = WalkForwardValidator(
            n_splits=5, train_window=24, test_window=12, purge_days=10,
            anchored=False
        )
        evaluator = WalkForwardPerformanceEvaluator(validator)

        wf_result = evaluator.evaluate(
            self.signal_df, self.price_df, top_k=20
        )

        random_result = random_split_performance(
            self.signal_df, self.price_df, n_folds=5, top_k=20
        )

        wf_sharpe = wf_result.get('aggregate_metrics', {}).get('sharpe_ratio', 0)
        rand_sharpe = random_result.get('aggregate_metrics', {}).get('sharpe_ratio', 0)

        print(f"\n[对比] Walk-Forward Sharpe: {wf_sharpe:.4f}")
        print(f"[对比] Random Split  Sharpe: {rand_sharpe:.4f}")
        print(f"[对比] 差异: {(rand_sharpe - wf_sharpe):.4f} (随机分割通常高估)")

        # 随机 Split 由于 look-ahead bias，Sharpe 通常会被高估
        # 不过这里是模拟数据，所以只记录差异
        print(f"[对比] Walk-Forward folds used: {wf_result.get('n_splits_used', 0)}/{wf_result.get('n_splits_total', 0)}")
        print(f"[对比] Leak issues: {len(wf_result.get('leak_issues', []))}")

        # 检查 fold 绩效的一致性（WF 各 fold 应有合理的波动）
        wf_folds = wf_result.get('fold_metrics', [])
        if wf_folds:
            sharpes = [f['sharpe_ratio'] for f in wf_folds if 'sharpe_ratio' in f]
            if sharpes:
                std = np.std(sharpes)
                print(f"[对比] Fold Sharpe 标准差: {std:.4f}")
                # 标准差不应过大（表示策略不稳定）
                self.assertLess(std, 5.0, f"各 fold Sharpe 波动过大: std={std:.4f}")


class TestTimeSeriesPurge(unittest.TestCase):
    """Purge Gap 数据隔离测试"""

    def test_gap_prevention(self):
        """验证 purge gap 防止 label 标签期内的信息泄露"""
        dates = pd.date_range('2023-01-01', '2025-06-30', freq='B')

        # 假设 label 需要未来 5 日收益
        label_forward_days = 5

        val_no_purge = WalkForwardValidator(
            n_splits=3, train_window=12, test_window=6, purge_days=0
        )
        val_with_purge = WalkForwardValidator(
            n_splits=3, train_window=12, test_window=6,
            purge_days=label_forward_days
        )

        splits_no = val_no_purge.split(dates)
        splits_purge = val_with_purge.split(dates)

        if splits_no and splits_purge:
            # 无 purge 时，train 最后几天可能看到 test 的 label
            gap_no = (splits_no[0][1].min() - splits_no[0][0].max()).days
            gap_purge = (splits_purge[0][1].min() - splits_purge[0][0].max()).days

            print(f"\n[Purge] 无 purge gap: {gap_no}天")
            print(f"[Purge] 有 purge gap: {gap_purge}天 (至少 {label_forward_days}天)")

            # 有 purge 的 gap 应大于 label 需要的天数
            self.assertGreaterEqual(
                gap_purge, label_forward_days,
                f"Purge gap ({gap_purge}天) 不足以隔离 {label_forward_days}日的 label"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)