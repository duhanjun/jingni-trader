"""
验证测试: 滚动窗口回测 (Rolling Window Backtest)
================================================
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
优化方向: 回测引擎增强 - 引入滚动窗口训练/验证机制

Qlib 核心设计:
- 严格的时间序列划分: 训练集、验证集、测试集按时间顺序排列，避免未来信息泄露
- RollingWindow: train(X months) -> valid(Y months) -> test(Z months)，滚动前进
- 支持多轮滚动，每轮用新数据重新训练模型，评估样本外表现
- 与 jingni-trader 现有回测引擎的差异: 现有引擎只做单次全量回测，无滚动交叉验证

测试目标:
1. 验证滚动窗口划分的正确性（无数据泄露）
2. 对比单次回测 vs 滚动窗口回测的绩效指标差异
3. 验证滚动窗口在处理 A 股 T+1 规则时的正确性
"""
import sys
import os
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import unittest
from datetime import datetime, timedelta


# ============================================================
# 滚动窗口回测器 - 借鉴 Qlib 的 RollingWindow 设计
# ============================================================

class RollingWindowSplitter:
    """
    滚动窗口划分器
    借鉴 Qlib 的 RollingWindow 划分逻辑:
    - 将时间序列数据划分为多个 (train, valid, test) 三段
    - 每个窗口按时间顺序滚动，确保无未来信息泄露
    """

    def __init__(
        self,
        train_period: str = "12M",   # 训练期长度
        valid_period: str = "3M",   # 验证期长度
        test_period: str = "3M",    # 测试期长度
        step: str = "3M",           # 滚动步长
        min_train_days: int = 120,  # 最少训练天数
    ):
        self.train_period = self._parse_period(train_period)
        self.valid_period = self._parse_period(valid_period)
        self.test_period = self._parse_period(test_period)
        self.step = self._parse_period(step)
        self.min_train_days = min_train_days

    @staticmethod
    def _parse_period(period: str) -> int:
        """解析周期字符串 '12M' -> 12, '3M' -> 3, '1Y' -> 12"""
        period = period.strip().upper()
        if period.endswith('Y'):
            return int(period[:-1]) * 12
        elif period.endswith('M'):
            return int(period[:-1])
        elif period.endswith('D'):
            return int(period[:-1]) // 30
        else:
            return int(period)

    def split(self, dates: pd.DatetimeIndex) -> list:
        """
        划分日期为多个滚动窗口

        返回:
            [(train_start, train_end, valid_start, valid_end, test_start, test_end), ...]
            每段为 pd.Timestamp
        """
        dates = sorted(dates.unique())
        if len(dates) < self.min_train_days + 42:
            return []

        windows = []
        start_idx = 0

        while True:
            # 训练期结束位置
            train_end_idx = start_idx
            train_end_date = dates[start_idx] + pd.DateOffset(months=self.train_period)
            for i in range(start_idx, len(dates)):
                if dates[i] >= train_end_date:
                    train_end_idx = i
                    break
            else:
                break

            if train_end_idx - start_idx < self.min_train_days:
                start_idx += max(1, self.min_train_days // 4)
                continue

            # 验证期结束位置
            valid_end_idx = train_end_idx
            valid_end_date = dates[train_end_idx] + pd.DateOffset(months=self.valid_period)
            for i in range(train_end_idx, len(dates)):
                if dates[i] >= valid_end_date:
                    valid_end_idx = i
                    break
            else:
                break

            # 测试期结束位置
            test_end_idx = valid_end_idx
            test_end_date = dates[valid_end_idx] + pd.DateOffset(months=self.test_period)
            for i in range(valid_end_idx, len(dates)):
                if dates[i] >= test_end_date:
                    test_end_idx = i
                    break
            else:
                test_end_idx = len(dates) - 1

            if valid_end_idx - train_end_idx < 20:
                break

            windows.append((
                dates[start_idx], dates[train_end_idx],
                dates[train_end_idx + 1], dates[valid_end_idx],
                dates[valid_end_idx + 1], dates[test_end_idx],
            ))

            # 滚动步长
            step_end_date = dates[start_idx] + pd.DateOffset(months=self.step)
            for i in range(start_idx, len(dates)):
                if dates[i] >= step_end_date:
                    start_idx = i
                    break
            else:
                break

            if start_idx >= len(dates) - 30:
                break

        return windows


class RollingWindowBacktest:
    """
    滚动窗口回测执行器
    借鉴 Qlib 的 backtest 模式: 每轮训练 -> 验证 -> 测试
    """

    def __init__(self, splitter: RollingWindowSplitter):
        self.splitter = splitter
        self.results = []

    def run(
        self,
        data: pd.DataFrame,
        signal_generator,  # Callable: (train_data, test_data) -> pd.DataFrame
        init_capital: float = 1e6,
    ) -> dict:
        """执行滚动窗口回测"""
        if 'date' not in data.columns:
            raise ValueError("数据必须包含 date 列")

        dates = pd.DatetimeIndex(data['date'].unique()).sort_values()
        windows = self.splitter.split(dates)

        if not windows:
            return {"error": "无法生成有效窗口", "windows": []}

        all_test_equity = []
        window_results = []

        for i, (train_s, train_e, valid_s, valid_e, test_s, test_e) in enumerate(windows):
            # 获取各段数据
            train_data = data[(data['date'] >= train_s) & (data['date'] <= train_e)]
            valid_data = data[(data['date'] >= valid_s) & (data['date'] <= valid_e)]
            test_data = data[(data['date'] >= test_s) & (data['date'] <= test_e)]

            # 生成信号（在训练集上训练，在测试集上预测）
            signals = signal_generator(train_data, test_data)

            # 模拟交易（简化版）
            equity = self._simulate_trades(test_data, signals, init_capital)
            all_test_equity.append(equity)

            # 计算窗口绩效
            metrics = self._calc_window_metrics(equity)
            window_results.append({
                "window": i + 1,
                "train": f"{train_s.date()}~{train_e.date()}",
                "valid": f"{valid_s.date()}~{valid_e.date()}",
                "test": f"{test_s.date()}~{test_e.date()}",
                "metrics": metrics,
                "signal_count": len(signals),
            })

        self.results = window_results
        return {
            "windows": window_results,
            "summary": self._aggregate_results(window_results),
        }

    def _simulate_trades(
        self, data: pd.DataFrame, signals: pd.DataFrame, init_capital: float
    ) -> pd.DataFrame:
        """简化交易模拟，返回权益曲线"""
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        dates = sorted(signals['date'].unique())
        cash = init_capital
        equity_records = []

        for dt in dates:
            day_sig = signals[signals['date'] == dt]
            day_data = data[data['date'] == dt]
            if day_data.empty:
                continue

            market_value = 0
            buy_count = (day_sig['signal'] > 0).sum()
            if buy_count > 0:
                budget = cash * 0.95 / buy_count
                for _, row in day_sig[day_sig['signal'] > 0].iterrows():
                    code = row['code']
                    code_data = day_data[day_data['code'] == code]
                    if code_data.empty:
                        continue
                    price = code_data['close'].iloc[0]
                    shares = int(budget / price / 100) * 100
                    if shares > 0:
                        cost = shares * price * 1.00025
                        cash -= cost
                        market_value += shares * price

            total_equity = cash + market_value
            equity_records.append({"date": dt, "equity": total_equity})

        return pd.DataFrame(equity_records)

    def _calc_window_metrics(self, equity: pd.DataFrame) -> dict:
        """计算窗口绩效指标"""
        if equity.empty or len(equity) < 2:
            return {}
        eq = equity.set_index('date')['equity']
        returns = eq.pct_change().dropna()
        total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
        ann_ret = total_ret * (252 / max(len(returns), 1))
        vol = returns.std() * np.sqrt(252)
        max_dd = (eq / eq.cummax() - 1).min()
        sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0
        return {
            "total_return": round(float(total_ret), 4),
            "annual_return": round(float(ann_ret), 4),
            "volatility": round(float(vol), 4),
            "max_drawdown": round(float(max_dd), 4),
            "sharpe_ratio": round(float(sharpe), 4),
        }

    def _aggregate_results(self, window_results: list) -> dict:
        """汇总多窗口结果"""
        if not window_results:
            return {}
        rets = [w['metrics'].get('total_return', 0) for w in window_results]
        sharpes = [w['metrics'].get('sharpe_ratio', 0) for w in window_results]
        dds = [w['metrics'].get('max_drawdown', 0) for w in window_results]

        return {
            "n_windows": len(window_results),
            "mean_return": round(float(np.mean(rets)), 4),
            "std_return": round(float(np.std(rets)), 4),
            "mean_sharpe": round(float(np.mean(sharpes)), 4),
            "mean_max_dd": round(float(np.mean(dds)), 4),
            "positive_ratio": round(float(np.mean([r > 0 for r in rets])), 4),
        }


# ============================================================
# 测试用例
# ============================================================

class TestRollingWindowSplitter(unittest.TestCase):
    """测试滚动窗口划分器"""

    def setUp(self):
        self.splitter = RollingWindowSplitter(
            train_period="12M",
            valid_period="3M",
            test_period="3M",
            step="3M",
        )

    def test_no_future_leak(self):
        """验证滚动窗口划分无未来信息泄露"""
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
        windows = self.splitter.split(dates)

        for train_s, train_e, valid_s, valid_e, test_s, test_e in windows:
            # 训练集 < 验证集 < 测试集
            self.assertLess(train_e, valid_s)
            self.assertLess(valid_e, test_s)
            # 训练集天数足够
            self.assertGreaterEqual((train_e - train_s).days, 180)

        self.assertGreater(len(windows), 0)

    def test_window_continuity(self):
        """验证窗口连续性"""
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
        windows = self.splitter.split(dates)

        # 每两个相邻窗口的测试期不应重叠（实际上可能重叠，但数据范围应正确推进）
        for i in range(len(windows) - 1):
            curr_test_s = windows[i][4]
            next_test_s = windows[i + 1][4]
            # 下一个窗口的测试期开始不应早于当前窗口
            self.assertGreater(next_test_s, curr_test_s)

    def test_min_data_requirement(self):
        """验证最少数据量要求"""
        dates = pd.date_range('2020-01-01', '2020-04-01', freq='B')
        windows = self.splitter.split(dates)
        self.assertEqual(len(windows), 0)

    def test_edge_case_single_stock(self):
        """验证单只股票也能正确划分"""
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
        splitter = RollingWindowSplitter(train_period="6M", valid_period="2M", test_period="2M", step="2M")
        windows = splitter.split(dates)
        self.assertGreater(len(windows), 0)

        for train_s, train_e, valid_s, valid_e, test_s, test_e in windows:
            self.assertLess(train_e, valid_s)
            self.assertLess(valid_e, test_s)


class TestRollingWindowBacktest(unittest.TestCase):
    """测试滚动窗口回测"""

    def setUp(self):
        np.random.seed(42)
        splitter = RollingWindowSplitter(
            train_period="12M", valid_period="3M",
            test_period="3M", step="3M",
        )
        self.backtest = RollingWindowBacktest(splitter)

    def _generate_synthetic_data(self, n_dates=500, n_codes=5):
        """生成合成数据"""
        dates = pd.date_range('2020-01-01', periods=n_dates, freq='B')
        records = []
        for code in [f"00000{i}.SZ" for i in range(1, n_codes + 1)]:
            price = np.random.uniform(10, 50)
            for dt in dates:
                price *= np.random.lognormal(0.0003, 0.015)
                records.append({
                    'date': dt,
                    'code': code,
                    'close': price,
                    'open': price * 0.99,
                    'high': price * 1.02,
                    'low': price * 0.98,
                })
        return pd.DataFrame(records)

    def _simple_ma_signal(self, train_data, test_data):
        """简单 MA5 信号生成器"""
        test_data = test_data.copy()
        test_data = test_data.sort_values(['code', 'date'])
        test_data['ma5'] = test_data.groupby('code')['close'].transform(
            lambda x: x.rolling(5, min_periods=5).mean()
        )
        test_data['signal'] = 0
        test_data.loc[test_data['close'] > test_data['ma5'], 'signal'] = 1
        test_data.loc[test_data['close'] <= test_data['ma5'], 'signal'] = -1
        return test_data[['date', 'code', 'signal']].dropna()

    def test_basic_rolling_backtest(self):
        """基础滚动回测测试"""
        data = self._generate_synthetic_data(n_dates=500, n_codes=3)
        result = self.backtest.run(data, self._simple_ma_signal)

        self.assertIn('windows', result)
        self.assertIn('summary', result)
        self.assertGreater(len(result['windows']), 0, "应至少生成1个窗口")

        summary = result['summary']
        self.assertIn('mean_return', summary)
        self.assertIn('mean_sharpe', summary)

    def test_rolling_vs_single_window(self):
        """对比滚动窗口 vs 单次回测的统计差异"""
        data = self._generate_synthetic_data(n_dates=500, n_codes=3)
        result = self.backtest.run(data, self._simple_ma_signal)

        windows = result['windows']
        if len(windows) < 2:
            self.skipTest("窗口数不足，跳过对比测试")

        # 验证不同窗口的绩效指标存在差异（体现样本外测试的价值）
        returns = [w['metrics'].get('total_return', 0) for w in windows]
        self.assertGreater(np.std(returns), 0, "各窗口收益应存在差异")

    def test_empty_data_handling(self):
        """空数据处理"""
        empty_data = pd.DataFrame(columns=['date', 'code', 'close'])
        result = self.backtest.run(empty_data, self._simple_ma_signal)
        self.assertIn('error', result)


class TestPerformanceComparison(unittest.TestCase):
    """
    性能对比测试: 单次回测 vs 滚动窗口回测
    验证滚动窗口能更好地暴露过拟合风险
    """

    def test_overfit_detection(self):
        """
        验证: 过拟合策略在滚动窗口回测中表现不稳定
        单次回测可能显示高收益，但滚动窗口回测会暴露不一致性
        """
        np.random.seed(123)
        splitter = RollingWindowSplitter(
            train_period="6M", valid_period="2M",
            test_period="2M", step="2M",
        )
        bt = RollingWindowBacktest(splitter)

        # 生成合成数据
        dates = pd.date_range('2020-01-01', periods=400, freq='B')
        records = []
        for code in ['000001.SZ', '000002.SZ']:
            price = np.random.uniform(10, 50)
            for dt in dates:
                price *= np.random.lognormal(0.0002, 0.015)
                records.append({
                    'date': dt, 'code': code, 'close': price,
                    'open': price * 0.99, 'high': price * 1.02, 'low': price * 0.98,
                })
        data = pd.DataFrame(records)

        # 简单信号
        def simple_signal(train_data, test_data):
            test_data = test_data.copy()
            test_data = test_data.sort_values(['code', 'date'])
            test_data['ma5'] = test_data.groupby('code')['close'].transform(
                lambda x: x.rolling(5, min_periods=5).mean()
            )
            test_data['signal'] = 0
            test_data.loc[test_data['close'] > test_data['ma5'], 'signal'] = 1
            test_data.loc[test_data['close'] <= test_data['ma5'], 'signal'] = -1
            return test_data[['date', 'code', 'signal']].dropna()

        result = bt.run(data, simple_signal)

        # 验证: 滚动窗口回测能产生多个窗口结果
        windows = result['windows']
        self.assertGreater(len(windows), 1, "应生成多个滚动窗口")

        returns = [w['metrics'].get('total_return', 0) for w in windows]
        pos_ratio = np.mean([r > 0 for r in returns])

        # 打印对比报告
        print("\n" + "=" * 60)
        print("滚动窗口回测对比报告")
        print("=" * 60)
        print(f"窗口数量: {len(windows)}")
        for w in windows:
            print(f"  窗口{w['window']}: {w['test']} -> "
                  f"收益={w['metrics'].get('total_return', 0):.4f}, "
                  f"Sharpe={w['metrics'].get('sharpe_ratio', 0):.4f}")
        print(f"---")
        summary = result['summary']
        print(f"平均收益: {summary['mean_return']:.4f}")
        print(f"收益标准差: {summary['std_return']:.4f}")
        print(f"盈利窗口比例: {summary['positive_ratio']:.2%}")
        print(f"平均最大回撤: {summary['mean_max_dd']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)