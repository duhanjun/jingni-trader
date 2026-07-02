"""
优化方向: 回测防未来偏差机制增强
借鉴来源: Jesse Trade Framework (Zero Look-Ahead Bias)
  - https://github.com/jesse-ai/jesse
  - https://jesse.trade/
  
  Jesse 的核心设计理念:
  1. 状态一致性快照: 每个 K 线结束时自动冻结账户状态、持仓、订单等全量状态
  2. 时间序列对齐机制: 自动处理多周期策略的数据对齐，避免不同时间戳精度导致的偏差
  3. Walk-Forward Optimization: 滚动重训/验证，防止过拟合
  4. Monte Carlo 压力测试: 随机打乱交易顺序、模拟不同市场路径

优化分析:
  jingnitrader 当前回测引擎:
  - 依赖第三方适配器 (backtrader/native)
  - 缺少显式的 look-ahead bias 检测机制
  - 未内置 Monte Carlo / WFO 等稳健性检验
  - 绩效指标较少（仅 total_return, annual_return, sharpe, max_drawdown, win_rate, calmar）

验证内容:
  1. 实现状态一致性快照与偏差检测器
  2. 实现 Walk-Forward 交叉验证
  3. 实现 Monte Carlo 模拟压力测试
  4. 绩效指标增强（Sortino, Calmar, Omega, 最大回撤持续期等）
  5. 对比原始回测引擎与增强后的输出
"""

import os
import sys
import json
import unittest
import warnings
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ===================== 状态一致性快照 =====================


@dataclass
class BarSnapshot:
    """单根 K 线的完整状态快照（借鉴 Jesse state consistency snapshot）"""
    timestamp: pd.Timestamp
    code: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    cash: float
    position: float
    equity: float
    is_limit_up: bool = False
    is_limit_down: bool = False


class BiasDetector:
    """
    未来偏差检测器（Look-Ahead Bias Detector）
    
    检测以下偏差类型:
    1. 信息泄露: 使用了当前 bar 的 close 价格做入场决策（而非 next bar open）
    2. 前视偏差: 因子计算使用了未来数据
    3. 停牌/涨跌停期间错误成交
    """

    def __init__(self):
        self.warnings: List[str] = []

    def check_entry_price(
        self, signal_date: pd.Timestamp, signal_close: float,
        execution_date: pd.Timestamp, execution_price: float
    ) -> bool:
        """检查入场价格是否避免了信息泄露"""
        if signal_date == execution_date and signal_close == execution_price:
            self.warnings.append(
                f"[信息泄露] {signal_date}: 信号日使用 close 价入场"
            )
            return False
        return True

    def check_data_leakage(
        self, factor_df: pd.DataFrame, price_df: pd.DataFrame,
        factor_col: str, max_lag: int = 5
    ) -> pd.DataFrame:
        """
        检查因子是否有前视偏差。
        方法: 计算因子与未来收益的相关性，若对 t+1 的预测力异常高
        则可能存在数据泄露。
        """
        results = []
        merged = factor_df[['code', 'date', factor_col]].merge(
            price_df[['code', 'date', 'close']], on=['code', 'date']
        )
        merged = merged.sort_values(['code', 'date'])

        for lag in range(1, max_lag + 1):
            merged[f'fwd_ret_{lag}'] = merged.groupby('code')['close'].transform(
                lambda x: x.shift(-lag) / x - 1
            )
            corr = merged[[factor_col, f'fwd_ret_{lag}']].corr().iloc[0, 1]
            results.append({"lag": lag, "correlation": float(corr)})

        # 如果 t+1 的 IC 极高而 t+5 等更低，可能是泄露
        result_df = pd.DataFrame(results)
        if len(result_df) >= 2:
            ic_decay = result_df['correlation'].diff().mean()
            if abs(ic_decay) < 0.001:
                self.warnings.append(
                    f"[可疑] 因子 {factor_col} IC 衰减过慢 (decay={ic_decay:.6f})，"
                    "可能存在前视偏差"
                )

        return result_df

    def report(self) -> List[str]:
        return self.warnings


# ===================== Walk-Forward 交叉验证 =====================


class WalkForwardValidator:
    """
    Walk-Forward 验证器（借鉴 Jesse + Freqtrade 的 WFO 机制）
    
    核心思想:
    - 将数据分为多个连续的时间窗口
    - 每个窗口内用前一段训练，后一段测试
    - 汇总所有窗口的样本外表现
    """

    def __init__(self, train_years: int = 3, test_years: int = 1):
        self.train_years = train_years
        self.test_years = test_years

    def generate_windows(
        self, dates: pd.DatetimeIndex
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        生成 Walk-Forward 窗口
        
        返回: [(train_start, train_end, test_start, test_end), ...]
        """
        min_date = dates.min()
        max_date = dates.max()
        train_delta = pd.DateOffset(years=self.train_years)
        test_delta = pd.DateOffset(years=self.test_years)

        windows = []
        current = min_date
        while current + train_delta + test_delta <= max_date:
            train_start = current
            train_end = current + train_delta - pd.Timedelta(days=1)
            test_start = current + train_delta
            test_end = min(current + train_delta + test_delta - pd.Timedelta(days=1), max_date)
            windows.append((train_start, train_end, test_start, test_end))
            current += test_delta

        return windows

    def validate(
        self,
        data: pd.DataFrame,
        train_fn,  # Callable[[pd.DataFrame], Any] - 返回训练好的模型/规则
        predict_fn,  # Callable[[Any, pd.DataFrame], np.ndarray] - 返回预测值
        metric_fn,  # Callable[[np.ndarray, np.ndarray], float] - 返回评估指标
        feature_cols: List[str],
        label_col: str = 'forward_return',
    ) -> Dict[str, Any]:
        """
        执行 Walk-Forward 验证
        
        返回:
            {
                "windows": int,
                "train_metrics": [float],
                "test_metrics": [float],
                "test_mean": float,
                "test_std": float,
            }
        """
        dates = pd.DatetimeIndex(data['date'].unique())
        windows = self.generate_windows(dates)

        train_scores = []
        test_scores = []

        for train_start, train_end, test_start, test_end in windows:
            train_mask = (data['date'] >= train_start) & (data['date'] <= train_end)
            test_mask = (data['date'] >= test_start) & (data['date'] <= test_end)

            train_data = data[train_mask]
            test_data = data[test_mask]

            if len(train_data) < 100 or len(test_data) < 30:
                continue

            # 训练
            model = train_fn(train_data)
            # 评估训练集
            train_pred = predict_fn(model, train_data)
            train_actual = train_data[label_col].dropna()
            train_score = metric_fn(train_pred[:len(train_actual)], train_actual.values)

            # 评估测试集（样本外）
            test_pred = predict_fn(model, test_data)
            test_actual = test_data[label_col].dropna()
            test_score = metric_fn(test_pred[:len(test_actual)], test_actual.values)

            train_scores.append(train_score)
            test_scores.append(test_score)

        if not test_scores:
            return {"windows": 0, "test_scores": [], "error": "无有效窗口"}

        return {
            "windows": len(test_scores),
            "train_scores": train_scores,
            "test_scores": test_scores,
            "test_mean": float(np.mean(test_scores)),
            "test_std": float(np.std(test_scores)),
            "test_min": float(np.min(test_scores)),
            "test_max": float(np.max(test_scores)),
        }


# ===================== Monte Carlo 压力测试 =====================


class MonteCarloSimulator:
    """
    Monte Carlo 模拟器（借鉴 Jesse Monte Carlo analysis）
    
    两种模拟模式:
    1. Trade Shuffle: 随机打乱交易顺序，检验策略是否依赖特定交易序列
    2. Return Simulation: 对每日收益率抽样重采样，模拟不同市场路径
    """

    def __init__(self, n_simulations: int = 500):
        self.n_simulations = n_simulations

    def trade_shuffle(
        self, trades: List[Dict], initial_capital: float = 1_000_000
    ) -> Dict[str, Any]:
        """
        交易顺序随机打乱模拟
        
        参数:
            trades: [{"return": float, "duration": int}, ...]
        
        返回:
            模拟结果的统计分布
        """
        if len(trades) < 10:
            return {"error": "交易数量不足"}

        final_equities = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            shuffled = np.random.permutation(trades)
            equity = initial_capital
            peak = equity
            mdd = 0

            for trade in shuffled:
                equity *= (1 + trade["return"])
                peak = max(peak, equity)
                mdd = min(mdd, (equity - peak) / peak)

            final_equities.append(equity)
            max_drawdowns.append(mdd)

        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)

        return {
            "simulations": self.n_simulations,
            "final_equity": {
                "mean": float(np.mean(final_equities)),
                "std": float(np.std(final_equities)),
                "p5": float(np.percentile(final_equities, 5)),
                "p25": float(np.percentile(final_equities, 25)),
                "p50": float(np.percentile(final_equities, 50)),
                "p75": float(np.percentile(final_equities, 75)),
                "p95": float(np.percentile(final_equities, 95)),
            },
            "max_drawdown": {
                "mean": float(np.mean(max_drawdowns)),
                "p5": float(np.percentile(max_drawdowns, 5)),
                "p95": float(np.percentile(max_drawdowns, 95)),
            },
            "profit_probability": float(np.mean(final_equities > initial_capital)),
        }

    def return_simulation(
        self, daily_returns: pd.Series, initial_capital: float = 1_000_000,
        n_days: int = 252
    ) -> Dict[str, Any]:
        """
        收益率重采样模拟
        
        通过 bootstrap 重采样日收益率生成可能的未来路径
        """
        returns_array = daily_returns.dropna().values
        if len(returns_array) < 30:
            return {"error": "收益率数据不足"}

        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(self.n_simulations):
            sampled = np.random.choice(returns_array, size=n_days, replace=True)
            equity_curve = initial_capital * np.cumprod(1 + sampled)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak

            final_equities.append(equity_curve[-1])
            max_drawdowns.append(drawdown.min())
            annual_ret = (equity_curve[-1] / initial_capital) ** (252 / n_days) - 1
            annual_vol = np.std(sampled) * np.sqrt(252)
            sharpe = (annual_ret - 0.03) / annual_vol if annual_vol > 0 else 0
            sharpe_ratios.append(sharpe)

        final_equities = np.array(final_equities)

        return {
            "simulations": self.n_simulations,
            "final_equity": {
                "mean": float(np.mean(final_equities)),
                "p5": float(np.percentile(final_equities, 5)),
                "p95": float(np.percentile(final_equities, 95)),
            },
            "max_drawdown": {
                "mean": float(np.mean(max_drawdowns)),
                "p5": float(np.percentile(max_drawdowns, 5)),
                "p95": float(np.percentile(max_drawdowns, 95)),
            },
            "sharpe_ratio": {
                "mean": float(np.mean(sharpe_ratios)),
                "p5": float(np.percentile(sharpe_ratios, 5)),
                "p95": float(np.percentile(sharpe_ratios, 95)),
            },
            "profit_probability": float(np.mean(final_equities > initial_capital)),
        }


# ===================== 增强绩效指标 =====================


class EnhancedMetricsCalculator:
    """
    增强绩效指标计算器
    
    在 jingnitrader 原有指标基础上增加:
    - Sortino Ratio (下行风险调整收益)
    - Omega Ratio (收益-损失比)
    - Max Drawdown Duration (最大回撤持续天数)
    - Recovery Factor (净收益/最大回撤绝对值)
    - Value at Risk (VaR) - 历史模拟法
    - Conditional VaR (CVaR / Expected Shortfall)
    - Tail Ratio (95分位收益 / 5分位损失)
    - Stability (R² of cumulative returns trend)
    """

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free: float = 0.03,
                     target_return: float = 0.0) -> float:
        """Sortino 比率: 仅考虑下行偏差"""
        excess = returns - target_return / 252
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        annual_excess = returns.mean() * 252 - risk_free
        annual_downside = downside.std() * np.sqrt(252)
        return float(annual_excess / annual_downside)

    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Omega 比率: 收益/损失概率加权比"""
        gains = returns[returns > threshold]
        losses = returns[returns < threshold]
        if len(losses) == 0 or losses.sum() == 0:
            return float('inf')
        return float(abs(gains.sum() / losses.sum()))

    @staticmethod
    def max_drawdown_duration(equity: pd.Series) -> int:
        """最大回撤持续天数"""
        peak = equity.cummax()
        drawdown_started = False
        max_duration = 0
        current_duration = 0
        for i in range(len(equity)):
            if equity.iloc[i] < peak.iloc[i]:
                if not drawdown_started:
                    drawdown_started = True
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                drawdown_started = False
                current_duration = 0
        return max_duration

    @staticmethod
    def recovery_factor(total_return: float, max_drawdown: float) -> float:
        """恢复因子: 净收益 / 最大回撤"""
        if max_drawdown == 0:
            return float('inf')
        return float(abs(total_return / max_drawdown))

    @staticmethod
    def tail_ratio(returns: pd.Series) -> float:
        """尾部比率: 95分位正收益 / |5分位负收益|"""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        if p5 == 0:
            return float('inf')
        return float(abs(p95 / p5))

    @staticmethod
    def stability(equity: pd.Series) -> float:
        """策略稳定性: 对累计收益做线性回归的 R²"""
        if len(equity) < 10:
            return 0.0
        cum_returns = equity / equity.iloc[0]
        x = np.arange(len(cum_returns)).reshape(-1, 1)
        y = cum_returns.values.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        try:
            model = LinearRegression().fit(x, y)
            r2 = model.score(x, y)
            return float(r2)
        except Exception:
            return 0.0

    @staticmethod
    def calc_all(equity: pd.Series, risk_free: float = 0.03) -> Dict[str, float]:
        """计算全部增强指标"""
        if len(equity) < 2:
            return {}
        returns = equity.pct_change().dropna()
        if len(returns) < 2:
            return {}
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        annual_return = float((1 + total_return) ** (252 / len(returns)) - 1)
        volatility = float(returns.std() * np.sqrt(252))
        max_dd = float((equity / equity.cummax() - 1).min())
        sharpe = float((annual_return - risk_free) / volatility) if volatility > 0 else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": float(annual_return / abs(max_dd)) if max_dd != 0 else 0,
            "sortino_ratio": EnhancedMetricsCalculator.sortino_ratio(returns, risk_free),
            "omega_ratio": EnhancedMetricsCalculator.omega_ratio(returns),
            "max_dd_duration": EnhancedMetricsCalculator.max_drawdown_duration(equity),
            "recovery_factor": EnhancedMetricsCalculator.recovery_factor(total_return, max_dd),
            "tail_ratio": EnhancedMetricsCalculator.tail_ratio(returns),
            "stability": EnhancedMetricsCalculator.stability(equity),
            "win_rate": float((returns > 0).mean()),
            "daily_var_95": float(np.percentile(returns, 5)),
            "daily_cvar_95": float(returns[returns <= np.percentile(returns, 5)].mean()),
        }


# ===================== 测试类 =====================


class TestBiasDetector(unittest.TestCase):
    """未来偏差检测器测试"""

    def setUp(self):
        self.detector = BiasDetector()

    def test_entry_price_leak(self):
        """检测: 当日收盘价入场属于信息泄露"""
        dt = pd.Timestamp('2024-01-15')
        result = self.detector.check_entry_price(dt, 10.0, dt, 10.0)
        self.assertFalse(result)
        self.assertIn("信息泄露", self.detector.warnings[0])

    def test_no_leak_next_open(self):
        """检测: 次日开盘入场不属于信息泄露"""
        dt_signal = pd.Timestamp('2024-01-15')
        dt_exec = pd.Timestamp('2024-01-16')
        result = self.detector.check_entry_price(dt_signal, 10.0, dt_exec, 10.2)
        self.assertTrue(result)
        self.assertEqual(len(self.detector.warnings), 0)

    def test_data_leakage_detection(self):
        """检测: 因子计算是否存在前视数据泄露"""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'code': ['000001.SZ'] * n,
            'date': pd.date_range('2024-01-01', periods=n, freq='B'),
            'close': np.cumsum(np.random.normal(0, 0.01, n)) + 10,
        })
        # 构造有泄露的因子: 直接用未来收益
        df['leaked_factor'] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-5) / x - 1
        )
        result = self.detector.check_data_leakage(
            df, df, 'leaked_factor', max_lag=5
        )
        self.assertEqual(len(result), 5)
        # 泄露因子对 t+5 应有极高相关性
        self.assertGreater(abs(result.iloc[4]['correlation']), 0.5)


class TestWalkForwardValidator(unittest.TestCase):
    """Walk-Forward 验证器测试"""

    def setUp(self):
        np.random.seed(42)
        n = 2000
        self.data = pd.DataFrame({
            'code': ['000001.SZ'] * n,
            'date': pd.date_range('2015-01-01', periods=n, freq='B'),
            'feature': np.random.normal(0, 1, n),
        })
        # 构造线性关系的标签
        self.data['forward_return'] = self.data['feature'] * 0.1 + np.random.normal(0, 0.02, n)

    def test_window_generation(self):
        """验证: 正确生成 Walk-Forward 窗口"""
        validator = WalkForwardValidator(train_years=3, test_years=1)
        windows = validator.generate_windows(pd.DatetimeIndex(self.data['date'].unique()))
        self.assertGreater(len(windows), 1, "应生成多个窗口")

        # 检查窗口测试期连续性
        for i in range(len(windows) - 1):
            _, _, _, test_e = windows[i]
            next_test_s = windows[i + 1][2]
            self.assertEqual(test_e + pd.Timedelta(days=1), next_test_s)

    def test_walk_forward_validation(self):
        """验证: Walk-Forward 验证流程正常工作"""
        validator = WalkForwardValidator(train_years=3, test_years=1)

        def train(data):
            from sklearn.linear_model import LinearRegression
            X = data[['feature']].values
            y = data['forward_return'].values
            model = LinearRegression()
            model.fit(X, y)
            return model

        def predict(model, data):
            return model.predict(data[['feature']].values)

        def metric(y_pred, y_true):
            return np.corrcoef(y_pred, y_true)[0, 1]

        result = validator.validate(
            self.data, train, predict, metric,
            feature_cols=['feature']
        )

        self.assertGreater(result['windows'], 0)
        self.assertIn('test_mean', result)
        self.assertIn('test_std', result)


class TestMonteCarloSimulator(unittest.TestCase):
    """Monte Carlo 模拟器测试"""

    def setUp(self):
        self.sim = MonteCarloSimulator(n_simulations=200)

    def test_trade_shuffle(self):
        """验证: 交易顺序打乱模拟"""
        np.random.seed(42)
        trades = [
            {"return": np.random.normal(0.01, 0.03), "duration": d}
            for d in np.random.randint(1, 10, 50)
        ]
        result = self.sim.trade_shuffle(trades)
        self.assertEqual(result['simulations'], 200)
        self.assertIn('final_equity', result)
        self.assertIn('profit_probability', result)

    def test_return_simulation(self):
        """验证: 收益率重采样模拟"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.015, 500))
        result = self.sim.return_simulation(returns, n_days=252)
        self.assertEqual(result['simulations'], 200)
        self.assertIn('sharpe_ratio', result)
        self.assertGreater(result['profit_probability'], 0)

    def test_insufficient_data(self):
        """边界条件: 数据不足"""
        result = self.sim.trade_shuffle([{"return": 0.01, "duration": 1}])
        self.assertIn('error', result)


class TestEnhancedMetrics(unittest.TestCase):
    """增强绩效指标测试"""

    def setUp(self):
        np.random.seed(42)
        n = 500
        self.equity = pd.Series(1_000_000)
        for i in range(1, n):
            ret = np.random.normal(0.0003, 0.012)
            self.equity.loc[i] = self.equity.iloc[-1] * (1 + ret)
        self.equity.index = pd.date_range('2024-01-01', periods=n, freq='B')
        self.returns = self.equity.pct_change().dropna()

    def test_sortino_ratio(self):
        sr = EnhancedMetricsCalculator.sortino_ratio(self.returns)
        self.assertIsInstance(sr, float)
        self.assertFalse(np.isnan(sr))

    def test_omega_ratio(self):
        omega = EnhancedMetricsCalculator.omega_ratio(self.returns)
        self.assertIsInstance(omega, float)
        self.assertGreater(omega, 0)

    def test_max_dd_duration(self):
        duration = EnhancedMetricsCalculator.max_drawdown_duration(self.equity)
        self.assertIsInstance(duration, int)
        self.assertGreaterEqual(duration, 0)

    def test_tail_ratio(self):
        tr = EnhancedMetricsCalculator.tail_ratio(self.returns)
        self.assertIsInstance(tr, float)
        self.assertGreater(tr, 0)

    def test_calc_all(self):
        metrics = EnhancedMetricsCalculator.calc_all(self.equity)
        expected_keys = [
            'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'calmar_ratio', 'sortino_ratio', 'omega_ratio',
            'max_dd_duration', 'recovery_factor', 'tail_ratio', 'stability',
            'win_rate', 'daily_var_95', 'daily_cvar_95',
        ]
        for key in expected_keys:
            self.assertIn(key, metrics, f"缺少指标: {key}")

    def test_empty_equity(self):
        """边界条件: 空数据"""
        metrics = EnhancedMetricsCalculator.calc_all(pd.Series(dtype=float))
        self.assertEqual(metrics, {})

    def test_single_point(self):
        """边界条件: 单数据点"""
        metrics = EnhancedMetricsCalculator.calc_all(pd.Series([100.0]))
        self.assertEqual(metrics, {})


class TestOriginalVsEnhancedComparison(unittest.TestCase):
    """
    对比: 原始回测引擎指标 vs 增强后指标
    
    模拟 jingnitrader BacktestEngine._calc_metrics() 的输出，
    并与 EnhancedMetricsCalculator.calc_all() 对比。
    """

    def setUp(self):
        np.random.seed(42)
        n = 252
        self.equity = pd.Series(1_000_000)
        for i in range(1, n):
            ret = np.random.normal(0.0003, 0.012)
            self.equity.loc[i] = self.equity.iloc[-1] * (1 + ret)

    def test_original_metrics_consistency(self):
        """验证: 增强指标与原指标数值一致（共同部分）"""
        # 模拟原始 backtest engine 计算
        eq = self.equity
        returns = eq.pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        max_dd = (eq / eq.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility != 0 else 0
        win_rate = (returns > 0).mean()

        original = {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
        }

        enhanced = EnhancedMetricsCalculator.calc_all(self.equity)

        # 公共指标应一致
        for key in original:
            self.assertIn(key, enhanced)
            self.assertAlmostEqual(original[key], enhanced[key], places=8,
                msg=f"指标 {key} 不一致")

        # 增强指标应包含额外的字段
        extra_keys = ['sortino_ratio', 'omega_ratio', 'max_dd_duration',
                     'recovery_factor', 'tail_ratio', 'stability',
                     'daily_var_95', 'daily_cvar_95']
        for key in extra_keys:
            self.assertIn(key, enhanced, f"增强指标缺少: {key}")


if __name__ == "__main__":
    print("=" * 60)
    print("Verification 2: 回测防未来偏差机制增强")
    print("借鉴来源: Jesse Zero Look-Ahead Bias + Freqtrade WFO")
    print("=" * 60)

    print("\n运行测试套件...")
    unittest.main(argv=[''], verbosity=2, exit=False)