"""
Backtrader 回测引擎适配器
高度灵活的事件驱动框架，需自行实现A股规则
"""
import os
from typing import Dict, Any
import pandas as pd
import numpy as np

try:
    import backtrader as bt
    HAS_BT = True
except ImportError:
    HAS_BT = False

from ..base.base_backtest_engine import BaseBacktestEngine


class AShareData(bt.feeds.PandasData):
    """A股数据格式"""
    params = (
        ('datetime', 'date'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )


class SignalStrategy(bt.Strategy):
    """基于预定义信号的策略"""
    params = (
        ('signals', None),   # DataFrame with columns: date, code, signal
        ('tplus1', True),
        ('price_limit', True),
    )

    def __init__(self):
        self.orders = {}

    def next(self):
        dt = self.datas[0].datetime.date(0)
        # 检查信号（需按股票分别处理）
        # 此处为简化示例，实际需完整实现
        pass


class BacktraderAdapter(BaseBacktestEngine):
    """Backtrader 适配器"""

    def __init__(self):
        if not HAS_BT:
            raise ImportError("Backtrader 未安装，请 pip install backtrader")

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1e6,
        benchmark: str = "000300.SH",
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        slippage: float = 0.001,
    ) -> Dict[str, Any]:
        cerebro = bt.Cerebro()

        # 为每只股票加载数据
        # Backtrader 需要每只股票一个数据源，这在多股票时较繁琐
        # 这里简化处理单只股票示例
        if data.empty:
            return self._empty_result()

        # 设置初始资金
        cerebro.broker.setcash(init_capital)

        # 设置佣金（双边）
        cerebro.broker.setcommission(commission=commission_rate)

        # 添加滑点
        cerebro.broker.set_slippage_perc(slippage)

        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.03)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        # 添加策略
        cerebro.addstrategy(SignalStrategy, signals=signals, tplus1=t_plus_1, price_limit=price_limit)

        # 运行
        results = cerebro.run()
        strat = results[0]

        # 提取结果
        metrics = {
            'sharpe_ratio': strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
            'max_drawdown': strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
            'annual_return': strat.analyzers.returns.get_analysis().get('rnorm100', 0),
        }

        # 生成净值曲线（简化）
        equity = pd.DataFrame({
            'date': data['date'].unique(),
            'equity': [init_capital]  # 应替换为实际权益
        })

        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": equity,
            "metrics": metrics,
            "report_path": "",
        }

    def _empty_result(self):
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
