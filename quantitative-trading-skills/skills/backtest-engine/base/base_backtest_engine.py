from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import pandas as pd
from datetime import datetime


class BaseBacktestEngine(ABC):
    """
    回测引擎抽象基类
    """

    @abstractmethod
    def run(
        self,
        strategy: Callable,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行回测

        Args:
            strategy: 策略函数
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            initial_capital: 初始资金
            **kwargs: 其他参数

        Returns:
            回测结果，包含：
            - trades_df: 交易记录 DataFrame
            - performance_metrics: 绩效指标 dict
            - equity_curve: 资金曲线 Series
            - benchmark_curve: 基准曲线 Series
        """
        pass

    @abstractmethod
    def get_trades(self) -> pd.DataFrame:
        """
        获取交易记录

        Returns:
            交易记录 DataFrame，包含以下字段：
            - trade_id: 交易ID
            - code: 股票代码
            - date: 交易日期
            - direction: 买卖方向 (buy/sell)
            - price: 成交价格
            - quantity: 成交数量
            - amount: 成交金额
            - commission: 佣金
            - stamp_duty: 印花税
            - transfer_fee: 过户费
        """
        pass

    @abstractmethod
    def get_equity_curve(self) -> pd.Series:
        """
        获取资金曲线

        Returns:
            资金曲线 Series，index 为日期
        """
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取绩效指标

        Returns:
            绩效指标 dict
        """
        pass
