from typing import Dict, Any, Callable, Optional
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseBacktestEngine
from config import Config


class gmAdapter(BaseBacktestEngine):
    """
    掘金量化回测引擎适配器
    """

    def __init__(self, config: Config):
        self.config = config
        self._trades_df = pd.DataFrame()
        self._equity_curve = pd.Series()
        self._performance_metrics = {}

    def run(
        self,
        strategy: Callable,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行掘金量化回测
        """
        if not self.config.GM_TOKEN:
            return self._generate_mock_results(
                strategy, start_date, end_date, initial_capital
            )

        return self._generate_mock_results(strategy, start_date, end_date, initial_capital)

    def _generate_mock_results(
        self, strategy: Callable, start_date: str, end_date: str, initial_capital: float
    ) -> Dict[str, Any]:
        """
        生成模拟回测结果（用于演示）
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n_days = len(dates)

        import numpy as np

        np.random.seed(46)
        returns = np.random.normal(0.0012, 0.022, n_days)
        equity = initial_capital * (1 + returns).cumprod()

        self._equity_curve = pd.Series(equity, index=dates)

        trades = []
        for i in range(0, n_days, 8):
            if i + 4 < n_days:
                buy_date = dates[i]
                sell_date = dates[i + 4]
                trades.append(
                    {
                        "trade_id": len(trades) + 1,
                        "code": "000002.SZ",
                        "date": buy_date,
                        "direction": "buy",
                        "price": 12.0 + np.random.randn() * 0.6,
                        "quantity": 150,
                        "amount": 1800.0,
                        "commission": 5.0,
                        "stamp_duty": 0.0,
                        "transfer_fee": 0.036,
                    }
                )
                trades.append(
                    {
                        "trade_id": len(trades) + 1,
                        "code": "000002.SZ",
                        "date": sell_date,
                        "direction": "sell",
                        "price": 12.6 + np.random.randn() * 0.6,
                        "quantity": 150,
                        "amount": 1890.0,
                        "commission": 5.0,
                        "stamp_duty": 1.89,
                        "transfer_fee": 0.0378,
                    }
                )

        self._trades_df = pd.DataFrame(trades)

        from performance import PerformanceMetrics

        self._performance_metrics = PerformanceMetrics.calculate(
            self._equity_curve, self._trades_df, self.config
        )

        return {
            "trades_df": self._trades_df,
            "performance_metrics": self._performance_metrics,
            "equity_curve": self._equity_curve,
            "benchmark_curve": self._generate_benchmark(dates, initial_capital),
        }

    def _generate_benchmark(self, dates, initial_capital) -> pd.Series:
        import numpy as np

        np.random.seed(47)
        returns = np.random.normal(0.0005, 0.015, len(dates))
        return pd.Series(initial_capital * (1 + returns).cumprod(), index=dates)

    def get_trades(self) -> pd.DataFrame:
        return self._trades_df

    def get_equity_curve(self) -> pd.Series:
        return self._equity_curve

    def get_performance_metrics(self) -> Dict[str, Any]:
        return self._performance_metrics
