"""
回测引擎抽象基类
所有回测后端适配器必须实现此接口
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd


class BaseBacktestEngine(ABC):
    """回测引擎基类"""

    @abstractmethod
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
        """
        执行回测

        参数:
            data: 行情数据，含 code, date, open, high, low, close, volume, is_st,
                  is_limit_up, is_limit_down
            signals: 交易信号，含 code, date, signal (1买入, -1卖出, 0持仓)
                     或 target_weight (目标权重)
            init_capital: 初始资金
            benchmark: 基准指数代码
            commission_rate: 佣金费率
            stamp_tax_rate: 印花税率（卖出）
            t_plus_1: 是否启用 T+1
            price_limit: 是否启用涨跌停限制
            slippage: 滑点比例

        返回:
            {
                "trades": DataFrame,       # 成交记录
                "positions": DataFrame,    # 每日持仓
                "equity_curve": DataFrame, # 净值曲线 (date, equity, benchmark)
                "metrics": dict,           # 绩效指标
                "report_path": str         # 报告文件路径
            }
        """
        ...
