"""
交易执行器抽象基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd


class BaseExecutor(ABC):
    """交易执行器基类"""

    @abstractmethod
    def query_account(self) -> Dict[str, Any]:
        """查询账户资产、可用资金、持仓"""
        ...

    @abstractmethod
    def send_order(
        self,
        code: str,
        side: str,
        volume: int,
        price: Optional[float] = None,
        order_type: str = "limit"
    ) -> Dict[str, Any]:
        """
        发送订单

        参数:
            code: 股票代码
            side: buy / sell
            volume: 数量（股）
            price: 价格（限价单时必需）
            order_type: limit / market

        返回:
            订单信息字典
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """撤单"""
        ...

    @abstractmethod
    def query_positions(self) -> pd.DataFrame:
        """查询当前持仓"""
        ...

    @abstractmethod
    def sync_positions(self, target_weights: Dict[str, float], prices: Dict[str, float]) -> List[Dict]:
        """
        同步目标仓位

        参数:
            target_weights: {code: weight}
            prices: {code: latest_price}

        返回:
            需要执行的订单列表
        """
        ...