from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    FAILED = "failed"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Order:
    """
    订单类
    """
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        status: OrderStatus = OrderStatus.PENDING,
        filled_quantity: int = 0,
        avg_fill_price: Optional[float] = None,
        create_time: Optional[datetime] = None,
        update_time: Optional[datetime] = None,
        **kwargs
    ):
        self.order_id = order_id
        self.symbol = symbol
        self.side = OrderSide(side) if isinstance(side, str) else side
        self.order_type = OrderType(order_type) if isinstance(order_type, str) else order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.status = OrderStatus(status) if isinstance(status, str) else status
        self.filled_quantity = filled_quantity
        self.avg_fill_price = avg_fill_price
        self.create_time = create_time or datetime.now()
        self.update_time = update_time or datetime.now()
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "create_time": self.create_time.isoformat() if self.create_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
            "extra": self.extra
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        return cls(
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=data["side"],
            order_type=data["order_type"],
            quantity=data["quantity"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            status=data.get("status", OrderStatus.PENDING),
            filled_quantity=data.get("filled_quantity", 0),
            avg_fill_price=data.get("avg_fill_price"),
            create_time=datetime.fromisoformat(data["create_time"]) if data.get("create_time") else None,
            update_time=datetime.fromisoformat(data["update_time"]) if data.get("update_time") else None,
            **data.get("extra", {})
        )


class Position:
    """
    持仓类
    """
    def __init__(
        self,
        symbol: str,
        quantity: int,
        avg_cost: float,
        current_price: Optional[float] = None,
        market_value: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        **kwargs
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.current_price = current_price
        self.market_value = market_value
        self.unrealized_pnl = unrealized_pnl
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "extra": self.extra
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        return cls(
            symbol=data["symbol"],
            quantity=data["quantity"],
            avg_cost=data["avg_cost"],
            current_price=data.get("current_price"),
            market_value=data.get("market_value"),
            unrealized_pnl=data.get("unrealized_pnl"),
            **data.get("extra", {})
        )


class BaseTrader(ABC):
    """
    交易抽象基类
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接交易接口
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开交易接口
        """
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        下单
        
        Args:
            symbol: 股票代码
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单需要）
            stop_price: 止损价格（止损单需要）
        
        Returns:
            Order 对象
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        撤单
        
        Args:
            order_id: 订单ID
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        获取订单
        
        Args:
            order_id: 订单ID
        
        Returns:
            Order 对象或 None
        """
        pass
    
    @abstractmethod
    def get_orders(self, symbol: Optional[str] = None, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        获取订单列表
        
        Args:
            symbol: 股票代码过滤
            status: 状态过滤
        
        Returns:
            订单列表
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        获取持仓
        
        Returns:
            持仓列表
        """
        pass
    
    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """
        获取账户信息
        
        Returns:
            账户信息字典
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        获取当前价格
        
        Args:
            symbol: 股票代码
        
        Returns:
            当前价格或 None
        """
        pass
    
    def normalize_code(self, code: str) -> str:
        """
        规范化股票代码格式
        """
        code = code.strip().upper()
        if "." in code:
            return code
        if code.startswith("6"):
            return f"{code}.SH"
        return f"{code}.SZ"
