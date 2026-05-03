import uuid
import random
from datetime import datetime
from typing import List, Optional, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseTrader, Order, Position, OrderStatus, OrderType, OrderSide


class SimAdapter(BaseTrader):
    """
    模拟交易适配器
    """
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.cash = config.INITIAL_CAPITAL
        self.initial_capital = config.INITIAL_CAPITAL
        self.price_cache: Dict[str, float] = {}
        self.order_counter = 0
    
    def connect(self) -> bool:
        self.connected = True
        return True
    
    def disconnect(self) -> bool:
        self.connected = False
        return True
    
    def _generate_order_id(self) -> str:
        self.order_counter += 1
        return f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.order_counter:04d}"
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        base_price = 10.0 if symbol.startswith("0") else 20.0
        price = base_price * (1 + (random.random() - 0.5) * 0.1)
        self.price_cache[symbol] = round(price, 2)
        return self.price_cache[symbol]
    
    def _calculate_commission(self, amount: float) -> float:
        commission = amount * self.config.COMMISSION_RATE
        return max(commission, self.config.MIN_COMMISSION)
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return round(price * (1 + self.config.SLIPPAGE), 2)
        else:
            return round(price * (1 - self.config.SLIPPAGE), 2)
    
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
        symbol = self.normalize_code(symbol)
        order_id = self._generate_order_id()
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.SUBMITTED
        )
        
        self.orders[order_id] = order
        self._execute_order(order)
        return order
    
    def _execute_order(self, order: Order):
        current_price = self.get_current_price(order.symbol)
        if not current_price:
            order.status = OrderStatus.REJECTED
            order.update_time = datetime.now()
            return
        
        execute_price = None
        if order.order_type == OrderType.MARKET:
            execute_price = self._apply_slippage(current_price, order.side)
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                execute_price = order.price
            elif order.side == OrderSide.SELL and current_price >= order.price:
                execute_price = order.price
            else:
                order.status = OrderStatus.PENDING
                order.update_time = datetime.now()
                return
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                execute_price = self._apply_slippage(current_price, order.side)
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                execute_price = self._apply_slippage(current_price, order.side)
            else:
                order.status = OrderStatus.PENDING
                order.update_time = datetime.now()
                return
        
        if execute_price:
            amount = execute_price * order.quantity
            commission = self._calculate_commission(amount)
            total_amount = amount + commission
            
            if order.side == OrderSide.BUY:
                if self.cash < total_amount:
                    order.status = OrderStatus.REJECTED
                    order.update_time = datetime.now()
                    return
                self.cash -= total_amount
                self._update_position(order.symbol, order.quantity, execute_price)
            else:
                position = self.positions.get(order.symbol)
                if not position or position.quantity < order.quantity:
                    order.status = OrderStatus.REJECTED
                    order.update_time = datetime.now()
                    return
                self.cash += (amount - commission)
                self._update_position(order.symbol, -order.quantity, execute_price)
            
            order.filled_quantity = order.quantity
            order.avg_fill_price = execute_price
            order.status = OrderStatus.FILLED
            order.update_time = datetime.now()
    
    def _update_position(self, symbol: str, quantity_change: int, price: float):
        if symbol in self.positions:
            position = self.positions[symbol]
            new_quantity = position.quantity + quantity_change
            
            if new_quantity > 0:
                if quantity_change > 0:
                    total_cost = position.avg_cost * position.quantity + price * quantity_change
                    position.avg_cost = round(total_cost / new_quantity, 2)
                position.quantity = new_quantity
            else:
                del self.positions[symbol]
        elif quantity_change > 0:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity_change,
                avg_cost=price
            )
    
    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELED
                order.update_time = datetime.now()
                return True
        return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)
    
    def get_orders(self, symbol: Optional[str] = None, status: Optional[OrderStatus] = None) -> List[Order]:
        orders = list(self.orders.values())
        if symbol:
            symbol = self.normalize_code(symbol)
            orders = [o for o in orders if o.symbol == symbol]
        if status:
            orders = [o for o in orders if o.status == status]
        return orders
    
    def get_positions(self) -> List[Position]:
        positions = []
        for pos in self.positions.values():
            current_price = self.get_current_price(pos.symbol)
            if current_price:
                pos.current_price = current_price
                pos.market_value = current_price * pos.quantity
                pos.unrealized_pnl = (current_price - pos.avg_cost) * pos.quantity
            positions.append(pos)
        return positions
    
    def get_account(self) -> Dict[str, Any]:
        total_value = self.cash
        positions = self.get_positions()
        for pos in positions:
            if pos.market_value:
                total_value += pos.market_value
        
        return {
            "cash": self.cash,
            "total_value": total_value,
            "unrealized_pnl": total_value - self.initial_capital,
            "initial_capital": self.initial_capital
        }
    
    def update_price(self, symbol: str, price: float):
        self.price_cache[symbol] = round(price, 2)
        for order in self.orders.values():
            if order.symbol == symbol and order.status == OrderStatus.PENDING:
                self._execute_order(order)
