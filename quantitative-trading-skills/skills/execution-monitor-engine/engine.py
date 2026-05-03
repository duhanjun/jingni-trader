import sys
import os
from typing import List, Optional, Dict, Any
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, get_config
from base import BaseTrader, Order, Position, OrderStatus, OrderType, OrderSide
from adapters import SimAdapter, xtquantAdapter, gmAdapter, vnpyAdapter
from circuit_breaker import CircuitBreaker
from audit_logger import AuditLogger
from position_manager import PositionManager
from trade_logger import TradeLogger


class ExecutionEngine:
    """
    交易执行监控主引擎
    """
    def __init__(self, config: Config):
        self.config = config
        self.paper_trade = config.PAPER_TRADE
        self.backend = config.EXECUTION_BACKEND
        
        self.trader: Optional[BaseTrader] = None
        self.circuit_breaker = CircuitBreaker(config)
        self.audit_logger = AuditLogger(config)
        self.position_manager = PositionManager(config)
        self.trade_logger = TradeLogger(config)
        
        self._initialize_trader()
    
    def _initialize_trader(self):
        if self.backend == "xtquant":
            self.trader = xtquantAdapter(self.config)
        elif self.backend == "gm":
            self.trader = gmAdapter(self.config)
        elif self.backend == "vnpy":
            self.trader = vnpyAdapter(self.config)
        else:
            self.trader = SimAdapter(self.config)
        
        self.trader.connect()
    
    def initialize_account(self, initial_capital: Optional[float] = None):
        if initial_capital is not None:
            self.config.INITIAL_CAPITAL = initial_capital
            self.circuit_breaker.initial_capital = initial_capital
        
        self.audit_logger.log_account_update({
            "initial_capital": self.config.INITIAL_CAPITAL,
            "mode": "paper_trade" if self.paper_trade else "live_trade"
        })
    
    def switch_paper_trade(self, enabled: bool):
        old_mode = "paper_trade" if self.paper_trade else "live_trade"
        self.paper_trade = enabled
        new_mode = "paper_trade" if self.paper_trade else "live_trade"
        
        self.audit_logger.log_switch_mode(old_mode, new_mode)
        
        if enabled and not isinstance(self.trader, SimAdapter):
            self.trader.disconnect()
            self.trader = SimAdapter(self.config)
            self.trader.connect()
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        side_enum = OrderSide(side)
        type_enum = OrderType(order_type)
        
        current_price = self.trader.get_current_price(symbol) or price or 10.0
        estimated_amount = current_price * quantity
        
        positions = self.get_positions()
        # 仅对买入订单检查持仓集中度
        check_position_concentration = (side_enum == OrderSide.BUY)
        
        if check_position_concentration:
            is_valid, reason = self.circuit_breaker.validate_order(
                estimated_amount, positions, symbol
            )
        else:
            # 卖出订单仅检查其他限制
            is_valid, reason = self.circuit_breaker.validate_order(
                estimated_amount, None, None
            )
        
        order_data = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "stop_price": stop_price
        }
        
        if not is_valid:
            temp_order = Order(
                order_id="REJECTED",
                symbol=symbol,
                side=side_enum,
                order_type=type_enum,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.REJECTED
            )
            self.audit_logger.log_order_rejected(temp_order, reason)
            self.circuit_breaker.record_order({"status": "rejected", **order_data})
            return temp_order
        
        self.audit_logger.log_risk_check_pass("all", order_data)
        
        order = self.trader.place_order(
            symbol=symbol,
            side=side_enum,
            order_type=type_enum,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            **kwargs
        )
        
        self.audit_logger.log_order_placed(order)
        self.circuit_breaker.record_order({"status": "placed", **order_data})
        
        if order.status == OrderStatus.FILLED:
            self.audit_logger.log_order_filled(order)
            self.trade_logger.log_trade(order, order.avg_fill_price, order.filled_quantity)
            self._sync_positions()
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        success = self.trader.cancel_order(order_id)
        if success:
            self.audit_logger.log_order_canceled(order_id, "user_requested")
        return success
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self.trader.get_order(order_id)
    
    def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> List[Order]:
        status_enum = OrderStatus(status) if status else None
        return self.trader.get_orders(symbol, status_enum)
    
    def get_positions(self) -> List[Position]:
        positions = self.trader.get_positions()
        self.position_manager.sync_positions(positions)
        return positions
    
    def get_account(self) -> Dict[str, Any]:
        account = self.trader.get_account()
        self.audit_logger.log_account_update(account)
        return account
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        return self.trader.get_current_price(symbol)
    
    def _sync_positions(self):
        positions = self.get_positions()
        for pos in positions:
            self.audit_logger.log_position_update(pos)
    
    def update_market_price(self, symbol: str, price: float):
        if hasattr(self.trader, 'update_price'):
            self.trader.update_price(symbol, price)
            self._sync_positions()
    
    def get_statistics(self) -> Dict[str, Any]:
        account = self.get_account()
        positions = self.get_positions()
        trade_stats = self.trade_logger.get_daily_statistics()
        
        return {
            "account": account,
            "position_count": len(positions),
            "total_market_value": self.position_manager.calculate_total_market_value(),
            "total_unrealized_pnl": self.position_manager.calculate_total_unrealized_pnl(),
            "daily_trades": trade_stats,
            "circuit_breaker_blocked": self.circuit_breaker.blocked,
            "paper_trade": self.paper_trade
        }
