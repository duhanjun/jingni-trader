import time
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from collections import deque


class CircuitBreaker:
    """
    硬断路器
    """
    def __init__(self, config):
        self.config = config
        self.order_history: deque = deque(maxlen=1000)
        self.daily_pnl: float = 0.0
        self.initial_capital: float = config.INITIAL_CAPITAL
        self.trading_day: date = date.today()
        self.blocked: bool = False
    
    def reset_daily(self):
        self.daily_pnl = 0.0
        self.trading_day = date.today()
        self.blocked = False
    
    def update_pnl(self, pnl_change: float):
        if date.today() != self.trading_day:
            self.reset_daily()
        self.daily_pnl += pnl_change
    
    def check_daily_loss_limit(self) -> bool:
        loss_rate = abs(self.daily_pnl) / self.initial_capital
        if loss_rate > self.config.MAX_DAILY_LOSS_RATE:
            self.blocked = True
            return False
        return True
    
    def check_order_amount(self, amount: float) -> bool:
        return amount <= self.config.MAX_SINGLE_ORDER_AMOUNT
    
    def check_order_frequency(self) -> bool:
        now = time.time()
        one_minute_ago = now - 60
        recent_orders = [o for o in self.order_history if o["timestamp"] > one_minute_ago]
        return len(recent_orders) < self.config.MAX_ORDER_FREQUENCY_PER_MINUTE
    
    def check_position_concentration(self, positions: List, new_order_symbol: str, new_order_value: float) -> bool:
        # 计算当前总市值
        current_total_value = sum(p.market_value or 0 for p in positions)
        
        # 如果没有持仓，使用初始资金作为参考
        if current_total_value == 0:
            # 检查新订单相对于初始资金的集中度
            concentration = new_order_value / self.initial_capital
            return concentration <= self.config.MAX_POSITION_CONCENTRATION
        
        # 有持仓的情况
        current_symbol_value = 0.0
        for p in positions:
            if p.symbol == new_order_symbol:
                current_symbol_value = p.market_value or 0
                break
        
        new_total_value = current_total_value + new_order_value
        new_concentration = (current_symbol_value + new_order_value) / new_total_value
        return new_concentration <= self.config.MAX_POSITION_CONCENTRATION
    
    def record_order(self, order_data: Dict[str, Any]):
        self.order_history.append({
            **order_data,
            "timestamp": time.time()
        })
    
    def validate_order(self, amount: float, positions: List = None, symbol: str = None) -> tuple[bool, Optional[str]]:
        if self.blocked:
            return False, "Trading blocked due to daily loss limit"
        
        if not self.check_daily_loss_limit():
            return False, f"Daily loss limit exceeded: {abs(self.daily_pnl):.2f}"
        
        if not self.check_order_amount(amount):
            return False, f"Single order amount limit: {self.config.MAX_SINGLE_ORDER_AMOUNT:.2f}"
        
        if not self.check_order_frequency():
            return False, f"Order frequency limit: {self.config.MAX_ORDER_FREQUENCY_PER_MINUTE} per minute"
        
        if positions is not None and symbol is not None:
            if not self.check_position_concentration(positions, symbol, amount):
                return False, f"Position concentration limit: {self.config.MAX_POSITION_CONCENTRATION:.1%}"
        
        return True, None
