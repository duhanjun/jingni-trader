from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import Position


class PositionManager:
    """
    仓位管理
    """
    def __init__(self, config):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.last_sync_time: Optional[datetime] = None
    
    def sync_positions(self, trader_positions: List[Position]):
        self.positions = {p.symbol: p for p in trader_positions}
        self.last_sync_time = datetime.now()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        symbol = self._normalize_code(symbol)
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        return list(self.positions.values())
    
    def calculate_total_market_value(self) -> float:
        return sum(p.market_value or 0 for p in self.positions.values())
    
    def calculate_total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl or 0 for p in self.positions.values())
    
    def get_position_concentration(self, symbol: str) -> Optional[float]:
        total_value = self.calculate_total_market_value()
        if total_value == 0:
            return None
        position = self.get_position(symbol)
        if not position:
            return 0.0
        return (position.market_value or 0) / total_value
    
    def get_top_positions(self, n: int = 5) -> List[Position]:
        sorted_positions = sorted(
            self.positions.values(),
            key=lambda x: x.market_value or 0,
            reverse=True
        )
        return sorted_positions[:n]
    
    def _normalize_code(self, code: str) -> str:
        code = code.strip().upper()
        if "." in code:
            return code
        if code.startswith("6"):
            return f"{code}.SH"
        return f"{code}.SZ"
