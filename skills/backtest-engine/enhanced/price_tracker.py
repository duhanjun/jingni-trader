"""
价格追踪器 - last_known_price 机制
借鉴: quant-stream "When price data is missing, use last_known_price rather than zero"
"""
import numpy as np
from typing import Dict, Optional


class PriceTracker:
    """价格追踪器：处理停牌/缺失价格"""

    def __init__(self):
        self._last_known: Dict[str, float] = {}

    def update(self, code: str, price: float):
        if not np.isnan(price) and price > 0:
            self._last_known[code] = price

    def get_price(self, code: str, current_price: Optional[float] = None) -> float:
        if current_price is not None and not np.isnan(current_price) and current_price > 0:
            self.update(code, current_price)
            return current_price
        return self._last_known.get(code, np.nan)

    def get_all_prices(self) -> Dict[str, float]:
        return dict(self._last_known)