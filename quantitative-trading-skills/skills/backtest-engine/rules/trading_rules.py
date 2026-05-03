from typing import Dict, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class FeeCalculator:
    """
    A股交易费用计算器
    """

    def __init__(self, config: Config):
        self.config = config

    def calculate(
        self, amount: float, direction: str
    ) -> Dict[str, float]:
        """
        计算交易费用

        Args:
            amount: 交易金额
            direction: 买卖方向 (buy/sell)

        Returns:
            费用 dict
        """
        commission = max(amount * self.config.COMMISSION_RATE, self.config.COMMISSION_MIN)
        stamp_duty = amount * self.config.STAMP_DUTY_RATE if direction == "sell" else 0.0
        transfer_fee = amount * self.config.TRANSFER_FEE_RATE

        return {
            "commission": commission,
            "stamp_duty": stamp_duty,
            "transfer_fee": transfer_fee,
            "total_fee": commission + stamp_duty + transfer_fee,
        }


class AShareTradingRules:
    """
    A股交易规则模拟
    """

    def __init__(self, config: Config):
        self.config = config
        self.fee_calculator = FeeCalculator(config)
        self._holdings = {}

    def check_limit_up_down(
        self, code: str, price: float, pre_close: float, is_st: bool = False
    ) -> Tuple[bool, str, float]:
        """
        检查涨跌停限制

        Args:
            code: 股票代码
            price: 委托价格
            pre_close: 前收盘价
            is_st: 是否ST

        Returns:
            (是否可成交, 原因, 合理价格)
        """
        limit_rate = self.config.ST_LIMIT_RATE if is_st else self.config.NORMAL_LIMIT_RATE
        limit_up = pre_close * (1 + limit_rate)
        limit_down = pre_close * (1 - limit_rate)

        if price > limit_up:
            if self.config.LIMIT_UP_DOWN_MODEL == "strict":
                return False, "price exceeds limit up", limit_up
            else:
                return True, "queue at limit up", limit_up
        elif price < limit_down:
            if self.config.LIMIT_UP_DOWN_MODEL == "strict":
                return False, "price below limit down", limit_down
            else:
                return True, "queue at limit down", limit_down

        return True, "price within limits", price

    def check_t1(self, code: str, sell_date: datetime) -> bool:
        """
        检查T+1规则

        Args:
            code: 股票代码
            sell_date: 卖出日期

        Returns:
            是否可卖出
        """
        if not self.config.ENABLE_T1:
            return True

        if code not in self._holdings:
            return False

        buy_date = self._holdings[code]["buy_date"]
        return sell_date > buy_date

    def update_holding(self, code: str, quantity: int, date: datetime):
        """
        更新持仓

        Args:
            code: 股票代码
            quantity: 数量
            date: 日期
        """
        if quantity > 0:
            self._holdings[code] = {
                "quantity": quantity,
                "buy_date": date,
            }
        elif code in self._holdings:
            self._holdings[code]["quantity"] += quantity
            if self._holdings[code]["quantity"] <= 0:
                del self._holdings[code]

    def get_holding_quantity(self, code: str) -> int:
        """
        获取持仓数量

        Args:
            code: 股票代码

        Returns:
            持仓数量
        """
        return self._holdings.get(code, {}).get("quantity", 0)

    def calculate_fee(self, amount: float, direction: str) -> Dict[str, float]:
        """
        计算交易费用

        Args:
            amount: 交易金额
            direction: 买卖方向 (buy/sell)

        Returns:
            费用 dict
        """
        return self.fee_calculator.calculate(amount, direction)
