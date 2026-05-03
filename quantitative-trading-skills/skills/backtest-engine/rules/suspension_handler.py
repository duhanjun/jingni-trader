from typing import Dict, Any, List, Set
import pandas as pd
from datetime import datetime


class SuspensionHandler:
    """
    停牌处理器
    """

    def __init__(self):
        self._suspension_info: Dict[str, List[datetime]] = {}
        self._frozen_assets: Dict[str, Dict] = {}

    def set_suspension_dates(self, code: str, dates: List[datetime]):
        """
        设置停牌日期

        Args:
            code: 股票代码
            dates: 停牌日期列表
        """
        self._suspension_info[code] = dates

    def is_suspended(self, code: str, date: datetime) -> bool:
        """
        检查是否停牌

        Args:
            code: 股票代码
            date: 日期

        Returns:
            是否停牌
        """
        if code not in self._suspension_info:
            return False

        return date in self._suspension_info[code]

    def freeze_asset(self, code: str, quantity: int, freeze_date: datetime):
        """
        冻结资产

        Args:
            code: 股票代码
            quantity: 数量
            freeze_date: 冻结日期
        """
        self._frozen_assets[code] = {
            "quantity": quantity,
            "freeze_date": freeze_date,
            "frozen": True,
        }

    def unfreeze_asset(self, code: str, unfreeze_date: datetime):
        """
        解冻资产

        Args:
            code: 股票代码
            unfreeze_date: 解冻日期
        """
        if code in self._frozen_assets:
            self._frozen_assets[code]["frozen"] = False
            self._frozen_assets[code]["unfreeze_date"] = unfreeze_date

    def is_frozen(self, code: str) -> bool:
        """
        检查资产是否冻结

        Args:
            code: 股票代码

        Returns:
            是否冻结
        """
        if code not in self._frozen_assets:
            return False

        return self._frozen_assets[code]["frozen"]

    def get_frozen_quantity(self, code: str) -> int:
        """
        获取冻结数量

        Args:
            code: 股票代码

        Returns:
            冻结数量
        """
        if code not in self._frozen_assets:
            return 0

        return self._frozen_assets[code]["quantity"] if self._frozen_assets[code]["frozen"] else 0

    def can_trade(self, code: str, date: datetime) -> bool:
        """
        检查是否可以交易

        Args:
            code: 股票代码
            date: 日期

        Returns:
            是否可以交易
        """
        if self.is_suspended(code, date):
            return False

        if self.is_frozen(code):
            return False

        return True

    def load_suspension_data(self, df: pd.DataFrame):
        """
        加载停牌数据

        Args:
            df: 停牌数据 DataFrame，包含 code 和 date 字段
        """
        for _, row in df.iterrows():
            code = row["code"]
            date = row["date"]

            if code not in self._suspension_info:
                self._suspension_info[code] = []

            if date not in self._suspension_info[code]:
                self._suspension_info[code].append(date)
