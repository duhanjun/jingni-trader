"""
迅投miniQMT (xtquant) 交易执行适配器
对接迅投交易终端，支持A股实盘及仿真交易
"""
import logging
from typing import Dict, List, Optional, Any
import pandas as pd

from ..base.base_executor import BaseExecutor
from ..config import XTP_ACCOUNT


class XtQuantExecutor(BaseExecutor):
    """迅投miniQMT交易执行适配器"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._connected = False
        self._xt_trader = None
        self._acc = None

    def connect(self, path: str = "", session_id: int = 0) -> bool:
        """
        连接miniQMT

        参数:
            path: miniQMT安装目录下的userdata_mini路径
            session_id: 会话ID，默认0
        """
        try:
            from xtquant import xtdata
            xtdata.connect()
        except ImportError:
            raise ImportError("xtquant 未安装，请先安装迅投miniQMT SDK")

        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
            self._xt_trader = XtQuantTrader(path, session_id)
            account_id = XTP_ACCOUNT or ""
            self._acc = StockAccount(account_id, "STOCK")
            callback = _XtQuantCallback(self._logger)
            self._xt_trader.register_callback(callback)
            self._xt_trader.start()
            connect_result = self._xt_trader.connect()
            if connect_result != 0:
                self._logger.error(f"连接交易终端失败，错误码: {connect_result}")
                return False
            subscribe_result = self._xt_trader.subscribe(self._acc)
            if subscribe_result != 0:
                self._logger.error(f"订阅账户失败，错误码: {subscribe_result}")
                return False
            self._connected = True
            self._logger.info("miniQMT交易终端连接成功")
            return True
        except Exception as e:
            self._logger.error(f"连接miniQMT失败: {e}")
            return False

    def _ensure_connected(self):
        if not self._connected:
            raise ConnectionError("miniQMT 未连接，请先调用 connect()")

    def query_account(self) -> Dict[str, Any]:
        """查询账户资产信息"""
        self._ensure_connected()
        try:
            asset = self._xt_trader.query_stock_asset(self._acc)
            if asset is None:
                return {}
            return {
                "total_assets": float(getattr(asset, "total_asset", 0)),
                "available_cash": float(getattr(asset, "available_cash", 0)),
                "market_value": float(getattr(asset, "market_value", 0)),
                "frozen_cash": float(getattr(asset, "frozen", 0)),
                "account_id": self._acc.account_id if self._acc else "",
            }
        except Exception as e:
            self._logger.error(f"查询账户失败: {e}")
            return {}

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
            code: 股票代码 (如 000001.SZ)
            side: buy / sell
            volume: 委托数量（股）
            price: 委托价格
            order_type: limit / market

        返回:
            订单信息
        """
        self._ensure_connected()
        try:
            from xtquant.xtconstant import STOCK_BUY, STOCK_SELL, FIX_PRICE, LATEST_PRICE

            xt_side = STOCK_BUY if side.lower() == "buy" else STOCK_SELL
            xt_price_type = LATEST_PRICE if order_type == "market" else FIX_PRICE

            order_price = price if price is not None else 0

            seq = self._xt_trader.order_stock(
                account=self._acc,
                stock_code=code,
                order_type=xt_side,
                order_volume=volume,
                price_type=xt_price_type,
                price=order_price,
                strategy_name="quant_agent",
                order_remark="agent auto order"
            )

            return {
                "order_id": str(seq),
                "code": code,
                "side": side,
                "volume": volume,
                "price": order_price,
                "status": "submitted",
            }
        except Exception as e:
            self._logger.error(f"发送订单失败 {code} {side}: {e}")
            return {"error": str(e), "code": code, "side": side}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """撤单"""
        self._ensure_connected()
        try:
            seq = int(order_id)
            result = self._xt_trader.cancel_order_stock(self._acc, seq)
            return {"order_id": order_id, "status": "cancelled" if result == 0 else "failed"}
        except Exception as e:
            self._logger.error(f"撤单失败 {order_id}: {e}")
            return {"order_id": order_id, "error": str(e)}

    def query_positions(self) -> pd.DataFrame:
        """查询当前持仓"""
        self._ensure_connected()
        try:
            positions = self._xt_trader.query_stock_positions(self._acc)
            if positions is None or len(positions) == 0:
                return pd.DataFrame()
            data = []
            for pos in positions:
                data.append({
                    "code": pos.stock_code,
                    "volume": pos.volume,
                    "available_volume": pos.can_use_volume,
                    "avg_cost": pos.avg_price,
                    "market_value": pos.market_value,
                    "profit_loss": pos.profit_amount,
                })
            return pd.DataFrame(data)
        except Exception as e:
            self._logger.error(f"查询持仓失败: {e}")
            return pd.DataFrame()

    def sync_positions(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float]
    ) -> List[Dict]:
        """
        同步目标仓位

        参数:
            target_weights: {code: weight}
            prices: {code: 最新价}

        返回:
            需要执行的订单列表
        """
        self._ensure_connected()
        orders = []
        try:
            account = self.query_account()
            total_assets = account.get("total_assets", 0)
            if total_assets <= 0:
                self._logger.warning("账户总资产为0，无法生成调仓订单")
                return orders

            current_positions = self.query_positions()
            if not current_positions.empty:
                current_holdings = dict(zip(
                    current_positions["code"],
                    current_positions["available_volume"]
                ))
            else:
                current_holdings = {}

            for code, target_weight in target_weights.items():
                target_value = total_assets * target_weight
                price = prices.get(code, 0)
                if price <= 0:
                    continue
                target_volume = int(target_value / price / 100) * 100
                current_volume = current_holdings.get(code, 0)

                diff = target_volume - current_volume
                if diff == 0:
                    continue
                side = "buy" if diff > 0 else "sell"
                volume = abs(diff)

                orders.append({
                    "code": code,
                    "side": side,
                    "volume": volume,
                    "price": price,
                    "target_weight": target_weight,
                })

            self._logger.info(f"生成 {len(orders)} 笔调仓订单")
            return orders

        except Exception as e:
            self._logger.error(f"同步仓位失败: {e}")
            return orders


class _XtQuantCallback:
    """miniQMT交易回调处理器"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def on_disconnected(self):
        self._logger.warning("miniQMT 连接断开")

    def on_stock_order(self, order):
        self._logger.info(
            f"订单回报: {order.stock_code} "
            f"状态={order.order_status} "
            f"成交={order.traded_volume}/{order.order_volume}"
        )

    def on_stock_trade(self, trade):
        self._logger.info(
            f"成交回报: {trade.stock_code} "
            f"价格={trade.traded_price} "
            f"量={trade.traded_volume}"
        )

    def on_order_error(self, order_error):
        self._logger.error(
            f"订单错误: {order_error.order_id} "
            f"错误={order_error.error_msg}"
        )

    def on_cancel_error(self, cancel_error):
        self._logger.error(f"撤单错误: {cancel_error}")

    def on_stock_asset(self, asset):
        pass

    def on_stock_position(self, position):
        pass