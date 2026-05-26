"""
掘金量化(GM)交易执行适配器
支持掘金终端实盘/仿真交易接口
"""
import logging
from typing import Dict, List, Optional, Any
import pandas as pd

from ..base.base_executor import BaseExecutor
from ..config import GM_TOKEN


class GMExecutor(BaseExecutor):
    """掘金量化交易执行适配器"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._connected = False
        self._gm = None
        self._account_id = None

    def _ensure_connected(self):
        """确保已连接到掘金终端"""
        if self._connected:
            return
        try:
            import gm.api as gm
            self._gm = gm
            token = GM_TOKEN
            if token:
                gm.set_token(token)
            gm.login()
            self._connected = True
            self._logger.info("掘金量化终端连接成功")
        except ImportError:
            raise ImportError("gm 未安装，请 pip install gm 安装掘金SDK")
        except Exception as e:
            self._logger.error(f"连接掘金终端失败: {e}")
            raise ConnectionError(f"无法连接掘金终端: {e}")

    def query_account(self) -> Dict[str, Any]:
        """查询账户资产信息"""
        self._ensure_connected()
        try:
            account = self._gm.get_fund()
            if account is None:
                return {}
            return {
                "total_assets": float(account.balance.get("nav", 0)),
                "available_cash": float(account.balance.get("available", 0)),
                "market_value": float(account.balance.get("market_value", 0)),
                "frozen_cash": float(account.balance.get("frozen", 0)),
                "total_return": float(account.balance.get("total_return", 0)),
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
            code: 股票代码 (如 SHSE.600000 / SZSE.000001)
            side: buy / sell
            volume: 委托数量（股）
            price: 委托价格
            order_type: limit / market

        返回:
            订单信息
        """
        self._ensure_connected()
        try:
            gm_side = 1 if side.lower() == "buy" else 2
            order_style = 1 if order_type == "market" else 2
            symbol = self._format_gm_code(code)

            if order_type == "market":
                order = self._gm.order_volume(
                    symbol=symbol,
                    volume=volume,
                    side=gm_side,
                    order_type=order_style,
                    position_effect=1
                )
            else:
                if price is None:
                    raise ValueError("限价单必须指定价格")
                order = self._gm.order_volume(
                    symbol=symbol,
                    volume=volume,
                    side=gm_side,
                    order_type=order_style,
                    position_effect=1,
                    price=price
                )

            return {
                "order_id": order.get("cl_ord_id", ""),
                "code": code,
                "side": side,
                "volume": volume,
                "price": price,
                "status": str(order.get("status", "unknown")),
                "message": str(order.get("ord_rej_reason_detail", "")),
            }
        except Exception as e:
            self._logger.error(f"发送订单失败 {code} {side}: {e}")
            return {"error": str(e), "code": code, "side": side}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """撤单"""
        self._ensure_connected()
        try:
            result = self._gm.order_cancel(cl_ord_id=order_id)
            return {"order_id": order_id, "status": "cancelled" if result else "failed"}
        except Exception as e:
            self._logger.error(f"撤单失败 {order_id}: {e}")
            return {"order_id": order_id, "error": str(e)}

    def query_positions(self) -> pd.DataFrame:
        """查询当前持仓"""
        self._ensure_connected()
        try:
            positions = self._gm.get_position()
            if positions is None or len(positions) == 0:
                return pd.DataFrame()
            return pd.DataFrame(positions)
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

        对比当前持仓与目标权重，生成调仓订单列表。

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
                    current_positions["symbol"],
                    current_positions["volume"]
                ))
            else:
                current_holdings = {}

            for code, target_weight in target_weights.items():
                gm_code = self._format_gm_code(code)
                target_value = total_assets * target_weight
                price = prices.get(code, 0)
                if price <= 0:
                    continue
                target_volume = int(target_value / price / 100) * 100
                current_volume = current_holdings.get(gm_code, 0)

                diff = target_volume - current_volume
                if diff == 0:
                    continue
                side = "buy" if diff > 0 else "sell"
                volume = abs(diff)

                order = {
                    "code": code,
                    "side": side,
                    "volume": volume,
                    "price": price,
                    "target_weight": target_weight,
                }
                orders.append(order)

            self._logger.info(f"生成 {len(orders)} 笔调仓订单")
            return orders

        except Exception as e:
            self._logger.error(f"同步仓位失败: {e}")
            return orders

    def _format_gm_code(self, code: str) -> str:
        """
        将通用代码格式转为掘金格式

        000001.SZ -> SZSE.000001
        600000.SH -> SHSE.600000
        """
        if "." in code:
            ticker, exchange = code.split(".")
            prefix = "SHSE" if exchange.upper() == "SH" else "SZSE"
            return f"{prefix}.{ticker}"
        return code