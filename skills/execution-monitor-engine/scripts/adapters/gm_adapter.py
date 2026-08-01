"""
掘金量化(GM)交易执行适配器
支持掘金终端实盘/仿真交易接口
"""
import logging
from typing import Dict, List, Optional, Any
import pandas as pd

from ..base.base_executor import BaseExecutor
from ..config import GM_TOKEN, GM_ACCOUNT_ID


class GMExecutor(BaseExecutor):
    """掘金量化交易执行适配器"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._connected = False
        self._available = False  # 是否可用(已连接)
        self._gm = None
        self._account_id = None

    @property
    def available(self) -> bool:
        """执行器是否可用(已连接)"""
        return self._available

    def _try_connect(self) -> bool:
        """尝试连接掘金终端,失败返回False不抛异常

        掘金SDK无login方法,连接流程: set_token + set_account_id
        """
        if self._available:
            return True
        try:
            import gm.api as gm
            self._gm = gm
            # 1. 设置 token
            token = GM_TOKEN
            if token:
                gm.set_token(token)
            else:
                self._logger.error("GM_TOKEN 未配置")
                self._available = False
                return False
            # 2. 设置账户ID(掘金SDK无login,通过set_account_id绑定账户)
            account_id = GM_ACCOUNT_ID
            if not account_id:
                self._logger.error("GM_ACCOUNT_ID 未配置(请在掘金终端获取)")
                self._available = False
                return False
            gm.set_account_id(account_id)
            self._account_id = account_id
            self._connected = True
            self._available = True
            self._logger.info("掘金量化终端连接成功")
            return True
        except ImportError:
            self._logger.error("gm 未安装")
            self._available = False
            return False
        except Exception as e:
            self._logger.error(f"连接掘金终端失败: {e}")
            self._available = False
            return False

    def _ensure_connected(self):
        """内部方法:确保已连接,未连接抛异常(保留向后兼容)"""
        if not self._available:
            raise ConnectionError(f"掘金终端未连接: {'gm未安装' if self._gm is None else '连接失败'}")

    def query_account(self) -> Dict[str, Any]:
        """查询账户资产信息

        掘金SDK用 get_cash() 查询资金(非get_fund)
        """
        if not self._available:
            return {}
        try:
            cash = self._gm.get_cash()
            if cash is None:
                return {}
            # 兼容对象属性与字典访问(字段名: nav/available/market_value/frozen)
            def _get(obj, key, default=0):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            return {
                "total_assets": float(_get(cash, "nav", 0)),
                "available_cash": float(_get(cash, "available", 0)),
                "market_value": float(_get(cash, "market_value", 0)),
                "frozen_cash": float(_get(cash, "frozen", 0)),
                "account_id": self._account_id or "",
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
            订单信息字典(含 success 字段)
        """
        if not self._available:
            return {"success": False, "error": "掘金终端未连接"}
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
                    return {"success": False, "error": "限价单必须指定价格",
                            "code": code, "side": side}
                order = self._gm.order_volume(
                    symbol=symbol,
                    volume=volume,
                    side=gm_side,
                    order_type=order_style,
                    position_effect=1,
                    price=price
                )

            return {
                "success": True,
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
            return {"success": False, "error": str(e), "code": code, "side": side}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """撤单

        掘金SDK order_cancel 接收 list[dict],每个含 cl_ord_id 和 account_id
        """
        if not self._available:
            return {"success": False, "error": "掘金终端未连接", "order_id": order_id}
        try:
            self._gm.order_cancel([{
                "cl_ord_id": order_id,
                "account_id": self._account_id or GM_ACCOUNT_ID,
            }])
            return {"success": True, "order_id": order_id, "status": "cancelled"}
        except Exception as e:
            self._logger.error(f"撤单失败 {order_id}: {e}")
            return {"success": False, "order_id": order_id, "error": str(e)}

    def query_positions(self) -> pd.DataFrame:
        """查询当前持仓"""
        if not self._available:
            return pd.DataFrame(columns=["code", "volume", "available_volume", "avg_cost", "market_value"])
        try:
            positions = self._gm.get_position()
            if positions is None or len(positions) == 0:
                return pd.DataFrame(columns=["code", "volume", "available_volume", "avg_cost", "market_value"])
            data = []
            for pos in positions:
                # 兼容对象属性访问与字典访问
                if isinstance(pos, dict):
                    data.append({
                        "code": pos.get("symbol", ""),
                        "volume": pos.get("volume", 0),
                        "available_volume": pos.get("available", 0),
                        "avg_cost": pos.get("vwap", 0),
                        "market_value": pos.get("market_value", 0),
                    })
                else:
                    data.append({
                        "code": getattr(pos, "symbol", str(pos)),
                        "volume": getattr(pos, "volume", 0),
                        "available_volume": getattr(pos, "available", 0),
                        "avg_cost": getattr(pos, "vwap", 0),
                        "market_value": getattr(pos, "market_value", 0),
                    })
            return pd.DataFrame(data)
        except Exception as e:
            self._logger.error(f"查询持仓失败: {e}")
            return pd.DataFrame(columns=["code", "volume", "available_volume", "avg_cost", "market_value"])

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
        if not self._available:
            self._logger.warning("掘金终端未连接,无法生成调仓订单")
            return []
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
                    "order_type": "limit",
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
