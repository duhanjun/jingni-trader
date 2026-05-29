"""
实盘执行与监控引擎主逻辑
支持 Paper 模拟 / Live 实盘双模式，含硬风控断路器和审计日志
"""
import os
import sys
import json
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import uuid
from datetime import datetime

for key in list(sys.modules.keys()):
    if key.startswith('scripts.') or key == 'scripts':
        del sys.modules[key]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from scripts.config import (
    EXECUTION_DIR, TRADE_MODE, TRADE_BACKEND, INIT_CAPITAL,
    MAX_DAILY_LOSS_RATIO, MAX_SINGLE_ORDER_RATIO, MAX_SINGLE_STOCK_WEIGHT,
    MAX_ORDER_FREQUENCY, MIN_COMMISSION, COMMISSION_RATE, STAMP_TAX_RATE,
    SLIPPAGE, AUDIT_LOG_PATH, ACCOUNT_STATE_PATH
)

logger = logging.getLogger("execution-monitor-engine")


@dataclass
class Account:
    """虚拟账户（模拟模式）"""
    nav: float = INIT_CAPITAL
    available_cash: float = INIT_CAPITAL
    start_of_day_nav: float = INIT_CAPITAL
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    daily_pnl: float = 0.0

    def reset_daily(self):
        self.start_of_day_nav = self.nav
        self.daily_pnl = 0.0

    def get_current_nav(self, prices: Optional[Dict[str, float]] = None) -> float:
        total = self.available_cash
        if prices:
            for code, pos in self.positions.items():
                price = prices.get(code, pos.get("avg_cost", 0))
                total += pos["volume"] * price
        return total

    def apply_buy(self, code: str, price: float, volume: int, commission: float):
        cost = price * volume + commission
        if cost > self.available_cash:
            raise ValueError(f"资金不足: 需要 {cost}, 可用 {self.available_cash}")
        self.available_cash -= cost
        if code not in self.positions:
            self.positions[code] = {"volume": 0, "avg_cost": 0.0}
        old_cost = self.positions[code]["avg_cost"] * self.positions[code]["volume"]
        self.positions[code]["volume"] += volume
        self.positions[code]["avg_cost"] = (old_cost + cost) / self.positions[code]["volume"]

    def apply_sell(self, code: str, price: float, volume: int, commission: float, stamp_tax: float):
        if code not in self.positions or self.positions[code]["volume"] < volume:
            raise ValueError(f"持仓不足: 需要 {volume}")
        total_fee = commission + stamp_tax
        revenue = price * volume - total_fee
        self.available_cash += revenue
        self.positions[code]["volume"] -= volume
        if self.positions[code]["volume"] <= 0:
            del self.positions[code]

    def calc_commission(self, amount: float, is_sell: bool = False) -> float:
        commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
        stamp_tax = amount * STAMP_TAX_RATE if is_sell else 0
        return commission + stamp_tax

    def to_dict(self) -> Dict:
        return {
            "nav": self.nav,
            "available_cash": self.available_cash,
            "positions": self.positions,
            "daily_pnl": self.daily_pnl,
        }


class CircuitBreaker:
    """硬风控断路器"""

    def __init__(self):
        self.last_order_times: List[float] = []

    def check_send_order(
        self,
        account: Account,
        code: str,
        order_value: float,
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """检查是否可以发单"""
        current_nav = account.get_current_nav(prices)
        daily_return = (current_nav - account.start_of_day_nav) / account.start_of_day_nav if account.start_of_day_nav > 0 else 0

        checks = {
            "daily_loss": daily_return > -MAX_DAILY_LOSS_RATIO,
            "single_order_size": order_value <= current_nav * MAX_SINGLE_ORDER_RATIO,
            "frequency": self._check_frequency(),
        }

        if not checks["daily_loss"]:
            return {"allowed": False, "reason": f"单日亏损 {daily_return:.2%} 超过阈值 {MAX_DAILY_LOSS_RATIO:.2%}"}
        if not checks["single_order_size"]:
            return {"allowed": False, "reason": f"单笔金额 {order_value:.0f} 超过上限 {current_nav * MAX_SINGLE_ORDER_RATIO:.0f}"}
        if not checks["frequency"]:
            return {"allowed": False, "reason": f"订单频率超过每秒 {MAX_ORDER_FREQUENCY} 次限制"}

        self.last_order_times.append(time.time())
        return {"allowed": True, "reason": ""}

    def _check_frequency(self) -> bool:
        now = time.time()
        self.last_order_times = [t for t in self.last_order_times if now - t < 1.0]
        return len(self.last_order_times) < MAX_ORDER_FREQUENCY


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, log_path: str = AUDIT_LOG_PATH):
        self.log_path = log_path

    def log_order(
        self,
        order_id: str,
        code: str,
        side: str,
        volume: int,
        price: float,
        status: str,
        metadata: Optional[Dict] = None,
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "order_id": order_id,
            "code": code,
            "side": side,
            "volume": volume,
            "price": price,
            "status": status,
            "metadata": metadata or {},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_logs(self, n: int = 100) -> List[Dict]:
        """读取最近 n 条日志"""
        if not os.path.exists(self.log_path):
            return []
        logs = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line.strip()))
        return logs[-n:]


class PaperExecutor:
    """模拟交易执行器"""

    def __init__(self, init_capital: float = INIT_CAPITAL):
        self.account = Account(nav=init_capital, available_cash=init_capital)
        self.circuit_breaker = CircuitBreaker()
        self.audit = AuditLogger()
        self.orders: Dict[str, Dict] = {}

    def query_account(self) -> Dict[str, Any]:
        return self.account.to_dict()

    def send_order(
        self,
        code: str,
        side: str,
        volume: int,
        price: Optional[float] = None,
        order_type: str = "limit"
    ) -> Dict[str, Any]:
        order_value = (price or 0) * volume
        check = self.circuit_breaker.check_send_order(self.account, code, order_value)

        if not check["allowed"]:
            return {"success": False, "error": check["reason"]}

        order_id = str(uuid.uuid4())[:12]
        try:
            if side == "buy":
                commission = self.account.calc_commission(order_value, is_sell=False)
                self.account.apply_buy(code, price or 0, volume, commission)
            elif side == "sell":
                total_fee = self.account.calc_commission(order_value, is_sell=True)
                stamp_tax = order_value * STAMP_TAX_RATE
                commission = total_fee - stamp_tax
                self.account.apply_sell(code, price or 0, volume, commission, stamp_tax)

            self.orders[order_id] = {
                "order_id": order_id, "code": code, "side": side,
                "volume": volume, "price": price, "status": "filled",
                "timestamp": datetime.now().isoformat(),
            }
            self.audit.log_order(order_id, code, side, volume, price, "filled")
            return {"success": True, "order_id": order_id, "status": "filled"}

        except Exception as e:
            self.audit.log_order(order_id, code, side, volume, price, "rejected", {"error": str(e)})
            return {"success": False, "order_id": order_id, "status": "rejected", "error": str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"
            return {"success": True, "order_id": order_id}
        return {"success": False, "error": "订单不存在"}

    def query_positions(self) -> pd.DataFrame:
        rows = []
        for code, pos in self.account.positions.items():
            rows.append({"code": code, "volume": pos["volume"], "avg_cost": pos["avg_cost"]})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["code", "volume", "avg_cost"])

    def sync_positions(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[Dict]:
        """同步目标仓位，生成买卖订单列表"""
        nav = self.account.get_current_nav(prices)
        orders_to_execute = []

        for code, target_weight in target_weights.items():
            if code not in prices:
                continue
            price = prices[code]
            target_value = nav * target_weight
            current_volume = self.account.positions.get(code, {}).get("volume", 0)
            target_volume = int((target_value / price) // 100 * 100)

            if target_volume > current_volume:
                diff = target_volume - current_volume
                if diff >= 100:
                    orders_to_execute.append({
                        "code": code, "side": "buy", "volume": diff,
                        "price": price, "order_type": "limit",
                    })
            elif target_volume < current_volume:
                diff = current_volume - target_volume
                if diff >= 100:
                    orders_to_execute.append({
                        "code": code, "side": "sell", "volume": diff,
                        "price": price, "order_type": "limit",
                    })

        return orders_to_execute

    def save_state(self):
        """持久化账户状态"""
        state = {
            "nav": self.account.nav,
            "available_cash": self.account.available_cash,
            "positions": self.account.positions,
            "updated_at": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(ACCOUNT_STATE_PATH), exist_ok=True)
        with open(ACCOUNT_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self) -> bool:
        """加载持久化的账户状态"""
        if not os.path.exists(ACCOUNT_STATE_PATH):
            logger.info("无账户状态文件，使用初始资金")
            return False
        with open(ACCOUNT_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.account.nav = state.get("nav", INIT_CAPITAL)
        self.account.available_cash = state.get("available_cash", INIT_CAPITAL)
        self.account.positions = state.get("positions", {})
        logger.info(f"已加载账户状态: nav={self.account.nav}")
        return True


def run(ctx) -> Dict[str, Any]:
    """
    execution-monitor-engine 的 run 函数

    参数:
        ctx: Context 对象，需包含:
            - artifacts['PORTFOLIO']: 目标权重 JSON 路径
            - artifacts['DATA']: 行情数据

    返回:
        {
            "success": bool,
            "artifact_path": str,
            "metadata": {...},
            "error": str
        }
    """
    try:
        os.makedirs(EXECUTION_DIR, exist_ok=True)

        portfolio_path = ctx.get_artifact("PORTFOLIO")
        if not portfolio_path or not os.path.exists(portfolio_path):
            return {"success": False, "artifact_path": "", "metadata": {}, "error": "目标权重文件不存在"}

        with open(portfolio_path, "r", encoding="utf-8") as f:
            target_weights = json.load(f)

        data_path = ctx.get_artifact("DATA")
        prices = {}
        if data_path and os.path.exists(data_path):
            data = pd.read_parquet(data_path)
            latest = data[data['date'] == data['date'].max()]
            prices = dict(zip(latest['code'], latest['close']))
        else:
            for code in target_weights:
                prices[code] = 10.0
            logger.warning("无行情数据，使用默认价格10元")

        mode = TRADE_MODE
        if mode == "paper":
            executor = PaperExecutor()
            executor.load_state()
        else:
            return {
                "success": False, "artifact_path": "", "metadata": {},
                "error": f"实盘模式 ({TRADE_BACKEND}) 需配置券商接口，当前不可用"
            }

        executor.account.reset_daily()
        orders = executor.sync_positions(target_weights, prices)

        success_count = 0
        fail_count = 0
        for order in orders:
            result = executor.send_order(
                code=order["code"],
                side=order["side"],
                volume=order["volume"],
                price=order["price"],
                order_type=order.get("order_type", "limit"),
            )
            if result["success"]:
                success_count += 1
            else:
                fail_count += 1
                logger.warning(f"订单失败: {result}")

        executor.save_state()
        account_snapshot = executor.query_account()

        return {
            "success": True,
            "artifact_path": AUDIT_LOG_PATH,
            "metadata": {
                "orders_executed": success_count,
                "orders_failed": fail_count,
                "account_snapshot": account_snapshot,
                "mode": mode,
            },
            "error": ""
        }

    except Exception as e:
        logger.exception("执行引擎执行失败")
        return {"success": False, "artifact_path": "", "metadata": {}, "error": str(e)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            ctx_dict = json.load(f)
        from scripts.context import Context
        ctx = Context.from_dict(ctx_dict)
    else:
        from scripts.context import Context
        ctx = Context(
            task_id="test_execution",
            stock_pool=[],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        ctx.update_artifact("PORTFOLIO", "./workspace/portfolio/portfolio_weights.json")
        ctx.update_artifact("DATA", "./workspace/data/cleaned_data.parquet")

    result = run(ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
