import os
import json
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional


class TradeLogger:
    """
    交易日志记录
    """
    def __init__(self, config):
        self.config = config
        self.log_dir = config.TRADE_LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self.trades: List[Dict[str, Any]] = []
        self.current_trade_file = self._get_trade_file()
    
    def _get_trade_file(self) -> str:
        today = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"trades_{today}.csv")
    
    def log_trade(self, order: Any, fill_price: float, fill_quantity: int, realized_pnl: Optional[float] = None):
        trade_data = {
            "trade_id": f"TRD_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, 'value') else order.side,
            "order_type": order.order_type.value if hasattr(order.order_type, 'value') else order.order_type,
            "quantity": fill_quantity,
            "fill_price": fill_price,
            "fill_amount": fill_price * fill_quantity,
            "realized_pnl": realized_pnl,
            "timestamp": datetime.now().isoformat(),
            "commission": self._calculate_commission(fill_price * fill_quantity)
        }
        self.trades.append(trade_data)
        self._write_trade_to_file(trade_data)
    
    def _calculate_commission(self, amount: float) -> float:
        commission = amount * self.config.COMMISSION_RATE
        return max(commission, self.config.MIN_COMMISSION)
    
    def _write_trade_to_file(self, trade_data: Dict[str, Any]):
        trade_file = self._get_trade_file()
        file_exists = os.path.exists(trade_file)
        
        with open(trade_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trade_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_data)
    
    def get_trades_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        return [t for t in self.trades if t["symbol"] == symbol]
    
    def get_trades_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        return [t for t in self.trades if t["timestamp"].startswith(date_str)]
    
    def get_daily_statistics(self) -> Dict[str, Any]:
        today_trades = self.get_trades_by_date(datetime.now().strftime("%Y-%m-%d"))
        total_buy = sum(t["fill_amount"] for t in today_trades if t["side"] == "buy")
        total_sell = sum(t["fill_amount"] for t in today_trades if t["side"] == "sell")
        total_commission = sum(t["commission"] for t in today_trades)
        total_pnl = sum(t["realized_pnl"] or 0 for t in today_trades)
        
        return {
            "total_trades": len(today_trades),
            "total_buy_amount": total_buy,
            "total_sell_amount": total_sell,
            "total_commission": total_commission,
            "total_realized_pnl": total_pnl
        }
    
    def export_trades(self, filepath: str):
        if not self.trades:
            return
        
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.trades[0].keys())
            writer.writeheader()
            writer.writerows(self.trades)
