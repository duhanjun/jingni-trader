import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class AuditAction(str, Enum):
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_CANCELED = "ORDER_CANCELED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_UPDATE = "POSITION_UPDATE"
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"
    RISK_CHECK_PASS = "RISK_CHECK_PASS"
    RISK_CHECK_FAIL = "RISK_CHECK_FAIL"
    SWITCH_MODE = "SWITCH_MODE"


class AuditLogger:
    """
    审计日志
    """
    def __init__(self, config):
        self.config = config
        self.log_dir = config.LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self.audit_logs: list = []
        self.current_log_file = self._get_log_file()
    
    def _get_log_file(self) -> str:
        today = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"audit_{today}.log")
    
    def _log(self, action: AuditAction, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data,
            "context": context or {}
        }
        self.audit_logs.append(log_entry)
        
        log_file = self._get_log_file()
        if log_file != self.current_log_file:
            self.current_log_file = log_file
        
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def log_order_placed(self, order: Any, context: Optional[Dict[str, Any]] = None):
        self._log(AuditAction.ORDER_PLACED, order.to_dict(), context)
    
    def log_order_canceled(self, order_id: str, reason: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        self._log(AuditAction.ORDER_CANCELED, {
            "order_id": order_id,
            "reason": reason
        }, context)
    
    def log_order_filled(self, order: Any, context: Optional[Dict[str, Any]] = None):
        self._log(AuditAction.ORDER_FILLED, order.to_dict(), context)
    
    def log_order_rejected(self, order: Any, reason: str, context: Optional[Dict[str, Any]] = None):
        self._log(AuditAction.ORDER_REJECTED, {
            **order.to_dict(),
            "reason": reason
        }, context)
    
    def log_position_update(self, position: Any, context: Optional[Dict[str, Any]] = None):
        self._log(AuditAction.POSITION_UPDATE, position.to_dict(), context)
    
    def log_account_update(self, account: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        self._log(AuditAction.ACCOUNT_UPDATE, account, context)
    
    def log_risk_check_pass(self, check_type: str, order_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        self._log(AuditAction.RISK_CHECK_PASS, {
            "check_type": check_type,
            "order_data": order_data
        }, context)
    
    def log_risk_check_fail(self, check_type: str, reason: str, order_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        self._log(AuditAction.RISK_CHECK_FAIL, {
            "check_type": check_type,
            "reason": reason,
            "order_data": order_data
        }, context)
    
    def log_switch_mode(self, from_mode: str, to_mode: str, context: Optional[Dict[str, Any]] = None):
        self._log(AuditAction.SWITCH_MODE, {
            "from_mode": from_mode,
            "to_mode": to_mode
        }, context)
    
    def get_recent_logs(self, limit: int = 100) -> list:
        return self.audit_logs[-limit:]
