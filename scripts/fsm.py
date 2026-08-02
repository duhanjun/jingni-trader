"""P1-2 显式 FSM + IncidentFSM

两个状态机：
1. DailyFSM: 11 状态主流程状态机（PRD P1-2.1 ~ P1-2.7）
   - 显式转移白名单，非法跳转 raise
   - DEGRADED 非终态（可恢复）
   - MANUAL_ATTENTION 终态（需人工清理）
   - REPORT 终态（run_manifest 落盘后流程结束）

2. IncidentFSM: 7 状态故障自愈状态机（PRD P1-2.8 ~ P1-2.9）
   - 纯内存，不落库（持久化由 P2 state.db 负责）
   - 重试回路：RETRYING ↔ CLASSIFIED

环境变量：
- QUANT_FSM_STRICT_MODE: 严格模式（默认 true），false 时非法转移仅 warning
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set

logger = logging.getLogger("fsm")


# ============================================================================
# DailyFSM: 11 状态主流程状态机
# ============================================================================

# 状态枚举（PRD P1-2.1）
STATE_INITIALIZED = "INITIALIZED"
STATE_DATA = "DATA"
STATE_FACTOR = "FACTOR"
STATE_MODEL = "MODEL"
STATE_BACKTEST = "BACKTEST"
STATE_PORTFOLIO = "PORTFOLIO"
STATE_EXECUTION = "EXECUTION"
STATE_REPORT = "REPORT"
STATE_DEGRADED = "DEGRADED"
STATE_FAILED = "FAILED"
STATE_MANUAL_ATTENTION = "MANUAL_ATTENTION"

ALL_STATES: Set[str] = {
    STATE_INITIALIZED, STATE_DATA, STATE_FACTOR, STATE_MODEL,
    STATE_BACKTEST, STATE_PORTFOLIO, STATE_EXECUTION, STATE_REPORT,
    STATE_DEGRADED, STATE_FAILED, STATE_MANUAL_ATTENTION,
}

# 终态
TERMINAL_STATES: Set[str] = {STATE_REPORT, STATE_FAILED, STATE_MANUAL_ATTENTION}

# 非终态（可继续转移）
NON_TERMINAL_STATES: Set[str] = ALL_STATES - TERMINAL_STATES


def _build_allowed_transitions() -> Dict[str, List[str]]:
    """构建显式转移白名单（PRD P1-2.2）

    主路径允许前向跳转（如 FACTOR→BACKTEST 跳过 MODEL），
    因为 target_stages 可跳过中间阶段（backtest-engine 可直接用 factor 信号）。
    """
    # 主路径（有序）
    main_path = [
        STATE_INITIALIZED, STATE_DATA, STATE_FACTOR, STATE_MODEL,
        STATE_BACKTEST, STATE_PORTFOLIO, STATE_EXECUTION, STATE_REPORT,
    ]
    transitions: Dict[str, List[str]] = {}

    # 主路径：任一阶段可前向跳转到后续任意阶段
    for i in range(len(main_path) - 1):
        current = main_path[i]
        # 允许跳到 i+1, i+2, ..., 末尾
        for j in range(i + 1, len(main_path)):
            transitions.setdefault(current, []).append(main_path[j])

    # 任何非终态 → DEGRADED / FAILED / MANUAL_ATTENTION
    for state in NON_TERMINAL_STATES:
        transitions.setdefault(state, []).extend([
            STATE_DEGRADED, STATE_FAILED, STATE_MANUAL_ATTENTION,
        ])

    # DEGRADED（非终态）可恢复
    transitions[STATE_DEGRADED] = [
        STATE_REPORT,      # 跳过失败阶段直接报告
        STATE_FAILED,      # 无法恢复 → 失败
        STATE_DATA,        # 恢复到前置阶段重试
        STATE_FACTOR,
        STATE_MODEL,
        STATE_BACKTEST,
        STATE_MANUAL_ATTENTION,
    ]

    # 终态：MANUAL_ATTENTION / FAILED / REPORT 无出边
    transitions[STATE_MANUAL_ATTENTION] = []
    transitions[STATE_FAILED] = []
    transitions[STATE_REPORT] = []

    return transitions


_ALLOWED_TRANSITIONS: Dict[str, List[str]] = _build_allowed_transitions()


class DailyFSM:
    """主流程状态机（PRD P1-2.1 ~ P1-2.7）

    使用方式：
        fsm = DailyFSM()
        fsm.transition("INITIALIZED", "DATA")   # 合法
        fsm.transition("DATA", "BACKTEST")       # 非法 → raise
    """

    def __init__(self, strict: Optional[bool] = None):
        """构造函数。

        参数：
            strict: 严格模式。None 时读环境变量 QUANT_FSM_STRICT_MODE（默认 true）。
                    false 时非法转移仅 warning 不 raise。
        """
        if strict is None:
            strict = os.environ.get("QUANT_FSM_STRICT_MODE", "true").lower() in (
                "1", "true", "yes",
            )
        self.strict = strict
        self.current: str = STATE_INITIALIZED
        self.transitions_log: List[Dict[str, str]] = []

    def transition(self, current: str, target: str) -> str:
        """状态转移（PRD P1-2.3）。

        参数：
            current: 当前状态
            target: 目标状态

        返回：
            新状态（= target）

        异常：
            ValueError: 非法跳转（strict=True 时）
        """
        # 校验状态合法性
        if current not in ALL_STATES:
            raise ValueError(f"未知状态: {current}")
        if target not in ALL_STATES:
            raise ValueError(f"未知状态: {target}")

        # MANUAL_ATTENTION 终态：任何转移都 raise（P1-2.5）
        if current == STATE_MANUAL_ATTENTION:
            msg = f"MANUAL_ATTENTION 是终态，禁止转移: {current}→{target}"
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)
            return current

        # REPORT / FAILED 终态
        if current in TERMINAL_STATES and current != STATE_MANUAL_ATTENTION:
            msg = f"{current} 是终态，禁止转移: {current}→{target}"
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)
            return current

        # 检查白名单
        allowed = _ALLOWED_TRANSITIONS.get(current, [])
        if target not in allowed:
            msg = f"illegal transition: {current}→{target}"
            if self.strict:
                raise ValueError(msg)
            logger.warning(f"FSM 非法转移（非严格模式放行）: {msg}")
            self.current = target
            self.transitions_log.append({"from": current, "to": target})
            return target

        self.current = target
        self.transitions_log.append({"from": current, "to": target})
        logger.debug(f"FSM 转移: {current}→{target}")
        return target

    def is_terminal(self, state: Optional[str] = None) -> bool:
        """判断是否终态"""
        s = state or self.current
        return s in TERMINAL_STATES


# ============================================================================
# IncidentFSM: 7 状态故障自愈状态机（P1-2.8 ~ P1-2.9）
# ============================================================================

# Incident 状态
INCIDENT_DETECTED = "DETECTED"
INCIDENT_CLASSIFIED = "CLASSIFIED"
INCIDENT_RETRYING = "RETRYING"
INCIDENT_DEGRADED = "DEGRADED"
INCIDENT_MANUAL_ATTENTION = "MANUAL_ATTENTION"
INCIDENT_RESOLVED = "RESOLVED"
INCIDENT_FAILED = "FAILED"

INCIDENT_STATES: Set[str] = {
    INCIDENT_DETECTED, INCIDENT_CLASSIFIED, INCIDENT_RETRYING,
    INCIDENT_DEGRADED, INCIDENT_MANUAL_ATTENTION,
    INCIDENT_RESOLVED, INCIDENT_FAILED,
}

INCIDENT_TERMINAL: Set[str] = {
    INCIDENT_RESOLVED, INCIDENT_MANUAL_ATTENTION, INCIDENT_FAILED,
}

_INCIDENT_TRANSITIONS: Dict[str, List[str]] = {
    INCIDENT_DETECTED: [INCIDENT_CLASSIFIED],
    INCIDENT_CLASSIFIED: [
        INCIDENT_RETRYING,      # 可重试
        INCIDENT_DEGRADED,      # 降级
        INCIDENT_MANUAL_ATTENTION,  # 需人工
        INCIDENT_RESOLVED,      # 直接解决
        INCIDENT_FAILED,        # 无法解决
    ],
    INCIDENT_RETRYING: [
        INCIDENT_CLASSIFIED,    # 重试回路：回到分类判断
        INCIDENT_RESOLVED,      # 重试成功
        INCIDENT_DEGRADED,      # 重试失败 → 降级
        INCIDENT_FAILED,        # 重试失败 → 失败
    ],
    INCIDENT_DEGRADED: [
        INCIDENT_RESOLVED,
        INCIDENT_MANUAL_ATTENTION,
        INCIDENT_FAILED,
    ],
    INCIDENT_MANUAL_ATTENTION: [],
    INCIDENT_RESOLVED: [],
    INCIDENT_FAILED: [],
}


class IncidentFSM:
    """故障自愈状态机（PRD P1-2.8）。

    纯内存，不落库。重试回路：RETRYING ↔ CLASSIFIED。

    使用方式：
        incident = IncidentFSM()
        incident.transition("DETECTED", "CLASSIFIED")
        incident.transition("CLASSIFIED", "RETRYING")
        incident.transition("RETRYING", "CLASSIFIED")  # 重试回路
        incident.transition("CLASSIFIED", "RESOLVED")   # 最终解决
    """

    def __init__(self):
        self.current: str = INCIDENT_DETECTED
        self.retry_count: int = 0
        self.max_retries: int = 1  # P1-2.9: 重试 1 次
        self.history: List[Dict[str, str]] = []

    def transition(self, current: str, target: str) -> str:
        """状态转移"""
        if current not in INCIDENT_STATES:
            raise ValueError(f"未知 Incident 状态: {current}")
        if target not in INCIDENT_STATES:
            raise ValueError(f"未知 Incident 状态: {target}")

        # 终态禁止转移
        if current in INCIDENT_TERMINAL:
            raise ValueError(
                f"Incident 终态禁止转移: {current}→{target}"
            )

        allowed = _INCIDENT_TRANSITIONS.get(current, [])
        if target not in allowed:
            raise ValueError(
                f"illegal incident transition: {current}→{target}"
            )

        # 重试计数
        if current == INCIDENT_CLASSIFIED and target == INCIDENT_RETRYING:
            self.retry_count += 1
            if self.retry_count > self.max_retries:
                # 超过重试上限 → 强制降级
                logger.warning(
                    f"Incident 重试上限({self.max_retries})，转入 DEGRADED"
                )
                self.current = INCIDENT_DEGRADED
                self.history.append({"from": current, "to": INCIDENT_DEGRADED})
                return INCIDENT_DEGRADED

        self.current = target
        self.history.append({"from": current, "to": target})
        return target

    def is_terminal(self) -> bool:
        return self.current in INCIDENT_TERMINAL
