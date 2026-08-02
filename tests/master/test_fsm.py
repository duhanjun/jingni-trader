"""P1-2 显式 FSM + IncidentFSM 测试

覆盖（PRD P1-2.10）：
- DailyFSM 合法/非法转移 × 主路径/分析路径/降级路径
- MANUAL_ATTENTION 终态
- DEGRADED 非终态（可恢复）
- IncidentFSM 重试回路
- IncidentFSM 重试上限
- 非严格模式（QUANT_FSM_STRICT_MODE=false）
"""
from __future__ import annotations

import pytest


# ============================================================================
# DailyFSM 合法转移
# ============================================================================

class TestDailyFSMLegalTransitions:
    def test_main_path_full(self):
        """主路径完整转移：INITIALIZED→DATA→...→REPORT"""
        from scripts.fsm import DailyFSM
        fsm = DailyFSM()
        path = ["DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"]
        current = "INITIALIZED"
        for stage in path:
            current = fsm.transition(current, stage)
        assert current == "REPORT"
        assert fsm.is_terminal()

    def test_analysis_path(self):
        """分析路径：INITIALIZED→DATA→FACTOR→REPORT"""
        from scripts.fsm import DailyFSM
        fsm = DailyFSM()
        current = fsm.transition("INITIALIZED", "DATA")
        current = fsm.transition(current, "FACTOR")
        current = fsm.transition(current, "REPORT")
        assert current == "REPORT"
        assert fsm.is_terminal()

    def test_degraded_recoverable(self):
        """DEGRADED 非终态，可恢复到 REPORT"""
        from scripts.fsm import DailyFSM
        fsm = DailyFSM()
        current = fsm.transition("INITIALIZED", "DATA")
        current = fsm.transition(current, "DEGRADED")
        assert not fsm.is_terminal()  # DEGRADED 非终态
        current = fsm.transition(current, "REPORT")
        assert current == "REPORT"

    def test_degraded_to_factor_retry(self):
        """DEGRADED → FACTOR 恢复重试"""
        from scripts.fsm import DailyFSM
        fsm = DailyFSM()
        current = fsm.transition("INITIALIZED", "DATA")
        current = fsm.transition(current, "FACTOR")
        current = fsm.transition(current, "DEGRADED")
        current = fsm.transition(current, "FACTOR")  # 回到 FACTOR 重试
        assert current == "FACTOR"


# ============================================================================
# DailyFSM 非法转移
# ============================================================================

class TestDailyFSMIllegalTransitions:
    def test_illegal_backward_jump(self):
        """非法后向跳转 BACKTEST→FACTOR raise"""
        from scripts.fsm import DailyFSM
        fsm = DailyFSM()
        current = fsm.transition("INITIALIZED", "DATA")
        current = fsm.transition(current, "FACTOR")
        current = fsm.transition(current, "BACKTEST")
        with pytest.raises(ValueError, match="illegal transition"):
            fsm.transition(current, "FACTOR")

    def test_illegal_factor_to_initialized(self):
        """非法后向跳转 FACTOR→INITIALIZED raise"""
        from scripts.fsm import DailyFSM
        fsm = DailyFSM()
        current = fsm.transition("INITIALIZED", "DATA")
        current = fsm.transition(current, "FACTOR")
        with pytest.raises(ValueError, match="illegal transition"):
            fsm.transition(current, "INITIALIZED")

    def test_manual_attention_terminal(self):
        """MANUAL_ATTENTION 终态：任何转移 raise"""
        from scripts.fsm import DailyFSM
        fsm = DailyFSM()
        current = fsm.transition("INITIALIZED", "DATA")
        current = fsm.transition(current, "MANUAL_ATTENTION")
        assert fsm.is_terminal()
        with pytest.raises(ValueError, match="终态"):
            fsm.transition(current, "DATA")

    def test_report_terminal(self):
        """REPORT 终态：任何转移 raise"""
        from scripts.fsm import DailyFSM
        fsm = DailyFSM()
        current = fsm.transition("INITIALIZED", "DATA")
        current = fsm.transition(current, "FACTOR")
        current = fsm.transition(current, "REPORT")
        with pytest.raises(ValueError, match="终态"):
            fsm.transition(current, "DATA")

    def test_failed_terminal(self):
        """FAILED 终态：任何转移 raise"""
        from scripts.fsm import DailyFSM
        fsm = DailyFSM()
        current = fsm.transition("INITIALIZED", "DATA")
        current = fsm.transition(current, "FAILED")
        with pytest.raises(ValueError, match="终态"):
            fsm.transition(current, "REPORT")


# ============================================================================
# 非严格模式
# ============================================================================

class TestFSMNonStrictMode:
    def test_non_strict_mode_warns_not_raises(self, monkeypatch):
        """QUANT_FSM_STRICT_MODE=false 时非法转移仅 warning"""
        from scripts.fsm import DailyFSM
        monkeypatch.setenv("QUANT_FSM_STRICT_MODE", "false")
        fsm = DailyFSM()
        current = fsm.transition("INITIALIZED", "DATA")
        # 非法转移不 raise，放行
        result = fsm.transition(current, "BACKTEST")
        assert result == "BACKTEST"


# ============================================================================
# IncidentFSM 重试回路
# ============================================================================

class TestIncidentFSM:
    def test_retry_loop(self):
        """重试回路：CLASSIFIED → RETRYING → CLASSIFIED"""
        from scripts.fsm import IncidentFSM
        incident = IncidentFSM()
        incident.transition("DETECTED", "CLASSIFIED")
        incident.transition("CLASSIFIED", "RETRYING")
        incident.transition("RETRYING", "CLASSIFIED")  # 回到分类
        incident.transition("CLASSIFIED", "RESOLVED")
        assert incident.is_terminal()

    def test_retry_limit_to_degraded(self):
        """重试上限 → 强制 DEGRADED"""
        from scripts.fsm import IncidentFSM
        incident = IncidentFSM()
        incident.transition("DETECTED", "CLASSIFIED")
        # 第一次重试
        result = incident.transition("CLASSIFIED", "RETRYING")
        assert result == "RETRYING"
        incident.transition("RETRYING", "CLASSIFIED")
        # 第二次重试 → 超上限 → DEGRADED
        result = incident.transition("CLASSIFIED", "RETRYING")
        assert result == "DEGRADED"

    def test_direct_resolved(self):
        """直接解决：DETECTED → CLASSIFIED → RESOLVED"""
        from scripts.fsm import IncidentFSM
        incident = IncidentFSM()
        incident.transition("DETECTED", "CLASSIFIED")
        incident.transition("CLASSIFIED", "RESOLVED")
        assert incident.is_terminal()

    def test_manual_attention_terminal(self):
        """Incident MANUAL_ATTENTION 终态"""
        from scripts.fsm import IncidentFSM
        incident = IncidentFSM()
        incident.transition("DETECTED", "CLASSIFIED")
        incident.transition("CLASSIFIED", "MANUAL_ATTENTION")
        assert incident.is_terminal()
        with pytest.raises(ValueError, match="终态"):
            incident.transition("MANUAL_ATTENTION", "RESOLVED")

    def test_illegal_incident_transition(self):
        """非法 Incident 转移 raise"""
        from scripts.fsm import IncidentFSM
        incident = IncidentFSM()
        with pytest.raises(ValueError, match="illegal"):
            incident.transition("DETECTED", "RESOLVED")  # DETECTED 只能到 CLASSIFIED


# ============================================================================
# 状态枚举完整性
# ============================================================================

class TestStateEnums:
    def test_all_states_count(self):
        """11 状态"""
        from scripts.fsm import ALL_STATES
        assert len(ALL_STATES) == 11

    def test_terminal_states(self):
        """3 终态：REPORT / FAILED / MANUAL_ATTENTION"""
        from scripts.fsm import TERMINAL_STATES
        assert TERMINAL_STATES == {"REPORT", "FAILED", "MANUAL_ATTENTION"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
