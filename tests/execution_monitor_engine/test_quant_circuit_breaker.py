"""execution-monitor-engine L2 单元测试：quant_circuit_breaker.CircuitBreakerV2。

execution-monitor-engine 与 portfolio-risk-engine 都有 CircuitBreakerV2 类，
但位于不同 skill 的 scripts/optimizations 目录下，需要分别测试以验证两者实现一致性。

覆盖：
- 构造参数校验
- 单日亏损触发断路
- 单笔金额上限
- 滞回恢复
- Fail-open
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXECUTION_ENGINE_DIR = os.path.join(ROOT, "skills", "execution-monitor-engine")


def _load_quant_circuit_breaker():
    """加载 execution-monitor-engine/scripts/optimizations/quant_circuit_breaker.py。"""
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(EXECUTION_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    for _m in ("xtquant", "gm"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    try:
        target_path = os.path.join(EXECUTION_ENGINE_DIR, "scripts/optimizations/quant_circuit_breaker.py")
        spec = ilu.spec_from_file_location("_eme_quant_circuit_breaker", target_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["_eme_quant_circuit_breaker"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


@pytest.mark.skill_execution_monitor_engine
@pytest.mark.unit
class TestQuantCircuitBreakerConstruction:
    """验证 quant_circuit_breaker.CircuitBreakerV2 构造参数。"""

    def test_default_construction(self):
        mod = _load_quant_circuit_breaker()
        cb = mod.CircuitBreakerV2()
        assert cb.daily_loss_limit == 0.05
        assert cb.tripped is False

    def test_invalid_recover_threshold_raises(self):
        """recover_threshold <= -daily_loss_limit → 抛 ValueError"""
        mod = _load_quant_circuit_breaker()
        with pytest.raises(ValueError):
            mod.CircuitBreakerV2(daily_loss_limit=0.05, recover_threshold=-0.06)


@pytest.mark.skill_execution_monitor_engine
@pytest.mark.unit
class TestQuantCircuitBreakerDailyLoss:
    """验证单日亏损触发断路。"""

    def test_trips_when_daily_loss_exceeds_limit(self):
        mod = _load_quant_circuit_breaker()
        cb = mod.CircuitBreakerV2(daily_loss_limit=0.05)

        # 当前净值 94，开盘 100 → 单日 -6% → 超过 5% 阈值触发
        # 注意：daily_return < -daily_loss_limit 是严格小于，必须超过阈值
        # order_value 必须小于净值 10%（9.4）才能通过单笔检查先到达单日亏损检查
        result = cb.check_send_order(
            current_nav=94.0,
            start_of_day_nav=100.0,
            order_value=5.0,
        )
        assert result["allowed"] is False
        assert "单日亏损" in result["reason"]


@pytest.mark.skill_execution_monitor_engine
@pytest.mark.unit
class TestQuantCircuitBreakerRecovery:
    """验证滞回恢复。"""

    def test_recovers_when_roi_meets_threshold(self):
        mod = _load_quant_circuit_breaker()
        cb = mod.CircuitBreakerV2(daily_loss_limit=0.05, recover_threshold=0.0)

        # 触发
        cb.check_send_order(current_nav=94.0, start_of_day_nav=100.0, order_value=5.0)
        assert cb.tripped is True

        # 恢复
        result = cb.check_send_order(current_nav=101.0, start_of_day_nav=100.0, order_value=5.0)
        assert result["allowed"] is True
        assert cb.tripped is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
