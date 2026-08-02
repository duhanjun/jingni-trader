"""P0-2 三态数据质量门 L2 单元测试

覆盖 DataQualityGate.check 的三态判定规则：
- normal：全部通过
- degraded：freshness > 5 天 或存在 PIT warning
- abort：任一 CORE 表缺失 或 freshness > 10 天

测试用例（PRD P0-2.7 要求 ≥ 7 个）：
1. normal 态 2 个用例
2. degraded 态 2 个用例
3. abort 态 2 个用例
4. freshness 边界（恰好 5/10 天）
5. PIT warning 触发 degraded
6. 别名映射（cleaned_data → daily, financial → fina_indicator）
7. 环境变量覆盖阈值
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest
import pandas as pd
import numpy as np


# ============================================================================
# 模块加载工具：把 data-engine/scripts 注册为 scripts 包，加载 quality_gate
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_DIR = os.path.join(ROOT, "skills", "data-engine")
SCRIPTS_DIR = os.path.join(DATA_ENGINE_DIR, "scripts")


def _load_quality_gate_module():
    """显式加载 data-engine/scripts/quality_gate.py 为独立模块。

    需要把 data-engine/scripts 注册为 sys.modules['scripts']，
    使 quality_gate.py 顶层的 `import pandas as pd` 等可解析。
    """
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    init_py = os.path.join(SCRIPTS_DIR, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[SCRIPTS_DIR],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    try:
        gate_path = os.path.join(SCRIPTS_DIR, "quality_gate.py")
        spec = ilu.spec_from_file_location("scripts.quality_gate", gate_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["scripts.quality_gate"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # 恢复 scripts 缓存
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


# ============================================================================
# 辅助：构造合成 DataFrame
# ============================================================================

def _make_daily_df(end_date: str = "20240920", n: int = 5) -> pd.DataFrame:
    """构造 daily 表，最新交易日为 end_date。

    end_date: YYYYMMDD 格式
    用 pd.date_range（日历日）确保长度恒定为 n，避免 bdate_range 在周末时的长度漂移
    """
    end = pd.Timestamp(end_date)
    dates = pd.date_range(end=end, periods=n)  # 日历日，包含周末
    return pd.DataFrame({
        "code": ["600000.SH"] * n,
        "date": dates,
        "open": np.linspace(10, 11, n),
        "high": np.linspace(11, 12, n),
        "low": np.linspace(9, 10, n),
        "close": np.linspace(10.5, 11.5, n),
        "volume": np.arange(100, 100 + n) * 1000,
    })


def _make_financial_df(n: int = 2) -> pd.DataFrame:
    """构造 fina_indicator 表"""
    return pd.DataFrame({
        "code": [f"60000{i}.SH" for i in range(n)],
        "report_date": ["20240930"] * n,
        "pe_ttm": [12.5, 8.3][:n],
        "roe": [15.0, 12.0][:n],
    })


# ============================================================================
# 单元测试
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestQualityGateNormal:
    """normal 态：全部通过"""

    def test_normal_when_all_core_present_and_freshness_within_5_days(self):
        """daily 表存在且 freshness ≤ 5 天 → normal"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        # daily 最新日 20240920，asof 20240925 → freshness 5 天（≤5，normal）
        df = _make_daily_df(end_date="20240920")
        verdict = gate.check(tables={"daily": df}, asof="20240925")
        assert verdict.mode == "normal"
        assert verdict.freshness_days == 5
        assert verdict.missing_core == []
        assert verdict.reason == "all checks passed"

    def test_normal_with_supplementary_optional_tables_present(self):
        """daily + financial + capital_flow 全部存在 → normal"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        df = _make_daily_df(end_date="20240920")
        fin_df = _make_financial_df()
        # capital_flow 也作为 optional 传入
        cf_df = pd.DataFrame({"code": ["600000.SH"], "net_inflow": [1e6]})
        verdict = gate.check(
            tables={"daily": df, "financial": fin_df, "capital_flow": cf_df},
            asof="20240925",
        )
        assert verdict.mode == "normal"
        assert verdict.missing_optional == []


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestQualityGateDegraded:
    """degraded 态：freshness > 5 天 或 PIT warning"""

    def test_degraded_when_freshness_between_6_and_10_days(self):
        """freshness 6 天（>5 且 ≤10）→ degraded"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        # daily 最新日 20240919，asof 20240925 → freshness 6 天
        df = _make_daily_df(end_date="20240919")
        verdict = gate.check(tables={"daily": df}, asof="20240925")
        assert verdict.mode == "degraded"
        assert verdict.freshness_days == 6
        assert "degraded 阈值" in verdict.reason

    def test_degraded_when_pit_warnings_present(self):
        """存在 PIT warning → degraded（即使 freshness 正常）"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        df = _make_daily_df(end_date="20240920")
        pit_warnings = [{"table": "financial", "code": "600000.SH", "disclosure_date": "20241201"}]
        verdict = gate.check(
            tables={"daily": df},
            asof="20240925",
            pit_warnings=pit_warnings,
        )
        assert verdict.mode == "degraded"
        assert verdict.pit_warnings == pit_warnings
        assert "PIT 违规" in verdict.reason


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestQualityGateAbort:
    """abort 态：CORE 表缺失 或 freshness > 10 天"""

    def test_abort_when_core_table_missing(self):
        """daily 表（CORE）缺失 → abort"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        # 只传 financial，不传 daily
        verdict = gate.check(
            tables={"financial": _make_financial_df()},
            asof="20240925",
        )
        assert verdict.mode == "abort"
        assert "daily" in verdict.missing_core
        assert "CORE 表缺失" in verdict.reason

    def test_abort_when_core_table_empty(self):
        """daily 表存在但为空 → abort"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        empty_df = pd.DataFrame()
        verdict = gate.check(tables={"daily": empty_df}, asof="20240925")
        assert verdict.mode == "abort"
        assert "daily" in verdict.missing_core

    def test_abort_when_freshness_exceeds_10_days(self):
        """freshness 11 天（>10）→ abort"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        # daily 最新日 20240914，asof 20240925 → freshness 11 天
        df = _make_daily_df(end_date="20240914")
        verdict = gate.check(tables={"daily": df}, asof="20240925")
        assert verdict.mode == "abort"
        assert verdict.freshness_days == 11
        assert "abort 阈值" in verdict.reason


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestQualityGateFreshnessBoundary:
    """freshness 边界测试"""

    def test_freshness_exactly_5_days_is_normal(self):
        """freshness 恰好 5 天 → normal（边界包含）"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        df = _make_daily_df(end_date="20240920")
        verdict = gate.check(tables={"daily": df}, asof="20240925")
        assert verdict.freshness_days == 5
        assert verdict.mode == "normal"

    def test_freshness_exactly_10_days_is_degraded(self):
        """freshness 恰好 10 天 → degraded（边界包含，>10 才 abort）"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        # daily 最新日 20240915，asof 20240925 → freshness 10 天
        df = _make_daily_df(end_date="20240915")
        verdict = gate.check(tables={"daily": df}, asof="20240925")
        assert verdict.freshness_days == 10
        assert verdict.mode == "degraded"


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestQualityGateAliasMapping:
    """别名映射测试（jingni-trader 产物 key → PRD 标准名）"""

    def test_cleaned_data_alias_maps_to_daily(self):
        """cleaned_data 别名应映射为 daily（CORE 校验通过）"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        df = _make_daily_df(end_date="20240920")
        # 用 cleaned_data 作为 key
        verdict = gate.check(tables={"cleaned_data": df}, asof="20240925")
        assert verdict.mode == "normal"
        assert verdict.missing_core == []

    def test_financial_alias_maps_to_fina_indicator(self):
        """financial 别名应映射为 fina_indicator"""
        mod = _load_quality_gate_module()
        # 显式声明 fina_indicator 为 core
        gate = mod.DataQualityGate(core_required=["daily", "fina_indicator"])
        df = _make_daily_df(end_date="20240920")
        fin_df = _make_financial_df()
        # 用 financial 作为 key
        verdict = gate.check(
            tables={"daily": df, "financial": fin_df},
            asof="20240925",
        )
        assert verdict.mode == "normal"
        assert verdict.missing_core == []

    def test_capital_flow_alias_maps_to_moneyflow_optional(self):
        """capital_flow 别名应映射为 moneyflow（OPTIONAL 表）"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        df = _make_daily_df(end_date="20240920")
        # capital_flow 为空 → 记入 missing_optional，但不影响 mode
        verdict = gate.check(
            tables={"daily": df, "capital_flow": pd.DataFrame()},
            asof="20240925",
        )
        assert verdict.mode == "normal"
        assert "moneyflow" in verdict.missing_optional


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestQualityGateEnvOverride:
    """环境变量覆盖阈值测试"""

    def test_env_override_freshness_abort_days(self, monkeypatch):
        """QUANT_QUALITY_GATE_FRESHNESS_ABORT_DAYS=3 → freshness 5 天触发 abort"""
        monkeypatch.setenv("QUANT_QUALITY_GATE_FRESHNESS_ABORT_DAYS", "3")
        monkeypatch.setenv("QUANT_QUALITY_GATE_FRESHNESS_DEGRADED_DAYS", "1")
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        df = _make_daily_df(end_date="20240920")
        # freshness 5 天 > abort 阈值 3 天 → abort
        verdict = gate.check(tables={"daily": df}, asof="20240925")
        assert verdict.mode == "abort"
        assert verdict.freshness_days == 5

    def test_constructor_override_takes_priority(self, monkeypatch):
        """构造函数参数优先于环境变量"""
        monkeypatch.setenv("QUANT_QUALITY_GATE_FRESHNESS_ABORT_DAYS", "3")
        mod = _load_quality_gate_module()
        # 构造函数显式传 20，应覆盖环境变量的 3
        gate = mod.DataQualityGate(
            core_required=["daily"],
            freshness_abort_days=20,
            freshness_degraded_days=10,
        )
        df = _make_daily_df(end_date="20240920")
        # freshness 5 天 ≤ 10 → normal
        verdict = gate.check(tables={"daily": df}, asof="20240925")
        assert verdict.mode == "normal"


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestQualityGateVerdictToDict:
    """QualityVerdict.to_dict 序列化测试"""

    def test_to_dict_contains_all_fields(self):
        """to_dict 返回的字典应包含所有字段"""
        mod = _load_quality_gate_module()
        gate = mod.DataQualityGate(core_required=["daily"])
        df = _make_daily_df(end_date="20240920")
        verdict = gate.check(tables={"daily": df}, asof="20240925")
        d = verdict.to_dict()
        assert set(d.keys()) == {
            "mode", "missing_core", "missing_optional",
            "freshness_days", "pit_warnings", "reason",
        }
        assert d["mode"] == "normal"
        assert isinstance(d["missing_core"], list)
        assert isinstance(d["freshness_days"], int)
