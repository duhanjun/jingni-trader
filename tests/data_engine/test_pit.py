"""P0-1 PIT 强制契约单元测试。

覆盖 scripts/pit.py 的三个核心函数：
- pit_filter: PIT 哨兵函数（缺列 raise / 过滤未来披露数据）
- scan_pit_warnings: 扫描违规行（不修改原 df）
- ensure_pit_filtered: 守卫函数（带 caller 审计）

测试用例：
1. 正常过滤：disclosure_date > asof 的行被剔除
2. 缺列 raise：QUANT_PIT_STRICT=true 时缺 disclosure_date 列 → ValueError
3. 缺列降级：QUANT_PIT_STRICT=false 时缺列 → warning + 返回原 df
4. 空 df：None / empty df 安全返回
5. asof 边界：disclosure_date == asof 的行保留（<= 判定）
6. disclosure_date 缺失回填场景模拟（adapter 层回填为 report_date）
7. scan_pit_warnings 不修改原 df
8. ensure_pit_filtered 带 caller 审计日志
9. 日期格式兼容：'YYYY-MM-DD' 与 'YYYYMMDD' 等价
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
import logging

import pytest
import pandas as pd
import numpy as np


# ============================================================================
# 模块加载：把 data-engine/scripts/pit.py 加载为独立模块
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_SCRIPTS = os.path.join(ROOT, "skills", "data-engine", "scripts")
PIT_PATH = os.path.join(DATA_ENGINE_SCRIPTS, "pit.py")


def _load_pit_module():
    """显式加载 data-engine/scripts/pit.py 为独立模块。"""
    spec = ilu.spec_from_file_location("_pit_test_module", PIT_PATH)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pit_mod(monkeypatch):
    """加载 pit 模块，默认 QUANT_PIT_STRICT=true。"""
    monkeypatch.setenv("QUANT_PIT_STRICT", "true")
    return _load_pit_module()


@pytest.fixture
def pit_mod_lenient(monkeypatch):
    """加载 pit 模块，QUANT_PIT_STRICT=false（宽松模式）。"""
    monkeypatch.setenv("QUANT_PIT_STRICT", "false")
    return _load_pit_module()


# ============================================================================
# 测试数据构造
# ============================================================================

def _make_financial_df():
    """构造含 disclosure_date 的财务数据 DataFrame。

    包含 3 行：
    - 000001.SZ: report_date=20240331, disclosure_date=20240430（Q1 报告，4 月底披露）
    - 600000.SH: report_date=20240630, disclosure_date=20240828（中报，8 月底披露）
    - 000002.SZ: report_date=20240930, disclosure_date=20241030（Q3 报告，10 月底披露）
    """
    return pd.DataFrame({
        'code': ['000001.SZ', '600000.SH', '000002.SZ'],
        'report_date': ['20240331', '20240630', '20240930'],
        'disclosure_date': ['20240430', '20240828', '20241030'],
        'roe': [15.2, 12.8, 18.5],
        'pe_ttm': [8.5, 5.2, 10.1],
    })


# ============================================================================
# Part 1: pit_filter 正常过滤
# ============================================================================

class TestPitFilterNormal:
    """pit_filter 正常过滤场景。"""

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_filter_future_disclosure_rows(self, pit_mod):
        """asof=20240701 应过滤掉 disclosure_date > 20240701 的行（中报和 Q3）"""
        df = _make_financial_df()
        result = pit_mod.pit_filter(df, asof='20240701')

        # 只保留 000001.SZ（disclosure_date=20240430 <= 20240701）
        assert len(result) == 1
        assert result.iloc[0]['code'] == '000001.SZ'

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_asof_boundary_inclusive(self, pit_mod):
        """disclosure_date == asof 的行应保留（<= 判定，边界包含）"""
        df = _make_financial_df()
        # asof=20240430 应保留 000001.SZ（disclosure_date=20240430 == asof）
        result = pit_mod.pit_filter(df, asof='20240430')
        assert len(result) == 1
        assert result.iloc[0]['code'] == '000001.SZ'

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_all_rows_pass_when_asof_late(self, pit_mod):
        """asof 足够晚时所有行都应保留"""
        df = _make_financial_df()
        result = pit_mod.pit_filter(df, asof='20251231')
        assert len(result) == 3

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_all_rows_filtered_when_asof_early(self, pit_mod):
        """asof 早于所有 disclosure_date 时所有行都被过滤"""
        df = _make_financial_df()
        result = pit_mod.pit_filter(df, asof='20240101')
        assert len(result) == 0

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_date_format_compatibility(self, pit_mod):
        """'YYYY-MM-DD' 与 'YYYYMMDD' 格式等价"""
        df = _make_financial_df()
        result_dash = pit_mod.pit_filter(df, asof='2024-07-01')
        result_nodash = pit_mod.pit_filter(df, asof='20240701')
        assert len(result_dash) == len(result_nodash) == 1
        assert result_dash.iloc[0]['code'] == result_nodash.iloc[0]['code']


# ============================================================================
# Part 2: pit_filter 缺列处理
# ============================================================================

class TestPitFilterMissingColumn:
    """pit_filter 缺 disclosure_date 列时的行为。"""

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_raise_when_strict_and_missing_column(self, pit_mod):
        """QUANT_PIT_STRICT=true 时缺 disclosure_date 列 → ValueError"""
        df = pd.DataFrame({
            'code': ['000001.SZ'],
            'report_date': ['20240331'],
            'roe': [15.2],
        })
        with pytest.raises(ValueError, match="disclosure_date"):
            pit_mod.pit_filter(df, asof='20240701')

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_warning_when_lenient_and_missing_column(self, pit_mod_lenient):
        """QUANT_PIT_STRICT=false 时缺列 → warning + 返回原 df（不过滤）"""
        df = pd.DataFrame({
            'code': ['000001.SZ'],
            'report_date': ['20240331'],
            'roe': [15.2],
        })
        result = pit_mod_lenient.pit_filter(df, asof='20240701')
        # 宽松模式返回原 df（不过滤）
        assert len(result) == 1
        assert result.iloc[0]['code'] == '000001.SZ'


# ============================================================================
# Part 3: pit_filter 空 df 处理
# ============================================================================

class TestPitFilterEmpty:
    """pit_filter 空 DataFrame / None 处理。"""

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_empty_dataframe_returns_empty(self, pit_mod):
        """空 DataFrame 安全返回（不 raise）"""
        df = pd.DataFrame()
        result = pit_mod.pit_filter(df, asof='20240701')
        assert result is df or len(result) == 0

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_none_returns_none(self, pit_mod):
        """None 输入安全返回 None（不 raise）"""
        result = pit_mod.pit_filter(None, asof='20240701')
        assert result is None


# ============================================================================
# Part 4: scan_pit_warnings 不修改原 df
# ============================================================================

class TestScanPitWarnings:
    """scan_pit_warnings 扫描违规行但不修改原 df。"""

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_scan_returns_warnings_without_modifying_df(self, pit_mod):
        """scan_pit_warnings 返回违规行信息，原 df 不变"""
        df = _make_financial_df()
        original_len = len(df)
        warnings = pit_mod.scan_pit_warnings(df, asof='20240701', table_name='financial')

        # 应有 2 行违规（600000.SH 和 000002.SZ 的 disclosure_date > 20240701）
        assert len(warnings) == 2
        assert all(w['table'] == 'financial' for w in warnings)
        assert all(w['asof'] == '20240701' for w in warnings)

        # 原 df 未被修改
        assert len(df) == original_len

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_scan_returns_empty_when_no_violation(self, pit_mod):
        """无违规时返回空列表"""
        df = _make_financial_df()
        warnings = pit_mod.scan_pit_warnings(df, asof='20251231')
        assert warnings == []

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_scan_returns_error_when_missing_column(self, pit_mod):
        """缺 disclosure_date 列时返回 error 信息（不 raise）"""
        df = pd.DataFrame({
            'code': ['000001.SZ'],
            'report_date': ['20240331'],
        })
        warnings = pit_mod.scan_pit_warnings(df, asof='20240701')
        assert len(warnings) == 1
        assert warnings[0]['error'] == 'missing_disclosure_date'
        assert warnings[0]['table'] == 'financial'


# ============================================================================
# Part 5: ensure_pit_filtered 守卫函数
# ============================================================================

class TestEnsurePitFiltered:
    """ensure_pit_filtered 守卫函数（带 caller 审计）。"""

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_guard_filters_and_logs_caller(self, pit_mod, caplog):
        """ensure_pit_filtered 过滤数据并记录 caller 信息"""
        df = _make_financial_df()
        with caplog.at_level(logging.INFO):
            result = pit_mod.ensure_pit_filtered(
                df, asof='20240701', caller='financial_factors.compute'
            )
        assert len(result) == 1
        assert result.iloc[0]['code'] == '000001.SZ'
        # 审计日志应包含 caller 标识
        assert any('financial_factors.compute' in r.message for r in caplog.records)

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_guard_raises_when_strict_and_missing_column(self, pit_mod):
        """严格模式 + 缺列 → raise（含 caller 信息）"""
        df = pd.DataFrame({
            'code': ['000001.SZ'],
            'report_date': ['20240331'],
        })
        with pytest.raises(ValueError, match="financial_factors"):
            pit_mod.ensure_pit_filtered(
                df, asof='20240701', caller='financial_factors.compute'
            )

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_guard_returns_empty_df_unchanged(self, pit_mod):
        """空 df 传入时安全返回"""
        df = pd.DataFrame()
        result = pit_mod.ensure_pit_filtered(df, asof='20240701', caller='test')
        assert result is df or len(result) == 0


# ============================================================================
# Part 6: disclosure_date 回填场景（模拟 adapter 保守降级）
# ============================================================================

class TestDisclosureDateBackfill:
    """模拟 adapter 缺原生披露日时回填为 report_date 的场景。"""

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_backfill_to_report_date_allows_pit_filter(self, pit_mod):
        """disclosure_date 回填为 report_date 时，pit_filter 仍能正常工作"""
        # 模拟 baostock/akshare 等 adapter：disclosure_date = report_date
        df = pd.DataFrame({
            'code': ['000001.SZ', '600000.SH'],
            'report_date': ['20240331', '20240630'],
            'disclosure_date': ['20240331', '20240630'],  # 回填为 report_date
            'roe': [15.2, 12.8],
        })
        # asof=20240515 应只保留 000001.SZ（report_date=20240331 <= 20240515）
        result = pit_mod.pit_filter(df, asof='20240515')
        assert len(result) == 1
        assert result.iloc[0]['code'] == '000001.SZ'

    @pytest.mark.skill_data_engine
    @pytest.mark.unit
    def test_mixed_real_and_backfill_disclosure_date(self, pit_mod):
        """混合真实披露日和回填值的 df 也能正常过滤"""
        df = pd.DataFrame({
            'code': ['000001.SZ', '600000.SH', '000002.SZ'],
            'report_date': ['20240331', '20240630', '20240930'],
            'disclosure_date': ['20240430', '20240630', '20241030'],  # 中间一个是回填
            'roe': [15.2, 12.8, 18.5],
        })
        result = pit_mod.pit_filter(df, asof='20240701')
        # 000001.SZ (20240430) 和 600000.SH (20240630) 都 <= 20240701
        assert len(result) == 2
        assert set(result['code']) == {'000001.SZ', '600000.SH'}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
