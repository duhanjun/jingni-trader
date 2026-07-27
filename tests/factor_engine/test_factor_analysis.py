"""factor-engine L2 单元测试：ic_analysis_v2 + factor_validator。

覆盖：
- calc_ic_series 对 spearman/pearson 两种方法的 IC 时间序列计算
- calc_ic_stats 对 IC 序列的统计量（ic_mean/ic_std/ic_ir/ic_positive_ratio/ic_t_stat）
- 空数据 / 缺失列 / 样本不足 等边界场景
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest
import pandas as pd
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACTOR_ENGINE_DIR = os.path.join(ROOT, "skills", "factor-engine")


def _load_ic_analysis_v2():
    """加载 factor-engine/scripts/optimizations/ic_analysis_v2.py 为独立模块。"""
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(FACTOR_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    for _m in ("talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    try:
        target_path = os.path.join(FACTOR_ENGINE_DIR, "scripts/optimizations/ic_analysis_v2.py")
        spec = ilu.spec_from_file_location("_fe_ic_analysis_v2", target_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["_fe_ic_analysis_v2"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


def _make_factor_data(n_days=20, n_stocks=15, seed=42):
    """构造测试用因子数据：含 date, factor_value, forward_return 列。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for d in dates:
        for c in codes:
            rows.append({
                "date": d,
                "code": c,
                "factor_value": rng.normal(0, 1),
                "forward_return": rng.normal(0, 0.02),
            })
    return pd.DataFrame(rows)


@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestCalcICSeries:
    """验证 calc_ic_series 的 IC 时间序列计算。"""

    def test_returns_series_with_spearman(self):
        """spearman 方法 → 返回以 date 为索引的 IC 序列"""
        ic_mod = _load_ic_analysis_v2()
        data = _make_factor_data()
        ic = ic_mod.calc_ic_series(data, "factor_value", "forward_return", method="spearman")
        assert isinstance(ic, pd.Series)
        assert ic.name == "ic"
        assert len(ic) > 0

    def test_returns_series_with_pearson(self):
        """pearson 方法 → 返回 IC 序列"""
        ic_mod = _load_ic_analysis_v2()
        data = _make_factor_data()
        ic = ic_mod.calc_ic_series(data, "factor_value", "forward_return", method="pearson")
        assert isinstance(ic, pd.Series)
        assert len(ic) > 0

    def test_invalid_method_raises(self):
        """未知方法 → 抛 ValueError"""
        ic_mod = _load_ic_analysis_v2()
        data = _make_factor_data()
        with pytest.raises(ValueError, match="不支持的 IC 方法"):
            ic_mod.calc_ic_series(data, "factor_value", "forward_return", method="unknown")

    def test_missing_column_returns_empty(self):
        """缺失列 → 返回空 Series"""
        ic_mod = _load_ic_analysis_v2()
        data = _make_factor_data()
        ic = ic_mod.calc_ic_series(data, "nonexistent_factor", "forward_return")
        assert ic.empty

    def test_empty_data_returns_empty(self):
        """空 DataFrame → 返回空 Series"""
        ic_mod = _load_ic_analysis_v2()
        ic = ic_mod.calc_ic_series(pd.DataFrame(), "factor_value", "forward_return")
        assert ic.empty

    def test_min_count_filter(self):
        """min_count 过滤：截面样本数不足时跳过该日"""
        ic_mod = _load_ic_analysis_v2()
        # 只 2 只股票，min_count=10 → 全部日期被过滤
        data = _make_factor_data(n_days=5, n_stocks=2)
        ic = ic_mod.calc_ic_series(data, "factor_value", "forward_return", min_count=10)
        assert ic.empty

    def test_ic_in_range(self):
        """IC 值应在 [-1, 1] 区间"""
        ic_mod = _load_ic_analysis_v2()
        data = _make_factor_data()
        ic = ic_mod.calc_ic_series(data, "factor_value", "forward_return", method="spearman")
        assert (ic >= -1.0001).all()
        assert (ic <= 1.0001).all()


@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestCalcICStats:
    """验证 calc_ic_stats 对 IC 序列的统计量计算。"""

    def test_returns_dict_with_required_fields(self):
        """返回 dict 含 5 个统计量字段"""
        ic_mod = _load_ic_analysis_v2()
        ic = pd.Series([0.05, 0.03, -0.02, 0.08, 0.01])
        stats = ic_mod.calc_ic_stats(ic)
        for field in ("ic_mean", "ic_std", "ic_ir", "ic_positive_ratio", "ic_t_stat"):
            assert field in stats

    def test_empty_series_returns_zeros(self):
        """空 Series → 全部 0"""
        ic_mod = _load_ic_analysis_v2()
        stats = ic_mod.calc_ic_stats(pd.Series(dtype=float))
        assert stats["ic_mean"] == 0.0
        assert stats["ic_ir"] == 0.0

    def test_none_series_returns_zeros(self):
        """None → 全部 0"""
        ic_mod = _load_ic_analysis_v2()
        stats = ic_mod.calc_ic_stats(None)
        assert stats["ic_mean"] == 0.0

    def test_positive_ic_series(self):
        """全正 IC 序列 → ic_positive_ratio=1.0, ic_mean>0"""
        ic_mod = _load_ic_analysis_v2()
        ic = pd.Series([0.1, 0.2, 0.05, 0.15])
        stats = ic_mod.calc_ic_stats(ic)
        assert stats["ic_positive_ratio"] == 1.0
        assert stats["ic_mean"] > 0

    def test_ic_ir_sign(self):
        """IC_IR 符号与 ic_mean 一致"""
        ic_mod = _load_ic_analysis_v2()
        ic = pd.Series([0.1, 0.2, 0.15, 0.25])
        stats = ic_mod.calc_ic_stats(ic)
        assert stats["ic_ir"] > 0  # 正 IC → 正 IR


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
