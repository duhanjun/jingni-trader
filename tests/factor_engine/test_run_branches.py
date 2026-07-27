"""factor-engine.run() 端到端分支测试。

来源：原 test_jingni_datafeed_integration.py::TestFactorEngineRunBranches（5 用例）。

覆盖：
- factor_source='jingni' + 取数成功 → 产物存在，metadata.factor_source='jingni'
- factor_source='jingni' + 取数失败 → 回退到本地计算
- factor_source='local' → 直接走本地计算
- 已有 FACTOR 产物 → 直接返回 cache
- DATA 产物不存在 → 失败 + 错误信息
"""
from __future__ import annotations

import os
from unittest import mock

import pytest
import pandas as pd


# ============================================================================
# 共用辅助函数（与原 test_jingni_datafeed_integration.py 保持一致）
# ============================================================================

def _make_ctx(factor_source="jingni", stock_pool=None,
              start="2024-01-01", end="2024-06-30"):
    """构造测试用 Context 对象"""
    from scripts.context import Context
    ctx = Context(
        task_id="test",
        stock_pool=stock_pool or [],
        start_date=start,
        end_date=end,
    )
    ctx.metadata["factor_source"] = factor_source
    return ctx


def _make_data_parquet(tmp_path):
    """构造最小可用 DATA parquet（OHLCV），供 factor-engine 本地计算路径使用"""
    import numpy as np
    codes = ["000001.SZ", "600000.SH"]
    frames = []
    rng = np.random.RandomState(42)
    for code in codes:
        dates = pd.bdate_range("2024-01-01", "2024-03-31")
        n = len(dates)
        closes = 10 * (1 + np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({
            "code": code,
            "date": dates,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": rng.randint(1_000_000, 5_000_000, n),
        }))
    df = pd.concat(frames, ignore_index=True)
    path = str(tmp_path / "cleaned_data.parquet")
    df.to_parquet(path, index=False)
    return path


# ============================================================================
# 测试用例
# ============================================================================

class TestFactorEngineRunBranches:
    """验证 factor-engine.run() 在不同 factor_source 下的产物路径。"""

    def test_run_with_jingni_source(self, monkeypatch, tmp_path):
        """factor_source='jingni' + 取数成功 → 产物存在，metadata.factor_source='jingni'"""
        factor_dir = str(tmp_path / "factors")
        os.makedirs(factor_dir, exist_ok=True)

        # 直接修改 engine 模块的 FACTOR_DIR 全局变量
        import factor_engine_engine as fe_engine
        monkeypatch.setattr(fe_engine, "FACTOR_DIR", factor_dir)

        data_path = _make_data_parquet(tmp_path)
        ctx = _make_ctx(factor_source="jingni", stock_pool=["000001.SZ", "600000.SH"])
        ctx.update_artifact("DATA", data_path)

        mock_factors = pd.DataFrame({
            "code": ["000001.SZ", "600000.SH"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "factor_name": ["momentum_20d", "momentum_20d"],
            "factor_value": [0.05, -0.03],
        })

        with mock.patch.object(fe_engine, "_try_load_factor_from_datafeed", return_value=mock_factors):
            result = fe_engine.run(ctx)

        assert result["success"] is True
        assert result["metadata"]["factor_source"] == "jingni"
        assert os.path.exists(result["artifact_path"])

        df = pd.read_parquet(result["artifact_path"])
        assert len(df) == 2
        assert "factor_value" in df.columns

    def test_run_fallback_to_local_when_jingni_fails(self, monkeypatch, tmp_path):
        """factor_source='jingni' + 取数失败 → 回退到本地计算"""
        factor_dir = str(tmp_path / "factors")
        os.makedirs(factor_dir, exist_ok=True)

        import factor_engine_engine as fe_engine
        monkeypatch.setattr(fe_engine, "FACTOR_DIR", factor_dir)

        data_path = _make_data_parquet(tmp_path)
        ctx = _make_ctx(factor_source="jingni", stock_pool=["000001.SZ", "600000.SH"])
        ctx.update_artifact("DATA", data_path)

        with mock.patch.object(fe_engine, "_try_load_factor_from_datafeed", return_value=None):
            result = fe_engine.run(ctx)

        assert result["success"] is True
        # 走了本地路径，metadata 不应有 factor_source='jingni'
        assert result["metadata"].get("factor_source") != "jingni"
        assert os.path.exists(result["artifact_path"])

    def test_run_with_local_source(self, monkeypatch, tmp_path):
        """factor_source='local' → 直接走本地计算，不调用 datafeed"""
        factor_dir = str(tmp_path / "factors")
        os.makedirs(factor_dir, exist_ok=True)

        import factor_engine_engine as fe_engine
        monkeypatch.setattr(fe_engine, "FACTOR_DIR", factor_dir)

        data_path = _make_data_parquet(tmp_path)
        ctx = _make_ctx(factor_source="local", stock_pool=["000001.SZ", "600000.SH"])
        ctx.update_artifact("DATA", data_path)

        result = fe_engine.run(ctx)

        assert result["success"] is True
        assert os.path.exists(result["artifact_path"])

    def test_run_returns_cached_when_factor_artifact_exists(self, monkeypatch, tmp_path):
        """已有 FACTOR 产物时直接返回缓存"""
        factor_dir = str(tmp_path / "factors")
        os.makedirs(factor_dir, exist_ok=True)

        import factor_engine_engine as fe_engine
        monkeypatch.setattr(fe_engine, "FACTOR_DIR", factor_dir)

        # 预置一个已存在的 FACTOR 产物
        cached_path = str(tmp_path / "cached_factor.parquet")
        pd.DataFrame({"code": ["000001.SZ"], "v": [1]}).to_parquet(cached_path, index=False)

        ctx = _make_ctx(factor_source="jingni")
        ctx.update_artifact("DATA", _make_data_parquet(tmp_path))
        ctx.update_artifact("FACTOR", cached_path)

        result = fe_engine.run(ctx)

        assert result["success"] is True
        assert result["metadata"]["source"] == "cache"
        assert result["artifact_path"] == cached_path

    def test_run_fails_when_data_missing(self, monkeypatch, tmp_path):
        """DATA 产物不存在 → 返回失败"""
        import factor_engine_engine as fe_engine

        ctx = _make_ctx(factor_source="local")
        ctx.update_artifact("DATA", "/nonexistent/path.parquet")

        result = fe_engine.run(ctx)

        assert result["success"] is False
        assert "数据产物不存在" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
