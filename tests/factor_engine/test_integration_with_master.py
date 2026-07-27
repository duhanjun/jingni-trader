"""factor-engine 与主调度器 parse_intent 的全链路集成测试。

来源：原 test_jingni_datafeed_integration.py::TestE2EIntegration（2 用例）。

覆盖：
- 配置 JINGNI 环境变量 + 用户明确要求 → factor_source=jingni → run() 走 datafeed
- 未配置 JINGNI 环境变量 → factor_source=local → run() 走本地计算

验证 parse_intent gate 与 factor-engine.run() 的协同。
"""
from __future__ import annotations

import os
from unittest import mock

import pytest
import pandas as pd


# ============================================================================
# 共用辅助函数
# ============================================================================

def _make_data_parquet(tmp_path):
    """构造最小可用 DATA parquet（OHLCV）"""
    import numpy as np
    codes = ["000001.SZ", "600000.SH"]
    frames = []
    rng = np.random.RandomState(42)
    for code in codes:
        dates = pd.bdate_range("2024-01-01", "2024-03-31")
        n = len(dates)
        closes = 10 * (1 + np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({
            "code": code, "date": dates,
            "open": closes, "high": closes * 1.01,
            "low": closes * 0.99, "close": closes,
            "volume": rng.randint(1_000_000, 5_000_000, n),
        }))
    df = pd.concat(frames, ignore_index=True)
    path = str(tmp_path / "cleaned_data.parquet")
    df.to_parquet(path, index=False)
    return path


# ============================================================================
# 测试用例
# ============================================================================

class TestE2EIntegration:
    """从 parse_intent 到 factor-engine.run 的全链路验证。"""

    def test_full_pipeline_jingni_path(self, monkeypatch, tmp_path):
        """配置 JINGNI 环境变量 + 用户明确要求 → factor_source=jingni → run() 走 datafeed"""
        factor_dir = str(tmp_path / "factors")
        os.makedirs(factor_dir, exist_ok=True)

        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")
        monkeypatch.setenv("JINGNI_DEFAULT_DATASOURCE_UID", "factor-store")

        # Step 1: parse_intent 识别 factor_source
        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年因子做回测，从惊泥因子库取数")
        assert ctx.metadata["factor_source"] == "jingni"

        # Step 2: factor-engine.run() 走 datafeed 路径
        data_path = _make_data_parquet(tmp_path)
        ctx.update_artifact("DATA", data_path)
        ctx.stock_pool = ["000001.SZ", "600000.SH"]

        import factor_engine_engine as fe_engine
        monkeypatch.setattr(fe_engine, "FACTOR_DIR", factor_dir)

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

    def test_full_pipeline_local_path(self, monkeypatch, tmp_path):
        """未配置 JINGNI 环境变量 → factor_source=local → run() 走本地计算"""
        factor_dir = str(tmp_path / "factors")
        os.makedirs(factor_dir, exist_ok=True)

        monkeypatch.delenv("JINGNI_URL", raising=False)
        monkeypatch.delenv("JINGNI_TOKEN", raising=False)

        # Step 1: parse_intent 识别 factor_source
        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年因子做回测")
        assert ctx.metadata["factor_source"] == "local"

        # Step 2: factor-engine.run() 走本地计算路径
        data_path = _make_data_parquet(tmp_path)
        ctx.update_artifact("DATA", data_path)
        ctx.stock_pool = ["000001.SZ", "600000.SH"]

        import factor_engine_engine as fe_engine
        monkeypatch.setattr(fe_engine, "FACTOR_DIR", factor_dir)

        result = fe_engine.run(ctx)

        assert result["success"] is True
        assert os.path.exists(result["artifact_path"])
        # 本地路径不应有 factor_source='jingni'
        assert result["metadata"].get("factor_source") != "jingni"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
