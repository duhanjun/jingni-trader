"""jingni-datafeed 整合到 jingni-trader 的端到端集成测试。

覆盖三个层面：
1. 主调度器 parse_intent 的 factor_source gate 逻辑（三态：local/auto/jingni）
2. factor-engine._try_load_factor_from_datafeed helper 的取数与 fallback
3. factor-engine.run() 在不同 factor_source 下的端到端分支

设计要点：
- 不依赖真实惊泥因子库服务，全部通过 mock 模拟
- 不依赖 sklearn/talib（已在 conftest.py 中 mock）
- 使用临时目录避免污染 workspace
- Windows GBK 控制台统一 utf-8 输出
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
from unittest import mock

import pytest
import pandas as pd

# Windows 控制台编码统一
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================================
# Part 1: parse_intent factor_source gate 逻辑
# ============================================================================

class TestParseIntentGate:
    """验证 MasterEngine.parse_intent 正确设置 ctx.metadata['factor_source']。"""

    def test_local_when_env_not_configured(self, monkeypatch):
        """未配置 JINGNI_URL/TOKEN → factor_source='local'"""
        monkeypatch.delenv("JINGNI_URL", raising=False)
        monkeypatch.delenv("JINGNI_TOKEN", raising=False)

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年 momentum 因子做回测")

        assert ctx.metadata["factor_source"] == "local"

    def test_auto_when_configured_but_not_requested(self, monkeypatch):
        """配置了凭证但用户没明确要求因子库 → factor_source='auto'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年因子做回测")

        assert ctx.metadata["factor_source"] == "auto"
        assert "FACTOR" in ctx.target_stages

    def test_jingni_when_explicitly_requested_chinese(self, monkeypatch):
        """配置了凭证 + 用户明确要求从因子库取数（中文）→ factor_source='jingni'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年因子做回测，从惊泥因子库取数")

        assert ctx.metadata["factor_source"] == "jingni"

    def test_jingni_when_explicitly_requested_english(self, monkeypatch):
        """配置了凭证 + 英文关键字 jingni → factor_source='jingni'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("use jingni factor store for backtest")

        assert ctx.metadata["factor_source"] == "jingni"

    def test_jingni_when_factor_store_keyword(self, monkeypatch):
        """factor_store 关键字也能触发"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("从 factor-store 取因子做回测")

        assert ctx.metadata["factor_source"] == "jingni"

    def test_local_when_no_factor_stage(self, monkeypatch):
        """没有 FACTOR 阶段时，factor_source 保持 'local'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("获取近3年数据")  # 只有 DATA 阶段

        assert ctx.metadata["factor_source"] == "local"
        assert "FACTOR" not in ctx.target_stages

    def test_auto_when_only_url_configured(self, monkeypatch):
        """只配了 JINGNI_URL 但没配 TOKEN → 视为未配置，factor_source='local'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.delenv("JINGNI_TOKEN", raising=False)

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年因子做回测")

        assert ctx.metadata["factor_source"] == "local"


# ============================================================================
# Part 2: _try_load_factor_from_datafeed helper
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


def _make_mock_query_result(rows):
    """构造 mock QueryResult，rows 为 list[dict]"""
    mock_result = mock.MagicMock()
    mock_result.to_table.return_value = rows
    return mock_result


class TestTryLoadFactorFromDatafeed:
    """验证 factor-engine._try_load_factor_from_datafeed 的取数与 fallback。"""

    def test_returns_none_when_local(self, monkeypatch):
        """factor_source='local' → 直接返回 None，不尝试取数"""
        from factor_engine_engine import _try_load_factor_from_datafeed

        ctx = _make_ctx(factor_source="local")
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        assert _try_load_factor_from_datafeed(ctx) is None

    def test_returns_none_when_env_not_configured(self, monkeypatch):
        """factor_source='jingni' 但环境变量未配 → 返回 None"""
        from factor_engine_engine import _try_load_factor_from_datafeed

        ctx = _make_ctx(factor_source="jingni")
        monkeypatch.delenv("JINGNI_URL", raising=False)
        monkeypatch.delenv("JINGNI_TOKEN", raising=False)

        assert _try_load_factor_from_datafeed(ctx) is None

    def test_returns_none_when_only_url(self, monkeypatch):
        """只配了 URL 没配 TOKEN → 返回 None"""
        from factor_engine_engine import _try_load_factor_from_datafeed

        ctx = _make_ctx(factor_source="jingni")
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.delenv("JINGNI_TOKEN", raising=False)

        assert _try_load_factor_from_datafeed(ctx) is None

    def test_returns_dataframe_when_query_succeeds(self, monkeypatch):
        """factor_source='jingni' + 环境变量已配 + 查询成功 → 返回 DataFrame"""
        from factor_engine_engine import _try_load_factor_from_datafeed

        ctx = _make_ctx(
            factor_source="jingni",
            stock_pool=["000001.SZ", "600000.SH"],
            start="2024-01-01",
            end="2024-06-30",
        )
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")
        monkeypatch.setenv("JINGNI_DEFAULT_DATASOURCE_UID", "factor-store")

        mock_rows = [
            {"code": "000001.SZ", "date": "20240102", "factor_name": "momentum_20d", "factor_value": 0.05},
            {"code": "600000.SH", "date": "20240102", "factor_name": "momentum_20d", "factor_value": -0.03},
        ]
        mock_result = _make_mock_query_result(mock_rows)

        mock_client_instance = mock.MagicMock()
        mock_client_instance.query_sql.return_value = mock_result

        with mock.patch("jingni_client.JingniClient", return_value=mock_client_instance):
            df = _try_load_factor_from_datafeed(ctx)

        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["code", "date", "factor_name", "factor_value"]
        assert df["factor_value"].dtype.kind == "f"  # 已转数值

        # 验证 SQL 包含股票池和时间范围
        call_kwargs = mock_client_instance.query_sql.call_args.kwargs
        sql = call_kwargs.get("raw_sql", "")
        assert "000001.SZ" in sql
        assert "600000.SH" in sql
        assert "20240101" in sql
        assert "20240630" in sql
        assert call_kwargs.get("uid") == "factor-store"

    def test_returns_none_when_query_raises(self, monkeypatch):
        """factor_source='jingni' + 查询异常 → 返回 None（fallback）"""
        from factor_engine_engine import _try_load_factor_from_datafeed

        ctx = _make_ctx(factor_source="jingni")
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        mock_client_instance = mock.MagicMock()
        mock_client_instance.query_sql.side_effect = RuntimeError("connection refused")

        with mock.patch("jingni_client.JingniClient", return_value=mock_client_instance):
            df = _try_load_factor_from_datafeed(ctx)

        assert df is None

    def test_returns_none_when_empty_rows(self, monkeypatch):
        """factor_source='jingni' + 查询返回空 → 返回 None"""
        from factor_engine_engine import _try_load_factor_from_datafeed

        ctx = _make_ctx(factor_source="jingni")
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        mock_result = _make_mock_query_result([])

        mock_client_instance = mock.MagicMock()
        mock_client_instance.query_sql.return_value = mock_result

        with mock.patch("jingni_client.JingniClient", return_value=mock_client_instance):
            df = _try_load_factor_from_datafeed(ctx)

        assert df is None

    def test_no_stock_pool_in_sql_when_empty(self, monkeypatch):
        """stock_pool 为空时，SQL 不应包含 IN 子句"""
        from factor_engine_engine import _try_load_factor_from_datafeed

        ctx = _make_ctx(factor_source="jingni", stock_pool=[])
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        mock_result = _make_mock_query_result([
            {"code": "000001.SZ", "date": "20240102", "factor_name": "f1", "factor_value": 0.1}
        ])

        mock_client_instance = mock.MagicMock()
        mock_client_instance.query_sql.return_value = mock_result

        with mock.patch("jingni_client.JingniClient", return_value=mock_client_instance):
            _try_load_factor_from_datafeed(ctx)

        sql = mock_client_instance.query_sql.call_args.kwargs.get("raw_sql", "")
        # WHERE 子句之后不应出现 IN
        where_clause = sql.split("WHERE", 1)[1] if "WHERE" in sql.upper() else ""
        assert "IN (" not in where_clause

    def test_auto_source_also_tries_datafeed(self, monkeypatch):
        """factor_source='auto' 也应尝试取数（与 'jingni' 相同路径）"""
        from factor_engine_engine import _try_load_factor_from_datafeed

        ctx = _make_ctx(factor_source="auto")
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        mock_result = _make_mock_query_result([
            {"code": "000001.SZ", "date": "20240102", "factor_name": "f1", "factor_value": 0.1}
        ])

        mock_client_instance = mock.MagicMock()
        mock_client_instance.query_sql.return_value = mock_result

        with mock.patch("jingni_client.JingniClient", return_value=mock_client_instance):
            df = _try_load_factor_from_datafeed(ctx)

        assert df is not None  # auto 也应该能取到数据


# ============================================================================
# Part 3: factor-engine.run() 端到端分支
# ============================================================================

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


# ============================================================================
# Part 4: 端到端集成（parse_intent → factor-engine.run 全链路）
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
