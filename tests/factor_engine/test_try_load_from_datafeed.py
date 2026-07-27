"""factor-engine._try_load_factor_from_datafeed helper 取数与 fallback 测试。

来源：原 test_jingni_datafeed_integration.py::TestTryLoadFactorFromDatafeed（8 用例）。

覆盖：
- factor_source='local' → 直接返回 None
- 环境变量缺失 / 只配 URL → 返回 None
- 查询成功 → 返回 DataFrame 且 SQL 含股票池、时间范围、datasource UID
- 查询异常 / 空结果 → 返回 None（fallback）
- stock_pool 为空时 SQL 不含 IN 子句
- auto 路径也尝试取数

不依赖真实惊泥因子库服务，全部通过 mock 模拟。
"""
from __future__ import annotations

import os
from unittest import mock

import pytest


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


def _make_mock_query_result(rows):
    """构造 mock QueryResult，rows 为 list[dict]"""
    mock_result = mock.MagicMock()
    mock_result.to_table.return_value = rows
    return mock_result


# ============================================================================
# 测试用例
# ============================================================================

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
