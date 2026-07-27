"""mock JingniClient 工具。

提供 mock 惊泥因子库客户端的构造器，供 factor-engine 测试使用。
不依赖真实惊泥服务，全部通过 mock 模拟。

来源：从 tests/test_jingni_datafeed_integration.py 抽出共用 mock 构造逻辑。
"""
from __future__ import annotations

from unittest import mock


def make_mock_query_result(rows: list[dict]):
    """构造 mock QueryResult，rows 为 list[dict]。

    Args:
        rows: 查询返回的行数据

    Returns:
        MagicMock，其 to_table() 返回 rows
    """
    mock_result = mock.MagicMock()
    mock_result.to_table.return_value = rows
    return mock_result


def make_mock_jingni_client(rows: list[dict] | None = None, side_effect=None):
    """构造 mock JingniClient 实例。

    Args:
        rows: query_sql 正常返回的行数据（与 side_effect 互斥）
        side_effect: query_sql 抛出的异常（与 rows 互斥）

    Returns:
        MagicMock，模拟 JingniClient 实例的 query_sql 行为
    """
    mock_instance = mock.MagicMock()
    if side_effect is not None:
        mock_instance.query_sql.side_effect = side_effect
    else:
        mock_instance.query_sql.return_value = make_mock_query_result(rows or [])
    return mock_instance


def make_mocked_github_api_response(sha: str = "newsha456", date: str = "2026-07-26T10:00:00Z") -> bytes:
    """构造 GitHub API commits 端点返回的字节流。

    用于 skill_sync 测试，模拟远程最新 commit。

    Args:
        sha: commit SHA
        date: commit 日期 ISO 字符串

    Returns:
        bytes，JSON 编码的 GitHub API 响应
    """
    import json
    return json.dumps({
        "sha": sha,
        "commit": {"committer": {"date": date}},
    }).encode("utf-8")
