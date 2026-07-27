"""7 个 skill 的标准输入 Context 构造器。

每个 skill 的 run(ctx) 都有自己的"标准输入契约"：
- 上游产物路径必须存在
- 必需字段必须填充（stock_pool、start_date、end_date 等）

本模块提供每个 skill 的标准输入 Context 构造器，供该 skill 的契约测试
与下游集成测试使用。所有构造器都使用临时目录，不污染真实 workspace。

来源：从 tests/test_jingni_datafeed_integration.py::_make_ctx 与
tests/test_integration_e2e.py 中的 ctx 构造逻辑抽出。
"""
from __future__ import annotations

import os
import sys
from typing import Any

# 确保项目根在 sys.path 中（测试可能在不同子目录下被 pytest 调用）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_master_context(user_intent: str = "获取近3年A股数据做一个反转因子选股回测并生成绩效报告"):
    """主调度器的标准输入：纯用户意图字符串。

    Returns:
        tuple: (MasterEngine 实例, Context 对象)
    """
    import engine
    from scripts.context import Context

    master = engine.MasterEngine()
    ctx = master.parse_intent(user_intent)
    return master, ctx


def make_data_engine_context(
    stock_pool: list[str] | None = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
) -> "Context":
    """data-engine 的标准输入 Context。

    data-engine 不依赖上游产物，只需 stock_pool + 时间范围。
    """
    from scripts.context import Context

    return Context(
        task_id="test_data",
        stock_pool=stock_pool if stock_pool is not None else ["000001.SZ", "600000.SH"],
        start_date=start,
        end_date=end,
    )


def make_factor_engine_context(
    data_artifact_path: str,
    factor_source: str = "local",
    stock_pool: list[str] | None = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
) -> "Context":
    """factor-engine 的标准输入 Context。

    必须包含上游 DATA 产物路径（data_artifact_path 应已存在）。
    """
    from scripts.context import Context

    ctx = Context(
        task_id="test_factor",
        stock_pool=stock_pool if stock_pool is not None else ["000001.SZ", "600000.SH"],
        start_date=start,
        end_date=end,
    )
    ctx.metadata["factor_source"] = factor_source
    ctx.update_artifact("DATA", data_artifact_path)
    return ctx


def make_strategy_model_engine_context(
    factor_artifact_path: str,
    stock_pool: list[str] | None = None,
) -> "Context":
    """strategy-model-engine 的标准输入 Context。

    必须包含上游 FACTOR 产物路径。
    """
    from scripts.context import Context

    ctx = Context(
        task_id="test_model",
        stock_pool=stock_pool if stock_pool is not None else ["000001.SZ", "600000.SH"],
        start_date="2024-01-01",
        end_date="2024-06-30",
    )
    ctx.update_artifact("FACTOR", factor_artifact_path)
    return ctx


def make_backtest_engine_context(
    factor_artifact_path: str,
    strategy_name: str = "single_factor",
    strategy_params: dict | None = None,
) -> "Context":
    """backtest-engine 的标准输入 Context。

    必须包含上游 FACTOR 产物路径 + 策略参数。
    """
    from scripts.context import Context

    ctx = Context(
        task_id="test_backtest",
        stock_pool=["000001.SZ", "600000.SH"],
        start_date="2024-01-01",
        end_date="2024-06-30",
        strategy_name=strategy_name,
        strategy_params=strategy_params or {},
    )
    ctx.update_artifact("FACTOR", factor_artifact_path)
    return ctx


def make_portfolio_risk_engine_context(backtest_artifact_path: str) -> "Context":
    """portfolio-risk-engine 的标准输入 Context。"""
    from scripts.context import Context

    ctx = Context(
        task_id="test_portfolio",
        stock_pool=["000001.SZ", "600000.SH"],
    )
    ctx.update_artifact("BACKTEST", backtest_artifact_path)
    return ctx


def make_execution_monitor_engine_context(portfolio_artifact_path: str) -> "Context":
    """execution-monitor-engine 的标准输入 Context。"""
    from scripts.context import Context

    ctx = Context(
        task_id="test_execution",
        stock_pool=["000001.SZ", "600000.SH"],
    )
    ctx.update_artifact("PORTFOLIO", portfolio_artifact_path)
    return ctx


def make_reports_engine_context(backtest_artifact_path: str) -> "Context":
    """reports-engine 的标准输入 Context。

    至少需要 BACKTEST 产物路径以生成报告。
    """
    from scripts.context import Context

    ctx = Context(
        task_id="test_report",
        stock_pool=["000001.SZ", "600000.SH"],
    )
    ctx.update_artifact("BACKTEST", backtest_artifact_path)
    return ctx
