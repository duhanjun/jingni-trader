"""L3 集成测试：验证相邻 skill 对的数据契约与产物传递。

每个测试聚焦一对相邻 skill 的真实对接：
1. 上游 skill run(ctx) 真实执行并产出约定产物
2. 下游 skill run(ctx) 读取上游产物并产出自己的约定产物
3. 验证产物之间的契约字段对齐（列名、JSON 结构、路径耦合等）

5 对相邻 skill 链路（第 6 对 DATA→FACTOR→BACKTEST→REPORT 已在
test_pipeline_minimal.py 中覆盖）：
- DATA → FACTOR
- FACTOR → BACKTEST
- BACKTEST → PORTFOLIO
- PORTFOLIO → EXECUTION
- BACKTEST → REPORT

设计要点：
- 不依赖任何外部数据源/网络，注入合成 OHLCV 数据
- 使用临时 QUANT_WORK_DIR，避免污染真实归档目录
- 使用 monkeypatch 隔离环境变量，pytest 自动清理
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# 共享辅助：构造合成数据 + 运行管道前缀
# ============================================================================

def _build_synthetic_daily():
    """构造最小可用 OHLCV DataFrame，注入 ctx.external_data 以确定性跑通 DATA 阶段。"""
    import numpy as np
    import pandas as pd

    codes = ["000001.SZ", "600000.SH"]
    frames = []
    rng = np.random.RandomState(20240101)
    for code in codes:
        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        n = len(dates)
        base = rng.uniform(8, 20)
        closes = base * (1 + np.cumsum(rng.normal(0, 0.01, n)))
        opens = closes * (1 + rng.normal(0, 0.002, n))
        highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n)))
        vol = rng.randint(1_000_000, 10_000_000, n)
        frames.append(pd.DataFrame({
            "code": code, "date": dates,
            "open": opens.round(2), "high": highs.round(2),
            "low": lows.round(2), "close": closes.round(2),
            "volume": vol,
        }))
    return pd.concat(frames, ignore_index=True)


def _run_pipeline_prefix(stages: list[str], monkeypatch, tmp_path):
    """运行管道前缀阶段，返回 (results, ctx, work_dir)。

    stages 必须按依赖顺序排列（如 ["DATA","FACTOR"]）。
    使用合成数据 + 临时工作目录，不依赖外部环境。
    """
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    exec_dir = str(work_dir / "execution")
    monkeypatch.setenv("QUANT_WORK_DIR", str(work_dir))
    # EXECUTION_DIR 是 execution-monitor-engine 自己的配置（不同于 QUANT_WORK_DIR）
    monkeypatch.setenv("EXECUTION_DIR", exec_dir)
    # 放宽单笔订单比例上限，避免等权组合（2 只股票各 50%）触发风控拒绝
    monkeypatch.setenv("MAX_SINGLE_ORDER_RATIO", "1.0")
    monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "true")
    monkeypatch.setenv("DATA_BACKENDS", "websearch")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    # 强制使用模拟交易执行器，避免环境变量 TRADE_MODE=live/TRADE_BACKEND=gm
    # 导致集成测试尝试连接真实交易服务而失败（测试应保持 hermetic）
    monkeypatch.setenv("TRADE_MODE", "paper")
    monkeypatch.delenv("TRADE_BACKEND", raising=False)

    # 关键：重新加载 engine + scripts 包，使 QUANT_WORK_DIR 等环境变量生效。
    # engine.py 在首次被 import 时把 WORK_DIR/DATA_DIR/FACTOR_DIR/PORTFOLIO_DIR 等
    # 从 scripts.config 读取并固定为模块级变量；如果其他测试先 import engine，
    # 这些变量就被绑定到当时的临时目录，后续 monkeypatch.setenv 不会重新读取。
    #
    # 此外，每个子 skill 的 engine.py 顶层都有 `sys.path.insert(0, 自己的目录)`，
    # 一旦其他测试触发了 master.run_pipeline → importlib.import_module("skills.X.engine")，
    # 对应的 skills/X 目录就被插入 sys.path 最前面。此时 `import engine` 会优先
    # 解析到 skills/X/engine.py（子 skill），而不是根目录 engine.py（主调度器），
    # reload(engine) 会重新执行子 skill engine.py，触发顶层 `from scripts.optimizations...`
    # 但此时 sys.modules['scripts'] 是主 scripts 包（无 optimizations 子目录），导致
    # ModuleNotFoundError。
    #
    # 解决方案：
    # 1) 清理 sys.path 中被注入的 skills/* 子目录（保留 ROOT 自身）
    # 2) 清理所有 scripts.* 和 engine 缓存
    # 3) 用 importlib.util 显式重新注册主 scripts 包（指向 ROOT/scripts）
    # 4) 用 importlib.util 显式重新加载根目录的 engine.py
    import importlib
    import importlib.util as _ilu

    # (1) 清理 sys.path 中被注入的子 skill 目录
    for _p in list(sys.path):
        if _p.startswith(os.path.join(ROOT, "skills") + os.sep):
            sys.path.remove(_p)

    # (2) 清理所有 scripts.* 和 engine 缓存
    for _k in list(sys.modules.keys()):
        if _k == "scripts" or _k.startswith("scripts.") or _k == "engine":
            sys.modules.pop(_k, None)

    # (3) 用 importlib.util 显式重新注册主 scripts 包
    scripts_dir = os.path.join(ROOT, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        _spec = _ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        _pkg = _ilu.module_from_spec(_spec)
        _pkg.__path__ = [scripts_dir]
        sys.modules["scripts"] = _pkg
        _spec.loader.exec_module(_pkg)
    else:
        import types
        _pkg = types.ModuleType("scripts")
        _pkg.__path__ = [scripts_dir]
        sys.modules["scripts"] = _pkg

    # (4) 用 importlib.util 显式重新加载根目录的 engine.py
    #    不依赖 sys.path 查找，避免被 skills/<子skill>/engine.py 抢先匹配
    _engine_path = os.path.join(ROOT, "engine.py")
    _engine_spec = _ilu.spec_from_file_location("engine", _engine_path)
    engine = _ilu.module_from_spec(_engine_spec)
    sys.modules["engine"] = engine
    _engine_spec.loader.exec_module(engine)

    master = engine.MasterEngine()
    ctx = master.parse_intent("获取近3年A股数据做一个反转因子选股回测并生成绩效报告")
    ctx.target_stages = stages
    ctx.stock_pool = ["000001.SZ", "600000.SH"]
    ctx.start_date = "2024-01-01"
    ctx.end_date = "2024-06-30"
    ctx.external_data = {"daily": _build_synthetic_daily(), "source": "l3-test"}
    results = master.run_pipeline(ctx=ctx)
    return results, ctx, str(work_dir)


# ============================================================================
# L3-1: DATA → FACTOR
# ============================================================================

@pytest.mark.integration
@pytest.mark.skill_data_engine
@pytest.mark.skill_factor_engine
class TestDataToFactorContract:
    """验证 DATA → FACTOR 的产物传递契约。

    契约要点：
    - DATA 产出 cleaned_data.parquet，必含 code/date/open/high/low/close/volume
    - FACTOR 读取 DATA 产物，产出 factor_data.parquet，必含 code/date/alpha_score
    - FACTOR metadata 包含 factor_names 列表
    """

    def test_data_artifact_has_required_columns(self, tmp_path, monkeypatch):
        """DATA 产物 cleaned_data.parquet 必含 OHLCV 列"""
        import pandas as pd

        results, ctx, work_dir = _run_pipeline_prefix(["DATA", "FACTOR"], monkeypatch, tmp_path)
        assert results["success"] is True

        data_path = ctx.artifacts.get("DATA")
        assert data_path and os.path.exists(data_path), f"DATA 产物缺失: {data_path}"

        df = pd.read_parquet(data_path)
        for col in ("code", "date", "open", "high", "low", "close", "volume"):
            assert col in df.columns, f"DATA 产物缺少列: {col}"

    def test_factor_reads_data_and_produces_alpha_score(self, tmp_path, monkeypatch):
        """FACTOR 读取 DATA 产物，产出 factor_data.parquet 必含 alpha_score 列"""
        import pandas as pd

        results, ctx, work_dir = _run_pipeline_prefix(["DATA", "FACTOR"], monkeypatch, tmp_path)
        assert results["success"] is True

        factor_path = ctx.artifacts.get("FACTOR")
        assert factor_path and os.path.exists(factor_path), f"FACTOR 产物缺失: {factor_path}"

        df = pd.read_parquet(factor_path)
        for col in ("code", "date", "alpha_score"):
            assert col in df.columns, f"FACTOR 产物缺少列: {col}"

    def test_factor_metadata_has_factor_names(self, tmp_path, monkeypatch):
        """FACTOR metadata 应包含 factor_names 列表"""
        results, ctx, work_dir = _run_pipeline_prefix(["DATA", "FACTOR"], monkeypatch, tmp_path)
        assert results["success"] is True

        factor_meta = ctx.metadata.get("FACTOR", {})
        # cache 路径下只有 source=cache，非 cache 才有 factor_names
        if factor_meta.get("source") != "cache":
            factor_names = factor_meta.get("factor_names")
            assert factor_names is not None, "FACTOR metadata 缺少 factor_names"
            assert isinstance(factor_names, list) and len(factor_names) > 0


# ============================================================================
# L3-2: FACTOR → BACKTEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.skill_factor_engine
@pytest.mark.skill_backtest_engine
class TestFactorToBacktestContract:
    """验证 FACTOR → BACKTEST 的产物传递契约。

    契约要点：
    - FACTOR 产出 factor_data.parquet 含 alpha_score 列
    - BACKTEST 读取 alpha_score（pct_rank > 0.8 → signal=1），产出 backtest_result.json
    - backtest_result.json 含 metrics/backend/timestamp
    - 同目录 equity_curve.parquet 存在（REPORT 依赖此约定）
    """

    def test_backtest_reads_factor_and_produces_result_json(self, tmp_path, monkeypatch):
        """BACKTEST 读取 FACTOR 产物，产出 backtest_result.json"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST"], monkeypatch, tmp_path
        )
        assert results["success"] is True

        bt_path = ctx.artifacts.get("BACKTEST")
        assert bt_path and os.path.exists(bt_path), f"BACKTEST 产物缺失: {bt_path}"

        with open(bt_path, "r", encoding="utf-8") as f:
            bt = json.load(f)
        for key in ("metrics", "backend", "timestamp"):
            assert key in bt, f"backtest_result.json 缺少字段: {key}"

    def test_backtest_metrics_has_required_fields(self, tmp_path, monkeypatch):
        """backtest_result.json 的 metrics 包含核心绩效指标"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST"], monkeypatch, tmp_path
        )
        assert results["success"] is True

        bt_path = ctx.artifacts.get("BACKTEST")
        with open(bt_path, "r", encoding="utf-8") as f:
            bt = json.load(f)
        metrics = bt.get("metrics", {})
        for key in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown"):
            assert key in metrics, f"backtest metrics 缺少: {key}"

    def test_equity_curve_parquet_exists_alongside_backtest(self, tmp_path, monkeypatch):
        """equity_curve.parquet 与 backtest_result.json 同目录（REPORT 依赖此约定）"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST"], monkeypatch, tmp_path
        )
        assert results["success"] is True

        bt_path = ctx.artifacts.get("BACKTEST")
        bt_dir = os.path.dirname(bt_path)
        equity_path = os.path.join(bt_dir, "equity_curve.parquet")
        assert os.path.exists(equity_path), f"equity_curve.parquet 缺失: {equity_path}"

        import pandas as pd
        eq = pd.read_parquet(equity_path)
        assert "date" in eq.columns and "equity" in eq.columns


# ============================================================================
# L3-3: BACKTEST → PORTFOLIO
# ============================================================================

@pytest.mark.integration
@pytest.mark.skill_backtest_engine
@pytest.mark.skill_portfolio_risk_engine
class TestBacktestToPortfolioContract:
    """验证 BACKTEST → PORTFOLIO 的产物传递契约。

    契约要点：
    - PORTFOLIO 读取 DATA（必需）+ 可选 FACTOR（alpha_score 作为预期收益）
    - PORTFOLIO 产出 portfolio_weights.json：{code: weight} dict
    - 权重非负、过滤掉 ≤0.0001 的微小值
    """

    def test_portfolio_produces_weights_json(self, tmp_path, monkeypatch):
        """PORTFOLIO 产出 portfolio_weights.json"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST", "PORTFOLIO"], monkeypatch, tmp_path
        )
        assert results["success"] is True
        assert "PORTFOLIO" in results["completed_stages"]

        pf_path = ctx.artifacts.get("PORTFOLIO")
        assert pf_path and os.path.exists(pf_path), f"PORTFOLIO 产物缺失: {pf_path}"

    def test_portfolio_weights_is_code_to_float_dict(self, tmp_path, monkeypatch):
        """portfolio_weights.json 结构为 {股票代码: 浮点权重}"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST", "PORTFOLIO"], monkeypatch, tmp_path
        )
        assert results["success"] is True

        pf_path = ctx.artifacts.get("PORTFOLIO")
        with open(pf_path, "r", encoding="utf-8") as f:
            weights = json.load(f)
        assert isinstance(weights, dict), "portfolio_weights.json 应为 dict"
        assert len(weights) > 0, "权重 dict 不应为空"
        for code, w in weights.items():
            assert isinstance(code, str), f"权重 key 应为 str: {code}"
            assert isinstance(w, (int, float)), f"权重 value 应为数值: {w}"
            assert w > 0, f"权重应 > 0（已过滤 ≤0.0001）: {code}={w}"

    def test_portfolio_metadata_has_metrics(self, tmp_path, monkeypatch):
        """PORTFOLIO metadata 包含 metrics 和 optimization_method"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST", "PORTFOLIO"], monkeypatch, tmp_path
        )
        assert results["success"] is True

        pf_meta = ctx.metadata.get("PORTFOLIO", {})
        if pf_meta.get("source") != "cache":
            assert "optimization_method" in pf_meta, "PORTFOLIO metadata 缺少 optimization_method"
            assert "num_assets" in pf_meta, "PORTFOLIO metadata 缺少 num_assets"


# ============================================================================
# L3-4: PORTFOLIO → EXECUTION
# ============================================================================

@pytest.mark.integration
@pytest.mark.skill_portfolio_risk_engine
@pytest.mark.skill_execution_monitor_engine
class TestPortfolioToExecutionContract:
    """验证 PORTFOLIO → EXECUTION 的产物传递契约。

    契约要点：
    - EXECUTION 读取 portfolio_weights.json（{code: weight}）
    - EXECUTION 产出 audit log（JSONL），每行含 order_id/code/side/volume/price/status
    - EXECUTION metadata 含 orders_executed 和 account_snapshot
    """

    def test_execution_reads_portfolio_and_produces_audit_log(self, tmp_path, monkeypatch):
        """EXECUTION 读取 PORTFOLIO 产物，产出 audit log"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST", "PORTFOLIO", "EXECUTION"],
            monkeypatch, tmp_path,
        )
        assert results["success"] is True
        assert "EXECUTION" in results["completed_stages"]

        exec_path = ctx.artifacts.get("EXECUTION")
        assert exec_path and os.path.exists(exec_path), f"EXECUTION 产物缺失: {exec_path}"

    def test_execution_audit_log_has_order_records(self, tmp_path, monkeypatch):
        """audit log（JSONL）每行是一条订单记录，含必需字段"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST", "PORTFOLIO", "EXECUTION"],
            monkeypatch, tmp_path,
        )
        assert results["success"] is True

        exec_path = ctx.artifacts.get("EXECUTION")
        with open(exec_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) > 0, "audit log 不应为空"

        for entry in lines:
            assert "code" in entry, f"audit log 条目缺少 code: {entry}"
            assert "side" in entry, f"audit log 条目缺少 side: {entry}"

    def test_execution_metadata_has_orders_executed(self, tmp_path, monkeypatch):
        """EXECUTION metadata 含 orders_executed 计数"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST", "PORTFOLIO", "EXECUTION"],
            monkeypatch, tmp_path,
        )
        assert results["success"] is True

        exec_meta = ctx.metadata.get("EXECUTION", {})
        if exec_meta.get("source") != "cache":
            assert "orders_executed" in exec_meta, "EXECUTION metadata 缺少 orders_executed"
            assert isinstance(exec_meta["orders_executed"], int)
            assert exec_meta["orders_executed"] >= 0


# ============================================================================
# L3-5: BACKTEST → REPORT
# ============================================================================

@pytest.mark.integration
@pytest.mark.skill_backtest_engine
@pytest.mark.skill_reports_engine
class TestBacktestToReportContract:
    """验证 BACKTEST → REPORT 的产物传递契约。

    契约要点：
    - REPORT 读取 BACKTEST 产物（同目录 equity_curve.parquet）
    - REPORT 产出 report.html + report_data.json
    - report_data.json 含 metrics 字段
    """

    def test_report_reads_backtest_and_produces_html(self, tmp_path, monkeypatch):
        """REPORT 读取 BACKTEST 产物，产出 report.html"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST", "REPORT"], monkeypatch, tmp_path
        )
        assert results["success"] is True
        assert "REPORT" in results["completed_stages"]

        rpt_path = ctx.artifacts.get("REPORT")
        assert rpt_path and os.path.exists(rpt_path), f"REPORT 产物缺失: {rpt_path}"
        assert rpt_path.endswith(".html"), f"REPORT 产物应为 .html: {rpt_path}"

        with open(rpt_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 0, "report.html 不应为空"

    def test_report_data_json_has_metrics(self, tmp_path, monkeypatch):
        """report_data.json 含 metrics 字段"""
        results, ctx, work_dir = _run_pipeline_prefix(
            ["DATA", "FACTOR", "BACKTEST", "REPORT"], monkeypatch, tmp_path
        )
        assert results["success"] is True

        rpt_meta = ctx.metadata.get("REPORT", {})
        report_data_path = rpt_meta.get("report_data_path")
        if report_data_path and os.path.exists(report_data_path):
            with open(report_data_path, "r", encoding="utf-8") as f:
                rpt_data = json.load(f)
            assert "metrics" in rpt_data, "report_data.json 缺少 metrics"
            assert "generated_at" in rpt_data, "report_data.json 缺少 generated_at"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
