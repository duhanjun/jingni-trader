"""非量化投资者分析路径全链路集成测试。

覆盖：
- 非量化意图 '分析000001.SZ技术面' 触发 DATA → FACTOR → REPORT 路径
- 管道产出 REPORT 阶段产物（report.html）
- 量化意图仍走全量量化管道（DATA → FACTOR → MODEL → BACKTEST → REPORT）

设计要点：
- 使用合成 OHLCV 数据注入 ctx.external_data，避免任何网络/外部数据源依赖
- 使用临时 QUANT_WORK_DIR，避免污染真实归档目录
- 开启 ALLOW_SYNTHETIC_FALLBACK=true，保证无外部依赖时仍能跑完管道
"""
from __future__ import annotations

import os

import pytest


def _build_synthetic_external_data():
    """构造最小可用 OHLCV DataFrame 注入 ctx.external_data。

    复用 tests/fixtures/synthetic_data.py 的 make_synthetic_daily，
    保证可复现且无任何外部数据源依赖。
    """
    from synthetic_data import make_synthetic_daily
    return make_synthetic_daily(
        codes=["000001.SZ", "600000.SH"],
        start="2024-01-01",
        end="2024-06-30",
    )


class TestStockAnalysisPipeline:
    """非量化个人投资者分析路径端到端集成测试。"""

    def _run_non_quant_pipeline(self, work_dir: str) -> dict:
        """跑一次非量化意图的完整管道，返回 run_pipeline 结果。"""
        import engine

        intent = "分析000001.SZ技术面"
        master = engine.MasterEngine()
        ctx = master.parse_intent(intent)
        # 明确股票池与时间范围，避免触发全市场列表获取
        ctx.stock_pool = ["000001.SZ", "600000.SH"]
        ctx.start_date = "2024-01-01"
        ctx.end_date = "2024-06-30"
        # 注入合成数据，避免任何外部数据源/网络依赖
        ctx.external_data = {
            "daily": _build_synthetic_external_data(),
            "source": "integration-test",
        }
        return master.run_pipeline(ctx=ctx)

    def _setup_env(self, monkeypatch, tmp_path):
        """统一设置测试环境变量与临时工作目录。"""
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        monkeypatch.setenv("QUANT_WORK_DIR", str(work_dir))
        monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "true")
        monkeypatch.setenv("DATA_BACKENDS", "websearch")
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        return str(work_dir)

    @pytest.mark.integration
    @pytest.mark.skill_master
    @pytest.mark.slow
    def test_non_quant_intent_triggers_data_factor_report_path(self, tmp_path, monkeypatch):
        """非量化意图 '分析000001.SZ技术面' 触发 DATA → FACTOR → REPORT 路径"""
        work_dir = self._setup_env(monkeypatch, tmp_path)

        results = self._run_non_quant_pipeline(work_dir)

        # 非量化路径的 target_stages 必须是 DATA → FACTOR → REPORT
        target_stages = results.get("context", {}).get("target_stages")
        assert target_stages == ["DATA", "FACTOR", "REPORT"], (
            f"非量化意图应路由到 DATA → FACTOR → REPORT，实际: {target_stages}"
        )

        # 分析意图应设置 report_template
        metadata = results.get("context", {}).get("metadata") or {}
        assert "report_template" in metadata, (
            "分析意图应设置 ctx.metadata['report_template']"
        )

    @pytest.mark.integration
    @pytest.mark.skill_master
    @pytest.mark.slow
    def test_non_quant_pipeline_produces_report_artifact(self, tmp_path, monkeypatch):
        """非量化路径产出 REPORT 阶段产物（report.html）"""
        work_dir = self._setup_env(monkeypatch, tmp_path)

        results = self._run_non_quant_pipeline(work_dir)

        # REPORT 阶段必须出现在已完成列表中
        completed = results.get("completed_stages", [])
        assert "REPORT" in completed, (
            f"REPORT 阶段未完成: completed={completed}, "
            f"failed={results.get('failed_stages')}"
        )

        # report.html 产物文件必须在磁盘上存在
        import engine
        ctx = results.get("context", {})
        artifacts = ctx.get("artifacts") or {}
        report_artifact = artifacts.get("REPORT") or os.path.join(
            engine.REPORT_DIR, engine.EXPECTED_ARTIFACTS["REPORT"]
        )
        assert report_artifact and os.path.exists(report_artifact), (
            f"REPORT 产物缺失: resolved={report_artifact}"
        )

    @pytest.mark.integration
    @pytest.mark.skill_master
    @pytest.mark.slow
    def test_quant_intent_goes_through_full_quant_pipeline(self, tmp_path, monkeypatch):
        """量化意图仍走全量量化管道（DATA → FACTOR → MODEL → BACKTEST → REPORT）"""
        work_dir = self._setup_env(monkeypatch, tmp_path)

        import engine
        intent = "获取近3年A股数据做一个反转因子选股回测并生成绩效报告"
        master = engine.MasterEngine()
        ctx = master.parse_intent(intent)
        # 限制为最小量化链路（避免 EXECUTION/PORTFOLIO 真实下单/风控）
        ctx.target_stages = ["DATA", "FACTOR", "MODEL", "BACKTEST", "REPORT"]
        ctx.stock_pool = ["000001.SZ", "600000.SH"]
        ctx.start_date = "2024-01-01"
        ctx.end_date = "2024-06-30"
        ctx.external_data = {
            "daily": _build_synthetic_external_data(),
            "source": "integration-test",
        }

        results = master.run_pipeline(ctx=ctx)

        # 新单一工作流模型：strategy_required=True 触发完整量化管线
        metadata = results.get("context", {}).get("metadata") or {}
        assert metadata.get("strategy_required") is True, (
            "量化意图应设置 strategy_required=True"
        )

        # report_template 是正交维度，两条路径都会设置；
        # 策略路径下 REPORT 阶段靠 BACKTEST 产物存在性自动走绩效报告路径
        # （reports-engine/engine.py:run() 已统一路由），所以 report_template
        # 字段存在不影响策略路径的绩效报告输出。
        assert "report_template" in metadata, (
            "report_template 应在两条路径下都被设置（正交维度）"
        )

        # 量化路径应包含 MODEL/BACKTEST 阶段
        target_stages = results.get("context", {}).get("target_stages") or []
        assert "MODEL" in target_stages, (
            f"量化路径应包含 MODEL 阶段: {target_stages}"
        )
        assert "BACKTEST" in target_stages, (
            f"量化路径应包含 BACKTEST 阶段: {target_stages}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
