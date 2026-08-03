"""reports-engine 绩效归因报告 L3 集成测试。

验证端到端流程：
- 从 ledger.jsonl 到 attribution_report.html 的完整链路
- 无 EXECUTION 产物时返回友好错误
- LLM 不可用时规则模板兜底
- 报告 HTML 包含关键章节
"""
from __future__ import annotations

import os
import sys
import json
import importlib.util as ilu
from unittest import mock

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_ENGINE_DIR = os.path.join(ROOT, "skills", "reports-engine")
REPORTS_ENGINE_PATH = os.path.join(REPORTS_ENGINE_DIR, "engine.py")

_CONTEXT_MODULE = None


def _get_context_class():
    """获取 Context 类（处理 scripts 包切换问题）"""
    global _CONTEXT_MODULE
    if _CONTEXT_MODULE is not None:
        return _CONTEXT_MODULE

    context_path = os.path.join(ROOT, "scripts", "context.py")
    if os.path.exists(context_path):
        spec = ilu.spec_from_file_location("jingni_context", context_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["jingni_context"] = mod
        spec.loader.exec_module(mod)
        _CONTEXT_MODULE = mod.Context
        return mod.Context

    raise ImportError("无法加载 Context 类")


def _load_reports_engine_module():
    """显式加载 reports-engine/engine.py 为独立模块。"""
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(REPORTS_ENGINE_DIR, "scripts")
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

    spec = ilu.spec_from_file_location("reports_engine_engine", REPORTS_ENGINE_PATH)
    mod = ilu.module_from_spec(spec)
    sys.modules["reports_engine_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_ctx(stock_pool=None):
    """构造 Context，设置绩效复盘意图"""
    Context = _get_context_class()
    ctx = Context(
        task_id="test_attribution",
        stock_pool=stock_pool or ["000001.SZ"],
        start_date="2024-01-01",
        end_date="2024-06-30",
    )
    ctx.metadata["report_intent"] = "attribution"
    return ctx


def _write_ledger(tmp_path, records):
    """将记录列表写入 EXECUTION 目录下的 ledger.jsonl"""
    exec_dir = tmp_path / "execution"
    exec_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = exec_dir / "ledger.jsonl"
    with open(ledger_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(exec_dir)


def _make_record(
    code="000001.SZ", side="buy", shares=100, price=10.0,
    trade_date="2024-01-15", commission=5.0, stamp_tax=0.0,
    slippage_cost=0.5, nav_after=1000000.0, cash_after=999895.0,
    position_after=100, confirmed=True,
):
    """构造单条成交记录"""
    return {
        "execution_id": f"exec_{code}_{trade_date}_{side}",
        "trade_date": trade_date,
        "code": code,
        "side": side,
        "shares": shares,
        "price": price,
        "commission": commission,
        "stamp_tax": stamp_tax,
        "slippage_cost": slippage_cost,
        "position_after_shares": position_after,
        "cash_after": cash_after,
        "nav_after": nav_after,
        "confirmed": confirmed,
        "created_at": f"{trade_date}T10:00:00",
    }


@pytest.mark.skill_reports_engine
@pytest.mark.integration
class TestAttributionReportIntegration:
    """绩效归因报告端到端集成测试。"""

    def test_end_to_end_attribution_report(self, monkeypatch, tmp_path):
        """端到端：从 ledger.jsonl 到 attribution_report.html"""
        work_dir = str(tmp_path / "workspace")
        monkeypatch.setenv("QUANT_WORK_DIR", work_dir)

        # 构造成交记录：买→卖 形成闭环
        records = [
            _make_record(side="buy", price=10.0, trade_date="2024-01-15",
                         nav_after=1000000.0, cash_after=999895.0, position_after=100),
            _make_record(side="sell", price=11.0, trade_date="2024-02-15",
                         stamp_tax=5.5, commission=5.5,
                         nav_after=1000094.5, cash_after=1000094.5, position_after=0),
        ]
        exec_dir = _write_ledger(tmp_path, records)

        reports_mod = _load_reports_engine_module()
        ctx = _make_ctx()
        ctx.update_artifact("EXECUTION", exec_dir)

        result = reports_mod.run(ctx)

        assert result["success"] is True
        assert result["artifact_path"] != ""
        assert os.path.exists(result["artifact_path"])
        assert result["metadata"]["report_type"] == "attribution"

        # 验证 HTML 包含关键章节
        with open(result["artifact_path"], "r", encoding="utf-8") as f:
            html = f.read()
        assert "绩效归因报告" in html
        assert "交易统计概览" in html
        assert "Round-Trip" in html
        assert "执行质量分析" in html
        assert "深度解读" in html

    def test_attribution_report_no_ledger(self, monkeypatch, tmp_path):
        """无 EXECUTION 产物时返回友好错误（不崩溃）"""
        work_dir = str(tmp_path / "workspace")
        monkeypatch.setenv("QUANT_WORK_DIR", work_dir)

        reports_mod = _load_reports_engine_module()
        ctx = _make_ctx()
        # 不设置 EXECUTION artifact

        result = reports_mod.run(ctx)

        assert result["success"] is False
        assert "EXECUTION" in result["error"]

    def test_attribution_report_no_ledger_file(self, monkeypatch, tmp_path):
        """EXECUTION 目录存在但 ledger.jsonl 不存在 → 友好错误"""
        work_dir = str(tmp_path / "workspace")
        monkeypatch.setenv("QUANT_WORK_DIR", work_dir)

        exec_dir = tmp_path / "execution"
        exec_dir.mkdir(parents=True, exist_ok=True)

        reports_mod = _load_reports_engine_module()
        ctx = _make_ctx()
        ctx.update_artifact("EXECUTION", str(exec_dir))

        result = reports_mod.run(ctx)

        assert result["success"] is False
        assert "ledger" in result["error"]

    def test_llm_fallback(self, monkeypatch, tmp_path):
        """LLM 不可用时规则模板兜底（不崩溃，报告仍生成）"""
        work_dir = str(tmp_path / "workspace")
        monkeypatch.setenv("QUANT_WORK_DIR", work_dir)
        # 不设置 QUANT_LLM_API_KEY → LLM 不可用

        records = [
            _make_record(side="buy", price=10.0, trade_date="2024-01-15",
                         nav_after=1000000.0, position_after=100),
            _make_record(side="sell", price=12.0, trade_date="2024-03-15",
                         stamp_tax=6.0, commission=6.0,
                         nav_after=1000188.0, position_after=0),
        ]
        exec_dir = _write_ledger(tmp_path, records)

        reports_mod = _load_reports_engine_module()
        ctx = _make_ctx()
        ctx.update_artifact("EXECUTION", exec_dir)

        result = reports_mod.run(ctx)

        assert result["success"] is True
        assert result["metadata"]["llm_status"] in ("skipped", "failed")

        # 验证 HTML 中占位符已被替换为兜底内容
        with open(result["artifact_path"], "r", encoding="utf-8") as f:
            html = f.read()
        assert "LLM_ATTRIBUTION_PLACEHOLDER" not in html
        assert "总体评价" in html or "盈亏" in html

    def test_attribution_report_with_multiple_stocks(self, monkeypatch, tmp_path):
        """多标的交易 → 报告含按标的归因章节"""
        work_dir = str(tmp_path / "workspace")
        monkeypatch.setenv("QUANT_WORK_DIR", work_dir)

        records = [
            _make_record(code="000001.SZ", side="buy", price=10.0,
                         trade_date="2024-01-15", nav_after=1000000.0, position_after=100),
            _make_record(code="000001.SZ", side="sell", price=11.0,
                         trade_date="2024-02-15", stamp_tax=5.5, commission=5.5,
                         nav_after=1000089.0, position_after=0),
            _make_record(code="600000.SH", side="buy", price=20.0,
                         trade_date="2024-01-20", nav_after=1000089.0, position_after=100),
            _make_record(code="600000.SH", side="sell", price=19.0,
                         trade_date="2024-02-20", stamp_tax=9.5, commission=5.0,
                         nav_after=1000084.0, position_after=0),
        ]
        exec_dir = _write_ledger(tmp_path, records)

        reports_mod = _load_reports_engine_module()
        ctx = _make_ctx()
        ctx.update_artifact("EXECUTION", exec_dir)

        result = reports_mod.run(ctx)

        assert result["success"] is True

        with open(result["artifact_path"], "r", encoding="utf-8") as f:
            html = f.read()
        assert "按标的盈亏明细" in html
        assert "000001.SZ" in html
        assert "600000.SH" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
