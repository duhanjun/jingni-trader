"""P1-4 Pydantic V2 Schema 全链路强校验测试

覆盖（PRD P1-4.7）：
- 每个 schema 的合法/非法用例
- extra 字段 raise
- 字段约束 violation
- code 格式校验
- Literal 校验
- Context Pydantic V2 兼容接口
- validate_payload / safe_validate_payload 辅助函数
"""
from __future__ import annotations

import pytest
from datetime import datetime
from pydantic import ValidationError


# ============================================================================
# OrderIntentV1
# ============================================================================

class TestOrderIntentV1:
    def test_valid_buy_order(self):
        from scripts.schemas import OrderIntentV1
        order = OrderIntentV1(code="000001.SZ", side="buy", shares=200, price=12.34)
        assert order.code == "000001.SZ"
        assert order.side == "buy"
        assert order.shares == 200
        assert order.order_type == "limit"  # 默认值

    def test_extra_field_forbid(self):
        from scripts.schemas import OrderIntentV1
        with pytest.raises(ValidationError):
            OrderIntentV1(
                code="000001.SZ", side="buy", shares=200, price=12.34,
                typo_field="xxx",  # 多余字段
            )

    def test_invalid_code_format(self):
        from scripts.schemas import OrderIntentV1
        with pytest.raises(ValidationError):
            OrderIntentV1(code="000001", side="buy", shares=200, price=12.34)

    def test_invalid_code_suffix(self):
        from scripts.schemas import OrderIntentV1
        with pytest.raises(ValidationError):
            OrderIntentV1(code="000001.US", side="buy", shares=200, price=12.34)

    def test_invalid_side_literal(self):
        from scripts.schemas import OrderIntentV1
        with pytest.raises(ValidationError):
            OrderIntentV1(code="000001.SZ", side="hold", shares=200, price=12.34)

    def test_shares_must_be_multiple_of_100(self):
        from scripts.schemas import OrderIntentV1
        with pytest.raises(ValidationError):
            OrderIntentV1(code="000001.SZ", side="buy", shares=150, price=12.34)

    def test_shares_negative_invalid(self):
        from scripts.schemas import OrderIntentV1
        with pytest.raises(ValidationError):
            OrderIntentV1(code="000001.SZ", side="buy", shares=-100, price=12.34)

    def test_price_must_be_positive(self):
        from scripts.schemas import OrderIntentV1
        with pytest.raises(ValidationError):
            OrderIntentV1(code="000001.SZ", side="buy", shares=200, price=0)


# ============================================================================
# ExecutionReportV1
# ============================================================================

class TestExecutionReportV1:
    def test_valid_report(self):
        from scripts.schemas import ExecutionReportV1
        report = ExecutionReportV1(
            execution_id="exec_001",
            trade_date="2026-08-02",
            nav_after=100000.0,
            cash_after=50000.0,
            positions_after={"000001.SZ": 200},
            verdict="confirmed",
            created_at=datetime.now(),
        )
        assert report.version == "ExecutionReportV1"
        assert report.verdict == "confirmed"

    def test_extra_field_forbid(self):
        from scripts.schemas import ExecutionReportV1
        with pytest.raises(ValidationError):
            ExecutionReportV1(
                execution_id="exec_001",
                trade_date="2026-08-02",
                nav_after=100000.0,
                cash_after=50000.0,
                positions_after={},
                verdict="confirmed",
                created_at=datetime.now(),
                extra_field="bad",
            )

    def test_invalid_verdict_literal(self):
        from scripts.schemas import ExecutionReportV1
        with pytest.raises(ValidationError):
            ExecutionReportV1(
                execution_id="exec_001",
                trade_date="2026-08-02",
                nav_after=100000.0,
                cash_after=50000.0,
                positions_after={},
                verdict="pending",  # 非法值
                created_at=datetime.now(),
            )


# ============================================================================
# BacktestResultV1 + VerdictV1
# ============================================================================

class TestBacktestResultV1:
    def test_valid_with_verdict(self):
        from scripts.schemas import BacktestResultV1, VerdictV1
        verdict = VerdictV1(
            recommended_state="candidate",
            passed_gates=["sharpe", "calmar"],
            failed_gates=[],
        )
        result = BacktestResultV1(
            metrics={"sharpe_ratio": 1.2},
            backend="native",
            verdict=verdict,
            trade_count=120,
        )
        assert result.version == "BacktestResultV1"
        assert result.verdict.recommended_state == "candidate"

    def test_valid_without_verdict(self):
        from scripts.schemas import BacktestResultV1
        result = BacktestResultV1(metrics={}, trade_count=0)
        assert result.verdict is None

    def test_extra_field_forbid(self):
        from scripts.schemas import BacktestResultV1
        with pytest.raises(ValidationError):
            BacktestResultV1(metrics={}, trade_count=0, unknown="x")

    def test_trade_count_negative_invalid(self):
        from scripts.schemas import BacktestResultV1
        with pytest.raises(ValidationError):
            BacktestResultV1(metrics={}, trade_count=-1)


# ============================================================================
# CleanedDataV1 / FactorDataV1 / ReportV1
# ============================================================================

class TestDataSchemas:
    def test_cleaned_data_valid(self):
        from scripts.schemas import CleanedDataV1
        data = CleanedDataV1(path="/tmp/x.parquet", rows=100, columns=["code", "date"])
        assert data.version == "CleanedDataV1"
        assert data.quality_mode == "normal"

    def test_cleaned_data_invalid_quality_mode(self):
        from scripts.schemas import CleanedDataV1
        with pytest.raises(ValidationError):
            CleanedDataV1(path="/tmp/x", rows=0, quality_mode="bad")

    def test_factor_data_valid(self):
        from scripts.schemas import FactorDataV1
        f = FactorDataV1(path="/tmp/f.parquet", factor_names=["alpha1"], rows=50)
        assert f.version == "FactorDataV1"

    def test_report_valid(self):
        from scripts.schemas import ReportV1
        r = ReportV1(path="/tmp/r.html", template="technical")
        assert r.template == "technical"

    def test_report_invalid_template(self):
        from scripts.schemas import ReportV1
        with pytest.raises(ValidationError):
            ReportV1(path="/tmp/r.html", template="quant")


# ============================================================================
# Context Pydantic V2 兼容接口
# ============================================================================

class TestContextPydantic:
    def test_context_roundtrip_json(self):
        from scripts.context import Context
        ctx = Context(
            task_id="t1",
            user_intent="测试",
            target_stages=["DATA", "REPORT"],
            stock_pool=["000001.SZ"],
        )
        ctx.update_artifact("DATA", "/tmp/x.parquet")
        ctx.add_error("test error")

        s = ctx.to_json()
        ctx2 = Context.from_json(s)
        assert ctx2.task_id == "t1"
        assert ctx2.target_stages == ["DATA", "REPORT"]
        assert ctx2.stock_pool == ["000001.SZ"]
        assert ctx2.artifacts == {"DATA": "/tmp/x.parquet"}
        assert ctx2.errors == ["test error"]

    def test_context_get_artifact_missing(self):
        from scripts.context import Context
        ctx = Context()
        assert ctx.get_artifact("NOT_EXIST") is None

    def test_context_extra_ignored(self):
        """Context extra='ignore'，多余字段不报错（向后兼容）"""
        from scripts.context import Context
        ctx = Context(task_id="t1", unknown_field="ignored")
        assert ctx.task_id == "t1"

    def test_context_from_dict_filters_unknown(self):
        from scripts.context import Context
        ctx = Context.from_dict({
            "task_id": "t2",
            "unknown_field": "filtered",
            "stock_pool": ["000300.SH"],
        })
        assert ctx.task_id == "t2"
        assert ctx.stock_pool == ["000300.SH"]

    def test_context_metadata_mutable(self):
        """metadata 字典可变，支持 ctx.metadata[key]=val 赋值"""
        from scripts.context import Context
        ctx = Context()
        ctx.metadata["strategy_required"] = True
        assert ctx.metadata["strategy_required"] is True


# ============================================================================
# 校验辅助函数
# ============================================================================

class TestValidatePayload:
    def test_validate_payload_valid(self):
        from scripts.schemas import validate_payload, OrderIntentV1
        payload = {"code": "000001.SZ", "side": "buy", "shares": 200, "price": 10.0}
        is_valid, err = validate_payload(payload, OrderIntentV1)
        assert is_valid is True
        assert err is None

    def test_validate_payload_invalid(self):
        from scripts.schemas import validate_payload, OrderIntentV1
        payload = {"code": "BAD", "side": "buy", "shares": 200, "price": 10.0}
        is_valid, err = validate_payload(payload, OrderIntentV1)
        assert is_valid is False
        assert err is not None

    def test_validate_payload_none(self):
        from scripts.schemas import validate_payload, OrderIntentV1
        is_valid, err = validate_payload(None, OrderIntentV1)
        assert is_valid is False

    def test_safe_validate_payload_returns_original(self):
        """safe_validate_payload 校验失败也返回原始 payload"""
        from scripts.schemas import safe_validate_payload, OrderIntentV1
        payload = {"code": "BAD", "side": "buy", "shares": 200, "price": 10.0}
        result = safe_validate_payload(payload, OrderIntentV1, stage="TEST")
        assert result is payload  # 返回原对象


# ============================================================================
# STAGE_SCHEMA_MAP 映射
# ============================================================================

class TestStageSchemaMap:
    def test_all_stages_mapped(self):
        from scripts.schemas import STAGE_SCHEMA_MAP
        for stage in ["DATA", "FACTOR", "BACKTEST", "EXECUTION", "REPORT"]:
            assert stage in STAGE_SCHEMA_MAP


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
