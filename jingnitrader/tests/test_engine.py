"""
主调度引擎测试

测试 MasterEngine 的各个功能
"""
import pytest
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.context import Context
from engine import MasterEngine, run, STAGES, STAGE_ORDER, SKILL_MODULES


class TestMasterEngine:
    """测试 MasterEngine 类"""

    def test_init(self):
        """测试引擎初始化"""
        engine = MasterEngine()
        assert engine is not None
        assert engine.ctx is None
        assert engine._loaded_skills == {}

    def test_parse_intent_basic(self):
        """测试基本意图解析"""
        engine = MasterEngine()
        ctx = engine.parse_intent("帮我做一个回测")

        assert ctx is not None
        assert ctx.user_intent == "帮我做一个回测"
        assert ctx.task_id is not None
        assert ctx.current_stage == "IDLE"

    def test_parse_intent_with_stock_pool(self):
        """测试股票池解析"""
        engine = MasterEngine()
        ctx = engine.parse_intent("用沪深300做一个回测")

        assert "000300.SH" in ctx.stock_pool
        assert "BACKTEST" in ctx.target_stages

    def test_parse_intent_with_date_range(self):
        """测试时间范围解析"""
        engine = MasterEngine()
        ctx = engine.parse_intent("用近3年数据做一个回测")

        assert ctx.start_date == "2021-01-01"
        assert ctx.end_date == "2024-12-31"

    def test_parse_intent_full_pipeline(self):
        """测试完整流程意图解析"""
        engine = MasterEngine()
        ctx = engine.parse_intent("帮我用近3年A股数据做一个20日反转因子选股回测")

        assert "DATA" in ctx.target_stages
        assert "FACTOR" in ctx.target_stages
        assert "MODEL" in ctx.target_stages
        assert "BACKTEST" in ctx.target_stages
        assert "REPORT" in ctx.target_stages

    def test_parse_intent_empty_keywords(self):
        """测试无关键词时的默认流程"""
        engine = MasterEngine()
        ctx = engine.parse_intent("随便")

        # 默认应该有完整流程
        assert len(ctx.target_stages) > 0

    def test_parse_intent_single_stage(self):
        """测试单个阶段意图"""
        engine = MasterEngine()
        ctx = engine.parse_intent("获取数据")

        assert ctx.target_stages == ["DATA"]

    def test_generate_summary(self):
        """测试摘要生成"""
        engine = MasterEngine()
        ctx = Context(
            task_id="test_001",
            user_intent="测试意图",
            current_stage="IDLE",
            target_stages=["DATA", "BACKTEST"]
        )
        ctx.update_artifact("DATA", "/path/to/data.parquet")
        engine.ctx = ctx

        summary = engine._generate_summary()

        assert "test_001" in summary
        assert "测试意图" in summary
        assert "DATA" in summary
        assert "/path/to/data.parquet" in summary

    def test_generate_summary_with_errors(self):
        """测试带错误的摘要生成"""
        engine = MasterEngine()
        ctx = Context(
            task_id="test_001",
            user_intent="测试意图",
            current_stage="IDLE",
            target_stages=["DATA"]
        )
        ctx.add_error("数据获取失败")
        engine.ctx = ctx

        summary = engine._generate_summary()

        assert "错误" in summary
        assert "数据获取失败" in summary


class TestContext:
    """测试 Context 类"""

    def test_context_init(self):
        """测试 Context 初始化"""
        ctx = Context(
            task_id="test_001",
            user_intent="测试",
            current_stage="IDLE"
        )

        assert ctx.task_id == "test_001"
        assert ctx.user_intent == "测试"
        assert ctx.current_stage == "IDLE"
        assert ctx.stock_pool == []
        assert ctx.artifacts == {}
        assert ctx.metadata == {}
        assert ctx.errors == []

    def test_update_artifact(self):
        """测试产物更新"""
        ctx = Context(
            task_id="test_001",
            user_intent="测试",
            current_stage="IDLE"
        )

        ctx.update_artifact("DATA", "/path/to/data.parquet")

        assert "DATA" in ctx.artifacts
        assert ctx.artifacts["DATA"] == "/path/to/data.parquet"

    def test_add_error(self):
        """测试错误添加"""
        ctx = Context(
            task_id="test_001",
            user_intent="测试",
            current_stage="IDLE"
        )

        ctx.add_error("测试错误")

        assert len(ctx.errors) == 1
        assert ctx.errors[0] == "测试错误"

    def test_to_dict(self):
        """测试字典转换"""
        ctx = Context(
            task_id="test_001",
            user_intent="测试",
            current_stage="IDLE"
        )
        ctx.update_artifact("DATA", "/path/to/data.parquet")

        data = ctx.to_dict()

        assert data["task_id"] == "test_001"
        assert data["artifacts"]["DATA"] == "/path/to/data.parquet"

    def test_from_json(self):
        """测试 JSON 反序列化"""
        import json

        ctx = Context(
            task_id="test_001",
            user_intent="测试",
            current_stage="IDLE"
        )

        json_str = json.dumps(ctx.to_dict())
        ctx2 = Context.from_json(json_str)

        assert ctx2.task_id == ctx.task_id
        assert ctx2.user_intent == ctx.user_intent
        assert ctx2.current_stage == ctx.current_stage


class TestRunFunction:
    """测试 run() 函数"""

    def test_run_with_user_input(self):
        """测试使用用户输入运行"""
        result = run(user_input="测试")

        assert "success" in result
        assert "completed_stages" in result
        assert "failed_stages" in result
        assert "summary" in result
        assert "context" in result

    def test_run_with_context(self):
        """测试使用 Context 运行"""
        ctx = Context(
            task_id="test_001",
            user_intent="测试",
            current_stage="IDLE",
            target_stages=["DATA"]
        )

        result = run(ctx=ctx)

        assert "success" in result
        assert result["context"]["task_id"] == "test_001"

    def test_run_with_both_inputs(self):
        """测试同时提供用户输入和 Context"""
        ctx = Context(
            task_id="test_001",
            user_intent="原意图",
            current_stage="IDLE",
            target_stages=["DATA"]
        )

        result = run(ctx=ctx, user_input="新意图")

        # 应该使用 Context 中的意图
        assert result["context"]["user_intent"] == "原意图"


class TestConstants:
    """测试常量定义"""

    def test_stages_defined(self):
        """测试阶段列表已定义"""
        assert "IDLE" in STAGES
        assert "DATA" in STAGES
        assert "FACTOR" in STAGES
        assert "MODEL" in STAGES
        assert "BACKTEST" in STAGES
        assert "PORTFOLIO" in STAGES
        assert "EXECUTION" in STAGES
        assert "REPORT" in STAGES

    def test_stage_order_defined(self):
        """测试阶段顺序已定义"""
        assert STAGE_ORDER["DATA"] < STAGE_ORDER["FACTOR"]
        assert STAGE_ORDER["FACTOR"] < STAGE_ORDER["MODEL"]
        assert STAGE_ORDER["MODEL"] < STAGE_ORDER["BACKTEST"]

    def test_skill_modules_defined(self):
        """测试子 Skill 模块映射已定义"""
        assert "DATA" in SKILL_MODULES
        assert "FACTOR" in SKILL_MODULES
        assert "MODEL" in SKILL_MODULES
        assert "BACKTEST" in SKILL_MODULES
        assert "REPORT" in SKILL_MODULES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
