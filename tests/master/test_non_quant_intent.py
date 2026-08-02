"""MasterEngine._is_strategy_required 单一工作流路由测试。

新单一工作流模型：根据用户是否明确需要构建策略选择执行深度。
- strategy_required=False（默认）→ DATA → FACTOR → REPORT，仅出分析报告
- strategy_required=True（用户明确要求）→ 完整 7 阶段管线

覆盖：
- STRATEGY_KEYWORDS 常量存在性与关键字集合
- _is_strategy_required() 对各类输入的判定
- 分析路径下 target_stages = ["DATA", "FACTOR", "REPORT"]
- 分析路径下 ctx.metadata["report_template"] 被正确设置
- ctx.metadata["strategy_required"] 布尔标志正确传递
"""
from __future__ import annotations

import pytest


# ============================================================================
# Part 1: 关键字常量契约
# ============================================================================

class TestStrategyKeywordsConstant:
    """验证 STRATEGY_KEYWORDS 常量存在并包含预期关键字。"""

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_constant_exists(self):
        """STRATEGY_KEYWORDS 在 engine 模块中可访问"""
        import engine
        assert hasattr(engine, "STRATEGY_KEYWORDS")
        assert isinstance(engine.STRATEGY_KEYWORDS, (set, list, tuple))

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_contains_expected_keywords(self):
        """STRATEGY_KEYWORDS 包含策略构建动作关键字"""
        import engine
        expected = {"回测", "策略", "模型", "组合", "实盘", "选股"}
        missing = expected - set(engine.STRATEGY_KEYWORDS)
        assert not missing, f"STRATEGY_KEYWORDS 缺少: {missing}"

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_excludes_factor_analysis_keywords(self):
        """STRATEGY_KEYWORDS 不应包含因子/分析类关键词（这些是两条路径共用的）"""
        import engine
        # '因子'、'alpha'、'ic' 属于因子分析范畴，不应触发策略路径
        non_strategy = {"因子", "alpha", "ic"}
        leaked = non_strategy & set(engine.STRATEGY_KEYWORDS)
        assert not leaked, f"STRATEGY_KEYWORDS 不应包含分析类关键词: {leaked}"


# ============================================================================
# Part 2: _is_strategy_required 判定逻辑
# ============================================================================

class TestIsStrategyRequired:
    """验证 _is_strategy_required 对各种输入的判定。"""

    def _make_engine(self):
        import engine
        return engine.MasterEngine()

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_analyze_stock(self):
        """'分析一下000001.SZ' → False（分析意图，不构建策略）"""
        master = self._make_engine()
        assert master._is_strategy_required("分析一下000001.SZ") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_technical_analysis(self):
        """'000001.SZ技术面怎么样' → False"""
        master = self._make_engine()
        assert master._is_strategy_required("000001.SZ技术面怎么样") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_kline_pattern(self):
        """'帮我看看K线形态' → False"""
        master = self._make_engine()
        assert master._is_strategy_required("帮我看看K线形态") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_fundamental_analysis(self):
        """'茅台的基本面分析' → False"""
        master = self._make_engine()
        assert master._is_strategy_required("茅台的基本面分析") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_factor_only(self):
        """'计算这只股的因子' → False（因子计算是共用前置步骤，不构成策略构建）"""
        master = self._make_engine()
        assert master._is_strategy_required("计算这只股的因子") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_true_for_backtest(self):
        """'帮我做回测' → True（明确要求回测）"""
        master = self._make_engine()
        assert master._is_strategy_required("帮我做回测") is True

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_true_for_model_training(self):
        """'训练模型选股' → True（命中 '模型'/'选股'）"""
        master = self._make_engine()
        assert master._is_strategy_required("训练模型选股") is True

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_true_for_portfolio_optimization(self):
        """'优化组合' → True（命中 '组合'）"""
        master = self._make_engine()
        assert master._is_strategy_required("优化组合") is True

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_true_for_live_trading(self):
        """'实盘下单' → True（命中 '实盘'/'下单'）"""
        master = self._make_engine()
        assert master._is_strategy_required("实盘下单") is True

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_true_for_risk_control(self):
        """'做风控' → True（命中 '风控'）"""
        master = self._make_engine()
        assert master._is_strategy_required("做风控") is True

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_empty_string(self):
        """空字符串 → False（默认走分析路径）"""
        master = self._make_engine()
        assert master._is_strategy_required("") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_none(self):
        """None 输入 → False"""
        master = self._make_engine()
        assert master._is_strategy_required(None) is False


# ============================================================================
# Part 3: parse_intent 在分析路径下的副作用
# ============================================================================

class TestAnalysisIntentRouting:
    """验证 parse_intent 在分析路径下正确设置 target_stages 与 metadata。"""

    def _make_engine(self):
        import engine
        return engine.MasterEngine()

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_target_stages_set_to_data_factor_report(self):
        """分析意图 → target_stages = ['DATA', 'FACTOR', 'REPORT']"""
        master = self._make_engine()
        ctx = master.parse_intent("分析一下000001.SZ")
        assert ctx.target_stages == ["DATA", "FACTOR", "REPORT"]

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_strategy_required_flag_false(self):
        """分析意图 → ctx.metadata['strategy_required'] = False"""
        master = self._make_engine()
        ctx = master.parse_intent("分析一下000001.SZ")
        assert ctx.metadata["strategy_required"] is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_report_template_metadata_set(self):
        """分析意图 → ctx.metadata['report_template'] 被正确设置"""
        master = self._make_engine()
        ctx = master.parse_intent("茅台的基本面分析")
        assert "report_template" in ctx.metadata
        assert ctx.metadata["report_template"] in ("technical", "fundamental", "both")

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_technical_intent_sets_template(self):
        """技术面意图 → report_template = 'technical'"""
        master = self._make_engine()
        ctx = master.parse_intent("000001.SZ技术面怎么样")
        assert ctx.metadata.get("report_template") == "technical"

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_fundamental_intent_sets_template(self):
        """基本面意图 → report_template = 'fundamental'"""
        master = self._make_engine()
        ctx = master.parse_intent("茅台的基本面分析")
        assert ctx.metadata.get("report_template") == "fundamental"

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_strategy_path_sets_flag_true(self):
        """策略意图 → ctx.metadata['strategy_required'] = True 且走完整管线"""
        master = self._make_engine()
        ctx = master.parse_intent("帮我做回测")
        assert ctx.metadata["strategy_required"] is True
        assert ctx.target_stages == [
            "DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
