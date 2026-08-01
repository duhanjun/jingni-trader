"""MasterEngine._is_analysis_intent 分析意图路由测试。

覆盖：
- ANALYSIS_KEYWORDS / QUANT_KEYWORDS 常量存在性与关键字集合
- _is_analysis_intent() 对各类输入的判定（含 mixed 意图优先走量化路径）
- 分析意图触发时 target_stages = ["DATA", "FACTOR", "REPORT"]
- 分析意图触发时 ctx.metadata["report_template"] 被正确设置
"""
from __future__ import annotations

import pytest


# ============================================================================
# Part 1: 关键字常量契约
# ============================================================================

class TestAnalysisKeywordsConstant:
    """验证 ANALYSIS_KEYWORDS 常量存在并包含预期关键字。"""

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_constant_exists(self):
        """ANALYSIS_KEYWORDS 在 engine 模块中可访问"""
        import engine
        assert hasattr(engine, "ANALYSIS_KEYWORDS")
        assert isinstance(engine.ANALYSIS_KEYWORDS, (set, list, tuple))

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_contains_expected_keywords(self):
        """ANALYSIS_KEYWORDS 包含个人投资者分析常用关键字"""
        import engine
        expected = {
            "分析", "技术面", "基本面", "K线", "形态",
            "MACD", "RSI", "KDJ", "均线", "PE", "PB", "ROE",
        }
        missing = expected - set(engine.ANALYSIS_KEYWORDS)
        assert not missing, f"ANALYSIS_KEYWORDS 缺少: {missing}"


class TestQuantKeywordsConstant:
    """验证 QUANT_KEYWORDS 常量存在并包含预期关键字。"""

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_constant_exists(self):
        """QUANT_KEYWORDS 在 engine 模块中可访问"""
        import engine
        assert hasattr(engine, "QUANT_KEYWORDS")
        assert isinstance(engine.QUANT_KEYWORDS, (set, list, tuple))

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_contains_expected_keywords(self):
        """QUANT_KEYWORDS 包含量化交易常用关键字"""
        import engine
        expected = {"回测", "因子", "策略", "模型", "组合", "实盘", "选股"}
        missing = expected - set(engine.QUANT_KEYWORDS)
        assert not missing, f"QUANT_KEYWORDS 缺少: {missing}"


# ============================================================================
# Part 2: _is_analysis_intent 判定逻辑
# ============================================================================

class TestIsAnalysisIntent:
    """验证 _is_analysis_intent 对各种输入的判定。"""

    def _make_engine(self):
        import engine
        return engine.MasterEngine()

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_true_for_analyze_stock(self):
        """'分析一下000001.SZ' → True（命中 '分析'）"""
        master = self._make_engine()
        assert master._is_analysis_intent("分析一下000001.SZ") is True

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_true_for_technical_analysis(self):
        """'000001.SZ技术面怎么样' → True（命中 '技术面'/'怎么样'）"""
        master = self._make_engine()
        assert master._is_analysis_intent("000001.SZ技术面怎么样") is True

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_true_for_kline_pattern(self):
        """'帮我看看K线形态' → True（命中 'K线'/'形态'）"""
        master = self._make_engine()
        assert master._is_analysis_intent("帮我看看K线形态") is True

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_true_for_fundamental_analysis(self):
        """'茅台的基本面分析' → True（命中 '基本面'/'分析'）"""
        master = self._make_engine()
        assert master._is_analysis_intent("茅台的基本面分析") is True

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_backtest(self):
        """'帮我做回测' → False（命中 '回测' 量化关键字）"""
        master = self._make_engine()
        assert master._is_analysis_intent("帮我做回测") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_model_training(self):
        """'训练模型选股' → False（命中 '模型'/'选股' 量化关键字）"""
        master = self._make_engine()
        assert master._is_analysis_intent("训练模型选股") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_portfolio_optimization(self):
        """'优化组合' → False（命中 '组合' 量化关键字）"""
        master = self._make_engine()
        assert master._is_analysis_intent("优化组合") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_mixed_intent_quant_wins(self):
        """'分析一下这个因子回测效果' → False（量化关键字优先于分析）"""
        master = self._make_engine()
        assert master._is_analysis_intent("分析一下这个因子回测效果") is False

    @pytest.mark.skill_master
    @pytest.mark.unit
    def test_false_for_empty_string(self):
        """空字符串 → False"""
        master = self._make_engine()
        assert master._is_analysis_intent("") is False


# ============================================================================
# Part 3: parse_intent 在分析意图路径下的副作用
# ============================================================================

class TestAnalysisIntentRouting:
    """验证 parse_intent 在识别到分析意图时正确设置 target_stages 与 metadata。"""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])