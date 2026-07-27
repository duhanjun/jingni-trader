"""MasterEngine.parse_intent 意图解析测试。

来源：合并原 test_jingni_datafeed_integration.py::TestParseIntentGate（7 用例）
与原 test_system_smoke.py::TestParseIntentKeywords（11 用例），共 18 用例。

覆盖：
- factor_source gate 三态（local/auto/jingni）
- 关键词识别（中文/英文/混合/边界）
- stock_pool 与 date_range 解析
"""
from __future__ import annotations

import pytest


# ============================================================================
# Part 1: factor_source gate 三态逻辑（原 TestParseIntentGate）
# ============================================================================

class TestParseIntentGate:
    """验证 MasterEngine.parse_intent 正确设置 ctx.metadata['factor_source']。"""

    def test_local_when_env_not_configured(self, monkeypatch):
        """未配置 JINGNI_URL/TOKEN → factor_source='local'"""
        monkeypatch.delenv("JINGNI_URL", raising=False)
        monkeypatch.delenv("JINGNI_TOKEN", raising=False)

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年 momentum 因子做回测")

        assert ctx.metadata["factor_source"] == "local"

    def test_auto_when_configured_but_not_requested(self, monkeypatch):
        """配置了凭证但用户没明确要求因子库 → factor_source='auto'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年因子做回测")

        assert ctx.metadata["factor_source"] == "auto"
        assert "FACTOR" in ctx.target_stages

    def test_jingni_when_explicitly_requested_chinese(self, monkeypatch):
        """配置了凭证 + 用户明确要求从因子库取数（中文）→ factor_source='jingni'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年因子做回测，从惊泥因子库取数")

        assert ctx.metadata["factor_source"] == "jingni"

    def test_jingni_when_explicitly_requested_english(self, monkeypatch):
        """配置了凭证 + 英文关键字 jingni → factor_source='jingni'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("use jingni factor store for backtest")

        assert ctx.metadata["factor_source"] == "jingni"

    def test_jingni_when_factor_store_keyword(self, monkeypatch):
        """factor_store 关键字也能触发"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("从 factor-store 取因子做回测")

        assert ctx.metadata["factor_source"] == "jingni"

    def test_local_when_no_factor_stage(self, monkeypatch):
        """没有 FACTOR 阶段时，factor_source 保持 'local'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.setenv("JINGNI_TOKEN", "gsa_test_token")

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("获取近3年数据")  # 只有 DATA 阶段

        assert ctx.metadata["factor_source"] == "local"
        assert "FACTOR" not in ctx.target_stages

    def test_auto_when_only_url_configured(self, monkeypatch):
        """只配了 JINGNI_URL 但没配 TOKEN → 视为未配置，factor_source='local'"""
        monkeypatch.setenv("JINGNI_URL", "https://jingni.example.com")
        monkeypatch.delenv("JINGNI_TOKEN", raising=False)

        import engine
        master = engine.MasterEngine()
        ctx = master.parse_intent("用近3年因子做回测")

        assert ctx.metadata["factor_source"] == "local"


# ============================================================================
# Part 2: parse_intent 关键词识别（原 TestParseIntentKeywords）
# ============================================================================

class TestParseIntentKeywords:
    """验证 parse_intent 对不同关键词的识别能力。"""

    def _make_engine(self):
        import engine
        return engine.MasterEngine()

    def test_full_pipeline_keywords(self):
        """一句话包含全流程关键词 → 7 个阶段全识别（按 STAGE_ORDER 排序）"""
        ctx = self._make_engine().parse_intent(
            "获取近3年A股数据 做因子分析 训练模型 回测验证 组合优化 实盘执行 生成绩效报告"
        )
        assert ctx.target_stages == ["DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"]

    def test_english_keywords(self):
        """英文关键词也应识别"""
        ctx = self._make_engine().parse_intent("run backtest with factor data")
        # 包含 factor + backtest + data (含 data 关键字)
        assert "FACTOR" in ctx.target_stages
        assert "BACKTEST" in ctx.target_stages
        assert "DATA" in ctx.target_stages  # 'data' 命中

    def test_model_keyword_lightgbm(self):
        """lightgbm 关键词识别为 MODEL"""
        ctx = self._make_engine().parse_intent("用 lightgbm 训练模型")
        assert "MODEL" in ctx.target_stages
        assert "DATA" in ctx.target_stages  # 自动补 DATA

    def test_execution_keyword(self):
        """实盘下单关键词识别为 EXECUTION"""
        ctx = self._make_engine().parse_intent("实盘下单 买入 100 股")
        assert "EXECUTION" in ctx.target_stages

    def test_portfolio_keyword(self):
        """组合优化关键词识别为 PORTFOLIO"""
        ctx = self._make_engine().parse_intent("做组合优化和风控")
        assert "PORTFOLIO" in ctx.target_stages

    def test_no_keyword_fallback(self):
        """无任何已知关键词 → 默认 DATA→FACTOR→MODEL→BACKTEST→REPORT"""
        ctx = self._make_engine().parse_intent("今天天气真好")
        assert ctx.target_stages == ["DATA", "FACTOR", "MODEL", "BACKTEST", "REPORT"]

    def test_stock_pool_csi300(self):
        """'沪深300' → stock_pool 含 000300.SH"""
        ctx = self._make_engine().parse_intent("用沪深300回测")
        assert ctx.stock_pool == ["000300.SH"]

    def test_stock_pool_csi500(self):
        """'中证500' → stock_pool 含 000905.SH"""
        ctx = self._make_engine().parse_intent("用中证500回测")
        assert ctx.stock_pool == ["000905.SH"]

    def test_stock_pool_full_market(self):
        """'全A' → stock_pool 为空"""
        ctx = self._make_engine().parse_intent("全A回测")
        assert ctx.stock_pool == []

    def test_date_range_3y(self):
        """'近3年' → 时间范围正确"""
        ctx = self._make_engine().parse_intent("近3年回测")
        assert ctx.start_date == "2021-01-01"
        assert ctx.end_date == "2024-12-31"

    def test_date_range_5y(self):
        """'近5年' → 时间范围正确"""
        ctx = self._make_engine().parse_intent("近5年回测")
        assert ctx.start_date == "2019-01-01"
        assert ctx.end_date == "2024-12-31"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
