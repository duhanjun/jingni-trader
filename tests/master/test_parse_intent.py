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
        """无任何已知关键词 → 默认走分析路径 DATA→FACTOR→REPORT

        新单一工作流模型：默认不构建策略，仅做分析。用户明确要求回测/策略/实盘
        等动作时才升级到完整 7 阶段管线。
        """
        ctx = self._make_engine().parse_intent("今天天气真好")
        assert ctx.target_stages == ["DATA", "FACTOR", "REPORT"]
        assert ctx.metadata["strategy_required"] is False

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
        """'近3年' → 时间范围为从今天起往前3年"""
        from datetime import date as _date
        ctx = self._make_engine().parse_intent("近3年回测")
        today = _date.today()
        expected_start = today.replace(year=today.year - 3).strftime("%Y-%m-%d")
        assert ctx.start_date == expected_start
        assert ctx.end_date == today.strftime("%Y-%m-%d")

    def test_date_range_5y(self):
        """'近5年' → 时间范围为从今天起往前5年"""
        from datetime import date as _date
        ctx = self._make_engine().parse_intent("近5年回测")
        today = _date.today()
        expected_start = today.replace(year=today.year - 5).strftime("%Y-%m-%d")
        assert ctx.start_date == expected_start
        assert ctx.end_date == today.strftime("%Y-%m-%d")

    def test_date_range_default_5y(self):
        """未指定时间 → 默认取最近5年"""
        from datetime import date as _date
        ctx = self._make_engine().parse_intent("回测")
        today = _date.today()
        expected_start = today.replace(year=today.year - 5).strftime("%Y-%m-%d")
        assert ctx.start_date == expected_start
        assert ctx.end_date == today.strftime("%Y-%m-%d")


# ============================================================================
# Part 3: 数据源优先级意图解析（方案 D）
# ============================================================================

class TestParseDataSourcesIntent:
    """验证 MasterEngine._parse_data_sources_intent 与 parse_intent 的数据源优先级解析。

    方案 D 契约：
    - 用户明确指定数据源 → ctx.data_sources 非空，覆盖环境变量 DATA_BACKENDS
    - 未指定 → ctx.data_sources 保持 None，由 data-engine 走环境变量 → 默认值
    """

    def _make_engine(self):
        import engine
        return engine.MasterEngine()

    def test_single_source_wind(self):
        """'用 wind 取数据' → ['wind', ...默认链兜底(不含 tushare)]"""
        ctx = self._make_engine().parse_intent("用 wind 取数据")
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "wind"
        # 用户只指定了 wind，后面应自动追加默认免费降级链（baostock/akshare/websearch）
        assert "baostock" in ctx.data_sources
        assert "websearch" in ctx.data_sources
        # tushare 是 opt-in 源，不在默认兜底链里，用户没明确说就不应出现
        assert "tushare" not in ctx.data_sources

    def test_single_source_ifind(self):
        """'优先用 ifind' → ['ifind', ...默认链兜底]"""
        ctx = self._make_engine().parse_intent("优先用 ifind")
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "ifind"

    def test_chinese_name_wind(self):
        """'用万得取数据' → 万得 → wind"""
        ctx = self._make_engine().parse_intent("用万得取数据")
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "wind"

    def test_chinese_name_ifind(self):
        """'用同花顺取数据' → 同花顺 → ifind"""
        ctx = self._make_engine().parse_intent("用同花顺取数据")
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "ifind"

    def test_multiple_sources_in_order(self):
        """'优先用 ifind，失败用 tushare' → 顺序为 ['ifind', 'tushare', ...]"""
        ctx = self._make_engine().parse_intent("优先用 ifind，失败用 tushare")
        assert ctx.data_sources is not None
        # ifind 在 tushare 之前出现
        assert ctx.data_sources.index("ifind") < ctx.data_sources.index("tushare")
        assert ctx.data_sources[0] == "ifind"

    def test_multiple_sources_no_dup(self):
        """'用 baostock 和 akshare' → 去重后顺序为 ['baostock', 'akshare', ...]"""
        ctx = self._make_engine().parse_intent("用 baostock 和 akshare")
        assert ctx.data_sources is not None
        assert ctx.data_sources.count("baostock") == 1
        assert ctx.data_sources.count("akshare") == 1
        assert ctx.data_sources.index("baostock") < ctx.data_sources.index("akshare")

    def test_no_verb_no_match(self):
        """'今天天气真好' → 无动作动词 → None"""
        ctx = self._make_engine().parse_intent("今天天气真好")
        assert ctx.data_sources is None

    def test_verb_but_no_source_name(self):
        """'用 momentum 因子做回测' → 有'用'但无数据源名 → None"""
        ctx = self._make_engine().parse_intent("用 momentum 因子做回测")
        assert ctx.data_sources is None

    def test_source_name_but_no_verb(self):
        """'wind 数据源说明' → 有数据源名但无动作动词 → None"""
        ctx = self._make_engine().parse_intent("wind 数据源说明")
        assert ctx.data_sources is None

    def test_default_chain_unchanged_when_no_intent(self):
        """未指定数据源 → ctx.data_sources 保持 None（由 data-engine 走环境变量/默认值）"""
        ctx = self._make_engine().parse_intent("用近3年因子做回测")
        assert ctx.data_sources is None

    def test_english_verb_use(self):
        """'use wind for data' → 英文动词 use 也能触发"""
        ctx = self._make_engine().parse_intent("use wind for data")
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "wind"

    def test_switch_verb(self):
        """'切换到 baostock' → 切换动词触发"""
        ctx = self._make_engine().parse_intent("切换到 baostock")
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "baostock"

    def test_tushare_opt_in_explicit(self):
        """'用 tushare 取数据' → tushare 作为 opt-in 源被用户明确启用,进入链首"""
        ctx = self._make_engine().parse_intent("用 tushare 取数据")
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "tushare"
        # tushare 是 opt-in 源,只有用户明确说才进入链
        # 默认兜底链 baostock/akshare/websearch 仍应追加在后面
        assert "baostock" in ctx.data_sources

    def test_default_chain_excludes_opt_in_sources(self):
        """未指定数据源时 ctx.data_sources 为 None,由 data-engine 走默认链(不含 tushare)"""
        ctx = self._make_engine().parse_intent("用近3年因子做回测")
        assert ctx.data_sources is None
        # 验证 data-engine 的默认链确实不含 tushare
        import sys
        import importlib.util as ilu
        saved_scripts = {k: v for k, v in sys.modules.items() if k == "scripts" or k.startswith("scripts.")}
        for k in list(sys.modules.keys()):
            if k == "scripts" or k.startswith("scripts."):
                sys.modules.pop(k, None)
        scripts_dir = r"d:\codebuddy\jingni-trader\skills\data-engine\scripts"
        init_py = scripts_dir + r"\__init__.py"
        try:
            spec = ilu.spec_from_file_location("scripts", init_py, submodule_search_locations=[scripts_dir])
            pkg = ilu.module_from_spec(spec)
            sys.modules["scripts"] = pkg
            spec.loader.exec_module(pkg)
            from scripts.config import DEFAULT_DATA_SOURCES, PAID_OR_SPECIAL_BACKENDS
            assert "tushare" not in DEFAULT_DATA_SOURCES, "tushare 不应在默认链里"
            assert "tushare" in PAID_OR_SPECIAL_BACKENDS, "tushare 应在 opt-in 源列表里"
        finally:
            for k in list(sys.modules.keys()):
                if k == "scripts" or k.startswith("scripts."):
                    sys.modules.pop(k, None)
            for k, v in saved_scripts.items():
                sys.modules[k] = v

    def test_does_not_break_target_stages(self):
        """数据源意图解析不影响 target_stages 的正常解析"""
        ctx = self._make_engine().parse_intent("用 wind 取数据做近3年回测")
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "wind"
        assert "DATA" in ctx.target_stages
        assert "BACKTEST" in ctx.target_stages


# ============================================================================
# Part 4: 单一工作流 + 因子用途分支（strategy_required 契约）
# ============================================================================

class TestStrategyRequiredRouting:
    """验证单一工作流模型：根据 strategy_required 标志选择执行深度。

    契约：
    - strategy_required=False（默认）→ DATA → FACTOR → REPORT，仅出分析报告
    - strategy_required=True（用户明确要求）→ 完整 7 阶段管线，含策略回测绩效报告
    - 因子/分析/技术面/基本面等关键词不触发 strategy_required
    - 回测/策略/模型/组合/实盘/选股/风控/下单等动作关键词触发 strategy_required
    """

    def _make_engine(self):
        import engine
        return engine.MasterEngine()

    def test_default_analysis_path(self):
        """无任何关键词 → 默认分析路径"""
        ctx = self._make_engine().parse_intent("看看比亚迪")
        assert ctx.metadata["strategy_required"] is False
        assert ctx.target_stages == ["DATA", "FACTOR", "REPORT"]

    def test_analysis_keywords_not_trigger_strategy(self):
        """分析类关键词（技术面/基本面/K线/估值等）不触发策略路径"""
        for text in [
            "分析比亚迪技术面",
            "看看这只股的估值",
            "比亚迪怎么样",
            "诊股 002594",
            "查看MACD和KDJ",
        ]:
            ctx = self._make_engine().parse_intent(text)
            assert ctx.metadata["strategy_required"] is False, f"'{text}' 不应触发策略路径"
            assert ctx.target_stages == ["DATA", "FACTOR", "REPORT"]

    def test_factor_keyword_not_trigger_strategy(self):
        """'因子' 关键词本身不触发策略路径（因子计算是两条路径共用前置步骤）"""
        ctx = self._make_engine().parse_intent("计算比亚迪的因子")
        assert ctx.metadata["strategy_required"] is False
        assert ctx.target_stages == ["DATA", "FACTOR", "REPORT"]

    def test_alpha_ic_not_trigger_strategy(self):
        """alpha/ic 关键词不触发策略路径（属于因子分析范畴）"""
        ctx = self._make_engine().parse_intent("看看这只股的 alpha 和 IC")
        assert ctx.metadata["strategy_required"] is False
        assert ctx.target_stages == ["DATA", "FACTOR", "REPORT"]

    def test_backtest_trigger_full_pipeline(self):
        """'回测' 触发完整 7 阶段管线"""
        ctx = self._make_engine().parse_intent("用近3年A股数据做回测")
        assert ctx.metadata["strategy_required"] is True
        assert ctx.target_stages == [
            "DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"
        ]

    def test_strategy_keyword_trigger_full_pipeline(self):
        """'策略' 触发完整管线"""
        ctx = self._make_engine().parse_intent("构建一个选股策略")
        assert ctx.metadata["strategy_required"] is True
        assert "MODEL" in ctx.target_stages
        assert "BACKTEST" in ctx.target_stages

    def test_model_keyword_trigger_full_pipeline(self):
        """'模型' 触发完整管线（训练模型本质上是策略构建）"""
        ctx = self._make_engine().parse_intent("用 lightgbm 训练模型")
        assert ctx.metadata["strategy_required"] is True
        assert "MODEL" in ctx.target_stages

    def test_live_trading_keyword_trigger_full_pipeline(self):
        """'实盘/下单' 触发完整管线（实盘依赖已验证策略）"""
        for text in ["实盘下单 100 股", "执行交易", "启动实盘"]:
            ctx = self._make_engine().parse_intent(text)
            assert ctx.metadata["strategy_required"] is True, f"'{text}' 应触发策略路径"
            assert "EXECUTION" in ctx.target_stages

    def test_report_template_always_set(self):
        """报告模板检测独立于 strategy_required，两条路径都应设置"""
        # 分析路径
        ctx1 = self._make_engine().parse_intent("分析比亚迪技术面")
        assert ctx1.metadata["report_template"] == "technical"
        assert ctx1.metadata["strategy_required"] is False

        # 策略路径也应设置 report_template（虽然 REPORT 阶段会走绩效报告，但字段仍存在）
        ctx2 = self._make_engine().parse_intent("回测比亚迪的技术面策略")
        assert "report_template" in ctx2.metadata
        assert ctx2.metadata["strategy_required"] is True

    def test_data_sources_intent_still_works_in_analysis_path(self):
        """数据源优先级解析在分析路径下也正常工作"""
        ctx = self._make_engine().parse_intent("用 wind 取数据分析比亚迪")
        assert ctx.metadata["strategy_required"] is False
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "wind"
        assert ctx.target_stages == ["DATA", "FACTOR", "REPORT"]

    def test_data_sources_intent_still_works_in_strategy_path(self):
        """数据源优先级解析在策略路径下也正常工作"""
        ctx = self._make_engine().parse_intent("用 wind 取数据做回测")
        assert ctx.metadata["strategy_required"] is True
        assert ctx.data_sources is not None
        assert ctx.data_sources[0] == "wind"
        assert "BACKTEST" in ctx.target_stages


# ============================================================================
# Part 5: 绩效复盘意图解析（attribution intent routing）
# ============================================================================

class TestAttributionIntent:
    """验证 MasterEngine._is_attribution_intent 与 parse_intent 的复盘意图路由。

    契约：
    - 命中 ATTRIBUTION_KEYWORDS → ctx.metadata['report_intent'] == 'attribution'
    - target_stages 为 ['DATA', 'FACTOR', 'EXECUTION', 'REPORT']（按 STAGE_ORDER 排序）
    - 优先级高于策略构建意图：即使输入含 '回测' 等 STRATEGY_KEYWORDS，仍走复盘路径
    - 跳过 MODEL/BACKTEST/PORTFOLIO 阶段
    """

    def _make_engine(self):
        import engine
        return engine.MasterEngine()

    def test_parse_attribution_intent(self):
        """'复盘' 触发绩效复盘意图，target_stages 按 STAGE_ORDER 排序"""
        ctx = self._make_engine().parse_intent("复盘")
        assert ctx.metadata["report_intent"] == "attribution"
        assert ctx.target_stages == ["DATA", "FACTOR", "EXECUTION", "REPORT"]

    def test_parse_attribution_intent_variants(self):
        """多个 ATTRIBUTION_KEYWORDS 关键词均应触发复盘意图"""
        keywords = [
            "绩效归因", "归因分析", "实盘报告",
            "盈亏分析", "交易复盘", "绩效复盘", "attribution",
        ]
        for kw in keywords:
            ctx = self._make_engine().parse_intent(kw)
            assert ctx.metadata.get("report_intent") == "attribution", (
                f"关键词 '{kw}' 应触发 attribution 意图"
            )

    def test_parse_attribution_not_triggered(self):
        """普通输入不应触发复盘意图"""
        for text in ["今天天气真好", "分析比亚迪技术面"]:
            ctx = self._make_engine().parse_intent(text)
            assert ctx.metadata.get("report_intent") != "attribution", (
                f"'{text}' 不应触发 attribution 意图"
            )

    def test_attribution_priority_over_strategy(self):
        """'复盘回测' 同时命中复盘与策略关键词，应优先走复盘路径（不含 MODEL/BACKTEST）"""
        ctx = self._make_engine().parse_intent("复盘回测")
        assert ctx.metadata["report_intent"] == "attribution"
        assert "MODEL" not in ctx.target_stages
        assert "BACKTEST" not in ctx.target_stages

    def test_attribution_target_stages_order(self):
        """复盘路径 target_stages 严格按 STAGE_ORDER 排序"""
        import engine
        ctx = self._make_engine().parse_intent("复盘")
        assert ctx.target_stages == ["DATA", "FACTOR", "EXECUTION", "REPORT"]
        # 显式校验顺序与 STAGE_ORDER 一致
        order = [engine.STAGE_ORDER[s] for s in ctx.target_stages]
        assert order == sorted(order)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
