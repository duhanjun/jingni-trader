"""strategy-model-engine L2 单元测试：IntentParser。

覆盖：
- 阶段关键词识别（7 个 stage）
- 时间范围解析（近 N 年 / 近 N 月 / 显式日期）
- 股票池识别（沪深300/中证500/中证1000/上证50/全A）
- 策略识别（reversal/momentum/ma_cross/rsi/macd）
- 风控约束（最大回撤 / 年化收益）
- 调仓频率（daily/weekly/monthly/every_N_days）
- 置信度计算 + 缺失字段检测
- to_dict 序列化
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock
from datetime import datetime

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRATEGY_ENGINE_DIR = os.path.join(ROOT, "skills", "strategy-model-engine")


def _load_intent_parser():
    """加载 strategy-model-engine/scripts/optimizations/intent_parser.py 为独立模块。"""
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(STRATEGY_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    for _m in ("sklearn", "sklearn.linear_model", "sklearn.ensemble",
               "sklearn.model_selection", "lightgbm", "catboost"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    try:
        target_path = os.path.join(STRATEGY_ENGINE_DIR, "scripts/optimizations/intent_parser.py")
        spec = ilu.spec_from_file_location("_sme_intent_parser", target_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["_sme_intent_parser"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


@pytest.mark.skill_strategy_model_engine
@pytest.mark.unit
class TestIntentParserStages:
    """验证阶段关键词识别。"""

    def test_full_pipeline_keywords(self):
        """一句话含全流程关键词 → 7 阶段全识别"""
        mod = _load_intent_parser()
        parser = mod.IntentParser(today=datetime(2024, 12, 31))
        intent = parser.parse("获取数据 做因子 训练模型 回测 组合优化 实盘执行 生成报告")
        for stage in ("DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"):
            assert stage in intent.target_stages

    def test_default_stages_when_no_keyword(self):
        """无任何关键词 → 默认 5 阶段"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("今天天气真好")
        assert intent.target_stages == ["DATA", "FACTOR", "MODEL", "BACKTEST", "REPORT"]

    def test_data_auto_added(self):
        """有 FACTOR/MODEL/BACKTEST 但没 DATA → 自动补 DATA"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("做因子回测")
        assert "DATA" in intent.target_stages


@pytest.mark.skill_strategy_model_engine
@pytest.mark.unit
class TestIntentParserDateRange:
    """验证时间范围解析。"""

    def test_years_back(self):
        """近 N 年 → 正确的起止日期"""
        mod = _load_intent_parser()
        parser = mod.IntentParser(today=datetime(2024, 12, 31))
        intent = parser.parse("近3年回测")
        assert intent.start_date == "2021-11-30"
        assert intent.end_date == "2024-11-30"

    def test_months_back(self):
        """近 N 月 → 正确的起止日期"""
        mod = _load_intent_parser()
        parser = mod.IntentParser(today=datetime(2024, 12, 31))
        intent = parser.parse("近6月回测")
        # end = 2024-12-01 - 1天 = 2024-11-30
        # start = end 替换 month=11-6=5 → 2024-05-30
        # 注：用 replace 把月份减 6，得到 5 月 30 日
        assert intent.start_date == "2024-05-30"
        assert intent.end_date == "2024-11-30"

    def test_explicit_date_range(self):
        """显式日期范围 → 精确到日"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("2024-01-01 到 2024-06-30 回测")
        assert intent.start_date == "2024-01-01"
        assert intent.end_date == "2024-06-30"

    def test_no_date_returns_empty(self):
        """无日期关键词 → start_date/end_date 为空"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("做因子回测")
        assert intent.start_date == ""
        assert intent.end_date == ""


@pytest.mark.skill_strategy_model_engine
@pytest.mark.unit
class TestIntentParserStockPool:
    """验证股票池识别。"""

    @pytest.mark.parametrize("keyword,expected_code", [
        ("沪深300", "000300.SH"),
        ("hs300", "000300.SH"),
        ("中证500", "000905.SH"),
        ("csi500", "000905.SH"),
        ("中证1000", "000852.SH"),
        ("上证50", "000016.SH"),
    ])
    def test_known_pool_keywords(self, keyword, expected_code):
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse(f"用{keyword}做回测")
        assert intent.stock_pool == [expected_code]

    def test_full_market_returns_empty_list(self):
        """全A → 空列表"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("全A回测")
        assert intent.stock_pool == []

    def test_no_keyword_returns_empty(self):
        """无股票池关键词 → 空"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("做回测")
        assert intent.stock_pool == []


@pytest.mark.skill_strategy_model_engine
@pytest.mark.unit
class TestIntentParserStrategy:
    """验证策略识别。"""

    def test_reversal_n_days(self):
        """'N日反转' → reversal 策略 + lookback"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("20日反转回测")
        assert intent.strategy_name == "reversal"
        assert intent.strategy_params["factor"] == "reversal_20d"
        assert intent.strategy_params["lookback"] == 20

    def test_momentum_n_days(self):
        """'N日动量' → momentum 策略"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("5日动量回测")
        assert intent.strategy_name == "momentum"
        assert intent.strategy_params["lookback"] == 5

    def test_ma_cross(self):
        """'MA20' → ma_cross 策略"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("MA20 回测")
        assert intent.strategy_name == "ma_cross"
        assert intent.strategy_params["slow_ma"] == 20

    def test_rsi(self):
        """'RSI 14' → rsi 策略"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("RSI 14 回测")
        assert intent.strategy_name == "rsi"
        assert intent.strategy_params["period"] == 14

    def test_macd(self):
        """'MACD' → macd 策略"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("MACD 回测")
        assert intent.strategy_name == "macd"

    def test_default_strategy(self):
        """无策略关键词 → strategy_name='default'"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("做回测")
        assert intent.strategy_name == "default"


@pytest.mark.skill_strategy_model_engine
@pytest.mark.unit
class TestIntentParserRiskAndRebal:
    """验证风控约束与调仓频率。"""

    def test_max_drawdown_constraint(self):
        """'最大回撤控制在15%以内' → risk_constraints.max_drawdown=0.15"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("最大回撤控制在15%以内的回测")
        assert intent.risk_constraints["max_drawdown"] == 0.15

    def test_annualized_return_constraint(self):
        """'年化≥20%' → risk_constraints.target_annual_return=0.20"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("年化≥20%的回测")
        assert intent.risk_constraints["target_annual_return"] == 0.20

    @pytest.mark.parametrize("keyword,expected_freq", [
        ("每日调仓", "daily"),
        ("每周调仓", "weekly"),
        ("每月调仓", "monthly"),
    ])
    def test_rebalance_frequency(self, keyword, expected_freq):
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse(f"20日反转 {keyword}")
        assert intent.strategy_params.get("rebalance_freq") == expected_freq

    def test_rebal_every_n_days(self):
        """'每5日调仓' → rebalance_freq='every_5_days'"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("20日反转 每5日调仓")
        assert intent.strategy_params.get("rebalance_freq") == "every_5_days"


@pytest.mark.skill_strategy_model_engine
@pytest.mark.unit
class TestIntentParserConfidenceAndMissing:
    """验证置信度计算与缺失字段检测。"""

    def test_full_intent_high_confidence(self):
        """所有字段齐全 → 置信度较高"""
        mod = _load_intent_parser()
        parser = mod.IntentParser(today=datetime(2024, 12, 31))
        intent = parser.parse(
            "获取近3年沪深300数据做20日反转回测，"
            "最大回撤控制在15%以内，年化≥20%，每日调仓"
        )
        assert intent.confidence > 0.7
        assert len(intent.missing_fields) == 0

    def test_empty_intent_low_confidence(self):
        """无任何字段 → 置信度低 + 多个 missing_fields"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("今天天气真好")
        assert intent.confidence < 0.5
        assert "date_range" in intent.missing_fields
        assert "stock_pool" in intent.missing_fields
        assert "strategy_name" in intent.missing_fields

    def test_to_dict_serializable(self):
        """to_dict 应可序列化为 dict"""
        mod = _load_intent_parser()
        parser = mod.IntentParser()
        intent = parser.parse("20日反转回测")
        d = intent.to_dict()
        assert isinstance(d, dict)
        assert "target_stages" in d
        assert "strategy_name" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
