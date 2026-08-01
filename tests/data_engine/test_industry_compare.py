"""data-engine L2 单元测试：行业对比分析 (industry_compare.py)。

覆盖 IndustryComparator 的：
- compare：10 只同行业股票的多维度对比
- _calc_rankings：rank 1 = 最优
- industry_avg 排除目标股票自身
- _generate_insights：advantages / disadvantages 生成
- 边界：行业内仅一只股票 / 缺少 industry 列 / 缺少 code 列
"""
from __future__ import annotations

import os
import importlib.util as ilu

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# 模块加载：把 data-engine/scripts/industry_compare.py 加载为独立模块
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_DIR = os.path.join(ROOT, "skills", "data-engine")
INDUSTRY_COMPARE_PATH = os.path.join(DATA_ENGINE_DIR, "scripts", "industry_compare.py")


def _load_industry_compare_module():
    """显式加载 industry_compare.py。

    industry_compare.py 仅依赖 pandas/numpy/typing/logging，无 `from scripts.*` 导入，
    可直接以裸文件形式加载。
    """
    spec = ilu.spec_from_file_location("industry_compare_mod", INDUSTRY_COMPARE_PATH)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# 合成数据构造器
# ============================================================================

def _make_industry_financial_data(
    n_stocks: int = 10,
    industry: str = "银行",
    seed: int = 42,
) -> pd.DataFrame:
    """构造同行业内 n 只股票的合成财务数据。

    每只股票有：code, industry, name, pe_ttm, pb, ps_ttm, dv_ratio,
                roe, roa, gross_margin, net_margin, revenue_growth,
                profit_growth, debt_ratio, current_ratio, quick_ratio

    股票按 index 排序，index 越大 → PE 越高（估值越贵）、ROE 越高（盈利越强）。
    """
    rng = np.random.RandomState(seed)
    codes = [f"60000{i}.SH" for i in range(n_stocks)]
    names = [f"股票{i}" for i in range(n_stocks)]

    # 让指标随 index 单调变化，便于断言排名
    idx = np.arange(n_stocks)
    return pd.DataFrame({
        "code": codes,
        "industry": industry,
        "name": names,
        # 估值类：越低越好；index 越大 → pe 越大（越差）
        "pe_ttm": (5.0 + idx * 1.5 + rng.uniform(-0.1, 0.1, n_stocks)).round(4),
        "pb": (0.5 + idx * 0.2 + rng.uniform(-0.02, 0.02, n_stocks)).round(4),
        "ps_ttm": (1.0 + idx * 0.3 + rng.uniform(-0.05, 0.05, n_stocks)).round(4),
        "dv_ratio": (5.0 - idx * 0.3 + rng.uniform(-0.1, 0.1, n_stocks)).round(4),
        # 基本面类：越高越好；index 越大 → roe 越大（越好）
        "roe": (8.0 + idx * 1.2 + rng.uniform(-0.2, 0.2, n_stocks)).round(4),
        "roa": (1.0 + idx * 0.2 + rng.uniform(-0.05, 0.05, n_stocks)).round(4),
        "gross_margin": (20.0 + idx * 1.5 + rng.uniform(-0.3, 0.3, n_stocks)).round(4),
        "net_margin": (5.0 + idx * 0.8 + rng.uniform(-0.2, 0.2, n_stocks)).round(4),
        "revenue_growth": (5.0 + idx * 1.0 + rng.uniform(-0.3, 0.3, n_stocks)).round(4),
        "profit_growth": (8.0 + idx * 1.5 + rng.uniform(-0.4, 0.4, n_stocks)).round(4),
        # 越低越好：index 越大 → debt_ratio 越大（越差）
        "debt_ratio": (30.0 + idx * 2.0 + rng.uniform(-0.5, 0.5, n_stocks)).round(4),
        # 越高越好：index 越大 → current_ratio 越大（越好）
        "current_ratio": (1.0 + idx * 0.1 + rng.uniform(-0.02, 0.02, n_stocks)).round(4),
        "quick_ratio": (0.8 + idx * 0.08 + rng.uniform(-0.02, 0.02, n_stocks)).round(4),
    })


# ============================================================================
# 单元测试
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestIndustryCompareBasic:
    """验证 IndustryComparator.compare 基本流程。"""

    def test_compare_returns_required_fields(self):
        """compare 返回结构包含所有必需字段。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        result = comparator.compare("600000.SH", df)

        for field in (
            "stock", "industry", "industry_avg", "stock_values",
            "rankings", "advantages", "disadvantages", "summary",
        ):
            assert field in result, f"返回缺少字段: {field}"
        assert result["stock"] == "600000.SH"
        assert result["industry"] == "银行"

    def test_compare_with_10_stocks_same_industry(self):
        """10 只同行业股票对比，应有 rankings 和 industry_avg。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        result = comparator.compare("600000.SH", df)

        # industry_avg 应有多个指标
        assert len(result["industry_avg"]) > 0
        # rankings 应有多个指标
        assert len(result["rankings"]) > 0
        # summary 应非空
        assert result["summary"]
        # advantages / disadvantages 应为 list
        assert isinstance(result["advantages"], list)
        assert isinstance(result["disadvantages"], list)


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestRankingsCorrectness:
    """验证 _calc_rankings 中 rank=1 表示最优。"""

    def test_lowest_pe_gets_rank_1(self):
        """PE 最低的股票在 pe_ttm 指标上 rank=1。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        # 600000.SH 是 index=0，PE 最低 → rank=1
        result = comparator.compare("600000.SH", df)

        pe_rank_info = result["rankings"]["pe_ttm"]
        assert pe_rank_info["rank"] == 1
        assert pe_rank_info["total"] == 10
        # rank=1, total=10 → percentile = (10-1)/10*100 = 90.0
        assert pe_rank_info["percentile"] == pytest.approx(90.0, abs=0.05)

    def test_highest_pe_gets_rank_10(self):
        """PE 最高的股票在 pe_ttm 指标上 rank=10（最差）。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        # 600009.SH 是 index=9，PE 最高 → rank=10
        result = comparator.compare("600009.SH", df)

        pe_rank_info = result["rankings"]["pe_ttm"]
        assert pe_rank_info["rank"] == 10
        assert pe_rank_info["total"] == 10
        # rank=10, total=10 → percentile = (10-10)/10*100 = 0.0
        assert pe_rank_info["percentile"] == pytest.approx(0.0, abs=0.05)

    def test_highest_roe_gets_rank_1(self):
        """ROE 最高的股票（higher_is_better=True）rank=1。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        # 600009.SH 是 index=9，ROE 最高 → rank=1
        result = comparator.compare("600009.SH", df)

        roe_rank_info = result["rankings"]["roe"]
        assert roe_rank_info["rank"] == 1
        assert roe_rank_info["total"] == 10
        # rank=1, total=10 → percentile = 90.0
        assert roe_rank_info["percentile"] == pytest.approx(90.0, abs=0.05)

    def test_lowest_roe_gets_rank_10(self):
        """ROE 最低的股票 rank=10。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        # 600000.SH 是 index=0，ROE 最低 → rank=10
        result = comparator.compare("600000.SH", df)

        roe_rank_info = result["rankings"]["roe"]
        assert roe_rank_info["rank"] == 10
        assert roe_rank_info["percentile"] == pytest.approx(0.0, abs=0.05)

    def test_rankings_within_valid_range(self):
        """所有指标的 rank 应在 [1, total] 范围内，percentile 在 [0, 100]。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        result = comparator.compare("600005.SH", df)

        for metric, info in result["rankings"].items():
            assert 1 <= info["rank"] <= info["total"], \
                f"{metric} rank={info['rank']} 超出 [1, {info['total']}]"
            assert 0.0 <= info["percentile"] <= 100.0, \
                f"{metric} percentile={info['percentile']} 超出 [0, 100]"


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestIndustryAvgExcludesSelf:
    """验证 industry_avg 计算时排除了目标股票自身。"""

    def test_industry_avg_excludes_target_stock(self):
        """industry_avg 应为同行其他 9 只股票的均值（不含自身）。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        target = "600000.SH"
        result = comparator.compare(target, df)

        # 手动计算排除自身后的均值
        peers = df[df["code"] != target]
        expected_pe = round(float(peers["pe_ttm"].mean()), 4)
        expected_roe = round(float(peers["roe"].mean()), 4)

        assert result["industry_avg"]["pe_ttm"] == pytest.approx(expected_pe, abs=1e-4)
        assert result["industry_avg"]["roe"] == pytest.approx(expected_roe, abs=1e-4)

    def test_industry_avg_differs_from_full_industry_mean(self):
        """industry_avg 不应等于包含自身的全行业均值（除非目标值刚好等于同行均值）。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        target = "600000.SH"
        result = comparator.compare(target, df)

        full_mean_pe = float(df["pe_ttm"].mean())
        peers_mean_pe = float(df[df["code"] != target]["pe_ttm"].mean())

        # 600000.SH 是 PE 最低的，从同行集合中剔除它后同行均值会变高
        # 因此 full_mean < peers_mean
        assert full_mean_pe < peers_mean_pe
        assert result["industry_avg"]["pe_ttm"] == pytest.approx(peers_mean_pe, abs=1e-4)
        assert result["industry_avg"]["pe_ttm"] != pytest.approx(full_mean_pe, abs=1e-4)


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestGenerateInsights:
    """验证 _generate_insights 生成 advantages / disadvantages。"""

    def test_insights_for_best_valuation_stock(self):
        """PE/PB 最低的股票 → advantages 应包含估值类优势。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        # 600000.SH：PE/PB/PS 最低（估值分位 90%），但 ROE 最低（基本面分位 0%）
        result = comparator.compare("600000.SH", df)

        # 估值类指标（pe/pb/ps）分位 90% >= 60% → 优势
        advantages_text = " ".join(result["advantages"])
        assert "PE" in advantages_text or "估值" in advantages_text

        # 基本面指标（roe 等）分位 0% <= 40% → 劣势
        disadvantages_text = " ".join(result["disadvantages"])
        assert "ROE" in disadvantages_text or "盈利" in disadvantages_text

    def test_insights_for_worst_valuation_stock(self):
        """PE/PB 最高的股票 → disadvantages 应包含估值类劣势。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        # 600009.SH：PE/PB/PS 最高（估值分位 0%），但 ROE 最高（基本面分位 90%）
        result = comparator.compare("600009.SH", df)

        # 基本面优势
        advantages_text = " ".join(result["advantages"])
        assert "ROE" in advantages_text or "盈利" in advantages_text

        # 估值劣势
        disadvantages_text = " ".join(result["disadvantages"])
        assert "PE" in disadvantages_text or "估值" in disadvantages_text

    def test_insights_return_lists(self):
        """_generate_insights 直接调用应返回两个 list。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        stock_values = {"pe_ttm": 5.0, "roe": 20.0}
        industry_avg = {"pe_ttm": 15.0, "roe": 12.0}
        rankings = {
            "pe_ttm": {"rank": 1, "total": 10, "percentile": 90.0, "label": "PE(市盈率)"},
            "roe": {"rank": 1, "total": 10, "percentile": 90.0, "label": "ROE"},
        }
        adv, dis = comparator._generate_insights(stock_values, industry_avg, rankings)
        assert isinstance(adv, list)
        assert isinstance(dis, list)
        assert len(adv) == 2  # 两个指标都 >= 60%
        assert len(dis) == 0


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestEdgeCases:
    """验证边界场景。"""

    def test_single_stock_in_industry(self):
        """行业内仅含目标股票 → 返回 empty 结构（industry 已填）。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = pd.DataFrame([{
            "code": "000001.SZ",
            "industry": "银行",
            "name": "唯一银行",
            "pe_ttm": 8.0,
            "roe": 12.0,
        }])
        result = comparator.compare("000001.SZ", df)

        assert result["stock"] == "000001.SZ"
        assert result["industry"] == "银行"
        assert result["rankings"] == {}
        assert result["advantages"] == []
        assert result["disadvantages"] == []
        assert result["summary"] == "无数据"

    def test_missing_industry_column(self):
        """financial_data 缺少 industry 列 → 返回 empty 结构。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = pd.DataFrame([{
            "code": "000001.SZ",
            "name": "股票1",
            "pe_ttm": 8.0,
            "roe": 12.0,
            # 没有 industry 列
        }])
        result = comparator.compare("000001.SZ", df)

        assert result["stock"] == "000001.SZ"
        assert result["rankings"] == {}
        assert result["summary"] == "无数据"

    def test_missing_code_column(self):
        """financial_data 缺少 code 列 → 返回 empty 结构。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = pd.DataFrame([{
            "industry": "银行",
            "pe_ttm": 8.0,
            # 没有 code 列
        }])
        result = comparator.compare("000001.SZ", df)

        assert result["stock"] == "000001.SZ"
        assert result["rankings"] == {}
        assert result["summary"] == "无数据"

    def test_empty_financial_data(self):
        """空 DataFrame → 返回 empty 结构。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        result = comparator.compare("000001.SZ", pd.DataFrame())
        assert result["stock"] == "000001.SZ"
        assert result["summary"] == "无数据"

    def test_none_financial_data(self):
        """None → 返回 empty 结构。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        result = comparator.compare("000001.SZ", None)
        assert result["stock"] == "000001.SZ"
        assert result["summary"] == "无数据"

    def test_stock_not_in_data(self):
        """目标股票不在 financial_data 中 → 返回 empty 结构。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        result = comparator.compare("999999.SZ", df)
        assert result["stock"] == "999999.SZ"
        assert result["summary"] == "无数据"

    def test_explicit_industry_overrides_auto_detect(self):
        """显式传入 industry 参数优先于从数据中自动读取。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10, industry="银行")
        # 显式传入不存在的行业 → 找不到同行 → empty
        result = comparator.compare("600000.SH", df, industry="非银金融")
        assert result["industry"] == "非银金融"
        assert result["summary"] == "无数据"

    def test_stock_with_nan_industry_value(self):
        """目标股票的 industry 字段为 NaN → 返回 empty 结构。"""
        mod = _load_industry_compare_module()
        comparator = mod.IndustryComparator()

        df = _make_industry_financial_data(n_stocks=10)
        # 把目标股票的 industry 改为 NaN
        df.loc[df["code"] == "600000.SH", "industry"] = np.nan
        result = comparator.compare("600000.SH", df)
        assert result["summary"] == "无数据"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
