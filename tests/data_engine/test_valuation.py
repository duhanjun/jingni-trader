"""data-engine L2 单元测试：估值分位计算 (valuation.py)。

覆盖 ValuationAnalyzer 的：
- calculate_percentile：历史 PE/PB 分位计算
- _percentile_of_score：midrank 百分位算法（known data → known percentile）
- _verdict_from_percentile：偏低/适中/偏高 阈值
- batch_percentile：多只股票批量分位
- compare_valuation：综合估值得分与 overall_verdict
- 边界：空数据 / 全 NaN / 全负值（PE 应被过滤）

合成数据：5 年月度 PE 数据（60 个观测点），值已知 → 百分位可手算。
"""
from __future__ import annotations

import os
import importlib.util as ilu

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# 模块加载：把 data-engine/scripts/valuation.py 加载为独立模块
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_DIR = os.path.join(ROOT, "skills", "data-engine")
VALUATION_PATH = os.path.join(DATA_ENGINE_DIR, "scripts", "valuation.py")


def _load_valuation_module():
    """显式加载 valuation.py。

    valuation.py 仅依赖 pandas/numpy/typing/logging，无 `from scripts.*` 导入，
    可直接以裸文件形式加载。
    """
    spec = ilu.spec_from_file_location("valuation_mod", VALUATION_PATH)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# 合成数据构造器
# ============================================================================

def _make_monthly_pe_history(
    stock_code: str,
    pe_values: list[float],
    start: str = "2019-01-31",
) -> pd.DataFrame:
    """构造月频历史 PE 数据。

    Args:
        stock_code: 股票代码
        pe_values: PE 值列表，按时间升序；最后一个为"最新"观测。
        start: 起始月份末
    """
    dates = pd.date_range(start, periods=len(pe_values), freq="ME")
    return pd.DataFrame({
        "code": stock_code,
        "date": dates,
        "pe_ttm": pe_values,
    })


def _make_monthly_multi_metric_history(
    stock_code: str,
    n_months: int = 60,
    start: str = "2019-01-31",
    seed: int = 42,
) -> pd.DataFrame:
    """构造 4 指标月频历史数据（pe_ttm / pb / ps_ttm / dv_ratio）。"""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start, periods=n_months, freq="ME")
    return pd.DataFrame({
        "code": stock_code,
        "date": dates,
        "pe_ttm": rng.uniform(8, 25, n_months).round(4),
        "pb": rng.uniform(0.5, 3.0, n_months).round(4),
        "ps_ttm": rng.uniform(0.3, 5.0, n_months).round(4),
        "dv_ratio": rng.uniform(0.0, 5.0, n_months).round(4),
    })


# ============================================================================
# 单元测试
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestPercentileOfScore:
    """验证 _percentile_of_score midrank 算法（known data → known percentile）。"""

    def test_max_value_percentile(self):
        mod = _load_valuation_module()
        # 1..60，value=60 → below=59, equal=1, n=60
        # percentile = (59 + 0.5*1) / 60 * 100 = 99.166...
        scores = np.arange(1, 61, dtype=float)
        pct = mod._percentile_of_score(scores, 60.0)
        assert pct == pytest.approx((59 + 0.5) / 60 * 100, abs=1e-6)

    def test_min_value_percentile(self):
        mod = _load_valuation_module()
        # 1..60，value=1 → below=0, equal=1, n=60
        # percentile = (0 + 0.5*1) / 60 * 100 = 0.833...
        scores = np.arange(1, 61, dtype=float)
        pct = mod._percentile_of_score(scores, 1.0)
        assert pct == pytest.approx(0.5 / 60 * 100, abs=1e-6)

    def test_mid_value_percentile(self):
        mod = _load_valuation_module()
        # 1..60，value=30 → below=29, equal=1, n=60
        # percentile = (29 + 0.5) / 60 * 100 = 49.166...
        scores = np.arange(1, 61, dtype=float)
        pct = mod._percentile_of_score(scores, 30.0)
        assert pct == pytest.approx(29.5 / 60 * 100, abs=1e-6)

    def test_empty_scores_returns_zero(self):
        mod = _load_valuation_module()
        assert mod._percentile_of_score(np.array([]), 10.0) == 0.0

    def test_all_nan_scores_returns_zero(self):
        mod = _load_valuation_module()
        assert mod._percentile_of_score(np.array([np.nan, np.nan]), 10.0) == 0.0


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestVerdictFromPercentile:
    """验证 _verdict_from_percentile 阈值：<25% 偏低，25-75% 适中，>75% 偏高。"""

    def test_low_percentile_is_偏低(self):
        mod = _load_valuation_module()
        assert mod._verdict_from_percentile(0.0) == "偏低"
        assert mod._verdict_from_percentile(10.0) == "偏低"
        assert mod._verdict_from_percentile(24.99) == "偏低"

    def test_mid_percentile_is_适中(self):
        mod = _load_valuation_module()
        assert mod._verdict_from_percentile(25.0) == "适中"
        assert mod._verdict_from_percentile(50.0) == "适中"
        assert mod._verdict_from_percentile(75.0) == "适中"

    def test_high_percentile_is_偏高(self):
        mod = _load_valuation_module()
        assert mod._verdict_from_percentile(75.01) == "偏高"
        assert mod._verdict_from_percentile(99.0) == "偏高"
        assert mod._verdict_from_percentile(100.0) == "偏高"


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestCalculatePercentile:
    """验证 ValuationAnalyzer.calculate_percentile 主流程。"""

    def test_percentile_with_5y_monthly_pe_high_current(self):
        """5 年月度 PE 数据（60 个点），最新值为最大 → 分位应 ~99%。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        # PE 值 1..60 升序，最后一个月（最新）= 60
        pe_values = list(range(1, 61))
        df = _make_monthly_pe_history("000001.SZ", pe_values)

        result = analyzer.calculate_percentile("000001.SZ", "pe_ttm", df, years=5)

        assert result["stock"] == "000001.SZ"
        assert result["metric"] == "pe_ttm"
        assert result["current_value"] == 60.0
        # below=59, equal=1, n=60 → 99.166...
        assert result["percentile"] == pytest.approx(99.17, abs=0.05)
        assert result["verdict"] == "偏高"
        assert result["samples"] == 60
        assert result["history_years"] == 5
        assert result["min"] == 1.0
        assert result["max"] == 60.0
        assert result["median"] == pytest.approx(30.5, abs=0.01)

    def test_percentile_with_5y_monthly_pe_low_current(self):
        """最新值为最小 → 分位应 ~0.83%（偏低）。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        # PE 值 60..1 降序时间序列，最后一个月（最新）= 1
        pe_values = list(range(60, 0, -1))
        df = _make_monthly_pe_history("000001.SZ", pe_values)

        result = analyzer.calculate_percentile("000001.SZ", "pe_ttm", df, years=5)

        assert result["current_value"] == 1.0
        assert result["percentile"] == pytest.approx(0.83, abs=0.05)
        assert result["verdict"] == "偏低"

    def test_percentile_with_mid_current(self):
        """最新值居中 → 分位 ~50.83%（适中）。

        current=30.5 在 [1..59, 30.5] 中：
        below=30 (1..30), equal=1 (30.5 自身), n=60
        percentile = (30 + 0.5*1) / 60 * 100 = 50.833...
        """
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        pe_values = [float(x) for x in range(1, 60)] + [30.5]
        df = _make_monthly_pe_history("000001.SZ", pe_values)

        result = analyzer.calculate_percentile("000001.SZ", "pe_ttm", df, years=5)

        assert result["current_value"] == 30.5
        assert result["percentile"] == pytest.approx(50.83, abs=0.05)
        assert result["verdict"] == "适中"

    def test_negative_pe_filtered_out(self):
        """PE 为负或零时应被过滤（_POSITIVE_REQUIRED=True）。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        # 5 个正值 + 5 个负值/零，latest 为正值
        pe_values = [10.0, 12.0, -5.0, 0.0, 15.0, -3.0, 11.0, 13.0, 14.0, 16.0]
        df = _make_monthly_pe_history("000001.SZ", pe_values)

        result = analyzer.calculate_percentile("000001.SZ", "pe_ttm", df, years=5)

        # 有效值应为 7 个（10, 12, 15, 11, 13, 14, 16），latest=16 → below=6, equal=1
        # percentile = (6 + 0.5) / 7 * 100 = 92.857...
        assert result["samples"] == 7
        assert result["current_value"] == 16.0
        assert result["percentile"] == pytest.approx(92.86, abs=0.05)
        assert result["verdict"] == "偏高"

    def test_empty_data_returns_no_data(self):
        """空 DataFrame → verdict='无数据'。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        result = analyzer.calculate_percentile(
            "000001.SZ", "pe_ttm", pd.DataFrame(), years=5
        )
        assert result["verdict"] == "无数据"
        assert result["percentile"] is None
        assert result["samples"] == 0

    def test_none_data_returns_no_data(self):
        """historical_data=None → verdict='无数据'。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        result = analyzer.calculate_percentile(
            "000001.SZ", "pe_ttm", None, years=5
        )
        assert result["verdict"] == "无数据"

    def test_all_nan_data_returns_no_data(self):
        """全 NaN PE → 过滤后无有效值 → verdict='无数据'。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        dates = pd.date_range("2019-01-31", periods=12, freq="ME")
        df = pd.DataFrame({
            "code": "000001.SZ",
            "date": dates,
            "pe_ttm": [np.nan] * 12,
        })
        result = analyzer.calculate_percentile("000001.SZ", "pe_ttm", df, years=5)
        assert result["verdict"] == "无数据"
        assert result["samples"] == 0

    def test_stock_not_in_data_returns_no_data(self):
        """historical_data 中未找到目标股票 → '无数据'。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        df = _make_monthly_pe_history("600000.SH", list(range(1, 61)))
        result = analyzer.calculate_percentile("000001.SZ", "pe_ttm", df, years=5)
        assert result["verdict"] == "无数据"

    def test_missing_metric_column_returns_no_data(self):
        """historical_data 缺少 metric 列 → '无数据'。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        dates = pd.date_range("2019-01-31", periods=12, freq="ME")
        df = pd.DataFrame({
            "code": "000001.SZ",
            "date": dates,
            # 故意没有 pe_ttm 列
        })
        result = analyzer.calculate_percentile("000001.SZ", "pe_ttm", df, years=5)
        assert result["verdict"] == "无数据"


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestBatchPercentile:
    """验证 ValuationAnalyzer.batch_percentile 多只股票批量分位。"""

    def test_batch_with_multiple_stocks(self):
        """3 只股票批量计算，返回 DataFrame 按分位升序排列。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        # 三只股票，各自 PE 1..60，但最新值不同
        # stock_a 最新=60（分位 ~99）
        # stock_b 最新=1（分位 ~0.83）
        # stock_c 最新=30（分位 ~49）
        frames = []
        for code, latest in [("000001.SZ", 60), ("600000.SH", 1), ("000002.SZ", 30)]:
            # range(1, 61) = 1..60；移除 latest 后追加 latest，保证 latest 为最新观测
            pe_values = [float(x) for x in range(1, 61)]
            pe_values.remove(latest)
            pe_values.append(latest)
            frames.append(_make_monthly_pe_history(code, pe_values))
        df = pd.concat(frames, ignore_index=True)

        result = analyzer.batch_percentile(
            ["000001.SZ", "600000.SH", "000002.SZ"], "pe_ttm", df, years=5
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # 按分位升序：600000.SH(~0.83) < 000002.SZ(~49) < 000001.SZ(~99)
        assert result.iloc[0]["stock"] == "600000.SH"
        assert result.iloc[1]["stock"] == "000002.SZ"
        assert result.iloc[2]["stock"] == "000001.SZ"
        assert result["percentile"].is_monotonic_increasing

    def test_batch_empty_stocks_returns_empty_df(self):
        """空股票列表 → 空 DataFrame。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()
        result = analyzer.batch_percentile([], "pe_ttm", pd.DataFrame(), years=5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestCompareValuation:
    """验证 ValuationAnalyzer.compare_valuation 综合估值分析。"""

    def test_compare_returns_overall_verdict_and_score(self):
        """综合 4 指标估值 → 返回 overall_verdict + score + metrics。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        df = _make_monthly_multi_metric_history("000001.SZ", n_months=60)
        result = analyzer.compare_valuation("000001.SZ", df)

        assert result["stock"] == "000001.SZ"
        assert "metrics" in result
        assert "overall_verdict" in result
        assert "score" in result
        # 至少有一个指标被算出来
        assert len(result["metrics"]) > 0
        # score 应为 0-100 的浮点数（只要至少一个指标有效）
        if result["score"] is not None:
            assert 0.0 <= result["score"] <= 100.0
        # overall_verdict 应为已知枚举
        assert result["overall_verdict"] in (
            "估值偏低", "估值合理偏低", "估值合理偏高", "估值偏高", "无数据",
        )

    def test_compare_with_low_pe_high_dv_is_undervalued(self):
        """PE/PB/PS 居历史低位、股息率居高位 → score 高，估值偏低。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        # 构造 60 个月数据：pe/pb/ps 最新为最小值（分位低 → 得分高）
        # dv_ratio 最新为最大值（分位高 → 得分高）
        rng = np.random.RandomState(123)
        n = 60
        dates = pd.date_range("2019-01-31", periods=n, freq="ME")
        # pe: 60 个值，最新为最小（5）；其余随机 10..30
        pe_vals = rng.uniform(10, 30, n - 1).tolist() + [5.0]
        pb_vals = rng.uniform(1.0, 3.0, n - 1).tolist() + [0.5]
        ps_vals = rng.uniform(1.0, 4.0, n - 1).tolist() + [0.3]
        # dv_ratio: 最新为最大（8）；其余 0..3
        dv_vals = rng.uniform(0, 3, n - 1).tolist() + [8.0]
        df = pd.DataFrame({
            "code": "000001.SZ",
            "date": dates,
            "pe_ttm": pe_vals,
            "pb": pb_vals,
            "ps_ttm": ps_vals,
            "dv_ratio": dv_vals,
        })

        result = analyzer.compare_valuation("000001.SZ", df)

        # PE/PB/PS 分位都很低 → 得分 = 100 - 分位 ≈ 100
        # DV 分位很高 → 得分 = 分位 ≈ 100
        # 综合 score 应该 >= 75 → "估值偏低"
        assert result["score"] is not None
        assert result["score"] >= 75.0
        assert result["overall_verdict"] == "估值偏低"

    def test_compare_with_empty_data_returns_no_data(self):
        """空数据 → overall_verdict='无数据', score=None。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        result = analyzer.compare_valuation("000001.SZ", pd.DataFrame())
        assert result["overall_verdict"] == "无数据"
        assert result["score"] is None

    def test_compare_with_none_data_returns_no_data(self):
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        result = analyzer.compare_valuation("000001.SZ", None)
        assert result["overall_verdict"] == "无数据"
        assert result["score"] is None

    def test_compare_with_all_nan_metrics_returns_no_data(self):
        """所有指标列全 NaN → overall_verdict='无数据'。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        n = 12
        dates = pd.date_range("2019-01-31", periods=n, freq="ME")
        df = pd.DataFrame({
            "code": "000001.SZ",
            "date": dates,
            "pe_ttm": [np.nan] * n,
            "pb": [np.nan] * n,
            "ps_ttm": [np.nan] * n,
            "dv_ratio": [np.nan] * n,
        })
        result = analyzer.compare_valuation("000001.SZ", df)
        assert result["overall_verdict"] == "无数据"
        assert result["score"] is None

    def test_compare_with_no_metric_columns_returns_no_data(self):
        """数据中没有任何估值指标列 → '无数据'。"""
        mod = _load_valuation_module()
        analyzer = mod.ValuationAnalyzer()

        n = 12
        dates = pd.date_range("2019-01-31", periods=n, freq="ME")
        df = pd.DataFrame({
            "code": "000001.SZ",
            "date": dates,
            # 没有任何 pe_ttm/pb/ps_ttm/dv_ratio 列
        })
        result = analyzer.compare_valuation("000001.SZ", df)
        assert result["overall_verdict"] == "无数据"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
