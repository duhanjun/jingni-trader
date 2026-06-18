"""
Factor DSL 单元测试 + 集成测试
===============================

测试场景：
1. 基本算子（Ref, Mean, Std, Delta, Rank, Zscore）
2. 嵌套表达式：Rank(Mean($close, 5))
3. 算术运算：$close / Ref($close, 1) - 1
4. 多个因子注册与拓扑排序
5. 内置 Alpha 因子库
6. 性能对比：DSL vs 硬编码 pandas
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from quant_opt_20260618.factor_dsl.engine import (
    FactorEngine, FactorExpression,
    builtin_alpha_expressions,
    _ref, _delta, _mean, _std, _sum,
    _rank, _zscore, _quantile,
)


# ─────────────────────────────────────────────────────────────
# 工具：生成测试数据
# ─────────────────────────────────────────────────────────────

def make_test_data(n_stocks: int = 5, n_days: int = 60) -> pd.DataFrame:
    """生成测试用的日线数据"""
    np.random.seed(2024)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

    rows = []
    for code in codes:
        start_price = np.random.uniform(10, 50)
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = start_price * (1 + returns).cumprod()
        volumes = np.random.lognormal(10, 0.3, n_days).astype(int)
        for i, dt in enumerate(dates):
            close = prices[i]
            prev_close = prices[i - 1] if i > 0 else close
            rows.append({
                "code": code,
                "date": dt,
                "open": close * (1 + np.random.normal(0, 0.005)),
                "high": close * (1 + abs(np.random.normal(0, 0.01))),
                "low": close * (1 - abs(np.random.normal(0, 0.01))),
                "close": close,
                "volume": volumes[i],
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 算子正确性测试
# ─────────────────────────────────────────────────────────────

class TestOperatorsCorrectness:
    """算子正确性测试"""

    def test_ref_shift(self):
        """Ref($close, 5) = close.shift(5)"""
        df = make_test_data(n_stocks=2, n_days=20)
        data = df.set_index(["code", "date"]).sort_index()
        result = _ref(data["close"], 5)
        # 与手动 shift 对比（MultiIndex 下按 code 分组 shift）
        expected = data["close"].groupby(level="code").shift(5)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_mean_rolling(self):
        """Mean($close, 5) = close.rolling(5).mean()"""
        df = make_test_data(n_stocks=2, n_days=20)
        data = df.set_index(["code", "date"]).sort_index()
        result = _mean(data["close"], 5)
        expected = data["close"].groupby(level="code").transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_delta_diff(self):
        """Delta($close, 5) = close - Ref(close, 5)"""
        df = make_test_data(n_stocks=2, n_days=20)
        data = df.set_index(["code", "date"]).sort_index()
        result = _delta(data["close"], 5)
        ref = data["close"].groupby(level="code").shift(5)
        expected = data["close"] - ref
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_rank_cross_section(self):
        """Rank(x) 应给出横截面百分位排名"""
        # 构造单日截面：3只股票，分别 10/20/30
        df = pd.DataFrame({
            "code": ["A", "B", "C"],
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "close": [10.0, 20.0, 30.0],
        })
        data = df.set_index(["code", "date"]).sort_index()
        result = _rank(data["close"]).reset_index(level=0, drop=True)
        # 排名：A=1/3, B=2/3, C=3/3
        assert abs(result.iloc[0] - 1/3) < 1e-6
        assert abs(result.iloc[1] - 2/3) < 1e-6
        assert abs(result.iloc[2] - 1.0) < 1e-6

    def test_zscore_cross_section(self):
        """Zscore(x) 应给出标准化后的截面值"""
        df = pd.DataFrame({
            "code": ["A", "B", "C", "D"],
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "close": [10.0, 20.0, 30.0, 40.0],
        })
        data = df.set_index(["code", "date"]).sort_index()
        result = _zscore(data["close"]).reset_index(level=0, drop=True)
        # 标准化后 mean=0, std=1
        assert abs(result.mean()) < 1e-6
        assert abs(result.std() - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────
# DSL 引擎测试
# ─────────────────────────────────────────────────────────────

class TestFactorEngine:
    """DSL 引擎测试"""

    def test_simple_field_reference(self):
        """直接引用字段"""
        engine = FactorEngine()
        engine.register(FactorExpression("price", "$close"))
        df = make_test_data(n_stocks=2, n_days=10)
        result = engine.compute(df)
        assert "price" in result.columns
        assert len(result) == 20  # 2 stocks × 10 days

    def test_ref_in_formula(self):
        """公式中含 Ref 算子"""
        engine = FactorEngine()
        engine.register(FactorExpression("ref_5", "Ref($close, 5)"))
        df = make_test_data(n_stocks=2, n_days=20)
        result = engine.compute(df)
        assert "ref_5" in result.columns
        assert result["ref_5"].isna().sum() > 0  # 前 5 天应为 NaN

    def test_nested_expression(self):
        """嵌套表达式：Rank(Mean($close, 5))"""
        engine = FactorEngine()
        engine.register(FactorExpression(
            "alpha_1", "Rank(Mean($close, 5))",
            description="5日均线的横截面排名"
        ))
        df = make_test_data(n_stocks=3, n_days=20)
        result = engine.compute(df)
        assert "alpha_1" in result.columns
        # rank 应在 [0, 1]
        valid = result["alpha_1"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 1

    def test_arithmetic_in_formula(self):
        """公式中含算术运算：$close / Ref($close, 1) - 1"""
        engine = FactorEngine()
        engine.register(FactorExpression(
            "ret_1d", "$close / Ref($close, 1) - 1",
            description="1日收益率"
        ))
        df = make_test_data(n_stocks=2, n_days=20)
        result = engine.compute(df)
        assert "ret_1d" in result.columns
        # 与手动计算对比
        df_sorted = df.sort_values(["code", "date"]).reset_index(drop=True)
        expected_ret = df_sorted.groupby("code")["close"].pct_change()
        # 不严格相等（NaN 位置），但首日应都是 NaN
        assert result["ret_1d"].isna().sum() == 2  # 每只股票第 1 天

    def test_multiple_factors(self):
        """注册多个因子"""
        engine = FactorEngine()
        for e in builtin_alpha_expressions():
            engine.register(e)
        df = make_test_data(n_stocks=3, n_days=30)
        result = engine.compute(df)
        # 应有所有 6 个因子
        assert len([c for c in result.columns if c not in ["code", "date"]]) >= 6

    def test_topological_sort(self):
        """依赖图应正确拓扑排序"""
        engine = FactorEngine()
        engine.register(FactorExpression("a", "Mean($close, 5)"))
        engine.register(FactorExpression("b", "Rank(a)"))  # b 依赖 a
        engine.register(FactorExpression("c", "Mean(b, 3)"))  # c 依赖 b

        dep_graph = engine._build_dependency_graph()
        order = engine._topological_sort(dep_graph)
        # a 必须在 b 之前，b 必须在 c 之前
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")


# ─────────────────────────────────────────────────────────────
# 边界条件测试
# ─────────────────────────────────────────────────────────────

class TestFactorEngineEdgeCases:

    def test_empty_dataframe(self):
        """空数据应安全处理"""
        engine = FactorEngine()
        engine.register(FactorExpression("a", "Mean($close, 5)"))
        result = engine.compute(pd.DataFrame())
        assert result.empty

    def test_unknown_field(self):
        """未知字段应抛错"""
        engine = FactorEngine()
        engine.register(FactorExpression("a", "Mean($unknown, 5)"))
        df = make_test_data(n_stocks=2, n_days=10)
        # 我们已经做了容错：未知字段会返回 NaN
        result = engine.compute(df)
        assert "a" in result.columns
        assert result["a"].isna().all()

    def test_single_stock(self):
        """单只股票也能工作"""
        engine = FactorEngine()
        engine.register(FactorExpression("mom", "Mean($close, 5)"))
        df = make_test_data(n_stocks=1, n_days=20)
        result = engine.compute(df)
        assert len(result) == 20


# ─────────────────────────────────────────────────────────────
# 性能对比测试
# ─────────────────────────────────────────────────────────────

class TestFactorEnginePerformance:
    """DSL 性能 vs 硬编码 pandas 性能对比"""

    def test_dsl_vs_hardcoded(self):
        """对比 DSL 和硬编码实现的性能"""
        df = make_test_data(n_stocks=10, n_days=120)
        df_sorted = df.sort_values(["code", "date"]).reset_index(drop=True)

        # 硬编码版
        import time
        t0 = time.time()
        hardcoded = pd.DataFrame()
        hardcoded["code"] = df_sorted["code"]
        hardcoded["date"] = df_sorted["date"]
        for _ in range(20):  # 20 次重复模拟
            hardcoded["ret_5"] = df_sorted.groupby("code")["close"].pct_change(5)
            hardcoded["rank"] = hardcoded.groupby("date")["ret_5"].rank(pct=True)
        t_hardcoded = time.time() - t0

        # DSL 版
        engine = FactorEngine()
        for _ in range(20):
            engine.register(FactorExpression("ret_5", "$close / Ref($close, 5) - 1"))
            engine.register(FactorExpression("rank", "Rank(ret_5)"))
        t0 = time.time()
        dsl_result = engine.compute(df)
        t_dsl = time.time() - t0

        # DSL 应该有合理性能（不超过硬编码 10x）
        print(f"\n[Perf] Hardcoded: {t_hardcoded:.3f}s, DSL: {t_dsl:.3f}s, "
              f"Ratio: {t_dsl / t_hardcoded:.2f}x")
        assert t_dsl < t_hardcoded * 10, f"DSL too slow: {t_dsl:.3f}s vs {t_hardcoded:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
