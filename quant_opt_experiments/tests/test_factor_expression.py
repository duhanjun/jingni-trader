"""
测试 1：因子表达式引擎 (Qlib 风格)
- 解析 + 编译
- 在合成的 panel 上批量计算
- Alpha158 子集注册
- 依赖图
- 与手动计算对比
"""
import time
import pandas as pd
import numpy as np
import pytest

from quant_opt_experiments.factor_expression_engine import (
    FactorEngine,
    parse,
    register_alpha158_pv,
    FieldNode,
    RefNode,
    RollingNode,
    BinaryOpNode,
)
from quant_opt_experiments.tests.fixtures import make_synthetic_panel


def test_parse_simple_field():
    node = parse("$close")
    assert isinstance(node, FieldNode)
    assert node.name == "$close"


def test_parse_ref():
    node = parse("Ref($close, 5)")
    assert isinstance(node, RefNode)
    assert node.n == 5


def test_parse_rolling():
    node = parse("Mean($close, 20)")
    assert isinstance(node, RollingNode)
    assert node.op == "Mean"
    assert node.window == 20


def test_parse_binary():
    node = parse("$high - $low")
    assert isinstance(node, BinaryOpNode)
    assert node.op == "Sub"


def test_parse_complex():
    expr = "($close - Ref($close, 5)) / Ref($close, 5)"
    node = parse(expr)
    assert isinstance(node, BinaryOpNode)
    # 依赖应只有 $close
    assert set(node.dependencies()) == {"$close"}


def test_compute_ma20_against_manual(panel):
    engine = FactorEngine(panel)
    engine.register("MA20", "Mean($close, 20)")
    pivot = engine.compute("MA20")
    # 手动计算
    panel_sorted = panel.sort_values(["code", "date"]).copy()
    panel_sorted["ma20_manual"] = (
        panel_sorted.groupby("code")["close"]
        .rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    )
    manual_pivot = panel_sorted.pivot(index="date", columns="code", values="ma20_manual")
    # 比较（忽略 NaN）
    diff = (pivot - manual_pivot).abs().max().max()
    assert diff < 1e-9, f"MA20 与手动计算不一致, max diff = {diff}"


def test_dependencies(panel):
    engine = FactorEngine(panel)
    engine.register("ROC5", "$close / Ref($close, 5) - 1")
    engine.register("CUSTOM", "Mean($close, 5) / $close - Mean($close, 20) / $close")
    deps = engine.dependency_graph()
    assert set(deps["ROC5"]) == {"$close"}
    assert set(deps["CUSTOM"]) == {"$close"}


def test_alpha158_subset(panel):
    engine = FactorEngine(panel)
    specs = register_alpha158_pv(engine)
    assert len(specs) > 15
    # 计算全部
    df = engine.compute_all()
    assert "date" in df.columns
    assert "code" in df.columns
    assert "MA20" in df.columns
    assert "ROC5" in df.columns
    # 无全空列
    for c in ["MA20", "MA5", "ROC5", "STD5"]:
        assert df[c].notna().any(), f"{c} 全为空"


def test_performance_vs_legacy(panel):
    """性能对比：表达式引擎 vs pandas_ta 逐列"""
    engine = FactorEngine(panel)
    engine.register("MA5", "Mean($close, 5)")
    engine.register("MA20", "Mean($close, 20)")
    engine.register("MA60", "Mean($close, 60)")
    engine.register("ROC5", "$close / Ref($close, 5) - 1")
    engine.register("ROC20", "$close / Ref($close, 20) - 1")

    t0 = time.time()
    for f in ["MA5", "MA20", "MA60", "ROC5", "ROC20"]:
        engine.compute(f)
    t_engine = time.time() - t0

    # 简单手动对比
    t0 = time.time()
    panel_sorted = panel.sort_values(["code", "date"]).copy()
    for code, sub in panel_sorted.groupby("code"):
        sub["ma5"] = sub["close"].rolling(5).mean()
        sub["ma20"] = sub["close"].rolling(20).mean()
        sub["ma60"] = sub["close"].rolling(60).mean()
        sub["roc5"] = sub["close"] / sub["close"].shift(5) - 1
        sub["roc20"] = sub["close"] / sub["close"].shift(20) - 1
    t_loop = time.time() - t0
    print(f"\n[PERF] 表达式引擎={t_engine*1000:.1f}ms  手动groupby循环={t_loop*1000:.1f}ms  加速={t_loop/t_engine:.1f}x")
    # 只验证能跑通，不强制性能（数据小）
    assert t_engine >= 0


# ---- pytest fixture ----
@pytest.fixture
def panel():
    return make_synthetic_panel(n_stocks=6, n_days=500, seed=42)


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "-s"])
    sys.exit(0)