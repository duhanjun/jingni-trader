"""factor-engine 性能基准测试：Polars 后端 vs pandas 后端（方向二 T2-11）。

覆盖 PRD 第 6.3 节：
- @pytest.mark.slow + @pytest.mark.requires_polars
- 中等规模数据集（1000 股 × 200 日 × 10 因子，平衡性能差异可见性与 CI 时长）
- 验证 polars 后端能正常跑通
- 输出双后端耗时与提速比例日志（不强断言 ≥ 60%，避免 CI 噪音误报，
  但断言 polars 不显著慢于 pandas）

测试范围：
- IC 计算（Pearson + Spearman）
- 中性化（行业 + 市值）
- IC Decay（多 lag 扫描）
- 相关性分析（10 因子矩阵）
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from unittest import mock

import numpy as np
import pandas as pd
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACTOR_ENGINE_DIR = os.path.join(ROOT, "skills", "factor-engine")


# ---------------------------------------------------------------------------
# Fixture：在测试期间让 factor-engine 的 scripts.optimizations 包接管 sys.modules
# ---------------------------------------------------------------------------


@pytest.fixture
def fe_optimizations_env():
    """临时注册 factor-engine 的 scripts 包到 sys.modules。"""
    saved = {
        k: sys.modules.get(k)
        for k in list(sys.modules.keys())
        if k == "scripts" or k.startswith("scripts.")
    }
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(FACTOR_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = importlib.util.spec_from_file_location(
            "scripts", init_py, submodule_search_locations=[scripts_dir]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    try:
        yield
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


def _load_opt_module(name: str):
    if name == "__init__":
        return importlib.import_module("scripts.optimizations")
    return importlib.import_module(f"scripts.optimizations.{name}")


# ---------------------------------------------------------------------------
# 数据构造：1000 股 × 200 日 × 10 因子
# ---------------------------------------------------------------------------


def _make_bench_panel(n_days=200, n_stocks=1000, n_factors=10, seed=42):
    """构造性能基准测试数据集。

    - 1000 股 × 200 日 × 10 因子 = 200K 行 × 10 因子列
    - 满足所有 optimizations 模块的最小截面要求
    - 数据量足够大，使 polars 多线程优势显现，但 CI 时长可控（< 30s）
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]

    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            row = {"date": d, "code": c}
            for k in range(n_factors):
                row[f"factor_{k}"] = rng.normal(0, 1)
            base = 10.0 + j * 0.01
            row["close"] = base * (1 + row["factor_0"] * 0.001) + rng.normal(0, 0.02)
            row["industry"] = f"industry_{j % 10}"
            row["lncap"] = np.log(base * 1000)
            rows.append(row)

    return pd.DataFrame(rows)


def _timeit(fn, *args, **kwargs):
    """计时辅助：返回 (result, elapsed_seconds)。"""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 性能基准测试
# ---------------------------------------------------------------------------


@pytest.mark.skill_factor_engine
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_polars
def test_perf_ic_pearson(fe_optimizations_env, caplog):
    """T2-11 测试 1：Pearson IC 计算 polars vs pandas 性能对比。

    断言 polars 不显著慢于 pandas（atol 50%，考虑 CI 噪音）。
    日志输出提速比例。
    """
    ic_mod = _load_opt_module("ic_vectorized")
    df = _make_bench_panel(n_days=200, n_stocks=500, n_factors=3)

    # 构造 IC 输入：factor + 1d forward return
    df = df.sort_values(["code", "date"]).copy()
    df["fwd_1d"] = df.groupby("code")["close"].transform(
        lambda x: x.shift(-1) / x - 1.0
    )
    valid = df[["factor_0", "fwd_1d", "date"]].dropna()

    _, t_pd = _timeit(
        ic_mod.ic_series_pearson,
        valid["factor_0"], valid["fwd_1d"], valid["date"],
        min_obs=10, backend="pandas",
    )
    s_pl, t_pl = _timeit(
        ic_mod.ic_series_pearson,
        valid["factor_0"], valid["fwd_1d"], valid["date"],
        min_obs=10, backend="polars",
    )

    speedup = (t_pd - t_pl) / t_pd * 100 if t_pd > 0 else 0
    print(f"\n[Pearson IC] pandas={t_pd:.3f}s polars={t_pl:.3f}s speedup={speedup:.1f}%")

    # polars 应能正常输出
    assert len(s_pl) > 0, "polars 输出不应为空"
    # polars 不应显著慢于 pandas（允许 50% 容差，CI 噪音）
    assert t_pl <= t_pd * 1.5, (
        f"polars 显著慢于 pandas: pandas={t_pd:.3f}s polars={t_pl:.3f}s"
    )


@pytest.mark.skill_factor_engine
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_polars
def test_perf_ic_spearman(fe_optimizations_env, caplog):
    """T2-11 测试 2：Spearman IC 计算 polars vs pandas 性能对比。"""
    ic_mod = _load_opt_module("ic_vectorized")
    df = _make_bench_panel(n_days=200, n_stocks=500, n_factors=3)

    df = df.sort_values(["code", "date"]).copy()
    df["fwd_1d"] = df.groupby("code")["close"].transform(
        lambda x: x.shift(-1) / x - 1.0
    )
    valid = df[["factor_0", "fwd_1d", "date"]].dropna()

    _, t_pd = _timeit(
        ic_mod.ic_series_spearman,
        valid["factor_0"], valid["fwd_1d"], valid["date"],
        min_obs=10, backend="pandas",
    )
    s_pl, t_pl = _timeit(
        ic_mod.ic_series_spearman,
        valid["factor_0"], valid["fwd_1d"], valid["date"],
        min_obs=10, backend="polars",
    )

    speedup = (t_pd - t_pl) / t_pd * 100 if t_pd > 0 else 0
    print(f"\n[Spearman IC] pandas={t_pd:.3f}s polars={t_pl:.3f}s speedup={speedup:.1f}%")

    assert len(s_pl) > 0
    assert t_pl <= t_pd * 1.5, (
        f"polars 显著慢于 pandas: pandas={t_pd:.3f}s polars={t_pl:.3f}s"
    )


@pytest.mark.skill_factor_engine
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_polars
def test_perf_neutralize(fe_optimizations_env, caplog):
    """T2-11 测试 3：中性化 polars vs pandas 性能对比。

    中性化是优化收益最大的模块（PRD 标注 5-15× 提速）。
    """
    neu_mod = _load_opt_module("vectorized_neutralize")
    df = _make_bench_panel(n_days=200, n_stocks=500, n_factors=5)
    factor_names = [f"factor_{k}" for k in range(5)]

    _, t_pd = _timeit(
        neu_mod.neutralize_factor,
        df.copy(), factor_names,
        neutralize_mcap=True, neutralize_industry=True,
        min_count=30, backend="pandas",
    )
    df_pl, t_pl = _timeit(
        neu_mod.neutralize_factor,
        df.copy(), factor_names,
        neutralize_mcap=True, neutralize_industry=True,
        min_count=30, backend="polars",
    )

    speedup = (t_pd - t_pl) / t_pd * 100 if t_pd > 0 else 0
    print(f"\n[Neutralize] pandas={t_pd:.3f}s polars={t_pl:.3f}s speedup={speedup:.1f}%")

    # 验证 polars 输出列完整
    for f in factor_names:
        assert f"{f}_neutral" in df_pl.columns

    # 中性化通常能看到明显加速，断言 polars 不慢于 pandas
    assert t_pl <= t_pd * 1.2, (
        f"polars 显著慢于 pandas: pandas={t_pd:.3f}s polars={t_pl:.3f}s"
    )


@pytest.mark.skill_factor_engine
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_polars
def test_perf_ic_decay(fe_optimizations_env, caplog):
    """T2-11 测试 4：IC Decay 多 lag 扫描 polars vs pandas 性能对比。

    IC Decay 含 Python lag 循环，polars 提速有限。本测试仅验证功能正确性
    与可观察性，不强断言提速比例。
    """
    decay_mod = _load_opt_module("ic_decay")
    df = _make_bench_panel(n_days=100, n_stocks=500, n_factors=3)

    analyzer = decay_mod.ICDecayAnalyzer(min_lag=1, max_lag=5, min_cross_size=30)

    res_pd, t_pd = _timeit(
        analyzer.calc_ic_decay, df, "factor_0", backend="pandas"
    )
    res_pl, t_pl = _timeit(
        analyzer.calc_ic_decay, df, "factor_0", backend="polars"
    )

    speedup = (t_pd - t_pl) / t_pd * 100 if t_pd > 0 else 0
    print(f"\n[IC Decay] pandas={t_pd:.3f}s polars={t_pl:.3f}s speedup={speedup:.1f}%")

    assert len(res_pl) > 0, "polars IC Decay 应至少返回一个 lag 结果"
    # 双后端 lag 数量一致
    assert len(res_pd) == len(res_pl), "双后端 lag 数量不一致"


@pytest.mark.skill_factor_engine
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_polars
def test_perf_correlation(fe_optimizations_env, caplog):
    """T2-11 测试 5：相关性分析 polars vs pandas 性能对比。

    注：pandas.DataFrame.corr() 已是 numpy C 实现，在小矩阵（≤50 因子）上
    极快（< 50ms），polars 的 to_pandas/from_pandas 转换开销反而成为瓶颈。
    本测试主要验证功能正确性，性能日志仅供观察，不强断言提速。
    """
    corr_mod = _load_opt_module("vectorized_correlation")
    df = _make_bench_panel(n_days=200, n_stocks=500, n_factors=10)
    factor_names = [f"factor_{k}" for k in range(10)]

    res_pd, t_pd = _timeit(
        corr_mod.correlation_analysis,
        df.copy(), factor_names=factor_names,
        max_correlation=0.99, backend="pandas",
    )
    res_pl, t_pl = _timeit(
        corr_mod.correlation_analysis,
        df.copy(), factor_names=factor_names,
        max_correlation=0.99, backend="polars",
    )

    speedup = (t_pd - t_pl) / t_pd * 100 if t_pd > 0 else 0
    print(f"\n[Correlation] pandas={t_pd:.3f}s polars={t_pl:.3f}s speedup={speedup:.1f}%")

    # 功能正确性验证
    assert len(res_pl["selected_factors"]) > 0, "polars 应至少保留一个因子"
    # 双后端 selected_factors 集合一致
    assert set(res_pd["selected_factors"]) == set(res_pl["selected_factors"]), (
        "双后端 selected_factors 不一致"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-header"])
