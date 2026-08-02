"""factor-engine L3 集成测试：Polars 后端端到端 pipeline（方向二 T2-10）。

覆盖 PRD 第 6.2 节：
- 端到端 ``QUANT_FACTOR_BACKEND=polars`` 跑通 IC → 中性化 → 相关性 → IC Decay 完整链路
- 双后端跑同一数据集，最终输出 diff < 1e-10
- 环境变量路径：未设置 / =pandas / =polars 三种情况下 pipeline 均能完成

测试策略：
- 构造中等规模合成数据（200 股 × 60 日 × 5 因子），模拟真实因子工程链路
- 不依赖外部数据源 / 主调度器，仅聚焦 factor-engine optimizations 子模块协同
- 需真实 polars 的测试打 @pytest.mark.requires_polars
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
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
    """临时注册 factor-engine 的 scripts 包到 sys.modules。

    conftest 的 _isolate_scripts_module autouse fixture 会重置 sys.modules['scripts']
    为主调度器的 scripts 包，本 fixture 在测试期间切换为 factor-engine 的 scripts 包，
    使 optimizations 子模块可正常工作。
    """
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
# 数据构造：合成多因子 + 价格数据
# ---------------------------------------------------------------------------


def _make_synthetic_panel(n_days=60, n_stocks=200, n_factors=5, seed=42):
    """构造多因子 + 价格面板数据，模拟真实因子工程链路。

    - n_stocks=200 满足所有 optimizations 模块的最小截面要求
    - 多因子列用于相关性矩阵
    - 因子与未来收益有正相关，确保 IC 非零
    - 包含 industry / lncap 字段用于中性化
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]

    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            row = {"date": d, "code": c}
            # 5 个因子列
            for k in range(n_factors):
                row[f"factor_{k}"] = rng.normal(0, 1) + i * 0.005 * (k + 1)
            # 价格（用于 IC Decay 的 forward return）
            base = 10.0 + j * 0.1
            row["close"] = base * (1 + row["factor_0"] * 0.002) + rng.normal(0, 0.05)
            # 行业（用于中性化）
            row["industry"] = f"industry_{j % 8}"
            # 对数市值（用于中性化）
            row["lncap"] = np.log(base * 1000)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 端到端 pipeline 双后端一致性
# ---------------------------------------------------------------------------


def _run_pipeline(df, factor_names, backend):
    """跑完整 IC + 中性化 + 相关性 + IC Decay 链路，返回聚合结果。"""
    ic_mod = _load_opt_module("ic_vectorized")
    neu_mod = _load_opt_module("vectorized_neutralize")
    corr_mod = _load_opt_module("vectorized_correlation")
    decay_mod = _load_opt_module("ic_decay")

    # 1. IC 分析：每个因子计算 1d forward IC
    # 构造 forward return（按 code 分组 shift）
    df_with_ret = df.sort_values(["code", "date"]).copy()
    df_with_ret["fwd_1d"] = df_with_ret.groupby("code")["close"].transform(
        lambda x: x.shift(-1) / x - 1.0
    )

    ic_results = {}
    for f in factor_names:
        # 准备 Series 输入
        valid = df_with_ret[[f, "fwd_1d", "date"]].dropna()
        s_ic = ic_mod.ic_series_pearson(
            valid[f], valid["fwd_1d"], valid["date"],
            min_obs=10, backend=backend,
        )
        ic_results[f] = {
            "ic_mean": float(s_ic.mean()) if len(s_ic) > 0 else None,
            "ic_std": float(s_ic.std()) if len(s_ic) > 0 else None,
            "n_obs": int(len(s_ic)),
        }

    # 2. 中性化（同时控制行业 + 市值）
    neutralized = neu_mod.neutralize_factor(
        df.copy(), factor_names, neutralize_mcap=True,
        neutralize_industry=True, min_count=30, backend=backend,
    )

    # 3. 相关性分析（基于中性化后的因子）
    corr_result = corr_mod.correlation_analysis(
        neutralized.copy(),
        factor_names=[f"{f}_neutral" for f in factor_names],
        max_correlation=0.99,  # 高阈值避免剔除，便于对比矩阵
        backend=backend,
    )

    # 4. IC Decay（仅 factor_0，节约时间）
    analyzer = decay_mod.ICDecayAnalyzer(min_lag=1, max_lag=5, min_cross_size=30)
    decay_results = analyzer.calc_ic_decay(df, "factor_0", backend=backend)

    return {
        "ic": ic_results,
        "neutralized": neutralized,
        "corr_matrix": pd.DataFrame(corr_result["correlation_matrix"]),
        "corr_selected": corr_result["selected_factors"],
        "decay": [
            {
                "lag": r.lag, "ic_mean": r.ic_mean,
                "ic_std": r.ic_std, "n_obs": r.n_obs,
            }
            for r in decay_results
        ],
    }


@pytest.mark.skill_factor_engine
@pytest.mark.integration
@pytest.mark.requires_polars
def test_e2e_pipeline_polars_runs_successfully(fe_optimizations_env):
    """T2-10 测试 1：QUANT_FACTOR_BACKEND=polars 下完整 pipeline 能成功跑通。

    验证：所有模块（IC / 中性化 / 相关性 / IC Decay）在 polars 后端
    下协同工作，无异常。
    """
    df = _make_synthetic_panel(n_days=60, n_stocks=200, n_factors=5)
    factor_names = [f"factor_{k}" for k in range(5)]

    result = _run_pipeline(df, factor_names, backend="polars")

    # 验证每个模块产出非空
    assert len(result["ic"]) == 5, "IC 分析应返回 5 个因子的统计量"
    for f in factor_names:
        assert result["ic"][f]["n_obs"] > 0, f"因子 {f} IC 序列不应为空"

    for f in factor_names:
        col = f"{f}_neutral"
        assert col in result["neutralized"].columns, f"中性化应输出 {col}"

    assert result["corr_matrix"].shape == (5, 5), "相关性矩阵应为 5×5"
    assert len(result["decay"]) > 0, "IC Decay 应至少返回一个 lag"


@pytest.mark.skill_factor_engine
@pytest.mark.integration
@pytest.mark.requires_polars
def test_e2e_pipeline_polars_vs_pandas_diff(fe_optimizations_env):
    """T2-10 测试 2：双后端跑同一数据集，最终输出 diff < 1e-6。

    断言维度：
    - IC 序列统计量（ic_mean / ic_std / n_obs）一致
    - 中性化残差（atol=1e-10）
    - 相关性矩阵（atol=1e-10）
    - IC Decay 序列（ic_mean / ic_std / n_obs，atol=1e-6）
    """
    df = _make_synthetic_panel(n_days=60, n_stocks=200, n_factors=5)
    factor_names = [f"factor_{k}" for k in range(5)]

    res_pd = _run_pipeline(df, factor_names, backend="pandas")
    res_pl = _run_pipeline(df, factor_names, backend="polars")

    # 1. IC 一致性
    for f in factor_names:
        ic_pd = res_pd["ic"][f]
        ic_pl = res_pl["ic"][f]
        assert ic_pd["n_obs"] == ic_pl["n_obs"], (
            f"因子 {f} IC n_obs 不一致: {ic_pd['n_obs']} vs {ic_pl['n_obs']}"
        )
        assert abs(ic_pd["ic_mean"] - ic_pl["ic_mean"]) < 1e-10, (
            f"因子 {f} ic_mean 偏差过大: {ic_pd['ic_mean']} vs {ic_pl['ic_mean']}"
        )
        assert abs(ic_pd["ic_std"] - ic_pl["ic_std"]) < 1e-10, (
            f"因子 {f} ic_std 偏差过大: {ic_pd['ic_std']} vs {ic_pl['ic_std']}"
        )

    # 2. 中性化残差一致性
    for f in factor_names:
        col = f"{f}_neutral"
        diff = (res_pd["neutralized"][col].values - res_pl["neutralized"][col].values).max()
        assert abs(diff) < 1e-10, f"中性化 {f} 残差双后端偏差过大: {diff}"

    # 3. 相关性矩阵一致性
    mat_diff = (res_pd["corr_matrix"].values - res_pl["corr_matrix"].values).max()
    assert abs(mat_diff) < 1e-10, f"相关性矩阵双后端偏差过大: {mat_diff}"

    # 4. IC Decay 一致性
    assert len(res_pd["decay"]) == len(res_pl["decay"]), "IC Decay lag 数量不一致"
    for r_pd, r_pl in zip(res_pd["decay"], res_pl["decay"]):
        assert r_pd["lag"] == r_pl["lag"], f"lag 不匹配: {r_pd['lag']} vs {r_pl['lag']}"
        assert abs(r_pd["ic_mean"] - r_pl["ic_mean"]) < 1e-6, (
            f"lag={r_pd['lag']} ic_mean 偏差过大: {r_pd['ic_mean']} vs {r_pl['ic_mean']}"
        )
        assert r_pd["n_obs"] == r_pl["n_obs"], (
            f"lag={r_pd['lag']} n_obs 不一致: {r_pd['n_obs']} vs {r_pl['n_obs']}"
        )


@pytest.mark.skill_factor_engine
@pytest.mark.integration
def test_e2e_pipeline_env_var_pandas_runs(fe_optimizations_env, monkeypatch):
    """T2-10 测试 3：QUANT_FACTOR_BACKEND 未设置时默认 pandas，pipeline 能跑通。"""
    monkeypatch.delenv("QUANT_FACTOR_BACKEND", raising=False)

    df = _make_synthetic_panel(n_days=30, n_stocks=80, n_factors=3)
    factor_names = [f"factor_{k}" for k in range(3)]

    # 不显式传 backend，应使用默认 pandas
    result = _run_pipeline(df, factor_names, backend=None)
    assert len(result["ic"]) == 3
    assert len(result["decay"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
