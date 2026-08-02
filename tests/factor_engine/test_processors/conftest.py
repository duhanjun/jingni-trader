"""test_processors 共享 fixture。

为 Processor 单元测试提供 factor-engine scripts 包的隔离环境。
遵循项目硬约束：测试独立 sys.path 隔离，测试后恢复原始缓存。
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FACTOR_ENGINE_DIR = os.path.join(ROOT, "skills", "factor-engine")


@pytest.fixture
def fe_scripts_env():
    """临时让 factor-engine 的 scripts 包接管 sys.modules。

    conftest 的 _isolate_scripts_module autouse fixture 会把 sys.modules['scripts']
    重置为主调度器的 scripts 包，导致 processors 子模块内的相对导入
    （``from scripts.processors.base import ...``）失败。本 fixture 在测试期间注册
    factor-engine 的 scripts 包到 sys.modules，使 processors 子模块可正常工作。
    """
    saved = {
        k: sys.modules.get(k)
        for k in list(sys.modules.keys())
        if k == "scripts" or k.startswith("scripts.")
    }
    # 清掉所有 scripts.* 缓存
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    # 注册 factor-engine 的 scripts 包
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
        # 恢复原始 scripts 缓存
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


@pytest.fixture
def sample_panel():
    """构造测试用因子面板数据。

    返回 (factor_df, forward_returns, factor_names)
    - factor_df: 含 code/date/factor_0/factor_1/lncap/industry 列
    - forward_returns: 含 code/date/ret_forward_1d/5d/20d 列
    - factor_names: ["factor_0", "factor_1"]
    """
    rng = np.random.RandomState(42)
    n_days = 40
    n_stocks = 50
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    industries = ["银行", "地产", "科技", "消费", "医药"]

    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            rows.append({
                "code": c,
                "date": d,
                "factor_0": rng.normal(0, 1) + i * 0.001,
                "factor_1": rng.normal(0, 1) - i * 0.001,
                "lncap": rng.normal(22, 1),
                "industry": industries[j % len(industries)],
            })
    factor_df = pd.DataFrame(rows)

    # 前瞻收益
    fwd_rows = []
    for c in codes:
        sub = factor_df[factor_df["code"] == c].sort_values("date").copy()
        closes = 10.0 + np.cumsum(rng.normal(0, 0.1, len(sub)))
        for period in [1, 5, 20]:
            sub[f"ret_forward_{period}d"] = pd.Series(closes).shift(-period) / pd.Series(closes) - 1
        fwd_rows.append(sub[["code", "date", "ret_forward_1d", "ret_forward_5d", "ret_forward_20d"]])
    forward_returns = pd.concat(fwd_rows, ignore_index=True)

    return factor_df, forward_returns, ["factor_0", "factor_1"]


@pytest.fixture
def sample_panel_with_nan(sample_panel):
    """带 NaN 的测试面板（用于测试 FillnaProcessor）"""
    factor_df, fwd, names = sample_panel
    df = factor_df.copy()
    # 随机注入 5% 的 NaN
    rng = np.random.RandomState(123)
    mask = rng.random(df["factor_0"].shape) < 0.05
    df.loc[mask, "factor_0"] = np.nan
    return df, fwd, names
