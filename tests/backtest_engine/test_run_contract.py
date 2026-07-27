"""backtest-engine L1 契约测试。

验证 backtest-engine.run(ctx) 的接口契约：
- 上游 DATA 产物缺失 → success=False + error 含"数据产物不存在"
- 上游 DATA 存在但 MODEL/FACTOR 缺失 → 走默认策略或失败（不抛异常）
- result dict 含必需字段（success/artifact_path/metadata/error）

成功路径（实际跑回测生成 backtest_result.json）需要完整上游产物，
本测试只覆盖契约边界。
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest
import pandas as pd
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKTEST_ENGINE_DIR = os.path.join(ROOT, "skills", "backtest-engine")
BACKTEST_ENGINE_PATH = os.path.join(BACKTEST_ENGINE_DIR, "engine.py")


def _load_backtest_engine_module():
    """显式加载 backtest-engine/engine.py 为独立模块。"""
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(BACKTEST_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    for _m in ("backtrader", "talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    try:
        spec = ilu.spec_from_file_location("backtest_engine_engine", BACKTEST_ENGINE_PATH)
        mod = ilu.module_from_spec(spec)
        sys.modules["backtest_engine_engine"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


def _make_data_parquet(tmp_path):
    """构造最小可用 DATA parquet（OHLCV）"""
    codes = ["000001.SZ", "600000.SH"]
    frames = []
    rng = np.random.RandomState(42)
    for code in codes:
        dates = pd.bdate_range("2024-01-01", "2024-03-31")
        n = len(dates)
        closes = 10 * (1 + np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({
            "code": code, "date": dates,
            "open": closes, "high": closes * 1.01,
            "low": closes * 0.99, "close": closes,
            "volume": rng.randint(1_000_000, 5_000_000, n),
        }))
    df = pd.concat(frames, ignore_index=True)
    path = str(tmp_path / "cleaned_data.parquet")
    df.to_parquet(path, index=False)
    return path


def _make_ctx(stock_pool=None):
    """构造 backtest-engine 标准输入 Context"""
    from scripts.context import Context
    return Context(
        task_id="test_backtest",
        stock_pool=stock_pool or ["000001.SZ", "600000.SH"],
        start_date="2024-01-01",
        end_date="2024-06-30",
    )


@pytest.mark.skill_backtest_engine
@pytest.mark.contract
class TestBacktestEngineRunContract:
    """验证 backtest-engine.run(ctx) 接口契约。"""

    def test_returns_failure_when_data_artifact_missing(self, monkeypatch, tmp_path):
        """DATA 产物未注册 → success=False + error 含"数据产物不存在" """
        monkeypatch.setenv("QUANT_BACKTEST_DIR", str(tmp_path))

        backtest_mod = _load_backtest_engine_module()
        ctx = _make_ctx()
        # 不更新 DATA 产物

        result = backtest_mod.run(ctx)

        assert result["success"] is False
        assert "数据产物不存在" in result["error"]

    def test_returns_failure_when_data_path_not_exists(self, monkeypatch, tmp_path):
        """DATA 路径注册但文件不存在 → success=False"""
        monkeypatch.setenv("QUANT_BACKTEST_DIR", str(tmp_path))

        backtest_mod = _load_backtest_engine_module()
        ctx = _make_ctx()
        ctx.update_artifact("DATA", "/nonexistent/data.parquet")

        result = backtest_mod.run(ctx)

        assert result["success"] is False

    def test_result_has_required_fields(self, monkeypatch, tmp_path):
        """result dict 必含 success/artifact_path/metadata/error 四个字段"""
        monkeypatch.setenv("QUANT_BACKTEST_DIR", str(tmp_path))

        backtest_mod = _load_backtest_engine_module()
        ctx = _make_ctx()
        ctx.update_artifact("DATA", "/nonexistent/data.parquet")

        result = backtest_mod.run(ctx)

        for field in ("success", "artifact_path", "metadata", "error"):
            assert field in result, f"result 缺少必需字段: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
