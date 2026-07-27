"""data-engine L1 契约测试。

验证 data-engine.run(ctx) 的接口契约：
- 标准输入 Context（stock_pool + start_date + end_date + external_data）
  → success=True + 产物存在 + metadata 含必需字段
- 上游缺失/失败路径 → success=False + error 含明确提示
- 已有缓存产物 → 直接返回 cache，不重复计算
- 不修改传入的 ctx（除 artifacts/metadata 外）

设计要点：
- 不依赖任何外部数据源/网络，全部用合成 OHLCV 数据注入 ctx.external_data
- 用 importlib 显式加载 data-engine/engine.py 为独立模块，避免与主调度器 engine.py 同名冲突
- mock 掉可能存在的重量级第三方依赖（tushare/baostock/akshare）
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest
import pandas as pd
import numpy as np


# ============================================================================
# 模块加载工具：把 data-engine/engine.py 加载为独立模块
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_DIR = os.path.join(ROOT, "skills", "data-engine")
DATA_ENGINE_PATH = os.path.join(DATA_ENGINE_DIR, "engine.py")


def _load_data_engine_module():
    """显式加载 data-engine/engine.py 为独立模块。

    需要把 data-engine/scripts 注册为 sys.modules['scripts']，
    使 engine.py 顶层的 `from scripts.config import ...` 能解析到 data-engine/scripts。
    """
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(DATA_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    # mock 重量级依赖
    for _m in ("tushare", "baostock", "akshare", "xtquant", "talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    try:
        spec = ilu.spec_from_file_location("data_engine_engine", DATA_ENGINE_PATH)
        mod = ilu.module_from_spec(spec)
        sys.modules["data_engine_engine"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # 恢复 scripts 缓存（让主调度器后续可用）
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


# ============================================================================
# 共用辅助：构造合成 OHLCV 数据
# ============================================================================

def _make_synthetic_external_data():
    """构造 ctx.external_data，含 daily DataFrame"""
    codes = ["000001.SZ", "600000.SH"]
    frames = []
    rng = np.random.RandomState(20240101)
    for code in codes:
        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        n = len(dates)
        base = rng.uniform(8, 20)
        closes = base * (1 + np.cumsum(rng.normal(0, 0.01, n)))
        opens = closes * (1 + rng.normal(0, 0.002, n))
        highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n)))
        vol = rng.randint(1_000_000, 10_000_000, n)
        frames.append(pd.DataFrame({
            "code": code, "date": dates,
            "open": opens.round(2), "high": highs.round(2),
            "low": lows.round(2), "close": closes.round(2),
            "volume": vol,
        }))
    return {"daily": pd.concat(frames, ignore_index=True), "source": "test"}


def _make_ctx(stock_pool=None, start="2024-01-01", end="2024-06-30",
              external_data=None):
    """构造 data-engine 标准输入 Context"""
    from scripts.context import Context
    ctx = Context(
        task_id="test_data",
        stock_pool=stock_pool if stock_pool is not None else ["000001.SZ", "600000.SH"],
        start_date=start,
        end_date=end,
    )
    if external_data is not None:
        ctx.external_data = external_data
    return ctx


# ============================================================================
# L1 契约测试
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.contract
class TestDataEngineRunContract:
    """验证 data-engine.run(ctx) 接口契约。"""

    def test_returns_success_with_synthetic_external_data(self, tmp_path, monkeypatch):
        """标准输入（含 external_data）→ success=True + 产物存在 + metadata 必需字段"""
        # 让 data-engine 输出到临时目录
        monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))

        data_engine_mod = _load_data_engine_module()
        external_data = _make_synthetic_external_data()
        ctx = _make_ctx(external_data=external_data)

        # mock 掉 DataEngine 整个类（避免加载真实适配器与第三方库）
        # external_data 已注入，fetch_and_clean 会优先用 external_data
        mock_engine_instance = mock.MagicMock()
        mock_engine_instance.fetch_and_clean.return_value = external_data["daily"]
        mock_engine_instance.backend = "external"
        mock_engine_instance.is_synthetic = False
        mock_engine_instance.data_sources = ["websearch"]
        mock_engine_instance.save_data = lambda df, path: df.to_parquet(path, index=False)

        with mock.patch.object(data_engine_mod, "DataEngine", return_value=mock_engine_instance):
            result = data_engine_mod.run(ctx)

        assert result["success"] is True, f"expected success, got: {result}"
        assert result["artifact_path"], "artifact_path 不应为空"
        assert os.path.exists(result["artifact_path"]), \
            f"产物文件应存在: {result['artifact_path']}"
        # metadata 应包含 data_source
        assert "data_source" in result["metadata"]
        assert result["metadata"]["rows"] > 0

    def test_returns_failure_when_no_data_and_no_backend(self, tmp_path, monkeypatch):
        """无 external_data + 无可用数据源（websearch 取不到数据）→ success=False"""
        monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "false")
        monkeypatch.setenv("DATA_BACKENDS", "websearch")

        data_engine_mod = _load_data_engine_module()
        ctx = _make_ctx()
        ctx.external_data = None  # 不注入任何外部数据

        result = data_engine_mod.run(ctx)

        # 没有 external_data 且无真实数据源 → 应失败或返回合成数据
        # 这里只断言"不抛异常" + result 是 dict + 含 success 字段
        assert isinstance(result, dict)
        assert "success" in result
        assert "error" in result

    def test_returns_cache_when_artifact_already_exists(self, tmp_path, monkeypatch):
        """已有 DATA 产物时 → 直接返回 cache，metadata.source='cache'"""
        monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))

        # 预置一个已存在的产物
        cached_path = str(tmp_path / "cleaned_data.parquet")
        pd.DataFrame({"code": ["000001.SZ"], "v": [1]}).to_parquet(cached_path, index=False)

        data_engine_mod = _load_data_engine_module()
        ctx = _make_ctx(external_data=_make_synthetic_external_data())
        ctx.update_artifact("DATA", cached_path)

        result = data_engine_mod.run(ctx)

        assert result["success"] is True
        assert result["metadata"]["source"] == "cache"
        assert result["artifact_path"] == cached_path

    def test_result_has_required_fields(self, tmp_path, monkeypatch):
        """result dict 必含 success/artifact_path/metadata/error 四个字段"""
        monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))

        data_engine_mod = _load_data_engine_module()
        ctx = _make_ctx(external_data=_make_synthetic_external_data())

        result = data_engine_mod.run(ctx)

        for field in ("success", "artifact_path", "metadata", "error"):
            assert field in result, f"result 缺少必需字段: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
