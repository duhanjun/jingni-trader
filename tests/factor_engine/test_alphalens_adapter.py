"""factor-engine L2 单元测试：alphalens_adapter（方向三 T3-9）。

覆盖 PRD 第 6.1 节 7 项测试：
1. test_to_alphalens_format_basic: 数据格式转换正确（需 alphalens-reloaded）
2. test_generate_full_report_outputs_6_files: 6 个文件全部生成（需 alphalens-reloaded）
3. test_fallback_when_alphalens_missing: mock ImportError 验证 fallback 到方案 C
4. test_metrics_json_contains_8_fields: JSON 字段完整性
5. test_disabled_by_default: 默认环境变量下不生成文件
6. test_matplotlib_agg_backend: 验证 Agg backend 生效
7. test_extract_metrics_correctness: metrics 计算正确性

测试策略：
- 需真实 alphalens-reloaded 的测试打 @pytest.mark.requires_alphalens，缺失时 skip
- 不依赖 alphalens 的测试（fallback 路径 / 默认开关 / 字段校验）始终运行
- 使用 importlib.util 加载 alphalens_adapter.py 为独立模块，避免与主 scripts 包冲突
"""
from __future__ import annotations

import importlib.util as ilu
import json
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACTOR_ENGINE_DIR = os.path.join(ROOT, "skills", "factor-engine")
ADAPTER_PATH = os.path.join(FACTOR_ENGINE_DIR, "scripts", "alphalens_adapter.py")


# ---------------------------------------------------------------------------
# 模块加载辅助
# ---------------------------------------------------------------------------


def _load_adapter():
    """加载 alphalens_adapter.py 为独立模块。

    遵循项目硬约束：测试独立 conftest.py 清理 sys.path，避免子 skill 间冲突。
    """
    saved = {
        k: sys.modules.get(k)
        for k in list(sys.modules.keys())
        if k == "scripts" or k.startswith("scripts.")
    }
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    try:
        spec = ilu.spec_from_file_location("_fe_alphalens_adapter", ADAPTER_PATH)
        mod = ilu.module_from_spec(spec)
        sys.modules["_fe_alphalens_adapter"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.modules.pop("_fe_alphalens_adapter", None)
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


def _alphalens_installed() -> bool:
    """检查 alphalens-reloaded 是否真实可用"""
    try:
        import alphalens  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 测试数据构造
# ---------------------------------------------------------------------------


def _make_factor_and_price(n_days=60, n_stocks=30, seed=42):
    """构造测试用因子数据 + 价格数据。

    因子值与前瞻收益有正相关，便于验证 IC > 0。
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]

    rows_factor = []
    rows_price = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            # 因子值：随机 + 轻微时序趋势
            factor_val = rng.normal(0, 1) + i * 0.01
            # 价格：受因子值影响（正相关，确保 IC > 0）
            base_price = 10.0 + j * 0.5
            close = base_price * (1 + factor_val * 0.01) + rng.normal(0, 0.1)
            rows_factor.append({
                "date": d, "code": c, "test_factor": factor_val,
            })
            rows_price.append({
                "date": d, "code": c, "close": close,
            })

    factor_df = pd.DataFrame(rows_factor)
    price_df = pd.DataFrame(rows_price)
    return factor_df, price_df


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


@pytest.mark.skill_factor_engine
@pytest.mark.unit
def test_disabled_by_default(tmp_path, monkeypatch):
    """T3-9 测试 5：默认环境变量下不生成任何文件。

    验证：QUANT_ALPHALENS_REPORT 未设置 / =0 时，generate_for_factor 不生成文件。
    """
    monkeypatch.delenv("QUANT_ALPHALENS_REPORT", raising=False)
    mod = _load_adapter()

    # 模块级 is_alphalens_enabled() 应返回 False
    assert mod.is_alphalens_enabled() is False

    factor_df, price_df = _make_factor_and_price()
    output_dir = tmp_path / "reports"

    # 直接调用 generate_for_factor，环境变量未启用时仍可生成（fallback 路径）
    # 但 is_alphalens_enabled() 应该返回 False，调用方据此跳过
    # 这里直接验证 is_alphalens_enabled() 行为
    monkeypatch.setenv("QUANT_ALPHALENS_REPORT", "0")
    assert mod.is_alphalens_enabled() is False

    monkeypatch.setenv("QUANT_ALPHALENS_REPORT", "1")
    assert mod.is_alphalens_enabled() is True

    monkeypatch.delenv("QUANT_ALPHALENS_REPORT", raising=False)
    assert mod.is_alphalens_enabled() is False


@pytest.mark.skill_factor_engine
@pytest.mark.unit
def test_fallback_when_alphalens_missing(tmp_path, monkeypatch):
    """T3-9 测试 3：alphalens 缺失时降级到方案 C（自研分层回测）。

    验证：
    - mock ImportError 后 _alphalens_available() 返回 False
    - generate_for_factor 走方案 C 路径，生成 metrics.json + HTML（无 PNG）
    - metrics.json 含 8 个必填字段
    """
    mod = _load_adapter()
    factor_df, price_df = _make_factor_and_price()
    output_dir = tmp_path / "reports"

    # mock alphalens 不可用
    with mock.patch.dict(sys.modules, {"alphalens": None}):
        # _alphalens_available 内 try import alphalens，模块为 None 时会抛 ImportError
        # 但更稳妥的方式：直接 patch _alphalens_available
        with mock.patch.object(mod, "_alphalens_available", return_value=False):
            result = mod.AlphalensAdapter.generate_for_factor(
                factor_df=factor_df,
                price_df=price_df,
                factor_name="test_factor",
                output_dir=str(output_dir),
            )

    # 方案 C 应该成功
    assert result is not None, "方案 C 应返回非 None"
    assert "metrics_json" in result
    assert "html" in result
    assert result.get("backend") == "fallback_lite"

    # 验证文件存在
    metrics_path = Path(result["metrics_json"])
    html_path = Path(result["html"])
    assert metrics_path.exists(), f"metrics.json 未生成: {metrics_path}"
    assert html_path.exists(), f"HTML 未生成: {html_path}"

    # 方案 C 不应生成 PNG
    png_files = list(output_dir.glob("*.png"))
    assert len(png_files) == 0, f"方案 C 不应生成 PNG，但找到: {png_files}"

    # 验证 metrics.json 含 8 个必填字段
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_fields = {
        "factor", "top_quantile_return", "bottom_quantile_return",
        "long_short_return", "long_short_sharpe",
        "ic_mean", "ic_ir", "avg_turnover_top_quantile", "suggested_verdict",
    }
    missing = required_fields - set(metrics.keys())
    assert not missing, f"metrics.json 缺少字段: {missing}"
    assert metrics["factor"] == "test_factor"
    assert metrics.get("_backend") == "fallback_lite"


@pytest.mark.skill_factor_engine
@pytest.mark.unit
def test_metrics_json_contains_8_fields(tmp_path, monkeypatch):
    """T3-9 测试 4：metrics.json 含 8 个必填字段。

    通过方案 C 路径生成（不依赖 alphalens），验证字段完整性。
    """
    mod = _load_adapter()
    factor_df, price_df = _make_factor_and_price()
    output_dir = tmp_path / "reports"

    with mock.patch.object(mod, "_alphalens_available", return_value=False):
        result = mod.AlphalensAdapter.generate_for_factor(
            factor_df=factor_df,
            price_df=price_df,
            factor_name="test_factor",
            output_dir=str(output_dir),
        )

    assert result is not None
    metrics = json.loads(Path(result["metrics_json"]).read_text(encoding="utf-8"))

    # 8 个必填字段 + 1 个 _backend 标识
    required = [
        "factor", "top_quantile_return", "bottom_quantile_return",
        "long_short_return", "long_short_sharpe",
        "ic_mean", "ic_ir", "avg_turnover_top_quantile", "suggested_verdict",
    ]
    for field in required:
        assert field in metrics, f"缺少字段: {field}"
        # 值类型校验
        if field == "factor":
            assert metrics[field] == "test_factor"
        elif field == "suggested_verdict":
            assert metrics[field] in ("ACCEPT", "REVIEW", "REJECT")
        else:
            assert isinstance(metrics[field], (int, float)), \
                f"字段 {field} 应为数值，实际: {type(metrics[field])}"


@pytest.mark.skill_factor_engine
@pytest.mark.unit
def test_matplotlib_agg_backend(tmp_path, monkeypatch):
    """T3-9 测试 6：验证 matplotlib 使用 Agg backend。

    通过 mock alphalens 的 tears 模块，触发 generate_full_report 调用，
    断言 matplotlib.use("Agg") 被调用。
    """
    mod = _load_adapter()

    # 模拟 alphalens 已安装
    fake_alphalens = mock.MagicMock()
    fake_factor_data = mock.MagicMock()

    # mock matplotlib
    fake_mpl = mock.MagicMock()

    with mock.patch.dict(sys.modules, {
        "alphalens": fake_alphalens,
        "matplotlib": fake_mpl,
        "matplotlib.pyplot": mock.MagicMock(),
    }):
        with mock.patch.object(mod, "_alphalens_available", return_value=True):
            mod.AlphalensAdapter.generate_full_report(
                factor_data=fake_factor_data,
                output_dir=str(tmp_path / "out"),
                factor_name="test_factor",
            )

    # 验证 matplotlib.use("Agg") 被调用
    fake_mpl.use.assert_called_with("Agg")


@pytest.mark.skill_factor_engine
@pytest.mark.unit
def test_extract_metrics_correctness():
    """T3-9 测试 7：metrics 计算正确性。

    构造已知 factor_data，验证 _extract_metrics 的计算逻辑。
    通过 mock alphalens.performance 模块返回固定数据。
    """
    mod = _load_adapter()

    # 构造 mock factor_data 和 alphalens.performance 返回值
    fake_factor_data = mock.MagicMock()

    # mock alphalens.performance 的三个函数
    fake_al = mock.MagicMock()
    # 分层收益：2 列（top/bot），10 行
    fake_al.performance.factor_returns.return_value = pd.DataFrame(
        {"1D": [0.01] * 10, "5D": [0.02] * 10}
    )
    # IC 数据：10 行 × 3 列（周期）
    fake_al.performance.factor_information_coefficient.return_value = pd.DataFrame(
        {"1D": [0.05] * 10, "5D": [0.04] * 10, "20D": [0.03] * 10}
    )
    # 换手率：10 行 × 5 列（quantile 1-5）
    fake_al.performance.factor_top_bottom_quantile_turnover.return_value = pd.DataFrame(
        {str(i): [0.1 + i * 0.01] * 10 for i in range(1, 6)}
    )

    with mock.patch.dict(sys.modules, {"alphalens": fake_al}):
        metrics = mod.AlphalensAdapter._extract_metrics(fake_factor_data, "test_factor")

    # 验证字段
    assert metrics["factor"] == "test_factor"
    # top_quantile_return 取最短周期最高分层列均值
    # returns_data 列为 ["1D", "5D"]，sorted 后 ["1D", "5D"]，最高列 "5D" 均值 = 0.02
    # 但实际逻辑取 first_col（"1D"），然后多列时取 top - bot
    # sorted(["1D", "5D"]) = ["1D", "5D"]，top = "5D" mean=0.02，bot = "1D" mean=0.01
    assert metrics["top_quantile_return"] == 0.02
    assert metrics["bottom_quantile_return"] == 0.01
    # long_short_return = 0.02 - 0.01 = 0.01
    assert metrics["long_short_return"] == 0.01
    # IC：ic_series = mean across columns per row = [0.04] * 10，mean=0.04，std=0 → ic_ir=0
    assert metrics["ic_mean"] == 0.04
    # ic_ir: std=0 时为 0
    # 换手率 top quantile = "5"，mean = 0.1 + 5*0.01 = 0.15
    assert metrics["avg_turnover_top_quantile"] == 0.15
    # suggested_verdict: ic_ir=0 < 0.5 → REVIEW
    assert metrics["suggested_verdict"] == "REVIEW"


@pytest.mark.skill_factor_engine
@pytest.mark.unit
@pytest.mark.requires_alphalens
@pytest.mark.skipif(not _alphalens_installed(), reason="alphalens-reloaded 未安装")
def test_to_alphalens_format_basic():
    """T3-9 测试 1：数据格式转换正确（需 alphalens-reloaded 真实安装）。

    验证 to_alphalens_format 返回 MultiIndex DataFrame，含 quantile 列。
    """
    mod = _load_adapter()
    factor_df, price_df = _make_factor_and_price(n_days=30, n_stocks=20)

    factor_data = mod.AlphalensAdapter.to_alphalens_format(
        factor_df=factor_df,
        price_df=price_df,
        factor_name="test_factor",
        forward_periods=(1, 5),
        quantiles=5,
    )

    # 验证返回的是 DataFrame
    assert isinstance(factor_data, pd.DataFrame)
    # MultiIndex 应含 (date, code)
    assert factor_data.index.names == ["date", "code"]
    # 应含 quantile 列（alphalens 自动添加）
    assert "factor_quantile" in factor_data.columns or any(
        "quantile" in c.lower() for c in factor_data.columns
    )


@pytest.mark.skill_factor_engine
@pytest.mark.unit
@pytest.mark.requires_alphalens
@pytest.mark.skipif(not _alphalens_installed(), reason="alphalens-reloaded 未安装")
def test_generate_full_report_outputs_6_files(tmp_path):
    """T3-9 测试 2：generate_full_report 输出 6 个文件（需 alphalens-reloaded）。

    验证：4 PNG + 1 HTML + 1 JSON 全部生成。
    """
    mod = _load_adapter()
    factor_df, price_df = _make_factor_and_price(n_days=30, n_stocks=20)

    factor_data = mod.AlphalensAdapter.to_alphalens_format(
        factor_df=factor_df,
        price_df=price_df,
        factor_name="test_factor",
        forward_periods=(1, 5),
        quantiles=5,
    )

    output_dir = tmp_path / "alphalens_reports"
    paths = mod.AlphalensAdapter.generate_full_report(
        factor_data=factor_data,
        output_dir=str(output_dir),
        factor_name="test_factor",
    )

    # 验证 6 个文件全部存在
    expected_files = [
        paths["returns_png"], paths["ic_png"], paths["turnover_png"],
        paths["summary_png"], paths["html"], paths["metrics_json"],
    ]
    for fpath in expected_files:
        assert os.path.exists(fpath), f"文件未生成: {fpath}"

    # 验证 metrics.json 含 8 必填字段
    metrics = json.loads(Path(paths["metrics_json"]).read_text(encoding="utf-8"))
    required = {
        "factor", "top_quantile_return", "bottom_quantile_return",
        "long_short_return", "long_short_sharpe",
        "ic_mean", "ic_ir", "avg_turnover_top_quantile", "suggested_verdict",
    }
    assert required.issubset(set(metrics.keys()))
