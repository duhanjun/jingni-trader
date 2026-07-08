"""
jingni-trader 端到端集成测试（GAP-5）

打通 MasterEngine.run_pipeline 全链路，断言 DATA → FACTOR → BACKTEST → REPORT
四个阶段均产出约定产物（复用 engine.EXPECTED_ARTIFACTS 思路）。

设计要点：
- 不硬编码 /workspace 绝对路径；以仓库根（本文件所在 tests/ 的父目录）为 ROOT。
- 使用临时 QUANT_WORK_DIR，避免污染真实归档目录。
- 开启 ALLOW_SYNTHETIC_FALLBACK=true，使数据源全失败时仍能以合成数据跑完流程，
  验证整条管道在"无外部依赖"条件下的可达成性（与 test_engine_v3 的降级链验证互补）。
- Windows GBK 控制台下统一用 utf-8 输出，避免非 ASCII 日志触发 UnicodeEncodeError。
- 仅触发 DATA/FACTOR/BACKTEST/REPORT 四个阶段（意图不含 模型/组合/实盘 关键词），
  契合回测类需求的最小链路。
"""

import os
import sys
import tempfile
import shutil

# 仓库根（tests/ 的父目录），保证 `from scripts.xxx` 与 `import engine` 可解析
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Windows 控制台默认 GBK，无法编码非 ASCII 日志字符；统一用 utf-8 输出（GAP-1 同源修复）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# 测试前后环境清理（避免影响其它测试 / 真实运行）
_SAVED_ENV = {}


def _push_env(overrides: dict) -> None:
    for k, v in overrides.items():
        _SAVED_ENV[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _pop_env() -> None:
    for k, v in _SAVED_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _SAVED_ENV.clear()


def _build_synthetic_external_data() -> "pd.DataFrame":
    """构造最小可用 OHLCV DataFrame，注入 ctx.external_data 以确定性跑通 DATA 阶段。

    不依赖任何外部数据源/网络，验证全链路在无外部依赖条件下的可达成性。
    列需覆盖 data-engine 外部数据校验：code/date/open/high/low/close/volume。
    """
    import numpy as np
    import pandas as pd

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
            "code": code,
            "date": dates,
            "open": opens.round(2),
            "high": highs.round(2),
            "low": lows.round(2),
            "close": closes.round(2),
            "volume": vol,
        }))
    return pd.concat(frames, ignore_index=True)


def _run_e2e(temp_work_dir: str) -> dict:
    """在临时工作目录下跑一次最小回测链路，返回 run_pipeline 结果"""
    import engine  # MasterEngine（master scripts 包已接入 sys.path）

    intent = "获取近3年A股数据做一个反转因子选股回测并生成绩效报告"
    master = engine.MasterEngine()
    ctx = master.parse_intent(intent)
    # 明确只跑 DATA/FACTOR/BACKTEST/REPORT
    ctx.target_stages = ["DATA", "FACTOR", "BACKTEST", "REPORT"]
    # 指定具体股票池（避免空池触发「获取全市场列表」）
    ctx.stock_pool = ["000001.SZ", "600000.SH"]
    ctx.start_date = "2024-01-01"
    ctx.end_date = "2024-06-30"
    # 注入合成外部数据，确定性跑通 DATA 阶段（不依赖任何真实数据源/网络）
    ctx.external_data = {"daily": _build_synthetic_external_data(), "source": "e2e-test"}
    return master.run_pipeline(ctx=ctx)


def test_e2e_data_factor_backtest_report():
    """全链路 e2e：四个阶段均产出约定产物，且管道不在 Windows 下因日志编码崩溃"""
    temp_work_dir = tempfile.mkdtemp(prefix="jingni_trader_e2e_")
    try:
        _push_env({
            "QUANT_WORK_DIR": temp_work_dir,
            "ALLOW_SYNTHETIC_FALLBACK": "true",
            # 仅启用 websearch（无注入函数），触发合成数据兜底，
            # 验证无外部数据源依赖时仍能跑完整个管道
            "DATA_BACKENDS": "websearch",
            "LOG_LEVEL": "INFO",
        })

        results = _run_e2e(temp_work_dir)

        # 1) 管道整体成功（DATA/BACKTEST 为关键阶段，失败会整体 stop）
        assert results.get("success") is True, (
            f"全链路未成功: failed_stages={results.get('failed_stages')}, "
            f"errors={results.get('errors')}"
        )

        # 2) 四个目标阶段均被完成
        completed = set(results.get("completed_stages", []))
        for stage in ("DATA", "FACTOR", "BACKTEST", "REPORT"):
            assert stage in completed, f"阶段 {stage} 未出现在已完成列表: {completed}"

        # 3) 约定产物文件在磁盘上确实存在（复用 EXPECTED_ARTIFACTS 思路）
        import engine as engine_mod
        expected = engine_mod.EXPECTED_ARTIFACTS
        stage_dir_map = {
            "DATA": engine_mod.DATA_DIR,
            "FACTOR": engine_mod.FACTOR_DIR,
            "BACKTEST": engine_mod.BACKTEST_DIR,
            "REPORT": engine_mod.REPORT_DIR,
        }
        ctx = results.get("context", {})
        for stage in ("DATA", "FACTOR", "BACKTEST", "REPORT"):
            # 优先用 ctx.artifacts 中记录的相对/绝对路径
            artifact = (ctx.get("artifacts") or {}).get(stage) or os.path.join(
                stage_dir_map[stage], expected[stage]
            )
            assert artifact and os.path.exists(artifact), (
                f"阶段 {stage} 产物缺失: expected={expected[stage]}, resolved={artifact}"
            )
    finally:
        _pop_env()
        shutil.rmtree(temp_work_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        test_e2e_data_factor_backtest_report()
        print("总计: 1 通过, 0 失败")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"总计: 0 通过, 1 失败 -> {e}")
        sys.exit(1)
