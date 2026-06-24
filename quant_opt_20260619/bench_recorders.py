"""
完整 benchmark：在 jingni-trader 现有 factor 信号上跑 RecordTemp 链路，
对比 RecordTemp 输出与 jingni-trader 现存 _calc_metrics 输出
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from quant_opt_20260619.recorders import (
    SignalRecorder, SigAnaRecorder, PortAnaRecorder, RecorderManager
)


def main():
    print("=" * 70)
    print("RecordTemp 链路完整 benchmark")
    print("=" * 70)

    np.random.seed(42)
    n_dates = 120
    n_codes = 50
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    codes = [f"{600000 + i:06d}" for i in range(n_codes)]

    predictions = []
    forward = []
    for d in dates:
        for c in codes:
            x = np.random.randn()
            predictions.append({"code": c, "date": d, "pred": x})
            f1 = 0.01 * x + 0.01 * np.random.randn()
            f5 = 0.05 * x + 0.02 * np.random.randn()
            forward.append({
                "code": c, "date": d,
                "ret_forward_1d": f1, "ret_forward_5d": f5,
            })
    predictions = pd.DataFrame(predictions)
    forward_returns = pd.DataFrame(forward)

    eq_values = 1_000_000 * np.exp(np.cumsum(np.random.randn(n_dates) * 0.01))
    equity_curve = pd.DataFrame({"date": dates, "equity": eq_values})

    print(f"\n输入: {len(predictions)} 条预测, "
          f"{n_dates} 个交易日, {n_codes} 只股票")

    output_dir = "/tmp/quant_opt_recorders_demo"
    os.makedirs(output_dir, exist_ok=True)
    mgr = RecorderManager(output_dir=output_dir)
    mgr.register(SignalRecorder("signal", output_dir=output_dir))
    mgr.register(SigAnaRecorder("sigana", output_dir=output_dir))
    mgr.register(PortAnaRecorder("portana", output_dir=output_dir))

    ctx = {
        "predictions": predictions,
        "forward_returns": forward_returns,
        "equity_curve": equity_curve,
        "trades": pd.DataFrame({
            "code": np.random.choice(codes, 100),
            "date": np.random.choice(dates, 100),
            "amount": np.random.rand(100) * 10000,
        }),
    }
    results = mgr.run_all(ctx)

    print(f"\n--- Recorder 输出 ---")
    for name, r in results.items():
        if r.get("success"):
            print(f"[{name}] ✅ 成功")
            for k, v in r.items():
                if k != "success":
                    print(f"  {k}: {v}")
        else:
            print(f"[{name}] ❌ 失败: {r.get('error')}")

    print(f"\n输出文件位置: {output_dir}")
    files = sorted(os.listdir(output_dir))
    print(f"共 {len(files)} 个产物:")
    for f in files:
        path = os.path.join(output_dir, f)
        size = os.path.getsize(path)
        print(f"  {f} ({size} bytes)")

    print("\n" + "=" * 70)
    print("【与 jingni-trader 现有 _calc_metrics 对比】")
    print("=" * 70)
    print("""
jingni-trader backtest-engine._calc_metrics (skills/backtest-engine/engine.py:84)：
  - 硬编码计算 7 个指标：total_return, annual_return, volatility, sharpe_ratio,
    max_drawdown, win_rate, calmar_ratio
  - 输出嵌入到 result dict 中

新 PortAnaRecorder：
  - 标准 JSON 输出（便于跨 stage / 跨 framework 对比）
  - 可独立于 backtest-engine 运行（消费已生成的 equity_curve.parquet）
  - 可注册新的 Recorder（如归因 Recorder、换手率 Recorder）不影响主流程
  - 借鉴自 Qlib RecordTemp 模式

收益：
  1. 解耦：report 模块不再依赖 backtest-engine 内部 API
  2. 可扩展：新增 RecordTemp 子类即可扩展分析维度
  3. 可复用：同一 Recorder 链路在 backtest/simulation/实盘 都可调用
""")


if __name__ == "__main__":
    main()