"""
验证脚本: 向量化回测 vs native_adapter 性能与正确性对比

独立运行, 不依赖 unittest 框架, 输出详细对比结果.
解决 native_adapter 的相对导入问题.
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd

# 路径设置
_OPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_OPT_DIR, "..", ".."))
for p in [_OPT_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 注入 backtest-engine scripts 路径, 使其内部相对导入可用
_BT_SCRIPTS = os.path.join(_PROJECT_ROOT, "skills", "backtest-engine", "scripts")
if _BT_SCRIPTS not in sys.path:
    sys.path.insert(0, _BT_SCRIPTS)


def load_native_adapter():
    """加载 native_adapter, 修补其相对导入"""
    import importlib
    import importlib.util

    base_pkg_path = os.path.join(_BT_SCRIPTS, "base")
    adapters_path = os.path.join(_BT_SCRIPTS, "adapters")
    for p in [base_pkg_path, adapters_path]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # 先加载 base 模块 (无相对导入依赖)
    bbe_path = os.path.join(_BT_SCRIPTS, "base", "base_backtest_engine.py")
    spec = importlib.util.spec_from_file_location("base_backtest_engine", bbe_path)
    bbe_mod = importlib.util.module_from_spec(spec)
    sys.modules["base_backtest_engine"] = bbe_mod
    spec.loader.exec_module(bbe_mod)

    bb_path = os.path.join(_BT_SCRIPTS, "base", "base_backtest.py")
    spec = importlib.util.spec_from_file_location("base_backtest", bb_path)
    bb_mod = importlib.util.module_from_spec(spec)
    sys.modules["base_backtest"] = bb_mod
    spec.loader.exec_module(bb_mod)

    # 读取 native_adapter 源码, 替换相对导入为绝对导入
    native_path = os.path.join(_BT_SCRIPTS, "adapters", "native_adapter.py")
    with open(native_path, "r", encoding="utf-8") as f:
        source = f.read()

    source = source.replace(
        "from ..base.base_backtest_engine import BaseBacktestEngine",
        "from base_backtest_engine import BaseBacktestEngine"
    ).replace(
        "from ..base.base_backtest import BaseBacktestMetrics",
        "from base_backtest import BaseBacktestMetrics"
    )

    # 编译并执行
    module = type(sys)("native_adapter_patched")
    module.__file__ = native_path
    exec(compile(source, native_path, "exec"), module.__dict__)
    sys.modules["native_adapter_patched"] = module
    return module.NativeAdapter()


def main():
    """主验证流程"""
    print("=" * 70)
    print("向量化回测 vs native_adapter 性能与正确性对比验证")
    print("=" * 70)

    # 1. 生成测试数据
    from tests.test_data_generator import generate_synthetic_data, generate_signal_data
    from vectorized_backtest import VectorizedBacktest

    print("\n[1] 生成测试数据...")
    data = generate_synthetic_data(n_codes=30, n_days=250, seed=999)
    signals = generate_signal_data(data, fast_window=5, slow_window=20)
    print(f"    数据: {len(data)} 行, {data['code'].nunique()} 只股票, "
          f"{data['date'].nunique()} 个交易日")

    # 2. 准备信号矩阵
    entries = signals.pivot(index="date", columns="code", values="signal").fillna(0) > 0
    exits = signals.pivot(index="date", columns="code", values="signal").fillna(0) < 0
    n_entries = int(entries.values.sum())
    n_exits = int(exits.values.sum())
    print(f"    买入信号: {n_entries} 个, 卖出信号: {n_exits} 个")

    # 3. 向量化回测
    print("\n[2] 运行向量化回测...")
    bt_vec = VectorizedBacktest()
    # 预热
    bt_vec.from_signals(data, entries, exits)

    n_runs = 3
    t0 = time.perf_counter()
    for _ in range(n_runs):
        result_vec = bt_vec.from_signals(data, entries, exits, init_capital=1e6)
    t_vec = (time.perf_counter() - t0) / n_runs

    metrics_vec = result_vec["metrics"]
    print(f"    耗时: {t_vec*1000:.2f} ms")
    print(f"    交易笔数: {metrics_vec.get('total_trades', 0)}")
    print(f"    总收益: {metrics_vec.get('total_return', 0):.4%}")
    print(f"    夏普: {metrics_vec.get('sharpe_ratio', 0):.4f}")
    print(f"    最大回撤: {metrics_vec.get('max_drawdown', 0):.4%}")

    # 4. native_adapter 回测
    print("\n[3] 运行 native_adapter 回测...")
    try:
        native = load_native_adapter()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            result_native = native.run_backtest(data, signals, init_capital=1e6)
        t_native = (time.perf_counter() - t0) / n_runs

        metrics_native = result_native.get("metrics", {})
        print(f"    耗时: {t_native*1000:.2f} ms")
        print(f"    交易笔数: {metrics_native.get('total_trades', 0)}")
        print(f"    总收益: {metrics_native.get('total_return', 0):.4%}")
        print(f"    夏普: {metrics_native.get('sharpe_ratio', 0):.4f}")
        print(f"    最大回撤: {metrics_native.get('max_drawdown', 0):.4%}")

        # 5. 对比
        print("\n[4] 对比结果:")
        print(f"    {'指标':<20} {'向量化':<20} {'native':<20} {'差异':<15}")
        print(f"    {'-'*70}")
        print(f"    {'耗时(ms)':<20} {t_vec*1000:<20.2f} {t_native*1000:<20.2f} {t_native/t_vec:<15.2f}x 加速")
        for key in ["total_return", "sharpe_ratio", "max_drawdown", "total_trades"]:
            v = metrics_vec.get(key, 0)
            n = metrics_native.get(key, 0)
            diff = v - n if isinstance(v, (int, float)) else "N/A"
            print(f"    {key:<20} {v:<20} {n:<20} {diff:<15}")

        # 6. 结论
        speedup = t_native / t_vec
        print(f"\n[5] 结论:")
        print(f"    加速比: {speedup:.2f}x")
        if speedup > 1.0:
            print(f"    向量化回测比 native_adapter 快 {speedup:.2f} 倍")
        else:
            print(f"    向量化回测未比 native_adapter 快 (native 也是简化实现)")

        return {
            "vectorized_time_ms": t_vec * 1000,
            "native_time_ms": t_native * 1000,
            "speedup": speedup,
            "vectorized_metrics": metrics_vec,
            "native_metrics": metrics_native,
        }

    except Exception as e:
        print(f"    native_adapter 加载失败: {e}")
        print(f"    仅向量化回测结果: {t_vec*1000:.2f} ms")
        import traceback
        traceback.print_exc()
        return {
            "vectorized_time_ms": t_vec * 1000,
            "native_time_ms": None,
            "speedup": None,
            "vectorized_metrics": metrics_vec,
            "native_metrics": None,
            "error": str(e),
        }


if __name__ == "__main__":
    result = main()
    # 保存结果
    results_dir = os.path.join(_OPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "backtest_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {output_path}")
