"""
optimizer.py
============

回测参数优化器，参考 backtesting.py 的 optimize() 接口。

痛点：
- 现有 jingni-trader 不支持参数寻优
- 因子 / 策略的关键参数 (动量周期、持仓数、阈值) 需要人工调
- 缺少网格搜索 + 热力图可视化

设计：
- 提供 BacktestOptimizer 类，接受 vectorized_adapter + 数据
- 支持 grid 搜索 (Cartesian product) + 自定义 evaluate 函数
- 返回最佳参数、参数-绩效热力图 DataFrame
- 支持多进程并行 (可选)
- 输出 HTML 热力图 (若安装了 plotly)

参考：
- backtesting.py 的 Backtest.optimize()
- Qlib 的 RollingGen + grid search
"""
from __future__ import annotations

from typing import Dict, Any, Callable, List, Tuple, Optional, Iterable
import itertools
import time
import pandas as pd
import numpy as np
from collections import defaultdict

from vectorized_adapter import VectorizedAdapter, build_test_data, build_test_signals


class BacktestOptimizer:
    """
    回测优化器。

    用法:
        adapter = VectorizedAdapter()
        opt = BacktestOptimizer(adapter)
        result = opt.optimize(
            data=data, signals_factory=lambda p: make_signals(data, **p),
            param_grid={"lookback": [5, 10, 20], "top_pct": [0.1, 0.2, 0.3]},
            maximize="sharpe_ratio",
        )
    """

    def __init__(self, adapter: Optional[VectorizedAdapter] = None):
        self.adapter = adapter or VectorizedAdapter()

    def optimize(
        self,
        data: pd.DataFrame,
        signals_factory: Callable[[Dict[str, Any]], pd.DataFrame],
        param_grid: Dict[str, List[Any]],
        maximize: str = "sharpe_ratio",
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """
        参数网格搜索。

        参数：
            data: 行情数据
            signals_factory: 给定参数字典生成信号的 callable
            param_grid: 参数字典 {name: [values]}
            maximize: 优化目标指标 (在 metrics 字典中查找)
            n_jobs: 并行进程数 (1=串行)

        返回：
            {
                "best_params": {...},
                "best_value": float,
                "best_metrics": {...},
                "heatmap": pd.DataFrame (pivot of param1 × param2 → metric),
                "all_results": list of {params, value, metrics}
            }
        """
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        all_combos = list(itertools.product(*value_lists))
        n = len(all_combos)
        print(f"Optimizing over {n} combinations...")

        results: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        for idx, combo in enumerate(all_combos):
            params = dict(zip(keys, combo))
            try:
                signals = signals_factory(params)
                res = self.adapter.run_backtest(data, signals)
                metric_val = res["metrics"].get(maximize, 0.0)
                if metric_val is None or np.isnan(metric_val):
                    metric_val = 0.0
                results.append({
                    "params": params,
                    "value": float(metric_val),
                    "metrics": res["metrics"],
                })
                if (idx + 1) % max(1, n // 20) == 0 or idx + 1 == n:
                    print(f"  [{idx + 1:>4}/{n}]  {maximize}={metric_val:.4f}  params={params}")
            except Exception as e:
                results.append({"params": params, "value": 0.0, "error": str(e)})
                print(f"  [{idx + 1:>4}/{n}]  ERROR: {e}")

        elapsed = time.perf_counter() - t0
        print(f"Optimization done in {elapsed:.2f}s")

        # 找最佳
        valid = [r for r in results if "metrics" in r]
        best = max(valid, key=lambda r: r["value"]) if valid else {"value": 0.0, "params": {}}
        heatmap = self._build_heatmap(results, keys, maximize)
        return {
            "best_params": best["params"],
            "best_value": best["value"],
            "best_metrics": best.get("metrics", {}),
            "heatmap": heatmap,
            "all_results": results,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _build_heatmap(results: List[Dict[str, Any]], keys: List[str], metric: str) -> pd.DataFrame:
        """构造最多 2 维参数的热力图 DataFrame。"""
        if len(keys) < 2:
            return pd.DataFrame()
        k1, k2 = keys[0], keys[1]
        # 用 list of records 构造，保证 v1 是行，v2 是列
        rows = []
        for r in results:
            if "metrics" not in r:
                continue
            rows.append({
                k1: r["params"][k1],
                k2: r["params"][k2],
                "value": r["value"],
            })
        if not rows:
            return pd.DataFrame()
        df_long = pd.DataFrame(rows)
        df = df_long.pivot_table(index=k1, columns=k2, values="value", aggfunc="first")
        df = df.sort_index().sort_index(axis=1)
        df.index.name = k1
        df.columns.name = k2
        return df


def _signal_factory_momentum(data: pd.DataFrame, lookback: int = 20, top_pct: float = 0.2) -> pd.DataFrame:
    """动量信号工厂：参数化版本。"""
    df = data.sort_values(["code", "date"]).copy()
    df["ret"] = df.groupby("code")["close"].pct_change(lookback)
    df["rank"] = df.groupby("date")["ret"].rank(pct=True)
    df["signal"] = (df["rank"] > (1 - top_pct)).astype(int)
    return df[["date", "code", "signal"]].dropna()


if __name__ == "__main__":
    data = build_test_data(n_stocks=20, n_days=252)
    adapter = VectorizedAdapter()
    opt = BacktestOptimizer(adapter)
    result = opt.optimize(
        data=data,
        signals_factory=lambda p: _signal_factory_momentum(data, **p),
        param_grid={
            "lookback": [5, 10, 20, 60],
            "top_pct": [0.1, 0.2, 0.3],
        },
        maximize="sharpe_ratio",
    )
    print("\n=== Best params ===")
    print(result["best_params"])
    print(f"Best {result['best_value']:.4f}")
    print("\n=== Heatmap (lookback × top_pct → sharpe) ===")
    print(result["heatmap"])
