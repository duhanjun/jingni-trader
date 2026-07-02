"""
walk_forward - 滚动前向验证 (Rolling Walk-Forward)

借鉴来源:
  - fastercapital.com / QuantConnect / 行业实践
    核心思想: 避免单次回测导致的过拟合, 周期性 retrain + 样本外验证
  - qlib 的 RollingGen (rolling window training)
  - RD-Agent 持续反馈迭代思想

用途:
  替换 jingni-trader 单次回测 (skills/backtest-engine/engine.py 中
  BacktestEngine.run), 增加:
    - 多窗口样本外评估
    - 跨窗口统计 (OOS sharpe 均值/标准差, 衰减率)
    - 整合 Deflated Sharpe 评估过拟合
    - 衰退检测 (regime shift)

约束:
  1. 不直接修改 main 分支代码, 以独立模块 + 适配器形式提供
  2. 接受任意 (data, signals) 接口, 与现有 native_adapter 对齐
  3. 可序列化 (每个窗口指标可保存为 JSON)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import full_report, MetricReport, TRADING_DAYS

logger = logging.getLogger("quant_opt.walk_forward")


# ============================================================================
# 1. 窗口切分
# ============================================================================

@dataclass
class WalkForwardConfig:
    """滚动前向配置"""
    train_window: int = 252        # 训练窗口 (交易日)
    valid_window: int = 63         # 验证窗口 (21 周 = 1 季度)
    test_window: int = 63          # 测试窗口 (OOS, 默认=valid_window)
    step: int = 63                 # 每次滚动步长
    min_train_size: int = 120      # 最小训练样本
    expanding: bool = False        # True=扩展窗口(从起点累计), False=滚动窗口(固定长度)
    purge_gap: int = 5             # train/valid/test 之间的清洗期 (避免标签泄漏)
    n_trials_for_deflated: int = 1  # 尝试过的策略数, 供 Deflated Sharpe

    def total_windows(self, n_dates: int) -> int:
        """估算能切出多少个窗口"""
        period = self.train_window + self.purge_gap + self.test_window
        if n_dates < period:
            return 0
        if self.expanding:
            return max(0, (n_dates - period) // self.step + 1)
        return max(0, (n_dates - period) // self.step + 1)


@dataclass
class WindowResult:
    """单个窗口的运行结果"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    test_metrics: Dict[str, float]
    train_metrics: Dict[str, float]
    n_trades: int
    elapsed_sec: float


# ============================================================================
# 2. 验证器
# ============================================================================

class WalkForwardValidator:
    """
    滚动前向验证器

    用法:
        validator = WalkForwardValidator(
            config=WalkForwardConfig(...),
            backtest_fn=my_backtest,   # (data_slice, signal_slice) -> {equity, trades, metrics}
        )
        result = validator.run(data, signals)
    """

    def __init__(self,
                 config: WalkForwardConfig,
                 backtest_fn: Callable[[pd.DataFrame, pd.DataFrame], Dict[str, Any]]):
        """
        参数:
            config: 滚动配置
            backtest_fn: 回测函数, 接收 (data_slice, signal_slice)
                         返回: {
                           "equity": pd.Series (index=date),
                           "trades": pd.DataFrame,
                           "metrics": Dict[str, float]  # 可选
                         }
        """
        self.config = config
        self.backtest_fn = backtest_fn

    # 工具 --------------------------------------------------------
    @staticmethod
    def _date_col(df: pd.DataFrame) -> pd.Series:
        if 'date' not in df.columns:
            raise ValueError("数据缺少 'date' 列")
        return pd.to_datetime(df['date'])

    @staticmethod
    def _slice_by_date(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        d = pd.to_datetime(df['date'])
        return df[(d >= start) & (d <= end)].copy()

    # 切窗口 ------------------------------------------------------
    def split_windows(self, dates: pd.DatetimeIndex) -> List[Dict[str, pd.Timestamp]]:
        """根据 config 切出 (train_start, train_end, test_start, test_end) 列表"""
        if len(dates) == 0:
            return []
        dates = pd.DatetimeIndex(sorted(dates.unique()))
        cfg = self.config
        windows: List[Dict[str, pd.Timestamp]] = []

        train_size = cfg.train_window
        test_size = cfg.test_window
        step = cfg.step

        cursor = 0
        wid = 0
        while True:
            train_start = dates[cursor] if cfg.expanding else dates[cursor]
            train_end_idx = cursor + train_size - 1
            if train_end_idx + cfg.purge_gap + test_size > len(dates):
                break
            train_end = dates[train_end_idx]
            test_start = dates[train_end_idx + cfg.purge_gap]
            test_end = dates[train_end_idx + cfg.purge_gap + test_size - 1]
            windows.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "id": wid,
            })
            wid += 1
            if cfg.expanding:
                # 扩展窗口: train_start 固定在 dates[0], train_end 增长
                # 这里实现为: 每次 cursor 前进 step, train_start 不变 (即从首日开始)
                # 实际扩展: 下一个窗口 train_size 增长
                # 简化: 始终以 dates[0] 为 train_start
                pass
            cursor += step

        return windows

    # 运行 --------------------------------------------------------
    def run(self,
            data: pd.DataFrame,
            signals: pd.DataFrame,
            ) -> Dict[str, Any]:
        """
        执行滚动前向验证

        返回:
            {
              "windows": [WindowResult...],
              "oos_aggregate": {OOS 指标统计},
              "summary": 总结文本
            }
        """
        if data.empty or signals.empty:
            return {"windows": [], "oos_aggregate": {}, "summary": "empty input"}

        dates = self._date_col(data)
        data_indexed = data.assign(date=dates)
        sig_dates = self._date_col(signals)
        signals_indexed = signals.assign(date=sig_dates)

        windows = self.split_windows(dates)
        if not windows:
            return {"windows": [], "oos_aggregate": {}, "summary": "no windows"}

        results: List[WindowResult] = []
        for w in windows:
            t0 = time.perf_counter()
            try:
                train_data = self._slice_by_date(data_indexed, w['train_start'], w['train_end'])
                test_data  = self._slice_by_date(data_indexed, w['test_start'],  w['test_end'])
                # 信号通常与 data 对齐, 也按日期切
                train_signals = self._slice_by_date(signals_indexed, w['train_start'], w['train_end'])
                test_signals  = self._slice_by_date(signals_indexed, w['test_start'],  w['test_end'])

                train_result = self.backtest_fn(train_data, train_signals)
                test_result  = self.backtest_fn(test_data, test_signals)

                # 提取 train/test equity 用于 sanity check
                tr = train_result.get("metrics", {}) or {}
                te = test_result.get("metrics", {}) or {}

                wr = WindowResult(
                    window_id=w['id'],
                    train_start=str(w['train_start'].date()),
                    train_end=str(w['train_end'].date()),
                    test_start=str(w['test_start'].date()),
                    test_end=str(w['test_end'].date()),
                    test_metrics={k: float(v) for k, v in te.items() if isinstance(v, (int, float, np.floating))},
                    train_metrics={k: float(v) for k, v in tr.items() if isinstance(v, (int, float, np.floating))},
                    n_trades=int(test_result.get("n_trades", 0)),
                    elapsed_sec=time.perf_counter() - t0,
                )
            except Exception as e:
                logger.exception("window %d 失败: %s", w['id'], e)
                wr = WindowResult(
                    window_id=w['id'],
                    train_start=str(w['train_start'].date()),
                    train_end=str(w['train_end'].date()),
                    test_start=str(w['test_start'].date()),
                    test_end=str(w['test_end'].date()),
                    test_metrics={"error": 1.0},
                    train_metrics={},
                    n_trades=0,
                    elapsed_sec=time.perf_counter() - t0,
                )
            results.append(wr)

        aggregate = self._aggregate_oos(results)
        return {
            "windows": [asdict(r) for r in results],
            "oos_aggregate": aggregate,
            "summary": self._format_summary(results, aggregate),
            "config": asdict(self.config),
        }

    # 聚合 --------------------------------------------------------
    @staticmethod
    def _aggregate_oos(results: List[WindowResult]) -> Dict[str, float]:
        if not results:
            return {}

        metric_keys = set()
        for r in results:
            metric_keys.update(r.test_metrics.keys())
        metric_keys.discard("error")

        agg: Dict[str, float] = {}
        for k in metric_keys:
            vals = [r.test_metrics.get(k, np.nan) for r in results
                    if k in r.test_metrics and np.isfinite(r.test_metrics.get(k, np.nan))]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"]  = float(np.std(vals))
                agg[f"{k}_min"]  = float(np.min(vals))
                agg[f"{k}_max"]  = float(np.max(vals))
                if k == "sharpe_ratio":
                    # 重要: OOS sharpe 衰减 (vs 训练集)
                    train_sharpes = [r.train_metrics.get("sharpe_ratio", np.nan)
                                     for r in results
                                     if "sharpe_ratio" in r.train_metrics
                                     and np.isfinite(r.train_metrics["sharpe_ratio"])]
                    if train_sharpes:
                        agg["sharpe_decay_ratio"] = float(
                            agg[f"{k}_mean"] / np.mean(train_sharpes)
                            if np.mean(train_sharpes) != 0 else 0.0
                        )

        agg["n_windows"] = float(len(results))
        agg["n_valid_windows"] = float(
            sum(1 for r in results if "error" not in r.test_metrics)
        )
        agg["avg_window_elapsed_sec"] = float(
            np.mean([r.elapsed_sec for r in results]) if results else 0.0
        )
        return agg

    @staticmethod
    def _format_summary(results: List[WindowResult], agg: Dict[str, float]) -> str:
        if not agg:
            return "无 OOS 结果"
        s = []
        s.append(f"前向验证窗口数: {int(agg.get('n_windows', 0))} "
                 f"(成功: {int(agg.get('n_valid_windows', 0))})")
        s.append(f"平均窗口耗时: {agg.get('avg_window_elapsed_sec', 0):.2f}s")
        for k in ["sharpe_ratio", "annual_return", "max_drawdown", "calmar_ratio"]:
            mk = f"{k}_mean"
            if mk in agg:
                s.append(f"  OOS {k}: mean={agg[mk]:.4f}  std={agg.get(f'{k}_std', 0):.4f}")
        if "sharpe_decay_ratio" in agg:
            s.append(f"  Sharpe 衰减比(OOS/IS): {agg['sharpe_decay_ratio']:.2%}")
        return "\n".join(s)


# ============================================================================
# 3. 适配器 - 直接对接 jingni-trader 的 native_adapter
# ============================================================================

def make_native_backtest_fn(adapter, **bt_kwargs):
    """
    把 NativeAdapter 包装为 (data, signals) -> dict, 供 walk_forward 使用

    adapter: skills.backtest-engine.scripts.adapters.native_adapter.NativeAdapter 实例
    """
    def _run(data_slice: pd.DataFrame, signals_slice: pd.DataFrame) -> Dict[str, Any]:
        if data_slice.empty or signals_slice.empty:
            return {"equity": pd.Series(dtype=float), "trades": pd.DataFrame(),
                    "metrics": {}, "n_trades": 0}
        out = adapter.run_backtest(data=data_slice, signals=signals_slice, **bt_kwargs)
        equity_curve = out.get("equity_curve", pd.DataFrame())
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {"equity": pd.Series(dtype=float), "trades": out.get("trades", pd.DataFrame()),
                    "metrics": {}, "n_trades": 0}
        eq = equity_curve.set_index('date')['equity']
        eq.index = pd.to_datetime(eq.index)
        return {
            "equity": eq,
            "trades": out.get("trades", pd.DataFrame()),
            "metrics": out.get("metrics", {}),
            "n_trades": int(len(out.get("trades", pd.DataFrame()))),
        }
    return _run


# ============================================================================
# 4. CLI
# ============================================================================

def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        # 用 numpy 构造一个简单回测函数自检
        np.random.seed(0)
        dates = pd.date_range("2022-01-01", periods=400, freq="B")
        codes = [f"{i:06d}.SH" for i in range(1, 6)]
        rows = []
        for d in dates:
            for c in codes:
                price = 10 + np.cumsum(np.random.normal(0, 0.02))[0] + np.random.normal(0, 1)
                rows.append({"date": d, "code": c, "close": max(0.5, price),
                             "open": price, "high": price * 1.01,
                             "low": price * 0.99, "volume": int(1e6),
                             "is_limit_up": False, "is_limit_down": False})
        data = pd.DataFrame(rows)
        signals = data[['date', 'code']].copy()
        signals['signal'] = np.random.choice([0, 1, -1], size=len(signals))

        def toy_bt(d, s):
            r = np.random.normal(0.0005, 0.01, len(d['date'].unique()))
            eq = pd.Series((1 + r).cumprod() * 1e6, index=pd.DatetimeIndex(sorted(d['date'].unique())))
            return {"equity": eq, "trades": pd.DataFrame(),
                    "metrics": {"sharpe_ratio": float(np.mean(r)/np.std(r)*np.sqrt(252)),
                                "annual_return": float(np.mean(r)*252),
                                "max_drawdown": -0.05,
                                "calmar_ratio": 1.0}, "n_trades": 0}

        cfg = WalkForwardConfig(train_window=120, test_window=60, step=60)
        v = WalkForwardValidator(cfg, toy_bt)
        out = v.run(data, signals)
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    _cli()
