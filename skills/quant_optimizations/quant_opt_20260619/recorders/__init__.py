"""
可插拔 Recorder 系统

借鉴自 Microsoft Qlib 的 RecordTemp 设计：
  - SignalRecord:    记录模型预测信号
  - SigAnaRecord:    记录 IC / Rank IC 等信号分析
  - PortAnaRecord:   记录组合回测分析
  - BaseRecord:      抽象基类

参考：
  https://qlib.readthedocs.io/en/v0.9.6/component/recorder.html
  https://github.com/microsoft/qlib/blob/main/qlib/workflow/record_temp.py

jingni-trader 现状问题：
  - 各 stage (DATA/FACTOR/MODEL/BACKTEST/...) 的输出格式硬编码在 engine.py 中
  - 添加新分析维度（如换手率分析、归因分析）需要修改主流程
  - 缺少统一的 record 接口，artefact 命名规范不一致

本模块提供：
  - BaseRecorder 抽象基类
  - 三个具体实现：SignalRecorder / SigAnaRecorder / PortAnaRecorder
  - RecorderManager 统一管理多个 recorder 的注册与触发
"""
from __future__ import annotations

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger("quant_opt.recorders")


class BaseRecorder(ABC):
    """
    所有 Recorder 的抽象基类

    借鉴 Qlib 的 record_temp.RecordBase：
      - 命名：所有 record() 输出统一 prefix
      - 输出路径：可指向 jingni-trader 现有的 artifact 目录
    """

    def __init__(self, name: str, output_dir: str = "./quant_opt_20260619/reports"):
        self.name = name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._results: Dict[str, Any] = {}

    @abstractmethod
    def record(self, context: Dict) -> Dict:
        """执行 record 逻辑，返回 dict 形式的产物"""
        pass

    def save(self, name: str, obj: Any):
        path = os.path.join(self.output_dir, f"{self.name}__{name}.json")
        if isinstance(obj, (dict, list, str, int, float, bool)):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        elif isinstance(obj, pd.DataFrame):
            obj.to_parquet(path.replace(".json", ".parquet"), index=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(obj))
        logger.info(f"[{self.name}] saved {name} -> {path}")
        return path


class SignalRecorder(BaseRecorder):
    """
    记录模型预测信号

    借鉴自 Qlib.workflow.record_temp.SignalRecord
    """

    def record(self, context: Dict) -> Dict:
        predictions = context.get("predictions")
        if predictions is None:
            return {"success": False, "error": "predictions 不存在"}

        if isinstance(predictions, pd.DataFrame):
            self.save("predictions", predictions)
            n = len(predictions)
            signal = predictions.get("signal") or predictions.get("pred")
            metrics = {
                "n_records": n,
                "n_unique_codes": predictions["code"].nunique() if "code" in predictions.columns else 0,
                "n_unique_dates": predictions["date"].nunique() if "date" in predictions.columns else 0,
            }
            if signal is not None and pd.api.types.is_numeric_dtype(signal):
                metrics.update({
                    "signal_mean": float(signal.mean()),
                    "signal_std": float(signal.std()),
                    "signal_min": float(signal.min()),
                    "signal_max": float(signal.max()),
                })
            self.save("metrics", metrics)
            return {"success": True, "metrics": metrics}
        return {"success": False, "error": "predictions 必须是 DataFrame"}


class SigAnaRecorder(BaseRecorder):
    """
    记录信号分析 (IC / Rank IC / ICIR)

    借鉴自 Qlib.workflow.record_temp.SigAnaRecord
    """

    def record(self, context: Dict) -> Dict:
        predictions = context.get("predictions")
        forward_returns = context.get("forward_returns")
        if predictions is None or forward_returns is None:
            return {"success": False, "error": "predictions / forward_returns 缺失"}

        if isinstance(predictions, pd.DataFrame) and "pred" in predictions.columns:
            merged = predictions.merge(forward_returns, on=["code", "date"], how="inner")
        else:
            signal_col = "signal" if "signal" in predictions.columns else None
            if signal_col is None:
                return {"success": False, "error": "无有效信号列"}
            merged = predictions.rename(columns={signal_col: "pred"}).merge(
                forward_returns, on=["code", "date"], how="inner"
            )

        if merged.empty:
            return {"success": False, "error": "merge 后为空"}

        results = []
        for fwd_col in [c for c in forward_returns.columns if c.startswith("ret_forward_")]:
            sub = merged.dropna(subset=["pred", fwd_col])
            if sub.empty:
                continue
            ic_by_date = sub.groupby("date").apply(
                lambda g: g["pred"].corr(g[fwd_col], method="spearman")
            ).dropna()
            if len(ic_by_date) > 0:
                results.append({
                    "forward_period": fwd_col,
                    "ic_mean": float(ic_by_date.mean()),
                    "ic_std": float(ic_by_date.std()),
                    "ic_ir": float(ic_by_date.mean() / ic_by_date.std()) if ic_by_date.std() > 0 else 0.0,
                    "ic_positive_ratio": float((ic_by_date > 0).mean()),
                    "n_dates": int(len(ic_by_date)),
                })

        self.save("ic_analysis", results)
        return {"success": True, "n_periods": len(results), "results": results}


class PortAnaRecorder(BaseRecorder):
    """
    记录组合回测分析

    借鉴自 Qlib.workflow.record_temp.PortAnaRecord

    与 jingni-trader 现存 backtest-engine 输出的差异：
      - 不直接调用回测引擎，而是消费 backtest-engine 的 artifact (equity_curve, trades)
      - 输出标准化 JSON，便于跨 stage 对比
    """

    def record(self, context: Dict) -> Dict:
        equity_curve = context.get("equity_curve")
        trades = context.get("trades")
        risk_free_rate = context.get("risk_free_rate", 0.03)

        if equity_curve is None:
            return {"success": False, "error": "equity_curve 缺失"}

        if isinstance(equity_curve, pd.DataFrame) and "equity" in equity_curve.columns:
            eq = equity_curve.set_index("date")["equity"] if "date" in equity_curve.columns else equity_curve["equity"]
            returns = eq.pct_change().dropna()
            if len(returns) < 2:
                return {"success": False, "error": "收益序列过短"}

            total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
            n_days = len(returns)
            annual_ret = float((1 + total_ret) ** (252 / n_days) - 1)
            vol = float(returns.std() * np.sqrt(252))
            sharpe = float((annual_ret - risk_free_rate) / vol) if vol > 0 else 0.0
            max_dd = float((eq / eq.cummax() - 1).min())

            metrics = {
                "total_return": total_ret,
                "annual_return": annual_ret,
                "volatility": vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "calmar_ratio": float(annual_ret / abs(max_dd)) if max_dd != 0 else 0.0,
                "n_periods": int(n_days),
            }
        else:
            metrics = {"raw": str(equity_curve)}

        n_trades = int(len(trades)) if trades is not None and isinstance(trades, pd.DataFrame) else 0

        self.save("port_metrics", {**metrics, "n_trades": n_trades})
        return {"success": True, "metrics": metrics, "n_trades": n_trades}


class RecorderManager:
    """
    Recorder 注册中心与编排器

    借鉴自 Qlib.workflow.R (QlibRecorder)
    """

    def __init__(self, output_dir: str = "./quant_opt_20260619/reports"):
        self.output_dir = output_dir
        self.recorders: Dict[str, BaseRecorder] = {}
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"RecorderManager 初始化, output_dir={output_dir}")

    def register(self, recorder: BaseRecorder):
        self.recorders[recorder.name] = recorder
        logger.info(f"注册 recorder: {recorder.name}")
        return self

    def run_all(self, context: Dict) -> Dict[str, Dict]:
        results = {}
        for name, rec in self.recorders.items():
            try:
                results[name] = rec.record(context)
            except Exception as e:
                logger.exception(f"Recorder {name} 执行失败")
                results[name] = {"success": False, "error": str(e)}
        return results

    def list_recorders(self) -> List[str]:
        return list(self.recorders.keys())