"""P0-3 RuleJudge 五硬门 + 分段一致性

纯 Python 实现，无 LLM 依赖。
构造函数接收 config: dict 可覆盖默认阈值；阈值亦可通过环境变量调整。

五硬门（PRD P0-3.2，默认放宽档）：
1. sharpe >= 0.8
2. calmar >= 0.5
3. max_drawdown <= 0.35
4. completed_trades >= 50
5. segment_sharpe_ir_std <= 0.5

分段一致性算法（PRD P0-3.3）：
- 把回测期按 ~252 交易日切分（不足一段单独成段）
- 计算各段 Sharpe 的标准差
- 段数 < 2 时第 5 门跳过（记 warning 不阻塞）

环境变量（统一 QUANT_ 前缀）：
- QUANT_RULE_JUDGE_SHARPE_MIN（默认 0.8，严格档 1.0）
- QUANT_RULE_JUDGE_CALMAR_MIN（默认 0.5，严格档 0.8）
- QUANT_RULE_JUDGE_MDD_MAX（默认 0.35，严格档 0.30）
- QUANT_RULE_JUDGE_TRADES_MIN（默认 50，严格档 100）
- QUANT_RULE_JUDGE_SEG_IR_STD_MAX（默认 0.5，严格档 0.5）
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("rule-judge")


# ============================================================================
# 判定结果
# ============================================================================

@dataclass
class Verdict:
    """RuleJudge 判定结果（PRD P0-3.4）。

    字段：
        recommended_state: "candidate"（通过）或 "rejected"（未通过）
        passed_gates: 通过的门清单
        failed_gates: 未通过的门清单
        segment_stats: 分段一致性统计
                       {"segment_count": int, "segment_sharpes": List[float], "seg_ir_std": float}
        skipped_gates: 跳过的门清单（如段数 < 2 时第 5 门跳过）
    """
    recommended_state: Literal["candidate", "rejected"]
    passed_gates: List[str] = field(default_factory=list)
    failed_gates: List[str] = field(default_factory=list)
    segment_stats: Dict = field(default_factory=dict)
    skipped_gates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "recommended_state": self.recommended_state,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "segment_stats": self.segment_stats,
            "skipped_gates": self.skipped_gates,
        }


# ============================================================================
# RuleJudge
# ============================================================================

class RuleJudge:
    """回测结果五硬门评审（PRD P0-3.1）。

    纯 Python 实现，无 LLM 依赖。

    使用方式：
        judge = RuleJudge()  # 默认放宽档阈值
        verdict = judge.judge(
            metrics=result["metrics"],
            equity_curve=result["equity_curve"],
            trade_count=120,
        )
        if verdict.recommended_state == "rejected":
            logger.warning(f"策略未通过评审: {verdict.failed_gates}")
    """

    # 五硬门名称
    GATE_SHARPE = "sharpe"
    GATE_CALMAR = "calmar"
    GATE_MDD = "max_drawdown"
    GATE_TRADES = "completed_trades"
    GATE_SEG_IR_STD = "segment_sharpe_ir_std"

    # 分段长度（交易日）
    SEGMENT_LENGTH = 252

    # 严格档预设值（P0-3.7，供参考）
    STRICT_PRESET = {
        "sharpe_min": 1.0,
        "calmar_min": 0.8,
        "mdd_max": 0.30,
        "trades_min": 100,
        "seg_ir_std_max": 0.5,
    }

    def __init__(self, config: Optional[Dict] = None):
        """构造函数。

        参数：
            config: 可覆盖默认阈值，支持 keys:
                sharpe_min, calmar_min, mdd_max, trades_min, seg_ir_std_max
                未提供的 key 走环境变量，环境变量未设置走默认放宽档
        """
        config = config or {}
        self.sharpe_min = float(config.get("sharpe_min") or os.environ.get(
            "QUANT_RULE_JUDGE_SHARPE_MIN", "0.8"))
        self.calmar_min = float(config.get("calmar_min") or os.environ.get(
            "QUANT_RULE_JUDGE_CALMAR_MIN", "0.5"))
        self.mdd_max = float(config.get("mdd_max") or os.environ.get(
            "QUANT_RULE_JUDGE_MDD_MAX", "0.35"))
        self.trades_min = int(config.get("trades_min") or os.environ.get(
            "QUANT_RULE_JUDGE_TRADES_MIN", "50"))
        self.seg_ir_std_max = float(config.get("seg_ir_std_max") or os.environ.get(
            "QUANT_RULE_JUDGE_SEG_IR_STD_MAX", "0.5"))

    # ------------------------------------------------------------------------
    # 分段一致性算法（P0-3.3）
    # ------------------------------------------------------------------------

    def _compute_segment_stats(self, equity_curve: pd.DataFrame) -> Dict:
        """计算分段 Sharpe 及其标准差。

        返回：
            {
                "segment_count": int,
                "segment_sharpes": List[float],
                "seg_ir_std": float,  # 各段 Sharpe 的标准差
            }
        """
        if equity_curve is None or equity_curve.empty or "equity" not in equity_curve.columns:
            return {"segment_count": 0, "segment_sharpes": [], "seg_ir_std": 0.0}

        eq = equity_curve.copy()
        if "date" in eq.columns:
            eq = eq.sort_values("date").reset_index(drop=True)

        n = len(eq)
        if n < 2:
            return {"segment_count": 0, "segment_sharpes": [], "seg_ir_std": 0.0}

        # 按 SEGMENT_LENGTH 切分（不足一段单独成段）
        seg_len = self.SEGMENT_LENGTH
        segment_sharpes = []
        for start in range(0, n, seg_len):
            end = min(start + seg_len + 1, n)  # +1 让段间有重叠以计算 returns
            seg = eq.iloc[start:end]
            if len(seg) < 2:
                continue
            seg_eq = seg.set_index("date")["equity"] if "date" in seg.columns else seg["equity"]
            seg_returns = seg_eq.pct_change().dropna()
            if len(seg_returns) < 2:
                continue
            seg_vol = seg_returns.std()
            if seg_vol == 0 or np.isnan(seg_vol):
                continue
            # 年化：假设日线，252 交易日
            seg_annual_return = (1 + seg_returns.mean()) ** 252 - 1
            seg_annual_vol = seg_vol * np.sqrt(252)
            seg_sharpe = seg_annual_return / seg_annual_vol if seg_annual_vol != 0 else 0.0
            if not np.isnan(seg_sharpe):
                segment_sharpes.append(float(seg_sharpe))

        seg_count = len(segment_sharpes)
        if seg_count < 2:
            return {
                "segment_count": seg_count,
                "segment_sharpes": segment_sharpes,
                "seg_ir_std": 0.0,
            }

        seg_ir_std = float(np.std(segment_sharpes, ddof=1))
        return {
            "segment_count": seg_count,
            "segment_sharpes": segment_sharpes,
            "seg_ir_std": seg_ir_std,
        }

    # ------------------------------------------------------------------------
    # 五硬门评审（P0-3.4）
    # ------------------------------------------------------------------------

    def judge(
        self,
        metrics: Dict,
        equity_curve: pd.DataFrame,
        trade_count: int,
    ) -> Verdict:
        """五硬门评审。

        参数：
            metrics: 回测指标字典，需含 sharpe_ratio / calmar_ratio / max_drawdown
            equity_curve: 权益曲线 DataFrame，需含 date / equity 列
            trade_count: 完成交易笔数

        返回：
            Verdict
        """
        passed: List[str] = []
        failed: List[str] = []
        skipped: List[str] = []

        sharpe = float(metrics.get("sharpe_ratio", 0) or 0)
        calmar = float(metrics.get("calmar_ratio", 0) or 0)
        mdd = abs(float(metrics.get("max_drawdown", 0) or 0))

        # 门 1: sharpe >= sharpe_min
        if sharpe >= self.sharpe_min:
            passed.append(self.GATE_SHARPE)
        else:
            failed.append(self.GATE_SHARPE)

        # 门 2: calmar >= calmar_min
        if calmar >= self.calmar_min:
            passed.append(self.GATE_CALMAR)
        else:
            failed.append(self.GATE_CALMAR)

        # 门 3: max_drawdown <= mdd_max
        if mdd <= self.mdd_max:
            passed.append(self.GATE_MDD)
        else:
            failed.append(self.GATE_MDD)

        # 门 4: completed_trades >= trades_min
        if trade_count >= self.trades_min:
            passed.append(self.GATE_TRADES)
        else:
            failed.append(self.GATE_TRADES)

        # 门 5: segment_sharpe_ir_std <= seg_ir_std_max（段数 < 2 跳过）
        seg_stats = self._compute_segment_stats(equity_curve)
        if seg_stats["segment_count"] < 2:
            skipped.append(self.GATE_SEG_IR_STD)
            logger.warning(
                f"分段一致性跳过：段数 {seg_stats['segment_count']} < 2（不阻塞）"
            )
        else:
            if seg_stats["seg_ir_std"] <= self.seg_ir_std_max:
                passed.append(self.GATE_SEG_IR_STD)
            else:
                failed.append(self.GATE_SEG_IR_STD)

        recommended_state = "rejected" if failed else "candidate"

        return Verdict(
            recommended_state=recommended_state,
            passed_gates=passed,
            failed_gates=failed,
            segment_stats=seg_stats,
            skipped_gates=skipped,
        )
