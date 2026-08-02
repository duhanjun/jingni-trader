"""P0-2 三态数据质量门

纯函数式设计，不发起数据源调用，只检查已拉取的 DataFrame 字典。

三态判定规则（PRD P0-2.4）：
- abort：任一 CORE 表缺失 或 freshness > 10 交易日
- degraded：freshness > 5 交易日 或存在 PIT warning
- normal：全部通过

环境变量（统一 QUANT_ 前缀，遵循工程约定）：
- QUANT_QUALITY_GATE_FRESHNESS_ABORT_DAYS（默认 10）
- QUANT_QUALITY_GATE_FRESHNESS_DEGRADED_DAYS（默认 5）
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import pandas as pd

logger = logging.getLogger("quality-gate")


# ============================================================================
# PRD 标准表清单（参考 GemStar 原始设计，涵盖 A 股量化研究完整数据栈）
# ============================================================================

# P0-2.2 CORE 表：缺失即整体失败
STANDARD_CORE_TABLES = frozenset({
    "stock_basic",      # 股票基础信息（上市日、行业、ST 标记）
    "daily",            # 日线行情（OHLCV）
    "daily_basic",      # 日线指标（PE/PB/换手率）
    "adj_factor",       # 复权因子
    "fina_indicator",   # 财务指标
    "trade_cal",        # 交易日历
})

# P0-2.2 OPTIONAL 表：缺失仅降级
STANDARD_OPTIONAL_TABLES = frozenset({
    "forecast", "news", "top_list", "moneyflow", "margin",
    "hk_hold", "index_weight", "limit_list", "concept",
    "announcement", "express", "div", "fina_audit", "hold_ctrl",
})

# jingni-trader 适配：data-engine 实际产物的 artifact_key → PRD 标准表名
# data-engine 落盘 cleaned_data.parquet / financial.parquet / capital_flow.parquet 等，
# 通过此映射归一为 PRD 标准名后再做 CORE/OPTIONAL 判定
_ALIAS_TO_STANDARD = {
    "cleaned_data": "daily",
    "daily": "daily",
    "financial": "fina_indicator",
    "fina_indicator": "fina_indicator",
    "capital_flow": "moneyflow",
    "dragon_tiger": "top_list",
    "shareholder": "hold_ctrl",
}


# ============================================================================
# 判定结果
# ============================================================================

@dataclass
class QualityVerdict:
    """质量门判定结果（PRD P0-2.3）。

    字段：
        mode: 三态之一 "normal" / "degraded" / "abort"
        missing_core: 缺失的 CORE 表清单
        missing_optional: 缺失的 OPTIONAL 表清单
        freshness_days: daily 表最新交易日距离 asof 的自然日数（近似交易日）
        pit_warnings: P0-1 PIT 扫描产出的违规记录
        reason: 人类可读的判定原因
    """
    mode: Literal["normal", "degraded", "abort"]
    missing_core: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    freshness_days: int = 0
    pit_warnings: List[Dict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "missing_core": self.missing_core,
            "missing_optional": self.missing_optional,
            "freshness_days": self.freshness_days,
            "pit_warnings": self.pit_warnings,
            "reason": self.reason,
        }


# ============================================================================
# 质量门
# ============================================================================

class DataQualityGate:
    """三态数据质量门（PRD P0-2.1）。

    纯函数式设计，不发起数据源调用，只检查已拉取的 DataFrame 字典。

    使用方式：
        gate = DataQualityGate(core_required=["daily"])
        verdict = gate.check(
            tables={"daily": df, "fina_indicator": fin_df},
            asof="20240930",
            pit_warnings=ctx.metadata.get("pit_warnings", []),
        )
        if verdict.mode == "abort":
            return {"success": False, "error": "data_quality_abort"}
        elif verdict.mode == "degraded":
            logger.warning(f"数据质量降级: {verdict.reason}")
    """

    def __init__(
        self,
        core_required: Optional[List[str]] = None,
        freshness_abort_days: Optional[int] = None,
        freshness_degraded_days: Optional[int] = None,
    ):
        # jingni-trader 适配默认：daily 为唯一硬性 core
        # （fina_indicator 缺失时下游因子计算可降级到纯价格因子，故不强制）
        self.core_required = set(core_required) if core_required else {"daily"}

        # 环境变量覆盖（统一 QUANT_ 前缀）
        self.freshness_abort_days = freshness_abort_days or int(
            os.environ.get("QUANT_QUALITY_GATE_FRESHNESS_ABORT_DAYS", "10")
        )
        self.freshness_degraded_days = freshness_degraded_days or int(
            os.environ.get("QUANT_QUALITY_GATE_FRESHNESS_DEGRADED_DAYS", "5")
        )

    @staticmethod
    def _normalize_table_name(name: str) -> str:
        """把别名映射为 PRD 标准名（未命中则原样返回）"""
        return _ALIAS_TO_STANDARD.get(name, name)

    @staticmethod
    def _is_empty(df: Optional[pd.DataFrame]) -> bool:
        """判断 DataFrame 是否为空（None 或 empty）"""
        if df is None:
            return True
        if hasattr(df, "empty"):
            return df.empty
        return False

    @staticmethod
    def _compute_freshness(daily_df: Optional[pd.DataFrame], asof: str) -> int:
        """计算 daily 表最新交易日距离 asof 的自然日数（近似交易日）。

        asof 是 YYYYMMDD 格式。返回非负整数。
        若 daily_df 为空或无 date 列，返回大数（触发 abort）。
        """
        if daily_df is None or daily_df.empty:
            return 999
        if "date" not in daily_df.columns:
            return 999
        try:
            latest = pd.to_datetime(daily_df["date"]).max()
            asof_dt = pd.to_datetime(asof)
            delta_days = (asof_dt - latest).days
            return max(delta_days, 0)
        except Exception as e:
            logger.warning(f"freshness 计算异常: {e}")
            return 999

    def check(
        self,
        tables: Dict[str, pd.DataFrame],
        asof: str,
        pit_warnings: Optional[List[Dict]] = None,
    ) -> QualityVerdict:
        """检查已拉取的 DataFrame 字典，返回三态判定结果。

        参数：
            tables: Dict[str, pd.DataFrame] 已拉取的表
                    key 可为 PRD 标准名或别名（自动映射）
            asof: str YYYYMMDD 回测截止日（用于 freshness 计算）
            pit_warnings: Optional[List[Dict]] P0-1 PIT 扫描产出的违规记录

        返回：
            QualityVerdict
        """
        pit_warnings = pit_warnings or []

        # 把 tables 的 key 归一为 PRD 标准名
        normalized = {self._normalize_table_name(k): v for k, v in tables.items()}

        # P0-2.4 检查 core 表缺失
        missing_core = sorted([
            t for t in self.core_required
            if t not in normalized or self._is_empty(normalized.get(t))
        ])

        # P0-2.4 检查 optional 表缺失（仅记录实际传入但为空的）
        missing_optional = []
        for k, v in tables.items():
            std_name = self._normalize_table_name(k)
            if std_name in STANDARD_OPTIONAL_TABLES and self._is_empty(v):
                missing_optional.append(std_name)
        missing_optional = sorted(set(missing_optional))

        # 计算 freshness
        daily_df = normalized.get("daily")
        freshness_days = self._compute_freshness(daily_df, asof)

        # 三态判定（PRD P0-2.4 优先级：abort > degraded > normal）
        if missing_core:
            reason = f"CORE 表缺失: {missing_core}"
            return QualityVerdict(
                mode="abort",
                missing_core=missing_core,
                missing_optional=missing_optional,
                freshness_days=freshness_days,
                pit_warnings=pit_warnings,
                reason=reason,
            )

        if freshness_days > self.freshness_abort_days:
            reason = f"freshness {freshness_days} 天 > abort 阈值 {self.freshness_abort_days} 天"
            return QualityVerdict(
                mode="abort",
                missing_core=missing_core,
                missing_optional=missing_optional,
                freshness_days=freshness_days,
                pit_warnings=pit_warnings,
                reason=reason,
            )

        if freshness_days > self.freshness_degraded_days:
            reason = f"freshness {freshness_days} 天 > degraded 阈值 {self.freshness_degraded_days} 天"
            return QualityVerdict(
                mode="degraded",
                missing_core=missing_core,
                missing_optional=missing_optional,
                freshness_days=freshness_days,
                pit_warnings=pit_warnings,
                reason=reason,
            )

        if pit_warnings:
            reason = f"存在 {len(pit_warnings)} 条 PIT 违规记录"
            return QualityVerdict(
                mode="degraded",
                missing_core=missing_core,
                missing_optional=missing_optional,
                freshness_days=freshness_days,
                pit_warnings=pit_warnings,
                reason=reason,
            )

        return QualityVerdict(
            mode="normal",
            missing_core=missing_core,
            missing_optional=missing_optional,
            freshness_days=freshness_days,
            pit_warnings=pit_warnings,
            reason="all checks passed",
        )
