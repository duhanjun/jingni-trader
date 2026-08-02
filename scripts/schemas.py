"""P1-4 Pydantic V2 Schema 全链路强校验

定义全链路 Pydantic V2 模型，用于阶段间产物结构强校验。
所有模型 extra="forbid"（PRD P1-4.2），防止字段拼写错误。

字段约束（PRD P1-4.3）：
- shares: int = Field(ge=0, multiple_of=100)
- price: float = Field(gt=0)
- code: str = Field(min_length=8, max_length=10, pattern=r"^\\d{6}\\.(SH|SZ|BJ)$")
- side: Literal["buy", "sell"]

兼容性：
- 现有 run() 函数返回 dict，本模块提供 validate_payload 辅助函数做出口校验
- 校验失败时记 warning 并返回原始 dict（不阻断流程，遵循"零回归"原则）
- 各 Skill 可在出口处调用 validate_payload(result, schema_cls) 做结构校验
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger("schemas")


# ============================================================================
# 基础配置：所有 V1 模型共享 extra="forbid"
# ============================================================================

_MODEL_CONFIG = ConfigDict(extra="forbid")

# A股代码正则：6位数字 + .SH/.SZ/.BJ
_CODE_PATTERN = r"^\d{6}\.(SH|SZ|BJ)$"


# ============================================================================
# 订单与执行类 Schema
# ============================================================================

class OrderIntentV1(BaseModel):
    """下单意图（PRD P1-4.1 / 3.4）"""
    model_config = _MODEL_CONFIG

    code: str = Field(min_length=8, max_length=10, pattern=_CODE_PATTERN)
    side: Literal["buy", "sell"]
    shares: int = Field(ge=0, multiple_of=100)
    price: float = Field(gt=0)
    order_type: Literal["limit", "market"] = "limit"


class PositionSnapshotV1(BaseModel):
    """持仓快照"""
    model_config = _MODEL_CONFIG

    code: str = Field(min_length=8, max_length=10, pattern=_CODE_PATTERN)
    shares: int = Field(ge=0)
    available: int = Field(ge=0)  # 可卖数量（T+1 余额）
    cost: float = Field(ge=0)     # 持仓成本
    market_price: float = Field(ge=0)
    market_value: float = Field(ge=0)


class ExecutionReportV1(BaseModel):
    """执行报告（PRD P1-4.1 / 3.4）"""
    model_config = _MODEL_CONFIG

    version: Literal["ExecutionReportV1"] = "ExecutionReportV1"
    execution_id: str
    trade_date: str
    orders: List[OrderIntentV1] = Field(default_factory=list)
    fills: List[Dict[str, Any]] = Field(default_factory=list)
    nav_after: float
    cash_after: float
    positions_after: Dict[str, int] = Field(default_factory=dict)
    verdict: Literal["confirmed", "rejected"]
    created_at: datetime


class RiskLimitV1(BaseModel):
    """风控限额"""
    model_config = _MODEL_CONFIG

    max_position_pct: float = Field(ge=0, le=1.0)
    max_single_stock_pct: float = Field(ge=0, le=1.0)
    max_drawdown: float = Field(ge=0, le=1.0)
    daily_loss_limit: float = Field(ge=0, le=1.0)


# ============================================================================
# 回测类 Schema
# ============================================================================

class VerdictV1(BaseModel):
    """RuleJudge 判定结果 Schema（对应 P0-3 Verdict）"""
    model_config = _MODEL_CONFIG

    recommended_state: Literal["candidate", "rejected"]
    passed_gates: List[str] = Field(default_factory=list)
    failed_gates: List[str] = Field(default_factory=list)
    segment_stats: Dict[str, Any] = Field(default_factory=dict)
    skipped_gates: List[str] = Field(default_factory=list)


class BacktestResultV1(BaseModel):
    """回测结果（PRD P1-4.5）

    对应 backtest-engine run() 返回的 metadata 结构。
    兼容 dict 风格访问：result["metrics"] / BacktestResultV1.metrics
    """
    model_config = _MODEL_CONFIG

    version: Literal["BacktestResultV1"] = "BacktestResultV1"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    backend: str = "native"
    verdict: Optional[VerdictV1] = None
    trade_count: int = Field(ge=0, default=0)
    timestamp: Optional[str] = None


# ============================================================================
# 数据与因子类 Schema
# ============================================================================

class CleanedDataV1(BaseModel):
    """清洗后数据产物（PRD P1-4.5）"""
    model_config = _MODEL_CONFIG

    version: Literal["CleanedDataV1"] = "CleanedDataV1"
    path: str
    rows: int = Field(ge=0)
    columns: List[str] = Field(default_factory=list)
    asof: Optional[str] = None  # 数据截止日 YYYY-MM-DD
    quality_mode: Literal["normal", "degraded", "abort"] = "normal"


class FactorDataV1(BaseModel):
    """因子数据产物（PRD P1-4.5）"""
    model_config = _MODEL_CONFIG

    version: Literal["FactorDataV1"] = "FactorDataV1"
    path: str
    factor_names: List[str] = Field(default_factory=list)
    rows: int = Field(ge=0)
    asof: Optional[str] = None


class ReportV1(BaseModel):
    """报告产物（PRD P1-4.5）"""
    model_config = _MODEL_CONFIG

    version: Literal["ReportV1"] = "ReportV1"
    path: str
    template: Literal["technical", "fundamental", "both"] = "both"
    stock_pool: List[str] = Field(default_factory=list)
    asof: Optional[str] = None


# ============================================================================
# 校验辅助函数
# ============================================================================

def validate_payload(payload: Any, schema_cls: Type[BaseModel]) -> Tuple[bool, Optional[str]]:
    """校验 dict/payload 是否符合 schema。

    参数：
        payload: 待校验的字典或对象
        schema_cls: Pydantic V1 模型类

    返回：
        (is_valid, error_message)
        - is_valid=True 时 error_message 为 None
        - is_valid=False 时 error_message 包含校验失败详情
    """
    if payload is None:
        return False, "payload is None"
    try:
        if isinstance(payload, BaseModel):
            # 已是 Pydantic 模型，重新校验
            schema_cls.model_validate(payload.model_dump())
        else:
            schema_cls.model_validate(payload)
        return True, None
    except ValidationError as e:
        return False, str(e)


def safe_validate_payload(
    payload: Any,
    schema_cls: Type[BaseModel],
    stage: str = "",
) -> Any:
    """安全校验 payload（不阻断流程）。

    校验失败时仅记 warning 并返回原始 payload，遵循"零回归"原则。
    校验通过时也返回原始 payload（保持 dict 风格访问兼容）。

    参数：
        payload: 待校验的字典
        schema_cls: Pydantic V1 模型类
        stage: 阶段名（用于日志）

    返回：
        原始 payload（无论校验是否通过）
    """
    if payload is None:
        return payload
    is_valid, err = validate_payload(payload, schema_cls)
    if is_valid:
        logger.debug(f"[{stage}] schema 校验通过: {schema_cls.__name__}")
    else:
        logger.warning(
            f"[{stage}] schema 校验失败 ({schema_cls.__name__}): {err}"
        )
    return payload


# ============================================================================
# 各阶段产物 → 对应 Schema 映射
# ============================================================================

STAGE_SCHEMA_MAP: Dict[str, Type[BaseModel]] = {
    "DATA": CleanedDataV1,
    "FACTOR": FactorDataV1,
    "BACKTEST": BacktestResultV1,
    "EXECUTION": ExecutionReportV1,
    "REPORT": ReportV1,
}
