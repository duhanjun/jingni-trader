"""P1-1 追加式 JSONL Paper Trading 账本

替代原 PaperExecutor 的覆盖式 account_state.json，改为追加式事务日志：
- 每笔成交追加一条 record 到 ledger.jsonl
- 启动时 replay_ledger 重建账户状态
- confirmed=True 硬约束（防止 LLM 输出直接落账）
- T+1 强制（卖出当日买入标的 raise）

旧状态迁移（P1-1.6a）：
- 检测到 account_state.json 存在但 ledger.jsonl 不存在时
- 备份原文件 → 生成 opening_balance record → 写入 ledger.jsonl

环境变量：
- PAPER_LEDGER_PATH：账本路径（默认 <EXECUTION_DIR>/ledger.jsonl）
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger("paper-ledger")


# ============================================================================
# PaperTradeRecordV1（PRD P1-1.1）
# ============================================================================

_CODE_PATTERN = r"^\d{6}\.(SH|SZ|BJ)$"


class PaperTradeRecordV1(BaseModel):
    """单笔 Paper Trading 成交记录（JSONL 一行一条）"""

    model_config = ConfigDict(extra="forbid")

    execution_id: str  # 唯一标识
    trade_date: str    # YYYY-MM-DD
    code: str = Field(min_length=8, max_length=10, pattern=_CODE_PATTERN)
    side: Literal["buy", "sell"]
    shares: int = Field(ge=0, multiple_of=100)
    price: float = Field(gt=0)
    commission: float = Field(ge=0, default=0.0)
    stamp_tax: float = Field(ge=0, default=0.0)
    slippage_cost: float = 0.0
    position_after_shares: int = Field(ge=0)
    cash_after: float
    nav_after: float
    confirmed: bool = True  # 必填=True
    created_at: datetime

    @field_validator("confirmed")
    @classmethod
    def _confirmed_must_be_true(cls, v: bool) -> bool:
        """P1-1.7: confirmed=False 的 record 写入时 raise"""
        if not v:
            raise ValueError("confirmed must be True (P1-1.7 硬约束)")
        return v


# ============================================================================
# AccountSnapshot（replay_ledger 重建结果）
# ============================================================================

@dataclass
class PositionState:
    """单标的持仓状态"""
    shares: int = 0          # 总持仓
    available: int = 0       # 可卖（T+1 余额）
    cost: float = 0.0        # 持仓成本


@dataclass
class AccountSnapshot:
    """replay_ledger 重建的账户快照"""
    positions: Dict[str, PositionState] = field(default_factory=dict)
    cash: float = 0.0
    nav: float = 0.0
    bought_today: Set[str] = field(default_factory=set)
    last_trade_date: str = ""


# ============================================================================
# 路径配置
# ============================================================================

def get_default_ledger_path() -> str:
    """获取默认 ledger 路径（受环境变量 PAPER_LEDGER_PATH 覆盖）"""
    return os.environ.get(
        "PAPER_LEDGER_PATH",
        os.path.join(os.environ.get("EXECUTION_DIR", "./workspace/execution"), "ledger.jsonl"),
    )


# ============================================================================
# 追加 / 读取 / 重放（PRD P1-1.2 / P1-1.3 / P1-1.4）
# ============================================================================

def append_paper_trade(path: Path, record: PaperTradeRecordV1) -> None:
    """追加一条成交记录到 ledger.jsonl（PRD P1-1.2）。

    - 写入前扫整文件去重（execution_id 重复 raise）
    - confirmed=False raise（P1-1.7）
    - 追加模式写入
    """
    path = Path(path)
    # confirmed=False 在 Pydantic 构造时已校验，此处双重保险
    if not record.confirmed:
        raise ValueError("confirmed must be True (P1-1.7 硬约束)")

    # 去重检查
    if path.exists():
        existing_ids = _read_execution_ids(path)
        if record.execution_id in existing_ids:
            raise ValueError(
                f"execution_id 重复: {record.execution_id} (P1-1.2 去重)"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
    logger.debug(f"追加 paper trade: {record.execution_id} {record.side} {record.code}")


def _read_execution_ids(path: Path) -> Set[str]:
    """读取 ledger 中所有 execution_id（用于去重）"""
    ids: Set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                eid = obj.get("execution_id")
                if eid:
                    ids.add(eid)
            except Exception:
                continue
    return ids


def read_paper_trades(path: Path) -> List[PaperTradeRecordV1]:
    """读取全部成交记录（PRD P1-1.3）。

    逐行 model_validate_json；损坏行 skip + warning。
    """
    path = Path(path)
    if not path.exists():
        return []
    records: List[PaperTradeRecordV1] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = PaperTradeRecordV1.model_validate_json(line)
                records.append(rec)
            except (ValidationError, Exception) as e:
                logger.warning(
                    f"ledger 损坏行已跳过 (P1-1.3): {path}:{lineno} - {e}"
                )
                continue
    return records


def replay_ledger(path: Path, init_capital: float = 0.0) -> AccountSnapshot:
    """顺序重放 ledger 重建账户状态（PRD P1-1.4）。

    - 顺序读 JSONL 重建 AccountSnapshot
    - T+1 强制：sell/reduce 当日买入的标的 raise
    - ledger 不存在时返回空快照（cash=init_capital）
    """
    path = Path(path)
    records = read_paper_trades(path)

    snapshot = AccountSnapshot(cash=init_capital, nav=init_capital)

    for rec in records:
        # 日期切换：清空 bought_today + 释放 T+1 可卖额度
        if snapshot.last_trade_date and rec.trade_date != snapshot.last_trade_date:
            snapshot.bought_today.clear()
            # T+1: 次日所有持仓变为可卖
            for pos in snapshot.positions.values():
                pos.available = pos.shares
        snapshot.last_trade_date = rec.trade_date

        pos = snapshot.positions.get(rec.code, PositionState())

        if rec.side == "buy":
            # 买入：增加持仓，当日不可卖
            old_cost = pos.cost * pos.shares
            new_shares = pos.shares + rec.shares
            pos.cost = (old_cost + rec.price * rec.shares) / new_shares if new_shares > 0 else 0.0
            pos.shares = new_shares
            # available 不变（T+1）
            snapshot.bought_today.add(rec.code)
        elif rec.side == "sell":
            # T+1 强制：卖出当日买入标的 raise
            if rec.code in snapshot.bought_today:
                raise ValueError(
                    f"T+1 违规: 尝试卖出当日买入标的 {rec.code} (trade_date={rec.trade_date})"
                )
            if pos.available < rec.shares:
                raise ValueError(
                    f"可用持仓不足: {rec.code} 需要 {rec.shares}, 可用 {pos.available}"
                )
            pos.shares -= rec.shares
            pos.available -= rec.shares

        if pos.shares <= 0:
            snapshot.positions.pop(rec.code, None)
        else:
            snapshot.positions[rec.code] = pos

        # 用 record 中的 after 值更新快照
        snapshot.cash = rec.cash_after
        snapshot.nav = rec.nav_after

    return snapshot


# ============================================================================
# 旧状态迁移（PRD P1-1.6a）
# ============================================================================

def migrate_legacy_state(
    ledger_path: Path,
    state_path: Path,
    init_capital: float = 0.0,
) -> bool:
    """旧 account_state.json → ledger.jsonl 迁移（P1-1.6a）。

    触发条件：account_state.json 存在但 ledger.jsonl 不存在。

    流程：
    1. 备份 account_state.json → account_state.json.bak.<timestamp>
    2. 读取旧状态，校验字段完整性（nav/cash/positions 齐全）
    3. 校验通过 → 生成 opening_balance record 写入 ledger.jsonl
    4. 校验失败 → raise

    返回：True 表示已迁移，False 表示无需迁移（ledger 已存在或 state 不存在）
    """
    ledger_path = Path(ledger_path)
    state_path = Path(state_path)

    # ledger 已存在 → 无需迁移
    if ledger_path.exists():
        return False
    # state 不存在 → 无需迁移
    if not state_path.exists():
        return False

    logger.info(f"检测到旧状态文件，开始迁移: {state_path} → {ledger_path}")

    # 1. 备份原文件
    backup_path = state_path.with_suffix(
        f".json.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    shutil.copy2(state_path, backup_path)
    logger.info(f"已备份旧状态文件: {backup_path}")

    # 2. 读取旧状态
    with state_path.open("r", encoding="utf-8") as f:
        legacy = json.load(f)

    # 3. 字段完整性校验
    nav = legacy.get("nav")
    cash = legacy.get("available_cash", legacy.get("cash"))
    positions = legacy.get("positions", {})

    if nav is None or cash is None:
        raise ValueError(
            "旧状态文件字段缺失 (nav/available_cash)，"
            "请使用 INIT_CAPITAL 重新启动（P1-1.6a）"
        )
    if not isinstance(positions, dict):
        raise ValueError("旧状态文件 positions 字段格式错误（应为 dict）")

    # 4. 生成 opening_balance record
    ts = datetime.now()
    # positions → position_after_shares 取第一个标的（或 0 占位）
    # opening_balance 是特殊记录：side=buy, shares=0, price=0 占位
    # 为每个有持仓的标的生成一条 record
    if positions:
        for code, pos_data in positions.items():
            shares = pos_data.get("volume", pos_data.get("shares", 0))
            record = PaperTradeRecordV1(
                execution_id=f"opening_balance_{ts.strftime('%Y%m%d%H%M%S')}_{code}",
                trade_date=ts.strftime("%Y-%m-%d"),
                code=code,
                side="buy",
                shares=0,  # 占位，不改变持仓
                price=0.01,  # price>0 约束，用最小值占位
                commission=0.0,
                stamp_tax=0.0,
                slippage_cost=0.0,
                position_after_shares=shares,
                cash_after=cash,
                nav_after=nav,
                confirmed=True,
                created_at=ts,
            )
            append_paper_trade(ledger_path, record)
    else:
        # 无持仓，生成一条空 record
        record = PaperTradeRecordV1(
            execution_id=f"opening_balance_{ts.strftime('%Y%m%d%H%M%S')}",
            trade_date=ts.strftime("%Y-%m-%d"),
            code="000001.SZ",  # 占位代码
            side="buy",
            shares=0,
            price=0.01,
            commission=0.0,
            stamp_tax=0.0,
            slippage_cost=0.0,
            position_after_shares=0,
            cash_after=cash,
            nav_after=nav,
            confirmed=True,
            created_at=ts,
        )
        append_paper_trade(ledger_path, record)

    logger.info(f"旧状态迁移完成: {len(positions)} 个持仓标的")
    return True
