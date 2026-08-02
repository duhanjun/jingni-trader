"""PIT (Point-in-Time) 强制契约模块。

防止量化回测中的 look-ahead bias：确保财务数据只使用"截至某交易日已披露"的部分。

核心契约：
    1. 财务数据 DataFrame 必须含 `disclosure_date` 列（披露日）
    2. 下游使用财务数据时必须调用 `pit_filter(df, asof)` 过滤
    3. 缺 `disclosure_date` 列时 raise ValueError（fail fast，不静默放过）

设计原则：
    - 纯函数式，不依赖外部状态
    - 不发起数据源调用，只校验已拉取的 DataFrame
    - 日志优先级：warning（过滤）/ error（缺列 raise）

环境变量：
    QUANT_PIT_STRICT: "true"（默认）强制契约；"false" 时降级为 warning（仅记日志不过滤）
"""
from __future__ import annotations

import logging
import os
from typing import List, Dict, Any, Optional

import pandas as pd

logger = logging.getLogger("pit")


def _is_strict_mode() -> bool:
    """是否严格模式（缺列 raise）。false 时降级为 warning。"""
    return os.environ.get("QUANT_PIT_STRICT", "true").lower() in ("1", "true", "yes")


def pit_filter(df: pd.DataFrame, asof: str) -> pd.DataFrame:
    """PIT 哨兵函数：过滤掉披露日晚于 asof 的财务数据。

    防止 look-ahead bias：回测某交易日时，只能使用该交易日"已披露"的财务数据。
    例如某公司 2024-04-30 发布 2024Q1 报告（report_date=20240331, disclosure_date=20240430），
    若 asof=20240415（Q1 报告发布前），则该行必须被过滤掉。

    参数:
        df: 财务数据 DataFrame，必须含 `disclosure_date` 列
        asof: 截止日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'

    返回:
        过滤后的 DataFrame（仅保留 disclosure_date <= asof 的行）

    raises:
        ValueError: 当 df 缺 `disclosure_date` 列且 QUANT_PIT_STRICT=true 时
    """
    if df is None or df.empty:
        return df

    if "disclosure_date" not in df.columns:
        msg = (
            "PIT 契约违规：财务数据缺少 `disclosure_date` 列。"
            "请确认 adapter 出口已追加该字段（缺值时回填为 report_date）。"
        )
        if _is_strict_mode():
            raise ValueError(msg)
        logger.warning(msg + "（QUANT_PIT_STRICT=false，降级为 warning，未过滤）")
        return df

    # 统一日期格式为 YYYYMMDD 字符串（去掉横线）便于字符串比较
    asof_norm = str(asof).replace("-", "")
    disc = df["disclosure_date"].astype(str).str.replace("-", "")

    mask = disc <= asof_norm
    filtered_out = (~mask).sum()
    if filtered_out > 0:
        logger.warning(
            f"PIT 过滤：剔除 {filtered_out} 行未来披露数据"
            f"（disclosure_date > {asof_norm}）"
        )
    return df[mask].copy()


def scan_pit_warnings(
    df: pd.DataFrame,
    asof: str,
    table_name: str = "financial",
) -> List[Dict[str, Any]]:
    """扫描财务数据中的 PIT 违规行（不修改原 df，仅记录 warning）。

    用于 data-engine 出口扫描：检测是否存在披露日晚于 asof 的行，
    违规行信息写入 ctx.metadata["pit_warnings"] 供下游参考。

    参数:
        df: 财务数据 DataFrame（应含 disclosure_date 列）
        asof: 截止日期
        table_name: 表名标识（用于 warning 信息）

    返回:
        PIT 违规行信息列表，每项 {"code":..., "report_date":..., "disclosure_date":..., "asof":...}
        若 df 缺 disclosure_date 列，返回 [{"error": "missing_disclosure_date", "table": ...}]
    """
    if df is None or df.empty:
        return []

    if "disclosure_date" not in df.columns:
        logger.warning(
            f"PIT 扫描：{table_name} 表缺 disclosure_date 列，无法做 PIT 校验"
        )
        return [{"error": "missing_disclosure_date", "table": table_name}]

    asof_norm = str(asof).replace("-", "")
    disc = df["disclosure_date"].astype(str).str.replace("-", "")
    future_mask = disc > asof_norm
    if not future_mask.any():
        return []

    warnings: List[Dict[str, Any]] = []
    for _, row in df[future_mask].iterrows():
        warnings.append({
            "table": table_name,
            "code": str(row.get("code", "")),
            "report_date": str(row.get("report_date", "")),
            "disclosure_date": str(row.get("disclosure_date", "")),
            "asof": asof_norm,
        })
    logger.warning(
        f"PIT 扫描：{table_name} 表发现 {len(warnings)} 行未来披露数据"
        f"（disclosure_date > {asof_norm}），将在出口过滤"
    )
    return warnings


def ensure_pit_filtered(df: pd.DataFrame, asof: str, caller: str = "") -> pd.DataFrame:
    """强制 PIT 过滤的守卫函数（用于 factor-engine / backtest-engine）。

    与 pit_filter 的区别：
    - pit_filter: 通用过滤函数，可被任何模块调用
    - ensure_pit_filtered: 守卫函数，记录调用方信息，用于审计下游是否真的做了 PIT

    若 QUANT_PIT_STRICT=true 且 df 缺 disclosure_date 列，raise ValueError；
    若 QUANT_PIT_STRICT=false 且 df 缺 disclosure_date 列，记 warning 并返回原 df。

    参数:
        df: 财务数据 DataFrame
        asof: 截止日期
        caller: 调用方标识（如 "financial_factors.compute"），用于审计日志

    返回:
        PIT 过滤后的 DataFrame
    """
    caller_tag = f"[{caller}] " if caller else ""
    if df is None or df.empty:
        logger.debug(f"{caller_tag}PIT 守卫：df 为空，跳过过滤")
        return df

    if "disclosure_date" not in df.columns:
        msg = (
            f"{caller_tag}PIT 守卫违规：财务数据缺 `disclosure_date` 列。"
            f"factor-engine / backtest-engine 使用财务数据前必须经过 PIT 过滤。"
        )
        if _is_strict_mode():
            raise ValueError(msg)
        logger.warning(msg + "（QUANT_PIT_STRICT=false，降级为 warning）")
        return df

    filtered = pit_filter(df, asof)
    logger.info(
        f"{caller_tag}PIT 守卫：过滤前 {len(df)} 行 → 过滤后 {len(filtered)} 行"
        f"（asof={asof}）"
    )
    return filtered
