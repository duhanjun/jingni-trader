"""
全局上下文对象定义
在子 Skill 之间传递任务状态和产物路径
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import date
import json


@dataclass
class Context:
    """量化任务上下文，贯穿全流程"""

    # 任务标识
    task_id: str = ""
    session_id: str = ""

    # 用户意图
    user_intent: str = ""                      # 原始自然语言
    current_stage: str = "IDLE"                # 当前阶段: IDLE/DATA/FACTOR/MODEL/BACKTEST/PORTFOLIO/EXECUTION/REPORT
    target_stages: List[str] = field(default_factory=list)  # 目标阶段列表

    # 股票与时间
    stock_pool: List[str] = field(default_factory=list)     # 股票代码列表
    benchmark: str = "000300.SH"                             # 基准指数
    start_date: str = ""                                     # YYYY-MM-DD
    end_date: str = ""                                       # YYYY-MM-DD

    # 策略参数
    strategy_name: str = ""
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # 产物路径（各阶段填充）
    artifacts: Dict[str, str] = field(default_factory=dict)
    # 示例: {"data": "/path/to/cleaned.parquet", "factor": "/path/to/factor.parquet", ...}

    # 元信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def update_artifact(self, stage: str, path: str):
        """记录阶段产物路径"""
        self.artifacts[stage] = path

    def get_artifact(self, stage: str) -> Optional[str]:
        """获取阶段产物路径"""
        return self.artifacts.get(stage)

    def add_error(self, error: str):
        """记录错误"""
        self.errors.append(error)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> "Context":
        return cls.from_dict(json.loads(json_str))
