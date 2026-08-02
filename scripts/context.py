"""全局上下文对象定义（P1-4 Pydantic V2 改造）

在子 Skill 之间传递任务状态和产物路径。

P1-4.4 改造要点：
- Context 从 dataclass 升级为 Pydantic V2 BaseModel
- 保留 dataclass 兼容接口：update_artifact / get_artifact / add_error /
  to_dict / to_json / from_dict / from_json 同名方法
- extra="ignore"（向后兼容，允许旧代码传递多余字段不报错）
- 现有调用代码（ctx.update_artifact、ctx.get_artifact）零改动
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Context(BaseModel):
    """量化任务上下文，贯穿全流程（Pydantic V2）"""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # 任务标识
    task_id: str = ""
    session_id: str = ""

    # 用户意图
    user_intent: str = ""
    current_stage: str = "IDLE"
    target_stages: List[str] = Field(default_factory=list)

    # 股票与时间
    stock_pool: List[str] = Field(default_factory=list)
    benchmark: str = "000300.SH"
    start_date: str = ""
    end_date: str = ""

    # 策略参数
    strategy_name: str = ""
    strategy_params: Dict[str, Any] = Field(default_factory=dict)

    # 产物路径（各阶段填充）
    artifacts: Dict[str, str] = Field(default_factory=dict)

    # 系统内置工具传入的外部数据
    external_data: Dict[str, Any] = Field(default_factory=dict)

    # data-engine 专用参数：用户通过对话指定的数据源优先级链
    data_sources: Optional[List[str]] = None

    # 运行归档目录
    run_dir: str = ""
    step_dirs: Dict[str, str] = Field(default_factory=dict)

    # 元信息
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # 兼容接口（与原 dataclass 方法签名一致）
    # ------------------------------------------------------------------

    def update_artifact(self, stage: str, path: str) -> None:
        """记录阶段产物路径"""
        self.artifacts[stage] = path

    def get_artifact(self, stage: str) -> Optional[str]:
        """获取阶段产物路径"""
        return self.artifacts.get(stage)

    def add_error(self, error: str) -> None:
        """记录错误"""
        self.errors.append(error)

    def to_dict(self) -> dict:
        """转为字典（Pydantic model_dump，等价于原 asdict）"""
        return self.model_dump()

    def to_json(self) -> str:
        """转为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        """从字典构建 Context（过滤未知字段，向后兼容）"""
        known = set(cls.model_fields.keys())
        filtered = {k: v for k, v in data.items() if k in known}
        return cls.model_validate(filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "Context":
        """从 JSON 字符串构建 Context"""
        return cls.from_dict(json.loads(json_str))
