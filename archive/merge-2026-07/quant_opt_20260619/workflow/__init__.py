"""
YAML 驱动的 Workflow 配置

借鉴自 Microsoft Qlib 的 qrun 设计：
  https://qlib.readthedocs.io/en/v0.6.0/component/workflow.html

Qlib 的核心范式：
  - 一个 YAML 文件定义完整实验 (data, model, dataset, record, ...)
  - 使用 &anchor 引用避免重复
  - qrun <yaml> 一键运行

jingni-trader 现状问题：
  - engine.py 的 parse_intent() 用 keyword 匹配（如 "因子" -> FACTOR stage）
  - 用户配置散落在 Context、strategy_params、config.py
  - 实验不易复现：没有"实验定义文件"概念

本模块提供：
  - WorkflowConfig: 从 YAML 加载实验定义
  - render_jingni_intent(): 把 YAML 编译为 jingni-trader 现有的 user_intent 字符串
  - validate(): 配置合法性校验
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class StageConfig:
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    """
    jingni-trader 工作流配置 (YAML schema)

    示例：
        experiment_name: csi300_momentum_v1
        market: csi300
        benchmark: SH000300
        start_date: 2021-01-01
        end_date: 2024-12-31
        stages:
          - data: {provider: tushare}
          - factor: {method: ic_weighted, neutralize: industry}
          - model: {type: lightgbm, optuna_trials: 30}
          - backtest: {backend: native, init_capital: 1000000}
          - report: {format: html}
    """
    experiment_name: str
    market: str = "all"
    benchmark: str = "SH000300"
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"
    stages: List[StageConfig] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "WorkflowConfig":
        if not HAS_YAML:
            raise ImportError("需要 PyYAML: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict) -> "WorkflowConfig":
        stages = []
        for s in d.get("stages", []):
            if isinstance(s, dict):
                for name, params in s.items():
                    stages.append(StageConfig(
                        name=name.upper(),
                        enabled=params.get("enabled", True) if isinstance(params, dict) else True,
                        params=params if isinstance(params, dict) else {},
                    ))
        return cls(
            experiment_name=d.get("experiment_name", "unnamed"),
            market=d.get("market", "all"),
            benchmark=d.get("benchmark", "SH000300"),
            start_date=d.get("start_date", "2021-01-01"),
            end_date=d.get("end_date", "2024-12-31"),
            stages=stages,
            raw=d,
        )

    def to_jingni_intent(self) -> str:
        """
        编译为 jingni-trader 的 user_intent 字符串
        兼容现有 engine.parse_intent() 的 keyword 解析
        """
        parts = []
        stage_to_kw = {
            "DATA": "数据获取",
            "FACTOR": "因子构建",
            "MODEL": "模型训练",
            "BACKTEST": "回测",
            "PORTFOLIO": "组合优化",
            "EXECUTION": "实盘",
            "REPORT": "报告",
        }
        for st in self.stages:
            kw = stage_to_kw.get(st.name, st.name.lower())
            parts.append(kw)
        intent = " → ".join(parts)

        market_phrase = {
            "csi300": "沪深300",
            "csi500": "中证500",
            "all": "全A",
        }.get(self.market, self.market)
        intent += f"（{market_phrase}，{self.start_date} 至 {self.end_date}）"
        return intent

    def validate(self) -> List[str]:
        """返回错误信息列表，空列表表示通过"""
        errs = []
        if not self.experiment_name or not re.match(r"^[A-Za-z0-9_\-]+$", self.experiment_name):
            errs.append("experiment_name 必须为字母数字_-")
        valid_stages = {"DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"}
        for st in self.stages:
            if st.name not in valid_stages:
                errs.append(f"未知 stage: {st.name}")
        return errs
