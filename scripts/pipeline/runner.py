"""流水线执行器 - 借鉴 Qlib qrun"""
import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("pipeline")

STAGE_SKILL_MAP = {
    "data": "data-engine",
    "factor": "factor-engine",
    "model": "strategy-model-engine",
    "backtest": "backtest-engine",
    "portfolio": "portfolio-risk-engine",
    "execution": "execution-monitor-engine",
    "report": "reports-engine",
}

STAGE_ORDER = ["data", "factor", "model", "backtest", "portfolio", "execution", "report"]


@dataclass
class PipelineConfig:
    name: str = ""
    description: str = ""
    init: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    global_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            init=data.get("init", {}),
            stages=data.get("stages", {}),
            global_params=data.get("global", {}),
        )


class PipelineRunner:
    """配置化流水线执行器

    Usage:
        runner = PipelineRunner()
        results = runner.run("config/pipeline_momentum_factor.yaml")
    """

    def __init__(self):
        self.config: Optional[PipelineConfig] = None
        self.artifacts: Dict[str, Any] = {}
        self.stage_status: Dict[str, str] = {}

    def run(self, config_path: str, stages: Optional[List[str]] = None) -> Dict[str, Any]:
        self.config = PipelineConfig.from_yaml(config_path)
        logger.info(f"流水线加载: {self.config.name}")
        logger.info(f"  {self.config.description}")

        if stages is None:
            active = [s for s in STAGE_ORDER if s in self.config.stages]
        else:
            active = [s for s in STAGE_ORDER if s in stages and s in self.config.stages]

        logger.info(f"阶段: {active}")

        results = {}
        for stage in active:
            sc = self.config.stages[stage]
            if not sc.get("enabled", True):
                self.stage_status[stage] = "skipped"
                continue

            # 依赖检查
            for dep in sc.get("depends_on", []):
                if self.stage_status.get(dep) == "failed":
                    self.stage_status[stage] = "failed_dependency"
                    results[stage] = {"error": f"依赖 {dep} 失败"}
                    continue

            try:
                params = {**self.config.global_params, **sc.get("params", {})}
                result = self._execute_stage(stage, sc, params)
                self.artifacts[stage] = result
                self.stage_status[stage] = "completed"
                results[stage] = result
                logger.info(f"  [OK] {stage}")

                if sc.get("break_on_success", False):
                    logger.info("中断标记，停止后续阶段")
                    break

            except Exception as e:
                self.stage_status[stage] = "failed"
                results[stage] = {"error": str(e)}
                logger.error(f"  [FAIL] {stage}: {e}")
                if not sc.get("continue_on_error", False):
                    break

        self._print_summary()
        return results

    def _execute_stage(self, stage: str, config: dict, params: dict) -> dict:
        skill = STAGE_SKILL_MAP.get(stage, stage)
        info = {"stage": stage, "skill": skill, "params": params, "status": "ok"}

        if stage == "data":
            info["detail"] = f"数据源: {self.config.init.get('data_source')}, 范围: {params.get('start_date')}~{params.get('end_date')}"
        elif stage == "factor":
            info["detail"] = f"因子类型: {config.get('factor_type')}, 表达式: {list(config.get('expressions', {}).keys())}, 扩展因子: {config.get('extended_factors')}"
        elif stage == "backtest":
            info["detail"] = f"模式: {'T+1增强' if config.get('enhanced') else 'T+1'}, 资金: {params.get('init_capital')}"
        elif stage == "report":
            info["detail"] = f"输出: {config.get('output')}"
        elif stage == "model":
            info["detail"] = f"模型: {params.get('model_type')}, 特征: {params.get('features')}"
        return info

    def _print_summary(self):
        print("\n" + "=" * 50)
        print(f"流水线摘要: {self.config.name}")
        print("=" * 50)
        icons = {"completed": "[OK]", "failed": "[FAIL]", "skipped": "[SKIP]", "failed_dependency": "[DEP]"}
        for s in STAGE_ORDER:
            st = self.stage_status.get(s)
            if st:
                print(f"  {icons.get(st, '---')} {s}")
        print("=" * 50)

    def validate_config(self, config_path: str) -> List[str]:
        warnings = []
        try:
            cfg = PipelineConfig.from_yaml(config_path)
        except Exception as e:
            return [f"YAML 解析失败: {e}"]
        if not cfg.name:
            warnings.append("缺少 name")
        if not cfg.stages:
            warnings.append("未定义 stages")
        for s in cfg.stages:
            if s not in STAGE_ORDER:
                warnings.append(f"未知阶段: {s}")
        return warnings


def main():
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("用法: python scripts/pipeline/runner.py config.yaml [--stages a,b,c]")
        sys.exit(1)

    config_path = sys.argv[1]
    stage_filter = None
    if "--stages" in sys.argv:
        idx = sys.argv.index("--stages")
        stage_filter = sys.argv[idx + 1].split(",")

    runner = PipelineRunner()
    for w in runner.validate_config(config_path):
        logger.warning(f"配置警告: {w}")

    results = runner.run(config_path, stage_filter)
    print(f"\n完成: {len(results)} 个阶段执行")


if __name__ == "__main__":
    main()