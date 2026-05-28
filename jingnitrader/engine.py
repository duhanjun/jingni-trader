"""
jingnitrader - A股量化交易全流程主调度器

标准入口点，提供 run() 函数供外部调用
"""
import os
import sys
import json
import logging
import importlib
from typing import Dict, Optional, Any
from datetime import datetime

from scripts.config import (
    WORK_DIR, DATA_DIR, FACTOR_DIR, MODEL_DIR,
    BACKTEST_DIR, PORTFOLIO_DIR, REPORT_DIR, LOG_DIR
)
from scripts.context import Context


os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"master_{datetime.now():%Y%m%d}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("jingnitrader")


STAGES = ["IDLE", "DATA", "FACTOR", "MODEL", "BACKTEST",
          "PORTFOLIO", "EXECUTION", "REPORT"]

STAGE_ORDER = {
    "DATA": 1,
    "FACTOR": 2,
    "MODEL": 3,
    "BACKTEST": 4,
    "PORTFOLIO": 5,
    "EXECUTION": 6,
    "REPORT": 7,
}

SKILL_MODULES = {
    "DATA": "skills.data-engine.engine",
    "FACTOR": "skills.factor-engine.engine",
    "MODEL": "skills.strategy-model-engine.engine",
    "BACKTEST": "skills.backtest-engine.engine",
    "PORTFOLIO": "skills.portfolio-risk-engine.engine",
    "EXECUTION": "skills.execution-monitor-engine.engine",
    "REPORT": "skills.reports-engine.engine",
}

EXPECTED_ARTIFACTS = {
    "DATA": "cleaned_data.parquet",
    "FACTOR": "factor_data.parquet",
    "MODEL": "model.pkl",
    "BACKTEST": "backtest_result.json",
    "PORTFOLIO": "portfolio_weights.json",
    "EXECUTION": "trade_log.json",
    "REPORT": "report.html",
}


class MasterEngine:
    """主调度引擎"""

    def __init__(self):
        self.ctx: Optional[Context] = None
        self._loaded_skills: Dict[str, Any] = {}

    def parse_intent(self, user_input: str) -> Context:
        """解析用户自然语言，提取任务参数，生成 Context"""
        ctx = Context(
            task_id=datetime.now().strftime("%Y%m%d%H%M%S"),
            user_intent=user_input,
            current_stage="IDLE"
        )

        input_lower = user_input.lower()

        target_stages = []
        if any(kw in input_lower for kw in ["数据", "获取", "下载", "data"]):
            target_stages.append("DATA")
        if any(kw in input_lower for kw in ["因子", "factor", "alpha", "ic"]):
            target_stages.append("FACTOR")
        if any(kw in input_lower for kw in ["模型", "训练", "model", "train", "lightgbm", "机器学习"]):
            target_stages.append("MODEL")
        if any(kw in input_lower for kw in ["回测", "backtest", "模拟"]):
            target_stages.append("BACKTEST")
        if any(kw in input_lower for kw in ["组合", "优化", "portfolio", "风控", "仓位"]):
            target_stages.append("PORTFOLIO")
        if any(kw in input_lower for kw in ["实盘", "交易", "下单", "execution", "执行"]):
            target_stages.append("EXECUTION")
        if any(kw in input_lower for kw in ["报告", "report", "可视化", "绩效", "归因"]):
            target_stages.append("REPORT")

        if not target_stages:
            target_stages = ["DATA", "FACTOR", "MODEL", "BACKTEST", "REPORT"]

        if any(s in target_stages for s in ["FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "REPORT"]) and "DATA" not in target_stages:
            target_stages.insert(0, "DATA")

        target_stages = sorted(target_stages, key=lambda s: STAGE_ORDER.get(s, 99))
        ctx.target_stages = target_stages

        if "沪深300" in user_input:
            ctx.stock_pool = ["000300.SH"]
        elif "中证500" in user_input:
            ctx.stock_pool = ["000905.SH"]
        elif "全A" in user_input or "全市场" in user_input:
            ctx.stock_pool = []

        if "近3年" in user_input or "最近3年" in user_input:
            ctx.start_date = "2021-01-01"
            ctx.end_date = "2024-12-31"
        elif "近5年" in user_input or "最近5年" in user_input:
            ctx.start_date = "2019-01-01"
            ctx.end_date = "2024-12-31"

        logger.info(f"意图解析完成: 目标阶段={target_stages}, 股票池={ctx.stock_pool or '全市场'}")
        self.ctx = ctx
        return ctx

    def execute_stage(self, stage: str) -> bool:
        """执行单个阶段，调用对应子 Skill"""
        logger.info(f"=== 开始执行阶段: {stage} ===")

        artifact_file = EXPECTED_ARTIFACTS.get(stage)
        stage_dir = {
            "DATA": DATA_DIR,
            "FACTOR": FACTOR_DIR,
            "MODEL": MODEL_DIR,
            "BACKTEST": BACKTEST_DIR,
            "PORTFOLIO": PORTFOLIO_DIR,
            "EXECUTION": WORK_DIR,
            "REPORT": REPORT_DIR,
        }.get(stage, WORK_DIR)

        artifact_path = os.path.join(stage_dir, artifact_file) if artifact_file else None

        if artifact_path and os.path.exists(artifact_path):
            logger.info(f"阶段 {stage} 产物已存在，跳过: {artifact_path}")
            self.ctx.update_artifact(stage, artifact_path)
            return True

        module_name = SKILL_MODULES.get(stage)
        if not module_name:
            error_msg = f"未找到阶段 {stage} 对应的 Skill 模块"
            logger.error(error_msg)
            self.ctx.add_error(error_msg)
            return False

        try:
            skill_module = importlib.import_module(module_name)
        except ImportError as e:
            error_msg = f"加载子 Skill {module_name} 失败: {e}"
            logger.error(error_msg)
            self.ctx.add_error(error_msg)
            return False

        try:
            result = skill_module.run(self.ctx)
            if result.get("success"):
                artifact = result.get("artifact_path", "")
                self.ctx.update_artifact(stage, artifact)
                self.ctx.metadata[stage] = result.get("metadata", {})
                logger.info(f"阶段 {stage} 执行成功, 产物: {artifact}")
                return True
            else:
                error_msg = result.get("error", "未知错误")
                logger.error(f"阶段 {stage} 执行失败: {error_msg}")
                self.ctx.add_error(f"{stage}: {error_msg}")
                return False
        except Exception as e:
            error_msg = f"阶段 {stage} 执行异常: {str(e)}"
            logger.exception(error_msg)
            self.ctx.add_error(error_msg)
            return False

    def run_pipeline(self, user_input: str = None, ctx: Context = None) -> dict:
        """执行全流程管道"""
        if ctx:
            self.ctx = ctx
        elif user_input:
            self.ctx = self.parse_intent(user_input)
        else:
            return {"success": False, "error": "需要提供 user_input 或 ctx"}

        results = {"success": True, "completed_stages": [], "failed_stages": [], "summary": ""}

        for stage in self.ctx.target_stages:
            success = self.execute_stage(stage)
            if success:
                results["completed_stages"].append(stage)
            else:
                results["failed_stages"].append(stage)
                if stage in ["DATA", "BACKTEST"]:
                    results["success"] = False
                    logger.error(f"关键阶段 {stage} 失败，停止管道")
                    break

        results["summary"] = self._generate_summary()
        results["context"] = self.ctx.to_dict()

        return results

    def _generate_summary(self) -> str:
        """生成可读的管道执行摘要"""
        if not self.ctx:
            return "无执行记录"

        lines = [
            f"任务ID: {self.ctx.task_id}",
            f"用户意图: {self.ctx.user_intent}",
            f"目标阶段: {' → '.join(self.ctx.target_stages)}",
            f"产物列表:"
        ]
        for stage, path in self.ctx.artifacts.items():
            lines.append(f"  [{stage}] {path}")

        if self.ctx.errors:
            lines.append(f"错误: {'; '.join(self.ctx.errors)}")

        return "\n".join(lines)


def run(ctx: Context = None, user_input: str = None) -> dict:
    """
    Skill 标准入口函数

    所有 Skill 都应该实现此接口

    参数:
        ctx: 上下文对象（可选）
        user_input: 用户自然语言（可选）

    返回:
        {
            "success": bool,
            "completed_stages": [...],
            "failed_stages": [...],
            "summary": str,
            "context": dict
        }
    """
    engine = MasterEngine()
    return engine.run_pipeline(user_input=user_input, ctx=ctx)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A股量化交易主调度器")
    parser.add_argument("-i", "--input", type=str, required=True, help="用户需求描述")
    parser.add_argument("-c", "--context", type=str, default=None, help="已有的Context JSON文件路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出结果JSON路径")

    args = parser.parse_args()

    ctx = None
    if args.context:
        with open(args.context, "r", encoding="utf-8") as f:
            ctx = Context.from_json(f.read())

    result = run(ctx=ctx, user_input=args.input)

    output_json = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"结果已保存至: {args.output}")
    else:
        print(output_json)
