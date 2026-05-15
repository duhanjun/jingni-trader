"""
主调度引擎
负责阶段状态机、子 Skill 调用、产物校验
"""
import os
import sys
import json
import logging
import importlib
from typing import Dict, Optional, Any
from datetime import datetime

from .config import (
    WORK_DIR, DATA_DIR, FACTOR_DIR, MODEL_DIR,
    BACKTEST_DIR, PORTFOLIO_DIR, REPORT_DIR, LOG_DIR
)
from .context import Context


# ── 日志配置 ──────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"master_{datetime.now():%Y%m%d}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("quant-master")


# ── 阶段定义 ──────────────────────────────
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

# 子 Skill 模块名映射
SKILL_MODULES = {
    "DATA": "a_share_data_engine",
    "FACTOR": "a_share_factor_engine",
    "MODEL": "strategy_model_engine",
    "BACKTEST": "backtest_engine",
    "PORTFOLIO": "portfolio_risk_engine",
    "EXECUTION": "execution_monitor_engine",
    "REPORT": "reports_engine",
}

# 每个阶段的产物文件（用于校验是否存在）
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

    # ── 意图解析 ──────────────────────────
    def parse_intent(self, user_input: str) -> Context:
        """
        解析用户自然语言，提取任务参数，生成 Context

        参数:
            user_input: 用户原始输入

        返回:
            初始化的 Context 对象
        """
        ctx = Context(
            task_id=datetime.now().strftime("%Y%m%d%H%M%S"),
            user_intent=user_input,
            current_stage="IDLE"
        )

        # ── 基于关键词的阶段推断 ─────────────
        input_lower = user_input.lower()

        # 判断需要的阶段
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

        # 如果没有任何关键词，默认走完整流程
        if not target_stages:
            target_stages = ["DATA", "FACTOR", "MODEL", "BACKTEST", "REPORT"]

        # 按阶段顺序排序
        target_stages = sorted(target_stages, key=lambda s: STAGE_ORDER.get(s, 99))
        ctx.target_stages = target_stages

        # ── 简单参数提取（可增强为 LLM 解析）──
        # 股票池
        if "沪深300" in user_input:
            ctx.stock_pool = ["000300.SH"]
        elif "中证500" in user_input:
            ctx.stock_pool = ["000905.SH"]
        elif "全A" in user_input or "全市场" in user_input:
            ctx.stock_pool = []  # 表示全市场

        # 时间范围（简化版，实际可用 dateparser 库）
        if "近3年" in user_input or "最近3年" in user_input:
            ctx.start_date = "2021-01-01"
            ctx.end_date = "2024-12-31"
        elif "近5年" in user_input or "最近5年" in user_input:
            ctx.start_date = "2019-01-01"
            ctx.end_date = "2024-12-31"

        logger.info(f"意图解析完成: 目标阶段={target_stages}, 股票池={ctx.stock_pool or '全市场'}")
        self.ctx = ctx
        return ctx

    # ── 阶段执行 ──────────────────────────
    def execute_stage(self, stage: str) -> bool:
        """
        执行单个阶段，调用对应子 Skill

        参数:
            stage: 阶段名 (DATA/FACTOR/MODEL/BACKTEST/PORTFOLIO/EXECUTION/REPORT)

        返回:
            是否执行成功
        """
        logger.info(f"=== 开始执行阶段: {stage} ===")

        # 检查产物是否已存在（支持断点续跑）
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

        # 动态加载子 Skill
        module_name = SKILL_MODULES.get(stage)
        if not module_name:
            error_msg = f"未找到阶段 {stage} 对应的 Skill 模块"
            logger.error(error_msg)
            self.ctx.add_error(error_msg)
            return False

        try:
            skill_module = importlib.import_module(f"{module_name}.engine")
        except ImportError as e:
            error_msg = f"加载子 Skill {module_name} 失败: {e}"
            logger.error(error_msg)
            self.ctx.add_error(error_msg)
            return False

        # 调用子 Skill 的 run 函数
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

    # ── 全流程执行 ──────────────────────────
    def run_pipeline(self, user_input: str = None, ctx: Context = None) -> dict:
        """
        执行全流程管道

        参数:
            user_input: 用户输入（如果不提供 ctx）
            ctx: 已有的上下文对象（断点续跑时使用）

        返回:
            执行结果摘要
        """
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
                # 如果关键阶段失败，停止后续阶段
                if stage in ["DATA", "BACKTEST"]:
                    results["success"] = False
                    logger.error(f"关键阶段 {stage} 失败，停止管道")
                    break

        # 生成摘要
        results["summary"] = self._generate_summary()
        results["context"] = self.ctx.to_dict()

        return results

    # ── 生成摘要 ──────────────────────────
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


# ── 对外统一入口 ──────────────────────────
def run(ctx: Context = None, user_input: str = None) -> dict:
    """
    Skill 统一入口函数
    所有子 Skill 都需要实现此接口

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


# ── CLI 入口 ──────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A股量化交易主调度器")
    parser.add_argument("--input", "-i", type=str, required=True, help="用户需求描述")
    parser.add_argument("--context", "-c", type=str, default=None, help="已有的Context JSON文件路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出结果JSON路径")

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
