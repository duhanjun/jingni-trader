"""
jingnitrader - A股量化交易全流程主调度器

标准入口点，提供 run() 函数供外部调用
"""
import os
import sys
import json
import logging
import importlib
import importlib.util as _ilu
from typing import Dict, Optional, Any
from datetime import datetime

from scripts.config import (
    WORK_DIR, DATA_DIR, FACTOR_DIR, MODEL_DIR,
    BACKTEST_DIR, PORTFOLIO_DIR, REPORT_DIR, LOG_DIR,
    ARCHIVE_DIR
)
from scripts.context import Context
from scripts.archive import RunArchiver
from scripts.schemas import STAGE_SCHEMA_MAP, safe_validate_payload
from scripts.fsm import DailyFSM, IncidentFSM, STATE_MANUAL_ATTENTION, STATE_DEGRADED


os.makedirs(LOG_DIR, exist_ok=True)
# GAP-1 修复：Windows zh_CN(GBK) 控制台默认编码下，含中文/非 ASCII 字形的
# 日志消息会抛 UnicodeEncodeError 并中断主流程。统一将 stderr/stdout 设为 utf-8，
# FileHandler 显式指定 utf-8，保证跨平台不乱码、不崩。
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_handlers = [
    logging.FileHandler(
        os.path.join(LOG_DIR, f"master_{datetime.now():%Y%m%d}.log"),
        encoding="utf-8", errors="replace"
    ),
    # StreamHandler 不接受 encoding 参数，依赖已 reconfigure 的 stderr
    logging.StreamHandler(sys.stderr),
]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=_handlers,
)
logger = logging.getLogger("jingnitrader")


# 每个子技能自带的 scripts/ 包路径（相对项目根目录）。
# 主流程在加载子技能前，需把对应子技能的 scripts 包注册为 sys.modules['scripts']，
# 否则子技能内部 `from scripts.config import ...` 会解析到主 scripts 包而 ImportError。
_SUBSKILL_SCRIPTS = {
    "DATA": "skills/data-engine/scripts",
    "FACTOR": "skills/factor-engine/scripts",
    "MODEL": "skills/strategy-model-engine/scripts",
    "BACKTEST": "skills/backtest-engine/scripts",
    "PORTFOLIO": "skills/portfolio-risk-engine/scripts",
    "EXECUTION": "skills/execution-monitor-engine/scripts",
    "REPORT": "skills/reports-engine/scripts",
}


def _register_subskill_scripts(stage: str):
    """把子技能自带的 scripts 包注册为 sys.modules['scripts']。

    每个子技能都有自己的 scripts/ 包（含各自独立的 config.py），它们与主
    scripts 包共用 sys.modules['scripts'] 槽位，同一时刻只能有一个生效。
    加载子技能前必须先切换到该子技能的 scripts 包，否则会 ImportError。
    """
    rel = _SUBSKILL_SCRIPTS.get(stage)
    if not rel:
        return
    init_py = os.path.join(os.path.dirname(__file__), rel, "__init__.py")
    if not os.path.exists(init_py):
        return
    spec = _ilu.spec_from_file_location(
        "scripts", init_py,
        submodule_search_locations=[os.path.dirname(init_py)],
    )
    pkg = _ilu.module_from_spec(spec)
    sys.modules["scripts"] = pkg
    spec.loader.exec_module(pkg)


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


# 策略构建关键词（触发完整管线：DATA → FACTOR → MODEL → BACKTEST → ...）
# 只有用户明确需要"构建可回测/可交易策略"时才命中，因子计算/分析本身不触发
STRATEGY_KEYWORDS = {
    "回测", "策略", "模型", "组合", "实盘", "选股",
    "backtest", "夏普", "回撤", "仓位", "风控", "模拟", "下单", "执行",
}

# 绩效复盘关键词（触发复盘路径：读取 EXECUTION 产物 → REPORT）
# 优先级高于策略关键词，用于回答"实盘盈亏来自哪里？"
ATTRIBUTION_KEYWORDS = {
    "绩效归因", "归因分析", "复盘", "实盘报告", "盈亏分析",
    "执行报告", "交易复盘", "绩效复盘", "attribution",
}


# ── 数据源优先级意图解析（方案 D：用户对话切换数据源）──────────────
# 用户可通过自然语言指定数据源优先级，例如：
#   "用 wind 取数据"              → ctx.data_sources = ["wind", "tushare", "baostock", "akshare", "websearch"]
#   "优先用 ifind，失败用 tushare" → ctx.data_sources = ["ifind", "tushare", "baostock", "akshare", "websearch"]
#   "用 baostock 作为首选源"       → ctx.data_sources = ["baostock", "tushare", "akshare", "websearch"]
# 未匹配到数据源优先级意图时，ctx.data_sources 保持 None，由 data-engine 走环境变量/默认值。
#
# 触发关键词（必须同时命中"指定动作"和"数据源名称"才算明确意图，避免误触发）：
DATA_SOURCE_TRIGGER_VERBS = {"用", "使用", "优先", "首选", "改用", "切换", "换", "from", "use", "using"}
DATA_SOURCE_NAMES = {
    "tushare":   "tushare",
    "baostock":  "baostock",
    "akshare":   "akshare",
    "websearch": "websearch",
    "xtquant":   "xtquant",
    "gm":        "gm",
    "tdxquant":  "tdxquant",
    "wind":      "wind",
    "ifind":     "ifind",
    "万得":       "wind",
    "wind终端":   "wind",
    "同花顺":      "ifind",
    "ifind终端":  "ifind",
    "掘金":       "gm",
    "通达信":      "tdxquant",
    "迅投":       "xtquant",
}

# 默认免费降级链（用户只指定首选源时，自动追加在后面作为兜底）
# 仅含真正免费、无需 token/账号的源；tushare/wind/ifind 等 opt-in 源不在此列
_DEFAULT_FALLBACK_CHAIN = ["baostock", "akshare", "websearch"]


class MasterEngine:
    """主调度引擎"""

    def __init__(self):
        self.ctx: Optional[Context] = None
        self._loaded_skills: Dict[str, Any] = {}
        self.archiver: Optional[RunArchiver] = None
        self._force_refresh = os.environ.get("QUANT_FORCE_REFRESH", "").lower() in ("1", "true", "yes")

        # P0-4 Frozen Core 路径策略保护：注册 atexit 退出兜底钩子
        # 在 run_pipeline 开始时调用 pre_snapshot，进程退出时 post_diff 校验 frozen core
        self._git_tracker = None
        try:
            import atexit
            from scripts.path_policy_loader import get_git_tracker, post_run_guard
            self._git_tracker = get_git_tracker()
            atexit.register(post_run_guard, self._git_tracker, "master-engine")
        except Exception as e:
            logger.debug(f"P0-4 atexit 钩子注册跳过: {e}")

        # 版本检查：每次实例化时检查 GitHub 是否有新版本，落后则输出提示
        # 本调用只检查、不修改任何文件（详见 scripts/skill_sync.py）
        # 失败/网络异常/24h 内已检查 均静默跳过，不阻断主流程（用户无感）
        try:
            from scripts.skill_sync import sync_all, ensure_skill
            project_root = os.path.dirname(os.path.abspath(__file__))
            sync_all(project_root)

            # 自动部署 jingni-datafeed：若 skills/jingni-datafeed/ 不存在则从 GitHub 克隆
            # 这是 greenfield 操作（从零创建），无用户数据风险
            # 已存在时退化为正常的版本检查（只检测不修改）
            ensure_skill(project_root, "jingni-datafeed")
        except Exception as e:
            logger.debug(f"skill 版本检查跳过: {e}")

    def _is_attribution_intent(self, user_intent: str) -> bool:
        """判断用户是否触发绩效复盘意图。

        优先级高于策略构建意图，命中时路由到复盘路径：
        DATA → FACTOR → EXECUTION → REPORT（跳过 MODEL/BACKTEST/PORTFOLIO）。
        """
        if not user_intent:
            return False
        return any(kw in user_intent for kw in ATTRIBUTION_KEYWORDS)

    def _is_strategy_required(self, user_intent: str) -> bool:
        """判断用户是否明确需要构建可回测/可交易策略。

        设计原则：
        - 因子计算/IC 分析是两条路径共用的前置步骤，不构成"策略构建"意图
        - 仅当用户明确表达"回测/策略/模型/组合/实盘/选股/风控/下单"等动作时才为 True
        - 默认 False（走分析路径 DATA → FACTOR → REPORT），与产品定位一致
        """
        if not user_intent:
            return False
        return any(kw in user_intent for kw in STRATEGY_KEYWORDS)

    def _detect_report_template(self, user_input: str) -> str:
        """从用户输入检测报告模板类型"""
        text = user_input.lower()
        tech_keywords = [
            "技术面", "技术分析", "k线", "k 线", "趋势", "支撑", "阻力",
            "形态", "macd", "rsi", "kdj", "boll", "均线", "量价", "资金流",
            "龙虎榜", "涨跌停", "北向",
        ]
        fund_keywords = [
            "基本面", "财务", "估值", "roe", "毛利率", "净利率", "营收",
            "利润", "pe", "pb", "ps", "股东", "分红", "股息", "现金流",
            "资产负债", "成长性", "盈利能力",
        ]
        tech_hits = sum(1 for kw in tech_keywords if kw in text)
        fund_hits = sum(1 for kw in fund_keywords if kw in text)

        if tech_hits > 0 and fund_hits == 0:
            return "technical"
        if fund_hits > 0 and tech_hits == 0:
            return "fundamental"
        if tech_hits > 0 and fund_hits > 0:
            if tech_hits - fund_hits >= 2:
                return "technical"
            if fund_hits - tech_hits >= 2:
                return "fundamental"
            return "both"
        return "both"

    def _parse_data_sources_intent(self, user_input: str) -> Optional[list]:
        """解析用户对数据源优先级的明确指定。

        返回:
            - list[str]: 用户明确指定了数据源优先级链（已去重、已校验）
            - None:     未识别到数据源优先级意图，ctx.data_sources 保持 None

        判定规则（避免误触发）：
            1. 用户输入必须同时包含"动作动词"和"数据源名称"
            2. 动词如：用/使用/优先/首选/改用/切换/换/use/using/from
            3. 数据源名称如：tushare/baostock/akshare/wind/ifind/万得/同花顺/掘金/通达信/迅投

        示例：
            "用 wind 取数据"        → ["wind", "tushare", "baostock", "akshare", "websearch"]
            "优先用 ifind，失败用 tushare" → ["ifind", "tushare", "baostock", "akshare", "websearch"]
            "用 baostock 作为首选源" → ["baostock", "tushare", "akshare", "websearch"]
            "用 baostock 和 akshare" → ["baostock", "akshare", "tushare", "websearch"]
            "今天天气真好"          → None
            "用 momentum 因子做回测" → None（"用"是动词但未跟数据源名）
        """
        if not user_input:
            return None

        input_lower = user_input.lower()

        # 1) 必须命中至少一个动作动词
        has_verb = any(v in user_input or v in input_lower for v in DATA_SOURCE_TRIGGER_VERBS)
        if not has_verb:
            return None

        # 2) 按名称在用户输入中出现的位置排序，提取命中的数据源
        matched = []  # [(pos, source_name), ...]
        for keyword, source_name in DATA_SOURCE_NAMES.items():
            # 中文用原文匹配，英文用小写匹配
            if keyword.isascii():
                haystack = input_lower
            else:
                haystack = user_input
            idx = haystack.find(keyword)
            if idx >= 0:
                matched.append((idx, source_name))

        if not matched:
            return None

        # 按位置排序，去重
        matched.sort(key=lambda x: x[0])
        user_chain: List[str] = []
        for _, src in matched:
            if src not in user_chain:
                user_chain.append(src)

        # 3) 用户只指定了部分源 → 自动追加默认免费降级链作为兜底
        #    例如用户只说"用 wind"，自动补 tushare→baostock→akshare→websearch
        for src in _DEFAULT_FALLBACK_CHAIN:
            if src not in user_chain:
                user_chain.append(src)

        return user_chain

    def parse_intent(self, user_input: str) -> Context:
        """解析用户自然语言，提取任务参数，生成 Context

        单一工作流模型：根据是否需要构建策略选择执行深度。
        - strategy_required=True  → 完整 7 阶段管线（含 MODEL/BACKTEST/PORTFOLIO/EXECUTION）
        - strategy_required=False → DATA → FACTOR → REPORT（默认，分析路径）
        - EXECUTION 触发时自动补齐 MODEL 和 BACKTEST（实盘依赖回测验证的策略）
        """
        ctx = Context(
            task_id=datetime.now().strftime("%Y%m%d%H%M%S"),
            user_intent=user_input,
            current_stage="IDLE"
        )

        input_lower = user_input.lower()

        # 单一布尔标志驱动路由：用户是否明确需要构建可回测/可交易策略
        strategy_required = self._is_strategy_required(user_input)
        ctx.metadata["strategy_required"] = strategy_required

        # 报告模板检测：正交维度，无论是否构建策略都应工作
        ctx.metadata["report_template"] = self._detect_report_template(user_input)

        # 绩效复盘意图：最高优先级，独立路由（优先级 > 策略构建 > 个股分析）
        is_attribution = self._is_attribution_intent(user_input)
        if is_attribution:
            ctx.metadata["report_intent"] = "attribution"
            target_stages = ["DATA", "FACTOR", "EXECUTION", "REPORT"]
            logger.info(
                f"检测到绩效复盘意图，路由到复盘路径: {' → '.join(target_stages)}"
            )
        elif strategy_required:
            # 完整策略管线：DATA → FACTOR → MODEL → BACKTEST → PORTFOLIO → EXECUTION → REPORT
            # MODEL 和 BACKTEST 已在管线中，实盘依赖回测验证的策略自动满足
            target_stages = ["DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"]
            logger.info(f"检测到策略构建意图，路由到完整管线: {' → '.join(target_stages)}")
        else:
            # 默认分析路径：因子仅用于分析，不构建策略
            target_stages = ["DATA", "FACTOR", "REPORT"]
            logger.info(
                f"未检测到策略构建意图，路由到分析路径: DATA → FACTOR → REPORT "
                f"(报告模板: {ctx.metadata['report_template']})"
            )

        target_stages = sorted(target_stages, key=lambda s: STAGE_ORDER.get(s, 99))
        ctx.target_stages = target_stages

        if "沪深300" in user_input:
            ctx.stock_pool = ["000300.SH"]
        elif "中证500" in user_input:
            ctx.stock_pool = ["000905.SH"]
        elif "全A" in user_input or "全市场" in user_input:
            ctx.stock_pool = []

        # 日期范围：默认取最近5年数据；支持用户通过关键词自定义
        from datetime import date as _date
        today = _date.today()
        if "近3年" in user_input or "最近3年" in user_input:
            ctx.start_date = (today.replace(year=today.year - 3)).strftime("%Y-%m-%d")
            ctx.end_date = today.strftime("%Y-%m-%d")
        elif "近5年" in user_input or "最近5年" in user_input:
            ctx.start_date = (today.replace(year=today.year - 5)).strftime("%Y-%m-%d")
            ctx.end_date = today.strftime("%Y-%m-%d")
        elif "近1年" in user_input or "最近1年" in user_input:
            ctx.start_date = (today.replace(year=today.year - 1)).strftime("%Y-%m-%d")
            ctx.end_date = today.strftime("%Y-%m-%d")
        else:
            # 默认：从当天起最近5年
            ctx.start_date = (today.replace(year=today.year - 5)).strftime("%Y-%m-%d")
            ctx.end_date = today.strftime("%Y-%m-%d")

        # JINGNI_URL gate: 若配置了惊泥因子库凭证且用户明确要求从因子库取数，
        # 则在 metadata 中标记 factor_source=jingni，factor-engine 据此走 jingni-datafeed 路径。
        jingni_configured = bool(os.environ.get("JINGNI_URL")) and bool(os.environ.get("JINGNI_TOKEN"))
        jingni_keywords = ["jingni", "惊泥", "因子库", "factor_store", "factor-store", "已沉淀"]
        wants_jingni = any(kw in input_lower for kw in jingni_keywords)
        if jingni_configured and wants_jingni and "FACTOR" in target_stages:
            ctx.metadata["factor_source"] = "jingni"
            logger.info("检测到惊泥因子库配置且用户明确要求从因子库取数 → FACTOR 阶段将走 jingni-datafeed 路径")
        elif jingni_configured and "FACTOR" in target_stages:
            # 配置就绪但未明确要求 → 默认仍走本地计算路径，factor-engine 内部按需自行 fallback
            ctx.metadata["factor_source"] = "auto"
        else:
            ctx.metadata["factor_source"] = "local"

        # 数据源优先级意图解析（方案 D）：
        # 用户可通过对话明确指定数据源优先级（如"用 wind 取数据"）。
        # 匹配到 → 写入 ctx.data_sources，覆盖环境变量 DATA_BACKENDS。
        # 未匹配到 → ctx.data_sources 保持 None，由 data-engine 走环境变量 → 默认值。
        data_sources = self._parse_data_sources_intent(user_input)
        if data_sources:
            ctx.data_sources = data_sources
            logger.info(f"检测到数据源优先级意图 → ctx.data_sources={data_sources}（覆盖环境变量 DATA_BACKENDS）")

        logger.info(f"意图解析完成: 目标阶段={target_stages}, 股票池={ctx.stock_pool or '全市场'}, "
                    f"factor_source={ctx.metadata.get('factor_source')}, "
                    f"data_sources={ctx.data_sources or '(走默认链)'}, ")
        self.ctx = ctx
        return ctx

    def execute_stage(self, stage: str, step_num: int) -> bool:
        """执行单个阶段，调用对应子 Skill"""
        logger.info(f"=== 开始执行阶段 Step {step_num}: {stage} ===")

        artifact_file = EXPECTED_ARTIFACTS.get(stage)
        # 模板报告模式生成 technical_report.html + fundamental_report.html，
        # 缓存检查以 technical_report.html 为准（reports-engine 返回的第一份产物）
        if (stage == "REPORT"
                and getattr(self.ctx, 'metadata', {}).get("report_template") in ("both", "technical", "fundamental")
                and not self.ctx.get_artifact("BACKTEST")):
            artifact_file = "technical_report.html"
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

        if self.archiver:
            self.archiver.create_step_dir(step_num, stage)

        if artifact_path and os.path.exists(artifact_path) and not self._force_refresh:
            logger.info(f"阶段 {stage} 产物已存在，跳过: {artifact_path}")
            self.ctx.update_artifact(stage, artifact_path)
            if self.archiver:
                self.archiver.record_step_result(stage, {"success": True, "artifact_path": artifact_path, "metadata": {"source": "cache"}})
                # P1-3.5: 传入上游 inputs 血缘（ctx.artifacts 中除当前阶段外的产物）
                upstream_inputs = [
                    v for k, v in self.ctx.artifacts.items() if k != stage and v
                ]
                self.archiver.save_artifact_copy(stage, artifact_path, inputs=upstream_inputs)
                self.archiver.record_stage_end(stage, "success")
                self.archiver.write_step_summary(stage, step_num)
            return True

        module_name = SKILL_MODULES.get(stage)
        if not module_name:
            error_msg = f"未找到阶段 {stage} 对应的 Skill 模块"
            logger.error(error_msg)
            self.ctx.add_error(error_msg)
            if self.archiver:
                self.archiver.record_step_result(stage, {"success": False, "error": error_msg})
                self.archiver.record_stage_end(stage, "failed")
                self.archiver.write_step_summary(stage, step_num)
            return False

        try:
            for key in list(sys.modules.keys()):
                if key == 'scripts' or key.startswith('scripts.'):
                    del sys.modules[key]
            _register_subskill_scripts(stage)
            skill_module = importlib.import_module(module_name)
        except ImportError as e:
            error_msg = f"加载子 Skill {module_name} 失败: {e}"
            logger.error(error_msg)
            self.ctx.add_error(error_msg)
            if self.archiver:
                self.archiver.record_step_result(stage, {"success": False, "error": error_msg})
                self.archiver.record_stage_end(stage, "failed")
                self.archiver.write_step_summary(stage, step_num)
            return False

        try:
            result = skill_module.run(self.ctx)
            if self.archiver:
                self.archiver.record_step_result(stage, result)
            if result.get("success"):
                artifact = result.get("artifact_path", "")
                self.ctx.update_artifact(stage, artifact)
                self.ctx.metadata[stage] = result.get("metadata", {})
                # P1-4 阶段间 schema 校验（软校验，不阻断流程，遵循零回归）
                stage_meta = result.get("metadata", {})
                schema_cls = STAGE_SCHEMA_MAP.get(stage)
                if schema_cls is not None and stage_meta:
                    safe_validate_payload(stage_meta, schema_cls, stage=stage)
                if self.archiver:
                    # P1-3.5: 传入上游 inputs 血缘（ctx.artifacts 中除当前阶段外的产物）
                    upstream_inputs = [
                        v for k, v in self.ctx.artifacts.items() if k != stage and v
                    ]
                    self.archiver.save_artifact_copy(stage, artifact, inputs=upstream_inputs)
                    # 处理多产物场景（如非量化 both 模式生成技术面+基本面两份报告）
                    all_artifacts = result.get("metadata", {}).get("all_artifacts", [])
                    for extra in all_artifacts:
                        if extra and extra != artifact:
                            self.archiver.save_artifact_copy(stage, extra, inputs=upstream_inputs)
                    # T3-7: FACTOR 阶段额外归档 alphalens 报告目录（环境变量启用时存在）
                    if stage == "FACTOR":
                        alphalens_dir = result.get("metadata", {}).get("alphalens_report_dir", "")
                        if alphalens_dir and os.path.isdir(alphalens_dir):
                            self.archiver.save_artifact_copy(stage, alphalens_dir, inputs=upstream_inputs)
                logger.info(f"阶段 {stage} 执行成功, 产物: {artifact}")
                if self.archiver:
                    self.archiver.record_stage_end(stage, "success")
                    self.archiver.write_step_summary(stage, step_num)
                return True
            else:
                error_msg = result.get("error", "未知错误")
                logger.error(f"阶段 {stage} 执行失败: {error_msg}")
                self.ctx.add_error(f"{stage}: {error_msg}")
                if self.archiver:
                    self.archiver.record_stage_end(stage, "failed")
                    self.archiver.write_step_summary(stage, step_num)
                return False
        except Exception as e:
            error_msg = f"阶段 {stage} 执行异常: {str(e)}"
            logger.exception(error_msg)
            self.ctx.add_error(error_msg)
            if self.archiver:
                self.archiver.record_step_result(stage, {"success": False, "error": error_msg})
                self.archiver.record_stage_end(stage, "failed")
                self.archiver.write_step_summary(stage, step_num)
            return False

    def run_pipeline(self, user_input: str = None, ctx: Context = None,
                     llm_responses: dict = None) -> dict:
        """
        执行全流程管道

        参数:
            user_input: 用户自然语言输入（与 ctx 二选一）
            ctx: 预构建的 Context 对象（与 user_input 二选一）
            llm_responses: 可选，外部传入的 LLM 分析结果，用于覆盖 skill 内部生成的解读。
                          格式 {"technical": {...}, "fundamental": {...}}
                          注意：reports-engine 已在内部自动调用 LLM 并注入，
                          此参数仅在需要外部覆盖时使用。
        """
        if ctx:
            self.ctx = ctx
        elif user_input:
            self.ctx = self.parse_intent(user_input)
        else:
            return {"success": False, "error": "需要提供 user_input 或 ctx"}

        self.archiver = RunArchiver(ARCHIVE_DIR)
        run_dir = self.archiver.create_run(self.ctx.task_id)
        self.ctx.run_dir = run_dir
        logger.info(f"运行归档目录: {run_dir}")

        # P0-4 Frozen Core：run_pipeline 开始时记录 git status 快照
        if self._git_tracker is not None:
            try:
                self._git_tracker.pre_snapshot()
            except Exception as e:
                logger.debug(f"P0-4 pre_snapshot 跳过: {e}")

        results = {"success": True, "completed_stages": [], "failed_stages": [],
                   "summary": "", "archive_dir": run_dir, "llm_prompts": {}}

        # P1-2 显式 FSM 校验层（不替换 STAGES，只校验转移合法性）
        fsm = DailyFSM()
        fsm_current = "INITIALIZED"
        self.ctx.metadata["fsm_transitions"] = []

        for step_num, stage in enumerate(self.ctx.target_stages, 1):
            # P1-2.6: 阶段间 FSM 转移校验
            try:
                fsm_current = fsm.transition(fsm_current, stage)
                self.ctx.metadata["fsm_transitions"].append(
                    {"from": fsm_current, "to": stage}
                )
            except ValueError as fsm_err:
                logger.error(f"P1-2 FSM 非法转移: {fsm_err}")
                self.ctx.add_error(f"FSM: {fsm_err}")
                results["success"] = False
                break

            success = self.execute_stage(stage, step_num)
            if success:
                results["completed_stages"].append(stage)
                # P1-2.7: data_quality abort → MANUAL_ATTENTION
                if stage == "DATA":
                    dq = self.ctx.metadata.get("DATA", {}).get("data_quality", {})
                    if isinstance(dq, dict) and dq.get("mode") == "abort":
                        logger.error("P1-2 数据质量 abort → 转入 MANUAL_ATTENTION")
                        try:
                            fsm.transition(fsm_current, STATE_MANUAL_ATTENTION)
                        except ValueError:
                            pass
                        results["success"] = False
                        break
            else:
                results["failed_stages"].append(stage)
                # P1-2.9: 异常时创建 IncidentFSM 重试
                if stage not in ["DATA", "BACKTEST"]:
                    incident = IncidentFSM()
                    try:
                        incident.transition("DETECTED", "CLASSIFIED")
                        incident.transition("CLASSIFIED", "RETRYING")
                        incident.transition("RETRYING", "CLASSIFIED")
                        incident.transition("CLASSIFIED", "DEGRADED")
                        fsm.transition(fsm_current, STATE_DEGRADED)
                        fsm_current = STATE_DEGRADED
                        logger.warning(f"P1-2 阶段 {stage} 失败，FSM 转 DEGRADED")
                    except ValueError:
                        pass
                    continue
                if stage in ["DATA", "BACKTEST"]:
                    results["success"] = False
                    logger.error(f"关键阶段 {stage} 失败，停止管道")
                    break

        # 收集 llm_prompts 和 llm_status（从 REPORT 阶段 metadata 提取）
        if "REPORT" in results["completed_stages"]:
            report_meta = self.ctx.metadata.get("REPORT", {})
            llm_prompts = report_meta.get("llm_prompts", {})
            if llm_prompts:
                results["llm_prompts"] = llm_prompts
            results["llm_status"] = report_meta.get("llm_status", "unknown")

        # 外部覆盖：如果传入了 llm_responses，重新注入归档 HTML
        if llm_responses and "REPORT" in results["completed_stages"]:
            self._inject_llm_to_archive(run_dir, llm_responses, results)

        results["summary"] = self._generate_summary()

        if self.archiver:
            self.archiver.write_pipeline_summary(
                completed=results["completed_stages"],
                failed=results["failed_stages"],
                target_stages=self.ctx.target_stages,
                user_intent=self.ctx.user_intent,
                task_id=self.ctx.task_id,
                errors=self.ctx.errors
            )
            # P1-3.6: run 结束时落盘 run_manifest.json（含各阶段 sha256/latency/status）
            # 紧跟在 pipeline_summary 之后，作为血缘与可重放校验的权威记录
            try:
                manifest_path = self.archiver.write_run_manifest(
                    inputs_sha256=self._collect_inputs_sha256()
                )
                results["run_manifest"] = manifest_path
            except Exception as e:
                logger.warning(f"P1-3 write_run_manifest 失败（不阻断）: {e}")

        results["context"] = self.ctx.to_dict()

        return results

    def _collect_inputs_sha256(self) -> Dict[str, str]:
        """P1-3: 收集各阶段上游输入产物的 sha256，用于 run_manifest 的 inputs_sha256 字段。

        对每个已执行的阶段，计算其产物文件的 sha256 作为下一阶段的输入指纹。
        返回 {stage: sha256} 映射（产物不存在的阶段跳过）。
        """
        try:
            from scripts.artifact_store import compute_sha256
        except ImportError:
            return {}

        inputs_sha: Dict[str, str] = {}
        for stage, path in self.ctx.artifacts.items():
            if not path or not os.path.isfile(path):
                continue
            try:
                inputs_sha[stage] = compute_sha256(path)
            except Exception:
                continue
        return inputs_sha

    def _inject_llm_to_archive(self, run_dir: str, llm_responses: dict, results: dict):
        """将 agent 传入的 LLM 分析结果注入归档 HTML 中，替换占位符"""
        inject_result = self.inject_llm(run_dir, llm_responses)
        if inject_result.get("injected_count", 0) > 0:
            results["metadata"] = results.get("metadata", {})
            results["metadata"]["llm_injected"] = True
            results["metadata"]["llm_injected_files"] = inject_result["injected_count"]

    def inject_llm(self, run_dir: str, llm_responses: dict) -> dict:
        """Agent 调用 LLM 后，将结果注入已归档的 HTML 报告

        两阶段执行流程：
        1. run_pipeline(ctx) → 生成报告(含占位符) → 返回 llm_prompts
        2. agent 调用 LLM → 拿到 llm_responses
        3. inject_llm(run_dir, llm_responses) → 替换占位符

        参数:
            run_dir: 归档目录路径（run_pipeline 返回的 archive_dir）
            llm_responses: LLM 分析结果，格式 {"technical": {...}, "fundamental": {...}}

        返回:
            {"injected_count": int, "injected_files": [str]}
        """
        import html as _html_lib

        # 找到归档目录中的 REPORT artifacts
        report_artifacts_dir = None
        for step_name in os.listdir(run_dir):
            if step_name.startswith("step_") and "REPORT" in step_name:
                report_artifacts_dir = os.path.join(run_dir, step_name, "artifacts")
                break
        if not report_artifacts_dir or not os.path.isdir(report_artifacts_dir):
            logger.warning("未找到归档 REPORT artifacts 目录，跳过 LLM 注入")
            return {"injected_count": 0, "injected_files": []}

        def _render_tech(resp: dict) -> str:
            score = float(resp.get("technical_score", 0))
            sc = "positive" if score >= 60 else ("negative" if score < 40 else "")
            opt = []
            for key, title, tag in [("capital_flow_analysis", "资金面分析", True),
                                     ("dragon_tiger_analysis", "龙虎榜解读", True),
                                     ("price_limit_analysis", "涨跌停分析", False)]:
                v = resp.get(key, "")
                if v:
                    t = " <span class='llm-tag-a'>A股特色</span>" if tag else ""
                    opt.append(f"<h4>{title}{t}</h4><p>{_html_lib.escape(v)}</p>")
            return (
                f'<div class="llm-analysis-header">'
                f'<span class="llm-badge llm-badge-{sc}">技术评分 {score:.0f}</span>'
                f'<span class="llm-badge">趋势：{_html_lib.escape(resp.get("trend_direction", ""))}</span>'
                f'<span class="llm-badge">置信度：{_html_lib.escape(resp.get("trend_confidence", ""))}</span>'
                f'</div>'
                f'<div class="llm-analysis-body">'
                f'<h4>整体评估</h4><p>{_html_lib.escape(resp.get("overall_assessment", ""))}</p>'
                f'<h4>多周期趋势分析</h4><p>{_html_lib.escape(resp.get("trend_analysis", ""))}</p>'
                f'<h4>技术指标信号解读</h4><p>{_html_lib.escape(resp.get("indicator_analysis", ""))}</p>'
                f'<h4>关键价位分析</h4><p>{_html_lib.escape(resp.get("key_levels", ""))}</p>'
                f'<h4>风险信号</h4><p>{_html_lib.escape(resp.get("risk_signals", ""))}</p>'
                f'<h4>短期展望</h4><p>{_html_lib.escape(resp.get("short_term_outlook", ""))}</p>'
                f'{"".join(opt)}'
                f'</div>'
            )

        def _render_fund(resp: dict) -> str:
            score = float(resp.get("fundamental_score", 0))
            sc = "positive" if score >= 60 else ("negative" if score < 40 else "")
            opt = []
            for key, title, tag in [("industry_analysis", "行业分析与景气度", False),
                                     ("financial_statement_analysis", "财务报表分析", False),
                                     ("shareholder_analysis", "股东结构与资本运作", True)]:
                v = resp.get(key, "")
                if v:
                    t = " <span class='llm-tag-a'>A股特色</span>" if tag else ""
                    opt.append(f"<h4>{title}{t}</h4><p>{_html_lib.escape(v)}</p>")
            return (
                f'<div class="llm-analysis-header">'
                f'<span class="llm-badge llm-badge-{sc}">基本面评分 {score:.0f}</span>'
                f'<span class="llm-badge">估值：{_html_lib.escape(resp.get("valuation_level", ""))}</span>'
                f'</div>'
                f'<div class="llm-analysis-body">'
                f'<h4>整体评估</h4><p>{_html_lib.escape(resp.get("overall_assessment", ""))}</p>'
                f'<h4>估值分析</h4><p>{_html_lib.escape(resp.get("valuation_analysis", ""))}</p>'
                f'<h4>盈利能力分析</h4><p>{_html_lib.escape(resp.get("profitability_analysis", ""))}</p>'
                f'<h4>成长性分析</h4><p>{_html_lib.escape(resp.get("growth_analysis", ""))}</p>'
                f'<h4>风险因素</h4><p>{_html_lib.escape(resp.get("risk_factors", ""))}</p>'
                f'{"".join(opt)}'
                f'</div>'
            )

        injected_count = 0
        injected_files = []
        tech_resp = llm_responses.get("technical")
        fund_resp = llm_responses.get("fundamental")

        for fname in os.listdir(report_artifacts_dir):
            if not fname.endswith(".html"):
                continue
            fpath = os.path.join(report_artifacts_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                html = f.read()
            modified = False
            if tech_resp and "<!--LLM_TECHNICAL_ANALYSIS_PLACEHOLDER-->" in html:
                html = html.replace(
                    "<!--LLM_TECHNICAL_ANALYSIS_PLACEHOLDER-->",
                    _render_tech(tech_resp)
                )
                modified = True
            if fund_resp and "<!--LLM_FUNDAMENTAL_ANALYSIS_PLACEHOLDER-->" in html:
                html = html.replace(
                    "<!--LLM_FUNDAMENTAL_ANALYSIS_PLACEHOLDER-->",
                    _render_fund(fund_resp)
                )
                modified = True
            if modified:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(html)
                injected_count += 1
                injected_files.append(fname)
                logger.info(f"LLM 内容已注入归档报告: {fname}")

        return {"injected_count": injected_count, "injected_files": injected_files}

    def _generate_summary(self) -> str:
        """生成可读的管道执行摘要"""
        if not self.ctx:
            return "无执行记录"

        lines = [
            f"任务ID: {self.ctx.task_id}",
            f"用户意图: {self.ctx.user_intent}",
            f"目标阶段: {' → '.join(self.ctx.target_stages)}",
            f"归档目录: {self.ctx.run_dir or 'N/A'}",
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
            "context": dict,
            "archive_dir": str
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
    parser.add_argument("--force", action="store_true", help="强制刷新，忽略缓存产物重新执行所有阶段")

    args = parser.parse_args()

    if args.force:
        os.environ["QUANT_FORCE_REFRESH"] = "1"

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