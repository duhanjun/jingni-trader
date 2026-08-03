"""
A股因子引擎主逻辑
负责因子计算（行业中性化/IC分析/相关性去冗余/多因子融合已迁移到 ProcessorChain）
"""
import os
import sys
import logging
import json
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from scripts.config import FACTOR_BACKEND, FACTOR_DIR

# 表达式引擎和扩展因子库
from expression import FactorExpressionEngine as ExpressionEngine
from factors import Alpha158FactorEngine
from factors.factors_config import FACTOR_CATEGORIES

logger = logging.getLogger("a-share-factor-engine")


class FactorEngine:
    """A股因子引擎"""

    def __init__(self):
        self.calculator = self._load_calculator()

    def _load_calculator(self):
        """根据配置加载因子计算器"""
        try:
            if FACTOR_BACKEND == "talib":
                from adapters.talib_calculator import TalibCalculator
                return TalibCalculator()
            elif FACTOR_BACKEND == "pandas_ta":
                from adapters.pandas_ta_calculator import PandasTaCalculator
                return PandasTaCalculator()
            else:
                raise ValueError(f"不支持的因子后端: {FACTOR_BACKEND}")
        except ImportError:
            logger.warning(f"因子计算后端 {FACTOR_BACKEND} 不可用，使用内置计算")
            return None

    def compute_a_share_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算A股专有Alpha因子

        参数:
            data: 清洗后的日线数据

        返回:
            DataFrame，列为 code, date, [各Alpha因子]
        """
        if data.empty:
            return data

        logger.info("开始计算A股专用Alpha因子...")
        df = data.sort_values(['code', 'date']).copy()
        result = df[['code', 'date']].copy()

        # 一次 groupby，所有 transform 复用
        grouped_close = df.groupby('code')['close']
        result['ret_1d'] = grouped_close.pct_change()
        result['ret_5d'] = grouped_close.pct_change(5)
        result['ret_20d'] = grouped_close.pct_change(20)
        result['ret_60d'] = grouped_close.pct_change(60)

        result['reversal_5d'] = -result['ret_5d']
        result['reversal_20d'] = -result['ret_20d']

        has_amount = 'amount' in df.columns and not df['amount'].isna().all()
        has_turnover = 'turnover_rate' in df.columns and not df['turnover_rate'].isna().all()

        # estimated_mv 仅作为 lncap 的中间变量，不输出到结果
        if has_amount and has_turnover:
            mv = df['amount'] / df['turnover_rate'].replace(0, np.nan) * 100
            result['lncap'] = mv.replace(0, np.nan).apply(lambda x: np.log(x) if x > 0 else np.nan)
        else:
            result['lncap'] = np.nan

        if has_turnover:
            grouped_turnover = df.groupby('code')['turnover_rate']
            result['turnover_20d'] = grouped_turnover.transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            result['turnover_5d'] = grouped_turnover.transform(
                lambda x: x.rolling(5, min_periods=3).mean()
            )
            result['turnover_change'] = result['turnover_5d'] / result['turnover_20d'].replace(0, np.nan) - 1
        else:
            result['turnover_20d'] = np.nan
            result['turnover_5d'] = np.nan
            result['turnover_change'] = np.nan

        result['volatility_20d'] = grouped_close.transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )

        result['volume_20d'] = df.groupby('code')['volume'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        result['volume_ratio'] = df['volume'] / result['volume_20d'].replace(0, np.nan)

        if 'change_pct' in df.columns:
            result['money_flow_raw'] = df['change_pct'] * df.get('amount', df['volume'])
        else:
            result['money_flow_raw'] = result['ret_1d'] * df.get('amount', df['volume'])
        result['money_flow_20d'] = result.groupby('code')['money_flow_raw'].transform(
            lambda x: x.rolling(20, min_periods=5).sum()
        )

        logger.info(f"A股因子计算完成，共 {len(result.columns) - 2} 个因子")
        return result

    def compute_expression_factors(
        self, data: pd.DataFrame, expressions: dict = None
    ) -> pd.DataFrame:
        """
        使用表达式引擎计算因子（借鉴 quant-stream）
        支持声明式因子定义: RANK(DELTA($close, 5))

        参数:
            data: 含 code, date, close, volume 等列的 DataFrame
            expressions: {因子名: 表达式} 字典，默认使用预设表达式

        返回:
            含所有表达式因子的 DataFrame
        """
        engine = ExpressionEngine()
        if expressions is None:
            return engine.compute_preset(data)
        result = data[["code", "date"]].copy()
        for name, expr in expressions.items():
            factor_result = engine.compute(data, expr, name=name)
            result = result.merge(factor_result, on=["code", "date"], how="left")
        return result

    def compute_extended_factors(
        self, data: pd.DataFrame, factor_names: list = None
    ) -> pd.DataFrame:
        """
        计算扩展因子库（借鉴 Qlib Alpha158）
        47 个因子，覆盖动量/反转/波动率/成交量/技术指标/资金流向 6 大类

        参数:
            data: 含 code, date, close, volume, high, low, open 等列的 DataFrame
            factor_names: 要计算的因子名列表，默认全部

        返回:
            含所有扩展因子的 DataFrame
        """
        engine = Alpha158FactorEngine()
        return engine.compute(data, factor_names)


def _try_load_factor_from_datafeed(ctx) -> Optional[pd.DataFrame]:
    """尝试从jingni-datafeed(惊泥因子库)加载已沉淀的因子数据。

    返回 None 表示未启用或取数失败，调用方应回退到本地计算路径。
    """
    factor_source = ctx.metadata.get("factor_source", "local")
    if factor_source == "local":
        return None

    if not (os.environ.get("JINGNI_URL") and os.environ.get("JINGNI_TOKEN")):
        return None

    try:
        datafeed_scripts = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "jingni-datafeed", "scripts"
        ))
        if datafeed_scripts not in sys.path:
            sys.path.insert(0, datafeed_scripts)

        from jingni_client import JingniClient
        from config import JingniConfig

        client = JingniClient(JingniConfig.from_env())
        uid = os.environ.get("JINGNI_DEFAULT_DATASOURCE_UID", "factor-store")

        start = (ctx.start_date or "").replace("-", "")
        end = (ctx.end_date or "").replace("-", "")
        sql = (
            "SELECT ts_code AS code, trade_date AS date, factor_name, factor_value "
            "FROM factor_daily "
            f"WHERE trade_date BETWEEN '{start}' AND '{end}'"
        )
        if ctx.stock_pool:
            codes = ",".join(f"'{c}'" for c in ctx.stock_pool)
            sql += f" AND ts_code IN ({codes})"

        result = client.query_sql(uid=uid, raw_sql=sql, format="table")
        rows = result.to_table()
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["factor_value"] = pd.to_numeric(df["factor_value"], errors="coerce")
        logger.info(f"从惊泥因子库加载 {len(df)} 行因子数据")
        return df
    except Exception as e:
        logger.warning(f"从jingni-datafeed取因子数据失败，将回退到本地计算: {e}")
        return None


def _maybe_generate_alphalens_reports(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    factor_names: List[str],
    task_id: str,
) -> str:
    """T3-6: 可选生成 alphalens 完整因子分析报告。

    环境变量 QUANT_ALPHALENS_REPORT=1 启用；默认关闭，返回空字符串。
    每个因子输出 4 PNG + 1 HTML + 1 JSON 到 workspace/reports/alphalens/<task_id>/。
    alphalens-reloaded 不可用时自动降级到方案 C（自研轻量分层回测，仅 JSON+HTML）。
    失败时静默记录日志，不阻塞主流程。

    返回
    ----
    报告目录路径；未启用时返回空字符串
    """
    if os.environ.get("QUANT_ALPHALENS_REPORT", "0") != "1":
        return ""

    # 延迟导入避免 alphalens 缺失时影响模块加载
    try:
        from scripts.alphalens_adapter import AlphalensAdapter
    except ImportError:
        # 跨 skill 加载场景下 scripts 包可能指向子 skill，按相对路径加载
        import importlib.util as _ilu
        _adapter_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scripts", "alphalens_adapter.py"
        )
        _spec = _ilu.spec_from_file_location("_alphalens_adapter_tmp", _adapter_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        AlphalensAdapter = _mod.AlphalensAdapter

    # 输出目录：workspace/reports/alphalens/<task_id>/
    _work_dir = os.environ.get("QUANT_WORK_DIR", "./workspace")
    report_dir = os.path.join(_work_dir, "reports", "alphalens", task_id or "default")
    os.makedirs(report_dir, exist_ok=True)

    # 价格数据：从原始 df 提取 code/date/close
    price_cols = {"code", "date", "close"}
    if not price_cols.issubset(price_df.columns):
        logger.warning("alphalens 报告生成跳过：price_df 缺少 code/date/close 列")
        return report_dir

    success_count = 0
    fail_count = 0
    for factor_name in (factor_names or []):
        if factor_name not in factor_df.columns:
            continue
        try:
            result = AlphalensAdapter.generate_for_factor(
                factor_df=factor_df,
                price_df=price_df,
                factor_name=factor_name,
                output_dir=report_dir,  # 字符串路径，AlphalensAdapter 内部会转 Path
            )
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.warning(f"alphalens 报告生成失败（因子 {factor_name}）: {e}")
            fail_count += 1

    logger.info(
        f"alphalens 报告生成完成：成功 {success_count} 个，失败 {fail_count} 个，目录 {report_dir}"
    )
    return report_dir


# ---------------------------------------------------------------------------
# Processor Pipeline 路径（旧硬编码路径已在 v3.0 移除）
# ---------------------------------------------------------------------------


def _run_processor_chain(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: List[str],
    task_id: str,
    data_path: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str], List[str]]:
    """ProcessorChain 路径。

    通过 ``pipeline.yaml`` 声明式配置因子处理流程，自动记录实验元数据到
    ``ExperimentRecorder``（manifest.json 含 7 字段，接入 P1-3 sha256 机制）。

    使用模块顶部已导入的 `_ProcessorChain` / `_ProcessContext` / `_load_pipeline_config`
    / `_ExperimentRecorder`，避免在测试环境中 `sys.modules['scripts']` 被重置后
    延迟导入失败。

    Returns
    -------
    (final_df, ic_results, selected_factors, removed_factors)
    """
    # 加载 pipeline 配置（work_dir 优先 → skill 默认 → 兜底默认链）
    _work_dir = os.environ.get("QUANT_WORK_DIR", "./workspace")
    work_dir = Path(_work_dir) if _work_dir else None

    try:
        processor_list = _load_pipeline_config(work_dir=work_dir)
    except Exception as e:
        logger.warning(
            f"加载 pipeline.yaml 失败 ({e})，回退兜底默认链"
        )
        processor_list = _default_processors()

    # 构造 ProcessorChain
    chain = _ProcessorChain(processor_list, fail_fast=False)

    # 初始化 Recorder
    _archive_dir = os.path.join(_work_dir, "archives", "factor_engine")
    try:
        recorder = _ExperimentRecorder(
            archive_dir=Path(_archive_dir),
            pipeline_config=chain.describe_chain(),
            input_data_paths=[data_path] if data_path else None,
        )
    except Exception as e:
        logger.warning(f"ExperimentRecorder 初始化失败，降级为无记录模式: {e}")
        recorder = None

    # 构造 ProcessContext
    proc_ctx = _ProcessContext(
        forward_returns=forward_returns,
        factor_names=list(factor_names),
        task_id=task_id or "",
        work_dir=work_dir,
        recorder=recorder,
    )

    # 执行 ProcessorChain
    final_df = chain.run(factor_df, proc_ctx)

    # 从 ctx 提取 IC 结果与选中因子
    ic_results = proc_ctx.ic_results or {}
    selected_factors = proc_ctx.selected_factors or []
    corr_meta = proc_ctx.metadata.get("correlation_result", {})
    removed_factors = corr_meta.get("removed_factors", [])

    # 记录输出产物并 finalize Recorder
    if recorder is not None:
        try:
            recorder.log_output_artifact("factor_data", "<runtime>")
            recorder.finalize()
        except Exception as e:
            logger.warning(f"Recorder finalize 失败（不阻塞主流程）: {e}")

    # 如果 final_df 没有 alpha_score 列（如 Fusion 被禁用），补一个空列保持结构兼容
    if "alpha_score" not in final_df.columns:
        logger.warning("ProcessorChain 输出未包含 alpha_score 列（FusionProcessor 可能被禁用）")
        final_df["alpha_score"] = 0.0

    return final_df, ic_results, selected_factors, removed_factors


def run(ctx) -> Dict[str, Any]:
    """
    a-share-factor-engine 的 run 函数
    由 jingnitrader 调度

    参数:
        ctx: Context 对象，需包含:
            - artifacts['DATA']: 清洗后数据文件路径

    返回:
        {
            "success": bool,
            "artifact_path": str,
            "metadata": {...},
            "error": str
        }
    """
    try:
        data_path = ctx.get_artifact("DATA")
        if not data_path or not os.path.exists(data_path):
            return {
                "success": False,
                "artifact_path": "",
                "metadata": {},
                "error": "数据产物不存在，请先运行 a-share-data-engine"
            }

        existing = ctx.get_artifact("FACTOR")
        if existing and os.path.exists(existing):
            return {
                "success": True,
                "artifact_path": existing,
                "metadata": {"source": "cache"},
                "error": ""
            }

        # 优先尝试从惊泥因子库取数（factor_source=jingni/auto 时）
        datafeed_df = _try_load_factor_from_datafeed(ctx)

        if datafeed_df is not None:
            # 走 jingni-datafeed 路径
            os.makedirs(FACTOR_DIR, exist_ok=True)
            output_path = os.path.join(FACTOR_DIR, "factor_data.parquet")
            datafeed_df.to_parquet(output_path, index=False)
            return {
                "success": True,
                "artifact_path": output_path,
                "metadata": {
                    "factor_source": "jingni",
                    "rows": len(datafeed_df),
                    "source": "jingni-datafeed",
                },
                "error": ""
            }

        # 回退到本地计算路径
        df = pd.read_parquet(data_path)
        if df.empty:
            return {"success": False, "artifact_path": "", "metadata": {}, "error": "数据为空"}

        engine = FactorEngine()

        # ── 按 FACTOR_CATEGORIES 配置调度各类因子模块 ──
        # 每类因子独立计算，缺失依赖数据时自动跳过，互不影响
        all_factor_dfs: List[pd.DataFrame] = []
        computed_categories: List[str] = []
        skipped_categories: List[str] = []

        for category, cfg in FACTOR_CATEGORIES.items():
            if not cfg.get("enabled", True):
                continue

            requires = cfg.get("requires", "price_data")
            module_path = cfg.get("module", "")

            # 检查依赖数据产物是否存在
            if requires != "price_data":
                dep_path = ctx.get_artifact(requires) if hasattr(ctx, 'get_artifact') else None
                if not dep_path or not os.path.exists(dep_path):
                    logger.warning(f"跳过 {category} 因子：缺少依赖数据 {requires}")
                    skipped_categories.append(category)
                    continue

            # 调度因子计算
            try:
                if module_path == "engine.compute_a_share_factors":
                    # 动量/量价因子：调用 FactorEngine 内置方法
                    factor_df_part = engine.compute_a_share_factors(df)
                else:
                    # 其余类别：动态导入模块并调用 compute(price_data, ctx)
                    mod = importlib.import_module(module_path)
                    factor_df_part = mod.compute(df, ctx)

                if factor_df_part is not None and not factor_df_part.empty:
                    all_factor_dfs.append(factor_df_part)
                    computed_categories.append(category)
                    logger.info(f"因子类别 [{category}] 计算完成: {len(factor_df_part.columns) - 2} 个因子")
                else:
                    logger.warning(f"因子类别 [{category}] 返回空数据，已跳过")
                    skipped_categories.append(category)
            except Exception as e:
                logger.warning(f"因子类别 [{category}] 计算失败，已跳过: {e}")
                skipped_categories.append(category)

        if not all_factor_dfs:
            return {
                "success": False,
                "artifact_path": "",
                "metadata": {},
                "error": "所有因子类别计算失败或无数据"
            }

        # 合并所有因子列（按 code, date 对齐）
        factor_df = all_factor_dfs[0]
        for extra_df in all_factor_dfs[1:]:
            merge_cols = [c for c in ['code', 'date'] if c in extra_df.columns]
            factor_cols = [c for c in extra_df.columns if c not in merge_cols]
            if factor_cols:
                factor_df = factor_df.merge(
                    extra_df[merge_cols + factor_cols],
                    on=merge_cols,
                    how='left'
                )

        forward_returns = pd.DataFrame()
        forward_returns['code'] = df['code']
        forward_returns['date'] = df['date']
        for period in [1, 5, 20]:
            forward_returns[f'ret_forward_{period}d'] = df.groupby('code')['close'].transform(
                lambda x: x.shift(-period) / x - 1
            )

        factor_names = [c for c in factor_df.columns
                       if c not in ['code', 'date', 'industry',
                                    'money_flow_raw', 'ret_1d', 'ret_5d', 'ret_20d', 'ret_60d',
                                    'turnover_5d']]

        # ── ProcessorChain 路径（旧硬编码路径已在 v3.0 移除） ──
        logger.info("走 ProcessorChain 路径（pipeline.yaml 声明式配置）")
        final_df, ic_results, selected_factors, removed_factors = _run_processor_chain(
            factor_df=factor_df,
            forward_returns=forward_returns,
            factor_names=factor_names,
            task_id=ctx.task_id,
            data_path=data_path,
        )

        os.makedirs(FACTOR_DIR, exist_ok=True)
        output_path = os.path.join(FACTOR_DIR, "factor_data.parquet")
        final_df.to_parquet(output_path, index=False)

        ic_report_path = os.path.join(FACTOR_DIR, "ic_report.json")
        with open(ic_report_path, 'w', encoding='utf-8') as f:
            json.dump(ic_results, f, ensure_ascii=False, indent=2, default=str)

        # T3-6: 可选生成 alphalens 完整因子分析报告
        # 环境变量 QUANT_ALPHALENS_REPORT=1 启用，每个因子输出 4 PNG + 1 HTML + 1 JSON
        alphalens_report_dir = _maybe_generate_alphalens_reports(
            factor_df=factor_df,
            price_df=df,
            factor_names=factor_names,
            task_id=ctx.task_id,
        )

        return {
            "success": True,
            "artifact_path": output_path,
            "metadata": {
                "factor_names": factor_names,
                "selected_factors": selected_factors,
                "removed_factors": removed_factors,
                "ic_results": ic_results,
                "fusion_method": "ic_weighted",
                "computed_categories": computed_categories,
                "skipped_categories": skipped_categories,
                "alphalens_report_dir": alphalens_report_dir,
            },
            "error": ""
        }

    except Exception as e:
        logger.exception("因子引擎执行失败")
        return {
            "success": False,
            "artifact_path": "",
            "metadata": {},
            "error": str(e)
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            ctx_dict = json.load(f)
        from scripts.context import Context
        ctx = Context.from_dict(ctx_dict)
    else:
        from scripts.context import Context
        ctx = Context(
            task_id="test_factor",
            stock_pool=[],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        ctx.update_artifact("DATA", "./workspace/data/cleaned_data.parquet")

    result = run(ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# 优化模块入口（已整合到 scripts/optimizations/）
# 按需导入，例如: from scripts.optimizations.ic_vectorized import ic_analysis_batch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Processor Pipeline 模块入口
# 使用方式: from engine import processors
#           processors.ProcessorChain / processors.NeutralizeProcessor / ...
# ---------------------------------------------------------------------------
from scripts.processors.base import (
    Processor as _Processor,
    ProcessContext as _ProcessContext,
    ProcessorRequirementError as _ProcessorRequirementError,
)
from scripts.processors.chain import (
    ProcessorChain as _ProcessorChain,
    ChainValidationError as _ChainValidationError,
)
from scripts.processors.loader import (
    load_pipeline_config as _load_pipeline_config,
    parse_yaml_to_processors as _parse_yaml_to_processors,
    register_processor as _register_processor,
    PROCESSOR_REGISTRY as _PROCESSOR_REGISTRY,
    _default_processors as _default_processors,
)
from scripts.processors.neutralize import NeutralizeProcessor as _NeutralizeProcessor
from scripts.processors.winsorize import WinsorizeProcessor as _WinsorizeProcessor
from scripts.processors.fillna import FillnaProcessor as _FillnaProcessor
from scripts.processors.standardize import StandardizeProcessor as _StandardizeProcessor
from scripts.processors.ic_analysis import ICAnalysisProcessor as _ICAnalysisProcessor
from scripts.processors.correlation_filter import (
    CorrelationFilterProcessor as _CorrelationFilterProcessor,
)
from scripts.processors.fusion import FusionProcessor as _FusionProcessor
from scripts.recorder import ExperimentRecorder as _ExperimentRecorder


class processors:
    """因子引擎 Processor Pipeline 模块集合

    通过 ``from engine import processors`` 访问所有 Processor 相关类与工具函数。
    """
    # 基类与异常
    Processor = _Processor
    ProcessContext = _ProcessContext
    ProcessorRequirementError = _ProcessorRequirementError
    ChainValidationError = _ChainValidationError

    # 调度器与加载器
    ProcessorChain = _ProcessorChain
    load_pipeline_config = _load_pipeline_config
    parse_yaml_to_processors = _parse_yaml_to_processors
    register_processor = _register_processor
    PROCESSOR_REGISTRY = _PROCESSOR_REGISTRY

    # 7 个内置 Processor
    NeutralizeProcessor = _NeutralizeProcessor
    WinsorizeProcessor = _WinsorizeProcessor
    FillnaProcessor = _FillnaProcessor
    StandardizeProcessor = _StandardizeProcessor
    ICAnalysisProcessor = _ICAnalysisProcessor
    CorrelationFilterProcessor = _CorrelationFilterProcessor
    FusionProcessor = _FusionProcessor

    # 实验记录器
    ExperimentRecorder = _ExperimentRecorder

    @staticmethod
    def get_all_processors():
        """返回所有已注册的 Processor 类"""
        return dict(_PROCESSOR_REGISTRY)

    @staticmethod
    def create_default_chain():
        """创建默认 ProcessorChain（兜底默认链，仅 IC + Correlation + Fusion）"""
        return _ProcessorChain(_default_processors())
