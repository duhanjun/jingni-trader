"""
A股因子引擎主逻辑
负责因子计算、行业中性化、IC分析、相关性去冗余、多因子融合
"""
import os
import sys
import logging
import json
import importlib
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from scripts.config import (
    FACTOR_BACKEND, FACTOR_DIR, IC_TYPE,
    NEUTRALIZE_INDUSTRY, NEUTRALIZE_MARKET_CAP,
    QUANTILES, MIN_IC, MIN_IC_IR, MAX_CORRELATION
)

# 新增: 表达式引擎和扩展因子库
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

        result['ret_1d'] = df.groupby('code')['close'].pct_change()
        result['ret_5d'] = df.groupby('code')['close'].pct_change(5)
        result['ret_20d'] = df.groupby('code')['close'].pct_change(20)
        result['ret_60d'] = df.groupby('code')['close'].pct_change(60)

        result['reversal_5d'] = -result['ret_5d']
        result['reversal_20d'] = -result['ret_20d']

        has_amount = 'amount' in df.columns and not df['amount'].isna().all()
        has_turnover = 'turnover_rate' in df.columns and not df['turnover_rate'].isna().all()

        if has_amount and has_turnover:
            mv = result['estimated_mv'] = df['amount'] / df['turnover_rate'].replace(0, np.nan) * 100
            result['lncap'] = mv.replace(0, np.nan).apply(lambda x: np.log(x) if x > 0 else np.nan)
        else:
            result['estimated_mv'] = np.nan
            result['lncap'] = np.nan

        if has_turnover:
            result['turnover_20d'] = df.groupby('code')['turnover_rate'].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            result['turnover_5d'] = df.groupby('code')['turnover_rate'].transform(
                lambda x: x.rolling(5, min_periods=3).mean()
            )
            result['turnover_change'] = result['turnover_5d'] / result['turnover_20d'].replace(0, np.nan) - 1
        else:
            result['turnover_20d'] = np.nan
            result['turnover_5d'] = np.nan
            result['turnover_change'] = np.nan

        result['volatility_20d'] = df.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )

        result['volume_20d'] = df.groupby('code')['volume'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        result['volume_ratio'] = df['volume'] / result['volume_20d'].replace(0, np.nan)

        if 'change_pct' in df.columns:
            result['money_flow_raw'] = df['change_pct'] * df.get('amount', df['volume'])
            result['money_flow_20d'] = result.groupby('code')['money_flow_raw'].transform(
                lambda x: x.rolling(20, min_periods=5).sum()
            )
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

    def neutralize(
        self,
        factor_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        neutralize_mcap: bool = NEUTRALIZE_MARKET_CAP,
        neutralize_industry: bool = NEUTRALIZE_INDUSTRY
    ) -> pd.DataFrame:
        """因子行业中性化处理

        .. deprecated:: v2.0
            将在 v3.0 移除，请改用 ``NeutralizeProcessor`` 或 ``ProcessorChain``。
        """
        warnings.warn(
            "FactorEngine.neutralize() 将在 v3.0 移除，请改用 NeutralizeProcessor 或 ProcessorChain",
            DeprecationWarning,
            stacklevel=2,
        )
        if not neutralize_industry and not neutralize_mcap:
            return factor_df

        if factor_df.empty:
            return factor_df

        logger.info("开始因子中性化处理...")
        result = factor_df.copy()

        if 'industry' not in result.columns and neutralize_industry:
            result = result.merge(industry_df[['code', 'industry']], on='code', how='left')

        factor_cols = [c for c in factor_df.columns if c not in ['code', 'date', 'industry']]

        for factor in factor_cols:
            if factor not in result.columns:
                continue

            dates = result['date'].unique()
            neutralized_values = pd.Series(index=result.index, dtype=float)

            for dt in dates:
                cross = result[result['date'] == dt].copy()

                if len(cross) < 30:
                    neutralized_values.loc[cross.index] = cross[factor]
                    continue

                X_vars = []
                if neutralize_mcap and 'lncap' in cross.columns:
                    X_vars.append('lncap')
                if neutralize_industry and 'industry' in cross.columns:
                    industry_dummies = pd.get_dummies(cross['industry'], prefix='ind')
                    for col in industry_dummies.columns:
                        cross[col] = industry_dummies[col].values
                        X_vars.append(col)

                if not X_vars:
                    neutralized_values.loc[cross.index] = cross[factor]
                    continue

                X = cross[X_vars].fillna(0).values
                y = cross[factor].fillna(0).values

                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    residual = y - y_pred
                    neutralized_values.loc[cross.index] = residual
                except Exception:
                    neutralized_values.loc[cross.index] = cross[factor]

            result[f"{factor}_neutral"] = neutralized_values

        logger.info("因子中性化完成")
        return result

    def ic_analysis(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """计算因子的IC序列和统计量

        .. deprecated:: v2.0
            将在 v3.0 移除，请改用 ``ICAnalysisProcessor`` 或 ``ProcessorChain``。
        """
        warnings.warn(
            "FactorEngine.ic_analysis() 将在 v3.0 移除，请改用 ICAnalysisProcessor 或 ProcessorChain",
            DeprecationWarning,
            stacklevel=2,
        )
        if factor_df.empty or forward_returns.empty:
            return {}

        logger.info("开始因子IC分析...")

        data = factor_df.merge(
            forward_returns[['code', 'date', 'ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']],
            on=['code', 'date'],
            how='inner'
        )

        if factor_names is None:
            factor_names = [c for c in factor_df.columns
                           if c not in ['code', 'date', 'industry']]

        results = {}

        for forward_col in ['ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']:
            if forward_col not in data.columns:
                continue

            ic_results = []
            for factor in factor_names:
                if factor not in data.columns:
                    continue

                ic_series = self._calc_ic(data, factor, forward_col)
                if ic_series is None or ic_series.empty:
                    continue

                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                ic_ir = ic_mean / ic_std if ic_std > 0 else 0
                ic_positive_ratio = (ic_series > 0).mean()

                ic_results.append({
                    "factor": factor,
                    "forward_period": forward_col,
                    "ic_mean": round(float(ic_mean), 6),
                    "ic_std": round(float(ic_std), 6),
                    "ic_ir": round(float(ic_ir), 4),
                    "ic_positive_ratio": round(float(ic_positive_ratio), 4),
                    "ic_t_stat": round(float(ic_mean / (ic_std / np.sqrt(len(ic_series)))) if ic_std > 0 else 0, 4),
                })

            results[forward_col] = ic_results

        logger.info(f"IC分析完成，共分析 {len(factor_names)} 个因子")
        return results

    def _calc_ic(self, data: pd.DataFrame, factor_col: str, forward_col: str) -> Optional[pd.Series]:
        """计算单个因子的IC时间序列"""
        if forward_col not in data.columns:
            return None

        ic_list = []
        dates = sorted(data['date'].unique())

        for dt in dates:
            cross = data[data['date'] == dt].dropna(subset=[factor_col, forward_col])
            if len(cross) < 10:
                continue

            if IC_TYPE == "spearman":
                ic, _ = stats.spearmanr(cross[factor_col], cross[forward_col], nan_policy='omit')
            else:
                ic, _ = stats.pearsonr(cross[factor_col].fillna(0), cross[forward_col].fillna(0))

            if not np.isnan(ic):
                ic_list.append({"date": dt, "ic": ic})

        if not ic_list:
            return None

        ic_df = pd.DataFrame(ic_list)
        ic_df['date'] = pd.to_datetime(ic_df['date'])
        return ic_df.set_index('date')['ic']

    def correlation_analysis(
        self,
        factor_df: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
        max_correlation: float = MAX_CORRELATION,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """因子相关性分析

        参数:
            backend: ``"pandas"`` / ``"polars"`` / ``"auto"`` / ``None``
                ``None`` 时使用环境变量 ``QUANT_FACTOR_BACKEND`` 默认值

        .. deprecated:: v2.0
            将在 v3.0 移除，请改用 ``CorrelationFilterProcessor`` 或 ``ProcessorChain``。
        """
        warnings.warn(
            "FactorEngine.correlation_analysis() 将在 v3.0 移除，请改用 CorrelationFilterProcessor 或 ProcessorChain",
            DeprecationWarning,
            stacklevel=2,
        )
        return _correlation_analysis(
            factor_df=factor_df,
            factor_names=factor_names,
            max_correlation=max_correlation,
            backend=backend,
        )

    def factor_fusion(
        self,
        factor_df: pd.DataFrame,
        ic_results: Dict[str, Any],
        selected_factors: List[str],
        fusion_method: str = "ic_weighted"
    ) -> pd.DataFrame:
        """多因子融合为复合Alpha信号

        .. deprecated:: v2.0
            将在 v3.0 移除，请改用 ``FusionProcessor`` 或 ``ProcessorChain``。
        """
        warnings.warn(
            "FactorEngine.factor_fusion() 将在 v3.0 移除，请改用 FusionProcessor 或 ProcessorChain",
            DeprecationWarning,
            stacklevel=2,
        )
        if factor_df.empty or not selected_factors:
            return pd.DataFrame()

        logger.info(f"开始多因子融合，方法: {fusion_method}")

        if fusion_method == "ic_weighted":
            weights = self._get_ic_weights(ic_results, selected_factors)
        else:
            weights = {f: 1.0 / len(selected_factors) for f in selected_factors}

        normalized = factor_df[['code', 'date']].copy()
        for factor in selected_factors:
            if factor not in factor_df.columns:
                continue
            normalized[f"{factor}_rank"] = factor_df.groupby('date')[factor].transform(
                lambda x: x.rank(pct=True)
            ).fillna(0.5)  # NaN隔离：缺失因子的rank填充为中性值0.5，避免0权重×NaN污染整行

        rank_cols = [f"{f}_rank" for f in selected_factors if f"{f}_rank" in normalized.columns]
        normalized['alpha_score'] = 0.0
        for f, col in zip(selected_factors, rank_cols):
            w = weights.get(f, 0)
            normalized['alpha_score'] += w * normalized[col]

        result = normalized[['code', 'date', 'alpha_score']].copy()
        logger.info(f"多因子融合完成，权重: {weights}")
        return result

    def _get_ic_weights(self, ic_results: Dict, selected_factors: List[str]) -> Dict[str, float]:
        """根据IC_IR计算因子权重"""
        weights = {}
        total_ic_ir = 0

        ic_list = ic_results.get('ret_forward_5d', [])
        ic_map = {item['factor']: item['ic_ir'] for item in ic_list}

        for factor in selected_factors:
            ic_ir = abs(ic_map.get(factor, 0))
            weights[factor] = ic_ir
            total_ic_ir += ic_ir

        if total_ic_ir > 0:
            weights = {k: v / total_ic_ir for k, v in weights.items()}
        else:
            n = len(selected_factors)
            weights = {k: 1.0 / n for k in selected_factors}

        return weights


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
# T1-6/T1-8: Processor Pipeline 新路径 + 旧路径兼容层
# ---------------------------------------------------------------------------


def _run_legacy_factor_pipeline(
    engine: "FactorEngine",
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str], Dict[str, Any]]:
    """旧 4 步硬编码因子处理路径（兼容回滚用）。

    保留 v1.x 的 5 步逻辑（IC 分析 → 相关性去冗余 → 选因子 → 融合 → 合并 alpha_score），
    通过 ``QUANT_LEGACY_PIPELINE=1`` 环境变量触发。

    Returns
    -------
    (final_df, ic_results, selected_factors, corr_result)
    """
    # 旧路径直接调用 engine 内部方法（绕过 DeprecationWarning）
    # 使用 warnings.catch_warnings 临时抑制 DeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ic_results = engine.ic_analysis(factor_df, forward_returns, factor_names)
        corr_result = engine.correlation_analysis(factor_df, factor_names)
        selected_factors = corr_result['selected_factors']
        fusion_df = engine.factor_fusion(factor_df, ic_results, selected_factors)

    # 合并 alpha_score 到 factor_df
    final_df = factor_df.merge(
        fusion_df[['code', 'date', 'alpha_score']],
        on=['code', 'date'],
        how='left',
    )
    return final_df, ic_results, selected_factors, corr_result


def _run_processor_chain(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: List[str],
    task_id: str,
    data_path: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str], Dict[str, Any]]:
    """新 ProcessorChain 路径（默认路径）。

    通过 ``pipeline.yaml`` 声明式配置因子处理流程，自动记录实验元数据到
    ``ExperimentRecorder``（manifest.json 含 7 字段，接入 P1-3 sha256 机制）。

    Returns
    -------
    (final_df, ic_results, selected_factors, corr_result)
    """
    # 延迟导入避免循环依赖
    from scripts.processors import (
        ProcessorChain,
        ProcessContext,
        load_pipeline_config,
    )
    from scripts.recorder import ExperimentRecorder

    # 加载 pipeline 配置（work_dir 优先 → skill 默认 → 兜底默认链）
    _work_dir = os.environ.get("QUANT_WORK_DIR", "./workspace")
    work_dir = Path(_work_dir) if _work_dir else None

    try:
        processors = load_pipeline_config(work_dir=work_dir)
    except Exception as e:
        logger.warning(
            f"加载 pipeline.yaml 失败 ({e})，回退兜底默认链"
        )
        from scripts.processors.loader import _default_processors
        processors = _default_processors()

    # 构造 ProcessorChain
    chain = ProcessorChain(processors, fail_fast=False)

    # 初始化 Recorder
    _archive_dir = os.path.join(_work_dir, "archives", "factor_engine")
    try:
        recorder = ExperimentRecorder(
            archive_dir=Path(_archive_dir),
            pipeline_config=chain.describe_chain(),
            input_data_paths=[data_path] if data_path else None,
        )
    except Exception as e:
        logger.warning(f"ExperimentRecorder 初始化失败，降级为无记录模式: {e}")
        recorder = None

    # 构造 ProcessContext
    proc_ctx = ProcessContext(
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

    # 构造 corr_result 兼容旧返回结构
    corr_meta = proc_ctx.metadata.get("correlation_result", {})
    corr_result = {
        "selected_factors": selected_factors,
        "removed_factors": corr_meta.get("removed_factors", []),
        "correlation_matrix": {},
    }

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

    return final_df, ic_results, selected_factors, corr_result


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
                       if c not in ['code', 'date', 'industry', 'estimated_mv',
                                    'money_flow_raw', 'ret_1d', 'ret_5d', 'ret_20d', 'ret_60d',
                                    'turnover_5d']]

        # ── T1-6/T1-8: Processor Pipeline 新路径 vs 旧路径切换 ──
        # 环境变量 QUANT_LEGACY_PIPELINE=1 强制走旧 4 步硬编码路径（兼容回滚）
        # 默认走 ProcessorChain（新路径），通过 pipeline.yaml 声明式配置
        use_legacy = os.environ.get("QUANT_LEGACY_PIPELINE", "0") == "1"

        if use_legacy:
            logger.info("QUANT_LEGACY_PIPELINE=1，走旧 4 步硬编码路径")
            final_df, ic_results, selected_factors, corr_result = _run_legacy_factor_pipeline(
                engine=engine,
                factor_df=factor_df,
                forward_returns=forward_returns,
                factor_names=factor_names,
            )
        else:
            logger.info("走 ProcessorChain 新路径（默认）")
            final_df, ic_results, selected_factors, corr_result = _run_processor_chain(
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
                "removed_factors": corr_result['removed_factors'],
                "ic_results": ic_results,
                "correlation": {k: v for k, v in corr_result.items() if k != 'correlation_matrix'},
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
# 使用方式: from engine import optimizations
# ---------------------------------------------------------------------------
from scripts.optimizations.factor_dsl import (
    FactorEngine as _FactorEngine,
)
from scripts.optimizations.lookahead_detector import (
    detect_in_code as _detect_lookahead_in_code,
    detect_in_dataframe as _detect_lookahead_in_dataframe,
)
from scripts.optimizations.alpha158_lib import (
    AlphaEngine as _AlphaEngine,
    AlphaRegistry as _AlphaRegistry,
)
from scripts.optimizations.factor_validator import (
    validate_factor as _validate_factor,
    FactorVerdict as _FactorVerdict,
)
from scripts.optimizations.ic_vectorized import (
    ic_analysis_batch as _ic_analysis_batch,
    ic_summary as _ic_summary,
)
from scripts.optimizations.ic_decay import (
    ICDecayAnalyzer as _ICDecayAnalyzer,
)
from scripts.optimizations.factor_registry_v2 import (
    FactorRegistry as _FactorRegistryV2,
    Neutralizer as _NeutralizerV2,
)
from scripts.optimizations.vectorized_neutralize import (
    neutralize_factor as _neutralize_factor,
    neutralize_factors_batch as _neutralize_factors_batch,
)
from scripts.optimizations.vectorized_correlation import (
    correlation_analysis as _correlation_analysis,
)
from scripts.optimizations.ic_analysis_v2 import (
    calc_ic_series as _calc_ic_series,
    calc_ic_stats as _calc_ic_stats,
)

from scripts.optimizations.expression_dsl.evaluator import (
    Evaluator as _FactorDSLEvaluator,
)
from scripts.optimizations.expression_dsl.parser import (
    AstNode as _AstNode,
    FieldNode as _FieldNode,
)


class optimizations:
    """因子引擎优化模块集合"""
    FactorDSLEvaluator = _FactorDSLEvaluator
    AstNode = _AstNode
    FieldNode = _FieldNode
    FactorEngine = _FactorEngine
    detect_lookahead_in_code = _detect_lookahead_in_code
    detect_lookahead_in_dataframe = _detect_lookahead_in_dataframe
    AlphaEngine = _AlphaEngine
    AlphaRegistry = _AlphaRegistry
    validate_factor = _validate_factor
    FactorVerdict = _FactorVerdict
    ic_analysis_batch = _ic_analysis_batch
    ic_summary = _ic_summary
    ICDecayAnalyzer = _ICDecayAnalyzer
    FactorRegistryV2 = _FactorRegistryV2
    NeutralizerV2 = _NeutralizerV2
    neutralize_factor = _neutralize_factor
    neutralize_factors_batch = _neutralize_factors_batch
    correlation_analysis = _correlation_analysis
    calc_ic_series = _calc_ic_series
    calc_ic_stats = _calc_ic_stats

    @staticmethod
    def get_factor_dsl_engines():
        """返回所有可用的因子DSL引擎"""
        return {
            "factor_engine": _FactorEngine,
            "expression_dsl_evaluator": _FactorDSLEvaluator,
            "ast_node": _AstNode,
            "field_node": _FieldNode,
            "alpha_engine": _AlphaEngine,
        }

    @staticmethod
    def get_ic_analyzers():
        """返回所有可用的IC分析器"""
        return {
            "ic_analysis_batch": _ic_analysis_batch,
            "ic_decay_analyzer": _ICDecayAnalyzer,
            "calc_ic_series": _calc_ic_series,
            "calc_ic_stats": _calc_ic_stats,
        }


# ---------------------------------------------------------------------------
# T1-6: Processor Pipeline 模块入口（方向一）
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
    """因子引擎 Processor Pipeline 模块集合（方向一）

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
        from scripts.processors.loader import _default_processors
        return _ProcessorChain(_default_processors())
