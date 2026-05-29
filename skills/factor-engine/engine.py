"""
A股因子引擎主逻辑
负责因子计算、行业中性化、IC分析、相关性去冗余、多因子融合
"""
import os
import sys
import logging
import json
from typing import List, Dict, Any, Optional, Tuple

for key in list(sys.modules.keys()):
    if key.startswith('scripts.') or key == 'scripts':
        del sys.modules[key]
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

logger = logging.getLogger("a-share-factor-engine")


class FactorEngine:
    """A股因子引擎"""

    def __init__(self):
        self.calculator = self._load_calculator()

    def _load_calculator(self):
        """根据配置加载因子计算器"""
        if FACTOR_BACKEND == "talib":
            from scripts.adapters.talib_calculator import TalibCalculator
            return TalibCalculator()
        elif FACTOR_BACKEND == "pandas_ta":
            from scripts.adapters.pandas_ta_calculator import PandasTaCalculator
            return PandasTaCalculator()
        else:
            raise ValueError(f"不支持的因子后端: {FACTOR_BACKEND}")

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

        if 'amount' in df.columns:
            result['estimated_mv'] = df['amount'] / df['turnover_rate'].replace(0, np.nan) * 100
            result['lncap'] = np.log(result['estimated_mv'].replace(0, np.nan))
        else:
            result['lncap'] = np.nan

        if 'turnover_rate' in df.columns:
            result['turnover_20d'] = df.groupby('code')['turnover_rate'].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            result['turnover_5d'] = df.groupby('code')['turnover_rate'].transform(
                lambda x: x.rolling(5, min_periods=3).mean()
            )
            result['turnover_change'] = result['turnover_5d'] / result['turnover_20d'].replace(0, np.nan) - 1

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

    def neutralize(
        self,
        factor_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        neutralize_mcap: bool = NEUTRALIZE_MARKET_CAP,
        neutralize_industry: bool = NEUTRALIZE_INDUSTRY
    ) -> pd.DataFrame:
        """因子行业中性化处理"""
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
        """计算因子的IC序列和统计量"""
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
        max_correlation: float = MAX_CORRELATION
    ) -> Dict[str, Any]:
        """因子相关性分析"""
        if factor_df.empty:
            return {"correlation_matrix": pd.DataFrame(), "selected_factors": [], "removed_factors": []}

        if factor_names is None:
            factor_names = [c for c in factor_df.columns
                           if c not in ['code', 'date', 'industry']]

        factor_means = factor_df.groupby('date')[factor_names].mean()
        corr_matrix = factor_means.corr()

        to_remove = set()
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                fi, fj = factor_names[i], factor_names[j]
                if fi in to_remove or fj in to_remove:
                    continue
                if abs(corr_matrix.loc[fi, fj]) > max_correlation:
                    if len(fj) < len(fi):
                        to_remove.add(fi)
                    else:
                        to_remove.add(fj)

        selected = [f for f in factor_names if f not in to_remove]

        logger.info(f"因子相关性分析：原始 {len(factor_names)} 个，剔除 {len(to_remove)} 个，保留 {len(selected)} 个")

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "selected_factors": selected,
            "removed_factors": list(to_remove)
        }

    def factor_fusion(
        self,
        factor_df: pd.DataFrame,
        ic_results: Dict[str, Any],
        selected_factors: List[str],
        fusion_method: str = "ic_weighted"
    ) -> pd.DataFrame:
        """多因子融合为复合Alpha信号"""
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
            )

        rank_cols = [f"{f}_rank" for f in selected_factors if f"{f}_rank" in normalized.columns]
        normalized['alpha_score'] = 0
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

        df = pd.read_parquet(data_path)
        if df.empty:
            return {"success": False, "artifact_path": "", "metadata": {}, "error": "数据为空"}

        engine = FactorEngine()

        factor_df = engine.compute_a_share_factors(df)

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
        ic_results = engine.ic_analysis(factor_df, forward_returns, factor_names)

        corr_result = engine.correlation_analysis(factor_df, factor_names)
        selected_factors = corr_result['selected_factors']

        fusion_df = engine.factor_fusion(factor_df, ic_results, selected_factors)

        os.makedirs(FACTOR_DIR, exist_ok=True)
        output_path = os.path.join(FACTOR_DIR, "factor_data.parquet")
        final_df = factor_df.merge(fusion_df[['code', 'date', 'alpha_score']], on=['code', 'date'], how='left')
        final_df.to_parquet(output_path, index=False)

        ic_report_path = os.path.join(FACTOR_DIR, "ic_report.json")
        with open(ic_report_path, 'w', encoding='utf-8') as f:
            json.dump(ic_results, f, ensure_ascii=False, indent=2, default=str)

        return {
            "success": True,
            "artifact_path": output_path,
            "metadata": {
                "factor_names": factor_names,
                "selected_factors": selected_factors,
                "removed_factors": corr_result['removed_factors'],
                "ic_results": ic_results,
                "correlation": {k: v for k, v in corr_result.items() if k != 'correlation_matrix'},
                "fusion_method": "ic_weighted"
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
