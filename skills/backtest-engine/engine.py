"""
回测引擎主逻辑
统一接口，调度原生后端，计算绩效，生成报告
"""
import os
import sys
import json
import logging
from typing import Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from scripts.config import (
    BACKTEST_DIR, INIT_CAPITAL,
    COMMISSION_RATE, MIN_COMMISSION, STAMP_TAX_RATE,
    TRANSFER_FEE_RATE, SLIPPAGE, BENCHMARK, RISK_FREE_RATE
)

logger = logging.getLogger("backtest-engine")

try:
    import quantstats as qs
    HAS_QS = True
except ImportError:
    HAS_QS = False


class BacktestEngine:
    """统一回测引擎"""

    def __init__(self):
        from scripts.adapters.native_adapter import NativeAdapter
        self.adapter = NativeAdapter()

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = INIT_CAPITAL,
        benchmark: str = BENCHMARK,
        commission_rate: float = COMMISSION_RATE,
        stamp_tax_rate: float = STAMP_TAX_RATE,
        slippage: float = SLIPPAGE,
        t_plus_1: bool = True,
        price_limit: bool = True,
    ) -> Dict[str, Any]:
        """执行回测"""
        logger.info("开始回测，后端: native")
        return self.adapter.run_backtest(
            data=data,
            signals=signals,
            init_capital=init_capital,
            benchmark=benchmark,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            t_plus_1=t_plus_1,
            price_limit=price_limit,
            slippage=slippage,
            transfer_fee_rate=TRANSFER_FEE_RATE,
            min_commission=MIN_COMMISSION,
        )

    def generate_report(self, result: Dict[str, Any], output_dir: str = BACKTEST_DIR) -> str:
        """生成回测报告"""
        if not HAS_QS:
            logger.warning("quantstats 未安装，无法生成详细报告")
            return ""

        equity_curve = result.get('equity_curve')
        if equity_curve is None or equity_curve.empty:
            return ""

        returns = equity_curve.set_index('date')['equity'].pct_change().dropna()
        report_path = os.path.join(output_dir, f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

        qs.reports.html(returns, output=report_path, title="A股策略回测报告")
        logger.info(f"回测报告已生成: {report_path}")
        return report_path


def run(ctx) -> Dict[str, Any]:
    """
    backtest-engine 的 run 函数

    参数:
        ctx: Context 对象，需包含:
            - artifacts['DATA']: 清洗后数据路径
            - artifacts['MODEL'] 或 artifacts['FACTOR']

    返回:
        {
            "success": bool,
            "artifact_path": str,
            "report_path": str,
            "metadata": {...},
            "error": str
        }
    """
    try:
        data_path = ctx.get_artifact("DATA")
        if not data_path or not os.path.exists(data_path):
            return {"success": False, "artifact_path": "", "metadata": {}, "error": "数据产物不存在"}
        data = pd.read_parquet(data_path)

        signal_path = ctx.get_artifact("MODEL")
        if not signal_path or not os.path.exists(signal_path):
            factor_path = ctx.get_artifact("FACTOR")
            if factor_path and os.path.exists(factor_path):
                factor_df = pd.read_parquet(factor_path)
                if 'alpha_score' in factor_df.columns:
                    factor_df['rank'] = factor_df.groupby('date')['alpha_score'].rank(pct=True)
                    signals = factor_df[['code', 'date']].copy()
                    signals['signal'] = 0
                    signals.loc[factor_df['rank'] > 0.8, 'signal'] = 1
                else:
                    return {"success": False, "artifact_path": "", "metadata": {}, "error": "无有效信号"}
            else:
                return {"success": False, "artifact_path": "", "metadata": {}, "error": "无信号数据"}
        else:
            if signal_path.endswith('.parquet'):
                signals = pd.read_parquet(signal_path)
            else:
                import joblib
                model = joblib.load(signal_path)
                factor_path = ctx.get_artifact("FACTOR")
                if factor_path:
                    factor_df = pd.read_parquet(factor_path)
                    feature_cols = [c for c in factor_df.columns if c not in ['code', 'date', 'industry']]
                    feature_cols = [c for c in feature_cols if not factor_df[c].isna().all()]
                    if 'alpha_score' in feature_cols:
                        feature_cols = ['alpha_score'] + [c for c in feature_cols if c != 'alpha_score']
                    X = factor_df[feature_cols].fillna(0)
                    preds = model.predict(X)
                    signals = factor_df[['code', 'date']].copy()
                    signals['signal'] = 0
                    signals.loc[preds > np.percentile(preds, 80), 'signal'] = 1
                else:
                    return {"success": False, "artifact_path": "", "metadata": {}, "error": "无法生成信号"}

        if signals.empty:
            return {"success": False, "artifact_path": "", "metadata": {}, "error": "信号为空"}

        os.makedirs(BACKTEST_DIR, exist_ok=True)
        engine = BacktestEngine()
        result = engine.run(data=data, signals=signals)

        result_json = {
            "metrics": result['metrics'],
            "backend": "native",
            "timestamp": datetime.now().isoformat(),
        }
        json_path = os.path.join(BACKTEST_DIR, "backtest_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)

        report_path = engine.generate_report(result)

        equity_path = os.path.join(BACKTEST_DIR, "equity_curve.parquet")
        if 'equity_curve' in result and not result['equity_curve'].empty:
            result['equity_curve'].to_parquet(equity_path)

        # ============================================================
        # P0-3 RuleJudge 五硬门评审（PRD P0-3.5）
        # ============================================================
        # 计算完成交易笔数：每次 signal 从 0→1 或 1→0 算一笔
        trade_count = 0
        try:
            if not signals.empty and "signal" in signals.columns:
                sig_sorted = signals.sort_values(["code", "date"])
                sig_diff = sig_sorted.groupby("code")["signal"].diff().abs()
                trade_count = int((sig_diff > 0).sum())
        except Exception as tc_e:
            logger.warning(f"trade_count 计算异常（默认 0）: {tc_e}")

        verdict_dict = {}
        try:
            from scripts.rule_judge import RuleJudge
            judge = RuleJudge()
            equity_curve_for_judge = result.get("equity_curve", pd.DataFrame())
            verdict = judge.judge(
                metrics=result["metrics"],
                equity_curve=equity_curve_for_judge,
                trade_count=trade_count,
            )
            verdict_dict = verdict.to_dict()
            if verdict.recommended_state == "rejected":
                logger.warning(
                    f"P0-3 策略未通过 RuleJudge 评审: failed_gates={verdict.failed_gates}"
                )
            else:
                logger.info(
                    f"P0-3 策略通过 RuleJudge 评审: passed_gates={verdict.passed_gates}"
                )
        except Exception as rj_e:
            logger.warning(f"P0-3 RuleJudge 评审异常（不阻断流程）: {rj_e}")

        return {
            "success": True,
            "artifact_path": json_path,
            "report_path": report_path,
            "metadata": {
                "metrics": result['metrics'],
                "backend": "native",
                "equity_curve_path": equity_path,
                # P0-3.5 新增：评审结果写入 result["verdict"]
                "verdict": verdict_dict,
                "trade_count": trade_count,
            },
            "error": ""
        }

    except Exception as e:
        logger.exception("回测引擎执行失败")
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
            task_id="test_bt",
            stock_pool=[],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        ctx.update_artifact("DATA", "./workspace/data/cleaned_data.parquet")
        ctx.update_artifact("FACTOR", "./workspace/factors/factor_data.parquet")

    result = run(ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
