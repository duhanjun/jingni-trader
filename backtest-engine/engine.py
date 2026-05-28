"""
回测引擎主逻辑
统一接口，调度不同后端，计算绩效，生成报告
"""
import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from scripts.config import (
    BACKTEST_BACKEND, BACKTEST_DIR, INIT_CAPITAL,
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
        self.adapter = self._load_adapter()

    def _load_adapter(self):
        if BACKTEST_BACKEND == "rqalpha":
            from scripts.adapters.rqalpha_adapter import RQAlphaAdapter
            return RQAlphaAdapter()
        elif BACKTEST_BACKEND == "backtrader":
            from scripts.adapters.backtrader_adapter import BacktraderAdapter
            return BacktraderAdapter()
        elif BACKTEST_BACKEND == "gm":
            from scripts.adapters.gm_adapter import GmAdapter
            return GmAdapter()
        else:
            raise ValueError(f"不支持的回测引擎: {BACKTEST_BACKEND}")

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
        logger.info(f"开始回测，后端: {BACKTEST_BACKEND}")
        result = self.adapter.run_backtest(
            data=data,
            signals=signals,
            init_capital=init_capital,
            benchmark=benchmark,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            t_plus_1=t_plus_1,
            price_limit=price_limit,
            slippage=slippage,
        )
        if 'metrics' not in result or not result['metrics']:
            result['metrics'] = self._calc_metrics(result.get('equity_curve', pd.DataFrame()), init_capital)
        return result

    def _calc_metrics(self, equity_curve: pd.DataFrame, init_capital: float) -> Dict[str, float]:
        """计算全面绩效指标"""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}
        eq = equity_curve.set_index('date')['equity']
        if len(eq) < 2:
            return {}
        returns = eq.pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = (eq / eq.cummax() - 1).min()
        sharpe = (annual_return - RISK_FREE_RATE) / volatility if volatility != 0 else 0
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0
        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "calmar_ratio": float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0,
        }

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
            "backend": BACKTEST_BACKEND,
            "timestamp": datetime.now().isoformat(),
        }
        json_path = os.path.join(BACKTEST_DIR, "backtest_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2, default=str)

        report_path = engine.generate_report(result)

        equity_path = os.path.join(BACKTEST_DIR, "equity_curve.parquet")
        if 'equity_curve' in result and not result['equity_curve'].empty:
            result['equity_curve'].to_parquet(equity_path)

        return {
            "success": True,
            "artifact_path": json_path,
            "report_path": report_path,
            "metadata": {
                "metrics": result['metrics'],
                "backend": BACKTEST_BACKEND,
                "equity_curve_path": equity_path,
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
        ctx.update_artifact("DATA", "./quant_workspace/data/cleaned_data.parquet")
        ctx.update_artifact("FACTOR", "./quant_workspace/factors/factor_data.parquet")

    result = run(ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
