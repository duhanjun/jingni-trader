"""
因子表达式引擎
借鉴: quant-stream FactorExpressionEngine
在 per-code 分组下批量计算因子表达式
"""
import logging
import pandas as pd
from typing import Optional

from .operators import OperatorRegistry
from .parser import FactorExpressionParser

logger = logging.getLogger("factor-expression-engine")

# 预设的常用因子表达式（借鉴 Qlib Alpha158 + quant-stream）
PRESET_EXPRESSIONS = {
    # 动量因子
    "momentum_5d": "RANK(DELTA($close, 5))",
    "momentum_20d": "RANK(DELTA($close, 20))",
    "momentum_60d": "RANK(DELTA($close, 60))",
    # 反转因子
    "reversal_5d": "RANK(DELTA($close, 5))",  # 取负号由下游处理
    "reversal_20d": "RANK(DELTA($close, 20))",
    # 波动率因子
    "volatility_20d": "ZSCORE(TS_STD(DELTA($close, 1), 20))",
    # 成交量因子
    "volume_ratio": "RANK($volume)",
    "volume_trend": "ZSCORE(TS_MEAN($volume, 20))",
    # 换手率因子
    "turnover_rank": "RANK($turnover)",
    # 技术指标
    "rsi_14": "RSI($close, 14)",
    # 日内因子
    "intraday_range": "RANK(DELTA($close, 1))",
}


class FactorExpressionEngine:
    """因子表达式引擎：声明式定义 + per-code 批量计算"""

    def __init__(self):
        self.registry = OperatorRegistry()
        self.parser = FactorExpressionParser(self.registry)

    def compute(
        self,
        data: pd.DataFrame,
        expression: str,
        code_col: str = "code",
        date_col: str = "date",
        name: Optional[str] = None,
    ) -> pd.DataFrame:
        if name is None:
            cleaned = expression
            for var in self.parser.VARIABLE_MAP:
                cleaned = cleaned.replace(var, var.replace("$", ""))
            name = cleaned.replace("(", "_").replace(")", "").replace(", ", "_").replace(" ", "_")
            while "__" in name:
                name = name.replace("__", "_")
            name = name.strip("_")

        data = data.sort_values([code_col, date_col]).copy()
        results = []
        for code, group in data.groupby(code_col):
            group = group.copy()
            group[name] = self.parser.parse_and_compute(expression, group, date_col, code_col)
            results.append(group[[code_col, date_col, name]])

        return pd.concat(results, ignore_index=True)

    def compute_preset(
        self,
        data: pd.DataFrame,
        preset_names: Optional[list] = None,
    ) -> pd.DataFrame:
        """批量计算预设因子表达式"""
        if preset_names is None:
            preset_names = list(PRESET_EXPRESSIONS.keys())

        result = data[["code", "date"]].copy()
        for name in preset_names:
            if name not in PRESET_EXPRESSIONS:
                continue
            expr = PRESET_EXPRESSIONS[name]
            logger.info(f"计算表达式因子: {name} = {expr}")
            factor_result = self.compute(data, expr, name=name)
            result = result.merge(factor_result, on=["code", "date"], how="left")
        return result

    def custom_register(self, name: str, fn, category: str = "custom"):
        """注册自定义算子"""
        self.registry.register(name, fn, category)

    def list_presets(self) -> dict:
        return dict(PRESET_EXPRESSIONS)