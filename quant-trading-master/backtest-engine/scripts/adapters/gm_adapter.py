"""
掘金量化(gm)回测引擎适配器
云端或本地回测，支持A股规则
"""
from typing import Dict, Any
import pandas as pd
import numpy as np

try:
    import gm.api as gm
    HAS_GM = True
except ImportError:
    HAS_GM = False

from ..base.base_backtest_engine import BaseBacktestEngine


class GmAdapter(BaseBacktestEngine):
    """掘金适配器"""

    def __init__(self):
        if not HAS_GM:
            raise ImportError("gm 未安装，请 pip install gm")
        # 掘金需要 token，从环境变量获取
        token = os.environ.get("GM_TOKEN", "")
        if token:
            gm.login(token)

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1e6,
        benchmark: str = "000300.SH",
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        slippage: float = 0.001,
    ) -> Dict[str, Any]:
        # 掘金回测需要通过其回测引擎执行，不能直接调用
        # 这里仅返回框架示意，实际需要编写策略文件并调用 gm.backtest()
        return self._generate_mock_result()

    def _generate_mock_result(self):
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
