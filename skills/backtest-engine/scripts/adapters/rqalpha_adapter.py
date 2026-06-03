"""
RQAlpha 回测引擎适配器
专为A股设计，内置 T+1、涨跌停、费用模型
"""
import os
import json
import shutil
import tempfile
import subprocess
from typing import Dict, Any
import pandas as pd
import numpy as np

from ..base.base_backtest_engine import BaseBacktestEngine


class RQAlphaAdapter(BaseBacktestEngine):
    """RQAlpha 适配器"""

    def __init__(self):
        self._check_rqalpha()

    def _check_rqalpha(self):
        try:
            import rqalpha
        except ImportError:
            raise ImportError("RQAlpha 未安装，请 pip install rqalpha")

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
        # RQAlpha 需要特定的数据格式和配置文件
        # 这里采用简化方式：通过临时文件传递，然后调用 rqalpha 命令行
        # 实际生产环境可通过 RQAlpha Python API 更精细控制

        # 构建策略文件内容
        strategy_code = self._generate_strategy_code(signals)

        # 创建临时目录
        tmp_dir = tempfile.mkdtemp()

        try:
            # 保存策略
            strategy_path = os.path.join(tmp_dir, "strategy.py")
            with open(strategy_path, "w", encoding="utf-8") as f:
                f.write(strategy_code)

            # 保存数据为 bundle（简化：使用内置数据，实际需配置）
            # 这里假设已配置好RQAlpha的本地bundle
            # 生成配置文件
            config = {
                "base": {
                    "start_date": str(data['date'].min().date()),
                    "end_date": str(data['date'].max().date()),
                    "accounts": {"stock": init_capital},
                    "benchmark": benchmark,
                },
                "extra": {
                    "log_level": "error",
                },
                "mod": {
                    "sys_risk": {
                        "enabled": True,
                        "price_limit": price_limit,
                        "tplus": t_plus_1,
                    },
                    "sys_simulation": {
                        "enabled": True,
                        "slippage": slippage,
                        "commission_multiplier": commission_rate / 0.0003,  # 相对默认费率
                        "tax_multiplier": stamp_tax_rate / 0.001,
                        "min_commission": 5,
                    },
                },
            }
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            # 运行回测（RQAlpha 命令行）
            result_dir = os.path.join(tmp_dir, "result")
            os.makedirs(result_dir, exist_ok=True)

            # 实际调用命令（需要 rqalpha 已安装且配置好数据包）
            # cmd = f"rqalpha run -f {strategy_path} -c {config_path} -d {result_dir}"
            # subprocess.run(cmd, shell=True, check=True)

            # 由于RQAlpha的数据依赖复杂，此处返回模拟结果作为框架示例
            return self._generate_mock_result(data, signals, init_capital)

        finally:
            # 清理临时文件
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _generate_strategy_code(self, signals: pd.DataFrame) -> str:
        """生成RQAlpha策略Python代码"""
        # 将信号序列化到策略中
        signals_json = signals.to_dict(orient='records')

        code = f'''
import rqalpha
from rqalpha.api import *
import pandas as pd
import json

# 预加载信号
SIGNALS = pd.DataFrame(json.loads('''{json.dumps(signals_json[:1000])}'''))  # 限制大小

def init(context):
    context.signal_pos = 0
    context.signals = SIGNALS.set_index(['code', 'date']) if not SIGNALS.empty else pd.DataFrame()

def handle_bar(context, bar_dict):
    today = context.now.strftime("%Y-%m-%d")
    # 根据信号调仓（简化示例）
    # 实际应遍历股票池，执行信号
    pass
'''
        return code

    def _generate_mock_result(self, data, signals, init_capital):
        """生成模拟回测结果（演示用，实际需替换为真实引擎输出）"""
        # 简单模拟：假设每个信号以当日收盘价成交，次日开盘价计算盈亏
        trades = []
        equity = [{"date": data['date'].min(), "equity": init_capital}]
        # 简化处理，仅作结构示例
        return {
            "trades": pd.DataFrame(trades, columns=['date', 'code', 'side', 'price', 'volume', 'commission']),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(equity),
            "metrics": self._calculate_metrics(pd.DataFrame(equity), init_capital),
            "report_path": "",
        }

    def _calculate_metrics(self, equity_curve, init_capital):
        """计算基础绩效指标"""
        if equity_curve.empty:
            return {}
        returns = equity_curve['equity'].pct_change().dropna()
        return {
            "annual_return": float(returns.mean() * 252),
            "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
            "max_drawdown": float((equity_curve['equity'] / equity_curve['equity'].cummax() - 1).min()),
            "total_return": float(equity_curve['equity'].iloc[-1] / init_capital - 1),
            "volatility": float(returns.std() * np.sqrt(252)),
        }
