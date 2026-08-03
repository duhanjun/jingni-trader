# API 参考文档

本文档提供 backtest-engine 的完整 API 参考。

## 核心类

### BacktestEngine

统一回测引擎类。

#### 方法

##### `__init__()`

初始化回测引擎，加载原生 NativeAdapter。

```python
engine = BacktestEngine()
```

##### `run(data, signals, ...) -> Dict`

执行回测。

**参数：**
- `data` (pd.DataFrame): 行情数据
- `signals` (pd.DataFrame): 交易信号
- `init_capital` (float): 初始资金
- `commission_rate` (float): 佣金费率
- `stamp_tax_rate` (float): 印花税率
- `slippage` (float): 滑点比例（双侧应用）
- `t_plus_1` (bool): 是否启用T+1
- `price_limit` (bool): 是否启用涨跌停限制

**返回：**
```python
{
    "metrics": {...},
    "equity_curve": pd.DataFrame,
    "gross_equity_curve": pd.DataFrame,
    "trades": [...],
    "positions": pd.DataFrame,
}
```

##### `generate_report(result, output_dir) -> str`

生成HTML回测报告。

## 标准入口函数

### `run(ctx) -> Dict[str, Any]`

Skill 标准入口函数。

**参数：**
- `ctx` (Context): 上下文对象

**返回：**
```python
{
    "success": bool,
    "artifact_path": str,      # 回测结果JSON路径
    "report_path": str,        # HTML报告路径
    "metadata": {
        "metrics": {...},
        "backend": "native",
        "equity_curve_path": str,
        "verdict": {...},      # RuleJudge 评审结果
        "trade_count": int,
    },
    "error": str
}
```

**示例：**
```python
from engine import run

result = run(ctx)
if result['success']:
    print(f"回测结果: {result['artifact_path']}")
    print(f"绩效指标: {result['metadata']['metrics']}")
```

## 绩效指标

### 基础指标（BaseBacktestMetrics）

| 指标名 | 描述 |
|--------|------|
| total_return | 累计收益率 |
| annual_return | 年化收益率 |
| volatility | 年化波动率 |
| sharpe_ratio | 夏普比率 |
| max_drawdown | 最大回撤 |
| win_rate | 胜率（基于真实 PnL） |
| calmar_ratio | Calmar比率 |
| sortino_ratio | Sortino比率 |
| total_trades | 交易笔数 |

### 扩展指标（成本分离 + 基准对比）

| 指标名 | 描述 |
|--------|------|
| gross_total_return | 毛收益（不含费用） |
| gross_annual_return | 毛年化收益 |
| total_cost_drag | 成本拖累（毛收益-净收益） |
| benchmark_return | 基准收益 |
| benchmark_volatility | 基准波动率 |
| benchmark_max_drawdown | 基准最大回撤 |
| alpha | CAPM Alpha |
| beta | CAPM Beta |
| excess_return | 超额收益 |

## CLI 使用

```bash
python engine.py context.json
```
