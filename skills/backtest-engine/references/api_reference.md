# API 参考文档

本文档提供 backtest-engine 的完整 API 参考。

## 核心类

### BacktestEngine

统一回测引擎类。

#### 方法

##### `__init__()`

初始化回测引擎，自动加载配置的后端适配器。

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
- `t_plus_1` (bool): 是否启用T+1
- `price_limit` (bool): 是否启用涨跌停限制

**返回：**
```python
{
    "metrics": {...},
    "equity_curve": pd.DataFrame,
    "trades": [...],
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
        "backend": str,
        "equity_curve_path": str,
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

| 指标名 | 描述 |
|--------|------|
| total_return | 累计收益率 |
| annual_return | 年化收益率 |
| volatility | 年化波动率 |
| sharpe_ratio | 夏普比率 |
| max_drawdown | 最大回撤 |
| win_rate | 胜率 |
| calmar_ratio | Calmar比率 |

## CLI 使用

```bash
python engine.py context.json
```
