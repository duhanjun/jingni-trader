# API 参考文档

本文档提供 reports-engine 的完整 API 参考。

## 核心类

### ReportGenerator

报告生成器类。

#### 方法

##### `__init__(title)`

初始化报告生成器。

```python
generator = ReportGenerator(title="我的策略报告")
```

##### `calc_performance_metrics(equity_curve, risk_free_rate) -> Dict`

计算绩效指标。

**参数：**
- `equity_curve` (pd.DataFrame): 净值曲线数据

**返回：**
```python
{
    "total_return": float,
    "annual_return": float,
    "volatility": float,
    "sharpe_ratio": float,
    "max_drawdown": float,
    "calmar_ratio": float,
    "win_rate": float,
    ...
}
```

##### `make_equity_chart(equity_curve, benchmark_data) -> str`

生成净值曲线图。

##### `make_monthly_heatmap(equity_curve) -> str`

生成月度收益热力图。

##### `make_style_exposure_chart(exposures) -> str`

生成风格暴露图。

##### `make_industry_attribution_chart(contributions) -> str`

生成行业归因图。

##### `calc_brinson_attribution(...) -> Dict`

计算 Brinson 归因。

##### `build_html_report() -> str`

构建完整 HTML 报告。

## 标准入口函数

### `run(ctx) -> Dict[str, Any]`

Skill 标准入口函数。

**参数：**
- `ctx` (Context): 上下文对象

**返回：**
```python
{
    "success": bool,
    "artifact_path": str,      # HTML报告路径
    "metadata": {
        "metrics": {...},
        "report_data_path": str,
        "num_charts": int,
    },
    "error": str
}
```

**示例：**

```python
from engine import run

result = run(ctx)
if result['success']:
    print(f"报告已生成: {result['artifact_path']}")
    print(f"绩效指标: {result['metadata']['metrics']}")
```

## CLI 使用

```bash
python engine.py context.json
```
