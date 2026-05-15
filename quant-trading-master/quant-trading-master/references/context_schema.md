# Context 对象 JSON Schema

## 顶层结构

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| task_id | string | 是 | 任务唯一ID，格式 YYYYMMDDHHMMSS |
| session_id | string | 否 | 会话ID |
| user_intent | string | 是 | 用户原始输入 |
| current_stage | string | 是 | 当前阶段: IDLE/DATA/FACTOR/MODEL/BACKTEST/PORTFOLIO/EXECUTION/REPORT |
| target_stages | string[] | 是 | 需要执行的目标阶段列表 |
| stock_pool | string[] | 否 | 股票代码列表，空列表表示全市场 |
| benchmark | string | 否 | 基准指数代码，默认 000300.SH |
| start_date | string | 否 | 开始日期 YYYY-MM-DD |
| end_date | string | 否 | 结束日期 YYYY-MM-DD |
| strategy_name | string | 否 | 策略名称 |
| strategy_params | object | 否 | 策略参数字典 |
| artifacts | object | 是 | 各阶段产物路径映射 |
| metadata | object | 否 | 元信息 |
| errors | string[] | 否 | 错误列表 |

## artifacts 映射示例

```json
{
  "DATA": "/workspace/data/cleaned_data.parquet",
  "FACTOR": "/workspace/factors/factor_data.parquet",
  "MODEL": "/workspace/models/model.pkl",
  "BACKTEST": "/workspace/backtest_results/backtest_result.json",
  "PORTFOLIO": "/workspace/portfolio/portfolio_weights.json",
  "EXECUTION": "/workspace/trade_log.json",
  "REPORT": "/workspace/reports/report.html"
}
