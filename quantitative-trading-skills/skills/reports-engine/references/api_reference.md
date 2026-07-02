# API 参考

## ReportsEngine 主类

### `__init__(config=None)`

初始化报告引擎。

**参数:**
- `config`: 可选的配置对象

### `generate_full_report(...)`

生成完整报告。

**主要参数:**
- `portfolio_returns`: 策略收益率序列（pandas.Series）
- `benchmark_returns`: 基准收益率序列
- `factor_exposures`: 因子暴露 DataFrame
- `industry_weights`: 行业权重 DataFrame
- `is_returns`: 样本内收益率
- `oos_returns`: 样本外收益率
- `ic_series`: IC时间序列

**返回:** 报告字典

### `save_report(report, output_path)`

保存报告到JSON文件。
