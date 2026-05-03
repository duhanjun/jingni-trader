# QuantSkills - 专业量化交易系统

基于Python开源库构建的专业量化交易系统，包含数据获取、技术分析、策略回测、风险管理、组合优化等核心模块。

## 功能特性

### 1. 数据获取模块 (DataLoader)
- 支持多种数据源: Yahoo Finance, AKShare, Tushare
- 统一的数据获取接口
- 批量数据下载
- 收益率计算

### 2. 技术分析模块 (TechnicalAnalysis)
- 移动平均线: SMA, EMA
- 动量指标: RSI, MACD
- 趋势指标: 布林带, KDJ
- 波动率指标: ATR
- 自动添加所有常用指标

### 3. 策略回测模块 (BacktestEngine)
- 内置经典策略: 均线交叉, RSI策略
- 支持自定义策略
- 考虑手续费和滑点
- 完整的回测分析: 夏普比率, 最大回撤等
- 兼容Backtrader框架

### 4. 风险管理模块 (RiskManager)
- 风险价值 (VaR) 计算: 历史法, 方差-协方差法
- 条件风险价值 (CVaR)
- 最大回撤分析
- 夏普比率, 索提诺比率
- Beta计算
- 完整的风险指标摘要

### 5. 组合优化模块 (PortfolioOptimizer)
- 最小方差组合
- 最大夏普比率组合
- 风险平价 (Equal Risk Contribution)
- 兼容PyPortfolioOpt库
- 组合表现分析

## 安装依赖

```bash
pip install -r requirements.txt
```

可选依赖库:
- `yfinance`: Yahoo Finance数据
- `akshare`: A股数据
- `tushare`: Tushare数据
- `pandas-ta`: 技术分析
- `TA-Lib`: 技术分析库
- `backtrader`: 回测框架
- `PyPortfolioOpt`: 组合优化
- `Riskfolio-Lib`: 高级组合优化

## 快速开始

运行示例代码:

```bash
python examples/quick_start.py
```

### 基本使用

#### 1. 数据获取

```python
from quant_skills import DataLoader

loader = DataLoader()

# 获取Yahoo Finance数据
data = loader.get_yahoo_data("AAPL", start_date="2023-01-01", end_date="2024-01-01")

# 获取A股数据 (AKShare)
data = loader.get_akshare_stock("sh600000")

# 批量获取
symbols = ["AAPL", "MSFT", "GOOGL"]
data_dict = loader.get_multi_symbols(symbols, source="yahoo")
```

#### 2. 技术分析

```python
from quant_skills import TechnicalAnalysis

ta = TechnicalAnalysis()

# 计算单个指标
sma_20 = ta.sma(data, 20)
rsi_14 = ta.rsi(data, 14)
macd_data = ta.macd(data)

# 一次性添加所有指标
data_with_indicators = ta.add_all_indicators(data)
```

#### 3. 策略回测

```python
from quant_skills import BacktestEngine

engine = BacktestEngine()

# 使用均线交叉策略
sma_strategy = engine.SimpleMovingAverageCross({
    "fast_period": 20,
    "slow_period": 60
})

# 执行回测
result = engine.backtest(
    data,
    sma_strategy,
    initial_capital=100000,
    commission=0.001,
    slippage=0.001
)

# 查看结果
print("策略表现:", result['performance'])
print("交易记录:", result['trades'])
```

#### 4. 风险管理

```python
from quant_skills import RiskManager

risk_manager = RiskManager()

# 计算收益率
returns = data['close'].pct_change().dropna()

# 计算VaR
var_95 = risk_manager.calculate_var(returns, 0.95)

# 计算最大回撤
max_dd, dd_start, dd_end = risk_manager.calculate_max_drawdown(data['close'])

# 风险指标摘要
summary = risk_manager.risk_summary(returns)
```

#### 5. 组合优化

```python
from quant_skills import PortfolioOptimizer

optimizer = PortfolioOptimizer()

# 优化组合
min_var_weights = optimizer.optimize_portfolio(returns_df, method="min_variance")
max_sharpe_weights = optimizer.optimize_portfolio(returns_df, method="max_sharpe")
equal_risk_weights = optimizer.optimize_portfolio(returns_df, method="equal_risk")

# 分析组合表现
performance = optimizer.portfolio_performance(returns_df, min_var_weights)
```

## 项目结构

```
quant_skills/
├── __init__.py              # 包初始化
├── data_loader.py           # 数据获取模块
├── technical_analysis.py    # 技术分析模块
├── backtest_engine.py       # 策略回测模块
└── risk_manager.py          # 风险管理与组合优化模块

examples/
├── __init__.py
└── quick_start.py           # 快速开始示例

requirements.txt             # 依赖文件
README.md                   # 项目文档
```

## 模块设计原则

1. **兼容性**: 核心功能不依赖特定外部库，同时提供对流行库的集成支持
2. **易用性**: 提供简洁的API接口，降低使用门槛
3. **可扩展性**: 模块化设计，便于添加新功能
4. **完整性**: 覆盖量化交易的主要工作流程

## 注意事项

- 本项目仅供学习和研究使用
- 回测结果不构成投资建议
- 实盘交易需谨慎
- 部分数据源可能需要注册或付费

## License

MIT License
