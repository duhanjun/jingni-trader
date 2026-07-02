# 完整量化工作流程示例

本示例展示如何使用本Skill套件完成一个完整的A股量化投研流程。

## 目录

1. [流程概述
2. [阶段1: 数据获取
3. [阶段2: 因子构建
4. [阶段3: 策略建模
5. [阶段4: 回测验证
6. [阶段5: 组合优化
7. [阶段6: 模拟交易
8. [阶段7: 绩效报告

---

## 流程概述

```
[数据获取] → [因子构建] → [模型训练] → [回测验证] → [组合优化] → [模拟／实盘] → [绩效报告]
```

我们将构建一个基于多因子选股的策略，回测验证后进行组合优化，最后模拟交易并生成报告。

---

## 准备工作

```python
import sys
import os
import pandas as pd
import numpy as np

# 添加所有Skill路径
skill_dirs = [
    'a-share-data-engine',
    'a-share-factor-engine',
    'backtest-engine',
    'portfolio-risk-engine',
    'strategy-model-engine',
    'execution-monitor-engine',
    'reports-engine'
]

for skill_dir in skill_dirs:
    sys.path.append(os.path.join(os.getcwd(), f'skills/{skill_dir}'))

print("✅ 环境准备完成")
```

---

## 阶段1: 数据获取

使用 a-share-data-engine 获取A股历史数据。

```python
from a_share_data_engine import AShareDataEngine, get_config

# 配置使用免费的BaoStock
config = get_config(DATA_BACKEND="baostock")
data_engine = AShareDataEngine(config)

# 定义股票池
stock_pool = [
    "000001.SZ",  # 平安银行
    "000002.SZ",  # 万科A
    "600000.SH",  # 浦发银行
    "600036.SH",  # 招商银行
    "600519.SH",  # 贵州茅台
    "000858.SZ",  # 五粮液
    "601318.SH",  # 中国平安
    "002475.SZ",  # 立讯精密
]

# 获取2020-2023年的日线数据
print("📊 正在获取数据...")
price_data = data_engine.get_daily(
    codes=stock_pool,
    start_date="2020-01-01",
    end_date="2023-12-31",
)

print(f"✅ 获取完成，共 {len(price_data)} 条数据")
print(price_data.head())
```

---

## 阶段2: 因子构建

使用 a-share-factor-engine 构建多因子。

```python
from a_share_factor_engine import AShareFactorEngine
from a_share_factor_engine.config import get_config

factor_config = get_config()
factor_engine = AShareFactorEngine(factor_config)

# 计算因子
print("🔧 正在计算因子...")
factor_list = [
    "momentum_1m",  # 1月动量
    "momentum_3m",  # 3月动量
    "volatility_1m",  # 1月波动率
    "turnover",  # 换手率
    "lncap",  # 对数市值
]

factors = factor_engine.compute_factors(price_data, factor_list=factor_list)

# 计算远期收益（用于IC分析）
forward_returns = factor_engine.compute_forward_returns(price_data, horizon=20)

# IC分析
print("📈 正在进行IC分析...")
ic_report = factor_engine.analyze_ic(factors, forward_returns)
print("IC均值:")
print(ic_report['ic_mean'])

# 相关性分析
corr_report = factor_engine.analyze_correlation(factors, ic_report['ic_mean'])

# 因子融合
combined_alpha = factor_engine.combine_factors(
    factors,
    method='ic_weighted',
    ic_mean=ic_report['ic_mean']
)

print("✅ 因子构建完成")
```

---

## 阶段3: 策略建模

使用 strategy-model-engine 构建选股模型。

```python
from strategy_model_engine import StrategyModelEngine
from strategy_model_engine.config import get_config

model_config = get_config()
model_engine = StrategyModelEngine(model_config)

# 训练截面选股模型
print("🤖 正在训练选股模型...")
model = model_engine.train_stock_selection_model(
    factors,
    forward_returns,
    model_type='lightgbm'
)

# 生成策略模板
print("📝 正在生成策略模板...")
strategy_template = model_engine.generate_strategy_template('multi_factor')

print("✅ 策略建模完成")
```

---

## 阶段4: 回测验证

使用 backtest-engine 进行策略回测。

```python
from backtest_engine import BacktestEngine
from backtest_engine.config import get_config

bt_config = get_config()
bt_engine = BacktestEngine(bt_config)

# 定义策略函数
def multi_factor_strategy(context, data):
    """基于复合因子的选股策略"""
    current_date = context.current_date
    
    # 获取当前复合因子值
    if current_date in combined_alpha.index:
        today_alpha = combined_alpha.loc[current_date]
        
        # 选择因子值最高的3只股票
        selected = today_alpha.nlargest(3).index.tolist()
        
        # 等权分配
        weight = 1.0 / len(selected)
        
        # 调仓
        for stock in selected:
            context.order_target_percent(stock, weight)

# 运行回测
print("🔙 正在运行回测...")
backtest_results = bt_engine.run(
    strategy=multi_factor_strategy,
    start_date="2021-01-01",
    end_date="2023-12-31",
    initial_capital=1000000,
    price_data=price_data
)

# 查看绩效指标
metrics = backtest_results['performance_metrics']
print("🎯 回测绩效:")
print(f"总收益率: {metrics['total_return']:.2%}")
print(f"年化收益率: {metrics['annual_return']:.2%}")
print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
print(f"最大回撤: {metrics['max_drawdown']:.2%}")
print(f"胜率: {metrics['win_rate']:.2%}")

# 生成回测报告
print("📄 正在生成回测报告...")
report = bt_engine.generate_report(backtest_results)

print("✅ 回测验证完成")
```

---

## 阶段5: 组合优化

使用 portfolio-risk-engine 优化投资组合。

```python
from portfolio_risk_engine import PortfolioRiskEngine
from portfolio_risk_engine.config import get_config

risk_config = get_config()
risk_engine = PortfolioRiskEngine(risk_config)

# 计算收益率矩阵
returns = price_data.pivot(columns='code', values='close').pct_change().dropna()

# 优化组合
print("⚖️  正在优化投资组合...")
optimization_result = risk_engine.optimize_portfolio(
    returns,
    method='max_sharpe',  # 最大夏普比率
    constraints={
        'single_stock_max': 0.3,  # 单股票最大权重30%
        'industry_max': 0.5,  # 单行业最大权重50%
    }
)

print("最优权重:")
print(optimization_result['weights'])
print("\n组合表现:")
print(optimization_result['performance'])

# 计算VaR
var_result = risk_engine.calculate_var(returns, optimization_result['weights'])
print(f"\nVaR (95%): {var_result['var']:.4f}")
print(f"CVaR (95%): {var_result['cvar']:.4f}")

print("✅ 组合优化完成")
```

---

## 阶段6: 模拟交易

使用 execution-monitor-engine 进行模拟交易。

```python
from execution_monitor_engine import ExecutionEngine
from execution_monitor_engine.config import get_config

exec_config = get_config(EXECUTION_BACKEND="sim", PAPER_TRADE="true")
exec_engine = ExecutionEngine(exec_config)

# 初始化账户
print("💼 正在初始化模拟账户...")
exec_engine.initialize_account(initial_capital=1000000)

# 按照优化权重下单
optimal_weights = optimization_result['weights']
print("📤 正在执行模拟交易...")

for stock, weight in optimal_weights.items():
    if weight > 0:
        order = exec_engine.place_order(
            symbol=stock,
            side="buy",
            order_type="market",
            quantity=int(1000000 * weight / price_data.loc[price_data['code'] == stock, 'close'].iloc[-1] / 100) * 100
        )
        print(f"已下单 {stock}: {order.quantity} 股")

# 获取持仓
positions = exec_engine.get_positions()
print("\n📊 当前持仓:")
for pos in positions:
    print(f"{pos.symbol}: {pos.quantity} 股，市值: {pos.market_value:.2f}")

# 获取账户状态
account = exec_engine.get_account()
print(f"\n💰 账户权益: {account['equity']:.2f}")

print("✅ 模拟交易完成")
```

---

## 阶段7: 绩效报告

使用 reports-engine 生成完整的绩效报告。

```python
from reports_engine import ReportsEngine

reports_engine = ReportsEngine()

# 准备数据
portfolio_returns = backtest_results['equity_curve'].pct_change().dropna()
benchmark_returns = returns.mean(axis=1)  # 等权基准

# 生成完整报告
print("📊 正在生成绩效报告...")
full_report = reports_engine.generate_full_report(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    backtest_results=backtest_results,
    optimization_result=optimization_result
)

# 保存报告
report_dir = "reports"
os.makedirs(report_dir, exist_ok=True)

reports_engine.save_report(full_report, report_dir)
print(f"✅ 报告已保存到 {report_dir}/")

print("\n" + "="*50)
print("🎉 完整量化工作流程执行完毕！")
print("="*50)
```

---

## 使用主引擎自动化整个流程

如果不想手动执行每个阶段，可以使用 quant-trading-master：

```python
from quant_trading_master.scripts.main_workflow import run, QuantTradingContext

# 创建上下文
ctx = QuantTradingContext(
    task_id="workflow_example_001",
    session_id="demo_session",
    stock_pool=stock_pool,
    time_range={"start_date": "2020-01-01", "end_date": "2023-12-31"},
    config={
        "data_backend": "baostock",
        "backtest_backend": "backtrader",
        "execution_backend": "sim"
    }
)

# 运行完整流程
result = run(ctx)

print(f"完成阶段: {result['completed_stages']}")
```

---

## 总结

本示例演示了：
1. ✅ 多源数据获取
2. ✅ 多因子构建与分析
3. ✅ 机器学习策略建模
4. ✅ 真实规则回测
5. ✅ 投资组合优化
6. ✅ 模拟交易执行
7. ✅ 完整绩效报告

每个阶段都可以独立使用，也可以通过主引擎串联自动化执行！
