# 快速入门: 5分钟上手量化交易

欢迎来到量化交易Skill套件！本指南将帮助您在5分钟内快速上手。

## 目录
1. [环境准备 (1分钟)
2. [安装依赖 (1分钟)
3. [你的第一个策略 (2分钟)
4. [下一步 (1分钟)

---

## 1. 环境准备 (1分钟)

### 系统要求
- Python 3.9 或更高版本
- 操作系统：Linux/macOS/Windows (WSL2推荐)

### 检查Python版本
```bash
python --version
```

---

## 2. 安装依赖 (1分钟)

### 克隆项目（如需要）
```bash
cd quantitative-trading-skills
```

### 安装核心依赖
```bash
pip install -r requirements.txt
```

### 检查依赖
```bash
python scripts/check_dependencies.py
```

---

## 3. 你的第一个策略 (2分钟)

让我们用BaoStock（免费数据源）来获取数据并做一个简单的回测。

### 步骤1: 导入数据引擎
```python
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'skills/a-share-data-engine'))
from a_share_data_engine import AShareDataEngine, get_config

# 配置使用免费的BaoStock数据源
config = get_config(DATA_BACKEND="baostock")
engine = AShareDataEngine(config)

print("✅ 数据引擎初始化成功！")
```

### 步骤2: 获取数据
```python
# 获取平安银行(000001.SZ)的数据
df = engine.get_daily(
    codes=["000001.SZ"],
    start_date="2024-01-01",
    end_date="2024-12-31",
)

print(f"✅ 获取到 {len(df)} 条数据")
print(df.head())
```

### 步骤3: 简单策略回测
让我们实现一个简单的5日均线策略。
```python
import pandas as pd
import numpy as np

# 计算5日均线
df['ma5'] = df['close'].rolling(window=5).mean()

# 生成交易信号
df['signal'] = 0
df.loc[df['close'] > df['ma5'], 'signal'] = 1  # 买入信号
df.loc[df['close'] < df['ma5'], 'signal'] = -1  # 卖出信号

# 计算策略收益
df['return'] = df['close'].pct_change()
df['strategy_return'] = df['signal'].shift(1) * df['return']

# 计算累计收益
df['cum_return'] = (1 + df['return']).cumprod()
df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod()

print("✅ 策略回测完成")
print(f"持有收益: {(df['cum_return'].iloc[-1] - 1:.2%}")
print(f"策略收益: {(df['cum_strategy_return'].iloc[-1] - 1:.2%}")
```

---

## 4. 下一步 (1分钟)

### 深入学习
- 阅读 [Example_Workflow.md](Example_Workflow.md) - 完整的量化工作流程
- 查看各Skill的README.md - 详细功能介绍
- 运行examples/目录下的示例脚本 - 实际代码示例

### Skill套件包含
- **数据获取** - a-share-data-engine
- **因子构建** - a-share-factor-engine
- **策略建模** - strategy-model-engine
- **回测验证** - backtest-engine
- **组合优化** - portfolio-risk-engine
- **模拟交易** - execution-monitor-engine
- **报告生成** - reports-engine

### 完整工作流
使用主引擎运行完整工作流
```python
# 准备好后可以尝试
# from skills.quant-trading-master.scripts.main_workflow import run, QuantTradingContext
```

---

## 常见问题

### Q: 需要付费数据源？
A: 推荐先用免费的BaoStock或AkShare开始，之后可配置Tushare等付费数据源。

### Q: 如何运行测试？
A: 每个Skill都有tests目录，运行test_engine.py即可。

### Q: 需要OpenClaw？
A: 本Skill套件可以独立使用，也可以配合OpenClaw平台。

---

**恭喜！你已经完成了快速入门！🎉 现在去探索更强大的功能吧！
