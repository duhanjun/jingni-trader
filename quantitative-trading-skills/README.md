# 量化交易Skill套件 - A股研究级组合方案

基于专业量化交易机构架构设计，专为A股市场优化的完整量化交易系统。

---

## 🎯 系统架构（研究级组合方案）

```
┌─────────────────────────────────────────────────────────────────────┐
│                       量化交易系统架构                              │
├─────────────────────────────────────────────────────────────────────┤
│  实盘层                                                           │
│  ├── xtquant (迅投QMT)   ── 专业级交易执行                        │
│  └── gm (掘金量化)       ── 云端仿真/实盘交易                      │
├─────────────────────────────────────────────────────────────────────┤
│  回测层                                                           │
│  └── Backtrader          ── 事件驱动回测验证                      │
├─────────────────────────────────────────────────────────────────────┤
│  研究层                                                           │
│  └── Qlib (微软)         ── AI量化因子研究平台                    │
├─────────────────────────────────────────────────────────────────────┤
│  数据层                                                           │
│  ├── Tushare             ── A股核心数据                          │
│  └── AKShare             ── 补充数据（龙虎榜、资金流等）            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 项目结构

```
quantitative-trading-skills/
├── skills/                              # Skill集合（按架构分层）
│   ├── a-share-data-engine/            # 数据层：Tushare+AKShare
│   ├── qlib-research-engine/           # 研究层：Qlib因子研究
│   ├── backtest-engine/                # 回测层：Backtrader
│   ├── execution-monitor-engine/       # 实盘层：xtquant+gm
│   ├── portfolio-risk-engine/          # 组合优化与风控
│   ├── reports-engine/                 # 绩效报告与归因
│   └── quant-trading-master/           # 主Skill（协调中枢）
├── examples/                           # 示例脚本
├── scripts/                            # 自动化脚本
├── requirements.txt                    # Python依赖列表
├── QuickStart.md                       # 新手5分钟入门指南
├── Example_Workflow.md                 # 完整量化工作流程示例
└── README.md                           # 项目说明
```

---

## 🏗️ 核心Skill详解

### 1️⃣ 数据层 - a-share-data-engine

| 数据源 | 用途 | 特点 |
|-------|------|------|
| **Tushare** | A股日线/分钟线、财务数据、指数成分 | 机构级稳定数据源 |
| **AKShare** | 龙虎榜、大宗交易、资金流向 | 免费补充数据 |

```python
# 数据获取示例
from skills.a-share-data-engine.engine import AShareDataEngine

engine = AShareDataEngine()
df = engine.get_daily(
    codes=["000001.SZ", "600000.SH"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

### 2️⃣ 研究层 - qlib-research-engine

**Qlib（微软AI量化平台）核心功能**：

| 功能模块 | 说明 |
|---------|------|
| **因子库** | 内置360+因子（Alpha158/Alpha360） |
| **模型训练** | LightGBM、LSTM、Transformer支持 |
| **因子挖掘** | 自动化因子生成与验证 |
| **解释性分析** | SHAP值、特征重要性 |

```python
# Qlib研究示例
import qlib
from qlib.contrib.model.gbdt import LGBModel

qlib.init()

# 使用Alpha360因子
handler = qlib.contrib.data.handler.Alpha360(
    start_time="2020-01-01",
    end_time="2024-12-31"
)

# 训练多因子模型
model = LGBModel()
model.fit(handler)
```

### 3️⃣ 回测层 - backtest-engine

**Backtrader事件驱动回测**：

```python
# Backtrader回测示例
import backtrader as bt

class MyStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
    
    def next(self):
        if self.data.close[0] > self.sma[0]:
            self.buy()

cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)
cerebro.run()
```

### 4️⃣ 实盘层 - execution-monitor-engine

| 后端 | 说明 | 适用场景 |
|-----|------|---------|
| **xtquant** | 迅投QMT接口 | 专业级机构交易 |
| **gm** | 掘金量化 | 云端仿真/实盘 |

```python
# 实盘执行示例
from skills.execution-monitor-engine.engine import ExecutionEngine

engine = ExecutionEngine(backend="xtquant")
engine.connect()

# 下单
engine.place_order(
    symbol="000001.SZ",
    side="buy",
    order_type="market",
    quantity=100
)
```

---

## 🔄 完整工作流

```
1. 数据获取 ──▶ 2. 因子研究 ──▶ 3. 策略回测 ──▶ 4. 组合优化 ──▶ 5. 实盘执行
     ↓                ↓               ↓                ↓               ↓
  Tushare        Qlib因子         Backtrader      风险平价         xtquant/gm
  AKShare        模型训练          绩效分析        约束优化          风控监控
```

---

## 📊 阶段状态机

```
数据获取 → 因子构建 → 模型训练 → 回测验证 → 组合优化 → 模拟／实盘 → 绩效报告
              ↓              ↓
         分支逻辑        分支逻辑
         ↓              ↓
    回测失败→返回因子调优   模型过拟合→样本外再验证
```

---

## 🚀 快速开始

### 环境要求
- Python 3.9+
- 环境变量配置（参考 `.env.example`）

### 安装流程

```bash
# 1. 进入项目目录
cd quantitative-trading-skills

# 2. 安装核心依赖
pip install -r requirements.txt

# 3. 安装Qlib（因子研究）
pip install qlib

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 Tushare Token 等信息

# 5. 运行示例
python examples/01_quick_start.py
```

### 环境变量配置

| 变量名 | 说明 | 是否必需 |
|-------|------|---------|
| `TUSHARE_TOKEN` | Tushare Pro API Token | ✅ 必需 |
| `GM_TOKEN` | 掘金量化 API Token | ⚠️ 实盘时需要 |
| `XTQUANT_PATH` | 迅投QMT安装路径 | ⚠️ 实盘时需要 |
| `QLIB_DATA_DIR` | Qlib数据目录 | ⚠️ Qlib需要 |

---

## 🧪 核心技术栈

| 层级 | 技术 | 版本 |
|-----|------|------|
| **数据层** | Tushare | >=1.2.60 |
| | AKShare | >=1.11.0 |
| **研究层** | Qlib | >=0.2.0 |
| **回测层** | Backtrader | >=1.9.78 |
| **实盘层** | xtquant | >=1.0.0 |
| | gm | >=1.0.0 |
| **基础库** | pandas | >=2.0.0 |
| | numpy | >=1.24.0 |
| | scikit-learn | >=1.3.0 |

---

## 📚 学习资源

| 文档 | 说明 |
|-----|------|
| [QuickStart.md](QuickStart.md) | 5分钟快速入门 |
| [Example_Workflow.md](Example_Workflow.md) | 完整工作流示例 |
| [Qlib官方文档](https://qlib.readthedocs.io) | 因子研究平台 |
| [Backtrader文档](https://www.backtrader.com) | 回测框架 |

---

## ⚠️ 风险提示

1. 量化交易存在高风险，请谨慎使用
2. 先在模拟环境充分验证策略
3. 本项目仅供学习和研究使用
4. 请确保遵守当地相关法律法规

---

**项目位置**：`/workspace/quantitative-trading-skills/`

---

## 🎉 开始您的量化之旅

```python
# 完整流程示例
from skills.quant-trading-master.scripts.main_workflow import run, QuantTradingContext

# 创建上下文
ctx = QuantTradingContext(
    task_id="research_task_001",
    stock_pool=["000001.SZ", "600000.SH", "600519.SH"],
    time_range={"start": "2020-01-01", "end": "2024-12-31"},
    config={
        "data_backend": "tushare",
        "research_backend": "qlib",
        "backtest_backend": "backtrader",
        "execution_backend": "gm"  # 或 "xtquant"
    }
)

# 运行完整量化流程
result = run(ctx)

# 查看结果
print(result.summary())
```

---

*Powered by 专业量化交易机构架构 | 专为A股市场优化*
