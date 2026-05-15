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
│   ├── a-share-factor-engine/          # 因子层：因子构建与IC分析
│   ├── strategy-model-engine/          # 策略层：策略模型与超参优化
│   ├── backtest-engine/                # 回测层：Backtrader
│   ├── portfolio-risk-engine/          # 风控层：组合优化与风控
│   ├── execution-monitor-engine/       # 实盘层：xtquant+gm+模拟
│   ├── reports-engine/                 # 报告层：绩效报告与归因
│   └── quant-trading-master/           # 主Skill（协调中枢）
├── examples/                           # 示例脚本
│   ├── 01_quick_start.py              # 快速入门
│   ├── 02_full_workflow.py            # 完整工作流
│   ├── backtest_demo.py               # Backtrader回测演示
│   └── demo_full_workflow.py          # 全流程演示
├── scripts/                            # 工具脚本
│   ├── prepare_qlib_data.py           # Qlib数据准备
│   ├── check_dependencies.py          # 依赖检查
│   ├── install.sh                     # 安装脚本
│   └── setup_env.sh                   # 环境配置
├── docs/                               # 文档
│   └── LIVE_TRADING_GUIDE.md          # 实盘对接指南
├── qlib_data/                          # Qlib数据目录
├── requirements.txt                    # Python依赖列表
├── .env.example                        # 环境变量模板
├── env_config.py                       # 环境配置管理
└── README.md                           # 项目说明
```

---

## 🏗️ 核心Skill详解

### 1️⃣ 数据层 - a-share-data-engine

| 数据源 | 用途 | 特点 |
|-------|------|------|
| **Tushare** | A股日线/分钟线、财务数据、指数成分 | 机构级稳定数据源 |
| **AKShare** | 龙虎榜、大宗交易、资金流向 | 免费补充数据 |
| **BaoStock** | 历史行情 | 免费备选 |

```python
import sys; sys.path.insert(0, 'skills/a-share-data-engine')
from engine import AShareDataEngine
from config import get_config

config = get_config()
engine = AShareDataEngine(config)
df = engine.get_daily(
    codes=["000001.SZ", "600000.SH", "600519.SH"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

### 2️⃣ 研究层 - qlib-research-engine

**Qlib（微软AI量化平台）核心功能**：

| 功能模块 | 状态 | 说明 |
|---------|------|------|
| Alpha158因子库 | ✅ | 经典158因子 |
| Alpha360因子库 | ✅ | 扩展360因子 |
| LightGBM模型 | ✅ | 梯度提升树训练 |
| SHAP解释性分析 | ✅ | 因子重要性分析 |
| 风险评估 | ✅ | IC/IR分析 |

```python
import qlib
from qlib.data import D
from qlib.contrib.model.gbdt import LGBModel

qlib.init(provider_uri='qlib_data', region='cn')

# 使用Alpha158因子
from qlib.contrib.data.handler import Alpha158
handler = Alpha158(start_time="2020-01-01", end_time="2024-12-31")

model = LGBModel()
model.fit(handler)
```

### 3️⃣ 回测层 - backtest-engine

**Backtrader事件驱动回测 + RSI/MACD多信号策略**：

买入信号（满足任一）：
- RSI < 30（超卖）且 MACD金叉
- MA5 > MA20（金叉趋势）
- RSI < 40 且 MA金叉

卖出信号（满足任一）：
- RSI > 70（超买）
- MA5 < MA20（死叉）
- 止损：亏损超过8%
- 止盈：盈利超过20%

```python
# 完整回测示例
python examples/backtest_demo.py
```

**历史回测结果（2023-2024，5只A股）：**

| 股票 | 收益率 | 交易次数 |
|------|--------|---------|
| 600000.SH | +18.28% | 14 |
| 600036.SH | +2.83% | 13 |
| 600519.SH | +0.00% | 0 |
| 000001.SZ | -7.67% | 17 |
| 000002.SZ | -15.59% | 15 |
| **组合** | **-0.43%** | **59** |

### 4️⃣ 实盘层 - execution-monitor-engine

| 后端 | 说明 | 适用场景 |
|-----|------|---------|
| **xtquant** | 迅投QMT接口 | 专业级机构交易 |
| **gm** | 掘金量化 | 云端仿真/实盘 |
| **sim** | 模拟交易 | 本地测试（默认） |

```python
import sys; sys.path.insert(0, 'skills/execution-monitor-engine')
from engine import ExecutionEngine
from config import get_config

config = get_config()
engine = ExecutionEngine(config)
engine.initialize_account(initial_capital=1000000)

order = engine.place_order(
    symbol="000001.SZ", side="buy",
    order_type="market", quantity=100
)
```

### 5️⃣ 组合优化与风控 - portfolio-risk-engine

| 功能 | 说明 |
|------|------|
| Barra归因 | 风格/行业归因分析 |
| 组合优化 | 风险平价、均值方差 |
| 止损管理 | 移动止损、时间止损 |
| 硬断路器 | 单日亏损>2%拒新开仓 |

### 6️⃣ 报告引擎 - reports-engine

| 功能 | 说明 |
|------|------|
| Quantstats | 完整绩效分析报告 |
| 权益曲线 | 多周期图表 |
| 回撤分析 | 回撤序列图 |
| Brinson归因 | 超额收益分解 |

---

## 🔄 完整工作流

```
数据获取 → 因子研究 → 因子构建 → 策略回测 → 组合优化 → 模拟交易 → 绩效报告
    ↓          ↓          ↓           ↓          ↓          ↓          ↓
 Tushare    Qlib      IC分析     Backtrader  风险平价     sim/gm   Quantstats
 AKShare   Alpha360   因子组合    RSI+MACD   约束优化    xtquant   Brinson
```

### 工作流阶段状态机

```
数据获取 → 因子研究 → 因子构建 → 模型训练 → 回测验证 → 组合优化 → 模拟／实盘 → 绩效报告
              ↓              ↓
         分支逻辑        分支逻辑
         ↓              ↓
    回测失败→返回因子调优   模型过拟合→样本外再验证
```

---

## 🚀 快速开始

### 环境要求
- Python 3.9-3.11
- Tushare Pro Token（[注册获取](https://tushare.pro)）

### 安装流程

```bash
# 1. 进入项目目录
cd quantitative-trading-skills

# 2. 创建虚拟环境（推荐）
python3.11 -m venv .venv
source .venv/bin/activate

# 3. 安装核心依赖
pip install -r requirements.txt

# 4. 安装Qlib（因子研究）
pip install git+https://github.com/microsoft/qlib.git

# 5. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 TUSHARE_TOKEN

# 6. 准备Qlib数据
python scripts/prepare_qlib_data.py

# 7. 运行回测演示
python examples/backtest_demo.py

# 8. 运行完整工作流演示
python examples/demo_full_workflow.py
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
| **研究层** | Qlib | >=0.9.8 |
| **回测层** | Backtrader | >=1.9.78 |
| **实盘层** | xtquant | >=1.0.0 |
| | gm | >=1.0.0 |
| **基础库** | pandas | >=2.0.0 |
| | numpy | >=1.24.0 |
| | scikit-learn | >=1.3.0 |
| | LightGBM | >=4.0.0 |

---

## 📚 文档索引

| 文档 | 说明 |
|-----|------|
| [README.md](README.md) | 项目总览（本文档） |
| [QuickStart.md](QuickStart.md) | 5分钟快速入门 |
| [Example_Workflow.md](Example_Workflow.md) | 完整工作流示例 |
| [docs/LIVE_TRADING_GUIDE.md](docs/LIVE_TRADING_GUIDE.md) | 实盘对接指南 |
| [Qlib官方文档](https://qlib.readthedocs.io) | 因子研究平台 |
| [Backtrader文档](https://www.backtrader.com) | 回测框架 |

---

## 📊 开发状态

| 模块 | 状态 | 验证 |
|------|------|------|
| a-share-data-engine | ✅ 完成 | Tushare数据获取正常 |
| qlib-research-engine | ✅ 完成 | Qlib核心模块全部通过 |
| backtest-engine | ✅ 完成 | Backtrader回测成功（5股/59笔交易） |
| execution-monitor-engine | ✅ 完成 | 模拟交易正常 |
| portfolio-risk-engine | ✅ 完成 | 风控模块就绪 |
| reports-engine | ✅ 完成 | 报告引擎就绪 |
| quant-trading-master | ✅ 完成 | 8阶段工作流就绪 |
| Qlib数据 | ✅ 完成 | 30只股票/5年数据(430MB) |

---

## ⚠️ 风险提示

1. 量化交易存在高风险，请谨慎使用
2. 先在模拟环境充分验证策略
3. 本项目仅供学习和研究使用
4. 请确保遵守当地相关法律法规

---

**项目位置**：`/workspace/quantitative-trading-skills/`