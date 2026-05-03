# 交易执行监控引擎 (Execution Monitor Engine)

A股交易执行与监控引擎，支持多后端切换、模拟账户、硬断路器、审计日志和仓位管理。

## 功能特性

- **多后端支持**: 迅投(xtquant)、掘金(gm)、vn.py、模拟(sim)
- **模拟账户**: 资金初始化、状态跟踪、实时盈亏计算
- **订单执行**: 市价单、限价单、止损单模拟撮合
- **硬断路器**: 单日亏损>2%拒新开仓、单笔限额、订单频率限制、持仓集中度限制
- **确认模式与演习模式**: 仿真模式（真实行情+模拟撮合）、一键切换paper_trade
- **审计日志完整性**: 所有订单包括被风控拒绝都记录完整上下文
- **仓位同步与成本计算**: 实时仓位管理
- **交易日志记录**: 完整的CSV格式交易记录

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础使用

```python
from engine import ExecutionEngine
from config import get_config

# 创建配置
config = get_config(EXECUTION_BACKEND="sim", PAPER_TRADE="true")

# 创建引擎
engine = ExecutionEngine(config)

# 初始化账户
engine.initialize_account(initial_capital=1000000)

# 下单
order = engine.place_order(
    symbol="000001.SZ",
    side="buy",
    order_type="market",
    quantity=100
)

# 获取持仓
positions = engine.get_positions()

# 获取账户状态
account = engine.get_account()
print(f"账户权益: {account['equity']:.2f}")
```

## 项目结构

```
execution-monitor-engine/
├── SKILL.md              # Skill 描述文件
├── README.md             # 项目说明
├── requirements.txt      # 依赖列表
├── __init__.py           # 包入口
├── engine.py             # 主引擎
├── config.py             # 配置管理
├── audit_logger.py       # 审计日志
├── circuit_breaker.py    # 断路器
├── position_manager.py   # 仓位管理
├── trade_logger.py       # 交易日志
├── base/                 # 基础类
│   ├── __init__.py
│   └── base_trader.py
├── adapters/             # 执行适配器
│   ├── __init__.py
│   ├── gm_adapter.py
│   ├── sim_adapter.py
│   ├── vnpy_adapter.py
│   └── xtquant_adapter.py
├── references/           # 文档
│   ├── broker_integration.md
│   ├── config_guide.md
│   ├── order_types.md
│   └── sim_trading_rules.md
├── tests/                # 测试
│   ├── __init__.py
│   └── test_engine.py
└── trade_logs/           # 交易日志目录
```

## 后端选择

| 后端 | 说明 | 模拟交易 | 实盘交易 |
|------|------|---------|---------|
| sim | 模拟后端（默认） | ✅ | ❌ |
| xtquant | 迅投量化 | ✅ | ✅ |
| gm | 掘金量化 | ✅ | ✅ |
| vnpy | vn.py框架 | ✅ | ✅ |

## 硬断路器规则

- **单日亏损限制**: 单日亏损 > 2% 拒绝新开仓
- **单笔限额**: 单次下单金额不超过账户权益的 10%
- **订单频率**: 每分钟最多 10 笔订单
- **持仓集中度**: 单股票持仓不超过账户权益的 30%

详见 [references/sim_trading_rules.md](references/sim_trading_rules.md)

## 文档

- [券商集成指南](references/broker_integration.md)
- [配置指南](references/config_guide.md)
- [订单类型说明](references/order_types.md)
- [模拟交易规则](references/sim_trading_rules.md)

## 测试

运行测试:

```bash
cd tests
python test_engine.py
```

## 许可证

MIT License
