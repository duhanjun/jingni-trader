# 券商对接说明

## 支持的券商/接口

### 1. 迅投量化 (xtquant)
- 提供完整的行情和交易接口
- 支持多账户管理
- 支持多种订单类型

#### 配置
```python
config = get_config(
    EXECUTION_BACKEND="xtquant",
    XTQUANT_TOKEN="your_token_here"
)
```

#### 安装依赖
```bash
pip install xtquant
```

### 2. 掘金量化 (gm)
- 专业的量化交易平台
- 提供丰富的历史数据
- 支持实盘和模拟交易

#### 配置
```python
config = get_config(
    EXECUTION_BACKEND="gm",
    GM_TOKEN="your_token_here"
)
```

#### 安装依赖
```bash
pip install gm-python
```

### 3. vn.py
- 开源量化交易框架
- 支持多种券商接口
- 支持CTP、飞马等

#### 配置
```python
config = get_config(
    EXECUTION_BACKEND="vnpy",
    VNPY_CONFIG={
        "gateway_name": "CTP",
        "setting": {...}
    }
)
```

#### 安装依赖
```bash
pip install vnpy
```

### 4. 模拟交易 (sim)
- 不需要真实券商账户
- 支持完整的交易模拟
- 适合策略回测和验证

#### 配置
```python
config = get_config(
    EXECUTION_BACKEND="sim",
    INITIAL_CAPITAL=1000000
)
```

## 切换后端

```python
# 切换到掘金量化
config = get_config(EXECUTION_BACKEND="gm")
engine = ExecutionEngine(config)

# 切换回模拟交易
engine.switch_paper_trade(True)
```

## 注意事项

1. 实盘交易前请充分测试策略
2. 建议先使用模拟交易验证
3. 注意控制风险，设置合理的止损
4. 保护好API Token，不要泄露
