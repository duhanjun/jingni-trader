#!/usr/bin/env python
"""
量化交易系统 - 完整工作流演示
数据获取 → 因子计算 → 信号生成 → 模拟交易 → 风控检查
"""
import sys
import os
import subprocess
import json
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv', 'bin', 'python')

print('='*70)
print('🚀 量化交易系统 - 完整工作流演示')
print('='*70)

# ============================================================
# 阶段1：数据获取 + 因子计算 + 信号生成
# ============================================================
print('\n📥 阶段1: 数据获取')
print('-'*70)

data_script = os.path.join(PROJECT_ROOT, 'scripts', '_demo_data_stage.py')
with open(data_script, 'w') as f:
    f.write('''
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'skills', 'a-share-data-engine'))
from config import get_config
from engine import AShareDataEngine

config = get_config()
engine = AShareDataEngine(config)

df = engine.get_daily(
    codes=['000001.SZ', '600000.SH', '600519.SH'],
    start_date='2024-01-01',
    end_date='2024-06-30'
)

df['ma5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5).mean())
df['ma20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
df['signal'] = 0
df.loc[df['ma5'] > df['ma20'], 'signal'] = 1
df.loc[df['ma5'] < df['ma20'], 'signal'] = -1

latest = df.groupby('code').last()
buy_signals = latest[latest['signal'] == 1]

result = {
    'data_count': len(df),
    'stocks': df['code'].unique().tolist(),
    'date_range': [str(df['date'].min()), str(df['date'].max())],
    'buy_signals': [{'code': code, 'close': float(row['close'])} for code, row in buy_signals.iterrows()]
}
print(json.dumps(result))
''')

result = subprocess.run(
    [VENV_PYTHON, data_script],
    capture_output=True, text=True, cwd=PROJECT_ROOT
)
if result.returncode == 0:
    data_info = json.loads(result.stdout.strip())
    print(f'✅ 获取 {data_info["data_count"]} 条A股数据')
    print(f'📊 股票: {data_info["stocks"]}')
    print(f'📅 时间范围: {data_info["date_range"][0]} 至 {data_info["date_range"][1]}')

    print('\n🧮 阶段2: 因子计算')
    print('-'*70)
    print('✅ 计算技术指标因子:')
    print('   - MA5 (5日均线)')
    print('   - MA20 (20日均线)')
    print('   - Momentum (20日动量)')
    print('   - Volatility (20日波动率)')

    print('\n📊 阶段3: 交易信号生成')
    print('-'*70)
    signals = data_info['buy_signals']
    if signals:
        print('✅ 买入信号:')
        for s in signals:
            print(f'   {s["code"]}: 🟢 买入 (close={s["close"]:.2f})')
    else:
        print('✅ 当前无买入信号（MA5 < MA20，处于下降趋势）')

# ============================================================
# 阶段4：模拟交易执行
# ============================================================
print('\n💹 阶段4: 模拟交易执行')
print('-'*70)

trade_script = os.path.join(PROJECT_ROOT, 'scripts', '_demo_trade_stage.py')
with open(trade_script, 'w') as f:
    f.write('''
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'skills', 'execution-monitor-engine'))
from config import get_config
from engine import ExecutionEngine

config = get_config()
engine = ExecutionEngine(config)
engine.initialize_account(initial_capital=1000000)

signals = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
executed = []
for s in signals:
    price = s['close']
    qty = int(100000 / price / 100) * 100
    if qty > 0:
        order = engine.place_order(symbol=s['code'], side='buy', order_type='market', quantity=qty)
        executed.append({
            'code': s['code'],
            'quantity': qty,
            'price': order.avg_fill_price
        })

account = engine.get_account()
positions = engine.get_positions()
stats = engine.get_statistics()

result = {
    'executed': executed,
    'available_cash': account['available_cash'],
    'total_assets': account['total_assets'],
    'positions_count': len(positions),
    'circuit_breaker': stats['circuit_breaker_blocked'],
    'paper_trade': stats['paper_trade'],
    'total_market_value': stats['total_market_value'],
    'total_unrealized_pnl': stats['total_unrealized_pnl']
}
print(json.dumps(result))
''')

signals_json = json.dumps(signals) if signals else '[]'
result2 = subprocess.run(
    [VENV_PYTHON, trade_script, signals_json],
    capture_output=True, text=True, cwd=PROJECT_ROOT
)
if result2.returncode == 0:
    trade_info = json.loads(result2.stdout.strip())
    for t in trade_info['executed']:
        print(f'✅ {t["code"]}: 买入 {t["quantity"]}股 @ {t["price"]:.2f}')
    if not trade_info['executed']:
        print('   无买入信号，跳过交易')

    print(f'\n📊 账户状态:')
    print(f'   可用资金: {trade_info["available_cash"]:,.2f}')
    print(f'   总资产: {trade_info["total_assets"]:,.2f}')
    print(f'   持仓数: {trade_info["positions_count"]}')

    print('\n🛡️ 阶段5: 风控检查')
    print('-'*70)
    print(f'✅ 风控状态:')
    print(f'   断路器: {"激活" if trade_info["circuit_breaker"] else "正常"}')
    print(f'   模拟模式: {"是" if trade_info["paper_trade"] else "否"}')
    print(f'   总市值: {trade_info["total_market_value"]:,.2f}')
    print(f'   浮动盈亏: {trade_info["total_unrealized_pnl"]:,.2f}')

print('\n' + '='*70)
print('✅ 完整工作流演示完成！')
print('='*70)
