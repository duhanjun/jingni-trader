"""
快速开始示例
展示如何使用量化交易技能包
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant_skills import DataLoader, TechnicalAnalysis, BacktestEngine, RiskManager, PortfolioOptimizer


def example_data_loader():
    """数据获取示例"""
    print("=" * 60)
    print("数据获取模块示例")
    print("=" * 60)
    
    loader = DataLoader()
    print(f"可用数据源: {loader.get_available_sources()}")
    
    # 生成模拟数据 (避免依赖外部API)
    print("\n生成模拟OHLCV数据...")
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # 模拟价格
    close = 100 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))) * 100
    high = close + np.random.uniform(0, 2, len(dates))
    low = close - np.random.uniform(0, 2, len(dates))
    open_price = low + np.random.uniform(0, 1, len(dates))
    volume = np.random.randint(1000000, 10000000, len(dates))
    
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    print(f"数据生成完成，共{len(data)}条记录")
    print(f"数据预览:\n{data.head()}")
    
    return data


def example_technical_analysis(data):
    """技术分析示例"""
    print("\n" + "=" * 60)
    print("技术分析模块示例")
    print("=" * 60)
    
    ta = TechnicalAnalysis()
    
    # 计算SMA
    sma_20 = ta.sma(data, 20)
    sma_60 = ta.sma(data, 60)
    print(f"\nSMA(20) 最后5个值:\n{sma_20.tail()}")
    
    # 计算RSI
    rsi = ta.rsi(data, 14)
    print(f"\nRSI(14) 最后5个值:\n{rsi.tail()}")
    
    # 计算MACD
    macd_data = ta.macd(data)
    print(f"\nMACD 最后5个值:\n{pd.DataFrame(macd_data).tail()}")
    
    # 添加所有指标
    data_with_indicators = ta.add_all_indicators(data)
    print(f"\n添加指标后的数据列: {data_with_indicators.columns.tolist()}")
    
    return data_with_indicators


def example_backtest(data):
    """策略回测示例"""
    print("\n" + "=" * 60)
    print("策略回测模块示例")
    print("=" * 60)
    
    engine = BacktestEngine()
    
    # 均线交叉策略
    print("\n测试均线交叉策略...")
    sma_strategy = engine.SimpleMovingAverageCross({
        "fast_period": 20,
        "slow_period": 60
    })
    
    backtest_result = engine.backtest(
        data,
        sma_strategy,
        initial_capital=100000,
        commission=0.001,
        slippage=0.001
    )
    
    print(f"\n策略表现:")
    perf = backtest_result['performance']
    print(f"  总收益率: {perf['total_return']:.2%}")
    print(f"  年化收益率: {perf['annual_return']:.2%}")
    print(f"  年化波动率: {perf['annual_volatility']:.2%}")
    print(f"  夏普比率: {perf['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {perf['max_drawdown']:.2%}")
    print(f"  最终资产: {perf['final_value']:.2f}")
    
    print(f"\n交易记录数: {len(backtest_result['trades'])}")
    if backtest_result['trades']:
        print(f"首笔交易: {backtest_result['trades'][0]}")
        print(f"末笔交易: {backtest_result['trades'][-1]}")
    
    # RSI策略
    print("\n测试RSI策略...")
    rsi_strategy = engine.RSIStrategy({
        "period": 14,
        "oversold": 30,
        "overbought": 70
    })
    
    rsi_result = engine.backtest(
        data,
        rsi_strategy,
        initial_capital=100000
    )
    
    rsi_perf = rsi_result['performance']
    print(f"\nRSI策略表现:")
    print(f"  总收益率: {rsi_perf['total_return']:.2%}")
    print(f"  夏普比率: {rsi_perf['sharpe_ratio']:.2f}")
    
    return backtest_result


def example_risk_management(data):
    """风险管理示例"""
    print("\n" + "=" * 60)
    print("风险管理模块示例")
    print("=" * 60)
    
    risk_manager = RiskManager()
    
    # 计算收益率
    returns = data['close'].pct_change().dropna()
    print(f"\n收益率统计:")
    print(f"  均值: {returns.mean():.4%}")
    print(f"  标准差: {returns.std():.4%}")
    
    # 计算VaR
    var_95 = risk_manager.calculate_var(returns, 0.95, "historical")
    var_99 = risk_manager.calculate_var(returns, 0.99, "historical")
    print(f"\n风险价值 (VaR):")
    print(f"  95% 置信度: {var_95:.4%}")
    print(f"  99% 置信度: {var_99:.4%}")
    
    # 计算CVaR
    cvar_95 = risk_manager.calculate_cvar(returns, 0.95)
    print(f"\n条件风险价值 (CVaR):")
    print(f"  95% 置信度: {cvar_95:.4%}")
    
    # 计算最大回撤
    max_dd, dd_start, dd_end = risk_manager.calculate_max_drawdown(data['close'])
    print(f"\n最大回撤:")
    print(f"  回撤幅度: {max_dd:.2%}")
    print(f"  开始日期: {dd_start}")
    print(f"  结束日期: {dd_end}")
    
    # 计算风险指标
    summary = risk_manager.risk_summary(returns)
    print(f"\n风险指标摘要:")
    for key, value in summary.items():
        if isinstance(value, float):
            if 'return' in key or 'volatility' in key or 'var' in key or 'cvar' in key:
                print(f"  {key}: {value:.4%}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def example_portfolio_optimization():
    """组合优化示例"""
    print("\n" + "=" * 60)
    print("组合优化模块示例")
    print("=" * 60)
    
    optimizer = PortfolioOptimizer()
    
    # 生成模拟收益率数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_assets = 5
    assets = ['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D', 'Asset_E']
    
    # 生成相关性较强的收益率
    base_return = np.random.normal(0.0005, 0.01, len(dates))
    returns_data = {}
    for i, asset in enumerate(assets):
        asset_return = base_return + np.random.normal(0, 0.005, len(dates))
        returns_data[asset] = asset_return
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    print(f"\n模拟收益率数据:")
    print(f"  资产数量: {n_assets}")
    print(f"  数据长度: {len(returns_df)}")
    print(f"\n收益率统计:\n{returns_df.describe()}")
    
    # 最小方差组合
    print("\n最小方差组合优化...")
    min_var_weights = optimizer.calculate_min_variance_weights(returns_df)
    print("\n资产权重:")
    for asset, weight in min_var_weights.items():
        print(f"  {asset}: {weight:.4%}")
    
    # 等权重对比
    equal_weights = {asset: 1.0 / n_assets for asset in assets}
    
    # 比较表现
    min_var_perf = optimizer.portfolio_performance(returns_df, min_var_weights)
    equal_perf = optimizer.portfolio_performance(returns_df, equal_weights)
    
    print("\n组合表现对比:")
    print(f"{'指标':<20} {'最小方差':<20} {'等权重':<20}")
    print("-" * 60)
    print(f"{'总收益率':<20} {min_var_perf['total_return']:<20.4%} {equal_perf['total_return']:<20.4%}")
    print(f"{'年化收益率':<20} {min_var_perf['annual_return']:<20.4%} {equal_perf['annual_return']:<20.4%}")
    print(f"{'年化波动率':<20} {min_var_perf['annual_volatility']:<20.4%} {equal_perf['annual_volatility']:<20.4%}")
    print(f"{'夏普比率':<20} {min_var_perf['sharpe_ratio']:<20.4f} {equal_perf['sharpe_ratio']:<20.4f}")


def main():
    """主函数"""
    print("量化交易技能包 - 快速开始示例")
    print("=" * 60)
    
    # 1. 数据获取
    data = example_data_loader()
    
    # 2. 技术分析
    data_with_indicators = example_technical_analysis(data)
    
    # 3. 策略回测
    backtest_result = example_backtest(data)
    
    # 4. 风险管理
    example_risk_management(data)
    
    # 5. 组合优化
    example_portfolio_optimization()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
