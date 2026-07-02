"""
测试数据生成器

生成与 A 股日线数据统计特征近似的模拟数据，用于验证测试。
不依赖外部数据源，保证测试可重复。
"""
import numpy as np
import pandas as pd
import polars as pl


def generate_panel_data(
    n_stocks: int = 200,
    n_days: int = 250,
    start_date: str = "2024-01-01",
    seed: int = 42,
    n_factors: int = 5,
    with_industry: bool = True,
) -> pl.DataFrame:
    """
    生成面板数据（股票 × 日期）

    返回 Polars DataFrame，包含:
        date, code, close, volume, turnover_rate, amount,
        lncap, industry, factor_1..factor_n, ret_forward_5d
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    industries = ["银行", "地产", "医药", "消费", "科技", "能源", "材料", "工业"]

    rows = []
    for code in codes:
        start_price = rng.uniform(8, 50)
        daily_drift = rng.uniform(-0.0005, 0.0015)
        daily_vol = rng.uniform(0.012, 0.025)
        rets = rng.normal(daily_drift, daily_vol, n_days)
        prices = start_price * np.cumprod(1 + rets)
        mcap = rng.uniform(5e9, 5e11)
        lncap = np.log(mcap)
        industry = rng.choice(industries)
        base_turnover = rng.uniform(0.005, 0.05)

        for i in range(n_days):
            close = prices[i]
            volume = int(rng.lognormal(14, 0.5))
            turnover = base_turnover * (1 + rng.normal(0, 0.3))
            turnover = max(turnover, 0.001)
            amount = close * volume

            row = {
                "date": dates[i],
                "code": code,
                "close": round(float(close), 4),
                "volume": volume,
                "turnover_rate": round(float(turnover), 6),
                "amount": round(float(amount), 2),
                "lncap": round(float(lncap), 6),
            }
            if with_industry:
                row["industry"] = industry

            # 生成因子（与未来收益有一定相关性，使 IC 非零）
            for f in range(1, n_factors + 1):
                factor_val = rng.normal(0, 1) + 0.05 * lncap + 0.1 * turnover
                row[f"factor_{f}"] = round(float(factor_val), 6)

            rows.append(row)

    df_pl = pl.DataFrame(rows)

    # 计算未来 5 日收益率
    df_pl = df_pl.sort(["code", "date"])
    df_pl = df_pl.with_columns(
        (pl.col("close").shift(-5).over("code") / pl.col("close") - 1).alias("ret_forward_5d")
    )

    return df_pl


def generate_equity_curve(
    n_days: int = 250,
    init_capital: float = 1e6,
    annual_return: float = 0.15,
    annual_vol: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成净值曲线（用于绩效指标测试）

    返回 DataFrame: date, equity, cash, market_value, position_count
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2024-01-01", periods=n_days)
    daily_mu = annual_return / 252
    daily_sigma = annual_vol / np.sqrt(252)
    rets = rng.normal(daily_mu, daily_sigma, n_days)
    equity = init_capital * np.cumprod(1 + rets)

    df = pd.DataFrame({
        "date": dates,
        "equity": equity,
        "cash": init_capital * 0.3,
        "market_value": equity - init_capital * 0.3,
        "position_count": 10,
    })
    return df


def generate_trades(n_trades: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成交易记录"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2024-01-01", periods=250)
    actions = rng.choice(["buy", "sell"], n_trades)
    return pd.DataFrame({
        "date": rng.choice(dates, n_trades),
        "code": [f"{rng.integers(1, 200):06d}.SZ" for _ in range(n_trades)],
        "action": actions,
        "price": rng.uniform(8, 50, n_trades),
        "shares": rng.integers(100, 10000, n_trades) * 100,
        "amount": rng.uniform(1e4, 5e5, n_trades),
        "commission": rng.uniform(5, 100, n_trades),
        "tax": rng.uniform(0, 50, n_trades),
    })
