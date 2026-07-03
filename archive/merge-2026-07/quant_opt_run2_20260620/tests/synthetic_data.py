"""
测试用合成数据生成器

生成确定性的多股票 OHLCV 数据，便于复现测试结果。
"""
import numpy as np
import pandas as pd


def make_synthetic_data(
    n_codes: int = 10,
    n_days: int = 250,
    start_date: str = "2023-01-03",
    seed: int = 42,
    base_price: float = 10.0,
    add_limit_flags: bool = True,
) -> pd.DataFrame:
    """
    生成合成 A 股 OHLCV 数据

    参数:
        n_codes: 股票数量
        n_days: 交易日数
        seed: 随机种子（保证可复现）
        base_price: 起始价格
        add_limit_flags: 是否添加涨跌停标记

    返回:
        DataFrame: code, date, open, high, low, close, volume, amount,
                   is_limit_up, is_limit_down
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_codes + 1)]

    rows = []
    for code in codes:
        # 几何布朗运动 + 不同漂移
        mu = rng.uniform(-0.0005, 0.001)
        sigma = rng.uniform(0.015, 0.03)
        log_ret = rng.normal(mu, sigma, n_days)
        price = base_price * np.exp(np.cumsum(log_ret))
        for i, dt in enumerate(dates):
            close = round(float(price[i]), 2)
            open_ = round(close * (1 + rng.normal(0, 0.005)), 2)
            high = round(max(open_, close) * (1 + abs(rng.normal(0, 0.004))), 2)
            low = round(min(open_, close) * (1 - abs(rng.normal(0, 0.004))), 2)
            vol = int(rng.integers(1_000_000, 10_000_000))
            amount = round(close * vol, 2)
            row = {
                "code": code, "date": dt,
                "open": open_, "high": high, "low": low, "close": close,
                "volume": vol, "amount": amount,
            }
            if add_limit_flags:
                row["is_limit_up"] = False
                row["is_limit_down"] = False
            rows.append(row)
    df = pd.DataFrame(rows)
    # 给个别股票打上涨跌停标记便于测试（使用相对位置避免越界）
    if add_limit_flags and len(df) > 10:
        df.loc[df.index[len(df) // 3], "is_limit_up"] = True
        df.loc[df.index[2 * len(df) // 3], "is_limit_down"] = True
    return df


def make_signals_from_factor(
    factor_df: pd.DataFrame,
    factor_col: str,
    top_pct: float = 0.2,
    date_col: str = "date",
    code_col: str = "code",
) -> pd.DataFrame:
    """
    根据因子值生成交易信号：每日取 top_pct 的股票发出买入信号

    返回:
        DataFrame: code, date, signal (1/-1/0)
    """
    tmp = factor_df[[code_col, date_col, factor_col]].copy()
    tmp["rank"] = tmp.groupby(date_col)[factor_col].rank(pct=True)
    tmp["signal"] = 0
    tmp.loc[tmp["rank"] >= (1 - top_pct), "signal"] = 1
    # 持有 5 天后卖出
    tmp = tmp.sort_values([code_col, date_col]).reset_index(drop=True)
    tmp["hold"] = tmp.groupby(code_col)["signal"].transform(
        lambda s: s.rolling(5).sum().shift(1)
    )
    tmp.loc[(tmp["hold"] > 0) & (tmp["signal"] == 0), "signal"] = -1
    return tmp[[code_col, date_col, "signal"]]


def make_benchmark_returns(data: pd.DataFrame, code_col: str = "code",
                           date_col: str = "date", price_col: str = "close") -> pd.Series:
    """用第一只股票作为伪基准收益率"""
    first_code = data[code_col].iloc[0]
    sub = data[data[code_col] == first_code].sort_values(date_col)
    rets = sub.set_index(date_col)[price_col].pct_change().fillna(0.0)
    return rets
