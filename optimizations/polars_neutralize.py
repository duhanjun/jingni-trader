"""
Polars 向量化因子中性化模块

借鉴来源：
- Microsoft Qlib: 向量化因子中性化
- AKQuant: Polars 驱动的因子处理

优化点：
原 factor-engine/engine.py 的 neutralize 方法通过 Python for 循环逐日做 OLS：
    for dt in dates:                              # Python 循环
        cross = result[result['date'] == dt]      # pandas 索引拷贝
        model = LinearRegression()                # sklearn 对象开销
        model.fit(X, y)                           # 逐日拟合
        residual = y - model.predict(X)

本模块使用 Frisch-Waugh-Lovell (FWL) 定理实现全向量化中性化：
- 市值中性化：用 cov/var/mean 的向量化公式逐组计算 OLS 残差
- 行业中性化：先按 (date, industry) 去均值（向量化），再对去均值后的
  变量做市值回归（向量化）。FWL 定理保证结果与完整 OLS 完全一致。
- 所有分组与聚合在 Polars/Rust 层多线程并行完成，无 Python 循环。
"""
from typing import Optional, List
import polars as pl


def neutralize_mcap_polars(
    df: pl.DataFrame,
    factor_col: str,
    mcap_col: str = "lncap",
    date_col: str = "date",
    min_samples: int = 30,
) -> pl.DataFrame:
    """
    市值中性化（向量化）

    对每个日期截面，回归 factor = a + b * lncap + e，取残差 e。
    使用向量化公式：
        slope = cov(x, y) / var(x)
        intercept = mean(y) - slope * mean(x)
        residual = y - (intercept + slope * x)

    参数:
        df: Polars DataFrame
        factor_col: 待中性化的因子列
        mcap_col: 对数市值列
        date_col: 日期列
        min_samples: 截面最少样本数，低于此值不中性化（返回原值）

    返回:
        增加 {factor_col}_neutral 列的 DataFrame
    """
    if factor_col not in df.columns or mcap_col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{factor_col}_neutral"))

    # 逐日计算回归统计量（向量化，Rust 多线程）
    stats = df.group_by(date_col).agg(
        pl.cov(factor_col, mcap_col).alias("_cov_xy"),
        pl.var(mcap_col).alias("_var_x"),
        pl.mean(factor_col).alias("_mean_y"),
        pl.mean(mcap_col).alias("_mean_x"),
        pl.len().alias("_n"),
    )

    # slope = cov / var，var 为 0 时 slope = 0
    stats = stats.with_columns(
        pl.when(pl.col("_var_x").is_not_null() & (pl.col("_var_x") != 0))
        .then(pl.col("_cov_xy") / pl.col("_var_x"))
        .otherwise(0.0)
        .alias("_slope")
    )

    # 连接回原表并计算残差
    result = df.join(stats, on=date_col, how="left")
    result = result.with_columns(
        pl.when(pl.col("_n") >= min_samples)
        .then(
            pl.col(factor_col)
            - (pl.col("_mean_y") + pl.col("_slope") * (pl.col(mcap_col) - pl.col("_mean_x")))
        )
        .otherwise(pl.col(factor_col))
        .alias(f"{factor_col}_neutral")
    )

    # 清理临时列
    return result.drop(
        "_cov_xy", "_var_x", "_mean_y", "_mean_x", "_n", "_slope"
    )


def neutralize_industry_mcap_polars(
    df: pl.DataFrame,
    factor_col: str,
    industry_col: str = "industry",
    mcap_col: str = "lncap",
    date_col: str = "date",
    min_samples: int = 30,
) -> pl.DataFrame:
    """
    行业 + 市值中性化（向量化，基于 FWL 定理）

    完整 OLS: factor = a + b*lncap + Σ c_k * industry_k + e
    FWL 定理等价计算：
        1. 按 (date, industry) 对 factor 和 lncap 去均值（行业效应）
        2. 对去均值后的变量做市值回归，取残差

    FWL 定理保证残差与完整 OLS 的残差在数值上完全一致。

    参数:
        df: Polars DataFrame
        factor_col: 待中性化的因子列
        industry_col: 行业列
        mcap_col: 对数市值列
        date_col: 日期列
        min_samples: 截面最少样本数

    返回:
        增加 {factor_col}_neutral 列的 DataFrame
    """
    if (
        factor_col not in df.columns
        or industry_col not in df.columns
        or mcap_col not in df.columns
    ):
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{factor_col}_neutral"))

    # Step 1: 按 (date, industry) 去均值 —— 剥离行业效应（向量化）
    ind_means = df.group_by([date_col, industry_col]).agg(
        pl.mean(factor_col).alias("_y_ind_mean"),
        pl.mean(mcap_col).alias("_x_ind_mean"),
    )
    df_dm = df.join(ind_means, on=[date_col, industry_col], how="left")
    df_dm = df_dm.with_columns(
        (pl.col(factor_col) - pl.col("_y_ind_mean")).alias("_y_dm"),
        (pl.col(mcap_col) - pl.col("_x_ind_mean")).alias("_x_dm"),
    )

    # Step 2: 对去均值后的变量做市值回归（向量化，同 neutralize_mcap_polars 逻辑）
    stats = df_dm.group_by(date_col).agg(
        pl.cov("_y_dm", "_x_dm").alias("_cov_xy"),
        pl.var("_x_dm").alias("_var_x"),
        pl.len().alias("_n"),
    )
    stats = stats.with_columns(
        pl.when(pl.col("_var_x").is_not_null() & (pl.col("_var_x") != 0))
        .then(pl.col("_cov_xy") / pl.col("_var_x"))
        .otherwise(0.0)
        .alias("_slope")
    )

    result = df_dm.join(stats, on=date_col, how="left")
    # 去均值后回归无需截距项（因 x_dm 和 y_dm 均值已为 0），残差 = y_dm - slope * x_dm
    result = result.with_columns(
        pl.when(pl.col("_n") >= min_samples)
        .then(pl.col("_y_dm") - pl.col("_slope") * pl.col("_x_dm"))
        .otherwise(pl.col(factor_col))
        .alias(f"{factor_col}_neutral")
    )

    return result.drop(
        "_y_ind_mean", "_x_ind_mean", "_y_dm", "_x_dm",
        "_cov_xy", "_var_x", "_n", "_slope",
    )


def neutralize_factors_batch_polars(
    df: pl.DataFrame,
    factor_names: List[str],
    industry_col: str = "industry",
    mcap_col: str = "lncap",
    date_col: str = "date",
    neutralize_industry: bool = True,
    neutralize_mcap: bool = True,
    min_samples: int = 30,
) -> pl.DataFrame:
    """
    批量因子中性化

    参数:
        df: Polars DataFrame
        factor_names: 待中性化的因子列表
        neutralize_industry: 是否行业中性化
        neutralize_mcap: 是否市值中性化

    返回:
        每个因子增加 {factor}_neutral 列的 DataFrame
    """
    result = df
    for factor in factor_names:
        if factor not in result.columns:
            continue
        if neutralize_industry and neutralize_mcap:
            result = neutralize_industry_mcap_polars(
                result, factor, industry_col, mcap_col, date_col, min_samples
            )
        elif neutralize_mcap:
            result = neutralize_mcap_polars(
                result, factor, mcap_col, date_col, min_samples
            )
        else:
            # 仅行业中性化：按 (date, industry) 去均值
            ind_means = result.group_by([date_col, industry_col]).agg(
                pl.mean(factor).alias(f"_{factor}_ind_mean"),
                pl.len().alias(f"_{factor}_n"),
            )
            result = result.join(ind_means, on=[date_col, industry_col], how="left")
            result = result.with_columns(
                pl.when(pl.col(f"_{factor}_n") >= min_samples)
                .then(pl.col(factor) - pl.col(f"_{factor}_ind_mean"))
                .otherwise(pl.col(factor))
                .alias(f"{factor}_neutral")
            ).drop([f"_{factor}_ind_mean", f"_{factor}_n"])
    return result
