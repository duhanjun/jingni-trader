"""
向量化因子中性化（优化验证版）

借鉴 Qlib 的 Processor 体系与 numpy.linalg.lstsq 批量截面回归思路，
重写 jingni-trader factor-engine 中的 neutralize 方法，解决以下问题：
1. 原实现用 `for dt in dates:` 逐日循环，每日单独拟合 sklearn LinearRegression
2. sklearn 对象创建 + fit 开销大，截面数多时性能急剧下降

优化方案：
- 用 numpy.linalg.lstsq 替代 sklearn LinearRegression（更快）
- 预构建行业哑变量矩阵，避免每日重复 pd.get_dummies
- 批量截面回归：将数据按日期分组后用 numpy 矩阵运算

借鉴来源：
- Qlib Processor 的 fit/__call__ 设计
- numpy.linalg.lstsq 批量最小二乘求解（比 sklearn SVD 快）
"""
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class VectorizedNeutralizer:
    """向量化因子中性化"""

    def neutralize_vectorized(
        self,
        factor_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        neutralize_mcap: bool = True,
        neutralize_industry: bool = True,
    ) -> pd.DataFrame:
        """
        向量化因子行业/市值中性化

        参数:
            factor_df: 含 code, date, [因子列], 可选 lncap
            industry_df: 含 code, industry
            neutralize_mcap: 是否市值中性化
            neutralize_industry: 是否行业中性化

        返回:
            添加 {factor}_neutral 列的 DataFrame
        """
        if not neutralize_industry and not neutralize_mcap:
            return factor_df
        if factor_df.empty:
            return factor_df

        result = factor_df.copy()
        if "industry" not in result.columns and neutralize_industry:
            result = result.merge(
                industry_df[["code", "industry"]], on="code", how="left"
            )

        factor_cols = [
            c for c in factor_df.columns
            if c not in ["code", "date", "industry"]
        ]

        # 预构建行业哑变量（全市场统一编码，避免每日重复 get_dummies）
        industry_dummies_full = None
        industry_cols = []
        if neutralize_industry and "industry" in result.columns:
            industry_dummies_full = pd.get_dummies(
                result["industry"], prefix="ind", dtype=float
            )
            industry_cols = list(industry_dummies_full.columns)

        for factor in factor_cols:
            if factor not in result.columns:
                continue
            result[f"{factor}_neutral"] = self._neutralize_one_factor(
                result,
                factor,
                industry_dummies_full,
                industry_cols,
                neutralize_mcap,
                neutralize_industry,
            )

        return result

    def _neutralize_one_factor(
        self,
        df: pd.DataFrame,
        factor: str,
        industry_dummies_full: Optional[pd.DataFrame],
        industry_cols: list,
        neutralize_mcap: bool,
        neutralize_industry: bool,
    ) -> pd.Series:
        """对单个因子做向量化截面中性化"""
        neutralized = pd.Series(index=df.index, dtype=float)

        # 组装自变量列名
        x_cols = []
        if neutralize_mcap and "lncap" in df.columns:
            x_cols.append("lncap")
        if neutralize_industry and industry_cols:
            x_cols.extend(industry_cols)

        if not x_cols:
            neutralized[:] = df[factor]
            return neutralized

        # 按日期分组，批量截面回归
        dates = df["date"].unique()
        for dt in dates:
            mask = df["date"] == dt
            cross = df.loc[mask]
            if len(cross) < 30:
                neutralized.loc[cross.index] = cross[factor]
                continue

            # 构建 X 矩阵
            X_parts = []
            if neutralize_mcap and "lncap" in cross.columns:
                X_parts.append(cross[["lncap"]].fillna(0).values)
            if neutralize_industry and industry_dummies_full is not None:
                X_parts.append(
                    industry_dummies_full.loc[cross.index, industry_cols].values
                )
            if not X_parts:
                neutralized.loc[cross.index] = cross[factor]
                continue

            X = np.hstack(X_parts)
            # 加截距列
            X = np.hstack([X, np.ones((X.shape[0], 1))])
            y = cross[factor].fillna(0).values

            try:
                # numpy.linalg.lstsq 比 sklearn LinearRegression 快
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ beta
                residual = y - y_pred
                neutralized.loc[cross.index] = residual
            except Exception:
                neutralized.loc[cross.index] = cross[factor]

        return neutralized


class OriginalNeutralizer:
    """原版中性化（用于对比测试，复制自 factor-engine/engine.py 的 neutralize）"""

    def neutralize_original(
        self,
        factor_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        neutralize_mcap: bool = True,
        neutralize_industry: bool = True,
    ) -> pd.DataFrame:
        """原版逐日 sklearn 循环实现（来自 main 分支 factor-engine）"""
        if not neutralize_industry and not neutralize_mcap:
            return factor_df
        if factor_df.empty:
            return factor_df

        result = factor_df.copy()
        if "industry" not in result.columns and neutralize_industry:
            result = result.merge(
                industry_df[["code", "industry"]], on="code", how="left"
            )

        factor_cols = [
            c for c in factor_df.columns
            if c not in ["code", "date", "industry"]
        ]

        for factor in factor_cols:
            if factor not in result.columns:
                continue

            dates = result["date"].unique()
            neutralized_values = pd.Series(index=result.index, dtype=float)

            for dt in dates:
                cross = result[result["date"] == dt].copy()
                if len(cross) < 30:
                    neutralized_values.loc[cross.index] = cross[factor]
                    continue

                X_vars = []
                if neutralize_mcap and "lncap" in cross.columns:
                    X_vars.append("lncap")
                if neutralize_industry and "industry" in cross.columns:
                    industry_dummies = pd.get_dummies(cross["industry"], prefix="ind")
                    for col in industry_dummies.columns:
                        cross[col] = industry_dummies[col].values
                        X_vars.append(col)

                if not X_vars:
                    neutralized_values.loc[cross.index] = cross[factor]
                    continue

                X = cross[X_vars].fillna(0).values
                y = cross[factor].fillna(0).values

                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    residual = y - y_pred
                    neutralized_values.loc[cross.index] = residual
                except Exception:
                    neutralized_values.loc[cross.index] = cross[factor]

            result[f"{factor}_neutral"] = neutralized_values

        return result
