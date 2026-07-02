"""
验证测试：增强因子分析框架
====================================================
借鉴来源: Alphalens / Alphalens-Reloaded
          (https://github.com/stefan-jansen/alphalens-reloaded)
优化方向: factor-engine - 增强因子分析能力（分层收益、换手率、IC衰减）
日期: 2026-06-14

Alphalens 的核心亮点：
  - 标准化的因子分析框架：IC分析 → 分层收益 → 换手率 → 全面 tear sheet
  - MultiIndex 数据结构统一因子值、收益、分组信息
  - 支持行业中性化 IC 分析
  - 因子衰减分析（多周期 forward returns）
  - 因子自相关和稳定性分析

当前 jingni-trader factor-engine 的不足：
  - IC 分析仅有均值/标准差/ICIR，缺少完整的 IC 序列统计
  - 无分层收益分析（quantile returns）
  - 无换手率/因子稳定性分析
  - 无因子衰减分析（IC decay over horizons）

本测试验证：
  1. 分层收益分析的正确性
  2. IC 衰减分析的正确性
  3. 换手率分析的正确性
  4. 行业中性化 IC 的正确性
  5. 与现有 factor-engine 的 IC 分析结果一致性
"""

import unittest
import sys
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


# =====================================================
# 增强因子分析器（借鉴 Alphalens 设计）
# =====================================================

class EnhancedFactorAnalyzer:
    """
    增强因子分析器

    借鉴 Alphalens 核心 API 设计：
      - get_clean_factor_and_forward_returns: 数据准备
      - factor_information_coefficient: IC 序列
      - mean_return_by_quantile: 分层收益
      - quantile_turnover: 换手率分析
      - factor_rank_autocorrelation: 因子稳定性
    """

    def __init__(self,
                 factor_data: pd.DataFrame,
                 prices: Optional[pd.DataFrame] = None,
                 periods: Tuple[int, ...] = (1, 5, 20),
                 quantiles: int = 5,
                 groupby: Optional[pd.Series] = None):
        """
        参数:
            factor_data: 因子数据, 列为 [code, date] + 因子列
            prices: 价格数据, 列为 [code, date, close]
            periods: 前向收益周期 (天)
            quantiles: 分位数数量
            groupby: 分组标签 (行业等), Series 索引为 code
        """
        self.factor_data = factor_data.copy()
        self.prices = prices
        self.periods = periods
        self.quantiles = quantiles
        self.groupby = groupby
        self._clean_data = None

    def get_clean_factor_and_forward_returns(
        self,
        factor_col: str,
        max_loss: float = 0.35
    ) -> pd.DataFrame:
        """
        清理数据并计算前向收益

        借鉴 Alphalens get_clean_factor_and_forward_returns

        返回:
            MultiIndex DataFrame [date, code]:
                - factor: 因子值
                - factor_quantile: 分位数分组
                - period_1d, period_5d, period_20d: 各周期前向收益
        """
        df = self.factor_data.copy()

        # 确保 date 列是 datetime
        if df["date"].dtype != "datetime64[ns]":
            df["date"] = pd.to_datetime(df["date"])

        # 计算前向收益
        if self.prices is not None:
            price_df = self.prices.copy()
            if price_df["date"].dtype != "datetime64[ns]":
                price_df["date"] = pd.to_datetime(price_df["date"])

            df = df.merge(
                price_df[["code", "date", "close"]],
                on=["code", "date"],
                how="left"
            )

            for period in self.periods:
                col_name = f"period_{period}d"
                df[col_name] = df.groupby("code")["close"].transform(
                    lambda x: x.shift(-period) / x - 1
                )
        else:
            # 当没有价格数据时，使用因子数据中已包含的前向收益
            pass

        # 计算分位数
        df["factor_quantile"] = df.groupby("date")[factor_col].transform(
            lambda x: pd.qcut(x, self.quantiles, labels=False, duplicates="drop")
            if x.notna().sum() >= self.quantiles else np.nan
        )

        # 添加分组标签
        if self.groupby is not None:
            df["group"] = df["code"].map(self.groupby)

        # 设置 MultiIndex
        result = df.set_index(["date", "code"])
        self._clean_data = result
        return result

    def factor_information_coefficient(
        self,
        factor_col: str,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        计算完整的 IC 时间序列

        借鉴 Alphalens factor_information_coefficient

        返回:
            DataFrame: 每行一个日期，每列一个 forward period 的 IC 值
        """
        if self._clean_data is None:
            self.get_clean_factor_and_forward_returns(factor_col)

        data = self._clean_data
        ic_results = []

        for date in data.index.get_level_values("date").unique():
            cross = data.loc[date].dropna(subset=[factor_col])
            row = {"date": date}

            for period in self.periods:
                col = f"period_{period}d"
                if col not in cross.columns:
                    row[f"IC_{period}d"] = np.nan
                    continue

                valid = cross.dropna(subset=[factor_col, col])
                if len(valid) < 10:
                    row[f"IC_{period}d"] = np.nan
                    continue

                if method == "spearman":
                    ic, _ = stats.spearmanr(valid[factor_col], valid[col],
                                           nan_policy="omit")
                else:
                    ic, _ = stats.pearsonr(valid[factor_col].fillna(0),
                                          valid[col].fillna(0))
                row[f"IC_{period}d"] = ic

            ic_results.append(row)

        return pd.DataFrame(ic_results).set_index("date")

    def ic_summary(self, factor_col: str) -> pd.DataFrame:
        """
        IC 汇总统计

        返回:
            DataFrame: 每行一个 forward period，每列一个统计指标
        """
        ic_df = self.factor_information_coefficient(factor_col)
        ic_cols = [f"IC_{p}d" for p in self.periods]

        summary = []
        for col in ic_cols:
            series = ic_df[col].dropna()
            if len(series) < 3:
                continue

            period_str = col.replace("IC_", "")
            summary.append({
                "period": period_str,
                "IC_mean": series.mean(),
                "IC_std": series.std(),
                "IC_IR": series.mean() / series.std() if series.std() > 0 else 0,
                "IC_positive_ratio": (series > 0).mean(),
                "IC_t_stat": series.mean() / (series.std() / np.sqrt(len(series))) if series.std() > 0 else 0,
                "IC_skew": series.skew(),
                "IC_kurtosis": series.kurtosis(),
                "n_dates": len(series),
            })

        return pd.DataFrame(summary)

    def mean_return_by_quantile(
        self,
        factor_col: str
    ) -> pd.DataFrame:
        """
        分层收益分析

        借鉴 Alphalens mean_return_by_quantile

        返回:
            DataFrame: 每行一个分位组，每列一个 forward period 的平均收益
        """
        if self._clean_data is None:
            self.get_clean_factor_and_forward_returns(factor_col)

        data = self._clean_data
        result = []

        for q in range(self.quantiles):
            row = {"quantile": q}
            q_data = data[data["factor_quantile"] == q]

            for period in self.periods:
                col = f"period_{period}d"
                if col in q_data.columns:
                    row[f"mean_return_{period}d"] = q_data[col].mean()
                else:
                    row[f"mean_return_{period}d"] = np.nan

            row["count"] = len(q_data)
            result.append(row)

        return pd.DataFrame(result)

    def compute_return_spread(
        self,
        factor_col: str
    ) -> pd.DataFrame:
        """
        计算多空收益差（top quantile - bottom quantile）

        返回:
            DataFrame: 包含各周期 top-bottom 收益差
        """
        returns = self.mean_return_by_quantile(factor_col)
        if returns.empty or len(returns) < 2:
            return pd.DataFrame()

        top = returns.iloc[-1]
        bottom = returns.iloc[0]

        spreads = {"metric": "top_minus_bottom"}
        for period in self.periods:
            col = f"mean_return_{period}d"
            if col in returns.columns:
                spreads[f"spread_{period}d"] = top[col] - bottom[col]

        return pd.DataFrame([spreads])

    def ic_decay_analysis(self, factor_col: str) -> pd.DataFrame:
        """
        IC 衰减分析

        分析因子预测能力随 forward period 增加而衰减的速度

        返回:
            DataFrame: 包含各周期的 IC_mean 和衰减速率
        """
        summary = self.ic_summary(factor_col)
        if summary.empty:
            return pd.DataFrame()

        result = summary[["period", "IC_mean", "IC_IR", "IC_positive_ratio"]].copy()

        # 计算衰减速率（相对于 1d IC）
        ic_1d = result[result["period"] == "1d"]["IC_mean"].values
        if len(ic_1d) > 0:
            result["decay_ratio"] = result["IC_mean"].abs() / abs(ic_1d[0])
        else:
            result["decay_ratio"] = 1.0

        return result

    def quantile_turnover(
        self,
        factor_col: str
    ) -> pd.DataFrame:
        """
        分位数换手率分析

        借鉴 Alphalens quantile_turnover

        衡量因子不同分位组之间的成分变化频率。
        高换手率意味着因子值不稳定，交易成本高。

        返回:
            DataFrame: 每行一个分位组，每列一个周期的平均换手率
        """
        if self._clean_data is None:
            self.get_clean_factor_and_forward_returns(factor_col)

        data = self._clean_data
        dates = sorted(data.index.get_level_values("date").unique())

        if len(dates) < 2:
            return pd.DataFrame()

        turnover_results = {q: [] for q in range(self.quantiles)}

        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]

            prev = data.loc[prev_date]
            curr = data.loc[curr_date]

            # 计算共同代码
            common_codes = prev.index.intersection(curr.index)
            if len(common_codes) < self.quantiles * 2:
                continue

            for q in range(self.quantiles):
                in_prev = set(prev.loc[prev["factor_quantile"] == q].index)
                in_curr = set(curr.loc[curr["factor_quantile"] == q].index)
                common = in_prev & in_curr

                if len(in_prev) > 0:
                    turnover = 1 - len(common) / len(in_prev)
                else:
                    turnover = np.nan

                turnover_results[q].append(turnover)

        result = []
        for q in range(self.quantiles):
            vals = [v for v in turnover_results[q] if not np.isnan(v)]
            if vals:
                result.append({
                    "quantile": q,
                    "mean_turnover": np.mean(vals),
                    "std_turnover": np.std(vals),
                    "max_turnover": np.max(vals),
                })

        return pd.DataFrame(result)

    def factor_rank_autocorrelation(
        self,
        factor_col: str
    ) -> pd.DataFrame:
        """
        因子排名自相关分析

        衡量因子排名的稳定性。高自相关意味着因子值随时间缓慢变化。

        返回:
            DataFrame: 包含各滞后期的自相关系数
        """
        if self._clean_data is None:
            self.get_clean_factor_and_forward_returns(factor_col)

        data = self._clean_data
        dates = sorted(data.index.get_level_values("date").unique())

        # 计算每日截面排名
        rank_df = pd.DataFrame()
        for date in dates:
            cross = data.loc[date][[factor_col]].dropna()
            ranks = cross[factor_col].rank(pct=True)
            rank_df[date] = ranks

        # 计算不同滞后期的平均自相关
        lag_results = []
        lags = [1, 5, 20]

        for lag in lags:
            if lag >= len(rank_df.columns):
                continue

            autocorrs = []
            for i in range(lag, len(rank_df.columns)):
                s1 = rank_df.iloc[:, i - lag].dropna()
                s2 = rank_df.iloc[:, i].dropna()

                common = s1.index.intersection(s2.index)
                if len(common) < 10:
                    continue

                corr = stats.spearmanr(s1.loc[common], s2.loc[common])[0]
                if not np.isnan(corr):
                    autocorrs.append(corr)

            if autocorrs:
                lag_results.append({
                    "lag_days": lag,
                    "mean_autocorrelation": np.mean(autocorrs),
                    "std_autocorrelation": np.std(autocorrs),
                })

        return pd.DataFrame(lag_results)

    def group_neutral_ic(
        self,
        factor_col: str,
        group_col: str = "group"
    ) -> pd.DataFrame:
        """
        行业中性化 IC 分析

        在每个行业组内部计算 IC，然后取平均值。
        这消除了行业偏好对因子评估的干扰。

        返回:
            DataFrame: 行业中性化后的 IC 汇总
        """
        if self._clean_data is None:
            self.get_clean_factor_and_forward_returns(factor_col)

        if group_col not in self._clean_data.columns:
            return self.ic_summary(factor_col)

        data = self._clean_data
        groups = data[group_col].dropna().unique()

        group_ic_list = []
        for group in groups:
            group_data = data[data[group_col] == group].copy()

            # 对每个组内计算 IC
            ic_results = []
            for date in group_data.index.get_level_values("date").unique():
                cross = group_data.loc[date].dropna(subset=[factor_col])
                row = {"date": date, "group": group}

                for period in self.periods:
                    col = f"period_{period}d"
                    if col not in cross.columns:
                        row[f"IC_{period}d"] = np.nan
                        continue

                    valid = cross.dropna(subset=[factor_col, col])
                    if len(valid) < 5:
                        row[f"IC_{period}d"] = np.nan
                        continue

                    ic, _ = stats.spearmanr(valid[factor_col], valid[col],
                                           nan_policy="omit")
                    row[f"IC_{period}d"] = ic

                ic_results.append(row)

            if ic_results:
                group_ic_list.append(pd.DataFrame(ic_results))

        if not group_ic_list:
            return self.ic_summary(factor_col)

        # 汇总所有组的 IC
        all_group_ic = pd.concat(group_ic_list, ignore_index=True)

        summary = []
        for period in self.periods:
            col = f"IC_{period}d"
            if col not in all_group_ic.columns:
                continue

            series = all_group_ic[col].dropna()
            if len(series) < 3:
                continue

            period_str = f"{period}d"
            summary.append({
                "period": period_str,
                "IC_mean_neutral": series.mean(),
                "IC_std_neutral": series.std(),
                "IC_IR_neutral": series.mean() / series.std() if series.std() > 0 else 0,
                "IC_positive_ratio_neutral": (series > 0).mean(),
                "n_obs": len(series),
            })

        return pd.DataFrame(summary)

    def generate_full_report(self, factor_col: str) -> Dict:
        """
        生成完整因子分析报告

        借鉴 Alphalens create_full_tear_sheet

        返回:
            包含所有分析结果的字典
        """
        ic_summary = self.ic_summary(factor_col)
        returns_by_q = self.mean_return_by_quantile(factor_col)
        spread = self.compute_return_spread(factor_col)
        ic_decay = self.ic_decay_analysis(factor_col)
        turnover = self.quantile_turnover(factor_col)
        autocorr = self.factor_rank_autocorrelation(factor_col)

        if self.groupby is not None:
            neutral_ic = self.group_neutral_ic(factor_col)
        else:
            neutral_ic = pd.DataFrame()

        return {
            "factor": factor_col,
            "ic_summary": ic_summary,
            "return_by_quantile": returns_by_q,
            "return_spread": spread,
            "ic_decay": ic_decay,
            "turnover": turnover,
            "rank_autocorrelation": autocorr,
            "group_neutral_ic": neutral_ic,
        }


# =====================================================
# 辅助：模拟行业分类
# =====================================================

def create_mock_industry_map(codes: List[str]) -> pd.Series:
    """为股票代码创建模拟行业映射"""
    industries = ["银行", "科技", "消费", "医药", "能源"]
    np.random.seed(42)
    mapping = {code: industries[i % len(industries)] for i, code in enumerate(codes)}
    return pd.Series(mapping)


# =====================================================
# 单元测试
# =====================================================

class TestEnhancedFactorAnalysis(unittest.TestCase):
    """测试增强因子分析功能"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_codes = 100
        n_dates = 252  # 约1年交易日

        codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")

        # 创建面板数据
        rows = []
        for code in codes:
            # 模拟价格
            start_price = np.random.uniform(5, 100)
            returns = np.random.normal(0.0001, 0.02, n_dates)
            prices = start_price * np.cumprod(1 + returns)

            # 模拟因子值（与收益存在正相关）
            # 因子 = 0.3 * 未来5日收益 + 噪声
            # 未来5日收益: 从当天到5天后的收益
            future_price = np.roll(prices, -5)
            future_price[-5:] = np.nan  # 最后5天没有未来数据
            future_ret_5 = future_price / prices - 1
            future_ret_5[np.isnan(future_ret_5)] = 0
            noise = np.random.normal(0, 0.01, n_dates)
            factor = 0.3 * future_ret_5 + noise

            df_one = pd.DataFrame({
                "date": dates,
                "code": code,
                "close": prices,
                "factor_value": factor,
            })
            rows.append(df_one)

        cls.full_data = pd.concat(rows, ignore_index=True)

        # 价格数据
        cls.prices = cls.full_data[["code", "date", "close"]].copy()

        # 因子数据
        cls.factor_data = cls.full_data[["code", "date", "factor_value"]].copy()

        # 行业映射
        cls.industry_map = create_mock_industry_map(codes)

    def test_clean_data_preparation(self):
        """测试数据清理和前向收益计算"""
        analyzer = EnhancedFactorAnalyzer(
            self.factor_data, self.prices,
            periods=(1, 5, 20), quantiles=5
        )
        clean = analyzer.get_clean_factor_and_forward_returns("factor_value")

        self.assertIsNotNone(clean)
        self.assertIn("factor_quantile", clean.columns)
        self.assertIn("period_1d", clean.columns)
        self.assertIn("period_5d", clean.columns)

    def test_ic_analysis(self):
        """测试 IC 分析"""
        analyzer = EnhancedFactorAnalyzer(
            self.factor_data, self.prices,
            periods=(1, 5, 20), quantiles=5
        )

        ic_summary = analyzer.ic_summary("factor_value")
        self.assertGreater(len(ic_summary), 0)

        # 由于因子包含了未来收益信息，IC 应该为正
        ic_5d = ic_summary[ic_summary["period"] == "5d"]
        if len(ic_5d) > 0:
            self.assertGreater(ic_5d["IC_mean"].values[0], 0,
                "包含未来信息的因子 IC 应为正")

        # IC_IR 应该合理（不应是 NaN 或 inf）
        self.assertFalse(ic_summary["IC_IR"].isna().any(), "IC_IR should not be NaN")
        self.assertFalse((ic_summary["IC_IR"].abs() == np.inf).any(), "IC_IR should not be inf")

        print(f"\n  IC Summary:")
        print(f"  {ic_summary.to_string()}")

    def test_quantile_returns(self):
        """测试分层收益分析"""
        analyzer = EnhancedFactorAnalyzer(
            self.factor_data, self.prices,
            periods=(1, 5, 20), quantiles=5
        )

        returns = analyzer.mean_return_by_quantile("factor_value")
        self.assertEqual(len(returns), 5)
        self.assertIn("mean_return_5d", returns.columns)

        # 高因子组应该有更高的平均收益
        if len(returns) >= 5:
            top_ret = returns.iloc[-1]["mean_return_5d"]
            bottom_ret = returns.iloc[0]["mean_return_5d"]
            if not (np.isnan(top_ret) or np.isnan(bottom_ret)):
                self.assertGreater(top_ret, bottom_ret,
                    "高因子组收益应高于低因子组")

        print(f"\n  Quantile Returns:")
        print(f"  {returns.to_string()}")

    def test_return_spread(self):
        """测试多空收益差"""
        analyzer = EnhancedFactorAnalyzer(
            self.factor_data, self.prices,
            periods=(1, 5, 20), quantiles=5
        )

        spread = analyzer.compute_return_spread("factor_value")
        self.assertGreater(len(spread), 0)
        # 多空收益差应为正
        if "spread_5d" in spread.columns:
            self.assertGreater(spread["spread_5d"].values[0], 0)

    def test_ic_decay(self):
        """测试 IC 衰减分析"""
        analyzer = EnhancedFactorAnalyzer(
            self.factor_data, self.prices,
            periods=(1, 5, 20), quantiles=5
        )

        decay = analyzer.ic_decay_analysis("factor_value")
        self.assertGreater(len(decay), 0)
        self.assertIn("decay_ratio", decay.columns)

        print(f"\n  IC Decay:")
        print(f"  {decay.to_string()}")

    def test_turnover_analysis(self):
        """测试换手率分析"""
        analyzer = EnhancedFactorAnalyzer(
            self.factor_data, self.prices,
            periods=(1, 5, 20), quantiles=5
        )

        turnover = analyzer.quantile_turnover("factor_value")
        # 换手率应在 0-1 之间
        if len(turnover) > 0:
            means = turnover["mean_turnover"].dropna()
            if len(means) > 0:
                self.assertTrue(all(means >= 0))
                self.assertTrue(all(means <= 1))

        print(f"\n  Turnover:")
        print(f"  {turnover.to_string()}")

    def test_autocorrelation(self):
        """测试因子自相关分析"""
        analyzer = EnhancedFactorAnalyzer(
            self.factor_data, self.prices,
            periods=(1, 5, 20), quantiles=5
        )

        autocorr = analyzer.factor_rank_autocorrelation("factor_value")
        self.assertGreater(len(autocorr), 0)
        self.assertIn("mean_autocorrelation", autocorr.columns)

        # 自相关应在合理范围
        means = autocorr["mean_autocorrelation"].dropna()
        if len(means) > 0:
            self.assertTrue(all(means.abs() <= 1))

    def test_group_neutral_ic(self):
        """测试行业中性化 IC"""
        analyzer = EnhancedFactorAnalyzer(
            self.factor_data, self.prices,
            periods=(1, 5, 20), quantiles=5,
            groupby=self.industry_map
        )

        # 先调用 clean data 以添加 group 列
        analyzer.get_clean_factor_and_forward_returns("factor_value")
        neutral_ic = analyzer.group_neutral_ic("factor_value")

        if len(neutral_ic) > 0:
            self.assertIn("IC_mean_neutral", neutral_ic.columns)

        print(f"\n  Group-Neutral IC:")
        print(f"  {neutral_ic.to_string()}")

    def test_full_report(self):
        """测试完整报告生成"""
        analyzer = EnhancedFactorAnalyzer(
            self.factor_data, self.prices,
            periods=(1, 5, 20), quantiles=5,
            groupby=self.industry_map
        )

        report = analyzer.generate_full_report("factor_value")

        required_sections = [
            "ic_summary", "return_by_quantile", "return_spread",
            "ic_decay", "turnover", "rank_autocorrelation"
        ]
        for section in required_sections:
            self.assertIn(section, report, f"缺少报告章节: {section}")
            self.assertIsNotNone(report[section])


class TestConsistencyWithExistingIC(unittest.TestCase):
    """
    验证增强分析器的 IC 计算与现有 factor-engine 的一致性
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_codes = 50
        n_dates = 252
        codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")

        rows = []
        for code in codes:
            start_price = np.random.uniform(5, 100)
            returns = np.random.normal(0.0001, 0.02, n_dates)
            prices = start_price * np.cumprod(1 + returns)
            factor = np.random.normal(0, 0.01, n_dates)  # 纯随机因子

            df_one = pd.DataFrame({
                "date": dates,
                "code": code,
                "close": prices,
                "factor_value": factor,
            })
            rows.append(df_one)

        cls.full_data = pd.concat(rows, ignore_index=True)

    def test_ic_calculation_match(self):
        """测试 IC 计算与现有方法的一致性"""
        # 使用现有方法手动计算 IC（模拟 factor-engine 的 _calc_ic）
        df = self.full_data.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 计算前向收益
        df["ret_forward_5d"] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-5) / x - 1
        )

        # 手动计算 IC
        ic_list = []
        for dt in sorted(df["date"].unique()):
            cross = df[df["date"] == dt].dropna(subset=["factor_value", "ret_forward_5d"])
            if len(cross) < 10:
                continue
            ic, _ = stats.spearmanr(cross["factor_value"], cross["ret_forward_5d"],
                                   nan_policy="omit")
            if not np.isnan(ic):
                ic_list.append({"date": dt, "ic": ic})

        manual_ic = pd.DataFrame(ic_list).set_index("date")["ic"]
        manual_mean = manual_ic.mean()

        # 使用增强分析器
        analyzer = EnhancedFactorAnalyzer(
            self.full_data[["code", "date", "factor_value"]],
            self.full_data[["code", "date", "close"]],
            periods=(5,),
            quantiles=5
        )

        ic_summary = analyzer.ic_summary("factor_value")
        analyzer_mean = ic_summary[ic_summary["period"] == "5d"]["IC_mean"].values[0]

        # 由于随机因子，IC 应该接近 0
        msg = f"手动 IC_mean={manual_mean:.6f}, 分析器 IC_mean={analyzer_mean:.6f}"
        self.assertAlmostEqual(manual_mean, analyzer_mean, delta=0.02, msg=msg)
        print(f"\n  一致性验证: 手动IC={manual_mean:.6f}, 分析器IC={analyzer_mean:.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("增强因子分析框架 - 验证测试")
    print("借鉴来源: Alphalens / Alphalens-Reloaded")
    print("=" * 60)
    unittest.main(verbosity=2)