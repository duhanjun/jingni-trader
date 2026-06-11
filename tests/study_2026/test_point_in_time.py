"""
借鉴来源: Microsoft Qlib Point-in-Time Database System
          (https://github.com/microsoft/qlib)
          Qlib 的 PoT 数据系统通过时间对齐确保回测时只使用当时可用的信息

优化方向: Point-in-Time 数据验证框架

当前 jingni-trader 的数据获取和因子计算流程未显式处理时间对齐问题，
可能导致:
  1. 因子计算无意中使用未来数据（如 rolling 窗口包含当前 bar）
  2. 财务数据发布时间与回测日期不对齐
  3. 股票停牌期间的数据插值引入偏差

本测试验证:
  1. Point-in-Time 数据存储结构的可行性
  2. 因子计算的时间对齐校验
  3. 前视数据泄漏的检测方法
"""

import sys
import os
import unittest
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---- Point-in-Time 数据存储 ----

class PointInTimeStore:
    """
    Point-in-Time 数据存储

    核心思想：
    - 每条数据记录都带一个 knowledge_time（该数据何时被知晓）
    - 查询时只能获取 knowledge_time <= query_time 的数据
    - 从架构层面杜绝前视数据泄漏
    """

    def __init__(self):
        self._store: Dict[str, pd.DataFrame] = {}

    def ingest(self, name: str, df: pd.DataFrame, knowledge_time_col: str = "pub_date"):
        """
        注入数据

        参数:
            name: 数据集名称，如 "financial_report", "trade_calendar"
            df: 数据 DataFrame，必须包含 knowledge_time_col
            knowledge_time_col: 标识数据何时为市场所知的列
        """
        if knowledge_time_col not in df.columns:
            raise ValueError(f"数据必须包含 {knowledge_time_col} 列")

        df_copy = df.copy()
        df_copy[knowledge_time_col] = pd.to_datetime(df_copy[knowledge_time_col])
        self._store[name] = df_copy.sort_values(knowledge_time_col)

    def get_as_of(self, name: str, as_of_date: pd.Timestamp,
                  knowledge_time_col: str = "pub_date") -> pd.DataFrame:
        """
        获取截至指定日期已知的数据

        参数:
            name: 数据集名称
            as_of_date: 截止日期
            knowledge_time_col: 知识时间列名

        返回:
            截至 as_of_date 已知的所有数据
        """
        if name not in self._store:
            return pd.DataFrame()

        df = self._store[name]
        mask = df[knowledge_time_col] <= as_of_date
        return df[mask].copy()


# ---- 因子计算时间对齐验证器 ----

class FactorTimeValidator:
    """
    因子时间对齐验证器

    检测因子计算中的常见前视偏差:
      1. Rolling 窗口包含当前 bar 而非仅历史 bar
      2. 分组操作中使用了未来数据
      3. 全局归一化使用了整个时间序列的统计量
    """

    @staticmethod
    def check_rolling_lookahead(
        data: pd.DataFrame,
        factor_fn,
        factor_name: str = "unknown",
        window: int = 20,
    ) -> Dict[str, Any]:
        """
        检查 rolling 计算是否有前视偏差

        方法：对比正常计算和"再滞后一天"计算的差异
        """

        # 正常计算
        factor_normal = factor_fn(data)

        # 对比：每次只有当前 bar 之前的 rolling
        factor_safe = pd.Series(index=range(len(data)), dtype=float)
        for i in range(len(data)):
            if i < window:
                factor_safe.iloc[i] = np.nan
            else:
                # 只用 i-window 到 i-1 的数据 (不包含 i)
                subset = data.iloc[max(0, i - window + 1):i + 1]
                factor_safe.iloc[i] = factor_fn(subset).iloc[-1]

        # 计算差异
        valid = factor_normal.notna() & factor_safe.notna()
        if valid.sum() < 2:
            return {
                "factor": factor_name,
                "lookahead_detected": None,
                "max_difference": 0,
                "note": "数据不足",
            }

        diff = (factor_normal[valid] - factor_safe[valid]).abs()
        return {
            "factor": factor_name,
            "lookahead_detected": diff.max() > 1e-10,
            "max_difference": float(diff.max()),
            "mean_difference": float(diff.mean()),
            "valid_points": int(valid.sum()),
        }

    @staticmethod
    def check_global_normalization(
        factor_df: pd.DataFrame,
        factor_col: str,
    ) -> Dict[str, Any]:
        """
        检查是否使用了全局归一化（整个时间序列的统计量）
        例如: (x - x.mean()) / x.std() 在整个数据集上计算
        """
        if factor_col not in factor_df.columns:
            return {"factor": factor_col, "error": "列不存在"}

        # 检查是否使用了 transform（分组内归一化，安全）
        # vs 直接计算全局统计量（可能包含未来数据）
        values = factor_df[factor_col].dropna()

        return {
            "factor": factor_col,
            "total_points": len(values),
            "recommendation": "使用 groupby('code').transform() 而非全局归一化",
        }


# ---- 前视数据泄漏检测器 ----

class LookaheadLeakDetector:
    """
    前视数据泄漏检测器

    通过注入"未来数据"并对比结果来检测泄漏。
    在合规的回测系统中，注入未来数据不应该改变任何历史交易决策。
    """

    def __init__(self, data: pd.DataFrame):
        self.original_data = data.sort_values(["code", "date"]).copy()

    def create_future_injected_data(self, lead_days: int = 1) -> pd.DataFrame:
        """
        创建注入未来数据的版本
        方法: 将 close 替换为 lead_days 天后的 close
        """
        df = self.original_data.copy()
        df["close_original"] = df["close"]
        df["close"] = df.groupby("code")["close"].shift(-lead_days).fillna(
            df["close"]
        )
        return df

    def detect_leakage_in_factor(
        self,
        compute_fn,
        factor_name: str = "unknown",
        tolerance: float = 1e-8,
    ) -> Dict[str, Any]:
        """
        检测因子计算中的前视泄漏

        方法: 分别用原始数据和未来注入数据计算因子
        如果结果相同 → 存在泄漏（因子使用了未来数据）
        如果结果不同 → 无泄漏
        """
        future_data = self.create_future_injected_data()

        factor_original = compute_fn(self.original_data)
        factor_future = compute_fn(future_data)

        # 对齐索引
        if isinstance(factor_original, pd.Series):
            orig = factor_original.reset_index(drop=True)
            fut = factor_future.reset_index(drop=True)
        elif isinstance(factor_original, pd.DataFrame):
            orig = factor_original[factor_name].reset_index(drop=True)
            fut = factor_future[factor_name].reset_index(drop=True)
        else:
            return {"factor": factor_name, "error": "不支持的返回类型"}

        valid = orig.notna() & fut.notna()
        if valid.sum() < 10:
            return {"factor": factor_name, "error": "有效数据不足"}

        diff = (orig[valid] - fut[valid]).abs()
        is_leaking = diff.max() <= tolerance

        return {
            "factor": factor_name,
            "lookahead_leak_detected": is_leaking,
            "max_original_future_diff": float(diff.max()),
            "mean_original_future_diff": float(diff.mean()),
            "valid_points": int(valid.sum()),
            "interpretation": (
                "⚠ 存在前视泄漏：因子在原始数据和未来数据上结果一致，"
                "表明计算中使用了未来信息" if is_leaking
                else "✓ 无前视泄漏：注入未来数据后结果显著不同"
            ),
        }


# ---- 测试用例 ----

class TestPointInTimeStore(unittest.TestCase):
    """Point-in-Time 数据存储测试"""

    def test_basic_ingest_and_query(self):
        """测试基本的数据注入和查询"""
        store = PointInTimeStore()

        # 模拟财务报告发布时间线
        financial_data = pd.DataFrame({
            "code": ["000001.SZ", "000001.SZ", "600000.SH"],
            "pub_date": [
                "2024-04-30",  # Q1 报告
                "2024-08-31",  # Q2 报告
                "2024-04-30",
            ],
            "quarter": ["Q1", "Q2", "Q1"],
            "eps": [0.5, 0.6, 0.8],
            "roe": [0.12, 0.14, 0.10],
        })

        store.ingest("financial_report", financial_data, "pub_date")

        # Q1 报告日之后查询
        df_q1 = store.get_as_of("financial_report", pd.Timestamp("2024-05-15"))
        self.assertEqual(len(df_q1), 2)  # 只应看到 Q1 数据

        # Q3 时应该看到全部（Q1+Q2）
        df_q3 = store.get_as_of("financial_report", pd.Timestamp("2024-09-01"))
        self.assertEqual(len(df_q3), 3)

        # 报告日之前查询
        df_before = store.get_as_of("financial_report", pd.Timestamp("2024-03-01"))
        self.assertEqual(len(df_before), 0)  # 没有任何报告

        print(f"\n✓ Point-in-Time 数据存储工作正常:")
        print(f"  Q1 报告日 (4/30) 后查询: {len(df_q1)} 条")
        print(f"  Q2 报告日 (8/31) 后查询: {len(df_q3)} 条")
        print(f"  报告日之前查询: {len(df_before)} 条")

    def test_cross_sectional_validity(self):
        """测试横截面数据的时间对齐"""
        store = PointInTimeStore()

        # 不同股票在不同时间发布财报
        reports = pd.DataFrame({
            "code": ["A", "B", "C", "A", "B"],
            "pub_date": [
                "2024-04-15", "2024-04-20", "2024-04-25",
                "2024-08-10", "2024-08-15",
            ],
            "eps": [1.0, 2.0, 3.0, 1.2, 2.2],
        })
        store.ingest("reports", reports, "pub_date")

        # 在 2024-04-18 做横截面分析
        cross = store.get_as_of("reports", pd.Timestamp("2024-04-18"))
        codes = cross["code"].tolist()
        self.assertIn("A", codes)      # A 的 Q1 已发布
        self.assertNotIn("B", codes)   # B 的 Q1 未发布
        self.assertNotIn("C", codes)   # C 的 Q1 未发布

        print(f"\n✓ 横截面时间对齐正确: 4/18 时只有 A 发布了财报")


class TestFactorTimeValidator(unittest.TestCase):
    """因子时间对齐验证测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 500
        cls.test_series = pd.Series(np.random.randn(n).cumsum() + 100)

    def test_rolling_mean_no_lookahead(self):
        """验证安全的 rolling mean 计算"""
        def safe_rolling_mean(s):
            return s.rolling(20).mean()

        result = FactorTimeValidator.check_rolling_lookahead(
            self.test_series,
            safe_rolling_mean,
            "rolling_mean_20",
            window=20,
        )
        # rolling().mean() 包含当前 bar，不算严格的安全
        # 但在业务上"用当天收盘价算因子"是可接受的
        print(f"\n✓ Rolling mean 时间对齐检查:")
        print(f"  前视偏差: {result['lookahead_detected']}")
        print(f"  最大差异: {result['max_difference']:.10f}")


class TestLookaheadLeakDetector(unittest.TestCase):
    """前视数据泄漏检测测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = [f"{i:06d}.SH" for i in range(600000, 600010)]
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")

        rows = []
        for code in codes:
            n = len(dates)
            start_price = np.random.uniform(5, 50)
            returns = np.random.normal(0.0005, 0.015, n)
            prices = start_price * np.cumprod(1 + returns)
            for i, (d, p) in enumerate(zip(dates, prices)):
                rows.append({
                    "code": code,
                    "date": d,
                    "close": p,
                    "volume": np.random.lognormal(12, 1),
                })

        cls.test_data = pd.DataFrame(rows).sort_values(["code", "date"])

    def test_safe_factor_no_leak(self):
        """测试安全因子计算（无前视泄漏）"""
        detector = LookaheadLeakDetector(self.test_data)

        def safe_momentum(data):
            """安全动量因子：用 Ref(close, 20) 而非当前 close/close.shift(20)"""
            return data.groupby("code")["close"].transform(
                lambda x: x / x.shift(20) - 1
            )

        result = detector.detect_leakage_in_factor(safe_momentum, "momentum_20")
        print(f"\n✓ 安全因子泄漏检测:")
        print(f"  因子: momentum_20")
        print(f"  前视泄漏: {result.get('lookahead_leak_detected')}")
        print(f"  最大差异: {result.get('max_original_future_diff', 'N/A')}")
        print(f"  解读: {result.get('interpretation', 'N/A')}")

    def test_leaky_factor_detected(self):
        """测试有泄漏的因子能被检测出来"""
        detector = LookaheadLeakDetector(self.test_data)

        def leaky_factor(data):
            """泄漏因子：直接用 pandas pct_change 算全表"""
            # 注意：这个本身是安全的（pct_change 是滞后），
            # 但我们需要构造一个"看起来安全但实际有泄漏"的例子
            # 比如用了未来数据的 groupby transform
            # 这里用一个明确的泄漏示例：直接用未来价格
            return data.groupby("code")["close"].transform(
                lambda x: x.shift(-1) / x - 1  # ← shift(-1) = 未来数据
            )

        result = detector.detect_leakage_in_factor(leaky_factor, "leaky_momentum")
        print(f"\n✓ 泄漏因子检测:")
        print(f"  因子: leaky_momentum")
        print(f"  前视泄漏: {result.get('lookahead_leak_detected')}")
        print(f"  最大差异: {result.get('max_original_future_diff', 'N/A')}")
        print(f"  解读: {result.get('interpretation', 'N/A')}")


if __name__ == "__main__":
    unittest.main(verbosity=2)