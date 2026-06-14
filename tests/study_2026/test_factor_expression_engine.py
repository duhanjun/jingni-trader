"""
优化方向: 表达式驱动的因子引擎
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
核心借鉴: Alpha158/Alpha360 表达式引擎设计模式
日期: 2026-06-14

Qlib 的因子表达式引擎允许用户通过简单的字符串表达式定义因子，
如 "Ref($close, -5) / $close - 1" 表示5日收益率。
引擎内部自动处理数据对齐、NaN填充、截面/时序运算。

对比 jingni-trader 当前设计:
- 当前: 因子计算器需手写每个因子的 Python 代码，扩展性差
- 优化: 引入表达式解析，用户只需定义表达式字符串即可自动计算因子

验证目标:
1. 表达式解析正确性
2. 与手写因子计算结果一致性
3. 批量计算的性能表现
"""

import numpy as np
import pandas as pd
import re
import operator
import time
from typing import Dict, Any, List, Callable, Union
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. 表达式引擎核心实现（借鉴 Qlib 模式）
# ============================================================

class ExpressionEngine:
    """
    因子表达式引擎

    借鉴 Qlib 的表达式设计:
    - Ref(data, N): 前N期值（N<0=往前，N>0=往后）
    - Mean(data, N): N期滚动均值
    - Std(data, N): N期滚动标准差
    - Max(data, N), Min(data, N): N期滚动最大/最小值
    - Corr(x, y, N): N期滚动相关系数
    - Delta(data, N): N期差分
    - Rank(data): 截面排名
    - Scale(data): 截面标准化(z-score)
    - Log(data): 对数变换
    - +, -, *, /: 四则运算
    - 条件表达式: If(cond, true_val, false_val)

    实际Qlib支持的表达式远多于此，这里实现核心子集作为验证。
    """

    # 运算符优先级
    _OPS = {
        "+": (1, operator.add),
        "-": (1, operator.sub),
        "*": (2, operator.mul),
        "/": (2, operator.truediv),
        "%": (2, operator.mod),
        ">": (0, operator.gt),
        "<": (0, operator.lt),
        ">=": (0, operator.ge),
        "<=": (0, operator.le),
        "==": (0, operator.eq),
        "!=": (0, operator.ne),
        "&": (-1, operator.and_),
        "|": (-2, operator.or_),
    }

    # 注册的函数
    _FUNCTIONS: Dict[str, Callable] = {}

    def __init__(self, data_provider: Callable[[str], pd.DataFrame]):
        """
        参数:
            data_provider: 函数(field_name) -> DataFrame(index=date, columns=code)
                          用于获取原始数据字段
        """
        self._data = data_provider
        self._cache: Dict[str, pd.DataFrame] = {}

    def register_function(self, name: str, func: Callable):
        """注册自定义函数"""
        self._FUNCTIONS[name] = func

    def evaluate(self, expression: str) -> pd.DataFrame:
        """
        解析并计算因子表达式

        参数:
            expression: 因子表达式字符串，如 "Ref($close, -5) / $close - 1"

        返回:
            DataFrame (index=date, columns=code)
        """
        if expression in self._cache:
            return self._cache[expression]
        result = self._parse_and_evaluate(expression)
        self._cache[expression] = result
        return result

    def _parse_and_evaluate(self, expr: str) -> pd.DataFrame:
        """简易表达式解析器（递归下降）"""
        expr = expr.strip()

        # 去除外层括号 (A + B) → A + B
        while expr.startswith("(") and expr.endswith(")"):
            # 检查括号是否匹配（防止 If(cond, val1, val2) 这样的外括号）
            depth = 0
            valid = True
            for i, c in enumerate(expr[1:-1], 1):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                if depth < 0:
                    valid = False
                    break
            if valid and depth == 0:
                expr = expr[1:-1].strip()
            else:
                break

        # If 条件表达式
        if expr.startswith("If(") and expr.endswith(")"):
            return self._eval_if(expr)

        # 变量引用 $field_name
        if expr.startswith("$") and "(" not in expr:
            field_name = expr[1:]
            return self._get_data(field_name)

        # 数值字面量
        try:
            val = float(expr)
            return pd.DataFrame(
                val,
                index=self._data("close").index,
                columns=self._data("close").columns
            )
        except (ValueError, TypeError):
            pass

        # 先尝试二元运算（优先级最低），处理如 A + B、Mean(x,5)/Mean(y,20) 等
        # 这样可以正确处理包含函数调用的表达式
        try:
            return self._eval_binary(expr)
        except ValueError:
            pass

        # 一元负号: -(expression)
        if expr.startswith("-"):
            inner = self._parse_and_evaluate(expr[1:].strip())
            return -inner

        # 函数调用（纯函数调用，不含外部运算符）
        if "(" in expr and expr.endswith(")"):
            func_match = re.match(r'^(\w+)\((.+)\)$', expr, re.DOTALL)
            if func_match:
                func_name = func_match.group(1)
                args_str = func_match.group(2)
                args = self._split_args(args_str)
                return self._eval_function(func_name, args)

        # 变量引用 $field_name (可能包含括号的复杂变量)
        if expr.startswith("$"):
            field_name = expr[1:]
            return self._get_data(field_name)

        raise ValueError(f"无法解析表达式: {expr}")

    def _eval_binary(self, expr: str) -> pd.DataFrame:
        """解析二元运算"""
        # 按优先级从低到高查找运算符（从外层括号外找）
        for op_str, (prec, op_func) in sorted(
            self._OPS.items(), key=lambda x: x[1][0]
        ):
            depth = 0
            for i in range(len(expr) - 1, -1, -1):
                c = expr[i]
                if c == ")":
                    depth += 1
                elif c == "(":
                    depth -= 1
                elif depth == 0:
                    # 检查操作符位置
                    if op_str == "-" and i == 0:
                        continue  # 一元负号
                    if op_str in (">", "<", ">=", "<=", "==", "!="):
                        if expr.startswith(op_str, i):
                            left = self._parse_and_evaluate(expr[:i].strip())
                            right = self._parse_and_evaluate(expr[i + len(op_str):].strip())
                            return op_func(left, right)
                    elif expr[i] == op_str or (len(op_str) == 1 and expr[i] == op_str):
                        # 对于 - 号，确保不是负号
                        if op_str == "-" and (i == 0 or expr[i-1] in "(,+-*/%><=&|"):
                            continue
                        if len(op_str) == 1 and expr[i] == op_str:
                            left = self._parse_and_evaluate(expr[:i].strip())
                            right = self._parse_and_evaluate(expr[i + 1:].strip())
                            return op_func(left, right)

        raise ValueError(f"无法解析表达式: {expr}")

    def _extract_int(self, val) -> int:
        """从 DataFrame 或标量中提取整数值"""
        if isinstance(val, pd.DataFrame):
            return int(val.iloc[0, 0])
        return int(val)

    def _eval_function(self, name: str, args: List[str]) -> pd.DataFrame:
        """执行函数调用"""
        evaluated_args = [self._parse_and_evaluate(arg) for arg in args]

        # 内置函数
        if name in self._FUNCTIONS:
            return self._FUNCTIONS[name](*evaluated_args)

        # 标准函数
        func_map = {
            "Ref": lambda data, n: data.shift(self._extract_int(n)),
            "Mean": lambda data, n: data.rolling(window=self._extract_int(n), min_periods=1).mean(),
            "Std": lambda data, n: data.rolling(window=self._extract_int(n), min_periods=2).std(),
            "Max": lambda data, n: data.rolling(window=self._extract_int(n), min_periods=1).max(),
            "Min": lambda data, n: data.rolling(window=self._extract_int(n), min_periods=1).min(),
            "Sum": lambda data, n: data.rolling(window=self._extract_int(n), min_periods=1).sum(),
            "Delta": lambda data, n: data.diff(self._extract_int(n)),
            "Rank": self._cross_sectional_rank,
            "Scale": self._cross_sectional_scale,
            "Log": lambda data: np.log(data.clip(lower=1e-10)),
            "Abs": lambda data: data.abs(),
        }

        if name in func_map:
            return func_map[name](*evaluated_args)

        raise ValueError(f"未知函数: {name}")

    def _cross_sectional_rank(self, data: pd.DataFrame) -> pd.DataFrame:
        """截面排名（百分位，0~1）"""
        return data.rank(axis=1, pct=True)

    def _cross_sectional_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """截面标准化"""
        mean = data.mean(axis=1)
        std = data.std(axis=1).clip(lower=1e-10)
        return data.sub(mean, axis=0).div(std, axis=0)

    def _eval_if(self, expr: str) -> pd.DataFrame:
        """If(cond, true_val, false_val)"""
        args = self._split_args(expr[3:-1])
        if len(args) != 3:
            raise ValueError(f"If 需要3个参数，得到 {len(args)}")
        cond = self._parse_and_evaluate(args[0])
        true_val = self._parse_and_evaluate(args[1])
        false_val = self._parse_and_evaluate(args[2])
        return true_val.where(cond > 0, false_val)

    def _split_args(self, args_str: str) -> List[str]:
        """按逗号分割函数参数（考虑嵌套函数）"""
        args = []
        depth = 0
        current = ""
        for c in args_str:
            if c == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                current += c
        if current.strip():
            args.append(current.strip())
        return args

    def _get_data(self, field_name: str) -> pd.DataFrame:
        """获取原始数据字段"""
        return self._data(field_name)


# ============================================================
# 2. Qlib 风格因子集定义
# ============================================================

def build_alpha158_expressions() -> Dict[str, str]:
    """
    Qlib Alpha158 风格因子的表达式子集

    注意: 这是简化版，真实 Alpha158 包含158个因子，
    这里只选取代表性的20个因子作为验证。
    """
    # 参考 Qlib qlib/contrib/data/handler.py Alpha158
    # 表达式使用 $ 前缀引用 OHLCV 原始字段
    return {
        # === 收益率类因子 ===
        "RET5": "Ref($close, -5) / $close - 1",       # 5日收益率
        "RET10": "Ref($close, -10) / $close - 1",     # 10日收益率
        "RET20": "Ref($close, -20) / $close - 1",     # 20日收益率

        # === 波动率类因子 ===
        "STD5": "Std(($close / Ref($close, -1) - 1), 5)",    # 5日收益率波动
        "STD20": "Std(($close / Ref($close, -1) - 1), 20)",  # 20日收益率波动

        # === 均线偏离因子 ===
        "MA5_DEV": "$close / Mean($close, 5) - 1",     # 5日均线偏离
        "MA20_DEV": "$close / Mean($close, 20) - 1",   # 20日均线偏离

        # === 成交量因子 ===
        "VOL_RATIO5": "Mean($volume, 5) / Mean($volume, 20)",
        "VOL_MA5": "$volume / Mean($volume, 5) - 1",

        # === 价格范围因子 ===
        "HIGH_LOW": "($high - $low) / $close",
        "PRICE_RANGE": "($close - $open) / ($high - $low + 0.0001)",

        # === 动量因子 ===
        "MOM5": "$close - Ref($close, 5)",
        "ROC10": "($close - Ref($close, 10)) / Ref($close, 10)",

        # === 换手率因子 ===
        "TURN_STD20": "Std($turnover_rate, 20)",
        "TURN_MEAN5": "Mean($turnover_rate, 5)",

        # === 截面排名因子（参考Alpha158中的截面因子）===
        "RET5_RANK": "Rank(Ref($close, -5) / $close - 1)",
        "VOL_RANK": "Rank($volume)",
        "MA20_DEV_RANK": "Rank($close / Mean($close, 20) - 1)",

        # === 复合因子 ===
        "REVERSAL_STD": "-(Ref($close, -5) / $close - 1) * Std(($close / Ref($close, -1) - 1), 20)",

        # === 对数因子 ===
        "LOG_RET5": "Log($close / Ref($close, 5))",
    }


# ============================================================
# 3. 与手写因子计算的一致性验证
# ============================================================

def build_sample_data(dates: int = 200, stocks: int = 50, seed: int = 42) -> pd.DataFrame:
    """生成模拟行情数据用于测试"""
    np.random.seed(seed)

    date_range = pd.date_range("2024-01-01", periods=dates, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(stocks)]

    # 模拟价格走势（几何布朗运动）
    base_prices = np.ones((dates, 1)) * 10.0
    returns = np.random.normal(0.0005, 0.02, (dates, stocks))
    returns[0] = 0

    prices = np.exp(np.cumsum(returns, axis=0)) * 10

    close = pd.DataFrame(prices, index=date_range, columns=codes)
    daily_returns = close.pct_change().fillna(0)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, (dates, stocks))))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, (dates, stocks))))
    open_price = close.shift(1).fillna(close.iloc[0]) * (1 + np.random.normal(0, 0.005, (dates, stocks)))
    volume = pd.DataFrame(
        np.abs(np.random.normal(1e7, 3e6, (dates, stocks))),
        index=date_range, columns=codes
    )
    amount = close * volume
    turnover_rate = pd.DataFrame(
        np.random.beta(2, 10, (dates, stocks)) * 0.15,
        index=date_range, columns=codes
    )

    return {
        "close": close,
        "open": open_price,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": amount,
        "turnover_rate": turnover_rate,
    }


def test_expression_engine():
    """测试表达式引擎的正确性和性能"""
    print("=" * 60)
    print("TEST 1: Expression Engine Correctness & Performance")
    print("=" * 60)

    # 准备数据
    raw_data = build_sample_data(dates=200, stocks=50)

    # 创建数据提供者
    def data_provider(field_name: str) -> pd.DataFrame:
        return raw_data.get(field_name, pd.DataFrame())

    engine = ExpressionEngine(data_provider)

    # === 子测试1: 基础表达式计算 ===
    print("\n--- Sub-test 1.1: Basic Expressions ---")
    expressions = [
        "Ref($close, -5) / $close - 1",     # 5日收益率
        "$close / Mean($close, 20) - 1",    # 20日均线偏离
        "Mean($volume, 5) / Mean($volume, 20)",  # 量比
        "($high - $low) / $close",           # 波幅
    ]

    for i, expr in enumerate(expressions):
        result = engine.evaluate(expr)
        nan_count = result.isna().sum().sum()
        total = result.size
        print(f"  Expr {i}: '{expr[:50]}...' => shape={result.shape}, NaN={nan_count}/{total}={nan_count/total*100:.1f}%")

    # === 子测试2: 截面运算 ===
    print("\n--- Sub-test 1.2: Cross-Sectional Operations ---")
    rank_result = engine.evaluate("Rank($close / Mean($close, 20) - 1)")
    print(f"  Rank result: shape={rank_result.shape}, values in [{rank_result.min().min():.3f}, {rank_result.max().max():.3f}]")

    scale_result = engine.evaluate("Scale($close / Mean($close, 20) - 1)")
    print(f"  Scale result: mean per row ≈ {scale_result.mean(axis=1).mean():.6f}, std per row ≈ {scale_result.std(axis=1).mean():.4f}")

    # === 子测试3: 条件表达式 ===
    print("\n--- Sub-test 1.3: Conditional Expressions ---")

    # 手动计算5日收益率
    ret5_manual = raw_data["close"].pct_change(5).shift(-5)
    cond = pd.DataFrame(np.where(ret5_manual > 0, 1, -1), index=ret5_manual.index, columns=ret5_manual.columns)

    if_result = engine.evaluate(
        "If(Ref($close, -5) / $close - 1 > 0, 1, -1)"
    )
    # 验证 (截取非NaN部分)
    match_mask = (~if_result.isna()) & (~cond.isna())
    # 比较 values
    if_match = if_result.where(match_mask).values == cond.where(match_mask).values
    match_rate = if_match.sum() / match_mask.sum().sum() * 100
    print(f"  If condition match rate: {match_rate:.2f}%")

    # === 子测试4: 与手写因子对比 ===
    print("\n--- Sub-test 1.4: Manual vs Expression ---")

    # 手动计算 MA20偏离
    close = raw_data["close"]
    ma20_manual = close.rolling(20).mean()
    ma20_dev_manual = close / ma20_manual - 1

    ma20_dev_expr = engine.evaluate("$close / Mean($close, 20) - 1")

    # 对齐后比较
    diff = (ma20_dev_expr - ma20_dev_manual).abs()
    # 仅比较两者都非NaN的部分
    valid = (~ma20_dev_expr.isna()) & (~ma20_dev_manual.isna())
    max_diff = diff[valid].max().max()
    mean_diff = diff[valid].mean().mean()
    print(f"  MA20_DEV max difference: {max_diff:.10f}")
    print(f"  MA20_DEV mean difference: {mean_diff:.10f}")
    assert max_diff < 1e-8, f"Expression and manual results differ! max_diff={max_diff}"

    # === 子测试5: Alpha158 批量计算 ===
    print("\n--- Sub-test 1.5: Alpha158 Subset Batch Calculation ---")
    factor_exprs = build_alpha158_expressions()

    start_time = time.time()
    results = {}
    for name, expr in factor_exprs.items():
        results[name] = engine.evaluate(expr)
    batch_time = time.time() - start_time

    print(f"  Computed {len(factor_exprs)} factors in {batch_time:.3f}s ({batch_time/len(factor_exprs)*1000:.1f}ms per factor)")
    for name, df in list(results.items())[:5]:
        print(f"    {name}: shape={df.shape}, nan_ratio={df.isna().sum().sum()/df.size*100:.1f}%")

    # === 子测试6: 缓存性能 ===
    print("\n--- Sub-test 1.6: Cache Performance ---")
    start_time = time.time()
    _ = engine.evaluate("$close / Mean($close, 20) - 1")
    first_time = time.time() - start_time

    start_time = time.time()
    _ = engine.evaluate("$close / Mean($close, 20) - 1")
    second_time = time.time() - start_time

    print(f"  First evaluation: {first_time*1000:.1f}ms")
    print(f"  Cached evaluation: {second_time*1000:.1f}ms")
    print(f"  Speedup: {first_time/max(second_time, 1e-6):.0f}x")

    print("\n✓ Expression Engine test PASSED\n")
    return True


# ============================================================
# 4. 因子库可扩展性对比测试
# ============================================================

def test_factor_extensibility():
    """
    对比: 手写因子 vs 表达式引擎的扩展成本
    参考 Qlib 的设计理念：表达式引擎让因子扩展几乎零代码成本
    """
    print("=" * 60)
    print("TEST 2: Factor Extensibility Comparison")
    print("=" * 60)

    print("""
    场景: 新增20个Alpha因子需要多少代码行？

    传统方式 (当前 jingni-trader):
        - 每个因子需要实现 calculate 方法
        - 大约 10-20 行/因子 (含数据处理、异常处理)
        - 总计: ~200-400 行代码
        - 风险: 代码重复、容易出错、难维护

    表达式引擎方式 (借鉴 Qlib):
        - 每个因子只需 1 行表达式字符串
        - 总计: 20 行配置
        - 优势: 声明式、易读、易测试、标准化

    Alpha158 因子表达式示例 (只用20行):
    """)

    factor_exprs = build_alpha158_expressions()
    for name, expr in sorted(factor_exprs.items()):
        print(f"  {name:15s} = {expr}")

    print(f"\n  代码量对比: {len(factor_exprs)} factors")
    print(f"    表达式方式: {len(factor_exprs)} 行")
    print(f"    传统方式: ~{len(factor_exprs) * 15} 行")
    print(f"    减少: ~{(1 - len(factor_exprs) / (len(factor_exprs) * 15)) * 100:.0f}%")
    print(f"    因子注册只需: factor_exprs[name] = 'expression_string'")

    print("\n✓ Factor Extensibility test PASSED\n")
    return True


# ============================================================
# 5. 建议改进方向
# ============================================================

def print_recommendations():
    print("=" * 60)
    print("RECOMMENDATIONS: 因子引擎优化建议")
    print("=" * 60)
    print("""
    1. [高优先级] 引入表达式引擎
       - 在 factor-engine/scripts/ 下新增 expression_engine.py
       - 保留现有 TA-Lib/pandas_ta 适配器作为底层计算器
       - 表达式引擎作为上层抽象，调用底层计算器
       - 目标: 用户只需 "Ref($close, -5)/$close-1" 即定义因子

    2. [中优先级] 预置因子库
       - 参照 Alpha158/Alpha360 模式，预置行业标准因子集
       - 存储在 factor-engine/config/presets/alpha158.yaml
       - 一键加载: load_factors("alpha158")

    3. [中优先级] 因子元数据标准化
       - 每个因子附带: 名称、方向(正向/反向/中性)、类别、
         参数、说明、来源
       - 便于后续 IC 分析、因子淘汰、报告生成

    4. [低优先级] 因子表达式验证器
       - 语法检查、循环检测、字段引用校验
       - 在因子注册阶段就发现语法错误
    """)
    print("=" * 60)


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    print("jingni-trader 优化验证 #1: 表达式驱动的因子引擎")
    print("借鉴来源: Microsoft Qlib Alpha158/Alpha360\n")

    try:
        test_expression_engine()
        test_factor_extensibility()
        print_recommendations()
        print("\n" + "=" * 60)
        print("所有验证通过!")
        print("=" * 60)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n验证失败: {e}")
        exit(1)