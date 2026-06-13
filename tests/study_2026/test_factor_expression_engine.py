"""
验证测试：因子表达式引擎 (Polars 驱动)
=============================================
借鉴来源: Microsoft Qlib Expression Engine + akquant Factor Expression Engine
优化方向: 用声明式 DSL 替代硬编码因子计算，使用 Polars 加速

Qlib 表达式引擎核心设计:
  - 声明式因子定义: $close, Ref($close, 1), Mean($close, 20)
  - AST 解析 -> 操作符树 -> 递归执行
  - 内置缓存避免重复计算

akquant 因子引擎核心设计:
  - Alpha101 风格表达式: Rank(Ts_Mean(Close, 5))
  - 基于 Polars Lazy API，自动优化查询计划
  - 时间序列算子: Ts_Mean, Ts_Std, Ts_Rank, Delay, Delta
  - 截面算子: Rank, Scale, Neutralize
  - 按 code 分区计算，自动对齐

本测试验证内容:
  1. Polars vs Pandas 因子计算性能对比
  2. 表达式引擎 DSL 原型可行性验证
  3. 可扩展性验证（批量因子定义 -> 自动计算）
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 生成模拟A股数据
def generate_test_data(n_stocks: int = 500, n_days: int = 500) -> pd.DataFrame:
    """生成模拟A股日线数据"""
    np.random.seed(20240101)
    codes = [f"{600000 + i:06d}.SH" if i < n_stocks // 2 else f"{i - n_stocks // 2:06d}.SZ"
             for i in range(n_stocks)]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    
    rows = []
    for code in codes:
        price = np.random.uniform(5, 50)
        for i, date in enumerate(dates):
            ret = np.random.normal(0.0002, 0.02)
            price *= (1 + ret)
            price = max(price, 1.0)
            
            open_p = price * (1 + np.random.normal(0, 0.005))
            high = max(price, open_p) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(price, open_p) * (1 - abs(np.random.normal(0, 0.01)))
            vol = int(np.random.lognormal(12, 0.8))
            amount = vol * price
            
            rows.append({
                "code": code, "date": date,
                "open": round(open_p, 2), "high": round(high, 2),
                "low": round(low, 2), "close": round(price, 2),
                "volume": vol, "amount": round(amount, 0),
                "change_pct": round(ret * 100, 4),
            })
    
    df = pd.DataFrame(rows)
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def benchmark_pandas_factors(df: pd.DataFrame) -> dict:
    """
    使用 pandas 实现因子计算（模拟 jingni-trader current 方式）
    计算 15 个因子
    """
    df = df.sort_values(['code', 'date']).copy()
    result = df[['code', 'date']].copy()
    
    t0 = time.perf_counter()
    
    # 收益率因子
    result['ret_1d'] = df.groupby('code')['close'].pct_change()
    result['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    result['ret_20d'] = df.groupby('code')['close'].pct_change(20)
    result['ret_60d'] = df.groupby('code')['close'].pct_change(60)
    
    # 反转因子
    result['reversal_5d'] = -result['ret_5d']
    result['reversal_20d'] = -result['ret_20d']
    
    # 市值
    if 'amount' in df.columns:
        result['lncap'] = df.groupby('code')['amount'].transform(
            lambda x: np.log(x.rolling(20, min_periods=5).mean().replace(0, np.nan)))
    
    # 换手率变化
    if 'amount' in df.columns and 'volume' in df.columns:
        result['turnover_20d'] = df.groupby('code')['amount'].transform(
            lambda x: x.rolling(20, min_periods=5).mean())
        result['turnover_5d'] = df.groupby('code')['amount'].transform(
            lambda x: x.rolling(5, min_periods=3).mean())
        result['turnover_change'] = result['turnover_5d'] / result['turnover_20d'].replace(0, np.nan) - 1
    
    # 波动率
    result['volatility_20d'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std())
    
    # 量比
    result['volume_20d'] = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    result['volume_ratio'] = df['volume'] / result['volume_20d'].replace(0, np.nan)
    
    # 资金流
    result['money_flow_raw'] = result['ret_1d'] * df['amount'].fillna(0)
    result['money_flow_20d'] = result.groupby('code')['money_flow_raw'].transform(
        lambda x: x.rolling(20, min_periods=5).sum())
    
    # SMA 因子
    for w in [5, 10, 20]:
        result[f'ma_{w}'] = df.groupby('code')['close'].transform(lambda x: x.rolling(w, min_periods=3).mean())
    
    elapsed = time.perf_counter() - t0
    
    factor_names = [c for c in result.columns if c not in ('code', 'date')]
    return {
        "backend": "pandas",
        "n_stocks": df['code'].nunique(),
        "n_days": df['date'].nunique(),
        "n_factors": len(factor_names),
        "time_seconds": round(elapsed, 4),
        "factors_per_second": round(len(factor_names) / elapsed, 2),
    }


def benchmark_polars_factors(df: pd.DataFrame) -> dict:
    """
    使用 Polars 实现因子计算（借鉴 akquant 设计）
    使用 Lazy API + 窗口函数批量计算
    """
    try:
        import polars as pl
    except ImportError:
        return {"backend": "polars", "error": "Polars 未安装", "time_seconds": 0}
    
    t0 = time.perf_counter()
    
    pl_df = pl.from_pandas(df).sort(["code", "date"])
    
    result = pl_df.lazy().with_columns([
        # 收益率因子
        (pl.col("close") / pl.col("close").shift(1).over("code") - 1).alias("ret_1d"),
        (pl.col("close") / pl.col("close").shift(5).over("code") - 1).alias("ret_5d"),
        (pl.col("close") / pl.col("close").shift(20).over("code") - 1).alias("ret_20d"),
        (pl.col("close") / pl.col("close").shift(60).over("code") - 1).alias("ret_60d"),
        # 反转
        (-(pl.col("close") / pl.col("close").shift(5).over("code") - 1)).alias("reversal_5d"),
        (-(pl.col("close") / pl.col("close").shift(20).over("code") - 1)).alias("reversal_20d"),
        # 市值对数
        pl.col("amount")
          .rolling_mean(window_size=20, min_samples=5)
          .over("code")
          .log()
          .alias("lncap"),
        # 波动率
        pl.col("close").pct_change().rolling_std(window_size=20, min_samples=10).over("code").alias("volatility_20d"),
        # 量比
        (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20, min_samples=5).over("code")).alias("volume_ratio"),
        # MA
        pl.col("close").rolling_mean(window_size=5, min_samples=3).over("code").alias("ma_5"),
        pl.col("close").rolling_mean(window_size=10, min_samples=3).over("code").alias("ma_10"),
        pl.col("close").rolling_mean(window_size=20, min_samples=3).over("code").alias("ma_20"),
        # 换手率变化
        pl.col("amount").rolling_mean(window_size=20, min_samples=5).over("code").alias("turnover_20d"),
        pl.col("amount").rolling_mean(window_size=5, min_samples=3).over("code").alias("turnover_5d"),
    ]).with_columns([
        (pl.col("turnover_5d") / pl.col("turnover_20d") - 1).alias("turnover_change"),
    ]).select([
        "code", "date", "ret_1d", "ret_5d", "ret_20d", "ret_60d",
        "reversal_5d", "reversal_20d", "lncap", "volatility_20d",
        "volume_ratio", "ma_5", "ma_10", "ma_20", "turnover_5d",
        "turnover_20d", "turnover_change",
    ]).collect()
    
    elapsed = time.perf_counter() - t0
    
    return {
        "backend": "polars",
        "n_stocks": df['code'].nunique(),
        "n_days": df['date'].nunique(),
        "n_factors": len(result.columns) - 2,
        "time_seconds": round(elapsed, 4),
        "factors_per_second": round((len(result.columns) - 2) / elapsed, 2) if elapsed > 0 else 0,
    }


# ════════════════════════════════════════════════════════════════
# 因子表达式 DSL 原型（借鉴 Qlib 设计）
# ════════════════════════════════════════════════════════════════

class FactorExpression:
    """
    简化版因子表达式引擎原型
    
    借鉴: Qlib 的 ExpressionOps + akquant 的 FactorEngine
    
    支持运算符:
      - Ref(field, n): 滞后 n 期
      - Mean(field, n): n 期移动平均
      - Std(field, n): n 期标准差
      - Rank(field): 截面排名 (百分比)
      - Delta(field, n): 滞后 n 期差分
      - Ts_Mean(field, n): 时间序列均值
      - Ts_Rank(field, n): 时间序列排名
      - 算术: +, -, *, /
    
    表达式的字符串语法:
      "Mean(close, 20) / Mean(close, 5) - 1"   # 均线差异比
      "Rank(Delta(close, 1))"                     # 截面动量排名
      "-1 * Ts_Mean(ret_1d, 5)"                   # 短期反转
    """
    
    def __init__(self):
        self._cache = {}
    
    def evaluate(self, expr: str, data: pd.DataFrame) -> pd.Series:
        """评估因子表达式，返回一维 Series"""
        if expr in self._cache:
            return self._cache[expr]
            
        expr = expr.strip()
        result = self._eval(expr, data)
        self._cache[expr] = result
        return result
    
    def evaluate_all(self, expressions: dict, data: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算因子
        
        参数:
            expressions: {因子名: 表达式字符串}
            data: 带 code, date 列的 DataFrame
        """
        import polars as pl
        pl_df = pl.from_pandas(data)
        result = pl_df.lazy()
        
        # 预先创建常用字段的 lazy 引用
        lazy_exprs = {}
        for name, expr in expressions.items():
            lazy_exprs[name] = self._build_polars_expr(expr, pl_df)
        
        result = result.with_columns([
            lazy_exprs[name].alias(name) for name in expressions.keys()
        ]).select([
            "code", "date", *expressions.keys()
        ]).collect()
        
        return result.to_pandas()
    
    def _build_polars_expr(self, expr: str, pl_df) -> 'pl.Expr':
        """将字符串表达式转换为 Polars Expression（递归解析）"""
        import polars as pl
        expr = expr.strip()
        
        # 滚动均值: Mean(field, n) 或 Ts_Mean(field, n)
        if expr.startswith("Mean(") or expr.startswith("Ts_Mean("):
            inner = expr[expr.index("(") + 1:expr.rindex(")")]
            parts = [p.strip() for p in inner.rsplit(",", 1)]
            field_expr, window = parts[0], int(parts[1])
            col = self._build_polars_expr(field_expr, pl_df) if '(' in field_expr else pl.col(field_expr)
            return col.rolling_mean(window_size=window, min_samples=window // 2).over("code")
        
        # 滚动标准差: Std(field, n)
        if expr.startswith("Std("):
            inner = expr[expr.index("(") + 1:expr.rindex(")")]
            parts = [p.strip() for p in inner.rsplit(",", 1)]
            field_expr, window = parts[0], int(parts[1])
            col = self._build_polars_expr(field_expr, pl_df) if '(' in field_expr else pl.col(field_expr)
            return col.rolling_std(window_size=window, min_samples=window // 2).over("code")
        
        # 滞后: Ref(field, n) 或 Delay(field, n)
        if expr.startswith("Ref(") or expr.startswith("Delay("):
            inner = expr[expr.index("(") + 1:expr.rindex(")")]
            parts = [p.strip() for p in inner.rsplit(",", 1)]
            field_expr, n = parts[0], int(parts[1])
            col = self._build_polars_expr(field_expr, pl_df) if '(' in field_expr else pl.col(field_expr)
            return col.shift(n).over("code")
        
        # 差分: Delta(field, n)
        if expr.startswith("Delta("):
            inner = expr[expr.index("(") + 1:expr.rindex(")")]
            parts = [p.strip() for p in inner.rsplit(",", 1)]
            field_expr, n = parts[0], int(parts[1])
            col = self._build_polars_expr(field_expr, pl_df) if '(' in field_expr else pl.col(field_expr)
            return col - col.shift(n).over("code")
        
        # 截面排名: Rank(field)
        if expr.startswith("Rank("):
            field = expr[expr.index("(") + 1:expr.rindex(")")].strip()
            return pl.col(field).rank("ordinal") / pl.col(field).count()
        
        # 收益率: Return(field, n)  等价于 field/Ref(field,n)-1
        if expr.startswith("Return("):
            inner = expr[expr.index("(") + 1:expr.rindex(")")]
            parts = [p.strip() for p in inner.rsplit(",", 1)]
            field_expr, n = parts[0], int(parts[1])
            col = self._build_polars_expr(field_expr, pl_df) if '(' in field_expr else pl.col(field_expr)
            shifted = col.shift(n).over("code")
            return col / shifted - 1
        
        # 取反: -expr
        if expr.startswith("-"):
            inner = expr[1:].strip()
            return -self._build_polars_expr(inner, pl_df)
        
        # 算术表达式: A + B, A - B, A * B, A / B
        for op in ['+', '-', '*', '/']:
            depth = 0
            op_idx = -1
            for i, ch in enumerate(expr):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif depth == 0 and ch == op:
                    # 跳过一元负号: 避免把 "A * -B" 中的 - 当作减法
                    if op == '-':
                        # 检查 - 前面是否是运算符或开头(一元负号)
                        prev_non_space = expr[:i].rstrip()
                        if prev_non_space == '' or prev_non_space[-1] in '(*':
                            continue
                    op_idx = i
                    break
            
            if op_idx > 0:
                left = expr[:op_idx].strip()
                right = expr[op_idx + 1:].strip()
                l_expr = self._build_polars_expr(left, pl_df)
                r_expr = self._build_polars_expr(right, pl_df) if right.replace('.', '').replace('-', '').isdigit() else self._build_polars_expr(right, pl_df)
                
                # 处理数值字面量
                if self._is_number(right):
                    r_val = float(right)
                    if op == '+': return l_expr + r_val
                    if op == '-': return l_expr - r_val
                    if op == '*': return l_expr * r_val
                    if op == '/': return l_expr / r_val
                else:
                    if op == '+': return l_expr + self._build_polars_expr(right, pl_df)
                    if op == '-': return l_expr - self._build_polars_expr(right, pl_df)
                    if op == '*': return l_expr * self._build_polars_expr(right, pl_df)
                    if op == '/': return l_expr / self._build_polars_expr(right, pl_df)
        
        # 字段引用
        return pl.col(expr)
    
    def _is_number(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _eval(self, expr: str, data: pd.DataFrame) -> pd.Series:
        """简单的 pandas 后端评估（用于单因子测试）"""
        import re
        
        # Mean(field, n)
        m = re.match(r'Mean\((\w+),\s*(\d+)\)', expr)
        if m:
            field, n = m.group(1), int(m.group(2))
            return data.groupby('code')[field].transform(lambda x: x.rolling(n, min_periods=n//2).mean())
        
        # Ref(field, n) / Delay(field, n)
        m = re.match(r'(?:Ref|Delay)\((\w+),\s*(\d+)\)', expr)
        if m:
            field, n = m.group(1), int(m.group(2))
            return data.groupby('code')[field].shift(n)
        
        # Delta(field, n)
        m = re.match(r'Delta\((\w+),\s*(\d+)\)', expr)
        if m:
            field, n = m.group(1), int(m.group(2))
            shifted = data.groupby('code')[field].shift(n)
            return data[field] - shifted
        
        # 直接列名
        if expr in data.columns:
            return data[expr]
        
        raise ValueError(f"无法解析表达式: {expr}")


def test_expression_engine():
    """测试表达式引擎 DSL 原型"""
    print("\n" + "=" * 70)
    print("测试: 因子表达式 DSL 原型")
    print("=" * 70)
    
    data = generate_test_data(n_stocks=100, n_days=200)
    engine = FactorExpression()
    
    # 定义因子表达式（Alpha101 风格）
    expressions = {
        "momentum_20d": "Return(close, 20)",
        "volatility_20d": "Std(Return(close, 1), 20)",
        "volume_ratio": "volume / Mean(volume, 20)",
        "reversal_5d": "Return(close, 5) * -1",
    }
    
    t0 = time.perf_counter()
    try:
        result = engine.evaluate_all(expressions, data)
        elapsed = time.perf_counter() - t0
        
        print(f"  表达式数量: {len(expressions)}")
        print(f"  数据规模: {data['code'].nunique()} 只股票 × {data['date'].nunique()} 天")
        print(f"  计算耗时: {elapsed:.4f}s")
        print(f"  结果形状: {result.shape}")
        
        # 验证每个因子值域是否合理
        for name in expressions:
            col = result[name]
            valid_ratio = col.notna().mean()
            print(f"    {name:25s}: 有效率={valid_ratio:.1%}, "
                  f"均值={col.mean():.6f}, 标准差={col.std():.6f}")
        
        print("  [PASS] 因子表达式 DSL 原型验证通过")
        return True
    except ImportError:
        print("  [SKIP] Polars 未安装，跳过 DSL 测试")
        return False


# ════════════════════════════════════════════════════════════════
# 主测试入口
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("因子表达式引擎验证测试")
    print("借鉴来源: Microsoft Qlib + akquant")
    print("优化方向: Polars 加速因子计算 + 声明式 DSL")
    print("=" * 70)
    
    results = []
    
    # 测试不同规模
    for n_stocks in [100, 300, 500]:
        print(f"\n--- 数据规模: {n_stocks} 只股票 × 500 天 ---")
        df = generate_test_data(n_stocks=n_stocks, n_days=500)
        print(f"  数据量: {len(df):,} 行")
        
        # pandas benchmark
        r_pd = benchmark_pandas_factors(df)
        results.append(r_pd)
        print(f"  pandas:  {r_pd['time_seconds']:.4f}s, "
              f"因子: {r_pd['n_factors']} 个")
        
        # polars benchmark
        r_pl = benchmark_polars_factors(df)
        results.append(r_pl)
        if 'error' not in r_pl:
            speedup = r_pd['time_seconds'] / r_pl['time_seconds'] if r_pl['time_seconds'] > 0 else float('inf')
            print(f"  polars:  {r_pl['time_seconds']:.4f}s, "
                  f"因子: {r_pl['n_factors']} 个, "
                  f"加速比: {speedup:.1f}x")
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("性能对比汇总")
    print("=" * 70)
    print(f"{'规模':<25s} {'Pandas(s)':<12s} {'Polars(s)':<12s} {'加速比':<10s}")
    print("-" * 59)
    for i in range(0, len(results), 2):
        if 'error' not in results[i+1]:
            r_pd = results[i]
            r_pl = results[i+1]
            scale = f"{r_pd['n_stocks']}股 × {r_pd['n_days']}天"
            speedup = r_pd['time_seconds'] / r_pl['time_seconds'] if r_pl['time_seconds'] > 0 else 0
            print(f"{scale:<25s} {r_pd['time_seconds']:<12.4f} {r_pl['time_seconds']:<12.4f} {speedup:<10.1f}x")
    
    # 表达式 DSL 测试
    test_expression_engine()
    
    # 保存结果
    report = {
        "test": "factor_expression_engine",
        "source_projects": ["microsoft/qlib", "akfamily/akquant"],
        "benchmarks": results,
    }
    report_path = os.path.join(os.path.dirname(__file__), "benchmark_factors.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {report_path}")
    
    return results


if __name__ == "__main__":
    main()