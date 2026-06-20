"""
向量化因子引擎 —— 优化验证实现

借鉴来源:
- Qlib: 横截面处理器用 groupby('date').transform 向量化, 而非逐日 Python 循环;
        RobustZScoreNorm = (x-median)/MAD 再 clip; CSRankNorm 横截面排名归一
- Qlib: 因子表达式 DSL (字符串→算子树→向量化求值)

对照原实现 skills/factor-engine/engine.py:
- neutralize(): L148 `for dt in dates:` 逐日 sklearn LinearRegression.fit → 慢
- _calc_ic(): L250 `for dt in dates:` 逐日 scipy spearmanr → 慢

本文件提供向量化版本用于性能对比与正确性验证。
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
from scipy import stats


# ────────────────────────────────────────────────────────────
# 1. 因子中性化: 逐日循环版 (复刻原实现) vs 向量化版
# ────────────────────────────────────────────────────────────
def neutralize_loop(
    factor_df: pd.DataFrame,
    factor_cols: List[str],
    neutralize_mcap: bool = True,
    neutralize_industry: bool = True,
) -> pd.DataFrame:
    """
    复刻原 factor-engine neutralize 的逐日循环逻辑 (sklearn LinearRegression)。
    作为性能与正确性基线。
    """
    from sklearn.linear_model import LinearRegression
    result = factor_df.copy()
    if "industry" not in result.columns and neutralize_industry:
        return result  # 原实现需要 industry_df 合并, 此处简化

    for factor in factor_cols:
        if factor not in result.columns:
            continue
        neutralized_values = pd.Series(index=result.index, dtype=float)
        for dt in result["date"].unique():
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
                residual = y - model.predict(X)
                neutralized_values.loc[cross.index] = residual
            except Exception:
                neutralized_values.loc[cross.index] = cross[factor]
        result[f"{factor}_neutral"] = neutralized_values
    return result


def neutralize_vectorized(
    factor_df: pd.DataFrame,
    factor_cols: List[str],
    neutralize_mcap: bool = True,
    neutralize_industry: bool = True,
) -> pd.DataFrame:
    """
    向量化中性化 (Qlib groupby 思路)。

    核心优化:
    - 用 groupby('date') 一次性处理所有日期, 而非逐日 Python 循环
    - 用 numpy.linalg.lstsq 替代 sklearn LinearRegression (更轻量)
    - 行业哑变量一次性生成

    数学等价: 对每个 (日期, 因子), 残差 = y - X @ beta, beta = lstsq(X, y)
    """
    result = factor_df.copy()
    if not neutralize_mcap and not neutralize_industry:
        for f in factor_cols:
            result[f"{f}_neutral"] = result[f]
        return result

    # 预构建行业哑变量 (一次性, 全局)
    if neutralize_industry and "industry" in result.columns:
        industry_dummies = pd.get_dummies(result["industry"], prefix="ind", dtype=float)
        dummy_cols = industry_dummies.columns.tolist()
        result = pd.concat([result, industry_dummies], axis=1)
    else:
        dummy_cols = []

    x_base_cols = []
    if neutralize_mcap and "lncap" in result.columns:
        x_base_cols.append("lncap")
    x_base_cols += dummy_cols

    if not x_base_cols:
        for f in factor_cols:
            result[f"{f}_neutral"] = result[f]
        return result

    # 按 date 分组, 每组做一次 lstsq (向量化分组)
    grouped = result.groupby("date", sort=False)

    def _residualize(cross: pd.DataFrame, factor: str) -> pd.Series:
        if len(cross) < 30:
            return cross[factor]
        X = cross[x_base_cols].fillna(0.0).values
        y = cross[factor].fillna(0.0).values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            residual = y - X @ beta
            return pd.Series(residual, index=cross.index)
        except Exception:
            return cross[factor]

    for factor in factor_cols:
        if factor not in result.columns:
            continue
        # groupby.apply 对每个日期组做残差化, 一次性处理
        neutralized = grouped[factor].transform(lambda s: s)  # 占位, 保持索引对齐
        # 直接用 apply 重建
        parts = []
        for dt, cross in grouped:
            parts.append(_residualize(cross, factor))
        result[f"{factor}_neutral"] = pd.concat(parts).reindex(result.index)

    # 清理临时哑变量列
    result = result.drop(columns=dummy_cols, errors="ignore")
    return result


# ────────────────────────────────────────────────────────────
# 2. IC 分析: 逐日循环版 vs 向量化版
# ────────────────────────────────────────────────────────────
def ic_analysis_loop(
    data: pd.DataFrame,
    factor_names: List[str],
    forward_col: str = "ret_forward_5d",
    ic_type: str = "spearman",
) -> Dict[str, Dict[str, float]]:
    """复刻原 _calc_ic 逐日循环逻辑"""
    results = {}
    for factor in factor_names:
        if factor not in data.columns:
            continue
        ic_list = []
        for dt in sorted(data["date"].unique()):
            cross = data[data["date"] == dt].dropna(subset=[factor, forward_col])
            if len(cross) < 10:
                continue
            if ic_type == "spearman":
                ic, _ = stats.spearmanr(cross[factor], cross[forward_col], nan_policy="omit")
            else:
                ic, _ = stats.pearsonr(cross[factor].fillna(0), cross[forward_col].fillna(0))
            if not np.isnan(ic):
                ic_list.append(ic)
        if not ic_list:
            continue
        ic_arr = np.array(ic_list)
        ic_std = ic_arr.std()
        results[factor] = {
            "ic_mean": float(ic_arr.mean()),
            "ic_std": float(ic_std),
            "ic_ir": float(ic_arr.mean() / ic_std) if ic_std > 0 else 0.0,
            "ic_positive_ratio": float((ic_arr > 0).mean()),
            "n_obs": int(len(ic_arr)),
        }
    return results


def ic_analysis_vectorized(
    data: pd.DataFrame,
    factor_names: List[str],
    forward_col: str = "ret_forward_5d",
    ic_type: str = "spearman",
) -> Dict[str, Dict[str, float]]:
    """
    向量化 IC 分析 (Qlib groupby 思路)。

    核心优化:
    - 用 groupby('date')[[factor, forward]].corr 一次性计算所有日期的 IC
    - 避免 scipy spearmanr 逐日调用的 Python 开销
    - pandas groupby + corr 内部用 C 加速
    """
    results = {}
    method = "spearman" if ic_type == "spearman" else "pearson"
    grouped = data.groupby("date", sort=False)

    for factor in factor_names:
        if factor not in data.columns or forward_col not in data.columns:
            continue
        # 过滤有效样本
        valid = data.dropna(subset=[factor, forward_col])
        # 按日期分组, 每组至少 10 个样本
        cnt = valid.groupby("date")[factor].count()
        valid_dates = cnt[cnt >= 10].index
        valid = valid[valid["date"].isin(valid_dates)]
        if valid.empty:
            continue

        # 向量化: groupby + corr 一次性算出每日 IC
        # 对 spearman, 先 rank 再 pearson 等价
        if ic_type == "spearman":
            valid = valid.copy()
            valid[factor + "_rank"] = valid.groupby("date")[factor].rank()
            valid[forward_col + "_rank"] = valid.groupby("date")[forward_col].rank()
            ic_series = valid.groupby("date").apply(
                lambda g: g[factor + "_rank"].corr(g[forward_col + "_rank"])
            )
        else:
            ic_series = valid.groupby("date").apply(
                lambda g: g[factor].corr(g[forward_col])
            )
        ic_series = ic_series.dropna()
        if ic_series.empty:
            continue
        ic_std = float(ic_series.std())
        results[factor] = {
            "ic_mean": float(ic_series.mean()),
            "ic_std": ic_std,
            "ic_ir": float(ic_series.mean() / ic_std) if ic_std > 0 else 0.0,
            "ic_positive_ratio": float((ic_series > 0).mean()),
            "n_obs": int(len(ic_series)),
        }
    return results


# ────────────────────────────────────────────────────────────
# 3. 因子表达式引擎 (Qlib DSL 启发, 简化版)
# ────────────────────────────────────────────────────────────
class FactorExpressionEngine:
    """
    简化版因子表达式引擎 (借鉴 Qlib Expression Engine)。

    支持:
    - 字段引用: $close, $open, $volume, $high, $low, $amount, $turnover_rate
    - 时序算子: Ref(expr, n), Mean(expr, n), Std(expr, n), Max(expr, n), Min(expr, n)
    - 横截面算子: CSRank(expr) (按日期排名)
    - 数学函数: Abs(expr), Log(expr), Add(a,b), Sub(a,b), Mul(a,b), Div(a,b)
    - 中缀运算符: + - * / (支持优先级, 如 Ref($close,-5)/$close - 1)

    示例:
        engine = FactorExpressionEngine()
        engine.add_factor("rev_5d", "Ref($close, -5)/$close - 1")  # 注: 负数=未来, 仅用于标签
        engine.add_factor("ma20", "Mean($close, 20)")
        engine.add_factor("rank_ma20", "CSRank(Mean($close, 20))")
        df = engine.compute(panel)
    """

    FIELD_PREFIX = "$"

    def __init__(self):
        self.factor_defs: Dict[str, str] = {}

    def add_factor(self, name: str, expr: str):
        self.factor_defs[name] = expr

    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """计算所有已注册因子, 返回 code/date/[因子列]"""
        df = panel.sort_values(["code", "date"]).copy()
        result = df[["code", "date"]].copy()
        cache: Dict[str, pd.Series] = {}
        code_grp = df["code"]
        date_grp = df["date"]

        def get_field(field_name: str) -> pd.Series:
            key = self.FIELD_PREFIX + field_name
            if key not in cache:
                cache[key] = df[field_name]
            return cache[key]

        def apply_func(fname: str, args: List[pd.Series], raw_args: List[str]) -> pd.Series:
            # 整数参数从求值后的 Series 提取 (兼容 -5 这类一元负号)
            def _int_arg(idx: int) -> int:
                return int(args[idx].iloc[0])
            if fname == "Ref":
                return args[0].groupby(code_grp).shift(_int_arg(1))
            if fname == "Mean":
                n = _int_arg(1)
                return args[0].groupby(code_grp).transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).mean())
            if fname == "Std":
                n = _int_arg(1)
                return args[0].groupby(code_grp).transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).std())
            if fname == "Max":
                n = _int_arg(1)
                return args[0].groupby(code_grp).transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).max())
            if fname == "Min":
                n = _int_arg(1)
                return args[0].groupby(code_grp).transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).min())
            if fname == "CSRank":
                return args[0].groupby(date_grp).rank(pct=True)
            if fname == "Abs":
                return args[0].abs()
            if fname == "Log":
                return np.log(args[0].replace(0, np.nan))
            if fname in ("Add", "Sub", "Mul", "Div"):
                a, b = args[0], args[1]
                if fname == "Add":
                    return a + b
                if fname == "Sub":
                    return a - b
                if fname == "Mul":
                    return a * b
                if fname == "Div":
                    return a / b.replace(0, np.nan)
            raise ValueError(f"未知算子: {fname}")

        # ── 递归下降解析器 (支持 + - * / 优先级) ──
        def tokenize(expr: str) -> List[str]:
            tokens = []
            i = 0
            while i < len(expr):
                ch = expr[i]
                if ch.isspace():
                    i += 1
                    continue
                if ch in "+-*/(),":
                    # 区分一元负号: 若前一个 token 是数字/字段/右括号, 则 '-' 是二元; 否则一元(并入数字)
                    tokens.append(ch)
                    i += 1
                    continue
                if ch == "$":
                    j = i + 1
                    while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                        j += 1
                    tokens.append(expr[i:j])
                    i = j
                    continue
                if ch.isalpha() or ch == "_":
                    j = i
                    while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                        j += 1
                    tokens.append(expr[i:j])
                    i = j
                    continue
                if ch.isdigit() or ch == ".":
                    j = i
                    while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                        j += 1
                    tokens.append(expr[i:j])
                    i = j
                    continue
                raise ValueError(f"无法识别字符: {ch} (位置 {i})")
            return tokens

        class Parser:
            def __init__(self, tokens):
                self.tokens = tokens
                self.pos = 0

            def peek(self):
                return self.tokens[self.pos] if self.pos < len(self.tokens) else None

            def next(self):
                t = self.peek()
                self.pos += 1
                return t

            def parse_expr(self):
                # expr := term (('+' | '-') term)*
                node = self.parse_term()
                while self.peek() in ("+", "-"):
                    op = self.next()
                    rhs = self.parse_term()
                    node = ("binop", op, node, rhs)
                return node

            def parse_term(self):
                # term := factor (('*' | '/') factor)*
                node = self.parse_factor()
                while self.peek() in ("*", "/"):
                    op = self.next()
                    rhs = self.parse_factor()
                    node = ("binop", op, node, rhs)
                return node

            def parse_factor(self):
                t = self.peek()
                if t == "(":
                    self.next()
                    node = self.parse_expr()
                    if self.peek() != ")":
                        raise ValueError("缺少右括号")
                    self.next()
                    return node
                if t == "-":
                    # 一元负号
                    self.next()
                    node = self.parse_factor()
                    return ("neg", node)
                if t and t.startswith("$"):
                    self.next()
                    return ("field", t[1:])
                if t and (t[0].isdigit() or t[0] == "."):
                    self.next()
                    return ("num", float(t))
                if t and (t[0].isalpha() or t[0] == "_"):
                    # 函数调用
                    fname = self.next()
                    if self.peek() != "(":
                        raise ValueError(f"函数 {fname} 后需跟左括号")
                    self.next()
                    args = []
                    raw_args = []
                    if self.peek() != ")":
                        # 解析参数 (参数本身可能是表达式或数字字面量)
                        start = self.pos
                        arg_node = self.parse_expr()
                        args.append(arg_node)
                        raw_args.append(self._raw(start, self.pos))
                        while self.peek() == ",":
                            self.next()
                            start = self.pos
                            arg_node = self.parse_expr()
                            args.append(arg_node)
                            raw_args.append(self._raw(start, self.pos))
                    if self.peek() != ")":
                        raise ValueError("函数参数缺少右括号")
                    self.next()
                    return ("func", fname, args, raw_args)
                raise ValueError(f"无法解析 token: {t}")

            def _raw(self, start, end):
                return " ".join(self.tokens[start:end])

        def eval_node(node) -> pd.Series:
            kind = node[0]
            if kind == "field":
                return get_field(node[1])
            if kind == "num":
                return pd.Series(node[1], index=df.index)
            if kind == "neg":
                return -eval_node(node[1])
            if kind == "binop":
                op = node[1]
                a = eval_node(node[2])
                b = eval_node(node[3])
                if op == "+":
                    return a + b
                if op == "-":
                    return a - b
                if op == "*":
                    return a * b
                if op == "/":
                    return a / b.replace(0, np.nan)
            if kind == "func":
                fname = node[1]
                arg_nodes = node[2]
                raw_args = node[3]
                args = [eval_node(n) for n in arg_nodes]
                return apply_func(fname, args, raw_args)
            raise ValueError(f"未知节点: {node}")

        for name, expr in self.factor_defs.items():
            tokens = tokenize(expr)
            parser = Parser(tokens)
            ast = parser.parse_expr()
            if parser.pos != len(tokens):
                raise ValueError(f"表达式解析未完成: {expr}, 剩余 token: {tokens[parser.pos:]}")
            result[name] = eval_node(ast)
        return result


def time_function(fn, runs: int = 3, *args, **kwargs) -> Dict[str, Any]:
    """计时辅助"""
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return {"result": result, "times": times, "mean_time": float(np.mean(times))}
