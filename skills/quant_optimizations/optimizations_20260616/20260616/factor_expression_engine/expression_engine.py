"""
声明式因子表达式引擎（借鉴 Qlib / AKQuant 表达式引擎）
==========================================================

参考项目：
  - Microsoft Qlib (qlib/data/ops.py) 的 Operator Tree 架构
  - AKQuant (akfamily/akquant) 的 Polars 驱动因子表达式
  - WorldQuant Alpha101 公式范式

核心创新点：
  1. **声明式编程**：用户用公式字符串定义因子，引擎解析AST并执行
  2. **横/纵截面统一 API**：同一套算子同时支持时间序列与截面
  3. **pandas DataFrame 友好**：输入输出保持 DataFrame 格式，无缝接入现有 pipeline

算子清单（参考 Qlib ops.py，简化版）：
  - 基础算子：Abs, Log, Sign, Sqrt, SignedPower
  - 横截面算子：Rank, Scale, Quantile  （按日分组）
  - 时序算子：Ref, Delta, Mean, Std, Sum, Ts_Max, Ts_Min, Ts_ArgMax, Ts_ArgMin
  - 条件算子：If, Gt, Lt, Eq, And, Or
  - 配对算子：Corr, Cov

约定：
  - 算子参数：
      * 第一个位置参数：data（Series）
      * 后续参数：算子特定参数（窗口长度、阈值等）
  - 表达式中 $field 引用 DataFrame 的字段
  - 算子嵌套：Mean($close, 5) → Ref($close, 5) 的 5 日均值
"""

from __future__ import annotations
import ast
import re
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ───────────────────────── 全局上下文（线程安全） ─────────────────────────
# 存储当前 DataFrame，供横截面算子使用
_CONTEXT = threading.local()


@contextmanager
def expression_context(df: pd.DataFrame):
    """设置表达式引擎的全局上下文 DataFrame（横截面算子按此 df 的 date 分组）"""
    _CONTEXT.df = df
    try:
        yield df
    finally:
        if hasattr(_CONTEXT, "df"):
            del _CONTEXT.df


def _current_df() -> Optional[pd.DataFrame]:
    return getattr(_CONTEXT, "df", None)


# ───────────────────────── 算子实现 ─────────────────────────

SeriesLike = Union[pd.Series, pd.DataFrame]


def _as_series(x: SeriesLike, col: Optional[str] = None) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if col is None:
            raise ValueError("DataFrame 输入必须指定 col 参数")
        return x[col]
    return x


# ── 基础算子（按元素） ──
def op_abs(x: SeriesLike) -> pd.Series:
    return _as_series(x).abs()


def op_log(x: SeriesLike) -> pd.Series:
    return _as_series(x).apply(lambda v: np.log(v) if v and v > 0 else np.nan)


def op_sign(x: SeriesLike) -> pd.Series:
    s = _as_series(x)
    out = pd.Series(np.sign(s), index=s.index, dtype=float)
    out = out.fillna(0)
    return out


def op_sqrt(x: SeriesLike) -> pd.Series:
    return _as_series(x).apply(lambda v: np.sqrt(v) if v and v >= 0 else np.nan)


def op_signedpower(x: SeriesLike, exp: float) -> pd.Series:
    """sign(x) * |x|^exp  (WorldQuant Alpha101 风格)"""
    s = _as_series(x)
    return s.apply(lambda v: (1 if v > 0 else (-1 if v < 0 else 0)) * (abs(v) ** exp))


# ── 简单二元算子 ──
def op_add(a, b) -> pd.Series:
    return _as_series(a) + _as_series(b)


def op_sub(a, b) -> pd.Series:
    return _as_series(a) - _as_series(b)


def op_mul(a, b) -> pd.Series:
    return _as_series(a) * _as_series(b)


def op_div(a, b) -> pd.Series:
    return _as_series(a) / _as_series(b).replace(0, np.nan)


# ── 横截面算子（按日分组） ──
def op_rank(x: SeriesLike) -> pd.Series:
    """横截面 rank（按日分组，返回百分比排名）"""
    s = _as_series(x)
    df = _current_df()
    if df is None or "date" not in df.columns:
        return s.rank(pct=True)
    return df.assign(_v=s).groupby("date")["_v"].rank(pct=True)


def op_scale(x: SeriesLike) -> pd.Series:
    """横截面缩放：x / sum(|x|)，按日"""
    s = _as_series(x)
    df = _current_df()
    if df is None or "date" not in df.columns:
        denom = s.abs().sum()
        return s / denom if denom else s
    abs_s = s.abs()
    sums = df.assign(_v=abs_s).groupby("date")["_v"].transform("sum")
    return s / sums.replace(0, np.nan)


def op_quantile(x: SeriesLike, q: float) -> pd.Series:
    """横截面分位数分组（>q 标记为 1）"""
    s = _as_series(x)
    df = _current_df()
    if df is None or "date" not in df.columns:
        return (s.rank(pct=True) > q).astype(int)
    pct = df.assign(_v=s).groupby("date")["_v"].rank(pct=True)
    return (pct > q).astype(int)


# ── 时序算子（按 code 分组） ──
def op_ref(x: SeriesLike, n: int) -> pd.Series:
    """Ref(x, n) = x.shift(n)"""
    s = _as_series(x)
    df = _current_df()
    if df is None or "code" not in df.columns:
        return s.shift(n)
    return s.groupby(df["code"]).shift(n)


def op_delta(x: SeriesLike, n: int) -> pd.Series:
    """Delta(x, n) = x - x.shift(n)"""
    s = _as_series(x)
    df = _current_df()
    if df is None or "code" not in df.columns:
        return s - s.shift(n)
    shifted = s.groupby(df["code"]).shift(n)
    return s - shifted


def op_mean(x: SeriesLike, n: int) -> pd.Series:
    s = _as_series(x)
    df = _current_df()
    if df is None or "code" not in df.columns:
        return s.rolling(n, min_periods=1).mean()
    return s.groupby(df["code"]).transform(lambda v: v.rolling(n, min_periods=1).mean())


def op_std(x: SeriesLike, n: int) -> pd.Series:
    s = _as_series(x)
    df = _current_df()
    if df is None or "code" not in df.columns:
        return s.rolling(n, min_periods=2).std()
    return s.groupby(df["code"]).transform(lambda v: v.rolling(n, min_periods=2).std())


def op_sum(x: SeriesLike, n: int) -> pd.Series:
    s = _as_series(x)
    df = _current_df()
    if df is None or "code" not in df.columns:
        return s.rolling(n, min_periods=1).sum()
    return s.groupby(df["code"]).transform(lambda v: v.rolling(n, min_periods=1).sum())


def op_ts_max(x: SeriesLike, n: int) -> pd.Series:
    s = _as_series(x)
    df = _current_df()
    if df is None or "code" not in df.columns:
        return s.rolling(n, min_periods=1).max()
    return s.groupby(df["code"]).transform(lambda v: v.rolling(n, min_periods=1).max())


def op_ts_min(x: SeriesLike, n: int) -> pd.Series:
    s = _as_series(x)
    df = _current_df()
    if df is None or "code" not in df.columns:
        return s.rolling(n, min_periods=1).min()
    return s.groupby(df["code"]).transform(lambda v: v.rolling(n, min_periods=1).min())


def op_ts_rank(x: SeriesLike, n: int) -> pd.Series:
    """Ts_Rank(x, n) = rolling rank of last value (percentile)"""
    s = _as_series(x)

    def _ts_rank(v: pd.Series) -> pd.Series:
        return v.rolling(n, min_periods=2).apply(
            lambda w: float(pd.Series(w).rank(pct=True).iloc[-1]) if len(w) > 1 else np.nan,
            raw=False
        )

    df = _current_df()
    if df is None or "code" not in df.columns:
        return _ts_rank(s)
    return s.groupby(df["code"]).transform(_ts_rank)


def op_ts_argmax(x: SeriesLike, n: int) -> pd.Series:
    s = _as_series(x)

    def _argmax(v: pd.Series) -> pd.Series:
        return v.rolling(n, min_periods=1).apply(
            lambda w: float(np.argmax(w)) if len(w) > 0 else 0.0, raw=True
        )

    df = _current_df()
    if df is None or "code" not in df.columns:
        return _argmax(s)
    return s.groupby(df["code"]).transform(_argmax)


def op_ts_argmin(x: SeriesLike, n: int) -> pd.Series:
    s = _as_series(x)

    def _argmin(v: pd.Series) -> pd.Series:
        return v.rolling(n, min_periods=1).apply(
            lambda w: float(np.argmin(w)) if len(w) > 0 else 0.0, raw=True
        )

    df = _current_df()
    if df is None or "code" not in df.columns:
        return _argmin(s)
    return s.groupby(df["code"]).transform(_argmin)


# ── 配对算子 ──
def op_corr(a: SeriesLike, b: SeriesLike, n: int) -> pd.Series:
    a_s = _as_series(a)
    b_s = _as_series(b)
    df = _current_df()
    if df is None or "code" not in df.columns:
        return a_s.rolling(n, min_periods=2).corr(b_s)
    res = pd.Series(index=a_s.index, dtype=float)
    for code, idx in a_s.groupby(df["code"]).groups.items():
        a_g = a_s.loc[idx]
        b_g = b_s.loc[idx]
        res.loc[idx] = a_g.rolling(n, min_periods=2).corr(b_g).values
    return res


def op_cov(a: SeriesLike, b: SeriesLike, n: int) -> pd.Series:
    a_s = _as_series(a)
    b_s = _as_series(b)
    df = _current_df()
    if df is None or "code" not in df.columns:
        return a_s.rolling(n, min_periods=2).cov(b_s)
    res = pd.Series(index=a_s.index, dtype=float)
    for code, idx in a_s.groupby(df["code"]).groups.items():
        a_g = a_s.loc[idx]
        b_g = b_s.loc[idx]
        res.loc[idx] = a_g.rolling(n, min_periods=2).cov(b_g).values
    return res


# ── 条件算子 ──
def op_if(cond: SeriesLike, a, b) -> pd.Series:
    c = _as_series(cond).astype(bool)
    a_s = _as_series(a) if isinstance(a, (pd.Series, pd.DataFrame)) else pd.Series(a, index=c.index)
    b_s = _as_series(b) if isinstance(b, (pd.Series, pd.DataFrame)) else pd.Series(b, index=c.index)
    return pd.Series(np.where(c, a_s, b_s), index=c.index)


def op_gt(a, b) -> pd.Series:
    return (_as_series(a) > _as_series(b)).astype(int)


def op_lt(a, b) -> pd.Series:
    return (_as_series(a) < _as_series(b)).astype(int)


def op_eq(a, b) -> pd.Series:
    return (_as_series(a) == _as_series(b)).astype(int)


def op_and(a, b) -> pd.Series:
    return ((_as_series(a) > 0) & (_as_series(b) > 0)).astype(int)


def op_or(a, b) -> pd.Series:
    return ((_as_series(a) > 0) | (_as_series(b) > 0)).astype(int)


# ───────────────────────── 算子注册表 ─────────────────────────

OPERATORS: Dict[str, Callable] = {
    # 基础
    "Abs": op_abs, "Log": op_log, "Sign": op_sign, "Sqrt": op_sqrt,
    "SignedPower": op_signedpower,
    # 二元
    "Add": op_add, "Sub": op_sub, "Mul": op_mul, "Div": op_div,
    # 横截面
    "Rank": op_rank, "Scale": op_scale, "Quantile": op_quantile,
    # 时序
    "Ref": op_ref, "Delta": op_delta,
    "Mean": op_mean, "Std": op_std, "Sum": op_sum,
    "Ts_Max": op_ts_max, "Ts_Min": op_ts_min,
    "Ts_Rank": op_ts_rank, "Ts_ArgMax": op_ts_argmax, "Ts_ArgMin": op_ts_argmin,
    # 配对
    "Corr": op_corr, "Cov": op_cov,
    # 条件
    "If": op_if, "Gt": op_gt, "Lt": op_lt, "Eq": op_eq, "And": op_and, "Or": op_or,
}


# ───────────────────────── 表达式解析器 ─────────────────────────

def _preprocess(expr: str) -> str:
    """将 $field 替换为安全的 Python 标识符 VAR_<field>"""
    return re.sub(r"\$([a-zA-Z_]\w*)", r"VAR_\1", expr)


def _resolve_arg(node: ast.AST, context: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _resolve_arg(node.operand, context)
        return -inner
    if isinstance(node, ast.BinOp):
        left = _resolve_arg(node.left, context)
        right = _resolve_arg(node.right, context)
        if isinstance(node.op, ast.Add): return _as_series(left) + _as_series(right)
        if isinstance(node.op, ast.Sub): return _as_series(left) - _as_series(right)
        if isinstance(node.op, ast.Mult): return _as_series(left) * _as_series(right)
        if isinstance(node.op, ast.Div): return _as_series(left) / _as_series(right).replace(0, np.nan)
    if isinstance(node, ast.Name):
        name = node.id
        if name.startswith("VAR_"):
            return context[name[4:]]
        if name in OPERATORS:
            return name
        raise ValueError(f"未识别的名字: {name}")
    if isinstance(node, ast.Call):
        return _eval_call(node, context)
    raise ValueError(f"不支持的语法节点: {ast.dump(node)}")


def _eval_call(node: ast.Call, context: Dict[str, pd.Series]) -> pd.Series:
    if not isinstance(node.func, ast.Name):
        raise ValueError("只支持简单函数调用形式")
    op_name = node.func.id
    if op_name not in OPERATORS:
        raise ValueError(f"未知算子: {op_name}")
    args = [_resolve_arg(a, context) for a in node.args]
    kwargs = {kw.arg: _resolve_arg(kw.value, context) for kw in node.keywords}
    return OPERATORS[op_name](*args, **kwargs)


def parse_and_eval(expr: str, df: pd.DataFrame, fields: Optional[List[str]] = None) -> pd.Series:
    """
    解析并执行公式字符串，返回结果 Series

    参数:
        expr: 公式字符串
        df: 输入 DataFrame（必须含 code, date）
        fields: 可选字段白名单（不指定则使用 df 全部列）
    """
    if fields is None:
        fields = [c for c in df.columns if c not in ("code", "date")]
    context = {f: df[f] for f in fields}
    pre = _preprocess(expr)
    tree = ast.parse(pre, mode="eval")
    with expression_context(df):
        return _eval_call(tree.body, context)


# ───────────────────────── 预置因子 ─────────────────────────

ALPHA101_FORMULAS = {
    "Alpha_006": "Mul(-1.0, Corr($open, $volume, 10))",
    "Alpha_012": "Mul(Sign(Delta($volume, 1)), Mul(-1.0, Delta($close, 1)))",
    "Alpha_033": "Rank(Mul(-1.0, Sub(1.0, Div($open, $close))))",
    "Reversal_5d": "Mul(-1.0, Delta($close, 5))",
    "Momentum_20d": "Sub($close, Ref($close, 20))",
    "Volatility_20d": "Std($returns, 20)",
    "MeanRev_5d": "Sub(Ref($close, 5), $close)",
}


def _ensure_returns_column(df: pd.DataFrame) -> pd.DataFrame:
    if "returns" not in df.columns and "close" in df.columns:
        df = df.copy()
        df["returns"] = df.groupby("code")["close"].pct_change()
    return df


def compute_alpha(name: str, df: pd.DataFrame) -> pd.Series:
    """计算单个预置因子（自动注入 returns 列）"""
    df = _ensure_returns_column(df)
    expr = ALPHA101_FORMULAS[name]
    return parse_and_eval(expr, df)


# ───────────────────────── 自检 ─────────────────────────

def _self_test():
    """Correctness test on synthetic data"""
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    codes = [f"{i:06d}.SZ" for i in range(10)]
    rows = []
    for d in dates:
        for c in codes:
            px = 10 + np.cumsum(np.random.normal(0, 0.02, 1))[0]
            rows.append({"date": d, "code": c, "open": px, "high": px * 1.01,
                         "low": px * 0.99, "close": px, "volume": int(1e6 + np.random.rand() * 1e5)})
    df = pd.DataFrame(rows)
    df = _ensure_returns_column(df)

    tests = [
        ("Mean($close, 3)", lambda d: d.groupby("code")["close"].transform(lambda v: v.rolling(3, min_periods=1).mean())),
        ("Sub($close, Ref($close, 1))", lambda d: d["close"] - d.groupby("code")["close"].shift(1)),
        ("Abs(Log($close))", lambda d: d["close"].apply(lambda v: abs(np.log(v)))),
        ("Sign(Sub($close, Ref($close, 1)))",
         lambda d: (d["close"] - d.groupby("code")["close"].shift(1)).apply(
             lambda v: 0 if pd.isna(v) or v == 0 else (1 if v > 0 else -1))),
        ("Rank($close)", lambda d: d.groupby("date")["close"].rank(pct=True)),
        ("Ref($close, 2)", lambda d: d.groupby("code")["close"].shift(2)),
        ("Delta($close, 5)", lambda d: d["close"] - d.groupby("code")["close"].shift(5)),
        ("Add($close, $open)", lambda d: d["close"] + d["open"]),
        ("Mul(2.0, $close)", lambda d: 2.0 * d["close"]),
        ("Gt($close, $open)", lambda d: (d["close"] > d["open"]).astype(int)),
        ("If(Gt($close, $open), 1.0, 0.0)", lambda d: pd.Series(np.where(d["close"] > d["open"], 1.0, 0.0), index=d.index)),
    ]
    results = []
    for expr, ref_fn in tests:
        try:
            got = parse_and_eval(expr, df)
            expected = ref_fn(df)
            got_v = got.fillna(-9999).reset_index(drop=True)
            exp_v = expected.fillna(-9999).reset_index(drop=True)
            max_diff = float((got_v - exp_v).abs().max())
            ok = max_diff < 1e-6
            results.append((expr, ok, max_diff))
        except Exception as e:
            results.append((expr, False, str(e)))
    return results


if __name__ == "__main__":
    print("=== Factor Expression Engine self-test ===")
    for expr, ok, info in _self_test():
        print(f"  {'[OK]' if ok else '[FAIL]'} {expr:42s} diff={info}")
    print("\n=== Alpha101 demo on synthetic data ===")
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    codes = [f"{i:06d}.SZ" for i in range(10)]
    rows = []
    for d in dates:
        for c in codes:
            px = 10 + np.cumsum(np.random.normal(0, 0.02, 1))[0]
            rows.append({"date": d, "code": c, "close": px, "open": px, "volume": int(1e6)})
    df = pd.DataFrame(rows)
    for name in ALPHA101_FORMULAS.keys():
        try:
            s = compute_alpha(name, df)
            print(f"  {name:18s}  non-null={s.notna().sum():3d}  "
                  f"mean={float(s.mean()):+.4f}  std={float(s.std()):.4f}")
        except Exception as e:
            print(f"  {name:18s}  ERROR: {e}")