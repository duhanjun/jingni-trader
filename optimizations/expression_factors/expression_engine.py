"""
表达式因子引擎（借鉴 Microsoft Qlib Alpha158 设计思想）

核心优化点：
1. 表达式引擎：支持 $close, Ref, Mean, Std, Max, Min, Slope, Corr 等算子
   用户可用字符串公式定义因子，无需硬编码
2. Alpha158 风格因子库：158 个因子，分 6 大类
   - K 线基础因子 (9)
   - 静态价格因子 (4)
   - 趋势类因子 (25)
   - 波动类因子 (30)
   - 极值位置类因子 (15)
   - 价量统计类因子 (45) + 其他 (30)
3. 向量化计算：pandas groupby + rolling，避免逐日 Python 循环

借鉴来源：Microsoft Qlib (https://github.com/microsoft/qlib)
  - 表达式引擎设计：$close, Ref($close, 5), Mean($close, 20)
  - Alpha158 因子库：158 个量价因子
  - DataHandlerLP 配置即代码的设计哲学

对照 jingni-trader 现有 factor-engine 的改进：
  - 现有：~12 个硬编码因子，compute_a_share_factors 中逐个手写
  - 优化：表达式引擎 + 158 因子库，新增因子只需写公式字符串
"""
from typing import Dict, List, Optional, Callable, Any
import re
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("expression-factor-engine")


# ============================================================
# 表达式引擎：算子注册表
# ============================================================

class ExpressionEngine:
    """
    轻量表达式引擎

    支持的语法：
      $close, $open, $high, $low, $volume       # 字段引用
      Ref($close, 5)                              # 5 日前的 close
      Mean($close, 20)                            # 20 日均值
      Std($close, 20)                             # 20 日标准差
      Max($high, 20)                              # 20 日最高
      Min($low, 20)                               # 20 日最低
      Slope($close, 20)                           # 20 日线性回归斜率
      Rsquare($close, 20)                         # 20 日 R²
      Resi($close, 20)                            # 20 日残差
      Corr($close, $volume, 20)                   # 20 日相关系数
      Quantile($close, 20, 0.8)                   # 20 日 80% 分位数
      IdxMax($high, 20)                           # 20 日最高价位置
      IdxMin($low, 20)                            # 20 日最低价位置
      Rank($close)                                # 横截面排名
      Delta($close, 5)                            # close - Ref(close, 5)
      算术运算: +, -, *, /, (, )
    """

    # 字段映射
    FIELD_MAP = {
        "$close": "close",
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$volume": "volume",
        "$vwap": "vwap",
        "$amount": "amount",
        "$turnover": "turnover_rate",
    }

    def __init__(self):
        self.operators: Dict[str, Callable] = {}
        self._register_default_operators()

    def _register_default_operators(self):
        """注册默认算子"""
        self.operators["Ref"] = self._op_ref
        self.operators["Mean"] = self._op_mean
        self.operators["Sum"] = self._op_sum
        self.operators["Std"] = self._op_std
        self.operators["Var"] = self._op_var
        self.operators["Max"] = self._op_max
        self.operators["Min"] = self._op_min
        self.operators["Slope"] = self._op_slope
        self.operators["Rsquare"] = self._op_rsquare
        self.operators["Resi"] = self._op_resi
        self.operators["Corr"] = self._op_corr
        self.operators["Cov"] = self._op_cov
        self.operators["Quantile"] = self._op_quantile
        self.operators["IdxMax"] = self._op_idxmax
        self.operators["IdxMin"] = self._op_idxmin
        self.operators["Rank"] = self._op_rank
        self.operators["Delta"] = self._op_delta
        self.operators["WMA"] = self._op_wma
        self.operators["EMA"] = self._op_ema

    # ---------- 算子实现（向量化，基于 groupby + rolling） ----------

    @staticmethod
    def _group_rolling(series: pd.Series, window: int, func: str, min_periods: int = 1) -> pd.Series:
        """按 code 分组做 rolling"""
        return series.groupby(level="code", group_keys=False).transform(
            lambda x: getattr(x.rolling(window=window, min_periods=min_periods), func)()
        )

    def _op_ref(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        return df[field].groupby(level="code", group_keys=False).shift(n)

    def _op_mean(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        return self._group_rolling(df[field], n, "mean")

    def _op_sum(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        return self._group_rolling(df[field], n, "sum")

    def _op_std(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        return self._group_rolling(df[field], n, "std")

    def _op_var(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        return self._group_rolling(df[field], n, "var")

    def _op_max(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        return self._group_rolling(df[field], n, "max")

    def _op_min(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        return self._group_rolling(df[field], n, "min")

    def _op_slope(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        """线性回归斜率（向量化）"""
        s = df[field]
        x = np.arange(n)

        def _slope(y):
            if len(y) < n:
                return np.nan
            yv = y.values[-n:]
            if np.isnan(yv).any():
                return np.nan
            x_mean = x.mean()
            y_mean = yv.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom == 0:
                return np.nan
            return ((x - x_mean) * (yv - y_mean)).sum() / denom

        return s.groupby(level="code", group_keys=False).rolling(n, min_periods=n).apply(
            _slope, raw=False
        ).reset_index(level=0, drop=True)

    def _op_rsquare(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        """R² 决定系数"""
        s = df[field]
        x = np.arange(n)
        x_mean = x.mean()
        ss_xx = ((x - x_mean) ** 2).sum()

        def _r2(y):
            if len(y) < n:
                return np.nan
            yv = y.values[-n:]
            if np.isnan(yv).any():
                return np.nan
            y_mean = yv.mean()
            ss_yy = ((yv - y_mean) ** 2).sum()
            if ss_yy == 0:
                return np.nan
            ss_xy = ((x - x_mean) * (yv - y_mean)).sum()
            return (ss_xy ** 2) / (ss_xx * ss_yy)

        return s.groupby(level="code", group_keys=False).rolling(n, min_periods=n).apply(
            _r2, raw=False
        ).reset_index(level=0, drop=True)

    def _op_resi(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        """线性回归残差（最后一个点）"""
        s = df[field]
        x = np.arange(n)
        x_mean = x.mean()

        def _resi(y):
            if len(y) < n:
                return np.nan
            yv = y.values[-n:]
            if np.isnan(yv).any():
                return np.nan
            y_mean = yv.mean()
            ss_xx = ((x - x_mean) ** 2).sum()
            if ss_xx == 0:
                return np.nan
            slope = ((x - x_mean) * (yv - y_mean)).sum() / ss_xx
            intercept = y_mean - slope * x_mean
            pred_last = slope * x[-1] + intercept
            return yv[-1] - pred_last

        return s.groupby(level="code", group_keys=False).rolling(n, min_periods=n).apply(
            _resi, raw=False
        ).reset_index(level=0, drop=True)

    def _op_corr(self, df: pd.DataFrame, f1: str, f2: str, n: int) -> pd.Series:
        """滚动相关系数"""
        s1, s2 = df[f1], df[f2]

        def _corr(pair):
            a, b = pair[:, 0], pair[:, 1]
            if len(a) < n:
                return np.nan
            a, b = a[-n:], b[-n:]
            if np.isnan(a).any() or np.isnan(b).any():
                return np.nan
            sa, sb = a.std(), b.std()
            if sa == 0 or sb == 0:
                return np.nan
            return np.corrcoef(a, b)[0, 1]

        combined = pd.concat([s1, s2], axis=1)
        return combined.groupby(level="code", group_keys=False).rolling(n, min_periods=n).apply(
            _corr, raw=True
        ).reset_index(level=0, drop=True).iloc[:, 0] if hasattr(
            combined.groupby(level="code", group_keys=False).rolling(n, min_periods=n).apply(
                _corr, raw=True
            ), "iloc"
        ) else combined.groupby(level="code", group_keys=False).rolling(n, min_periods=n).apply(
            _corr, raw=True
        ).reset_index(level=0, drop=True)

    def _op_cov(self, df: pd.DataFrame, f1: str, f2: str, n: int) -> pd.Series:
        s1, s2 = df[f1], df[f2]

        def _cov(pair):
            a, b = pair[:, 0], pair[:, 1]
            if len(a) < n:
                return np.nan
            a, b = a[-n:], b[-n:]
            if np.isnan(a).any() or np.isnan(b).any():
                return np.nan
            return np.cov(a, b)[0, 1]

        combined = pd.concat([s1, s2], axis=1)
        result = combined.groupby(level="code", group_keys=False).rolling(n, min_periods=n).apply(
            _cov, raw=True
        )
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        else:
            result = result.reset_index(level=0, drop=True)
        return result

    def _op_quantile(self, df: pd.DataFrame, field: str, n: int, q: float) -> pd.Series:
        return df[field].groupby(level="code", group_keys=False).transform(
            lambda x: x.rolling(n, min_periods=max(n // 2, 1)).quantile(q)
        )

    def _op_idxmax(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        def _idxmax(x):
            r = x.rolling(n, min_periods=n)
            return r.apply(lambda y: np.argmax(y), raw=True)
        return df[field].groupby(level="code", group_keys=False).transform(_idxmax)

    def _op_idxmin(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        def _idxmin(x):
            r = x.rolling(n, min_periods=n)
            return r.apply(lambda y: np.argmin(y), raw=True)
        return df[field].groupby(level="code", group_keys=False).transform(_idxmin)

    def _op_rank(self, df: pd.DataFrame, field: str) -> pd.Series:
        """横截面排名（按 date 分组）"""
        return df.groupby(level="date", group_keys=False)[field].rank(pct=True)

    def _op_delta(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        return df[field].groupby(level="code", group_keys=False).diff(n)

    def _op_wma(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        """加权移动平均"""
        weights = np.arange(1, n + 1, dtype=float)

        def _wma(x):
            if len(x) < n:
                return np.nan
            yv = x.values[-n:]
            if np.isnan(yv).any():
                return np.nan
            return (yv * weights).sum() / weights.sum()

        return df[field].groupby(level="code", group_keys=False).rolling(n, min_periods=n).apply(
            _wma, raw=False
        ).reset_index(level=0, drop=True)

    def _op_ema(self, df: pd.DataFrame, field: str, n: int) -> pd.Series:
        """指数移动平均"""
        alpha = 2.0 / (n + 1)
        return df[field].groupby(level="code", group_keys=False).transform(
            lambda x: x.ewm(alpha=alpha, min_periods=n).mean()
        )

    # ---------- 表达式解析与求值 ----------

    def evaluate(self, expr: str, df: pd.DataFrame) -> pd.Series:
        """
        解析并求值表达式

        参数:
            expr: 表达式字符串，如 "Mean($close, 20) / $close"
            df: 数据，必须以 (code, date) 为 MultiIndex，含 close/open/high/low/volume 列

        返回:
            pd.Series，索引与 df 相同
        """
        # 替换字段引用
        for token, col in self.FIELD_MAP.items():
            expr = expr.replace(token, col)
        return self._eval_expr(expr, df)

    def _eval_expr(self, expr: str, df: pd.DataFrame) -> pd.Series:
        """递归下降解析器"""
        parser = _ExprParser(expr, self, df)
        return parser.parse()


class _ExprParser:
    """简单的递归下降表达式解析器"""

    def __init__(self, expr: str, engine: ExpressionEngine, df: pd.DataFrame):
        self.expr = expr.replace(" ", "")
        self.pos = 0
        self.engine = engine
        self.df = df

    def parse(self) -> pd.Series:
        result = self._parse_expr()
        return result

    def _peek(self) -> str:
        if self.pos >= len(self.expr):
            return ""
        return self.expr[self.pos]

    def _parse_expr(self) -> Any:
        left = self._parse_term()
        while self._peek() in ("+", "-"):
            op = self._peek()
            self.pos += 1
            right = self._parse_term()
            if op == "+":
                left = left + right
            else:
                left = left - right
        return left

    def _parse_term(self) -> Any:
        left = self._parse_factor()
        while self._peek() in ("*", "/"):
            op = self._peek()
            self.pos += 1
            right = self._parse_factor()
            if op == "*":
                left = left * right
            else:
                # 除零保护
                left = left / right.replace(0, np.nan)
        return left

    def _parse_factor(self) -> Any:
        ch = self._peek()
        if ch == "(":
            self.pos += 1
            val = self._parse_expr()
            if self._peek() == ")":
                self.pos += 1
            return val
        if ch == "-":
            self.pos += 1
            return -self._parse_factor()

        # 数字字面量
        if ch.isdigit() or ch == ".":
            return self._parse_number()

        # 标识符（字段名或函数名）
        if ch.isalpha() or ch == "_":
            return self._parse_identifier()

        raise ValueError(f"无法解析字符: {ch} 在位置 {self.pos}")

    def _parse_number(self) -> float:
        start = self.pos
        while self.pos < len(self.expr) and (self.expr[self.pos].isdigit() or self.expr[self.pos] == "."):
            self.pos += 1
        return float(self.expr[start:self.pos])

    def _parse_identifier(self) -> Any:
        start = self.pos
        while self.pos < len(self.expr) and (self.expr[self.pos].isalnum() or self.expr[self.pos] == "_"):
            self.pos += 1
        name = self.expr[start:self.pos]

        # 函数调用
        if self._peek() == "(":
            self.pos += 1
            args = self._parse_args()
            if self._peek() == ")":
                self.pos += 1
            return self._call_function(name, args)

        # 字段引用
        if name in self.df.columns:
            return self.df[name]
        if name == "close":
            return self.df["close"]
        raise ValueError(f"未知字段或函数: {name}")

    def _parse_args(self) -> List[Any]:
        args = []
        if self._peek() == ")":
            return args
        args.append(self._parse_expr())
        while self._peek() == ",":
            self.pos += 1
            args.append(self._parse_expr())
        return args

    def _call_function(self, name: str, args: List[Any]) -> pd.Series:
        op = self.engine.operators.get(name)
        if op is None:
            raise ValueError(f"未知算子: {name}")
        # args 可能是 pd.Series 或 float
        # 算子签名约定：op(df, field_or_series, n, [q])
        # 这里做适配
        if name in ("Corr", "Cov"):
            # 两个 Series 参数 + n
            f1 = args[0]
            f2 = args[1]
            n = int(args[1]) if isinstance(args[1], (int, float)) else int(args[2])
            # 判断参数顺序
            if isinstance(args[1], (int, float)):
                # Corr(field1, field2, n) 但 field2 是数字？不可能
                raise ValueError(f"{name} 需要两个 Series 参数")
            return op(self.df, self._series_to_field(f1), self._series_to_field(f2), n)
        elif name == "Quantile":
            field = self._series_to_field(args[0])
            n = int(args[1])
            q = float(args[2])
            return op(self.df, field, n, q)
        elif name == "Rank":
            field = self._series_to_field(args[0])
            return op(self.df, field)
        else:
            # 单 Series + n
            field = self._series_to_field(args[0])
            n = int(args[1])
            return op(self.df, field, n)

    @staticmethod
    def _series_to_field(s: Any) -> str:
        """从 Series 提取字段名（取 name）"""
        if isinstance(s, pd.Series):
            return s.name
        return str(s)


# ============================================================
# Alpha158 风格因子库
# ============================================================

class Alpha158FactorLibrary:
    """
    Alpha158 风格因子库（借鉴 Qlib Alpha158）

    6 大类 158 个因子：
      1. K 线基础因子 (9)
      2. 静态价格因子 (4)
      3. 趋势类因子 (25)
      4. 波动类因子 (30)
      5. 极值位置类因子 (15)
      6. 价量统计类因子 (45) + 其他 (30)
    """

    def __init__(self):
        self.engine = ExpressionEngine()
        self.factor_configs: Dict[str, str] = {}
        self._build_alpha158()

    def _build_alpha158(self):
        """构建 Alpha158 因子配置（表达式字符串）"""
        cfg = {}

        # ---- 1. K 线基础因子 (9) ----
        cfg["KMID"] = "($close - $open) / $open"
        cfg["KLEN"] = "($high - $low) / $open"
        cfg["KMID2"] = "($close - $open) / ($high - $low)"
        cfg["KUP"] = "($high - Max($open, $close)) / $open"
        cfg["KUP2"] = "($high - Max($open, $close)) / ($high - $low)"
        cfg["KLOW"] = "(Min($open, $close) - $low) / $open"
        cfg["KLOW2"] = "(Min($open, $close) - $low) / ($high - $low)"
        cfg["KSFT"] = "(2 * $close - $high - $low) / $open"
        cfg["KSFT2"] = "(2 * $close - $high - $low) / ($high - $low)"

        # ---- 2. 静态价格因子 (4) ----
        cfg["OPEN0"] = "$open / $close"
        cfg["HIGH0"] = "$high / $close"
        cfg["LOW0"] = "$low / $close"
        cfg["VWAP0"] = "$amount / $volume"

        # ---- 3. 趋势类因子 (25) = 5 类 × 5 周期 ----
        for n in (5, 10, 20, 30, 60):
            cfg[f"ROC{n}"] = f"Ref($close, {n}) / $close"
            cfg[f"MA{n}"] = f"Mean($close, {n}) / $close"
            cfg[f"BETA{n}"] = f"Slope($close, {n}) / $close"
            cfg[f"RSQR{n}"] = f"Rsquare($close, {n})"
            cfg[f"RESI{n}"] = f"Resi($close, {n}) / $close"

        # ---- 4. 波动类因子 (30) = 6 类 × 5 周期 ----
        for n in (5, 10, 20, 30, 60):
            cfg[f"STD{n}"] = f"Std($close, {n}) / $close"
            cfg[f"MAX{n}"] = f"Max($high, {n}) / $close"
            cfg[f"MIN{n}"] = f"Min($low, {n}) / $close"
            cfg[f"QTLU{n}"] = f"Quantile($close, {n}, 0.8) / $close"
            cfg[f"QTLD{n}"] = f"Quantile($close, {n}, 0.2) / $close"
            cfg[f"RSV{n}"] = f"($close - Min($low, {n})) / (Max($high, {n}) - Min($low, {n}))"

        # ---- 5. 极值位置类因子 (15) = 3 类 × 5 周期 ----
        for n in (5, 10, 20, 30, 60):
            cfg[f"IMAX{n}"] = f"IdxMax($high, {n}) / {n}"
            cfg[f"IMIN{n}"] = f"IdxMin($low, {n}) / {n}"
            cfg[f"IMXD{n}"] = f"(IdxMax($high, {n}) - IdxMin($low, {n})) / {n}"

        # ---- 6. 价量统计类因子 (45) = 9 类 × 5 周期 ----
        for n in (5, 10, 20, 30, 60):
            cfg[f"CORR{n}"] = f"Corr($close, $volume, {n})"
            cfg[f"CORD{n}"] = f"Corr($close / Ref($close, 1), $volume / Ref($volume, 1) + 1, {n})"
            cfg[f"CNTP{n}"] = f"Mean($close > Ref($close, 1), {n})"
            cfg[f"CNTN{n}"] = f"Mean($close < Ref($close, 1), {n})"
            cfg[f"CNTD{n}"] = f"Mean($close > Ref($close, 1), {n}) - Mean($close < Ref($close, 1), {n})"
            cfg[f"SUMP{n}"] = f"Sum(Max($close - Ref($close, 1), 0), {n}) / Sum(Abs($close - Ref($close, 1)), {n})"
            cfg[f"SUMN{n}"] = f"Sum(Max(Ref($close, 1) - $close, 0), {n}) / Sum(Abs($close - Ref($close, 1)), {n})"
            cfg[f"SUMD{n}"] = f"Sum(Max($close - Ref($close, 1), 0), {n}) - Sum(Max(Ref($close, 1) - $close, 0), {n})"
            cfg[f"VMA{n}"] = f"Mean($volume, {n}) / $volume"

        # ---- 7. 额外补充因子 (30) ----
        for n in (5, 10, 20, 30, 60):
            cfg[f"WMA{n}"] = f"WMA($close, {n}) / $close"
            cfg[f"EMA{n}"] = f"EMA($close, {n}) / $close"
            cfg[f"VSTD{n}"] = f"Std($volume, {n}) / Mean($volume, {n})"
            cfg[f"VRAP{n}"] = f"Std($close, {n}) / Mean($close, {n})"
            cfg[f"DROC{n}"] = f"Delta($close, {n}) / Ref($close, {n})"
            cfg[f"VROC{n}"] = f"Delta($volume, {n}) / Ref($volume, {n})"

        self.factor_configs = cfg
        return cfg

    def list_factors(self) -> List[str]:
        return list(self.factor_configs.keys())

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算全部 Alpha158 因子

        参数:
            data: 日线数据，含 date, code, open, high, low, close, volume, amount

        返回:
            DataFrame，索引为 (code, date)，列为各因子
        """
        # 准备数据：以 (code, date) 为 MultiIndex
        if not isinstance(data.index, pd.MultiIndex):
            df = data.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["code", "date"]).sort_index()
        else:
            df = data.copy().sort_index()

        # 确保 volume/amount 存在
        if "volume" not in df.columns and "vol" in df.columns:
            df["volume"] = df["vol"]
        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]
        if "turnover_rate" not in df.columns:
            df["turnover_rate"] = 0.0

        result = pd.DataFrame(index=df.index)
        n_factors = len(self.factor_configs)
        computed = 0
        failed = []

        for name, expr in self.factor_configs.items():
            try:
                series = self.engine.evaluate(expr, df)
                if isinstance(series, pd.Series):
                    result[name] = series
                    computed += 1
                else:
                    failed.append(name)
            except Exception as e:
                logger.warning(f"因子 {name} 计算失败: {e}")
                failed.append(name)

        logger.info(
            f"Alpha158 因子计算完成: 成功 {computed}/{n_factors}，失败 {len(failed)}"
        )
        if failed:
            logger.debug(f"失败因子: {failed}")

        return result.reset_index()

    def compute_factor(self, data: pd.DataFrame, factor_name: str) -> pd.Series:
        """计算单个因子"""
        if factor_name not in self.factor_configs:
            raise ValueError(f"未知因子: {factor_name}，可用: {list(self.factor_configs.keys())[:10]}...")

        if not isinstance(data.index, pd.MultiIndex):
            df = data.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["code", "date"]).sort_index()
        else:
            df = data.copy().sort_index()

        if "volume" not in df.columns and "vol" in df.columns:
            df["volume"] = df["vol"]
        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]

        return self.engine.evaluate(self.factor_configs[factor_name], df)


# ============================================================
# 向量化 IC 分析（替代 factor-engine 中逐日循环）
# ============================================================

class VectorizedICAnalysis:
    """
    向量化 IC 分析

    对照 factor-engine.py 中的 ic_analysis：
      原实现：for dt in dates: 逐日计算 spearmanr
      优化：groupby('date') + corr(method='spearman') 一次性计算
    """

    @staticmethod
    def calc_ic_series(
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_col: str,
        forward_col: str = "ret_forward_5d",
        method: str = "spearman",
    ) -> pd.Series:
        """
        计算因子 IC 时间序列（向量化）

        参数:
            factor_df: 含 date, code, factor_col
            forward_returns: 含 date, code, forward_col
            factor_col: 因子列名
            forward_col: 未来收益列名
            method: 'spearman' 或 'pearson'

        返回:
            pd.Series，索引为 date，值为 IC
        """
        merged = factor_df[["date", "code", factor_col]].merge(
            forward_returns[["date", "code", forward_col]],
            on=["date", "code"],
            how="inner",
        )
        # 过滤无效值
        merged = merged.dropna(subset=[factor_col, forward_col])

        # 按日期分组计算相关系数
        def _cross_corr(group):
            if len(group) < 10:
                return np.nan
            if method == "spearman":
                return group[factor_col].corr(group[forward_col], method="spearman")
            else:
                return group[factor_col].corr(group[forward_col], method="pearson")

        ic_series = merged.groupby("date").apply(_cross_corr)
        ic_series = ic_series.dropna()
        return ic_series

    @staticmethod
    def calc_ic_summary(ic_series: pd.Series) -> Dict[str, float]:
        """IC 统计摘要"""
        if ic_series.empty or len(ic_series) < 2:
            return {}
        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
        ic_t = float(ic_mean / (ic_std / np.sqrt(len(ic_series)))) if ic_std > 0 else 0.0
        return {
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "ic_ir": round(ic_ir, 4),
            "ic_t_stat": round(ic_t, 4),
            "ic_positive_ratio": round(float((ic_series > 0).mean()), 4),
            "n_periods": int(len(ic_series)),
        }
