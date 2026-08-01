"""
基于 pandas-ta 的因子计算器适配器（数据驱动注册表模式）

镜像 talib_calculator.py 的注册表设计，但使用 pandas-ta 后端。
因子名称与 TA-Lib 版本一一对应，便于无缝切换后端。

覆盖范围：
  - 重叠指标 Overlap Studies
  - 动量指标 Momentum Indicators
  - 成交量指标 Volume Indicators
  - 波动率指标 Volatility Indicators
  - 价格变换 Price Transform
  - 统计函数 Statistic Functions
  - 周期指标 Cycle Indicators（pandas-ta 不支持，返回 NaN）
  - K线形态识别 Pattern Recognition（61 个 CDL 函数）
    其中 14 个常用形态使用纯 pandas/numpy 实现，
    其余 47 个记录告警并返回 NaN。

注意：
  - pandas-ta 不支持希尔伯特变换系列（HT_*），返回 NaN。
  - pandas-ta 不支持 SAREXT、HT_TRENDLINE，返回 NaN。
  - pandas-ta 多输出函数返回 DataFrame（而非 TA-Lib 的元组），
    通过 output_key 列名前缀定位目标列。
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from ..base.base_factor_calculator import BaseFactorCalculator


# ------------------------------------------------------------------
# 注册表定义
# ------------------------------------------------------------------
# 每条记录字段说明：
#   func        : pandas-ta 函数名（字符串，懒加载 getattr(ta, func)）；
#                 None 表示 pandas-ta 不支持该因子（返回 NaN）
#   inputs      : 所需输入列名顺序列表（取自 OHLCV）
#   kwargs      : 调用 pandas-ta 时的默认关键字参数（pandas-ta 风格）
#   output_key  : 多输出函数的列名前缀（如 "MACD_" → 匹配 MACD_12_26_9）；
#                 None 表示单输出（返回 Series）
#   category    : 因子类别 (overlap/momentum/volume/volatility/price/
#                 statistic/cycle/pattern)
#   description : 中文名称/描述
#   direction   : 方向 (1=看多, -1=看空, 0=中性)
#   params      : 对外暴露的参数（默认与 kwargs 等价，便于前端展示）
#   pattern_name: CDL 形态原始函数名（仅 pattern 类别使用）
PANDAS_TA_FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _register(name: str,
              func: Optional[str],
              inputs: List[str],
              kwargs: Dict[str, Any],
              category: str,
              description: str,
              direction: int = 0,
              output_key: Optional[str] = None,
              params: Optional[Dict[str, Any]] = None,
              pattern_name: Optional[str] = None) -> None:
    """注册一个 pandas-ta 因子。"""
    PANDAS_TA_FUNCTION_REGISTRY[name] = {
        "func": func,
        "inputs": list(inputs),
        "kwargs": dict(kwargs),
        "output_key": output_key,
        "category": category,
        "description": description,
        "direction": direction,
        "params": dict(params) if params is not None else dict(kwargs),
        "pattern_name": pattern_name,
    }


# ==================================================================
# 1. 重叠指标 Overlap Studies
# ==================================================================
_register("dema", "dema", ["close"], {"length": 30},
          "overlap", "双指数移动平均线", 0)
_register("ema", "ema", ["close"], {"length": 30},
          "overlap", "指数移动平均线", 0)
# HT_TRENDLINE: pandas-ta 不支持希尔伯特变换
_register("ht_trendline", None, ["close"], {},
          "overlap", "希尔伯特瞬时趋势线（pandas-ta 不支持）", 0)
_register("kama", "kama", ["close"], {"length": 30},
          "overlap", "考夫曼自适应移动平均线", 0)
# MA: pandas-ta 无 matype 参数，matype=0 等价于 SMA
_register("ma", "sma", ["close"], {"length": 30},
          "overlap", "移动平均线（SMA，pandas-ta 无 matype）", 0)
_register("midpoint", "midpoint", ["close"], {"length": 14},
          "overlap", "区间中点", 0)
_register("midprice", "midprice", ["high", "low"], {"length": 14},
          "overlap", "区间中间价", 0)
_register("sar", "psar", ["high", "low"],
          {"af": 0.02, "max_af": 0.2},
          "overlap", "抛物线SAR（停损转向点）", 0,
          output_key="PSAR_")
# SAREXT: pandas-ta 不支持
_register("sarext", None, ["high", "low"], {},
          "overlap", "扩展抛物线SAR（pandas-ta 不支持）", 0)
_register("t3", "t3", ["close"], {"length": 5, "vfactor": 0.7},
          "overlap", "T3三重平滑移动平均线", 0)
_register("tema", "tema", ["close"], {"length": 30},
          "overlap", "三重指数移动平均线", 0)
_register("trima", "trima", ["close"], {"length": 30},
          "overlap", "三角移动平均线", 0)
_register("wma", "wma", ["close"], {"length": 30},
          "overlap", "加权移动平均线", 0)

# 布林带（多输出，统一缓存）
_register("bbands_upper", "bbands", ["close"],
          {"length": 20, "std": 2},
          "overlap", "布林带上轨", 1, output_key="BBU_")
_register("bbands_middle", "bbands", ["close"],
          {"length": 20, "std": 2},
          "overlap", "布林带中轨", 0, output_key="BBM_")
_register("bbands_lower", "bbands", ["close"],
          {"length": 20, "std": 2},
          "overlap", "布林带下轨", -1, output_key="BBL_")

# 向后兼容别名（保留旧命名）
_register("bollinger_upper", "bbands", ["close"],
          {"length": 20, "std": 2},
          "overlap", "布林带上轨（兼容别名）", 1, output_key="BBU_")
_register("bollinger_middle", "bbands", ["close"],
          {"length": 20, "std": 2},
          "overlap", "布林带中轨（兼容别名）", 0, output_key="BBM_")
_register("bollinger_lower", "bbands", ["close"],
          {"length": 20, "std": 2},
          "overlap", "布林带下轨（兼容别名）", -1, output_key="BBL_")

_register("ma_5", "sma", ["close"], {"length": 5},
          "overlap", "5日简单移动平均线", 0)
_register("ma_10", "sma", ["close"], {"length": 10},
          "overlap", "10日简单移动平均线", 0)
_register("ma_20", "sma", ["close"], {"length": 20},
          "overlap", "20日简单移动平均线", 0)
_register("ma_30", "sma", ["close"], {"length": 30},
          "overlap", "30日简单移动平均线", 0)
_register("ma_60", "sma", ["close"], {"length": 60},
          "overlap", "60日简单移动平均线", 0)
_register("ma_120", "sma", ["close"], {"length": 120},
          "overlap", "120日简单移动平均线", 0)
_register("ma_250", "sma", ["close"], {"length": 250},
          "overlap", "250日简单移动平均线（年线）", 0)
_register("ema_5", "ema", ["close"], {"length": 5},
          "overlap", "5日指数移动平均线", 0)
_register("ema_10", "ema", ["close"], {"length": 10},
          "overlap", "10日指数移动平均线", 0)
_register("ema_20", "ema", ["close"], {"length": 20},
          "overlap", "20日指数移动平均线", 0)
_register("ema_60", "ema", ["close"], {"length": 60},
          "overlap", "60日指数移动平均线", 0)


# ==================================================================
# 2. 动量指标 Momentum Indicators
# ==================================================================
# ADX 返回 DataFrame(ADX_, DMP_, DMN_)，取 ADX 列
_register("adx", "adx", ["high", "low", "close"], {"length": 14},
          "momentum", "平均趋向指数（趋势强度）", 0, output_key="ADX_")
_register("adxr", "adxr", ["high", "low", "close"], {"length": 14},
          "momentum", "ADX评级", 0)
_register("apo", "apo", ["close"],
          {"fast": 12, "slow": 26},
          "momentum", "绝对价格震荡指标", 0)
_register("aroon_up", "aroon", ["high", "low"], {"length": 14},
          "momentum", "Aroon上轨", 1, output_key="AROONU_")
_register("aroon_down", "aroon", ["high", "low"], {"length": 14},
          "momentum", "Aroon下轨", -1, output_key="AROOND_")
_register("aroonosc", "aroonosc", ["high", "low"], {"length": 14},
          "momentum", "Aroon震荡指标", 0)
_register("bop", "bop", ["open", "high", "low", "close"], {},
          "momentum", "力量指标（开盘-最高-最低-收盘）", 0)
_register("cci", "cci", ["high", "low", "close"], {"length": 14},
          "momentum", "商品通道指标", 0)
_register("cmo", "cmo", ["close"], {"length": 14},
          "momentum", "钱德动量摆动指标", 0)
_register("dx", "dx", ["high", "low", "close"], {"length": 14},
          "momentum", "趋向指标DX", 0)
# MACD（多输出：MACD_, MACDh_, MACDs_）
_register("macd", "macd", ["close"],
          {"fast": 12, "slow": 26, "signal": 9},
          "momentum", "MACD差离值", 0, output_key="MACD_")
_register("macd_signal", "macd", ["close"],
          {"fast": 12, "slow": 26, "signal": 9},
          "momentum", "MACD信号线", 0, output_key="MACDs_")
_register("macd_hist", "macd", ["close"],
          {"fast": 12, "slow": 26, "signal": 9},
          "momentum", "MACD柱状图", 0, output_key="MACDh_")
_register("mfi", "mfi", ["high", "low", "close", "volume"],
          {"length": 14}, "momentum", "资金流量指标", 0)
_register("minus_di", "minus_di", ["high", "low", "close"],
          {"length": 14}, "momentum", "负向趋向指标DI", -1)
_register("minus_dm", "minus_dm", ["high", "low"],
          {"length": 14}, "momentum", "负向趋向指标DM", -1)
_register("mom", "mom", ["close"], {"length": 10},
          "momentum", "动量指标", 0)
_register("plus_di", "plus_di", ["high", "low", "close"],
          {"length": 14}, "momentum", "正向趋向指标DI", 1)
_register("plus_dm", "plus_dm", ["high", "low"],
          {"length": 14}, "momentum", "正向趋向指标DM", 1)
_register("ppo", "ppo", ["close"],
          {"fast": 12, "slow": 26},
          "momentum", "价格百分比震荡指标", 0)
_register("roc", "roc", ["close"], {"length": 10},
          "momentum", "变动率指标", 0)
_register("rocp", "rocp", ["close"], {"length": 10},
          "momentum", "百分比变动率", 0)
_register("rocr", "rocr", ["close"], {"length": 10},
          "momentum", "比率变动率", 0)
_register("rocr100", "rocr100", ["close"], {"length": 10},
          "momentum", "ROCR100指标", 0)
_register("rsi", "rsi", ["close"], {"length": 14},
          "momentum", "相对强弱指标", 0)
# STOCH（多输出：STOCHk_, STOCHd_）
_register("stoch_k", "stoch", ["high", "low", "close"],
          {"k": 5, "d": 3, "smooth_k": 3},
          "momentum", "随机指标K线", 0, output_key="STOCHk_")
_register("stoch_d", "stoch", ["high", "low", "close"],
          {"k": 5, "d": 3, "smooth_k": 3},
          "momentum", "随机指标D线", 0, output_key="STOCHd_")
# STOCHF（快速随机）
_register("stochf_k", "stochf", ["high", "low", "close"],
          {"k": 5, "d": 3},
          "momentum", "快速随机指标K线", 0, output_key="STOCHFk_")
_register("stochf_d", "stochf", ["high", "low", "close"],
          {"k": 5, "d": 3},
          "momentum", "快速随机指标D线", 0, output_key="STOCHFd_")
# STOCHRSI（多输出：STOCHRSIk_, STOCHRSId_）
_register("stochrsi_k", "stochrsi", ["close"],
          {"length": 14, "rsi_length": 14, "k": 5, "d": 3},
          "momentum", "随机RSI指标K线", 0, output_key="STOCHRSIk_")
_register("stochrsi_d", "stochrsi", ["close"],
          {"length": 14, "rsi_length": 14, "k": 5, "d": 3},
          "momentum", "随机RSI指标D线", 0, output_key="STOCHRSId_")
_register("trix", "trix", ["close"], {"length": 30},
          "momentum", "三重平滑均线动量", 0)
_register("ultosc", "uosc", ["high", "low", "close"],
          {"fast": 7, "medium": 14, "slow": 28},
          "momentum", "终极震荡指标", 0)
_register("willr", "willr", ["high", "low", "close"], {"length": 14},
          "momentum", "威廉指标", 0)


# ==================================================================
# 3. 成交量指标 Volume Indicators
# ==================================================================
_register("ad", "ad", ["high", "low", "close", "volume"], {},
          "volume", "累积/派发线", 0)
_register("adosc", "adosc", ["high", "low", "close", "volume"],
          {"fast": 3, "slow": 10},
          "volume", "累积/派发震荡指标", 0)
_register("obv", "obv", ["close", "volume"], {},
          "volume", "能量潮指标", 0)


# ==================================================================
# 4. 波动率指标 Volatility Indicators
# ==================================================================
_register("atr", "atr", ["high", "low", "close"], {"length": 14},
          "volatility", "平均真实波幅", 0)
_register("natr", "natr", ["high", "low", "close"], {"length": 14},
          "volatility", "归一化平均真实波幅", 0)
_register("trange", "true_range", ["high", "low", "close"], {},
          "volatility", "真实波幅", 0)


# ==================================================================
# 5. 价格变换 Price Transform
# ==================================================================
_register("avgprice", "avgprice", ["open", "high", "low", "close"], {},
          "price", "平均价格", 0)
_register("medprice", "median_price", ["high", "low"], {},
          "price", "中间价格", 0)
_register("typprice", "typical_price", ["high", "low", "close"], {},
          "price", "典型价格", 0)
_register("wclprice", "weighted_close", ["high", "low", "close"], {},
          "price", "加权收盘价", 0)


# ==================================================================
# 6. 统计函数 Statistic Functions
# ==================================================================
_register("beta", "beta", ["high", "low"], {"length": 5},
          "statistic", "贝塔系数", 0)
_register("correl", "correlation", ["high", "low"], {"length": 30},
          "statistic", "皮尔逊相关系数", 0)
_register("linearreg", "linreg", ["close"], {"length": 14},
          "statistic", "线性回归", 0)
_register("linearreg_angle", "linreg_angle", ["close"],
          {"length": 14}, "statistic", "线性回归角度", 0)
_register("linearreg_intercept", "linreg_intercept", ["close"],
          {"length": 14}, "statistic", "线性回归截距", 0)
_register("linearreg_slope", "linreg_slope", ["close"],
          {"length": 14}, "statistic", "线性回归斜率", 0)
_register("stddev", "stdev", ["close"], {"length": 5},
          "statistic", "标准差", 0)
_register("tsf", "tsf", ["close"], {"length": 14},
          "statistic", "时间序列预测", 0)
_register("var", "variance", ["close"], {"length": 5},
          "statistic", "方差", 0)


# ==================================================================
# 7. 周期指标 Cycle Indicators（pandas-ta 不支持希尔伯特变换）
# ==================================================================
_register("ht_dcperiod", None, ["close"], {},
          "cycle", "希尔伯特变换-主导周期（pandas-ta 不支持）", 0)
_register("ht_dcphase", None, ["close"], {},
          "cycle", "希尔伯特变换-主导相位（pandas-ta 不支持）", 0)
_register("ht_phasor_inphase", None, ["close"], {},
          "cycle", "希尔伯特变换-同相分量（pandas-ta 不支持）", 0)
_register("ht_phasor_quadrature", None, ["close"], {},
          "cycle", "希尔伯特变换-正交分量（pandas-ta 不支持）", 0)
_register("ht_sine", None, ["close"], {},
          "cycle", "希尔伯特变换-正弦（pandas-ta 不支持）", 0)
_register("ht_leadsine", None, ["close"], {},
          "cycle", "希尔伯特变换-超前正弦（pandas-ta 不支持）", 0)


# ==================================================================
# 8. K线形态识别 Pattern Recognition（全部 61 个 CDL 函数）
# ==================================================================
# pandas-ta 不支持 CDL 系列形态识别。
# 其中 14 个最常用形态使用纯 pandas/numpy 实现（见下方 _PATTERN_HANDLERS）。
# 其余 47 个形态记录告警并返回 NaN。
#
# 所有 CDL 函数均接收 (open, high, low, close)，返回整数数组：
#   正值（如 100/200）= 看多形态，负值（如 -100/-200）= 看空形态，0 = 无形态
# direction 字段表示该形态的典型方向：1=看多形态, -1=看空形态, 0=中性/可多可空
_CDL_PATTERN_META: Dict[str, Tuple[str, int]] = {
    # 看空形态 direction=-1
    "CDL2CROWS":              ("两只乌鸦", -1),
    "CDL3BLACKCROWS":         ("三只乌鸦", -1),
    "CDLADVANCEBLOCK":        ("大敌当前", -1),
    "CDLDARKCLOUDCOVER":      ("乌云压顶", -1),
    "CDLEVENINGDOJISTAR":     ("黄昏十字星", -1),
    "CDLEVENINGSTAR":         ("黄昏之星", -1),
    "CDLGRAVESTONEDOJI":      ("墓碑十字", -1),
    "CDLHANGINGMAN":          ("上吊线", -1),
    "CDLIDENTICAL3CROWS":     ("相同的三只乌鸦", -1),
    "CDLINNECK":              ("颈内线", -1),
    "CDLONNECK":              ("颈上线", -1),
    "CDLSHOOTINGSTAR":        ("流星", -1),
    "CDLSTALLEDPATTERN":      ("停顿形态", -1),
    "CDLTHRUSTING":           ("插入形态", -1),
    "CDLUPSIDEGAP2CROWS":     ("向上跳空两只乌鸦", -1),
    # 看多形态 direction=1
    "CDL3STARSINSOUTH":       ("南方三星", 1),
    "CDL3WHITESOLDIERS":      ("三个白兵", 1),
    "CDLCONCEALBABYSWALL":    ("藏婴吞没", 1),
    "CDLHAMMER":              ("锤子线", 1),
    "CDLHOMINGPIGEON":        ("家鸽", 1),
    "CDLINVERTEDHAMMER":      ("倒锤子线", 1),
    "CDLLADDERBOTTOM":        ("梯底", 1),
    "CDLMATCHINGLOW":         ("相同低价", 1),
    "CDLMORNINGDOJISTAR":     ("早晨十字星", 1),
    "CDLMORNINGSTAR":         ("早晨之星", 1),
    "CDLPIERCING":            ("刺穿形态", 1),
    "CDLSTICKSANDWICH":       ("棍子三明治", 1),
    "CDLTAKURI":              ("探水竿", 1),
    "CDLUNIQUE3RIVER":        ("奇特三河床", 1),
    # 中性/可多可空形态 direction=0
    "CDL3INSIDE":             ("三内部上涨/下跌", 0),
    "CDL3LINESTRIKE":         ("三线打击", 0),
    "CDL3OUTSIDE":            ("三外部上涨/下跌", 0),
    "CDLABANDONEDBABY":       ("弃婴", 0),
    "CDLBELTHOLD":            ("捉腰带", 0),
    "CDLBREAKAWAY":           ("脱离", 0),
    "CDLCLOSINGMARUBOZU":     ("收盘秃头", 0),
    "CDLCOUNTERATTACK":       ("反击", 0),
    "CDLDOJI":                ("十字星", 0),
    "CDLDOJISTAR":            ("十字星之星", 0),
    "CDLDRAGONFLYDOJI":       ("蜻蜓十字", 0),
    "CDLENGULFING":           ("吞没形态", 0),
    "CDLGAPSIDESIDEWHITE":    ("跳空并列白线", 0),
    "CDLHARAMI":              ("母子线", 0),
    "CDLHARAMICROSS":         ("十字孕线", 0),
    "CDLHIGHWAVE":            ("长腿十字", 0),
    "CDLHIKKAKE":             ("Hikkake形态", 0),
    "CDLHIKKAKEMOD":          ("修正Hikkake形态", 0),
    "CDLKICKING":             ("反冲形态", 0),
    "CDLKICKINGBYLENGTH":     ("反冲形态(按长度)", 0),
    "CDLLONGLEGGEDDOJI":      ("长腿十字星", 0),
    "CDLLONGLINE":            ("长蜡烛", 0),
    "CDLMARUBOZU":            ("秃头光脚", 0),
    "CDLMATHOLD":             ("银河", 0),
    "CDLRICKSHAWMAN":         ("黄包车夫", 0),
    "CDLRISEFALL3METHODS":    ("上升/下降三法", 0),
    "CDLSEPARATINGLINES":     ("分离线", 0),
    "CDLSHORTLINE":           ("短蜡烛", 0),
    "CDLSPINNINGTOP":         ("纺锤顶", 0),
    "CDLTASUKIGAP":           ("跳空缺口(补缺)", 0),
    "CDLTRISTAR":             ("三星", 0),
    "CDLXSIDEGAP3STARS":      ("跳空三星", 0),
}

for _cdl_func, (_cdl_desc, _cdl_dir) in _CDL_PATTERN_META.items():
    _register(
        name=_cdl_func.lower(),
        func=None,  # pandas-ta 不支持 CDL 系列
        inputs=["open", "high", "low", "close"],
        kwargs={},
        category="pattern",
        description=_cdl_desc,
        direction=_cdl_dir,
        output_key=None,
        params={},
        pattern_name=_cdl_func,
    )

# 清理循环变量，避免污染模块命名空间
del _cdl_func, _cdl_desc, _cdl_dir


# ==================================================================
# 纯 pandas/numpy 实现的 K线形态识别（14 个最常用形态）
# ==================================================================
# 返回 float 数组：正值=看多形态, 负值=看空形态, 0=无形态
# 与 TA-Lib CDL 函数返回值约定一致


def _pat_doji(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """十字星：实体极小（不超过全幅的10%）。"""
    body = np.abs(c - o)
    rng = h - l
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(rng > 0, body / rng, 1.0)
    return np.where(ratio <= 0.1, 100.0, 0.0)


def _pat_hammer(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """锤子线：长下影线，小实体在顶部，极短上影线。"""
    body = c - o
    abs_body = np.abs(body)
    rng = h - l
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        body_ratio = np.where(rng > 0, abs_body / rng, 1.0)
    cond = (
        (rng > 0)
        & (lower_shadow >= 2 * abs_body)
        & (upper_shadow <= abs_body * 0.3)
        & (body_ratio <= 0.3)
    )
    return np.where(cond, 100.0, 0.0)


def _pat_hanging_man(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """上吊线：形态同锤子线（方向语义不同，由 direction 字段区分）。"""
    return _pat_hammer(o, h, l, c)


def _pat_engulfing(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """吞没形态：当前实体完全包裹前一根实体。"""
    prev_body = o[1:] - c[1:]
    curr_body = c[1:] - o[1:]
    prev_abs = np.abs(prev_body)
    curr_abs = np.abs(curr_body)
    # 看涨吞没：前阴后阳，当前实体 > 前实体
    bullish = (prev_body < 0) & (curr_body > 0) & (curr_abs > prev_abs) & \
              (c[1:] > o[:-1]) & (o[1:] < c[:-1])
    # 看跌吞没：前阳后阴，当前实体 > 前实体
    bearish = (prev_body > 0) & (curr_body < 0) & (curr_abs > prev_abs) & \
              (c[1:] < o[:-1]) & (o[1:] > c[:-1])
    result = np.zeros(len(o), dtype=float)
    result[1:] = np.where(bullish, 100.0, np.where(bearish, -100.0, 0.0))
    return result


def _pat_morning_star(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """早晨之星：三根K线看涨反转（大阴-小星-大阳收复第一根中点以上）。"""
    n = len(o)
    result = np.zeros(n, dtype=float)
    if n < 3:
        return result
    body0 = c[:-2] - o[:-2]
    body2 = c[2:] - o[2:]
    abs_body2 = np.abs(body2)
    avg_body = (np.abs(c[:-2] - o[:-2]) + np.abs(c[1:-1] - o[1:-1])) / 2.0
    # 第一根：大阴线
    cond0 = body0 < 0
    # 第二根：小实体（星线），向下跳空
    abs_body1 = np.abs(c[1:-1] - o[1:-1])
    cond1 = (abs_body1 <= avg_body * 0.3) & (np.maximum(o[1:-1], c[1:-1]) < np.minimum(o[:-2], c[:-2]))
    # 第三根：大阳线，收盘超过第一根中点
    mid0 = (o[:-2] + c[:-2]) / 2.0
    cond2 = (body2 > 0) & (abs_body2 > avg_body) & (c[2:] > mid0)
    cond = cond0 & cond1 & cond2
    result[2:] = np.where(cond, 100.0, 0.0)
    return result


def _pat_evening_star(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """黄昏之星：三根K线看跌反转（大阳-小星-大阴跌破第一根中点以下）。"""
    n = len(o)
    result = np.zeros(n, dtype=float)
    if n < 3:
        return result
    body0 = c[:-2] - o[:-2]
    body2 = c[2:] - o[2:]
    abs_body2 = np.abs(body2)
    avg_body = (np.abs(c[:-2] - o[:-2]) + np.abs(c[1:-1] - o[1:-1])) / 2.0
    # 第一根：大阳线
    cond0 = body0 > 0
    # 第二根：小实体（星线），向上跳空
    abs_body1 = np.abs(c[1:-1] - o[1:-1])
    cond1 = (abs_body1 <= avg_body * 0.3) & (np.minimum(o[1:-1], c[1:-1]) > np.maximum(o[:-2], c[:-2]))
    # 第三根：大阴线，收盘低于第一根中点
    mid0 = (o[:-2] + c[:-2]) / 2.0
    cond2 = (body2 < 0) & (abs_body2 > avg_body) & (c[2:] < mid0)
    cond = cond0 & cond1 & cond2
    result[2:] = np.where(cond, -100.0, 0.0)
    return result


def _pat_piercing(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """刺穿形态：前阴后阳，开盘低于前低，收盘超过前中点但低于前收。"""
    n = len(o)
    result = np.zeros(n, dtype=float)
    if n < 2:
        return result
    prev_body = c[:-1] - o[:-1]
    curr_body = c[1:] - o[1:]
    prev_mid = (o[:-1] + c[:-1]) / 2.0
    cond = (
        (prev_body < 0)  # 前阴
        & (curr_body > 0)  # 后阳
        & (o[1:] < l[:-1])  # 开盘低于前低
        & (c[1:] > prev_mid)  # 收盘超过前中点
        & (c[1:] < c[:-1])  # 收盘低于前收
    )
    result[1:] = np.where(cond, 100.0, 0.0)
    return result


def _pat_dark_cloud_cover(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """乌云压顶：前阳后阴，开盘高于前高，收盘低于前中点但高于前开。"""
    n = len(o)
    result = np.zeros(n, dtype=float)
    if n < 2:
        return result
    prev_body = c[:-1] - o[:-1]
    curr_body = c[1:] - o[1:]
    prev_mid = (o[:-1] + c[:-1]) / 2.0
    cond = (
        (prev_body > 0)  # 前阳
        & (curr_body < 0)  # 后阴
        & (o[1:] > h[:-1])  # 开盘高于前高
        & (c[1:] < prev_mid)  # 收盘低于前中点
        & (c[1:] > o[:-1])  # 收盘高于前开
    )
    result[1:] = np.where(cond, -100.0, 0.0)
    return result


def _pat_harami(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """母子线（孕线）：当前实体完全在前根实体范围内。"""
    n = len(o)
    result = np.zeros(n, dtype=float)
    if n < 2:
        return result
    prev_body = c[:-1] - o[:-1]
    curr_body = c[1:] - o[1:]
    prev_high = np.maximum(o[:-1], c[:-1])
    prev_low = np.minimum(o[:-1], c[:-1])
    curr_high = np.maximum(o[1:], c[1:])
    curr_low = np.minimum(o[1:], c[1:])
    inside = (curr_high <= prev_high) & (curr_low >= prev_low) & \
             (np.abs(curr_body) < np.abs(prev_body))
    # 看涨孕线：前阴后阳
    bullish = inside & (prev_body < 0) & (curr_body > 0)
    # 看跌孕线：前阳后阴
    bearish = inside & (prev_body > 0) & (curr_body < 0)
    result[1:] = np.where(bullish, 100.0, np.where(bearish, -100.0, 0.0))
    return result


def _pat_shooting_star(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """流星：长上影线，小实体在底部，极短下影线。"""
    body = c - o
    abs_body = np.abs(body)
    rng = h - l
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        body_ratio = np.where(rng > 0, abs_body / rng, 1.0)
    cond = (
        (rng > 0)
        & (upper_shadow >= 2 * abs_body)
        & (lower_shadow <= abs_body * 0.3)
        & (body_ratio <= 0.3)
    )
    return np.where(cond, -100.0, 0.0)


def _pat_inverted_hammer(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """倒锤子线：形态同流星（方向语义不同，由 direction 字段区分）。"""
    # 形态判定与流星相同，但返回正值（看多）
    body = c - o
    abs_body = np.abs(body)
    rng = h - l
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        body_ratio = np.where(rng > 0, abs_body / rng, 1.0)
    cond = (
        (rng > 0)
        & (upper_shadow >= 2 * abs_body)
        & (lower_shadow <= abs_body * 0.3)
        & (body_ratio <= 0.3)
    )
    return np.where(cond, 100.0, 0.0)


def _pat_3_white_soldiers(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """三个白兵：连续三根阳线，每根收盘创新高，每根开盘在前根实体范围内。"""
    n = len(o)
    result = np.zeros(n, dtype=float)
    if n < 3:
        return result
    b0 = c[:-2] - o[:-2]
    b1 = c[1:-1] - o[1:-1]
    b2 = c[2:] - o[2:]
    all_bull = (b0 > 0) & (b1 > 0) & (b2 > 0)
    # 每根收盘高于前根收盘
    higher_close = (c[1:-1] > c[:-2]) & (c[2:] > c[1:-1])
    # 每根开盘在前根实体范围内
    open_in_range = (o[1:-1] >= np.minimum(o[:-2], c[:-2])) & \
                    (o[1:-1] <= np.maximum(o[:-2], c[:-2])) & \
                    (o[2:] >= np.minimum(o[1:-1], c[1:-1])) & \
                    (o[2:] <= np.maximum(o[1:-1], c[1:-1]))
    cond = all_bull & higher_close & open_in_range
    result[2:] = np.where(cond, 100.0, 0.0)
    return result


def _pat_3_black_crows(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """三只乌鸦：连续三根阴线，每根收盘创新低，每根开盘在前根实体范围内。"""
    n = len(o)
    result = np.zeros(n, dtype=float)
    if n < 3:
        return result
    b0 = c[:-2] - o[:-2]
    b1 = c[1:-1] - o[1:-1]
    b2 = c[2:] - o[2:]
    all_bear = (b0 < 0) & (b1 < 0) & (b2 < 0)
    # 每根收盘低于前根收盘
    lower_close = (c[1:-1] < c[:-2]) & (c[2:] < c[1:-1])
    # 每根开盘在前根实体范围内
    open_in_range = (o[1:-1] >= np.minimum(o[:-2], c[:-2])) & \
                    (o[1:-1] <= np.maximum(o[:-2], c[:-2])) & \
                    (o[2:] >= np.minimum(o[1:-1], c[1:-1])) & \
                    (o[2:] <= np.maximum(o[1:-1], c[1:-1]))
    cond = all_bear & lower_close & open_in_range
    result[2:] = np.where(cond, -100.0, 0.0)
    return result


def _pat_spinning_top(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """纺锤顶：小实体，上下影线都较长。"""
    body = np.abs(c - o)
    rng = h - l
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    with np.errstate(divide='ignore', invalid='ignore'):
        body_ratio = np.where(rng > 0, body / rng, 1.0)
    cond = (
        (rng > 0)
        & (body_ratio <= 0.3)
        & (lower_shadow > body)
        & (upper_shadow > body)
    )
    return np.where(cond, 100.0, 0.0)


# 注册已实现的形态处理器
_PATTERN_HANDLERS: Dict[str, Any] = {
    "CDLDOJI":             _pat_doji,
    "CDLHAMMER":           _pat_hammer,
    "CDLHANGINGMAN":       _pat_hanging_man,
    "CDLENGULFING":        _pat_engulfing,
    "CDLMORNINGSTAR":      _pat_morning_star,
    "CDLEVENINGSTAR":      _pat_evening_star,
    "CDLPIERCING":         _pat_piercing,
    "CDLDARKCLOUDCOVER":   _pat_dark_cloud_cover,
    "CDLHARAMI":           _pat_harami,
    "CDLSHOOTINGSTAR":     _pat_shooting_star,
    "CDLINVERTEDHAMMER":   _pat_inverted_hammer,
    "CDL3WHITESOLDIERS":   _pat_3_white_soldiers,
    "CDL3BLACKCROWS":      _pat_3_black_crows,
    "CDLSPINNINGTOP":      _pat_spinning_top,
}


class PandasTaCalculator(BaseFactorCalculator):
    """pandas-ta 因子计算器（数据驱动注册表模式）"""

    def __init__(self):
        if not HAS_PANDAS_TA:
            raise ImportError("pandas-ta 未安装，请 pip install pandas-ta")
        # 缓存 getattr(ta, func)，避免每次计算都做属性查找
        self._func_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    def get_available_factors(self) -> List[str]:
        """返回所有已注册因子名称列表（按字母排序）。"""
        return sorted(PANDAS_TA_FUNCTION_REGISTRY.keys())

    def get_factor_info(self, factor_name: str) -> Dict:
        """返回因子元信息：名称/类别/方向/参数/输入列/底层pandas-ta函数。"""
        entry = PANDAS_TA_FUNCTION_REGISTRY.get(factor_name)
        if entry is None:
            return {}
        return {
            "name": entry["description"],
            "category": entry["category"],
            "direction": entry["direction"],
            "params": dict(entry["params"]),
            "inputs": list(entry["inputs"]),
            "func": entry["func"],
            "output_key": entry["output_key"],
        }

    def calculate(self, data: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """
        批量计算因子，按股票(code)分组。

        对于多输出函数（MACD/BBANDS/STOCH/AROON 等），
        组内会缓存完整输出 DataFrame，避免重复计算。
        例如同时计算 macd/macd_signal/macd_hist 时，底层 macd 只调用一次。
        """
        if data.empty:
            return data

        # 校验因子名
        unknown = [f for f in factor_names if f not in PANDAS_TA_FUNCTION_REGISTRY]
        if unknown:
            raise ValueError(f"不支持的因子: {unknown}")

        # 保留原始索引顺序，输出按 code/date 排序
        result = data[['code', 'date']].copy().reset_index(drop=True)
        for fn in factor_names:
            result[fn] = np.nan

        data_sorted = data.sort_values(['code', 'date']).reset_index(drop=True)

        # 按股票分组，组内缓存多输出函数
        for code, grp in data_sorted.groupby('code', sort=False):
            idx = grp.index
            cache: Dict[Tuple[str, Tuple[str, ...], Tuple[Tuple[str, Any], ...]], Any] = {}
            for fn in factor_names:
                try:
                    values = self._calc_factor(grp, fn, cache)
                    # CDL 形态返回整数，统一转 float 写入
                    result.loc[idx, fn] = np.asarray(values, dtype=float)
                except Exception as e:
                    print(f"计算 {code} 的 {fn} 失败: {e}")

        return result

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _get_pta_func(self, func_name: str):
        """带缓存的 pandas-ta 函数获取。"""
        fn = self._func_cache.get(func_name)
        if fn is None:
            fn = getattr(ta, func_name)
            self._func_cache[func_name] = fn
        return fn

    @staticmethod
    def _find_column(df: pd.DataFrame, prefix: str) -> str:
        """在 DataFrame 中查找以 prefix 开头的第一列。"""
        for col in df.columns:
            if isinstance(col, str) and col.startswith(prefix):
                return col
        raise KeyError(
            f"未找到以 '{prefix}' 开头的列，现有列: {list(df.columns)}"
        )

    def _calc_factor(
        self,
        df: pd.DataFrame,
        factor_name: str,
        cache: Dict[Tuple, Any],
    ) -> np.ndarray:
        """
        计算单只股票的单个因子。

        多输出函数按 (func, inputs, kwargs) 做缓存，避免重复计算。
        pandas-ta 不支持的函数（func=None）返回 NaN 数组并记录告警。
        """
        entry = PANDAS_TA_FUNCTION_REGISTRY[factor_name]
        func_name = entry["func"]
        inputs = tuple(entry["inputs"])
        kwargs = entry["kwargs"]
        output_key = entry["output_key"]
        pattern_name = entry["pattern_name"]

        # ---- pandas-ta 不支持的因子 ----
        if func_name is None:
            # K线形态：尝试纯 pandas/numpy 实现
            if pattern_name is not None:
                handler = _PATTERN_HANDLERS.get(pattern_name)
                if handler is not None:
                    o = df['open'].values.astype(float)
                    h = df['high'].values.astype(float)
                    l = df['low'].values.astype(float)
                    c = df['close'].values.astype(float)
                    return handler(o, h, l, c)
                # 未实现的形态：告警并返回 NaN
                print(f"警告: pandas-ta 不支持 K线形态 {pattern_name}，返回 NaN")
            else:
                # 希尔伯特/SAREXT 等不支持的函数
                print(f"警告: pandas-ta 不支持因子 {factor_name}，返回 NaN")
            return np.full(len(df), np.nan)

        # ---- 调用 pandas-ta 函数（带多输出缓存） ----
        kwargs_key = tuple(sorted(kwargs.items(), key=lambda kv: kv[0]))
        cache_key = (func_name, inputs, kwargs_key)

        result = cache.get(cache_key)
        if result is None:
            pta_func = self._get_pta_func(func_name)
            args = [df[col] for col in inputs]
            result = pta_func(*args, **kwargs)
            cache[cache_key] = result

        # ---- 从结果中提取目标输出 ----
        if isinstance(result, pd.DataFrame):
            if output_key is not None:
                col = self._find_column(result, output_key)
                return result[col].values.astype(float)
            # 单输出但返回了 DataFrame：取第一列
            return result.iloc[:, 0].values.astype(float)

        # Series 直接取值
        return np.asarray(result, dtype=float)
