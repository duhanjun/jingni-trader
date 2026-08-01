"""
基于 TA-Lib 的因子计算器适配器（数据驱动注册表模式）

覆盖全部约 130 个有意义的 TA-Lib 函数：
  - 重叠指标 Overlap Studies
  - 动量指标 Momentum Indicators
  - 成交量指标 Volume Indicators
  - 波动率指标 Volatility Indicators
  - 价格变换 Price Transform
  - 统计函数 Statistic Functions
  - 周期指标 Cycle Indicators
  - K线形态识别 Pattern Recognition (61 个 CDL 函数)

纯数学函数（ACOS/ASIN/ATAN/CEIL/COS/EXP/FLOOR/LN/LOG10/SIN/SQRT/TAN/ADD/DIV/MAX/MIN/MULT/SUB）
不计入因子库，故不在此注册。
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from ..base.base_factor_calculator import BaseFactorCalculator


# ------------------------------------------------------------------
# 注册表定义
# ------------------------------------------------------------------
# 每条记录字段说明：
#   func        : TA-Lib 函数名（字符串，懒加载 getattr(talib, func)）
#   inputs      : 所需输入列名顺序列表（取自 OHLCV）
#   kwargs      : 调用 TA-Lib 时的默认关键字参数
#   output_idx  : 多输出函数的输出索引；None 表示单输出
#   category    : 因子类别 (overlap/momentum/volume/volatility/price/
#                 statistic/cycle/pattern)
#   description : 中文名称/描述
#   direction   : 方向 (1=看多, -1=看空, 0=中性)
#   params      : 对外暴露的参数（默认与 kwargs 等价，便于前端展示）
TALIB_FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _register(name: str,
              func: str,
              inputs: List[str],
              kwargs: Dict[str, Any],
              category: str,
              description: str,
              direction: int = 0,
              output_idx: Optional[int] = None,
              params: Optional[Dict[str, Any]] = None) -> None:
    """注册一个 TA-Lib 因子。"""
    TALIB_FUNCTION_REGISTRY[name] = {
        "func": func,
        "inputs": list(inputs),
        "kwargs": dict(kwargs),
        "output_idx": output_idx,
        "category": category,
        "description": description,
        "direction": direction,
        "params": dict(params) if params is not None else dict(kwargs),
    }


# ==================================================================
# 1. 重叠指标 Overlap Studies
# ==================================================================
_register("dema", "DEMA", ["close"], {"timeperiod": 30},
          "overlap", "双指数移动平均线", 0)
_register("ema", "EMA", ["close"], {"timeperiod": 30},
          "overlap", "指数移动平均线", 0)
_register("ht_trendline", "HT_TRENDLINE", ["close"], {},
          "overlap", "希尔伯特瞬时趋势线", 0)
_register("kama", "KAMA", ["close"], {"timeperiod": 30},
          "overlap", "考夫曼自适应移动平均线", 0)
_register("ma", "MA", ["close"], {"timeperiod": 30, "matype": 0},
          "overlap", "移动平均线（matype: 0=SMA/1=EMA/2=WMA/3=DEMA/4=TEMA）", 0)
_register("midpoint", "MIDPOINT", ["close"], {"timeperiod": 14},
          "overlap", "区间中点", 0)
_register("midprice", "MIDPRICE", ["high", "low"], {"timeperiod": 14},
          "overlap", "区间中间价", 0)
_register("sar", "SAR", ["high", "low"], {"acceleration": 0.02, "maximum": 0.2},
          "overlap", "抛物线SAR（停损转向点）", 0)
_register("sarext", "SAREXT", ["high", "low"], {},
          "overlap", "扩展抛物线SAR", 0)
_register("t3", "T3", ["close"], {"timeperiod": 5, "vfactor": 0.7},
          "overlap", "T3三重平滑移动平均线", 0)
_register("tema", "TEMA", ["close"], {"timeperiod": 30},
          "overlap", "三重指数移动平均线", 0)
_register("trima", "TRIMA", ["close"], {"timeperiod": 30},
          "overlap", "三角移动平均线", 0)
_register("wma", "WMA", ["close"], {"timeperiod": 30},
          "overlap", "加权移动平均线", 0)

# 布林带（多输出，统一缓存）
_register("bbands_upper", "BBANDS", ["close"],
          {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0},
          "overlap", "布林带上轨", 1, output_idx=0)
_register("bbands_middle", "BBANDS", ["close"],
          {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0},
          "overlap", "布林带中轨", 0, output_idx=1)
_register("bbands_lower", "BBANDS", ["close"],
          {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0},
          "overlap", "布林带下轨", -1, output_idx=2)

# 向后兼容别名（保留旧命名）
_register("bollinger_upper", "BBANDS", ["close"],
          {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0},
          "overlap", "布林带上轨（兼容别名）", 1, output_idx=0)
_register("bollinger_middle", "BBANDS", ["close"],
          {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0},
          "overlap", "布林带中轨（兼容别名）", 0, output_idx=1)
_register("bollinger_lower", "BBANDS", ["close"],
          {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0},
          "overlap", "布林带下轨（兼容别名）", -1, output_idx=2)

_register("ma_5", "MA", ["close"], {"timeperiod": 5, "matype": 0},
          "overlap", "5日简单移动平均线", 0)
_register("ma_10", "MA", ["close"], {"timeperiod": 10, "matype": 0},
          "overlap", "10日简单移动平均线", 0)
_register("ma_20", "MA", ["close"], {"timeperiod": 20, "matype": 0},
          "overlap", "20日简单移动平均线", 0)
_register("ma_30", "MA", ["close"], {"timeperiod": 30, "matype": 0},
          "overlap", "30日简单移动平均线", 0)
_register("ma_60", "MA", ["close"], {"timeperiod": 60, "matype": 0},
          "overlap", "60日简单移动平均线", 0)
_register("ma_120", "MA", ["close"], {"timeperiod": 120, "matype": 0},
          "overlap", "120日简单移动平均线", 0)
_register("ma_250", "MA", ["close"], {"timeperiod": 250, "matype": 0},
          "overlap", "250日简单移动平均线（年线）", 0)
_register("ema_5", "EMA", ["close"], {"timeperiod": 5},
          "overlap", "5日指数移动平均线", 0)
_register("ema_10", "EMA", ["close"], {"timeperiod": 10},
          "overlap", "10日指数移动平均线", 0)
_register("ema_20", "EMA", ["close"], {"timeperiod": 20},
          "overlap", "20日指数移动平均线", 0)
_register("ema_60", "EMA", ["close"], {"timeperiod": 60},
          "overlap", "60日指数移动平均线", 0)


# ==================================================================
# 2. 动量指标 Momentum Indicators
# ==================================================================
_register("adx", "ADX", ["high", "low", "close"], {"timeperiod": 14},
          "momentum", "平均趋向指数（趋势强度）", 0)
_register("adxr", "ADXR", ["high", "low", "close"], {"timeperiod": 14},
          "momentum", "ADX评级", 0)
_register("apo", "APO", ["close"],
          {"fastperiod": 12, "slowperiod": 26, "matype": 0},
          "momentum", "绝对价格震荡指标", 0)
_register("aroon_up", "AROON", ["high", "low"], {"timeperiod": 14},
          "momentum", "Aroon上轨", 1, output_idx=0)
_register("aroon_down", "AROON", ["high", "low"], {"timeperiod": 14},
          "momentum", "Aroon下轨", -1, output_idx=1)
_register("aroonosc", "AROONOSC", ["high", "low"], {"timeperiod": 14},
          "momentum", "Aroon震荡指标", 0)
_register("bop", "BOP", ["open", "high", "low", "close"], {},
          "momentum", "力量指标（开盘-最高-最低-收盘）", 0)
_register("cci", "CCI", ["high", "low", "close"], {"timeperiod": 14},
          "momentum", "商品通道指标", 0)
_register("cmo", "CMO", ["close"], {"timeperiod": 14},
          "momentum", "钱德动量摆动指标", 0)
_register("dx", "DX", ["high", "low", "close"], {"timeperiod": 14},
          "momentum", "趋向指标DX", 0)
_register("macd", "MACD", ["close"],
          {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
          "momentum", "MACD差离值", 0, output_idx=0)
_register("macd_signal", "MACD", ["close"],
          {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
          "momentum", "MACD信号线", 0, output_idx=1)
_register("macd_hist", "MACD", ["close"],
          {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
          "momentum", "MACD柱状图", 0, output_idx=2)
_register("mfi", "MFI", ["high", "low", "close", "volume"],
          {"timeperiod": 14}, "momentum", "资金流量指标", 0)
_register("minus_di", "MINUS_DI", ["high", "low", "close"],
          {"timeperiod": 14}, "momentum", "负向趋向指标DI", -1)
_register("minus_dm", "MINUS_DM", ["high", "low"],
          {"timeperiod": 14}, "momentum", "负向趋向指标DM", -1)
_register("mom", "MOM", ["close"], {"timeperiod": 10},
          "momentum", "动量指标", 0)
_register("plus_di", "PLUS_DI", ["high", "low", "close"],
          {"timeperiod": 14}, "momentum", "正向趋向指标DI", 1)
_register("plus_dm", "PLUS_DM", ["high", "low"],
          {"timeperiod": 14}, "momentum", "正向趋向指标DM", 1)
_register("ppo", "PPO", ["close"],
          {"fastperiod": 12, "slowperiod": 26, "matype": 0},
          "momentum", "价格百分比震荡指标", 0)
_register("roc", "ROC", ["close"], {"timeperiod": 10},
          "momentum", "变动率指标", 0)
_register("rocp", "ROCP", ["close"], {"timeperiod": 10},
          "momentum", "百分比变动率", 0)
_register("rocr", "ROCR", ["close"], {"timeperiod": 10},
          "momentum", "比率变动率", 0)
_register("rocr100", "ROCR100", ["close"], {"timeperiod": 10},
          "momentum", "ROCR100指标", 0)
_register("rsi", "RSI", ["close"], {"timeperiod": 14},
          "momentum", "相对强弱指标", 0)
_register("stoch_k", "STOCH", ["high", "low", "close"],
          {"fastk_period": 5, "slowk_period": 3, "slowk_matype": 0,
           "slowd_period": 3, "slowd_matype": 0},
          "momentum", "随机指标K线", 0, output_idx=0)
_register("stoch_d", "STOCH", ["high", "low", "close"],
          {"fastk_period": 5, "slowk_period": 3, "slowk_matype": 0,
           "slowd_period": 3, "slowd_matype": 0},
          "momentum", "随机指标D线", 0, output_idx=1)
_register("stochf_k", "STOCHF", ["high", "low", "close"],
          {"fastk_period": 5, "fastd_period": 3, "fastd_matype": 0},
          "momentum", "快速随机指标K线", 0, output_idx=0)
_register("stochf_d", "STOCHF", ["high", "low", "close"],
          {"fastk_period": 5, "fastd_period": 3, "fastd_matype": 0},
          "momentum", "快速随机指标D线", 0, output_idx=1)
_register("stochrsi_k", "STOCHRSI", ["close"],
          {"timeperiod": 14, "fastk_period": 5,
           "fastd_period": 3, "fastd_matype": 0},
          "momentum", "随机RSI指标K线", 0, output_idx=0)
_register("stochrsi_d", "STOCHRSI", ["close"],
          {"timeperiod": 14, "fastk_period": 5,
           "fastd_period": 3, "fastd_matype": 0},
          "momentum", "随机RSI指标D线", 0, output_idx=1)
_register("trix", "TRIX", ["close"], {"timeperiod": 30},
          "momentum", "三重平滑均线动量", 0)
_register("ultosc", "ULTOSC", ["high", "low", "close"],
          {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28},
          "momentum", "终极震荡指标", 0)
_register("willr", "WILLR", ["high", "low", "close"], {"timeperiod": 14},
          "momentum", "威廉指标", 0)


# ==================================================================
# 3. 成交量指标 Volume Indicators
# ==================================================================
_register("ad", "AD", ["high", "low", "close", "volume"], {},
          "volume", "累积/派发线", 0)
_register("adosc", "ADOSC", ["high", "low", "close", "volume"],
          {"fastperiod": 3, "slowperiod": 10},
          "volume", "累积/派发震荡指标", 0)
_register("obv", "OBV", ["close", "volume"], {},
          "volume", "能量潮指标", 0)


# ==================================================================
# 4. 波动率指标 Volatility Indicators
# ==================================================================
_register("atr", "ATR", ["high", "low", "close"], {"timeperiod": 14},
          "volatility", "平均真实波幅", 0)
_register("natr", "NATR", ["high", "low", "close"], {"timeperiod": 14},
          "volatility", "归一化平均真实波幅", 0)
_register("trange", "TRANGE", ["high", "low", "close"], {},
          "volatility", "真实波幅", 0)


# ==================================================================
# 5. 价格变换 Price Transform
# ==================================================================
_register("avgprice", "AVGPRICE", ["open", "high", "low", "close"], {},
          "price", "平均价格", 0)
_register("medprice", "MEDPRICE", ["high", "low"], {},
          "price", "中间价格", 0)
_register("typprice", "TYPPRICE", ["high", "low", "close"], {},
          "price", "典型价格", 0)
_register("wclprice", "WCLPRICE", ["high", "low", "close"], {},
          "price", "加权收盘价", 0)


# ==================================================================
# 6. 统计函数 Statistic Functions
# ==================================================================
_register("beta", "BETA", ["high", "low"], {"timeperiod": 5},
          "statistic", "贝塔系数", 0)
_register("correl", "CORREL", ["high", "low"], {"timeperiod": 30},
          "statistic", "皮尔逊相关系数", 0)
_register("linearreg", "LINEARREG", ["close"], {"timeperiod": 14},
          "statistic", "线性回归", 0)
_register("linearreg_angle", "LINEARREG_ANGLE", ["close"],
          {"timeperiod": 14}, "statistic", "线性回归角度", 0)
_register("linearreg_intercept", "LINEARREG_INTERCEPT", ["close"],
          {"timeperiod": 14}, "statistic", "线性回归截距", 0)
_register("linearreg_slope", "LINEARREG_SLOPE", ["close"],
          {"timeperiod": 14}, "statistic", "线性回归斜率", 0)
_register("stddev", "STDDEV", ["close"], {"timeperiod": 5, "nbdev": 1},
          "statistic", "标准差", 0)
_register("tsf", "TSF", ["close"], {"timeperiod": 14},
          "statistic", "时间序列预测", 0)
_register("var", "VAR", ["close"], {"timeperiod": 5, "nbdev": 1},
          "statistic", "方差", 0)


# ==================================================================
# 7. 周期指标 Cycle Indicators
# ==================================================================
_register("ht_dcperiod", "HT_DCPERIOD", ["close"], {},
          "cycle", "希尔伯特变换-主导周期", 0)
_register("ht_dcphase", "HT_DCPHASE", ["close"], {},
          "cycle", "希尔伯特变换-主导相位", 0)
_register("ht_phasor_inphase", "HT_PHASOR", ["close"], {},
          "cycle", "希尔伯特变换-同相分量", 0, output_idx=0)
_register("ht_phasor_quadrature", "HT_PHASOR", ["close"], {},
          "cycle", "希尔伯特变换-正交分量", 0, output_idx=1)
_register("ht_sine", "HT_SINE", ["close"], {},
          "cycle", "希尔伯特变换-正弦", 0, output_idx=0)
_register("ht_leadsine", "HT_SINE", ["close"], {},
          "cycle", "希尔伯特变换-超前正弦", 0, output_idx=1)


# ==================================================================
# 8. K线形态识别 Pattern Recognition（全部 61 个 CDL 函数）
# ==================================================================
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
        func=_cdl_func,
        inputs=["open", "high", "low", "close"],
        kwargs={},
        category="pattern",
        description=_cdl_desc,
        direction=_cdl_dir,
        output_idx=None,
        params={},
    )

# 清理循环变量，避免污染模块命名空间
del _cdl_func, _cdl_desc, _cdl_dir


class TalibCalculator(BaseFactorCalculator):
    """TA-Lib 因子计算器（数据驱动注册表模式）"""

    def __init__(self):
        if not HAS_TALIB:
            raise ImportError("TA-Lib 未安装，请 pip install TA-Lib")
        # 缓存 getattr(talib, func)，避免每次计算都做属性查找
        self._func_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    def get_available_factors(self) -> List[str]:
        """返回所有已注册因子名称列表（按字母排序）。"""
        return sorted(TALIB_FUNCTION_REGISTRY.keys())

    def get_factor_info(self, factor_name: str) -> Dict:
        """返回因子元信息：名称/类别/方向/参数/输入列/底层TA-Lib函数。"""
        entry = TALIB_FUNCTION_REGISTRY.get(factor_name)
        if entry is None:
            return {}
        return {
            "name": entry["description"],
            "category": entry["category"],
            "direction": entry["direction"],
            "params": dict(entry["params"]),
            "inputs": list(entry["inputs"]),
            "func": entry["func"],
            "output_idx": entry["output_idx"],
        }

    def calculate(self, data: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """
        批量计算因子，按股票(code)分组。

        对于多输出函数（MACD/BBANDS/STOCH/AROON/HT_PHASOR/HT_SINE 等），
        组内会缓存完整输出元组，避免重复计算。
        例如同时计算 macd/macd_signal/macd_hist 时，底层 MACD 只调用一次。
        """
        if data.empty:
            return data

        # 校验因子名
        unknown = [f for f in factor_names if f not in TALIB_FUNCTION_REGISTRY]
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
            cache: Dict[Tuple[str, Tuple[str, ...], Tuple[Tuple[str, Any], ...]], Tuple] = {}
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
    def _get_talib_func(self, func_name: str):
        """带缓存的 TA-Lib 函数获取。"""
        fn = self._func_cache.get(func_name)
        if fn is None:
            fn = getattr(talib, func_name)
            self._func_cache[func_name] = fn
        return fn

    def _calc_factor(
        self,
        df: pd.DataFrame,
        factor_name: str,
        cache: Dict[Tuple, Tuple],
    ) -> np.ndarray:
        """
        计算单只股票的单个因子。

        多输出函数按 (func, inputs, kwargs) 做缓存键，避免重复计算。
        """
        entry = TALIB_FUNCTION_REGISTRY[factor_name]
        func_name = entry["func"]
        inputs = tuple(entry["inputs"])
        kwargs = entry["kwargs"]
        kwargs_key = tuple(sorted(kwargs.items(), key=lambda kv: kv[0]))
        cache_key = (func_name, inputs, kwargs_key)

        outputs = cache.get(cache_key)
        if outputs is None:
            talib_func = self._get_talib_func(func_name)
            args = [df[col].values.astype(float) for col in inputs]
            outputs = talib_func(*args, **kwargs)
            # 归一化为元组
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            else:
                outputs = tuple(outputs)
            cache[cache_key] = outputs

        output_idx = entry["output_idx"]
        if output_idx is None:
            return outputs[0]
        return outputs[output_idx]
