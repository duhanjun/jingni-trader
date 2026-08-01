"""
报告模板引擎

根据模板配置（YAML）从 factor_data.parquet 和 price_data.parquet 生成 HTML 报告。
报告结构 = 固定章节(行情数据 + 深度解读 + 风险提示) + 因子组合章节(N个)

数据流向：
- 行情数据章节：直接读 price_data.parquet（K线图）
- 因子组合章节：读 factor_data.parquet，调用渲染器生成 HTML
- 深度解读章节：基于模板因子生成 LLM prompt，留占位符待注入
- 风险提示章节：从因子异常值自动提取风险信号
"""
import os
import sys
import logging
import importlib.util
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd

logger = logging.getLogger("template_engine")

# 模板配置目录
_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "config")


def _load_yaml_config(template_id: str) -> Optional[Dict[str, Any]]:
    """加载模板 YAML 配置"""
    yaml_path = os.path.join(_TEMPLATE_DIR, f"{template_id}.yaml")
    if not os.path.exists(yaml_path):
        logger.error(f"模板配置不存在: {yaml_path}")
        return None
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML 未安装，尝试简单解析")
        return _parse_simple_yaml(yaml_path)
    except Exception as e:
        logger.error(f"加载模板配置失败: {e}")
        return None


def _parse_simple_yaml(yaml_path: str) -> Dict[str, Any]:
    """无 PyYAML 时的简单 YAML 解析（仅支持本模板格式）"""
    config: Dict[str, Any] = {"factor_groups": []}
    current_group: Optional[Dict[str, Any]] = None
    section = None

    with open(yaml_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if not line.startswith(" ") and ":" in stripped:
                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip()
                if val:
                    if key in ("template_id", "template_name", "analyst_type"):
                        config[key] = val
                    else:
                        config[key] = val
                else:
                    if key == "fixed_sections":
                        section = "fixed"
                        config["fixed_sections"] = {}
                    elif key == "factor_groups":
                        section = "groups"
                current_group = None
            elif line.startswith("  - "):
                # 新因子分组
                if section == "groups":
                    current_group = {}
                    config["factor_groups"].append(current_group)
                    item = stripped[2:].strip()
                    if ":" in item:
                        k, _, v = item.partition(":")
                        current_group[k.strip()] = v.strip().strip('"')
            elif line.startswith("    ") and current_group is not None:
                item = stripped
                if ":" in item:
                    k, _, v = item.partition(":")
                    k = k.strip()
                    v = v.strip().strip('"')
                    if k == "factors":
                        current_group[k] = [x.strip() for x in v.strip("[]").split(",") if x.strip()]
                    else:
                        current_group[k] = v
            elif line.startswith("    ") and section == "fixed":
                # fixed_sections 子项
                pass

    return config


class ReportTemplateEngine:
    """报告模板引擎"""

    def __init__(self):
        self._kline_chart_module = None

    def generate(
        self,
        template_id: str,
        ctx,
        output_path: str,
    ) -> Dict[str, Any]:
        """根据模板生成 HTML 报告

        参数:
            template_id: 模板ID (technical / fundamental)
            ctx: Context 对象，需包含 artifacts['DATA'] 和 artifacts['FACTOR']
            output_path: 输出文件路径

        返回:
            {
                "success": bool,
                "artifact_path": str,
                "llm_prompts": dict,  # 供 agent 调用 LLM
                "error": str
            }
        """
        # 1. 加载模板配置
        config = _load_yaml_config(template_id)
        if not config:
            return {"success": False, "artifact_path": "", "llm_prompts": {}, "error": f"模板 {template_id} 加载失败"}

        template_name = config.get("template_name", template_id)
        analyst_type = config.get("analyst_type", template_id)
        logger.info(f"开始生成报告: {template_name} (template_id={template_id})")

        # 2. 读取数据
        data_path = ctx.get_artifact("DATA") if hasattr(ctx, 'get_artifact') else None
        factor_path = ctx.get_artifact("FACTOR") if hasattr(ctx, 'get_artifact') else None

        if not data_path or not os.path.exists(data_path):
            return {"success": False, "artifact_path": "", "llm_prompts": {}, "error": "缺少 DATA 产物"}

        price_data = pd.read_parquet(data_path)
        if price_data.empty:
            return {"success": False, "artifact_path": "", "llm_prompts": {}, "error": "行情数据为空"}

        factor_data = pd.DataFrame()
        if factor_path and os.path.exists(factor_path):
            try:
                factor_data = pd.read_parquet(factor_path)
            except Exception as e:
                logger.warning(f"读取因子数据失败: {e}")

        # 取第一只股票
        stock_code = str(price_data['code'].iloc[0]) if 'code' in price_data.columns else ""
        stock_name = stock_code
        ohlcv = price_data[price_data['code'] == stock_code].sort_values('date') if stock_code else price_data
        current_price = float(ohlcv.iloc[-1]['close']) if len(ohlcv) > 0 else 0.0
        data_date = str(ohlcv.iloc[-1]['date'])[:10] if len(ohlcv) > 0 else ""

        # 3. 渲染固定章节
        fixed_cfg = config.get("fixed_sections", {})

        # 3a. 行情数据章节（K线图，直接读 price_data）
        market_html = self._render_market_data(fixed_cfg.get("market_data", {}), ohlcv, stock_code)

        # 3b. 因子组合章节
        factor_sections_html = ""
        all_factor_values_for_llm: Dict[str, Any] = {}
        for group_cfg in config.get("factor_groups", []):
            section_html, factor_values = self._render_factor_group(group_cfg, factor_data, stock_code)
            factor_sections_html += section_html
            all_factor_values_for_llm.update(factor_values)

        # 3c. 深度解读章节（LLM 占位符）
        llm_prompt = self._prepare_llm_prompt(analyst_type, stock_code, stock_name,
                                               current_price, data_date, all_factor_values_for_llm,
                                               template_config=config)
        deep_html = f'''
<div class="section" id="deep_analysis">
    <h2>深度解读</h2>
    <!--LLM_{"TECHNICAL" if analyst_type == "technical" else "FUNDAMENTAL"}_ANALYSIS_PLACEHOLDER-->
</div>'''

        # 3d. 风险提示章节
        risk_html = self._render_risk_warning(fixed_cfg.get("risk_warning", {}), all_factor_values_for_llm)

        # 4. 组装 HTML
        html = self._assemble_html(
            template_name=template_name,
            stock_code=stock_code,
            stock_name=stock_name,
            current_price=current_price,
            data_date=data_date,
            market_html=market_html,
            factor_sections_html=factor_sections_html,
            deep_html=deep_html,
            risk_html=risk_html,
        )

        # 5. 写入文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"报告已生成: {output_path}")

        llm_prompts = {}
        if llm_prompt:
            llm_prompts[analyst_type] = llm_prompt

        return {
            "success": True,
            "artifact_path": output_path,
            "llm_prompts": llm_prompts,
            "error": "",
        }

    # ------------------------------------------------------------------
    # 固定章节渲染
    # ------------------------------------------------------------------

    def _render_market_data(self, market_cfg: Dict, ohlcv: pd.DataFrame, stock_code: str) -> str:
        """渲染行情数据章节（K线图）"""
        kline_html = ""
        try:
            kline_html = self._render_kline_chart(ohlcv, market_cfg)
        except Exception as e:
            logger.warning(f"K线图渲染失败: {e}")
            kline_html = f'<p class="no-data">K线图渲染失败: {e}</p>'

        return f'''
<div class="section" id="market_data">
    <h2>行情数据</h2>
    <div class="chart-container">{kline_html}</div>
</div>'''

    def _render_kline_chart(self, ohlcv: pd.DataFrame, market_cfg: Dict) -> str:
        """调用现有 kline_chart 模块渲染 K线图"""
        if self._kline_chart_module is None:
            self._kline_chart_module = self._load_kline_module()
        if self._kline_chart_module is None:
            return '<p class="no-data">K线图模块不可用</p>'

        ma_periods = market_cfg.get("ma_periods", [5, 10, 20, 60])
        show_sr = market_cfg.get("show_support_resistance", False)

        try:
            gen = self._kline_chart_module.KlineChartGenerator()
            return gen.generate_tradingview_chart(
                ohlcv,
                ma_periods=ma_periods,
                show_support_resistance=show_sr,
            )
        except Exception as e:
            logger.warning(f"K线图渲染异常: {e}")
            return f'<p class="no-data">K线图异常: {e}</p>'

    def _load_kline_module(self):
        """加载 kline_chart 模块"""
        import importlib.util
        kline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts", "kline_chart.py")
        if not os.path.exists(kline_path):
            logger.warning(f"kline_chart.py 不存在: {kline_path}")
            return None
        try:
            spec = importlib.util.spec_from_file_location("kline_chart", kline_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            logger.warning(f"加载 kline_chart 模块失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 因子组合章节渲染
    # ------------------------------------------------------------------

    # 因子名别名映射：模板声明的因子名 → 数据 schema 实际列名
    # 解决模板（如 fundamental.yaml 用 roe_ttm）与数据 schema（用 roe）错位问题
    FACTOR_NAME_ALIASES = {
        # 盈利能力
        'roe_ttm': 'roe',
        'roe_avg': 'roe',
        'roa_ttm': 'roa',
        'gross_profit_margin': 'gross_margin',
        'net_profit_margin': 'net_margin',
        # 成长性
        'revenue_growth_yoy': 'revenue_growth',
        'profit_growth_yoy': 'profit_growth',
        'net_profit_growth_yoy': 'profit_growth',
        # 偿债能力
        'debt_to_asset': 'debt_ratio',
        'current': 'current_ratio',
        'quick': 'quick_ratio',
        # 估值
        'pe': 'pe_ttm',
        'pb_ratio': 'pb',
        'ps': 'ps_ttm',
        'dividend_yield': 'dv_ratio',
        # 现金流
        'ocf_per_share': 'ocf',
        'operating_cash_flow': 'ocf',
    }

    def _resolve_factor_name(self, factor_name: str, available_columns) -> str:
        """将模板因子名解析为数据中实际存在的列名。
        优先精确匹配，其次查别名表。返回 None 表示无匹配。"""
        # 1. 精确匹配
        if factor_name in available_columns:
            return factor_name
        # 2. 别名匹配
        alias = self.FACTOR_NAME_ALIASES.get(factor_name)
        if alias and alias in available_columns:
            return alias
        # 3. 反向匹配（数据列名是别名，模板用原名）
        for template_name, data_name in self.FACTOR_NAME_ALIASES.items():
            if data_name == factor_name and template_name in available_columns:
                return template_name
        return None

    def _render_factor_group(
        self,
        group_cfg: Dict,
        factor_data: pd.DataFrame,
        stock_code: str,
    ) -> tuple:
        """渲染一个因子分组，返回 (html, factor_values_dict)"""
        from scripts.renderers.registry import render_factor_group

        group_id = group_cfg.get("id", "")
        title = group_cfg.get("title", "")
        factors = group_cfg.get("factors", [])
        render_as = group_cfg.get("render_as", "metric_grid")
        hint = group_cfg.get("analysis_hint", "")

        if factor_data.empty:
            html = f'<div class="section" id="{group_id}"><h2>{title}</h2><p class="no-data">暂无因子数据</p></div>'
            return html, {}

        # 提取该股票的因子数据（取最新行）
        stock_factors = factor_data
        if 'code' in factor_data.columns and stock_code:
            stock_factors = factor_data[factor_data['code'] == stock_code]
        if 'date' in stock_factors.columns and len(stock_factors) > 0:
            stock_factors = stock_factors.sort_values('date').iloc[-1:]

        # 只保留配置中指定的因子列（含别名解析）
        available_factors = []
        factor_name_mapping = {}  # 模板名 -> 实际列名
        for f in factors:
            resolved = self._resolve_factor_name(f, stock_factors.columns)
            if resolved:
                available_factors.append(f)
                factor_name_mapping[f] = resolved

        if not available_factors:
            html = f'<div class="section" id="{group_id}"><h2>{title}</h2><p class="no-data">暂无数据</p></div>'
            return html, {}

        factor_values = {}
        for f in available_factors:
            actual_col = factor_name_mapping[f]
            if actual_col in stock_factors.columns:
                val = stock_factors[actual_col].iloc[0] if len(stock_factors) > 0 else None
                factor_values[f] = val

        html = render_factor_group(group_cfg, factor_data, stock_code)
        return html, factor_values

    # ------------------------------------------------------------------
    # 深度解读 LLM prompt 准备
    # ------------------------------------------------------------------

    def _prepare_llm_prompt(
        self,
        analyst_type: str,
        stock_code: str,
        stock_name: str,
        current_price: float,
        data_date: str,
        factor_values: Dict[str, Any],
        template_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """基于模板因子生成 LLM prompt（动态加载 factor_groups）

        返回:
            dict: {"system_prompt": str, "user_prompt": str, "response_schema": dict}
        """
        try:
            from scripts.llm_analyst import TechnicalAnalyst, FundamentalsAnalyst
        except ImportError:
            logger.warning("llm_analyst 模块不可用")
            return {}

        # 提取 factor_groups（优先使用传入的 template_config）
        factor_groups = None
        if template_config:
            factor_groups = template_config.get("factor_groups", [])
        else:
            # 回退：从模板文件加载
            config = _load_yaml_config(analyst_type)
            if config:
                factor_groups = config.get("factor_groups", [])

        # 从 factor_values 构造上下文
        ctx_data = {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "current_price": current_price,
            "data_date": data_date,
            "technical_indicators": factor_values,
            "fundamental_data": factor_values,
        }

        try:
            if analyst_type == "technical":
                return TechnicalAnalyst().prepare(ctx_data, factor_groups=factor_groups)
            else:
                return FundamentalsAnalyst().prepare(ctx_data, factor_groups=factor_groups)
        except Exception as e:
            logger.warning(f"LLM prompt 准备失败: {e}")
            return {}

    # ------------------------------------------------------------------
    # 风险提示章节
    # ------------------------------------------------------------------

    def _render_risk_warning(self, risk_cfg: Dict, factor_values: Dict[str, Any]) -> str:
        """从因子异常值提取风险信号"""
        risk_factors = risk_cfg.get("source_factors", [])
        risks: List[str] = []

        for name in risk_factors:
            val = factor_values.get(name)
            if val is None or (isinstance(val, float) and val != val):
                continue
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue

            risk = self._check_risk(name, v)
            if risk:
                risks.append(risk)

        if not risks:
            risks.append("当前未检测到明显异常风险信号。")

        risk_items = "".join(f'<li>{r}</li>' for r in risks)
        return f'''
<div class="section" id="risk_warning">
    <h2>风险提示</h2>
    <ul class="risk-list">{risk_items}</ul>
</div>'''

    def _check_risk(self, factor_name: str, value: float) -> Optional[str]:
        """检查单个因子的风险信号"""
        if factor_name == "volatility_20d" and value > 0.05:
            return f"20日波动率 {value*100:.2f}% 超过阈值 5%，价格波动较大"
        if factor_name == "volume_ratio" and value > 2.0:
            return f"量比 {value:.2f} 偏高，存在异常放量"
        if factor_name == "money_flow_20d" and value < 0:
            return f"20日累计资金流为负 ({value:.2f})，资金持续流出"
        if factor_name == "debt_ratio" and value > 0.7:
            return f"资产负债率 {value*100:.1f}% 偏高，偿债压力较大"
        if factor_name == "revenue_growth_yoy" and value < 0:
            return f"营收同比增速 {value:.2f}% 为负，业绩下滑"
        if factor_name == "pe_ttm" and value > 100:
            return f"PE(TTM) {value:.1f} 倍偏高，估值风险"
        return None

    # ------------------------------------------------------------------
    # HTML 组装
    # ------------------------------------------------------------------

    def _assemble_html(
        self,
        template_name: str,
        stock_code: str,
        stock_name: str,
        current_price: float,
        data_date: str,
        market_html: str,
        factor_sections_html: str,
        deep_html: str,
        risk_html: str,
    ) -> str:
        """组装完整 HTML 报告"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{template_name} - {stock_name}({stock_code})</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
        .header {{ background: linear-gradient(135deg, #1f77b4, #2ca02c); color: white;
                   padding: 30px 40px; border-radius: 12px; margin-bottom: 24px; }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .section {{ background: white; border-radius: 10px; padding: 24px; margin-bottom: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .section h2 {{ margin: 0 0 16px 0; font-size: 18px; color: #444;
                       border-bottom: 2px solid #eee; padding-bottom: 8px; }}
        .chart-container {{ width: 100%; overflow-x: auto; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                         gap: 16px; }}
        .metric-card {{ background: #f9f9f9; padding: 16px; border-radius: 8px; text-align: center;
                        border: 1px solid #eee; }}
        .metric-card.positive {{ border-left: 3px solid #2ca02c; }}
        .metric-card.negative {{ border-left: 3px solid #d62728; }}
        .metric-value {{ font-size: 24px; font-weight: 700; color: #333; }}
        .metric-card.positive .metric-value {{ color: #2ca02c; }}
        .metric-card.negative .metric-value {{ color: #d62728; }}
        .metric-label {{ font-size: 12px; color: #888; margin-top: 4px; }}
        .signal-tag {{ display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px;
                       margin-left: 6px; }}
        .signal-bullish {{ background: #e8f5e9; color: #2e7d32; }}
        .signal-bearish {{ background: #ffebee; color: #c62828; }}
        .signal-neutral {{ background: #f5f5f5; color: #666; }}
        .analysis-hint {{ color: #888; font-size: 13px; font-style: italic; margin-bottom: 12px; }}
        .no-data {{ color: #999; text-align: center; padding: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f9f9f9; font-weight: 600; color: #555; }}
        .percentile-list {{ display: flex; flex-direction: column; gap: 12px; }}
        .percentile-item {{ display: flex; align-items: center; gap: 12px; }}
        .percentile-label {{ width: 120px; font-size: 13px; color: #555; }}
        .percentile-bar-container {{ flex: 1; height: 20px; background: #eee; border-radius: 10px; overflow: hidden; }}
        .percentile-bar {{ height: 100%; border-radius: 10px; transition: width 0.3s; }}
        .percentile-bar.overvalued {{ background: #d62728; }}
        .percentile-bar.undervalued {{ background: #2ca02c; }}
        .percentile-bar.fair {{ background: #1f77b4; }}
        .percentile-value {{ width: 120px; font-size: 13px; font-weight: 600; }}
        .level-tag {{ font-size: 11px; padding: 1px 6px; border-radius: 8px; margin-left: 4px; }}
        .level-tag.overvalued {{ background: #ffebee; color: #c62828; }}
        .level-tag.undervalued {{ background: #e8f5e9; color: #2e7d32; }}
        .level-tag.fair {{ background: #e3f2fd; color: #1565c0; }}
        .band-table td:first-child {{ font-weight: 600; }}
        .band-table .upper td:last-child {{ color: #d62728; }}
        .band-table .lower td:last-child {{ color: #2ca02c; }}
        .band-width {{ margin-top: 8px; font-size: 13px; color: #666; }}
        .trend-list {{ display: flex; flex-direction: column; gap: 10px; }}
        .trend-item {{ display: flex; justify-content: space-between; padding: 10px 14px; background: #f9f9f9; border-radius: 6px; }}
        .trend-label {{ color: #555; }}
        .trend-value {{ font-weight: 600; }}
        .trend-up {{ color: #d62728; }}
        .trend-down {{ color: #2ca02c; }}
        .flow-table .flow-in {{ color: #d62728; font-weight: 600; }}
        .flow-table .flow-out {{ color: #2ca02c; font-weight: 600; }}
        .event-list {{ display: flex; flex-direction: column; gap: 10px; }}
        .event-item {{ display: flex; justify-content: space-between; padding: 10px 14px; background: #f9f9f9; border-radius: 6px; }}
        .holder-table .holder-increase {{ color: #d62728; font-weight: 600; }}
        .holder-table .holder-decrease {{ color: #2ca02c; font-weight: 600; }}
        .risk-list {{ padding-left: 20px; }}
        .risk-list li {{ margin-bottom: 8px; color: #555; }}
        .footer {{ text-align: center; margin-top: 40px; font-size: 12px; color: #aaa; }}
    </style>
</head>
<body>
<div class="header">
    <h1>{template_name} — {stock_name}({stock_code})</h1>
    <p>数据日期: {data_date} | 当前价: {current_price:.2f} | 生成时间: {now}</p>
</div>

{market_html}
{factor_sections_html}
{deep_html}
{risk_html}

<div class="footer">Generated by jingni-trader 投资分析系统</div>
</body>
</html>'''
