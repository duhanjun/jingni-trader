"""渲染器注册表

每种 render_as 类型对应一个渲染函数，从 factor_data 中提取因子值并生成 HTML 片段。
模板引擎通过 RENDERERS 字典按 render_as 类型查找对应渲染器。
"""
from typing import Dict, Any, Callable, Optional
import pandas as pd

from .metric_grid import render_metric_grid
from .indicator_panel import render_indicator_panel
from .percentile_chart import render_percentile_chart
from .band_chart import render_band_chart
from .trend_chart import render_trend_chart
from .flow_table import render_flow_table
from .event_list import render_event_list
from .info_card import render_info_card
from .pattern_list import render_pattern_list
from .holder_table import render_holder_table

# 渲染器注册表：render_as 类型 → 渲染函数
RENDERERS: Dict[str, Callable] = {
    "metric_grid": render_metric_grid,
    "indicator_panel": render_indicator_panel,
    "percentile_chart": render_percentile_chart,
    "band_chart": render_band_chart,
    "trend_chart": render_trend_chart,
    "flow_table": render_flow_table,
    "event_list": render_event_list,
    "info_card": render_info_card,
    "pattern_list": render_pattern_list,
    "holder_table": render_holder_table,
}


def get_renderer(render_as: str) -> Optional[Callable]:
    """根据 render_as 类型获取渲染器"""
    return RENDERERS.get(render_as)


def render_factor_group(
    group_config: Dict[str, Any],
    factor_data: pd.DataFrame,
    stock_code: str = "",
) -> str:
    """渲染一个因子分组为 HTML 章节

    参数:
        group_config: 因子分组配置，含 id, title, factors, render_as, analysis_hint
        factor_data: 因子数据 DataFrame（含 code, date, [各因子列]）
        stock_code: 股票代码

    返回:
        HTML 字符串
    """
    render_as = group_config.get("render_as", "metric_grid")
    factors = group_config.get("factors", [])
    title = group_config.get("title", "")
    hint = group_config.get("analysis_hint", "")
    group_id = group_config.get("id", "")

    renderer = get_renderer(render_as)
    if renderer is None:
        return f'<div class="section"><h2>{title}</h2><p>不支持的渲染器类型: {render_as}</p></div>'

    # 提取该股票的因子数据（取最新行）
    stock_factors = factor_data
    if 'code' in factor_data.columns and stock_code:
        stock_factors = factor_data[factor_data['code'] == stock_code]
    if 'date' in stock_factors.columns and len(stock_factors) > 0:
        stock_factors = stock_factors.sort_values('date').iloc[-1:]

    # 只保留配置中指定的因子列
    available_factors = [f for f in factors if f in stock_factors.columns]
    if not available_factors:
        return f'<div class="section"><h2>{title}</h2><p class="no-data">暂无数据</p></div>'

    factor_values = {}
    for f in available_factors:
        if f in stock_factors.columns:
            val = stock_factors[f].iloc[0] if len(stock_factors) > 0 else None
            factor_values[f] = val

    body_html = renderer(factor_values, hint=hint)

    return f'''
<div class="section" id="{group_id}">
    <h2>{title}</h2>
    {body_html}
</div>'''
