"""指标卡片网格渲染器"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.metric_grid")


def render_metric_grid(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染指标卡片网格"""
    try:
        from scripts.factors.registry import get_factor_meta
    except Exception:
        get_factor_meta = None

    cards = []
    for name, val in factor_values.items():
        label = name
        direction = ""
        if get_factor_meta:
            meta = get_factor_meta(name)
            if meta:
                label = meta.description
                if meta.direction > 0:
                    direction = "positive"
                elif meta.direction < 0:
                    direction = "negative"

        formatted = _format_value(name, val)
        cards.append(f'''
        <div class="metric-card {direction}">
            <div class="metric-value">{formatted}</div>
            <div class="metric-label">{label}</div>
        </div>''')

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    return f'{hint_html}<div class="metrics-grid">{"".join(cards)}</div>'


def _format_value(name: str, val: Any) -> str:
    """格式化数值显示"""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    # 百分比类因子
    pct_keywords = ['ratio', 'turnover', 'growth', 'margin', 'percentile', 'pe_', 'pb_', 'dv_']
    if any(kw in name.lower() for kw in pct_keywords):
        try:
            return f"{float(val):.2f}%"
        except (ValueError, TypeError):
            return str(val)
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)
