"""信息卡片渲染器"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.info_card")


def render_info_card(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染信息卡片"""
    try:
        from scripts.factors.registry import get_factor_meta
    except Exception:
        get_factor_meta = None

    rows = []
    for name, val in factor_values.items():
        label = name
        if get_factor_meta:
            meta = get_factor_meta(name)
            if meta:
                label = meta.description
        formatted = str(val) if val is not None else "—"
        rows.append(f'<tr><td class="info-label">{label}</td><td class="info-value">{formatted}</td></tr>')

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    table_html = f'<table class="info-table">{"".join(rows)}</table>'
    return f'{hint_html}{table_html}'
