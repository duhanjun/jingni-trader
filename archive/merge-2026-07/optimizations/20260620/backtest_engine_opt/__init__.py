"""
回测引擎优化模块（20260620）

借鉴来源：
- Microsoft Qlib：完整绩效指标体系（含 turnover / alpha / beta / IR）
- QuantStats：丰富的绩效归因与可视化
- Jesse：零 look-ahead bias 设计

本模块提供以下优化：
- enhanced_metrics: 换手率、Alpha/Beta、信息比率、回撤持续期等扩展指标
- look_ahead_guard: 前视偏差检测工具
"""
from .enhanced_metrics import (
    calc_turnover,
    calc_alpha_beta,
    calc_information_ratio,
    calc_max_drawdown_duration,
    calc_all_enhanced_metrics,
)
from .look_ahead_guard import (
    check_forward_return_leakage,
    check_signal_timestamp_order,
    check_feature_alignment,
    LookAheadBiasError,
)

__all__ = [
    "calc_turnover",
    "calc_alpha_beta",
    "calc_information_ratio",
    "calc_max_drawdown_duration",
    "calc_all_enhanced_metrics",
    "check_forward_return_leakage",
    "check_signal_timestamp_order",
    "check_feature_alignment",
    "LookAheadBiasError",
]
