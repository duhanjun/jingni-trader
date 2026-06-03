"""
Tushare 错误分类器

将 tushare 抛出的各种异常归类为：
- QuotaExceededError  → 积分/权限不足，切下一个源
- RateLimitError      → 频率限制，切下一个源
- NetworkError        → 网络问题，可重试或切下一个源
- InvalidParameterError → 参数错误，不切换
"""
import re
from typing import Optional
from .errors import QuotaExceededError, RateLimitError, NetworkError, InvalidParameterError, DataSourceError


# Tushare 错误消息中典型的限频提示
_RATE_LIMIT_PATTERNS = [
    re.compile(r"(\d+)\s*次/小时"),
    re.compile(r"(\d+)\s*次/分钟"),
    re.compile(r"(\d+)\s*次/天"),
    re.compile(r"访问频率超限"),
    re.compile(r"频率超限"),
    re.compile(r"访问频次超限"),
    re.compile(r"超过最大访问频次"),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"rate limit", re.IGNORECASE),
]

# Tushare 错误消息中典型的权限/积分不足提示
_QUOTA_PATTERNS = [
    re.compile(r"积分不足"),
    re.compile(r"权限不足"),
    re.compile(r"没有权限"),
    re.compile(r"需要更高权限"),
    re.compile(r"没有访问权限"),
    re.compile(r"权限认证失败"),
    re.compile(r"接口未授权"),
    re.compile(r"未购买"),
    re.compile(r"unauthorized", re.IGNORECASE),
    re.compile(r"permission denied", re.IGNORECASE),
    re.compile(r"forbidden", re.IGNORECASE),
]

# 参数错误
_INVALID_PARAM_PATTERNS = [
    re.compile(r"参数错误"),
    re.compile(r"invalid parameter", re.IGNORECASE),
    re.compile(r"invalid token", re.IGNORECASE),
    re.compile(r"token\s*不对"),
    re.compile(r"token\s*错误"),
    re.compile(r"token\s*无效"),
    re.compile(r"token\s*is\s*invalid", re.IGNORECASE),
    re.compile(r"token已过期"),
    re.compile(r"认证失败"),
    re.compile(r"authorization\s*failed", re.IGNORECASE),
]

# 网络错误
_NETWORK_PATTERNS = [
    re.compile(r"timeout", re.IGNORECASE),
    re.compile(r"timed out", re.IGNORECASE),
    re.compile(r"connection (refused|reset|aborted)", re.IGNORECASE),
    re.compile(r"network is unreachable", re.IGNORECASE),
    re.compile(r"dns", re.IGNORECASE),
    re.compile(r"getaddrinfo", re.IGNORECASE),
    re.compile(r"远程主机强迫关闭"),
    re.compile(r"broken pipe", re.IGNORECASE),
    re.compile(r"无法连接"),
    re.compile(r"ConnectionError"),
    re.compile(r"RemoteDisconnected"),
]


def _extract_retry_seconds(message: str) -> Optional[int]:
    """从限频消息中提取建议等待秒数"""
    # 1次/小时 -> 3600
    m = re.search(r"(\d+)\s*次/小时", message)
    if m:
        return 3600
    m = re.search(r"(\d+)\s*次/分钟", message)
    if m:
        return 60
    m = re.search(r"(\d+)\s*次/天", message)
    if m:
        return 86400
    return None


def classify_tushare_error(exc: Exception) -> DataSourceError:
    """
    将 tushare 抛出的异常归类为特定 DataSourceError

    优先级: invalid_param > rate_limit > quota > network
    """
    message = str(exc) or ""
    # 一些 tushare 错误把消息放在 exc.args[0]，可能含中文编码
    if not message and exc.args:
        message = str(exc.args[0])

    # 1) 参数错误
    for pat in _INVALID_PARAM_PATTERNS:
        if pat.search(message):
            return InvalidParameterError("tushare", message, original=exc)

    # 2) 限频
    for pat in _RATE_LIMIT_PATTERNS:
        if pat.search(message):
            return RateLimitError("tushare", message, retry_after=_extract_retry_seconds(message), original=exc)

    # 3) 权限/积分
    for pat in _QUOTA_PATTERNS:
        if pat.search(message):
            return QuotaExceededError("tushare", message, original=exc)

    # 4) 网络
    for pat in _NETWORK_PATTERNS:
        if pat.search(message):
            return NetworkError("tushare", message, original=exc)

    # 默认归类为网络错误（让上层可以切换到下一个源）
    return NetworkError("tushare", message, original=exc)
