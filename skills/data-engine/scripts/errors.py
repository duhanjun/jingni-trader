"""
数据源异常类定义

当某个数据源不可用时（积分不足 / 限频 / 网络故障等），
抛出一个明确的异常类型，DataEngine 据此判断是否切换到下一个数据源。
"""
from typing import Optional


class DataSourceError(Exception):
    """数据源通用错误基类"""
    def __init__(self, source: str, message: str, retriable: bool = True, original: Optional[Exception] = None):
        self.source = source
        self.message = message
        self.retriable = retriable  # 是否值得在当前源上重试
        self.original = original
        super().__init__(f"[{source}] {message}")


class QuotaExceededError(DataSourceError):
    """
    权限 / 积分不足错误（例如 tushare 错误码 40203）

    典型消息：
    - "您没有权限访问该接口"
    - "积分不足，请充值"
    - "需要更高权限"
    - "抱歉，您访问接口(stock_basic)频率超限" 实际 40203 在某些版本中表示权限
    """
    def __init__(self, source: str, message: str, original: Optional[Exception] = None):
        super().__init__(source, message, retriable=False, original=original)


class RateLimitError(DataSourceError):
    """
    频率限制错误（例如 tushare 错误码 40201）

    典型消息：
    - "1次/小时"
    - "1次/分钟"
    - "5次/天"
    - "访问频率超限"
    - "每分钟最多 XXX 次"
    """
    def __init__(self, source: str, message: str, retry_after: Optional[int] = None, original: Optional[Exception] = None):
        self.retry_after = retry_after  # 建议等待秒数（从错误消息解析）
        super().__init__(source, message, retriable=False, original=original)


class NetworkError(DataSourceError):
    """网络错误（连接超时、DNS 失败等）"""
    def __init__(self, source: str, message: str, original: Optional[Exception] = None):
        super().__init__(source, message, retriable=False, original=original)


class InvalidParameterError(DataSourceError):
    """参数错误（不应切换数据源，因为下一家也会失败）"""
    def __init__(self, source: str, message: str, original: Optional[Exception] = None):
        super().__init__(source, message, retriable=False, original=original)


class BlacklistedError(DataSourceError):
    """
    服务器黑名单错误（IP/账号/Token 被禁用）

    典型场景：
    - 短时间内请求过于频繁，IP 被临时拉黑
    - Token 长期滥用被永久封禁
    - GeoIP 限制（境外 IP 访问受限）

    典型消息：
    - "您的IP访问过于频繁，已被临时限制"
    - "access denied"
    - "forbidden"
    - "IP banned"
    """
    def __init__(self, source: str, message: str, original: Optional[Exception] = None):
        super().__init__(source, message, retriable=False, original=original)


class DataNotFoundError(DataSourceError):
    """
    标的未覆盖错误（数据源不提供此标的数据）

    典型场景：
    - Baostock 不支持 ETF/可转债/部分港股
    - AkShare 的某接口对特定品种未维护
    - WebSearch 搜索引擎无相关数据
    - 上市公司已退市/代码错误

    典型消息：
    - "no data found for symbol XXX"
    - "该代码不存在"
    - "no matching record"
    """
    def __init__(self, source: str, message: str, original: Optional[Exception] = None):
        super().__init__(source, message, retriable=False, original=original)


# 触发降级到下一源的异常类型集合
# 凡是该集合内的异常都应触发降级
FALLBACK_TRIGGERING_ERRORS = (
    QuotaExceededError,
    RateLimitError,
    NetworkError,
    BlacklistedError,
    DataNotFoundError,
    InvalidParameterError,  # token/auth 相关的 InvalidParameterError 也会触发（engine 内部判断）
)
