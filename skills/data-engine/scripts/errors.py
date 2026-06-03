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
