
import os
from typing import Optional, Dict, Any
from pathlib import Path


class EnvConfig:
    """
    统一环境配置管理类
    提供安全的环境变量读取、类型转换和默认值管理
    """
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file or ".env"
        self._load_env_file()
        
    def _load_env_file(self):
        """
        从 .env 文件加载环境变量
        """
        env_path = Path(self.env_file)
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if key and key not in os.environ:
                            os.environ[key] = value
    
    def get_str(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取字符串类型的环境变量
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值或默认值
        """
        return os.environ.get(key, default)
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        获取整数类型的环境变量
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值或默认值
        """
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
    
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        获取浮点数类型的环境变量
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值或默认值
        """
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        获取布尔类型的环境变量
        支持 true/false, yes/no, 1/0 (不区分大小写)
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值或默认值
        """
        value = os.environ.get(key)
        if value is None:
            return default
        value_lower = value.lower()
        if value_lower in ("true", "yes", "1", "on"):
            return True
        elif value_lower in ("false", "no", "0", "off"):
            return False
        return default
    
    def require_str(self, key: str, help_text: Optional[str] = None) -> str:
        """
        获取必填的字符串类型环境变量，缺失时抛出异常
        
        Args:
            key: 环境变量名
            help_text: 帮助文本，用于异常提示
            
        Returns:
            环境变量值
            
        Raises:
            ValueError: 当环境变量缺失时
        """
        value = self.get_str(key)
        if value is None:
            msg = f"必需的环境变量 {key} 未设置"
            if help_text:
                msg += f" - {help_text}"
            raise ValueError(msg)
        return value
    
    def get_db_url(self) -> str:
        """
        获取数据库连接 URL
        
        Returns:
            数据库连接 URL
        """
        db_type = self.get_str("DB_TYPE", "sqlite")
        data_dir = self.get_str("DATA_DIR", "./data")
        
        if db_type == "sqlite":
            db_name = self.get_str("DB_NAME", "a_share_data")
            db_path = Path(data_dir) / f"{db_name}.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path.absolute()}"
        elif db_type == "mysql":
            host = self.get_str("DB_HOST", "localhost")
            port = self.get_int("DB_PORT", 3306)
            db_name = self.require_str("DB_NAME", "请配置数据库名称")
            user = self.require_str("DB_USER", "请配置数据库用户名")
            password = self.require_str("DB_PASSWORD", "请配置数据库密码")
            return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
        elif db_type == "postgresql":
            host = self.get_str("DB_HOST", "localhost")
            port = self.get_int("DB_PORT", 5432)
            db_name = self.require_str("DB_NAME", "请配置数据库名称")
            user = self.require_str("DB_USER", "请配置数据库用户名")
            password = self.require_str("DB_PASSWORD", "请配置数据库密码")
            return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")
    
    def list_all(self) -> Dict[str, str]:
        """
        列出所有配置相关的环境变量（不包含敏感信息）
        
        Returns:
            环境变量字典
        """
        sensitive_keys = {"TUSHARE_TOKEN", "XCSC_TUSHARE_TOKEN", "GM_TOKEN", "DB_PASSWORD", "DB_USER"}
        result = {}
        for key in os.environ:
            if key in sensitive_keys:
                result[key] = "***" if os.environ[key] else ""
            elif any(prefix in key for prefix in ["DB_", "DATA_", "REPORT_", "FACTOR_", "BACKTEST_", "TA_", "PLOT_", "LOG_"]):
                result[key] = os.environ[key]
        return result


# 全局配置实例
_global_config: Optional[EnvConfig] = None


def get_env_config(env_file: Optional[str] = None) -> EnvConfig:
    """
    获取全局环境配置实例（单例模式）
    
    Args:
        env_file: 可选的 .env 文件路径
        
    Returns:
        EnvConfig 实例
    """
    global _global_config
    if _global_config is None:
        _global_config = EnvConfig(env_file)
    return _global_config


def reset_env_config():
    """
    重置全局配置实例（主要用于测试）
    """
    global _global_config
    _global_config = None

