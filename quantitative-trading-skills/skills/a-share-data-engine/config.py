
import os
from typing import Optional, Dict, Any

try:
    # 尝试导入项目根目录的统一环境配置模块
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from env_config import get_env_config
    HAS_ENV_CONFIG = True
except ImportError:
    HAS_ENV_CONFIG = False


class Config:
    """
    配置类 - 支持从统一环境配置模块或直接从环境变量读取
    """

    def __init__(self, **kwargs):
        if HAS_ENV_CONFIG:
            # 使用统一环境配置模块
            env = get_env_config()
            self.DATA_BACKEND: str = kwargs.get("DATA_BACKEND", env.get_str("DATA_BACKEND", "tushare"))
            self.DATA_DIR: str = kwargs.get("DATA_DIR", env.get_str("DATA_DIR", "./data"))
            self.TUSHARE_TOKEN: Optional[str] = kwargs.get("TUSHARE_TOKEN", env.get_str("TUSHARE_TOKEN"))
            self.GM_TOKEN: Optional[str] = kwargs.get("GM_TOKEN", env.get_str("GM_TOKEN"))
            
            self.DB_TYPE: str = kwargs.get("DB_TYPE", env.get_str("DB_TYPE", "sqlite"))
            self.DB_HOST: str = kwargs.get("DB_HOST", env.get_str("DB_HOST", "localhost"))
            self.DB_PORT: int = kwargs.get("DB_PORT", env.get_int("DB_PORT", 3306))
            self.DB_NAME: str = kwargs.get("DB_NAME", env.get_str("DB_NAME", "a_share_data"))
            self.DB_USER: Optional[str] = kwargs.get("DB_USER", env.get_str("DB_USER"))
            self.DB_PASSWORD: Optional[str] = kwargs.get("DB_PASSWORD", env.get_str("DB_PASSWORD"))
        else:
            # 回退到原始行为，直接从环境变量读取
            self.DATA_BACKEND: str = kwargs.get("DATA_BACKEND", os.environ.get("DATA_BACKEND", "tushare"))
            self.DATA_DIR: str = kwargs.get("DATA_DIR", os.environ.get("DATA_DIR", "./data"))
            self.TUSHARE_TOKEN: Optional[str] = kwargs.get("TUSHARE_TOKEN", os.environ.get("TUSHARE_TOKEN"))
            self.GM_TOKEN: Optional[str] = kwargs.get("GM_TOKEN", os.environ.get("GM_TOKEN"))
            
            self.DB_TYPE: str = kwargs.get("DB_TYPE", "sqlite")
            self.DB_HOST: str = kwargs.get("DB_HOST", "localhost")
            self.DB_PORT: int = kwargs.get("DB_PORT", 3306)
            self.DB_NAME: str = kwargs.get("DB_NAME", "a_share_data")
            self.DB_USER: Optional[str] = kwargs.get("DB_USER", os.environ.get("DB_USER"))
            self.DB_PASSWORD: Optional[str] = kwargs.get("DB_PASSWORD", os.environ.get("DB_PASSWORD"))
        
        self.CLEANING_ENABLED: bool = kwargs.get("CLEANING_ENABLED", True)
        self.FILTER_ST: bool = kwargs.get("FILTER_ST", False)
        self.FILTER_NEW_STOCK_DAYS: int = kwargs.get("FILTER_NEW_STOCK_DAYS", 60)
        self.ADJ_TYPE: str = kwargs.get("ADJ_TYPE", "qfq")


def get_config(**kwargs) -> Config:
    """
    获取配置实例
    
    Args:
        **kwargs: 可选的配置覆盖
        
    Returns:
        Config 实例
    """
    return Config(**kwargs)


def get_db_url(config: Config) -> str:
    """
    根据配置生成数据库连接 URL
    
    Args:
        config: Config 实例
        
    Returns:
        数据库连接 URL
    """
    if HAS_ENV_CONFIG:
        # 使用统一环境配置模块的数据库 URL 生成
        from env_config import get_env_config
        return get_env_config().get_db_url()
    else:
        # 回退到原始行为
        if config.DB_TYPE == "sqlite":
            db_path = os.path.join(config.DATA_DIR, f"{config.DB_NAME}.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite:///{db_path}"
        elif config.DB_TYPE == "mysql":
            return f"mysql+pymysql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        elif config.DB_TYPE == "postgresql":
            return f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        else:
            raise ValueError(f"Unsupported DB type: {config.DB_TYPE}")

