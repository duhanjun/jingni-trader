import pandas as pd
import os
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class FactorStorage:
    """
    因子存储器
    """

    def __init__(self, config: Config):
        self.config = config
        os.makedirs(self.config.FACTOR_DATA_DIR, exist_ok=True)

    def save_factors(
        self,
        factors: pd.DataFrame,
        filename: str,
        incremental: bool = False
    ) -> str:
        """
        保存因子
        
        Args:
            factors: 因子 DataFrame
            filename: 文件名
            incremental: 是否增量更新
        
        Returns:
            保存路径
        """
        filepath = os.path.join(self.config.FACTOR_DATA_DIR, filename)
        
        if incremental and os.path.exists(filepath):
            existing = self.load_factors(filename)
            combined = pd.concat([existing, factors], ignore_index=True)
            combined = combined.drop_duplicates(subset=["code", "date"], keep="last")
            combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
            combined.to_parquet(filepath, index=False)
        else:
            factors.to_parquet(filepath, index=False)
        
        return filepath

    def load_factors(
        self,
        filename: str
    ) -> pd.DataFrame:
        """
        加载因子
        
        Args:
            filename: 文件名
        
        Returns:
            因子 DataFrame
        """
        filepath = os.path.join(self.config.FACTOR_DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            return pd.DataFrame()
        
        return pd.read_parquet(filepath)

    def list_factor_files(self) -> list:
        """
        列出所有因子文件
        """
        files = []
        for filename in os.listdir(self.config.FACTOR_DATA_DIR):
            if filename.endswith(".parquet"):
                files.append(filename)
        return sorted(files)
