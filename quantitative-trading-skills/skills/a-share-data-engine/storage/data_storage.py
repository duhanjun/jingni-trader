import pandas as pd
import os
from typing import Optional
from sqlalchemy import create_engine, Table, Column, String, Float, DateTime, Boolean, MetaData
from sqlalchemy.sql import text
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_db_url


class DataStorage:
    """
    数据存储类
    """

    def __init__(self, config: Config):
        self.config = config
        self.db_url = get_db_url(config)
        self.engine = create_engine(self.db_url)
        self.metadata = MetaData()
        self._init_tables()

    def _init_tables(self):
        """
        初始化数据表
        """
        Table(
            "daily_data",
            self.metadata,
            Column("code", String(20), primary_key=True),
            Column("date", DateTime, primary_key=True),
            Column("open", Float),
            Column("high", Float),
            Column("low", Float),
            Column("close", Float),
            Column("volume", Float),
            Column("amount", Float),
            Column("pre_close", Float),
            Column("change_pct", Float),
            Column("turnover_rate", Float),
            Column("is_st", Boolean),
            Column("is_limit_up", Boolean),
            Column("is_limit_down", Boolean),
            Column("is_suspended", Boolean),
        )
        self.metadata.create_all(self.engine, checkfirst=True)

    def save_daily(self, df: pd.DataFrame, if_exists: str = "replace"):
        """
        保存日线数据
        
        Args:
            df: 数据 DataFrame
            if_exists: 处理方式 'fail', 'replace', 'append'
        """
        if df.empty:
            return
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df.to_sql("daily_data", self.engine, if_exists=if_exists, index=False, chunksize=10000)

    def load_daily(
        self,
        codes: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载日线数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            数据 DataFrame
        """
        query = "SELECT * FROM daily_data WHERE 1=1"
        params = {}
        
        if codes:
            placeholders = ",".join([f":code{i}" for i in range(len(codes))])
            query += f" AND code IN ({placeholders})"
            params.update({f"code{i}": code for i, code in enumerate(codes)})
        
        if start_date:
            query += " AND date >= :start_date"
            params["start_date"] = start_date
        
        if end_date:
            query += " AND date <= :end_date"
            params["end_date"] = end_date
        
        query += " ORDER BY code, date"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        return df

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """
        保存为 CSV
        
        Args:
            df: 数据 DataFrame
            filename: 文件名
        """
        filepath = os.path.join(self.config.DATA_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """
        从 CSV 加载
        
        Args:
            filename: 文件名
        
        Returns:
            数据 DataFrame
        """
        filepath = os.path.join(self.config.DATA_DIR, filename)
        df = pd.read_csv(filepath)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
