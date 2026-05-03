"""
数据获取模块
支持多种数据源：yfinance、akshare、tushare等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict


class DataLoader:
    """
    数据加载器
    统一接口获取各类金融市场数据
    """
    
    def __init__(self):
        self._yfinance_available = False
        self._akshare_available = False
        self._tushare_available = False
        
        # 尝试导入数据源库
        try:
            import yfinance as yf
            self._yf = yf
            self._yfinance_available = True
        except ImportError:
            pass
            
        try:
            import akshare as ak
            self._ak = ak
            self._akshare_available = True
        except ImportError:
            pass
            
        try:
            import tushare as ts
            self._ts = ts
            self._tushare_available = True
        except ImportError:
            pass
    
    def get_yahoo_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1d"
    ) -> pd.DataFrame:
        """
        从Yahoo Finance获取数据
        
        参数:
            symbol: 股票代码 (如 "AAPL", "600000.SS")
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            period: 时间周期 ("1d", "1wk", "1mo")
            
        返回:
            pd.DataFrame: OHLCV数据
        """
        if not self._yfinance_available:
            raise ImportError("yfinance库未安装，请运行: pip install yfinance")
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        data = self._yf.download(
            symbol,
            start=start_date,
            end=end_date,
            period=period
        )
        
        # 标准化列名
        data.columns = [col.lower() for col in data.columns]
        
        return data
    
    def get_akshare_stock(
        self,
        symbol: str,
        period: str = "daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        从AKShare获取A股数据
        
        参数:
            symbol: 股票代码 (如 "sh600000")
            period: 周期 ("daily", "weekly", "monthly")
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            pd.DataFrame: OHLCV数据
        """
        if not self._akshare_available:
            raise ImportError("akshare库未安装，请运行: pip install akshare")
        
        try:
            if period == "daily":
                data = self._ak.stock_zh_a_hist(
                    symbol=symbol.replace("sh", "").replace("sz", ""),
                    period="daily",
                    start_date=start_date.replace("-", "") if start_date else "",
                    end_date=end_date.replace("-", "") if end_date else "",
                    adjust="qfq"
                )
            else:
                data = self._ak.stock_zh_a_hist(
                    symbol=symbol.replace("sh", "").replace("sz", ""),
                    period=period,
                    adjust="qfq"
                )
            
            # 格式化数据
            if not data.empty:
                data = data.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount"
                })
                data["date"] = pd.to_datetime(data["date"])
                data = data.set_index("date")
            
            return data
        except Exception as e:
            print(f"获取AKShare数据失败: {e}")
            return pd.DataFrame()
    
    def get_tushare_data(
        self,
        symbol: str,
        token: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        从Tushare获取数据（需要token）
        
        参数:
            symbol: 股票代码
            token: Tushare token
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            pd.DataFrame: 数据
        """
        if not self._tushare_available:
            raise ImportError("tushare库未安装，请运行: pip install tushare")
        
        if token:
            self._ts.set_token(token)
        
        try:
            pro = self._ts.pro_api()
            data = pro.daily(
                ts_code=symbol,
                start_date=start_date.replace("-", "") if start_date else "",
                end_date=end_date.replace("-", "") if end_date else ""
            )
            
            if not data.empty:
                data["trade_date"] = pd.to_datetime(data["trade_date"])
                data = data.set_index("trade_date")
                data = data.sort_index()
            
            return data
        except Exception as e:
            print(f"获取Tushare数据失败: {e}")
            return pd.DataFrame()
    
    def get_multi_symbols(
        self,
        symbols: List[str],
        source: str = "yahoo",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据
        
        参数:
            symbols: 股票代码列表
            source: 数据源 ("yahoo", "akshare", "tushare")
            **kwargs: 其他参数
            
        返回:
            Dict[str, pd.DataFrame]: 股票代码到数据的映射
        """
        result = {}
        
        for symbol in symbols:
            try:
                if source == "yahoo":
                    data = self.get_yahoo_data(symbol, **kwargs)
                elif source == "akshare":
                    data = self.get_akshare_stock(symbol, **kwargs)
                elif source == "tushare":
                    data = self.get_tushare_data(symbol, **kwargs)
                else:
                    raise ValueError(f"不支持的数据源: {source}")
                
                if not data.empty:
                    result[symbol] = data
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
        
        return result
    
    def get_returns(
        self,
        data: pd.DataFrame,
        price_col: str = "close",
        periods: int = 1
    ) -> pd.Series:
        """
        计算收益率
        
        参数:
            data: OHLC数据
            price_col: 价格列名
            periods: 周期数
            
        返回:
            pd.Series: 收益率序列
        """
        returns = data[price_col].pct_change(periods)
        return returns
    
    def get_available_sources(self) -> List[str]:
        """
        获取可用的数据源列表
        
        返回:
            List[str]: 可用数据源名称
        """
        sources = []
        if self._yfinance_available:
            sources.append("yahoo")
        if self._akshare_available:
            sources.append("akshare")
        if self._tushare_available:
            sources.append("tushare")
        return sources
