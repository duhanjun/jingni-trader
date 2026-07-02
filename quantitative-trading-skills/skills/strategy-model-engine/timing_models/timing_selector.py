import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


class TimingModel:
    """
    择时模型 - 基于分钟线技术指标的3日涨跌分类器
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: 分钟线数据，包含 open, high, low, close, volume
        
        Returns:
            包含技术指标的DataFrame
        """
        df = data.copy()
        
        # 移动平均线
        for period in [5, 10, 20, 60]:
            df[f"MA_{period}"] = df["close"].rolling(window=period).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = df["close"].ewm(span=12, adjust=False).mean()
        exp26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
        
        # 布林带
        df["BB_MID"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["BB_UPPER"] = df["BB_MID"] + (bb_std * 2)
        df["BB_LOWER"] = df["BB_MID"] - (bb_std * 2)
        
        # 波动率
        df["VOLATILITY"] = df["close"].pct_change().rolling(window=20).std()
        
        # 成交量指标
        df["VOLUME_MA_5"] = df["volume"].rolling(window=5).mean()
        df["VOLUME_MA_20"] = df["volume"].rolling(window=20).mean()
        df["VOLUME_RATIO"] = df["volume"] / df["VOLUME_MA_20"]
        
        # 价格动量
        for period in [1, 5, 10, 20]:
            df[f"MOMENTUM_{period}"] = df["close"].pct_change(periods=period)
        
        return df
    
    def _create_target(self, data: pd.DataFrame, forward_days: int = 3) -> pd.Series:
        """
        创建目标变量 - 3日涨跌
        
        Args:
            data: 分钟线数据
            forward_days: 预测天数
        
        Returns:
            目标变量（1表示涨，0表示跌）
        """
        # 假设每天有240分钟
        minutes_per_day = 240
        future_close = data["close"].shift(-forward_days * minutes_per_day)
        target = (future_close > data["close"]).astype(int)
        return target
    
    def prepare_features_and_target(self, data: pd.DataFrame, 
                                  forward_days: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征和目标
        
        Args:
            data: 分钟线数据
            forward_days: 预测天数
        
        Returns:
            (特征DataFrame, 目标Series)
        """
        if forward_days is None:
            forward_days = self.config.FORWARD_RETURN_DAYS
        
        df_with_indicators = self._calculate_technical_indicators(data)
        target = self._create_target(df_with_indicators, forward_days)
        
        # 去除包含NaN的行
        df_clean = df_with_indicators.dropna()
        target_clean = target.loc[df_clean.index]
        
        # 只保留特征列
        feature_cols = [col for col in df_clean.columns if col not in ["open", "high", "low", "close", "volume"]]
        X = df_clean[feature_cols]
        
        return X, target_clean
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
             params: Optional[Dict[str, Any]] = None) -> lgb.LGBMClassifier:
        """
        训练择时模型
        
        Args:
            X: 特征数据
            y: 目标变量
            params: 模型参数
        
        Returns:
            训练好的模型
        """
        if params is None:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "num_leaves": 31,
                "verbose": -1,
                "random_state": self.config.RANDOM_STATE
            }
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_scaled, y)
        
        self.model = model
        self.feature_names = X.columns.tolist()
        
        return model
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        预测
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        predictions = self.model.predict(X_scaled)
        return pd.Series(predictions, index=X.index, name="predicted_direction")
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        预测概率
        
        Args:
            X: 特征数据
        
        Returns:
            预测概率DataFrame
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        proba = self.model.predict_proba(X_scaled)
        return pd.DataFrame(proba, index=X.index, columns=["prob_down", "prob_up"])
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            X: 特征数据
            y: 真实标签
        
        Returns:
            评估指标字典
        """
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
