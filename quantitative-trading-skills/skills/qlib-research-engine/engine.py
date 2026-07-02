import os
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class QlibResearchEngine:
    """
    Qlib因子研究引擎
    """
    
    def __init__(self, config):
        self.config = config
        self._qlib_initialized = False
        
    def init_qlib(self, data_dir: Optional[str] = None) -> None:
        """
        初始化Qlib环境
        
        Args:
            data_dir: Qlib数据目录，默认为配置中的QLIB_DATA_DIR
        """
        try:
            import qlib
            qlib.init(
                provider_uri=data_dir or self.config.QLIB_DATA_DIR,
                region="cn"
            )
            self._qlib_initialized = True
            logger.info("Qlib环境初始化成功")
        except ImportError:
            logger.warning("Qlib未安装，请运行: pip install qlib")
            raise
        except Exception as e:
            logger.error(f"Qlib初始化失败: {e}")
            raise
    
    def get_alpha158_handler(
        self,
        start_time: str,
        end_time: str,
        instruments: Optional[List[str]] = None
    ) -> Any:
        """
        获取Alpha158因子处理器
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            instruments: 股票列表，默认为全市场
            
        Returns:
            Qlib数据处理器
        """
        if not self._qlib_initialized:
            self.init_qlib()
            
        from qlib.contrib.data.handler import Alpha158
        
        handler = Alpha158(
            start_time=start_time,
            end_time=end_time,
            instruments=instruments
        )
        return handler
    
    def get_alpha360_handler(
        self,
        start_time: str,
        end_time: str,
        instruments: Optional[List[str]] = None
    ) -> Any:
        """
        获取Alpha360因子处理器
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            instruments: 股票列表，默认为全市场
            
        Returns:
            Qlib数据处理器
        """
        if not self._qlib_initialized:
            self.init_qlib()
            
        from qlib.contrib.data.handler import Alpha360
        
        handler = Alpha360(
            start_time=start_time,
            end_time=end_time,
            instruments=instruments
        )
        return handler
    
    def train_model(
        self,
        handler: Any,
        model_type: str = "lightgbm",
        **kwargs
    ) -> Any:
        """
        训练多因子模型
        
        Args:
            handler: Qlib数据处理器
            model_type: 模型类型，支持 lightgbm, xgboost, lstm
            **kwargs: 额外的模型参数
            
        Returns:
            训练好的模型
        """
        if not self._qlib_initialized:
            self.init_qlib()
            
        if model_type.lower() == "lightgbm":
            from qlib.contrib.model.gbdt import LGBModel
            model = LGBModel(**kwargs)
        elif model_type.lower() == "xgboost":
            from qlib.contrib.model.xgboost import XGBModel
            model = XGBModel(**kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model.fit(handler)
        logger.info(f"{model_type}模型训练完成")
        return model
    
    def generate_signals(
        self,
        model: Any,
        handler: Any
    ) -> pd.DataFrame:
        """
        生成选股信号
        
        Args:
            model: 训练好的模型
            handler: Qlib数据处理器
            
        Returns:
            选股信号DataFrame
        """
        predictions = model.predict(handler)
        return predictions
    
    def analyze_ic(
        self,
        predictions: pd.DataFrame,
        handler: Any
    ) -> Dict[str, float]:
        """
        IC分析
        
        Args:
            predictions: 预测结果
            handler: Qlib数据处理器
            
        Returns:
            IC分析结果字典
        """
        from qlib.contrib.evaluate import risk_analysis
        
        if hasattr(handler, 'get_label'):
            labels = handler.get_label()
            analysis = risk_analysis(predictions, labels)
            return analysis
        return {}
    
    def analyze_with_shap(
        self,
        model: Any,
        handler: Any,
        sample_size: int = 1000
    ) -> Any:
        """
        SHAP解释性分析
        
        Args:
            model: 训练好的模型
            handler: Qlib数据处理器
            sample_size: 采样大小
            
        Returns:
            SHAP值
        """
        try:
            import shap
            
            data = handler.get_feature().sample(min(sample_size, len(handler)))
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(data)
            
            logger.info("SHAP分析完成")
            return shap_values
        except ImportError:
            logger.warning("shap未安装，请运行: pip install shap")
            return None
        except Exception as e:
            logger.error(f"SHAP分析失败: {e}")
            return None
    
    def get_factor_importance(
        self,
        model: Any
    ) -> pd.DataFrame:
        """
        获取因子重要性
        
        Args:
            model: 训练好的模型
            
        Returns:
            因子重要性DataFrame
        """
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            return pd.DataFrame({
                'feature': importance.index,
                'importance': importance.values
            }).sort_values('importance', ascending=False)
        return pd.DataFrame()
