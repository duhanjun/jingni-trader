import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
from sklearn.model_selection._split import _BaseKFold


class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """
    Purged Group Time Series Cross-Validator
    滚动窗口：36训练、12验证、12测试、间隔2天、行业分组
    
    实现了：
    - 滚动窗口划分
    - 训练集和验证集/测试集之间的间隔（purge）
    - 按行业分组确保同一行业的样本不会同时出现在训练和验证中
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 train_window: Optional[int] = None,
                 val_window: Optional[int] = None,
                 test_window: Optional[int] = None,
                 purge_gap: Optional[int] = None,
                 embargo: int = 0):
        """
        初始化
        
        Args:
            n_splits: 折数
            train_window: 训练窗口大小（单位：时间步）
            val_window: 验证窗口大小
            test_window: 测试窗口大小
            purge_gap: 训练集和验证集之间的间隔
            embargo: 验证集和测试集之间的间隔
        """
        super().__init__(n_splits, shuffle=False, random_state=None)
        
        # 使用配置文件中的默认值
        from config import get_config
        config = get_config()
        
        self.train_window = train_window if train_window is not None else config.CV_TRAIN_WINDOW
        self.val_window = val_window if val_window is not None else config.CV_VALID_WINDOW
        self.test_window = test_window if test_window is not None else config.CV_TEST_WINDOW
        self.purge_gap = purge_gap if purge_gap is not None else config.CV_PURGE_GAP
        self.embargo = embargo
    
    def split(self, 
             X: pd.DataFrame, 
             y: Optional[pd.Series] = None, 
             groups: Optional[pd.Series] = None,
             time_col: Optional[str] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        生成索引划分
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            groups: 分组Series（如行业）
            time_col: 时间列名，如果为None则使用索引
        
        Returns:
            每个折的 (train_idx, val_idx, test_idx) 列表
        """
        # 获取时间索引
        if time_col is not None:
            time_idx = X[time_col].values
        else:
            time_idx = np.arange(len(X))
        
        # 确保数据按时间排序
        sort_idx = np.argsort(time_idx)
        X_sorted = X.iloc[sort_idx]
        if groups is not None:
            groups_sorted = groups.iloc[sort_idx].values
        else:
            groups_sorted = None
        
        n_samples = len(X_sorted)
        splits = []
        
        # 计算每个窗口的起始点
        total_window = self.train_window + self.purge_gap + self.val_window + self.embargo + self.test_window
        
        for i in range(self.n_splits):
            # 计算当前折的起始位置
            start_idx = n_samples - (self.n_splits - i) * self.test_window - total_window
            if start_idx < 0:
                start_idx = 0
            
            # 训练集
            train_end = start_idx + self.train_window
            train_idx = np.arange(start_idx, train_end)
            
            # 验证集（跳过purge_gap）
            val_start = train_end + self.purge_gap
            val_end = val_start + self.val_window
            val_idx = np.arange(val_start, val_end)
            
            # 测试集（跳过embargo）
            test_start = val_end + self.embargo
            test_end = test_start + self.test_window
            test_idx = np.arange(test_start, test_end)
            
            # 确保不超出范围
            val_idx = val_idx[val_idx < n_samples]
            test_idx = test_idx[test_idx < n_samples]
            
            # 如果有分组，确保分组的完整性
            if groups_sorted is not None:
                train_groups = set(groups_sorted[train_idx])
                val_idx = val_idx[~np.isin(groups_sorted[val_idx], list(train_groups))]
                test_idx = test_idx[~np.isin(groups_sorted[test_idx], list(train_groups))]
            
            # 恢复原始索引顺序
            train_idx_orig = sort_idx[train_idx]
            val_idx_orig = sort_idx[val_idx]
            test_idx_orig = sort_idx[test_idx]
            
            splits.append((train_idx_orig, val_idx_orig, test_idx_orig))
        
        return splits
    
    def get_n_splits(self, 
                    X: Optional[pd.DataFrame] = None, 
                    y: Optional[pd.Series] = None, 
                    groups: Optional[pd.Series] = None) -> int:
        """
        获取折数
        
        Returns:
            折数
        """
        return self.n_splits
    
    def visualize_splits(self, 
                        X: pd.DataFrame, 
                        time_col: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8)):
        """
        可视化划分
        
        Args:
            X: 特征DataFrame
            time_col: 时间列名
            figsize: 图表大小
        """
        import matplotlib.pyplot as plt
        
        # 获取时间索引
        if time_col is not None:
            time_idx = X[time_col].values
        else:
            time_idx = np.arange(len(X))
        
        # 生成划分
        splits = self.split(X, time_col=time_col)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制每个折
        for i, (train_idx, val_idx, test_idx) in enumerate(splits):
            # 训练集
            ax.scatter(time_idx[train_idx], [i] * len(train_idx), 
                      c='blue', label='Train' if i == 0 else "", s=10)
            # 验证集
            ax.scatter(time_idx[val_idx], [i] * len(val_idx), 
                      c='orange', label='Val' if i == 0 else "", s=10)
            # 测试集
            ax.scatter(time_idx[test_idx], [i] * len(test_idx), 
                      c='green', label='Test' if i == 0 else "", s=10)
        
        ax.set_yticks(range(self.n_splits))
        ax.set_ylabel('Fold')
        ax.set_xlabel('Time')
        ax.set_title('Purged Group Time Series Split')
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def apply_split_to_data(self, 
                           X: pd.DataFrame, 
                           y: pd.Series, 
                           fold_idx: int,
                           groups: Optional[pd.Series] = None,
                           time_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, 
                                                                     pd.DataFrame, pd.Series,
                                                                     pd.DataFrame, pd.Series]:
        """
        应用划分到数据
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            fold_idx: 折索引
            groups: 分组Series
            time_col: 时间列名
        
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        splits = self.split(X, groups=groups, time_col=time_col)
        train_idx, val_idx, test_idx = splits[fold_idx]
        
        return (
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx], y.iloc[val_idx],
            X.iloc[test_idx], y.iloc[test_idx]
        )
