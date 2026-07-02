import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional, List
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class SnapshotManager:
    """
    数据快照管理器
    """

    def __init__(self, config: Config):
        self.config = config
        self.snapshot_dir = os.path.join(config.DATA_DIR, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def create_snapshot(
        self,
        df: pd.DataFrame,
        name: str,
        description: str = "",
    ) -> str:
        """
        创建快照
        
        Args:
            df: 数据 DataFrame
            name: 快照名称
            description: 描述
        
        Returns:
            快照 ID
        """
        snapshot_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_id)
        os.makedirs(snapshot_path, exist_ok=True)

        data_path = os.path.join(snapshot_path, "data.parquet")
        df.to_parquet(data_path, index=False)

        meta = {
            "id": snapshot_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "rows": len(df),
            "columns": list(df.columns),
        }
        meta_path = os.path.join(snapshot_path, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> pd.DataFrame:
        """
        加载快照
        
        Args:
            snapshot_id: 快照 ID
        
        Returns:
            数据 DataFrame
        """
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_id)
        data_path = os.path.join(snapshot_path, "data.parquet")
        df = pd.read_parquet(data_path)
        return df

    def list_snapshots(self) -> List[dict]:
        """
        列出所有快照
        
        Returns:
            快照元数据列表
        """
        snapshots = []
        for item in os.listdir(self.snapshot_dir):
            item_path = os.path.join(self.snapshot_dir, item)
            if os.path.isdir(item_path):
                meta_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    snapshots.append(meta)
        return sorted(snapshots, key=lambda x: x["created_at"], reverse=True)

    def delete_snapshot(self, snapshot_id: str):
        """
        删除快照
        
        Args:
            snapshot_id: 快照 ID
        """
        import shutil
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_id)
        if os.path.exists(snapshot_path):
            shutil.rmtree(snapshot_path)
