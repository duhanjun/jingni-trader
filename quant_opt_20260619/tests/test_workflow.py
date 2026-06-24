"""
Workflow YAML 配置模块单元测试
"""
import os
import sys
import unittest
import tempfile

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from quant_opt_20260619.workflow import WorkflowConfig, StageConfig


class TestWorkflowConfig(unittest.TestCase):
    def test_from_dict_basic(self):
        d = {
            "experiment_name": "csi300_momentum_v1",
            "market": "csi300",
            "start_date": "2021-01-01",
            "end_date": "2024-12-31",
            "stages": [
                {"data": {"provider": "tushare"}},
                {"factor": {"method": "ic_weighted"}},
                {"model": {"type": "lightgbm"}},
                {"backtest": {"backend": "native"}},
                {"report": {"format": "html"}},
            ],
        }
        cfg = WorkflowConfig.from_dict(d)
        self.assertEqual(cfg.experiment_name, "csi300_momentum_v1")
        self.assertEqual(len(cfg.stages), 5)
        self.assertEqual(cfg.stages[0].name, "DATA")
        self.assertEqual(cfg.stages[3].name, "BACKTEST")

    def test_from_yaml(self):
        d = {
            "experiment_name": "test_yaml",
            "stages": [{"data": {}}, {"backtest": {}}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(d, f)
            yaml_path = f.name
        try:
            cfg = WorkflowConfig.from_yaml(yaml_path)
            self.assertEqual(cfg.experiment_name, "test_yaml")
            self.assertEqual(len(cfg.stages), 2)
        finally:
            os.unlink(yaml_path)

    def test_to_jingni_intent(self):
        cfg = WorkflowConfig(
            experiment_name="x",
            market="csi300",
            start_date="2021-01-01",
            end_date="2024-12-31",
            stages=[
                StageConfig("DATA"),
                StageConfig("FACTOR"),
                StageConfig("MODEL"),
                StageConfig("BACKTEST"),
                StageConfig("REPORT"),
            ],
        )
        intent = cfg.to_jingni_intent()
        self.assertIn("数据获取", intent)
        self.assertIn("因子构建", intent)
        self.assertIn("回测", intent)
        self.assertIn("沪深300", intent)
        self.assertIn("2021-01-01", intent)

    def test_validate(self):
        cfg = WorkflowConfig(
            experiment_name="valid_name_1",
            stages=[StageConfig("DATA"), StageConfig("BACKTEST")],
        )
        self.assertEqual(cfg.validate(), [])

        bad = WorkflowConfig(
            experiment_name="",
            stages=[StageConfig("INVALID_STAGE")],
        )
        errs = bad.validate()
        self.assertGreater(len(errs), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)