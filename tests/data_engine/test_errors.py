"""data-engine L2 单元测试：tushare_error_classifier + errors 异常体系。

来源：data-engine/scripts/tushare_error_classifier.py 与 errors.py。

覆盖：
- classify_tushare_error 对限频/权限/网络/参数 4 类异常的归类
- _extract_retry_seconds 从消息中提取等待秒数
- DataSourceError 异常类的属性传递（source/message/retriable/original）
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_DIR = os.path.join(ROOT, "skills", "data-engine")


def _load_module(rel_path: str, mod_name: str):
    """显式加载 data-engine/scripts 下的子模块为独立模块。

    data-engine/scripts/tushare_error_classifier.py 用了 `from .errors import ...`
    相对导入，所以必须以 scripts.errors / scripts.tushare_error_classifier 的
形式加载，而不是裸文件加载。

    关键：已加载的模块直接复用，避免重复加载产生不同的类对象
    （否则 isinstance 检查会失败，因为两个 errors 模块会各自创建不同的类）。
    """
    # 计算模块在 scripts 包下的相对路径
    # rel_path 如 "scripts/tushare_error_classifier.py" → 模块名 scripts.tushare_error_classifier
    sub_path = rel_path.replace("scripts/", "").replace(".py", "")
    full_mod_name = f"scripts.{sub_path}"

    # 已加载过则直接返回，确保所有测试引用同一个模块对象
    if full_mod_name in sys.modules:
        return sys.modules[full_mod_name]

    # 确保 scripts 包已加载（仅首次加载时执行）
    scripts_dir = os.path.join(DATA_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if "scripts" not in sys.modules:
        if os.path.exists(init_py):
            spec = ilu.spec_from_file_location(
                "scripts", init_py,
                submodule_search_locations=[scripts_dir],
            )
            pkg = ilu.module_from_spec(spec)
            # 显式设置 __path__，确保相对导入能找到子模块
            pkg.__path__ = [scripts_dir]
            sys.modules["scripts"] = pkg
            spec.loader.exec_module(pkg)
        else:
            # 没有 __init__.py 也要能作为命名空间包工作
            import types
            pkg = types.ModuleType("scripts")
            pkg.__path__ = [scripts_dir]
            sys.modules["scripts"] = pkg

    # mock 重型依赖（仅在首次加载时注入）
    for _m in ("tushare", "baostock", "akshare", "xtquant"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    # 对于依赖 .errors 的模块，预加载 errors.py 以确保相对导入可用
    if sub_path != "errors":
        errors_mod_name = "scripts.errors"
        if errors_mod_name not in sys.modules:
            errors_path = os.path.join(scripts_dir, "errors.py")
            if os.path.exists(errors_path):
                _load_module("scripts/errors.py", "")

    target_path = os.path.join(DATA_ENGINE_DIR, rel_path)
    spec = ilu.spec_from_file_location(
        full_mod_name, target_path,
        submodule_search_locations=None,  # 子模块不需要再 search
    )
    mod = ilu.module_from_spec(spec)
    sys.modules[full_mod_name] = mod
    # 关键：设置 __package__ 让相对导入 from .errors 能解析
    mod.__package__ = "scripts"
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestClassifyTushareError:
    """验证 classify_tushare_error 对 4 类异常的归类。"""

    def test_classify_rate_limit(self):
        """限频消息 → RateLimitError"""
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        errors_mod = _load_module("scripts/errors.py", "")

        exc = Exception("抱歉，您每天最多访问该接口 1次/小时")
        result = classifier.classify_tushare_error(exc)
        assert isinstance(result, errors_mod.RateLimitError)
        assert result.source == "tushare"
        assert result.retry_after == 3600  # 1次/小时 → 3600秒

    def test_classify_rate_limit_per_minute(self):
        """1次/分钟 → 60秒等待"""
        classifier = _load_module("scripts/tushare_error_classifier.py", "")

        exc = Exception("访问频率超限，每分钟 1次/分钟")
        result = classifier.classify_tushare_error(exc)
        assert result.retry_after == 60

    def test_classify_quota_exceeded(self):
        """权限/积分不足 → QuotaExceededError"""
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        errors_mod = _load_module("scripts/errors.py", "")

        exc = Exception("抱歉，您积分不足，没有权限访问该接口")
        result = classifier.classify_tushare_error(exc)
        assert isinstance(result, errors_mod.QuotaExceededError)
        assert result.source == "tushare"

    def test_classify_invalid_parameter(self):
        """参数错误 → InvalidParameterError（不切换数据源）"""
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        errors_mod = _load_module("scripts/errors.py", "")

        exc = Exception("参数错误：ts_code 不合法")
        result = classifier.classify_tushare_error(exc)
        assert isinstance(result, errors_mod.InvalidParameterError)
        assert result.retriable is False  # 参数错误不应重试

    def test_classify_network_error(self):
        """网络错误 → NetworkError"""
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        errors_mod = _load_module("scripts/errors.py", "")

        exc = Exception("ConnectionError: connection timed out")
        result = classifier.classify_tushare_error(exc)
        assert isinstance(result, errors_mod.NetworkError)

    def test_classify_unknown_falls_to_network(self):
        """未知错误 → 默认归类为 NetworkError（让上层可以切换源）"""
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        errors_mod = _load_module("scripts/errors.py", "")

        exc = Exception("unknown error xyz")
        result = classifier.classify_tushare_error(exc)
        assert isinstance(result, errors_mod.NetworkError)

    def test_priority_invalid_param_over_rate_limit(self):
        """优先级：参数错误 > 限频"""
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        errors_mod = _load_module("scripts/errors.py", "")

        # 同时含"参数错误"和"频率超限" → 应判为参数错误
        exc = Exception("参数错误，访问频率超限")
        result = classifier.classify_tushare_error(exc)
        assert isinstance(result, errors_mod.InvalidParameterError)


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestExtractRetrySeconds:
    """验证 _extract_retry_seconds 提取等待秒数。"""

    def test_extract_per_hour(self):
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        assert classifier._extract_retry_seconds("1次/小时") == 3600
        assert classifier._extract_retry_seconds("2 次/小时") == 3600

    def test_extract_per_minute(self):
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        assert classifier._extract_retry_seconds("1次/分钟") == 60

    def test_extract_per_day(self):
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        assert classifier._extract_retry_seconds("1次/天") == 86400

    def test_extract_returns_none_when_no_pattern(self):
        classifier = _load_module("scripts/tushare_error_classifier.py", "")
        assert classifier._extract_retry_seconds("普通错误消息") is None


@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestDataSourceErrorHierarchy:
    """验证 DataSourceError 异常类的属性传递。"""

    def test_data_source_error_attributes(self):
        errors_mod = _load_module("scripts/errors.py", "")

        original = ValueError("原始异常")
        err = errors_mod.DataSourceError("tushare", "test msg", retriable=False, original=original)
        assert err.source == "tushare"
        assert err.message == "test msg"
        assert err.retriable is False
        assert err.original is original

    def test_quota_exceeded_inherits_data_source_error(self):
        errors_mod = _load_module("scripts/errors.py", "")
        err = errors_mod.QuotaExceededError("tushare", "积分不足")
        assert isinstance(err, errors_mod.DataSourceError)
        # QuotaExceededError 默认 retriable=False（切换源，不重试同源）
        assert err.retriable is False

    def test_rate_limit_has_retry_after(self):
        errors_mod = _load_module("scripts/errors.py", "")
        err = errors_mod.RateLimitError("tushare", "限频", retry_after=3600)
        assert err.retry_after == 3600

    def test_invalid_parameter_not_retriable(self):
        errors_mod = _load_module("scripts/errors.py", "")
        err = errors_mod.InvalidParameterError("tushare", "参数错误")
        assert err.retriable is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
