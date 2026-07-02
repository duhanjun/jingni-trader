"""
集成测试：验证 v3 数据引擎的精准降级 + 模拟数据兜底

测试场景：
1. 加载 data-engine 子技能
2. 验证 _ADAPTER_REGISTRY 包含全部 7 个源
3. 验证 DATA_FALLBACK_RULES 表结构
4. 模拟所有源失败：仅启用 websearch（无注入）+ synthetic 兜底
5. 验证 _should_fallback 决策逻辑
6. 验证 _generate_synthetic_data 能产出合规 schema
"""
import os
import sys
import logging
import importlib.util
import importlib
import types

# 设置 sys.path 让 from scripts.xxx 能解析
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _register_scripts_package(skill_scripts_path: str):
    """与 run_bond_etf_ma20.py 中 use_skill 等价"""
    if not os.path.isdir(skill_scripts_path):
        return

    init_py = os.path.join(skill_scripts_path, '__init__.py')
    if not os.path.exists(init_py):
        return
    spec = importlib.util.spec_from_file_location(
        'scripts', init_py,
        submodule_search_locations=[skill_scripts_path],
    )
    scripts_pkg = importlib.util.module_from_spec(spec)
    sys.modules['scripts'] = scripts_pkg
    spec.loader.exec_module(scripts_pkg)

    for root, dirs, files in os.walk(skill_scripts_path):
        rel = os.path.relpath(root, skill_scripts_path)
        if rel == '.':
            package_prefix = 'scripts'
        else:
            package_prefix = 'scripts.' + rel.replace(os.sep, '.')

        if package_prefix != 'scripts':
            parent_pkg, _, _ = package_prefix.rpartition('.')
            if parent_pkg and parent_pkg not in sys.modules:
                parts = rel.split(os.sep)[:-1]
                parent_rel = os.path.join(*parts) if parts else ''
                parent_dir = os.path.join(skill_scripts_path, parent_rel) if parent_rel else skill_scripts_path
                parent_init = os.path.join(parent_dir, '__init__.py')
                if os.path.exists(parent_init):
                    pspec = importlib.util.spec_from_file_location(
                        parent_pkg, parent_init,
                        submodule_search_locations=[parent_dir],
                    )
                    pmod = importlib.util.module_from_spec(pspec)
                    sys.modules[parent_pkg] = pmod
                    pspec.loader.exec_module(pmod)

        for f in files:
            if f.endswith('.py') and f != '__init__.py':
                mod_name = f[:-3]
                full_name = f'{package_prefix}.{mod_name}'
                fpath = os.path.join(root, f)
                if full_name in sys.modules:
                    continue
                try:
                    fspec = importlib.util.spec_from_file_location(full_name, fpath)
                    fmod = importlib.util.module_from_spec(fspec)
                    sys.modules[full_name] = fmod
                    fspec.loader.exec_module(fmod)
                except Exception as e:
                    print(f"⚠️ 预加载失败 {full_name}: {e}")


# 加载 data-engine 子技能
DE_SCRIPTS = os.path.join(ROOT, 'skills', 'data-engine', 'scripts')
_register_scripts_package(DE_SCRIPTS)

import pandas as pd

# 现在可以 import scripts.* 形式的模块
from scripts import config, errors
from scripts.errors import (
    DataSourceError, QuotaExceededError, RateLimitError, NetworkError,
    BlacklistedError, DataNotFoundError, InvalidParameterError,
    FALLBACK_TRIGGERING_ERRORS,
)

# 引擎需要 import 时是顶层 `import` (from engine module)
# engine.py 自己 import scripts.* 是基于包内路径
# 重新切回主 scripts 包
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # 主 scripts 包要在 sys.path
# 重新把主 scripts 包注册到 sys.modules，覆盖子技能的
main_scripts_init = os.path.join(ROOT, 'scripts', '__init__.py')
if os.path.exists(main_scripts_init):
    spec = importlib.util.spec_from_file_location(
        'scripts', main_scripts_init,
        submodule_search_locations=[os.path.join(ROOT, 'scripts')],
    )
    main_pkg = importlib.util.module_from_spec(spec)
    sys.modules['scripts'] = main_pkg
    spec.loader.exec_module(main_pkg)
    # 再次预加载子技能 scripts 包（否则 from scripts.adapters.xxx 会找不到）
    _register_scripts_package(DE_SCRIPTS)

# 加载 engine.py
engine_path = os.path.join(ROOT, 'skills', 'data-engine', 'engine.py')
spec = importlib.util.spec_from_file_location('data_engine_main', engine_path)
engine_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_mod)

# 切回主 scripts
if os.path.exists(main_scripts_init):
    spec = importlib.util.spec_from_file_location(
        'scripts', main_scripts_init,
        submodule_search_locations=[os.path.join(ROOT, 'scripts')],
    )
    main_pkg = importlib.util.module_from_spec(spec)
    sys.modules['scripts'] = main_pkg
    spec.loader.exec_module(main_pkg)
    _register_scripts_package(DE_SCRIPTS)

DataEngine = engine_mod.DataEngine
_ADAPTER_REGISTRY = engine_mod._ADAPTER_REGISTRY


# ============== 实际测试 ==============

def test_registry():
    """测试 1：适配器注册表覆盖全部 7 个源"""
    print("\n=== 测试 1: 适配器注册表 ===")
    expected = {"tushare", "baostock", "akshare", "websearch", "xtquant", "gm", "tdxquant"}
    actual = set(_ADAPTER_REGISTRY.keys())
    print(f"  注册: {sorted(actual)}")
    assert expected == actual, f"缺少源: {expected - actual}"
    print("  ✓ 全部 7 个源都注册")


def test_fallback_rules():
    """测试 2：DATA_FALLBACK_RULES 表结构"""
    print("\n=== 测试 2: 降级条件表 ===")
    rules = config.DATA_FALLBACK_RULES
    for src, rule in rules.items():
        print(f"  • {src}: 触发 {rule.get('trigger_errors')}")
        print(f"          降级到 {rule.get('downgrade_to')}")
    # 关键源必须有降级规则
    for src in ["tushare", "baostock", "akshare", "websearch"]:
        assert src in rules, f"{src} 缺少降级规则"
    print("  ✓ 4 个默认源都有降级规则")


def test_should_fallback():
    """测试 3：_should_fallback 决策逻辑"""
    print("\n=== 测试 3: 降级决策逻辑 ===")
    # 创建一个临时引擎（不需要 provider，因为 _should_fallback 不依赖 provider）
    # 我们 mock 掉 __init__ 中的 self.provider
    class _T:
        pass
    tmp = _T()
    tmp.backend = "tushare"
    # 借用 DataEngine 的方法
    should = DataEngine._should_fallback

    # tushare 收到 QuotaExceededError → 应降级（baostock）
    e1 = QuotaExceededError("tushare", "积分不足")
    assert should(tmp, "tushare", e1), "tushare QuotaExceededError 应降级"
    print("  ✓ tushare + QuotaExceededError → 降级")

    # tushare 收到 RateLimitError → 应降级
    e2 = RateLimitError("tushare", "1次/小时")
    assert should(tmp, "tushare", e2), "tushare RateLimitError 应降级"
    print("  ✓ tushare + RateLimitError → 降级")

    # tushare 收到 DataNotFoundError → 不应降级（不在 tushare 允许列表）
    e3 = DataNotFoundError("tushare", "标的未覆盖")
    assert not should(tmp, "tushare", e3), "tushare DataNotFoundError 不在降级列表，不应降级"
    print("  ✓ tushare + DataNotFoundError → 不降级（不在触发列表）")

    # baostock 收到 DataNotFoundError → 应降级到 akshare
    e4 = DataNotFoundError("baostock", "未覆盖")
    assert should(tmp, "baostock", e4), "baostock DataNotFoundError 应降级"
    print("  ✓ baostock + DataNotFoundError → 降级")

    # baostock 收到 QuotaExceededError → 不应降级（不在 baostock 允许列表）
    e5 = QuotaExceededError("baostock", "积分")
    assert not should(tmp, "baostock", e5), "baostock QuotaExceededError 不在降级列表"
    print("  ✓ baostock + QuotaExceededError → 不降级")

    # websearch 收到 DataNotFoundError → 应降级（到 synthetic）
    e6 = DataNotFoundError("websearch", "无搜索结果")
    assert should(tmp, "websearch", e6), "websearch DataNotFoundError 应触发 synthetic 兜底"
    print("  ✓ websearch + DataNotFoundError → 触发 synthetic 兜底")

    # InvalidParameterError with token → 应降级
    e7 = InvalidParameterError("tushare", "您的token不对，请确认")
    assert should(tmp, "tushare", e7), "token 错误应降级"
    print("  ✓ tushare + InvalidParameterError(token) → 降级")

    # InvalidParameterError 普通 → 不应降级
    e8 = InvalidParameterError("tushare", "日期格式错误")
    assert not should(tmp, "tushare", e8), "普通参数错误不应降级"
    print("  ✓ tushare + InvalidParameterError(非token) → 不降级")


def test_synthetic_data():
    """测试 4：模拟数据生成"""
    print("\n=== 测试 4: 模拟数据生成 ===")
    # 创建一个不走网络初始化的引擎实例
    # 我们直接调用 _generate_synthetic_data 方法
    class _MockEngine:
        pass
    mock = _MockEngine()
    # 借用方法
    gen = DataEngine._generate_synthetic_data
    df = gen(mock, ["511090.SH", "000001.SZ"], "2024-06-01", "2024-12-31")
    print(f"  生成 {len(df)} 行, {df['code'].nunique()} 只标的")
    print(f"  列: {list(df.columns)}")
    print(f"  时间范围: {df['date'].min()} ~ {df['date'].max()}")
    # 校验 schema
    required = {"date", "code", "open", "high", "low", "close", "vol"}
    assert required.issubset(set(df.columns)), f"缺少列: {required - set(df.columns)}"
    assert df['code'].nunique() == 2
    assert (df['close'] > 0).all()
    print("  ✓ schema 合规，价格为正数，2 个标的都覆盖")


def test_full_fallback_chain():
    """测试 5：完整降级链走到 synthetic"""
    print("\n=== 测试 5: 完整降级链 → synthetic ===")
    # 我们构造一个完全无法工作的环境
    # 1) tushare: 无 token → 初始化失败
    # 2) baostock: 在没 baostock 包的容器里 → 初始化失败
    # 3) akshare: 类似
    # 4) websearch: 无 web_search_fn 注入 → 调用时报错
    # 5) 最终走 synthetic
    import os as _os
    # 确保 TUSHARE_TOKEN 没设
    _os.environ.pop("TUSHARE_TOKEN", None)

    # 因为 akshare/baostock 可能装了，我们用 monkey patch 让它们直接 raise
    import scripts.adapters as _adapters_pkg
    # 把 _ADAPTER_REGISTRY 中 akshare/baostock/tushare 的模块都替换成 raise 异常
    from scripts.errors import NetworkError, DataNotFoundError
    import pandas as pd

    class _BrokenAdapter:
        def __init__(self, *args, **kwargs):
            raise NetworkError("test", "test broken")

    # 直接替换 _ADAPTER_REGISTRY 中的工厂
    original = dict(_ADAPTER_REGISTRY)
    try:
        # 替换 websearch: 注入一个总是返回 "未找到" 的 fn
        def _always_not_found(_):
            return "未找到相关结果"

        _ADAPTER_REGISTRY["tushare"] = ("scripts.adapters.tushare_adapter", "_BrokenModule", {})
        # 给 tushare 加一个会报 NetworkError 的占位模块
        sys.modules["_broken_tushare"] = types.SimpleNamespace(TushareAdapter=_BrokenAdapter)
        # 用一个会 raise 的实际类
        class _RAISE:
            def __init__(self, *a, **k):
                raise NetworkError("tushare", "测试：模拟 tushare 不可用")
        # 直接 monkey-patch tushare_adapter
        ta_mod = sys.modules.get("scripts.adapters.tushare_adapter")
        if ta_mod is not None:
            _orig_Tushare = ta_mod.TushareAdapter
            ta_mod.TushareAdapter = _RAISE

        # baostock: 同样
        ba_mod = sys.modules.get("scripts.adapters.baostock_adapter")
        if ba_mod is not None:
            class _RAISE_BS:
                def __init__(self, *a, **k):
                    raise NetworkError("baostock", "测试：模拟 baostock 不可用")
            ba_mod.BaostockAdapter = _RAISE_BS

        # akshare: 同样
        ak_mod = sys.modules.get("scripts.adapters.akshare_adapter")
        if ak_mod is not None:
            class _RAISE_AK:
                def __init__(self, *a, **k):
                    raise NetworkError("akshare", "测试：模拟 akshare 不可用")
            ak_mod.AkshareAdapter = _RAISE_AK

        # websearch: 注入总是返回 "未找到"
        ws_mod = sys.modules.get("scripts.adapters.websearch_adapter")
        if ws_mod is not None:
            class _WebSearchNotFound:
                def __init__(self, web_search_fn=None):
                    self.web_search_fn = _always_not_found
                def get_daily(self, *a, **k):
                    raise DataNotFoundError("websearch", "测试：websearch 找不到")
            ws_mod.WebSearchAdapter = _WebSearchNotFound

        # 创建引擎
        engine = DataEngine(
            data_sources=["tushare", "baostock", "akshare", "websearch"],
        )
        print(f"  引擎初始化完成，backend = {engine.backend}")

        # 调 fetch_and_clean
        df = engine.fetch_and_clean(
            symbols=["511090.SH", "000001.SZ"],
            start_date="2024-06-01",
            end_date="2024-12-31",
        )
        print(f"  fetch_and_clean 返回 {len(df)} 行, is_synthetic = {engine.is_synthetic}")
        assert engine.is_synthetic, "应该走了 synthetic 兜底"
        assert engine.backend == "synthetic"
        assert len(df) > 0
        print("  ✓ 完整降级链 → synthetic 兜底成功")
    finally:
        # 恢复 monkey patch
        _ADAPTER_REGISTRY.clear()
        _ADAPTER_REGISTRY.update(original)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    test_registry()
    test_fallback_rules()
    test_should_fallback()
    test_synthetic_data()
    test_full_fallback_chain()

    print("\n🎉 全部测试通过")
