"""
conftest.py for skills/quant_optimizations/

自动设置 sys.path，确保：
1. /workspace 在路径中（用于 import skills.quant_optimizations.xxx）
2. skills/backtest-engine 在路径中（用于 import scripts.adapters.xxx）
3. scripts 包被正确注册
"""
import os
import sys
import importlib.util
import importlib

# 计算 /workspace 路径
# 当前文件: /workspace/skills/quant_optimizations/conftest.py
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 确保 /workspace 在 sys.path 中
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

# 注册 scripts 包（来自 skills/backtest-engine/scripts/）
# 与 test_engine_v3.py 中的 _register_scripts_package 等价
def _register_scripts_package():
    skill_scripts_path = os.path.join(WORKSPACE, "skills", "backtest-engine", "scripts")
    if not os.path.isdir(skill_scripts_path):
        return

    init_py = os.path.join(skill_scripts_path, "__init__.py")
    if not os.path.exists(init_py):
        return

    if "scripts" in sys.modules:
        return  # 已注册

    spec = importlib.util.spec_from_file_location(
        "scripts", init_py,
        submodule_search_locations=[skill_scripts_path],
    )
    scripts_pkg = importlib.util.module_from_spec(spec)
    sys.modules["scripts"] = scripts_pkg
    spec.loader.exec_module(scripts_pkg)

    # 注册子包
    for root, dirs, files in os.walk(skill_scripts_path):
        rel = os.path.relpath(root, skill_scripts_path)
        if rel == ".":
            package_prefix = "scripts"
        else:
            package_prefix = "scripts." + rel.replace(os.sep, ".")

        if package_prefix != "scripts":
            parent_pkg, _, _ = package_prefix.rpartition(".")
            if parent_pkg and parent_pkg not in sys.modules:
                parts = rel.split(os.sep)[:-1]
                parent_rel = os.path.join(*parts) if parts else ""
                if parent_rel:
                    parent_path = os.path.join(skill_scripts_path, parent_rel)
                    parent_init = os.path.join(parent_path, "__init__.py")
                    if os.path.exists(parent_init):
                        p_spec = importlib.util.spec_from_file_location(
                            parent_pkg, parent_init,
                            submodule_search_locations=[parent_path],
                        )
                        p_pkg = importlib.util.module_from_spec(p_spec)
                        sys.modules[parent_pkg] = p_pkg
                        p_spec.loader.exec_module(p_pkg)

        init_file = os.path.join(root, "__init__.py")
        if os.path.exists(init_file) and package_prefix not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                package_prefix, init_file,
                submodule_search_locations=[root],
            )
            pkg = importlib.util.module_from_spec(spec)
            sys.modules[package_prefix] = pkg
            spec.loader.exec_module(pkg)


_register_scripts_package()
