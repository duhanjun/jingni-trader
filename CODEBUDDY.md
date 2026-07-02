# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Commands

**Install dependencies** (no `requirements.txt`/`pyproject.toml` exists; deps are listed inline in the script):
```bash
bash install.sh
```
Note: `install.sh` also runs `pip install -e .`, but there is no `setup.py`/`pyproject.toml`, so that step fails — install dependencies manually from the lists in `install.sh` / `README.md` if needed.

**Run the main pipeline** (natural-language intent → stages):
```bash
python engine.py -i "帮我用近3年A股数据做一个20日反转因子选股回测"
python engine.py --input "优化当前组合，最大回撤控制在15%以内" --output result.json
```

**Run the standalone MA20 bond-ETF backtest** (self-contained demo that wires sub-skills directly):
```bash
python run_bond_etf_ma20.py
```

**Run the integration test** (validates the data-engine fallback chain + synthetic fallback):
```bash
python test_engine_v3.py
```
Caveat: this test hardcodes the path `/workspace` (e.g. `sys.path.insert(0, '/workspace')`, loads `skills/data-engine/scripts`). It only works when the repo is checked out such that `/workspace` resolves to the project root, or after editing those paths.

**Key environment variables**: `TUSHARE_TOKEN`, `GM_TOKEN`, `QUANT_WORK_DIR` (default `./workspace`), `DATA_BACKENDS` (fallback chain, default `tushare,baostock,akshare,websearch`), `BACKTEST_BACKEND` (default `native`), `TRADE_BACKEND`, `FACTOR_BACKEND`, `LOG_LEVEL`.

## Architecture

jingni-trader is a **master-orchestrator** for an A-share quant workflow. The top-level `engine.py` (`MasterEngine`) parses a natural-language intent, decides which stages to run, then drives seven independent sub-skills in pipeline order. It performs **no quant computation itself** — it only schedules, passes a shared `Context`, and records artifacts.

### Pipeline & stages
`STAGES = [IDLE, DATA, FACTOR, MODEL, BACKTEST, PORTFOLIO, EXECUTION, REPORT]`. `parse_intent()` maps Chinese/English keywords in the user input to an ordered `target_stages` list (see `SKILL_MODULES` for stage→module mapping). `run_pipeline()` iterates `target_stages`, calling `execute_stage()` for each.

**Breakpoint resume**: before invoking a sub-skill, `execute_stage()` checks whether `EXPECTED_ARTIFACTS[stage]` already exists on disk; if so it skips execution and reuses the cached artifact. This means re-running a finished pipeline short-circuits completed stages — delete files under `./workspace/<stage_dir>/` to force recomputation.

**Critical-stage semantics**: a failure in `DATA` or `BACKTEST` aborts the whole pipeline (`results["success"]=False`); other stage failures are recorded but do not stop the run.

### Shared state: `Context` and `RunArchiver`
- `scripts/context.py` — the `Context` dataclass is the single carrier threaded through every sub-skill. Stages read `stock_pool`, `start_date`, `end_date`, `external_data` and write back `artifacts[stage]` (path) and `metadata[stage]`. Add new cross-stage fields here.
- `scripts/archive.py` — `RunArchiver` writes a timestamped run folder under `workspace/archives/YYYYMMDD_HHMMSS/` with `step_N_<STAGE>/summary.md` + `artifacts/` and a top-level `pipeline_summary.md`.

### The seven sub-skills
Each lives in `skills/<name>/` and exposes `engine.py` with a uniform `run(ctx) -> {"success", "artifact_path", "metadata", "error"}` interface. They are loaded lazily via `importlib.import_module(SKILL_MODULES[stage])` and invoked as `skill_module.run(self.ctx)`. `DATA` is by far the most complex (see below); `EXECUTION`/`PORTFOLIO`/`MODEL` depend on heavy optional deps (xtquant/gm, cvxpy, lightgbm) that may be absent.

### Data engine (the core subsystem)
`skills/data-engine/engine.py` `DataEngine` implements a **precise multi-source fallback chain**: `tushare → baostock → akshare → websearch → synthetic`. Key parts:
- `_ADAPTER_REGISTRY`: maps backend name → `(module, class, kwargs)`. Adapters live in `skills/data-engine/scripts/adapters/` and subclass `BaseDataProvider` (`scripts/base/base_data_provider.py`), which mandates `get_daily()` and `get_stock_list()`.
- `scripts/errors.py` defines typed exceptions; `FALLBACK_TRIGGERING_ERRORS` + `config.DATA_FALLBACK_RULES` govern **why** a source downgrades (e.g. tushare degrades only on `QuotaExceededError`/`RateLimitError`, not on `DataNotFoundError`). `_should_fallback()` encodes this per-source logic.
- Final safety net: `ALLOW_SYNTHETIC_FALLBACK` → `_generate_synthetic_data()` returns statistically plausible but **fake** OHLCV (flagged via `engine.is_synthetic`), so the rest of the pipeline can still run for testing. Synthetic data must never be treated as tradeable.
- `external_data` (from agent built-in tools) is tried first and bypasses the adapter chain.

### ⚠️ Critical: the dual `scripts` namespace
There are **two distinct `scripts` packages**: the master one (`scripts/` with `config.py`, `context.py`, `archive.py`) and a **separate `scripts/` package inside every sub-skill** (`skills/<name>/scripts/`, each with its own `config.py`, `errors.py`, `base/`, `adapters/`). Sub-skill code does `from scripts.config import DEFAULT_DATA_SOURCES` expecting **its own** package, whose symbols differ from the master's (e.g. data-engine `scripts/config` has `DEFAULT_DATA_SOURCES`/`DATA_FALLBACK_RULES`, not `DATA_DIR`).

Both packages occupy the same `sys.modules['scripts']` slot, so **only one is active at a time**. Any code that imports or runs a sub-skill must first point `sys.modules['scripts']` at that sub-skill's package. The canonical, correct pattern is in `run_bond_etf_ma20.py`:
- `_register_scripts_package(skill_scripts_path)` registers a sub-skill's `scripts/` (and walks/preloads its submodules) into `sys.modules`.
- `use_skill(name)` is a context manager that swaps in the registered sub-skill `scripts.*` for the duration of a `with` block, then restores the master package.

`test_engine_v3.py` does the same via `sys.path.insert(0, '/workspace')` + `_register_scripts_package`. The master `engine.py` takes a lighter approach — it deletes `scripts.*` from `sys.modules` before `importlib`-importing the skill module, relying on a fresh import to rebind `scripts`. **When editing or invoking a sub-skill, always be explicit about which `scripts` package is active; never assume master config symbols exist inside a sub-skill or vice-versa.**

### Adding a data source / backend
Implement the adapter subclassing `BaseDataProvider`, register it in `_ADAPTER_REGISTRY`, and (for data sources) add its downgrade trigger to `DATA_FALLBACK_RULES` in `skills/data-engine/scripts/config.py` plus the backend to `SUPPORTED_BACKENDS`. For backtest frameworks, subclass the analogous base in `backtest-engine/scripts/` and register it there.

### Documentation
`SKILL.md` is the master skill descriptor (intent keywords, sub-skill map, data-source priority, archive layout). `references/` holds `config_guide.md`, `api_reference.md`, `context_protocol.md`, `context_schema.md`, `workflow_architecture.md`. README lists the CLI examples and the per-engine responsibilities.
