import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from dialogue_benchmarks._paths import DATA_ROOT, PLUGINS_ROOT

PLUGINS_DIR = PLUGINS_ROOT / "datasets"
RAW_DIR = DATA_ROOT / "raw"

# not real benchmarks -- e.g. conversation_chronicles_exp only contains symlinks
# into an unrelated project (LTPD), kept on disk but excluded from the registry.
EXCLUDED = {"conversation_chronicles_exp"}


@dataclass
class DatasetInfo:
    name: str
    plugin_path: Path  # code: <repo>/plugins/datasets/<name>/ (meta.json, adapter.py)
    raw_path: Path  # data: <DATA_ROOT>/raw/<name>/ (original files, cache/)
    meta: dict
    has_adapter: bool

    @property
    def tier(self):
        return self.meta.get("tier")

    @property
    def axes(self):
        return self.meta.get("axes", [])

    @property
    def task_type(self):
        return self.meta.get("task_type")


def _load_meta(plugin_path: Path) -> dict:
    with open(plugin_path / "meta.json") as f:
        return json.load(f)


def list_datasets() -> list[DatasetInfo]:
    infos = []
    if not PLUGINS_DIR.exists():
        return infos
    for d in sorted(PLUGINS_DIR.iterdir()):
        if not d.is_dir() or d.name in EXCLUDED:
            continue
        if not (d / "meta.json").exists():
            continue
        infos.append(
            DatasetInfo(
                name=d.name,
                plugin_path=d,
                raw_path=RAW_DIR / d.name,
                meta=_load_meta(d),
                has_adapter=(d / "adapter.py").exists(),
            )
        )
    return infos


def get_dataset_info(name: str) -> DatasetInfo:
    plugin_path = PLUGINS_DIR / name
    meta_path = plugin_path / "meta.json"
    if name in EXCLUDED or not meta_path.exists():
        raise KeyError(f"'{name}' is not a registered dataset (no plugins/datasets/{name}/meta.json)")
    return DatasetInfo(
        name=name,
        plugin_path=plugin_path,
        raw_path=RAW_DIR / name,
        meta=_load_meta(plugin_path),
        has_adapter=(plugin_path / "adapter.py").exists(),
    )


def _import_adapter(name: str):
    info = get_dataset_info(name)
    adapter_path = info.plugin_path / "adapter.py"
    if not adapter_path.exists():
        raise NotImplementedError(
            f"dataset '{name}' has no adapter yet (plugins/datasets/{name}/adapter.py missing). "
            f"See README.md for how to add one."
        )
    module_name = f"dialogue_benchmarks._adapters.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # so pickle/importlib can resolve dataclasses defined here
    spec.loader.exec_module(module)
    return module
