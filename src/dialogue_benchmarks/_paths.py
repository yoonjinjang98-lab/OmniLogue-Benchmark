import os
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"could not locate project root (no pyproject.toml found above {here})")


PROJECT_ROOT = _find_project_root()

# Code (adapter.py / method.py / metric.py + meta.json) lives in the repo and is
# version-controlled.
PLUGINS_ROOT = PROJECT_ROOT / "plugins"

# Raw data / caches / results are large and shared across the team on the HDD.
# DIALOGUE_BENCHMARKS_DATA overrides the default `<repo>/data` symlink -- this is
# what lets a git worktree (a second checkout of this repo) point at the same
# shared data without needing its own symlink.
_data_env = os.environ.get("DIALOGUE_BENCHMARKS_DATA")
DATA_ROOT = Path(_data_env).expanduser().resolve() if _data_env else PROJECT_ROOT / "data"
