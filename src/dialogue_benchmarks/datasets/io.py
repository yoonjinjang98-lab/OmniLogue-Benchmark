import pickle
from pathlib import Path

from dialogue_benchmarks.datasets.registry import get_dataset_info


def save_variant(name: str, data, filename: str) -> Path:
    """Save a user-customized version of a dataset into its cache/ folder.

    This is the only way cache/ ever gets populated -- plain load_dataset(name)
    calls never write here.
    """
    info = get_dataset_info(name)
    path = info.raw_path / "cache" / f"{filename}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def load_variant(name: str, filename: str):
    info = get_dataset_info(name)
    path = info.raw_path / "cache" / f"{filename}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"no cached variant '{filename}' for dataset '{name}' at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
