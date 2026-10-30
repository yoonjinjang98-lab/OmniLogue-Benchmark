from typing import Optional

from dialogue_benchmarks.datasets.registry import _import_adapter, get_dataset_info


def load_dataset(name: str, variant: Optional[str] = None):
    """Load a dataset by name.

    variant=None (default): re-parses the raw files fresh every call via that
    dataset's adapter.py, and touches nothing on disk.

    variant="some_name": instead loads a user-saved customization from
    data/raw/<name>/cache/some_name.pkl (see save_variant / io.py).
    """
    if variant is not None:
        from dialogue_benchmarks.datasets.io import load_variant

        return load_variant(name, variant)

    info = get_dataset_info(name)
    adapter = _import_adapter(name)
    return adapter.load(info.raw_path)
