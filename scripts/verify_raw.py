#!/usr/bin/env python3
"""Regenerate data/raw/_verify_report.json: per-dataset file inventory + hashes.

Full sha256 for files under PARTIAL_THRESHOLD; for larger files, a cheap
"partial:" hash (first/last chunk + size) so re-running this after a reorg
doesn't take forever on multi-GB pickles/parquets.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

from dialogue_benchmarks._paths import DATA_ROOT, PLUGINS_ROOT

RAW_DIR = DATA_ROOT / "raw"
DATASETS_PLUGIN_DIR = PLUGINS_ROOT / "datasets"
PARTIAL_THRESHOLD = 100 * 1024 * 1024  # 100MB
CHUNK = 8 * 1024 * 1024  # 8MB


def _hash_file(path: Path, size: int) -> str:
    if size <= PARTIAL_THRESHOLD:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
        return f"sha256:{h.hexdigest()}"

    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(CHUNK))
        if size > CHUNK:
            f.seek(max(size - CHUNK, 0))
            h.update(f.read(CHUNK))
    h.update(str(size).encode())
    return f"partial:{h.hexdigest()}"


def _dataset_report(d: Path) -> dict:
    # meta.json lives in the repo now (plugins/datasets/<name>/meta.json), not
    # alongside the raw data files this script hashes.
    meta_path = DATASETS_PLUGIN_DIR / d.name / "meta.json"
    has_meta = meta_path.exists()
    meta = json.loads(meta_path.read_text()) if has_meta else None

    files = []
    total_size = 0
    for path in sorted(d.rglob("*")):
        if path.is_dir():
            continue
        size = path.stat().st_size
        total_size += size
        files.append({"path": str(path.relative_to(d)), "size": size, "hash": _hash_file(path, size)})

    return {"has_meta": has_meta, "meta": meta, "n_files": len(files), "total_size": total_size, "files": files}


def main() -> None:
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "datasets": {}}
    for d in sorted(RAW_DIR.iterdir()):
        if not d.is_dir():
            continue
        print(f"hashing {d.name} ...")
        report["datasets"][d.name] = _dataset_report(d)

    out_path = RAW_DIR / "_verify_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
