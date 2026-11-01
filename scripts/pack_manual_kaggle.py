"""Package only manual Kaggle code; no checkpoints, results or nested archives."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output=None):
    output = Path(output or ROOT / "runs/manual_delivery/manual_kaggle_code.zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [
        "scripts/prepare_data.py",
        "scripts/manual_experiment.py",
        "scripts/failure_analysis_addendum.py",
        "src/__init__.py",
        "src/evaluation/__init__.py",
        "src/evaluation/metrics.py",
        "MANUAL_KAGGLE_GUIDE.md",
        "configs/manual/yolo26m_baseline.yaml",
        "configs/manual/yolo26x_baseline.yaml",
    ]
    names += [
        p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "notebooks/kaggle").glob("*.ipynb"))
    ]
    contents = {name: (ROOT / name).read_bytes() for name in names}
    # The original notebook is preserved on disk. Only outputs/metadata are omitted in the ZIP.
    nb = json.loads((ROOT / "notebooks/baseline.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    nb["metadata"] = {}
    contents["notebooks/baseline.ipynb"] = (json.dumps(nb, ensure_ascii=False) + "\n").encode()
    manifest = {
        "base_commit": "c39c9ac",
        "notebook_mode": "run_all_v1",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr("manual_code_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip():
            raise ValueError("Code ZIP failed CRC validation")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Code ZIP SHA256 mismatch: {name}")
    print(f"{output} ({output.stat().st_size:,} bytes)")
    return output


if __name__ == "__main__":
    build()
