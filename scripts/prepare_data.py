"""Prepare VisDrone2019-DET without running the whole baseline notebook.

Executes only the data cells of ``notebooks/baseline.ipynb`` (setup, download,
YOLO conversion, ``visdrone.yaml``, dataset checks) — every code cell before the
"## 4. Initialize both models" section — so the data pipeline stays defined in
one place. Skips the vanilla scoring and the 5-epoch stage-1 fine-tune.

Two robustness additions around those cells (the cells themselves are untouched):

* if an already extracted ``data/raw/VisDrone2019-DET-*`` tree is attached as a
  Kaggle input (e.g. the output of the fine-tune notebook), it is linked into
  ``data/raw/`` so nothing is downloaded (``VISDRONE_RAW`` overrides the search);
* downloads are retried, and a partially written zip is removed between attempts.

Usage (from anywhere)::

    python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # the sample-image cell plots; no display on a server

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "baseline.ipynb"
STOP_HEADING = "## 4."
RAW_DIR = ROOT / "data" / "raw"
RAW_FOLDERS = ("VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev")
KAGGLE_INPUT = Path("/kaggle/input")
DOWNLOAD_ATTEMPTS = 5


def data_cells(notebook: Path) -> list[str]:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    sources: list[str] = []
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.lstrip().startswith(STOP_HEADING):
            break
        if cell["cell_type"] == "code":
            sources.append(source)
    return sources


def find_raw_source() -> Path | None:
    """An attached ``data/raw`` holding all three extracted VisDrone folders."""
    if os.environ.get("VISDRONE_RAW"):
        return Path(os.environ["VISDRONE_RAW"])
    if not KAGGLE_INPUT.is_dir():
        return None
    for depth in range(1, 5):  # /kaggle/input/<input>[/<sub>...]/data/raw
        for candidate in sorted(KAGGLE_INPUT.glob("/".join(["*"] * depth) + "/data/raw")):
            if all((candidate / folder / "images").is_dir() for folder in RAW_FOLDERS):
                return candidate
    return None


def link_raw(source: Path) -> None:
    """Symlink already extracted raw folders into data/raw (read-only is fine)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for folder in RAW_FOLDERS:
        target = RAW_DIR / folder
        if (target / "images").is_dir():
            continue
        target.symlink_to((source / folder).resolve(), target_is_directory=True)
        print(f"linked {target} -> {source / folder}")


def retrying_urlretrieve(url: str, filename: Any = None, *args: Any, **kwargs: Any) -> Any:
    """``urlretrieve`` that retries dropped connections and removes partial files."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return _urlretrieve(url, filename, *args, **kwargs)
        except OSError as exc:  # ConnectionResetError, URLError, timeouts
            if filename is not None:
                Path(filename).unlink(missing_ok=True)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            print(f"download failed ({exc}); retry {attempt}/{DOWNLOAD_ATTEMPTS - 1} in 15 s")
            time.sleep(15)
    raise AssertionError("unreachable")


_urlretrieve = urllib.request.urlretrieve


def main() -> None:
    os.chdir(ROOT)  # the notebook resolves data/ relative to the repo root
    raw_source = find_raw_source()
    if raw_source is not None:
        link_raw(raw_source)
    urllib.request.urlretrieve = retrying_urlretrieve  # used by download_visdrone()
    namespace: dict[str, object] = {"__name__": "__prepare_data__"}
    for index, source in enumerate(data_cells(NOTEBOOK)):
        print(f"--- baseline.ipynb data cell {index} ---")
        exec(compile(source, f"baseline.ipynb[data:{index}]", "exec"), namespace)
    print(f"data ready under {ROOT / 'data'}")


if __name__ == "__main__":
    main()
