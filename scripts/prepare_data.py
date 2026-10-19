"""Prepare VisDrone2019-DET without running the whole baseline notebook.

Executes only the data cells of ``notebooks/baseline.ipynb`` (setup, download,
YOLO conversion, ``visdrone.yaml``, dataset checks) — every code cell before the
"## 4. Initialize both models" section — so the data pipeline stays defined in
one place. Skips the vanilla scoring and the 5-epoch stage-1 fine-tune.

Usage (from anywhere)::

    python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the sample-image cell plots; no display on a server

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "baseline.ipynb"
STOP_HEADING = "## 4."


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


def main() -> None:
    os.chdir(ROOT)  # the notebook resolves data/ relative to the repo root
    namespace: dict[str, object] = {"__name__": "__prepare_data__"}
    for index, source in enumerate(data_cells(NOTEBOOK)):
        print(f"--- baseline.ipynb data cell {index} ---")
        exec(compile(source, f"baseline.ipynb[data:{index}]", "exec"), namespace)
    print(f"data ready under {ROOT / 'data'}")


if __name__ == "__main__":
    main()
