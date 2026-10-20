"""Pack the final artifacts into one zip: report, all computed results, optionally weights.

Reads ``runs/`` (fine-tune + evaluation outputs), ``results/`` and ``docs/results.md``;
writes the layout described in ``results/ARCHIVE_README.md`` (copied into the archive as
``README.md``):

* default: ``results/small_object_detection_results.zip`` without checkpoints (committed);
* ``--with-weights``: ``results/small_object_detection_results_with_weights.zip`` including
  both ``best.pt`` files (~160 MB, kept out of git: GitHub rejects files over 100 MB).

Usage::

    python scripts/pack_results.py [--with-weights] [--stage finetune]
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("yolo26m", "yolo26x")


def build_manifest(stage: str, with_weights: bool) -> list[tuple[Path, str]]:
    """(source file, path inside the archive) for every packed file."""
    runs = ROOT / "runs"
    train = runs / stage
    entries: list[tuple[Path, str]] = [
        (ROOT / "results" / "ARCHIVE_README.md", "README.md"),
        (ROOT / "docs" / "results.md", "report/results.md"),
    ]
    entries += [(p, f"report/figures/{p.name}") for p in sorted((ROOT / "docs/figures").glob("*"))]

    for model in MODELS:
        run = train / f"{model}_s640"
        entries.append((run / "args.yaml", f"checkpoints/{model}_s640/args.yaml"))
        if with_weights:
            entries.append((run / "weights" / "best.pt", f"checkpoints/{model}_s640/best.pt"))
        entries += [
            (p, f"experiments/training/{model}_s640/{p.name}")
            for p in sorted(run.iterdir())
            if p.is_file()
        ]
        entries.append((train / f"{model}.log", f"experiments/training/{model}.log"))
        entries.append(
            (train / f"finetune_{model}.ipynb", f"experiments/notebooks/finetune_{model}.ipynb")
        )
    entries += [
        (ROOT / "results/training_summary.csv", "experiments/training/training_summary.csv"),
        (ROOT / "results/training_config.json", "experiments/training/training_config.json"),
        (runs / "pilot/pilot_result.json", "experiments/training/pilot_result.json"),
        (runs / "pilot/pilot.log", "experiments/training/pilot.log"),
        (train / "finetune_curves.png", "experiments/training/finetune_val_curves.png"),
    ]

    metrics = "experiments/metrics"
    entries += [
        (ROOT / "results/metrics_all.csv", f"{metrics}/metrics_all.csv"),
        (ROOT / "results/metrics_by_native_size.csv", f"{metrics}/metrics_by_native_size.csv"),
        (
            runs / f"highres/{stage}/appendix_by_native_size_delta_test.csv",
            f"{metrics}/metrics_by_native_size_delta.csv",
        ),
    ]
    tables = runs / "results" / stage
    entries += [(p, f"{metrics}/{p.name}") for p in sorted(tables.glob("*.csv"))]
    entries.append((tables / "results_tables.md", f"{metrics}/results_tables.md"))

    cond = "experiments/conditions"
    highres, tiling = runs / "highres" / stage, runs / "tiling" / stage
    entries += [
        (highres / "highres_scores_test_640.csv", f"{cond}/full_640.csv"),
        (highres / "highres_scores_test_1280.csv", f"{cond}/full_1280.csv"),
        (tiling / "tiling_scores_test_ov00.csv", f"{cond}/tiled_ov00.csv"),
        (tiling / "tiling_scores_test_ov20.csv", f"{cond}/tiled_ov20.csv"),
        (tiling / "tiling_scores_test_ov40.csv", f"{cond}/tiled_ov40.csv"),
        (
            tiling / "tiling_overlap_sweep_test_yolo26m.csv",
            f"{cond}/tiling_overlap_sweep_yolo26m.csv",
        ),
        (tiling / "tiling_vs_baseline_test.csv", f"{cond}/tiling_vs_baseline.csv"),
    ]

    figs = "experiments/figures"
    entries += [(p, f"{figs}/{p.name}") for p in sorted(tables.glob("*.png"))]
    entries += [(p, f"{figs}/{p.name}") for p in sorted(tiling.glob("*.png"))]
    entries += [
        (ROOT / "docs/figures" / name, f"{figs}/{name}")
        for name in (
            "failure_crowding_tiny.png",
            "failure_large_object.png",
            "failure_tile_boundary.png",
        )
    ]
    entries += [
        (p, f"experiments/notebooks/{p.name}")
        for p in sorted((runs / "notebooks").glob(f"*_{stage}.ipynb"))
    ]
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="finetune")
    parser.add_argument("--with-weights", action="store_true", help="include best.pt files")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    suffix = "_with_weights" if args.with_weights else ""
    default_out = ROOT / "results" / f"small_object_detection_results{suffix}.zip"
    entries = build_manifest(args.stage, args.with_weights)
    missing = [str(src) for src, _ in entries if not src.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs:\n  " + "\n  ".join(missing))

    out = Path(args.out) if args.out else default_out
    with zipfile.ZipFile(out, "w") as archive:
        for src, arcname in entries:
            # Checkpoints/PNGs are already compressed: store them, deflate the rest.
            stored = src.suffix in {".pt", ".png", ".jpg"}
            method = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
            archive.write(src, arcname, compress_type=method)
    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB, {len(entries)} files)")


if __name__ == "__main__":
    main()
