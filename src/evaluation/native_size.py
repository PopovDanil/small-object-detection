"""Appendix analysis: every inference condition scored per native image size.

VisDrone test-dev mixes images of different native resolutions (960x540,
1360x765, 1920x1080), and the inference conditions treat them differently:
``imgsz=1280`` upscales any image whose long side is below 1280, and tiling
needs 2 crops for a 960x540 image but 8 for a 1920x1080 one. This module splits
a split into native-size groups and scores each group separately with the
shared :func:`~src.evaluation.metrics.score_predictions`, so the per-group
numbers use exactly the same metrics as the main results.

It is a supplementary table only: the main results come from
:func:`~src.evaluation.metrics.evaluate_split` on the whole split. Groups differ
in camera, altitude and scene, so compare conditions *within* a group (see
:func:`delta_vs_reference`), not groups against each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from ultralytics import YOLO

from src.evaluation.metrics import (
    CONF_EVAL,
    IGNORE_DIR,
    TILE_OVERLAP,
    TILE_SIZE,
    benchmark_runtime,
    benchmark_runtime_tiled,
    predict_images,
    predict_images_tiled,
    score_predictions,
)

# Condition name -> evaluate_split-style parameters. ``inference`` is "full"
# (letterbox the whole image to ``imgsz``) or "tiled" (native-resolution crops).
DEFAULT_CONDITIONS: dict[str, dict[str, Any]] = {
    "full-640": {"inference": "full", "imgsz": 640},
    "full-1280": {"inference": "full", "imgsz": 1280},
    "tiled-ov20": {"inference": "tiled", "tile": TILE_SIZE, "overlap": TILE_OVERLAP},
}

RUNTIME_PER_GROUP = 60  # timed images per native-size group (first 10 are warm-up)

SIZE_METRICS = (
    "map50_95",
    "map50",
    "recall50_25",
    "ap_tiny",
    "recall_tiny",
    "ap_small",
    "recall_small",
    "ap_medium",
    "recall_medium",
    "ap_large",
    "recall_large",
    "runtime_mean_ms",
    "runtime_p95_ms",
)


def native_size_groups(images: list[Path]) -> dict[str, list[int]]:
    """Indices of ``images`` grouped by native size ``"WxH"`` (header read only)."""
    groups: dict[str, list[int]] = {}
    for index, image_path in enumerate(images):
        with Image.open(image_path) as img:
            width, height = img.size
        groups.setdefault(f"{width}x{height}", []).append(index)
    return groups


def _predict(
    model: YOLO, images: list[Path], params: dict[str, Any], batch: int
) -> dict[str, Any]:
    if params["inference"] == "tiled":
        return predict_images_tiled(
            model,
            images,
            tile=params.get("tile", TILE_SIZE),
            overlap=params.get("overlap", TILE_OVERLAP),
            conf=CONF_EVAL,
        )
    return predict_images(model, images, imgsz=params["imgsz"], conf=CONF_EVAL, batch=batch)


def _time(model: YOLO, images: list[Path], params: dict[str, Any]) -> dict[str, float]:
    if params["inference"] == "tiled":
        return benchmark_runtime_tiled(
            model,
            images,
            tile=params.get("tile", TILE_SIZE),
            overlap=params.get("overlap", TILE_OVERLAP),
        )
    return benchmark_runtime(model, images, imgsz=params["imgsz"])


def evaluate_by_native_size(
    models: dict[str, YOLO],
    images: list[Path],
    labels: list[Path],
    conditions: dict[str, dict[str, Any]] | None = None,
    runtime_per_group: int = RUNTIME_PER_GROUP,
    batch: int = 4,
    ignore_dir: Path = IGNORE_DIR,
) -> pd.DataFrame:
    """All shared metrics + runtime per (model, condition, native image size).

    One prediction pass per model and condition over the whole split; the
    predictions are then scored group by group with ``score_predictions``.
    Runtime uses the first ``runtime_per_group`` images of each group with the
    shared benchmark protocol (batch 1, warm-up, CUDA sync).
    """
    conditions = DEFAULT_CONDITIONS if conditions is None else conditions
    groups = native_size_groups(images)
    rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        for condition, params in conditions.items():
            print(f"=== {model_name} {condition}: per-native-size pass ({len(images)} images) ===")
            preds = _predict(model, images, params, batch)
            for size_name, indices in groups.items():
                group_images = [images[i] for i in indices]
                metrics = score_predictions(
                    [labels[i] for i in indices], group_images, preds, ignore_dir
                )
                metrics.update(_time(model, group_images[:runtime_per_group], params))
                rows.append(
                    {
                        "model": model_name,
                        "native": size_name,
                        "images": len(indices),
                        "inference": condition,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def delta_vs_reference(
    table: pd.DataFrame,
    reference: str = "full-640",
    metrics: tuple[str, ...] = SIZE_METRICS,
) -> pd.DataFrame:
    """``condition - reference`` inside each (model, native size) group.

    Positive is better for accuracy metrics and worse for runtime.
    """
    base = table[table["inference"] == reference].set_index(["model", "native"])
    rows: list[dict[str, Any]] = []
    for _, row in table[table["inference"] != reference].iterrows():
        ref = base.loc[(row["model"], row["native"])]
        rows.append(
            {
                "model": row["model"],
                "native": row["native"],
                "condition": row["inference"],
                **{metric: row[metric] - ref[metric] for metric in metrics},
            }
        )
    return pd.DataFrame(rows).set_index(["model", "native", "condition"]).sort_index()
