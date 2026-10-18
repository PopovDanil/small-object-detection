"""Shared evaluation for every small-object-detection experiment.

The project measures inference-time resolution (640 full-image, 1280
full-image, native-resolution tiling) on VisDrone2019-DET, so all conditions
must be scored by exactly one evaluator. The entry point is
:func:`evaluate_split`, which accepts the ground-truth labels, the
original-resolution images and a YOLO model and returns every metric required
by the proposal:

* ``map50_95`` / ``map50`` — COCO mAP at IoU 0.50:0.95 / 0.50, low conf;
* ``recall50_25`` — recall at IoU 0.5, confidence 0.25, fixed everywhere;
* ``ap_tiny`` / ``recall_tiny`` — tiny objects (original area < 16^2 px);
* ``ap_small|medium|large`` / ``recall_small|medium|large`` / ``n_*`` — COCO
  size groups (small < 32^2, 32^2 <= medium < 96^2, large >= 96^2) with object
  counts;
* ``runtime_mean_ms`` / ``runtime_p95_ms`` — mean and p95 ms per original
  image, batch size 1, with warm-up and CUDA synchronization.

All metrics are computed in original-image coordinates and ignored/don't-care
regions are respected. Results apply to this protocol and are not official
leaderboard scores.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import batched_nms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
IGNORE_DIR = DATA_DIR / "ignore"
MANIFESTS_DIR = DATA_DIR / "manifests"
RUNS_DIR = ROOT / "runs"

# Fixed protocol constants: decided before any result is examined.
SEED = 42
IMG_SIZE = 640
CONF_EVAL = 0.001  # low threshold for AP (proposal)
CONF_RECALL = 0.25  # fixed recall confidence (proposal)
IOU_RECALL = 0.5  # fixed recall IoU (proposal)
IOU_NMS = 0.7  # fixed NMS IoU (proposal)
TINY_AREA = 16**2  # tiny objects: original-image area < 16^2 px
SMALL_AREA = 32**2  # COCO small upper bound
MEDIUM_AREA = 96**2  # COCO medium upper bound
IGNORE_IOU_THR = 0.5  # drop predictions overlapping ignored regions at this IoU
TILE_SIZE = 640  # tiling: native-resolution crops of 640x640
TILE_OVERLAP = 0.20  # tiling: default 20% overlap between neighbouring crops

CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)

# COCO-pretrained (vanilla) models predict 80 COCO classes. For the zero-shot
# baseline we keep only classes with an unambiguous VisDrone counterpart and
# drop the rest: COCO has no van/tricycle/awning-tricycle analogue, and the
# single COCO ``person`` class cannot be split into VisDrone pedestrian/people.
COCO_TO_VISDRONE_NAMES = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motor",
    "bus": "bus",
    "truck": "truck",
}

AREA_RANGES = [
    ("all", 0.0, 1e10),
    ("tiny", 0.0, float(TINY_AREA)),
    ("small", 0.0, float(SMALL_AREA)),
    ("medium", float(SMALL_AREA), float(MEDIUM_AREA)),
    ("large", float(MEDIUM_AREA), 1e10),
]
SIZE_GROUPS = ("all", "tiny", "small", "medium", "large")


# --------------------------------------------------------------------------- #
# Paths, labels and ignored regions
# --------------------------------------------------------------------------- #
def load_split_paths(
    split: str,
    manifests_dir: Path = MANIFESTS_DIR,
    images_dir: Path = IMAGES_DIR,
    labels_dir: Path = LABELS_DIR,
) -> tuple[list[Path], list[Path]]:
    """Aligned (images, labels) lists for one split, from its seed-42 manifest."""
    with (manifests_dir / f"{split}.csv").open(newline="") as handle:
        names = [row["image"] for row in csv.DictReader(handle)]
    images = [images_dir / split / name for name in names]
    labels = [labels_dir / split / f"{Path(name).stem}.txt" for name in names]
    return images, labels


def read_yolo_labels(label_path: Path) -> np.ndarray:
    """Return (N, 5) [class, xc, yc, w, h] normalized; shape (0, 5) when empty."""
    rows = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                rows.append([float(v) for v in parts])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def yolo_to_xyxy(labels: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalized YOLO boxes -> (boxes_xyxy, classes) in original-image pixels."""
    if len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    cls = labels[:, 0].astype(np.int64)
    xc, yc, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
    boxes = np.stack(
        [(xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height],
        axis=1,
    ).astype(np.float64)
    return boxes, cls


def read_ignore_boxes(image_path: Path, ignore_dir: Path = IGNORE_DIR) -> np.ndarray:
    """(M, 4) absolute xyxy ignored regions for one image (empty when none)."""
    path = ignore_dir / image_path.parent.name / f"{image_path.stem}.txt"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                rows.append([float(v) for v in parts[:4]])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


# --------------------------------------------------------------------------- #
# IoU and ignored-region handling
# --------------------------------------------------------------------------- #
def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two xyxy box sets -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def filter_ignored(
    predictions: np.ndarray,
    ignore_boxes: np.ndarray,
    iou_thr: float = IGNORE_IOU_THR,
) -> np.ndarray:
    """Respect ignored regions: drop predictions overlapping them at IoU >= thr."""
    if len(predictions) == 0 or len(ignore_boxes) == 0:
        return predictions
    ious = box_iou_matrix(predictions[:, :4], ignore_boxes)
    return predictions[ious.max(axis=1) < iou_thr]


# --------------------------------------------------------------------------- #
# Inference + timing
# --------------------------------------------------------------------------- #
def _boxes_to_numpy(result: Any) -> np.ndarray:
    """Ultralytics result -> (N, 6) [x1, y1, x2, y2, conf, cls] in original coords."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float64)
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
    confs = boxes.conf.cpu().numpy().astype(np.float64)
    cls = boxes.cls.cpu().numpy().astype(np.float64)
    return np.concatenate([xyxy, confs[:, None], cls[:, None]], axis=1)


def remap_classes(predictions: np.ndarray, class_map: dict[int, int] | None) -> np.ndarray:
    """Keep predictions whose raw class id is in ``class_map`` and renumber them.

    ``class_map=None`` is the identity and is used for the fine-tuned 10-class
    models. A map is used for zero-shot COCO models and drops unmapped classes.
    """
    if class_map is None or len(predictions) == 0:
        return predictions
    raw = predictions[:, 5].astype(np.int64)
    mapped = np.array([class_map.get(int(c), -1) for c in raw], dtype=np.float64)
    keep = mapped >= 0
    out = predictions[keep].copy()
    out[:, 5] = mapped[keep]
    return out


def coco_class_map(model: YOLO) -> dict[int, int]:
    """Map a COCO-pretrained model's 80 class ids onto the 10 VisDrone ids."""
    visdrone_id = {name: index for index, name in enumerate(CLASSES)}
    mapping: dict[int, int] = {}
    for raw_id, name in model.names.items():
        target = COCO_TO_VISDRONE_NAMES.get(str(name))
        if target is not None:
            mapping[int(raw_id)] = visdrone_id[target]
    return mapping


def predict_images(
    model: YOLO,
    image_paths: list[Path],
    imgsz: int = IMG_SIZE,
    conf: float = CONF_EVAL,
    batch: int = 16,
    iou: float = IOU_NMS,
    device: str | None = None,
    class_map: dict[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Prediction pass for many images (same math as batch=1, faster)."""
    preds: dict[str, np.ndarray] = {}
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start : start + batch]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        for path, result in zip(chunk, results):
            preds[path.name] = remap_classes(_boxes_to_numpy(result), class_map)
    return preds


def benchmark_runtime(
    model: YOLO,
    image_paths: list[Path],
    imgsz: int = IMG_SIZE,
    warmup: int = 10,
    conf: float = CONF_EVAL,
    iou: float = IOU_NMS,
    device: str | None = None,
) -> dict[str, float]:
    """Mean/p95 ms per original image: batch 1, warm-up, CUDA synchronization."""
    timings: list[float] = []
    use_cuda = torch.cuda.is_available()
    for image_path in image_paths:
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        if use_cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    measured = timings[warmup:] or timings  # tiny splits: keep every timing
    array = np.asarray(measured, dtype=np.float64)
    return {
        "runtime_mean_ms": float(array.mean()),
        "runtime_p95_ms": float(np.percentile(array, 95)),
    }


# --------------------------------------------------------------------------- #
# Tiled inference at native resolution
# --------------------------------------------------------------------------- #
def tile_starts(length: int, tile: int = TILE_SIZE, overlap: float = TILE_OVERLAP) -> list[int]:
    """Top-left coordinates covering ``[0, length)`` with ``tile`` windows.

    The stride is ``tile * (1 - overlap)``. The last window is flush with the
    far edge so the whole image is covered even when ``length`` is not an exact
    multiple of the stride (this can add a little extra overlap at the edge).
    """
    if length <= tile:
        return [0]
    stride = max(1, round(tile * (1.0 - overlap)))
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def merge_tile_detections(detections: np.ndarray, iou_thr: float = IOU_NMS) -> np.ndarray:
    """Merge per-tile detections in original coordinates with class-aware NMS.

    Overlapping tiles produce duplicate boxes for the same object; ``batched_nms``
    keeps the highest-confidence box per class and suppresses the rest at
    ``iou_thr`` (the same NMS IoU fixed for every condition).
    """
    if len(detections) == 0:
        return detections
    boxes = torch.as_tensor(detections[:, :4], dtype=torch.float32)
    scores = torch.as_tensor(detections[:, 4], dtype=torch.float32)
    classes = torch.as_tensor(detections[:, 5].astype(np.int64))
    keep = batched_nms(boxes, scores, classes, iou_thr)
    return detections[keep.numpy()]


def predict_image_tiled(
    model: YOLO,
    image_path: Path,
    tile: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
    conf: float = CONF_EVAL,
    iou: float = IOU_NMS,
    tile_batch: int = 1,
    device: str | None = None,
    nms_iou: float = IOU_NMS,
) -> np.ndarray:
    """Run one native-resolution image tile-by-tile and merge to original coords.

    Crops are ``tile`` x ``tile`` with the requested overlap; every tile is
    resized by Ultralytics to ``tile`` (already the native tile size). Tile
    predictions are offset back to original-image coordinates, clipped to the
    image bounds, and merged with class-aware NMS.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return np.zeros((0, 6), dtype=np.float64)
    height, width = image.shape[:2]
    crops: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for y0 in tile_starts(height, tile, overlap):
        for x0 in tile_starts(width, tile, overlap):
            crops.append(image[y0 : y0 + tile, x0 : x0 + tile])
            origins.append((x0, y0))

    tile_preds: list[np.ndarray] = []
    for start in range(0, len(crops), tile_batch):
        chunk = crops[start : start + tile_batch]
        results = model.predict(
            source=chunk,
            imgsz=tile,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        for result, (x0, y0) in zip(results, origins[start : start + tile_batch]):
            dets = _boxes_to_numpy(result)
            if len(dets):
                dets[:, [0, 2]] += x0
                dets[:, [1, 3]] += y0
                tile_preds.append(dets)

    if not tile_preds:
        return np.zeros((0, 6), dtype=np.float64)
    merged = np.concatenate(tile_preds, axis=0)
    merged[:, [0, 2]] = np.clip(merged[:, [0, 2]], 0, width)
    merged[:, [1, 3]] = np.clip(merged[:, [1, 3]], 0, height)
    return merge_tile_detections(merged, nms_iou)


def predict_images_tiled(
    model: YOLO,
    image_paths: list[Path],
    tile: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
    conf: float = CONF_EVAL,
    iou: float = IOU_NMS,
    tile_batch: int = 1,
    device: str | None = None,
    class_map: dict[int, int] | None = None,
    nms_iou: float = IOU_NMS,
) -> dict[str, np.ndarray]:
    """Tiled prediction pass for many images; output is in original coordinates."""
    preds: dict[str, np.ndarray] = {}
    for image_path in image_paths:
        dets = predict_image_tiled(
            model,
            image_path,
            tile=tile,
            overlap=overlap,
            conf=conf,
            iou=iou,
            tile_batch=tile_batch,
            device=device,
            nms_iou=nms_iou,
        )
        preds[image_path.name] = remap_classes(dets, class_map)
    return preds


def benchmark_runtime_tiled(
    model: YOLO,
    image_paths: list[Path],
    tile: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
    warmup: int = 10,
    conf: float = CONF_EVAL,
    iou: float = IOU_NMS,
    tile_batch: int = 1,
    device: str | None = None,
) -> dict[str, float]:
    """Mean/p95 ms per original image for the tiled path (tiles + merge).

    Same protocol as :func:`benchmark_runtime`: batch size 1 (one original image
    per timed call), warm-up, and explicit CUDA synchronization. Everything is
    counted: image decode, all tile crops, tile inference and NMS merging.
    """
    timings: list[float] = []
    use_cuda = torch.cuda.is_available()
    for image_path in image_paths:
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        predict_image_tiled(
            model,
            image_path,
            tile=tile,
            overlap=overlap,
            conf=conf,
            iou=iou,
            tile_batch=tile_batch,
            device=device,
        )
        if use_cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    measured = timings[warmup:] or timings
    array = np.asarray(measured, dtype=np.float64)
    return {
        "runtime_mean_ms": float(array.mean()),
        "runtime_p95_ms": float(np.percentile(array, 95)),
    }


# --------------------------------------------------------------------------- #
# AP (pycocotools) overall and per size group
# --------------------------------------------------------------------------- #
def build_coco_gt(image_paths: list[Path], label_paths: list[Path]) -> dict[str, Any]:
    """COCO gt dict from YOLO labels, in original-image coordinates."""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id, (image_path, label_path) in enumerate(zip(image_paths, label_paths), start=1):
        with Image.open(image_path) as img:
            width, height = img.size
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        boxes, classes = yolo_to_xyxy(read_yolo_labels(label_path), width, height)
        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = box
            bw, bh = float(x2 - x1), float(y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)],
    }


def _load_results(coco_gt: COCO, detections: list[dict[str, Any]]) -> COCO:
    """Load detections into a COCO results object (empty-list safe)."""
    if detections:
        return coco_gt.loadRes(detections)
    coco_dt = COCO()
    coco_dt.dataset = {
        "images": list(coco_gt.dataset["images"]),
        "categories": list(coco_gt.dataset["categories"]),
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


def compute_map50_95(
    gt: dict[str, Any],
    preds_by_image: dict[str, np.ndarray],
) -> dict[str, float]:
    """mAP50-95, mAP50 and AP50-95 per COCO/tiny size group via pycocotools."""
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    detections: list[dict[str, Any]] = []
    for image in gt["images"]:
        preds = preds_by_image.get(image["file_name"], np.zeros((0, 6)))
        for x1, y1, x2, y2, score, cls_id in preds:
            detections.append(
                {
                    "image_id": image["id"],
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    coco_dt = _load_results(coco_gt, detections)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.areaRng = [[lo, hi] for _, lo, hi in AREA_RANGES]
    evaluator.params.areaRngLbl = [label for label, _, _ in AREA_RANGES]
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]  # (T, R, K, A, M)
    metrics: dict[str, float] = {}
    for area_index, (label, _, _) in enumerate(AREA_RANGES):
        values = precision[:, :, :, area_index, -1]
        values = values[values > -1]
        key = "map50_95" if label == "all" else f"ap_{label}"
        metrics[key] = float(values.mean()) if values.size else 0.0

    # mAP50 (IoU=0.50), from the same precision array, IoU thresholds are
    # ordered 0.50:0.05:0.95 (the summarize() path is capped at maxDets=20).
    values = precision[0, :, :, 0, -1]
    values = values[values > -1]
    metrics["map50"] = float(values.mean()) if values.size else 0.0
    return metrics


# --------------------------------------------------------------------------- #
# Recall @ IoU=0.5, conf=0.25 per size group (with object counts)
# --------------------------------------------------------------------------- #
def compute_recall(
    image_paths: list[Path],
    label_paths: list[Path],
    preds_by_image: dict[str, np.ndarray],
    conf: float = CONF_RECALL,
    iou_thr: float = IOU_RECALL,
) -> dict[str, float]:
    """Recall at fixed IoU/conf, overall and per size group, with counts."""
    total: Counter[str] = Counter()
    matched_total: Counter[str] = Counter()
    for image_path, label_path in zip(image_paths, label_paths):
        with Image.open(image_path) as img:
            width, height = img.size
        gt_boxes, _gt_classes = yolo_to_xyxy(read_yolo_labels(label_path), width, height)
        if len(gt_boxes):
            areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        else:
            areas = np.zeros(0, dtype=np.float64)
        groups = {
            "all": np.ones(len(areas), dtype=bool),
            "tiny": areas < TINY_AREA,
            "small": areas < SMALL_AREA,
            "medium": (areas >= SMALL_AREA) & (areas < MEDIUM_AREA),
            "large": areas >= MEDIUM_AREA,
        }

        preds = preds_by_image.get(image_path.name, np.zeros((0, 6)))
        preds = preds[preds[:, 4] >= conf]
        preds = preds[np.argsort(-preds[:, 4])] if len(preds) else preds

        matched = np.zeros(len(gt_boxes), dtype=bool)
        for pred in preds:
            if len(gt_boxes) == 0:
                break
            ious = box_iou_matrix(pred[None, :4], gt_boxes)[0]
            ious[matched] = -1.0
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr:
                matched[best] = True

        for group, mask in groups.items():
            total[group] += int(mask.sum())
            matched_total[group] += int(matched[mask].sum())

    metrics: dict[str, float] = {}
    for group in SIZE_GROUPS:
        metrics[f"recall_{group}" if group != "all" else "recall50_25"] = (
            matched_total[group] / total[group] if total[group] else 0.0
        )
        metrics[f"n_{group}"] = float(total[group])
    return metrics


# --------------------------------------------------------------------------- #
# Single shared entry point
# --------------------------------------------------------------------------- #
def score_predictions(
    labels: list[Path],
    images: list[Path],
    preds_by_image: dict[str, np.ndarray],
    ignore_dir: Path = IGNORE_DIR,
) -> dict[str, float]:
    """Score precomputed original-coordinate predictions with the shared metrics.

    Used by :func:`evaluate_split` for every inference condition; keeping the
    AP/recall computation here means the 640, 1280 and tiled conditions are all
    scored by exactly the same code.
    """
    filtered = {
        image_path.name: filter_ignored(
            preds_by_image[image_path.name], read_ignore_boxes(image_path, ignore_dir)
        )
        for image_path in images
    }
    metrics = compute_map50_95(build_coco_gt(images, labels), filtered)
    metrics.update(compute_recall(images, labels, filtered))
    return metrics


def evaluate_split(
    labels: list[Path],
    images: list[Path],
    model: YOLO,
    imgsz: int = IMG_SIZE,
    conf: float = CONF_EVAL,
    runtime_images: int | None = None,
    batch: int = 16,
    class_map: dict[int, int] | None = None,
    device: str | None = None,
    ignore_dir: Path = IGNORE_DIR,
    warmup: int = 10,
    inference: str = "full",
    tile: int = TILE_SIZE,
    overlap: float = TILE_OVERLAP,
    tile_batch: int = 1,
) -> dict[str, float]:
    """Score one model on one split with the shared protocol.

    Parameters
    ----------
    labels:
        ground-truth YOLO label files (one per image, same order as ``images``).
    images:
        original-resolution image paths.
    model:
        YOLO detector; fine-tuned 10-class checkpoints or zero-shot COCO models.
    class_map:
        optional raw-class -> VisDrone-id map. ``None`` keeps the model's own
        class ids (fine-tuned models); :func:`coco_class_map` builds the
        zero-shot mapping and drops unmapped COCO classes.
    runtime_images:
        number of images used for the timed pass (all when ``None``).
    inference:
        ``"full"`` for full-image resizing to ``imgsz`` (640/1280) or
        ``"tiled"`` for native-resolution ``tile`` x ``tile`` crops with
        ``overlap`` and NMS merging.
    tile, overlap, tile_batch:
        tiling parameters, used only when ``inference == "tiled"``.

    Returns
    -------
    Flat dict with mAP50-95, mAP50, recall@IoU0.5/conf0.25, tiny (<16^2 px) AP
    and recall, COCO small/medium/large AP and recall with per-group object
    counts, and mean/p95 runtime per original image.
    """
    if inference == "tiled":
        preds_by_image = predict_images_tiled(
            model,
            images,
            tile=tile,
            overlap=overlap,
            conf=conf,
            tile_batch=tile_batch,
            device=device,
            class_map=class_map,
        )
    else:
        preds_by_image = predict_images(
            model, images, imgsz=imgsz, conf=conf, batch=batch, device=device, class_map=class_map
        )
    metrics = score_predictions(labels, images, preds_by_image, ignore_dir)
    n_runtime = len(images) if runtime_images is None else min(runtime_images, len(images))
    if inference == "tiled":
        metrics.update(
            benchmark_runtime_tiled(
                model,
                images[:n_runtime],
                tile=tile,
                overlap=overlap,
                warmup=warmup,
                tile_batch=tile_batch,
                device=device,
            )
        )
    else:
        metrics.update(
            benchmark_runtime(model, images[:n_runtime], imgsz=imgsz, warmup=warmup, device=device)
        )
    return metrics
