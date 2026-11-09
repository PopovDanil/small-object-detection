"""Reproduce the three fixed failure cases without changing the shared evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = (
    "9999938_00000_d_0000207.jpg",
    "9999986_00000_d_0000043.jpg",
    "0000006_00611_d_0000002.jpg",
)


def match_details(boxes, classes, predictions, metrics):
    """Diagnostic trace of current recall matching (class-agnostic, greedy).

    Not a second evaluator: score_predictions remains the metric authority.
    Records class correctness separately so localization TP is not a class TP.
    """
    import numpy as np

    predictions = predictions[predictions[:, 4] >= metrics.CONF_RECALL]
    predictions = predictions[np.argsort(-predictions[:, 4])]
    matched = np.zeros(len(boxes), dtype=bool)
    rows = []
    for index, pred in enumerate(predictions):
        ious = metrics.box_iou_matrix(pred[None, :4], boxes)[0]
        ious[matched] = -1
        best = int(ious.argmax()) if len(ious) else -1
        tp = best >= 0 and ious[best] >= metrics.IOU_RECALL
        if tp:
            matched[best] = True
        rows.append(
            {
                "prediction": index,
                "gt_index": best if tp else None,
                "tp_localization": bool(tp),
                "class_correct": bool(tp and int(pred[5]) == classes[best]),
            }
        )
    return predictions, matched, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/failure_addendum")
    parser.add_argument("--device", default="0")
    parser.add_argument("--pedestrian-roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()
    # Validate inputs before importing the GPU runtime or allowing auto-downloads.
    for path in [
        args.weights,
        *(args.data / "images/test" / name for name in CASES),
        *(args.data / "labels/test" / f"{Path(name).stem}.txt" for name in CASES),
        *(args.data / "ignore/test" / f"{Path(name).stem}.txt" for name in CASES),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    from PIL import Image
    from ultralytics import YOLO

    from scripts.manual_experiment import environment, sha256, write_json
    from src.evaluation import metrics as m

    args.output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(args.weights.resolve()))
    if tuple(model.names.values()) != m.CLASSES:
        raise ValueError(
            "Expected original YOLO26m VisDrone checkpoint with the ten shared classes"
        )
    trained_from = model.ckpt.get("train_args", {}).get("model", "")
    if Path(trained_from).stem != "yolo26m" or model.ckpt.get("train_args", {}).get("imgsz") != 640:
        raise ValueError("This comparison requires YOLO26m; verify train_args.model")
    write_json(
        args.output / "provenance.json",
        {
            "weights_sha256": sha256(args.weights),
            "evaluator_sha256": sha256(Path(m.__file__)),
            "environment": environment(),
            "conf_ap": m.CONF_EVAL,
            "conf_display": m.CONF_RECALL,
            "recall_matching": "class-agnostic (existing evaluator)",
            "overlap": 0.20,
        },
    )

    def draw(ax, image, boxes, predictions, grid=False):
        ax.imshow(image)
        for color, array in (("lime", boxes), ("red", predictions[:, :4])):
            for x1, y1, x2, y2 in array:
                ax.add_patch(
                    Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=0.6
                    )
                )
        if grid:
            w, h = image.size
            # Both near and far tile edges; the old notebook drew starts only.
            for x in sorted(
                {edge for start in m.tile_starts(w) for edge in (start, min(start + 640, w))}
            ):
                ax.axvline(x, color="deepskyblue", linewidth=0.6)
            for y in sorted(
                {edge for start in m.tile_starts(h) for edge in (start, min(start + 640, h))}
            ):
                ax.axhline(y, color="deepskyblue", linewidth=0.6)
        ax.set_axis_off()

    for name in CASES:
        path = args.data / "images/test" / name
        label = args.data / "labels/test" / f"{path.stem}.txt"
        image = Image.open(path).convert("RGB")
        w, h = image.size
        boxes, classes = m.yolo_to_xyxy(m.read_yolo_labels(label), w, h)
        ignored = m.read_ignore_boxes(path, args.data / "ignore")
        predictions = {}
        for size in (640, 1280):
            predictions[f"full-{size}"] = m.predict_images(
                model, [path], imgsz=size, batch=1, device=args.device
            )[name]

        captured = {}
        tile_rows = []
        original_merge = m.merge_tile_detections
        original_predict = model.predict
        origins = [(x, y) for y in m.tile_starts(h) for x in m.tile_starts(w)]

        def trace_predict(
            *positional,
            original_predict=original_predict,
            tile_rows=tile_rows,
            origins=origins,
            **keywords,
        ):
            results = original_predict(*positional, **keywords)
            for result in results:
                tile_rows.append(
                    {
                        "origin": origins[len(tile_rows)],
                        "detections": len(result.boxes) if result.boxes is not None else 0,
                    }
                )
            return results

        def trace(detections, iou_thr, captured=captured, original_merge=original_merge):
            captured["before_nms"] = detections.copy()
            merged = original_merge(detections, iou_thr)
            captured["after_nms_before_cap"] = merged.copy()
            return merged

        with (
            patch.object(m, "merge_tile_detections", side_effect=trace),
            patch.object(model, "predict", side_effect=trace_predict),
        ):
            predictions["tiled-ov20"] = m.predict_image_tiled(model, path, device=args.device)
        for key in ("before_nms", "after_nms_before_cap"):
            captured.setdefault(key, np.zeros((0, 6)))
        np.savez_compressed(args.output / f"{path.stem}_predictions.npz", **predictions, **captured)
        summary = {
            "image": name,
            "size": [w, h],
            "gt_boxes": boxes.tolist(),
            "gt_classes": classes.tolist(),
            "ignore_boxes": ignored.tolist(),
            "tiles_in_raw_array_order": tile_rows,
            "modes": {},
        }
        visible = {}
        for mode, raw in predictions.items():
            filtered = m.filter_ignored(raw, ignored)
            scores = m.score_predictions([label], [path], {name: raw}, args.data / "ignore")
            shown, matched, trace_rows = match_details(boxes, classes, filtered, m)
            visible[mode] = shown
            tp = int(matched.sum())
            if len(boxes) and not np.isclose(tp / len(boxes), scores["recall50_25"]):
                raise AssertionError(
                    "Diagnostic matching differs from shared recall; review evaluator changes"
                )
            summary["modes"][mode] = {
                "shared_metrics": scores,
                "tp_localization": tp,
                "fp_localization": len(shown) - tp,
                "fn_localization": len(boxes) - tp,
                "matches": trace_rows,
                "wrong_class_localization_matches": sum(
                    r["tp_localization"] and not r["class_correct"] for r in trace_rows
                ),
            }
            # Evidence for misses; a low score/IoU is observed, its physical cause is a hypothesis.
            best_candidates = []
            for index, box in enumerate(boxes):
                same = filtered[filtered[:, 5] == classes[index]]
                ious = m.box_iou_matrix(box[None], same[:, :4])[0]
                best = int(ious.argmax()) if len(ious) else None
                best_candidates.append(
                    None
                    if best is None
                    else {
                        "prediction": same[best].tolist(),
                        "iou": float(ious[best]),
                        "passes_conf025": bool(same[best, 4] >= 0.25),
                    }
                )
            summary["modes"][mode]["best_same_class_candidate_per_gt_at_conf0001"] = best_candidates
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        for ax, (mode, shown) in zip(axes, visible.items()):
            draw(ax, image, boxes, shown, grid=mode == "tiled-ov20")
            r = summary["modes"][mode]
            ax.set_title(
                f"{mode}: localization TP/FP/FN {r['tp_localization']}/{r['fp_localization']}/{r['fn_localization']}"
            )
        fig.suptitle(f"{name}: green GT, red predictions >=0.25; class-agnostic matching")
        fig.tight_layout()
        fig.savefig(args.output / f"{path.stem}_comparison.png", dpi=180)
        plt.close(fig)

        if name == CASES[0]:
            tiny_pedestrians = (classes == 0) & (np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) < 256)
            candidates = np.flatnonzero(tiny_pedestrians)
            if not len(candidates):
                raise ValueError("No tiny pedestrian GT; verify the fixed case and labels")
            if args.pedestrian_roi:
                roi = np.array(args.pedestrian_roi)
            else:
                centres = (boxes[:, :2] + boxes[:, 2:]) / 2
                # Fixed GT-only rule: densest 240x240 window centred on a tiny pedestrian.
                counts = [
                    np.all(abs(centres[candidates] - centres[i]) <= 120, axis=1).sum()
                    for i in candidates
                ]
                centre = centres[candidates[int(np.argmax(counts))]]
                roi = np.array([*(centre - 120), *(centre + 120)])
            roi[[0, 2]] = roi[[0, 2]].clip(0, w)
            roi[[1, 3]] = roi[[1, 3]].clip(0, h)
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                raise ValueError("ROI must have positive size")
            centres = (boxes[:, :2] + boxes[:, 2:]) / 2
            selected = (
                tiny_pedestrians
                & np.all(centres >= roi[:2], axis=1)
                & np.all(centres <= roi[2:], axis=1)
            )
            summary["tiny_pedestrian_roi"] = {
                "xyxy": roi.tolist(),
                "gt_indices": np.flatnonzero(selected).tolist(),
            }
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, (mode, shown) in zip(axes, visible.items()):
                draw(ax, image, boxes, shown)
                ax.set(xlim=(roi[0], roi[2]), ylim=(roi[3], roi[1]))
                matched_ids = {
                    r["gt_index"] for r in summary["modes"][mode]["matches"] if r["tp_localization"]
                }
                tp = sum(int(i) in matched_ids for i in np.flatnonzero(selected))
                # FP is the full-image match status of predictions centred in ROI (all classes).
                centres_p = (shown[:, :2] + shown[:, 2:4]) / 2
                inside = np.all(centres_p >= roi[:2], axis=1) & np.all(centres_p <= roi[2:], axis=1)
                fp = sum(
                    bool(inside[i]) and not r["tp_localization"]
                    for i, r in enumerate(summary["modes"][mode]["matches"])
                )
                counts = {
                    "tp_tiny_pedestrian_localization": tp,
                    "fn_tiny_pedestrian_localization": int(selected.sum()) - tp,
                    "fp_all_classes_centred_in_roi": fp,
                }
                summary["modes"][mode]["roi_counts"] = counts
                ax.set_title(f"{mode}: tiny ped TP/FN={tp}/{int(selected.sum()) - tp}; ROI FP={fp}")
            fig.tight_layout()
            fig.savefig(args.output / f"{path.stem}_tiny_roi.png", dpi=180)
            plt.close(fig)

        if name == CASES[1]:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
            trucks = np.flatnonzero(classes == 5)
            summary["largest_gt_index"] = int(areas.argmax())
            # Verify the report's truck identification instead of guessing from the figure.
            summary["truck_gt_comparison"] = []
            for index in trucks:
                box = boxes[index]
                row = {
                    "gt_index": int(index),
                    "xyxy": box.tolist(),
                    "width": box[2] - box[0],
                    "height": box[3] - box[1],
                    "modes": {},
                }
                for mode, raw in predictions.items():
                    filtered = m.filter_ignored(raw, ignored)
                    same_class = filtered[filtered[:, 5] == 5]
                    ious = m.box_iou_matrix(box[None], same_class[:, :4])[0]
                    best = int(ious.argmax()) if len(ious) else None
                    row["modes"][mode] = (
                        None
                        if best is None
                        else {
                            "prediction": same_class[best].tolist(),
                            "iou": float(ious[best]),
                            "passes_conf025": bool(same_class[best, 4] >= 0.25),
                        }
                    )
                summary["truck_gt_comparison"].append(row)
        if name == CASES[2]:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            for ax, (title, raw) in zip(
                axes,
                [
                    ("before cross-tile NMS", captured["before_nms"]),
                    ("after NMS and cap", predictions["tiled-ov20"]),
                ],
            ):
                draw(ax, image, boxes, raw[raw[:, 4] >= 0.25], grid=True)
                ax.set_title(title)
            fig.tight_layout()
            fig.savefig(args.output / f"{path.stem}_nms.png", dpi=180)
            plt.close(fig)
            summary["nms"] = {key: len(value) for key, value in captured.items()}
            # Save all same-class overlapping post-merge pairs, including low-IoU fragments.
            shown = visible["tiled-ov20"]
            ious = m.box_iou_matrix(shown[:, :4], shown[:, :4])
            summary["remaining_overlap_pairs"] = [
                {"i": i, "j": j, "iou": float(ious[i, j])}
                for i in range(len(shown))
                for j in range(i + 1, len(shown))
                if shown[i, 5] == shown[j, 5] and ious[i, j] > 0
            ]
        (args.output / f"{path.stem}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
