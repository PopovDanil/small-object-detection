# Small-object detection on VisDrone — report and results

Results of project **P17 Small-Object Detection in Aerial Images**: standard resizing (640)
vs higher-resolution inference (1280) vs tiling (640×640 crops at native resolution, 20 %
overlap) for YOLO26m and YOLO26x fine-tuned on VisDrone2019-DET.
Code: <https://github.com/PopovDanil/small-object-detection>.

**Start here:** `report/results.pdf` (or `report/results.md`) — the full report (tables, figures, conclusions, failure
analysis). Every number in it comes from `experiments/metrics/metrics_all.csv`.

```
report/                       the report
  results.pdf                 report, PDF
  results.md                  the same report in markdown (any viewer / GitHub)
  figures/                    images referenced by the report
checkpoints/                  weights (best epoch by val mAP50-95) — full archive only
  yolo26m_s640/best.pt        YOLO26m, 21.8 M parameters, 44 MB
  yolo26m_s640/args.yaml      exact Ultralytics training arguments
  yolo26x_s640/best.pt        YOLO26x, 58.8 M parameters, 118 MB
  yolo26x_s640/args.yaml
experiments/                  everything computed in the experiments
  metrics/                    MAIN: every metric as raw numbers
  conditions/                 raw output of each evaluation run
  training/                   training logs and curves, pilot, config
  figures/                    all experiment plots
  notebooks/                  executed notebooks (with cell outputs)
README.md                     this file
```

Two versions of the archive are built by `python scripts/pack_results.py`:

| Archive | Checkpoints | Where |
|---|---|---|
| `small_object_detection_results.zip` | no (`args.yaml` only) | committed to git |
| `small_object_detection_results_with_weights.zip` | yes (162 MB of weights) | kept out of git (GitHub rejects files > 100 MB); shared separately |

## checkpoints/ — using the weights

The models are trained on the 10 VisDrone classes (ids 0–9: pedestrian, people, bicycle, car,
van, truck, tricycle, awning-tricycle, bus, motor) at `imgsz=640`, but accept any input size.

```python
from ultralytics import YOLO  # ultralytics==8.4.175

model = YOLO("checkpoints/yolo26m_s640/best.pt")
model.predict("frame.jpg", imgsz=1280, conf=0.25, max_det=1000)   # best mode in the report
```

To score them with the same evaluator as the report (from the repository):

```python
from src.evaluation.metrics import evaluate_split, load_split_paths
images, labels = load_split_paths("test")
evaluate_split(labels, images, model, imgsz=1280)                       # 1280
evaluate_split(labels, images, model, inference="tiled", overlap=0.20)  # tiling
```

Or put the `yolo26*_s640/` folders (with `best.pt` under `weights/`) into the repository's
`runs/finetune/` — `highres.ipynb` / `tiling.ipynb` / `results.ipynb` find them automatically
(`EVAL_STAGE=finetune`).

## experiments/metrics/ — every metric (test-dev, 1,610 images, 75,101 objects)

| File | Content |
|---|---|
| `metrics_all.csv` | **every metric, values 0–1**, one row per model × mode (`full-640`, `full-1280`, `tiled-ov20`; for yolo26m also `tiled-ov00`, `tiled-ov40`) |
| `metrics_by_native_size.csv` | the same metrics per native frame size (6 groups: 960×540 … 1920×1080) |
| `metrics_by_native_size_delta.csv` | each mode minus 640 within each frame-size group |
| `results_tables.md` | Tables 1–4 in markdown, values ×100 |
| `table1_overall.csv` | overall metrics + runtime, deltas vs 640 |
| `table2_size_yolo26m.csv`, `table2_size_yolo26x.csv` | per object size (tiny / small / medium / large) |
| `table3_capacity.csv` | yolo26x − yolo26m in each mode |
| `table4_overlap.csv` | tile overlap 0 / 20 / 40 % (yolo26m) |

Columns of `metrics_all.csv`:

| Column | Meaning |
|---|---|
| `map50_95`, `map50` | COCO mAP at IoU 0.50:0.95 / 0.50 (conf 0.001, ≤1000 detections per image) |
| `recall50_25` | recall at IoU ≥ 0.5 and confidence ≥ 0.25 |
| `ap_<g>`, `ap50_<g>`, `recall_<g>`, `n_<g>` | the same for size group `<g>` + object count |
| `runtime_mean_ms`, `runtime_p95_ms` | ms per original image: Tesla T4, batch 1, 10 warm-up images, CUDA sync; tiling counts all tiles + merging; first 200 images |

Size groups (box area in **original**-image pixels): `tiny` < 16², `small_only` 16²–32²,
`medium` 32²–96², `large` ≥ 96² — disjoint, summing to all objects
(27,345 + 23,491 + 21,856 + 2,409 = 75,101). `small` (< 32²) is the COCO definition
(= tiny + small_only), kept for comparability with the literature.

## experiments/conditions/ — raw evaluation runs

| File | Run |
|---|---|
| `full_640.csv`, `full_1280.csv` | full image at 640 / 1280, both models (`highres.ipynb`) |
| `tiled_ov00.csv`, `tiled_ov20.csv`, `tiled_ov40.csv` | tiling with 0 / 20 / 40 % overlap (`tiling.ipynb`; 0 and 40 % — yolo26m only) |
| `tiling_overlap_sweep_yolo26m.csv` | the three overlaps in one table |
| `tiling_vs_baseline.csv` | tiling 20 % vs 640 (from `tiling.ipynb`) |

## experiments/training/ — training

| File | Content |
|---|---|
| `training_summary.csv` | epochs, best epoch, best val mAP50–95 / mAP50, batch, wall time (h) |
| `training_config.json` | shared config of both runs + pilot peak memory |
| `pilot_result.json`, `pilot.log` | batch-size pilot (yolo26x, 1 epoch on 2 % of train: batch 4/8/16) |
| `yolo26m_s640/`, `yolo26x_s640/` | `results.csv` (per-epoch val metrics), `args.yaml`, P/R/F1/PR curves, confusion matrix, train/val batch samples |
| `finetune_val_curves.png` | val mAP per epoch for both models |
| `yolo26m.log`, `yolo26x.log` | Kaggle run logs |

> Training val metrics come from the Ultralytics validator (ignored regions not respected,
> different detection cap) and cannot be compared directly with the test-dev numbers.

## experiments/figures/ and experiments/notebooks/

- `figures/`: accuracy vs runtime, change per object size, tile overlap, tile grid example,
  three failure cases (crowd of tiny objects, large object, tile boundary).
- `notebooks/`: executed `highres`, `tiling`, `results` (evaluation) and `finetune_*`
  (training) — with all cell outputs, viewable without running.

## Protocol in two lines

One fine-tune per model (640, 50 epochs, batch 16, seed 42, best epoch on val); three inference
modes on the **same** weights; one evaluator for all modes; val only for epoch selection, all
numbers on test-dev. Single run without confidence intervals: differences < ~0.5 points are noise.
