# Results — raw numbers

Evaluation of the full fine-tune (`EVAL_STAGE=finetune`) on VisDrone2019-DET
**test-dev** (1,610 images, 75,101 objects), produced by `scripts/kaggle_eval.sh`
(`highres.ipynb` → `tiling.ipynb` → `results.ipynb`) on a Kaggle Tesla T4. The written
report with tables, figures and discussion is [docs/results.md](../docs/results.md).

| File | Content |
|---|---|
| `metrics_all.csv` | **every metric, raw (0–1)**, one row per model × inference condition |
| `metrics_by_native_size.csv` | appendix: the same metrics split by native image size |
| `training_summary.csv` | epochs, selected epoch, best validation mAP, batch, wall time |
| `training_config.json` | the shared training config of both runs + pilot peak memory |

## `metrics_all.csv`

Rows: `yolo26m` / `yolo26x` × `full-640`, `full-1280`, `tiled-ov20` (main conditions) and,
for `yolo26m` only, `tiled-ov00` / `tiled-ov40` (overlap sensitivity sweep).

| Column | Meaning |
|---|---|
| `condition` | `full-640` / `full-1280`: whole image letterboxed to that long side; `tiled-ovXX`: 640×640 crops at native resolution, XX % overlap, merged with class-aware NMS |
| `map50_95`, `map50` | COCO mAP at IoU 0.50:0.95 / 0.50 (conf 0.001, ≤1000 detections per image) |
| `recall50_25` | recall at IoU ≥ 0.5 and confidence ≥ 0.25 |
| `ap_<g>`, `ap50_<g>`, `recall_<g>` | the same per object-size group `<g>` |
| `n_<g>` | number of ground-truth objects in group `<g>` |
| `runtime_mean_ms`, `runtime_p95_ms` | ms per **original** image (batch 1, 10 warm-up, CUDA sync; tiling counts all crops + merge), first `runtime_images` images |

Object-size groups (area in original-image pixels):

| `<g>` | Area | Disjoint? |
|---|---|---|
| `tiny` | < 16² | yes |
| `small_only` | 16² – 32² | yes |
| `medium` | 32² – 96² | yes |
| `large` | ≥ 96² | yes |
| `small` | < 32² (COCO definition, = tiny + small_only) | no, kept for COCO comparability |

`tiny + small_only + medium + large = all` (27,345 + 23,491 + 21,856 + 2,409 = 75,101).

## `metrics_by_native_size.csv`

Same metrics per `native` image size (`960x540`, `1360x765`, `1400x788`, `1400x1050`,
`1916x1078`, `1920x1080`) and `inference` condition; runtime there is measured on the
first 60 images of each group. Groups differ in camera, altitude and scene — compare
conditions **within** a group, not groups with each other.
