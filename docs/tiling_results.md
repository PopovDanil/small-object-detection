# Tiling results — VisDrone2019-DET (test-dev)

Stage-2 results for the small-object-detection study
([docs/proposal.md](proposal.md)): **tiled inference at native resolution**
compared with the 640 full-image baseline in
[docs/baseline_results.md](baseline_results.md).

Reported scores are computed on the official VisDrone2019-DET **test-dev** split
(1,610 images) with the single shared evaluator
[`src/evaluation/metrics.py`](../src/evaluation/metrics.py) called with
`inference="tiled"`. The official **validation** split is used only for
checkpoint/hyperparameter selection and is never reported as a result.

Two conditions are reported:

1. **Main tiling condition (primary result)** — 640 × 640 native-resolution
   crops with **20 % overlap**, for **both** models;
2. **Overlap sensitivity sweep** — 0 %, 20 %, 40 % overlap for `yolo26m`, all
   other factors fixed.

The detections are merged with class-aware NMS at the same fixed IoU 0.7 used
everywhere, predictions are mapped back to original-image coordinates, and
ignored regions are respected, exactly as in the baseline.

> These numbers follow our protocol and are **not** official VisDrone
> leaderboard scores.

## 1. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) |
| PyTorch | 2.14.1+cu130 |
| Ultralytics | 8.4.175 |
| Python | 3.12.13 |
| Data | VisDrone2019-DET, official splits, native resolution |

All timings come from the same machine with batch size 1, warm-up and explicit
`torch.cuda.synchronize()` around each timed call; the tiled timing loop counts
image decode, every tile crop, tile inference and NMS merging.

## 2. Data and models

Data splits, class mapping, size-group definitions and object counts are
identical to the baseline (see
[docs/baseline_results.md](baseline_results.md) §2). Test-dev contains 75,101
objects, of which 27,345 are tiny (<16² px) and 50,836 small (<32² px).

The two checkpoints are the **stage-1 fine-tuned** models reused unchanged from
the baseline; only inference changes (training stays at `imgsz=640`).

| Model | Weights | Parameters | Initialization |
|---|---|---:|---|
| yolo26m | `runs/baseline/yolo26m_s640/weights/best.pt` | 21.8 M | COCO → VisDrone (10 classes), stage-1 |
| yolo26x | `runs/baseline/yolo26x_s640/weights/best.pt` | 58.8 M | COCO → VisDrone (10 classes), stage-1 |

To score the later full-protocol fine-tuned weights, change the `WEIGHTS` dict at
the top of [`notebooks/tiling.ipynb`](../notebooks/tiling.ipynb) and re-run; no
other edit is required.

## 3. Fixed tiling parameters

All constants are fixed before any result is examined and live in
[`src/evaluation/metrics.py`](../src/evaluation/metrics.py).

| Parameter | Value |
|---|---|
| Tile size | 640 × 640 px, native resolution (no resize) |
| Main overlap | 20 % (stride 512 px) |
| Sweep overlaps | 0 %, 20 %, 40 % (yolo26m only) |
| Tile coverage | last window flush with the far edge, so the whole image is covered |
| Cross-tile merging | class-aware NMS at IoU 0.7 (`torchvision.ops.batched_nms`) |
| Allocation of a detection | highest-confidence box per class kept among duplicates |
| AP confidence threshold | 0.001 |
| Recall confidence threshold | 0.25 |
| Recall IoU threshold | 0.5 |
| Ignored-region IoU | 0.5 (predictions overlapping ignored regions are dropped) |
| Tiny threshold | area < 16² px = 256 px² |
| Small threshold | area < 32² px = 1024 px² |
| Medium/large boundary | 96² px = 9216 px² |
| Runtime protocol | batch size 1, warm-up 10, CUDA sync, all tiles + merge counted, first 200 test images |
| Seed | 42 |

Tiles per image (three native sizes present in test-dev):

| Image size | Tiles @ 0 % | Tiles @ 20 % | Tiles @ 40 % |
|---|---:|---:|---:|
| 960 × 540 | 2 | 2 | 2 |
| 1360 × 765 | 6 | 6 | 6 |
| 1920 × 1080 | 6 | 8 | 15 |

## 4. Main tiling condition (20 % overlap) — all metrics, both models

Command: `run_tiled_eval(checkpoints, overlap=0.20, split="test")` in
[`notebooks/tiling.ipynb`](../notebooks/tiling.ipynb) (section 3), saving
`runs/tiling/tiling_scores_test_ov20.csv`. Scores are on the 1,610 test-dev
images.

| Metric | yolo26m | yolo26x |
|---|---:|---:|
| mAP50–95 | 0.1948 | 0.2252 |
| mAP50 | 0.3393 | 0.3857 |
| Recall@IoU0.5/conf0.25 | 0.6303 | 0.6679 |
| AP50–95 tiny (<16² px) | 0.0643 | 0.0857 |
| Recall tiny | 0.3849 | 0.4406 |
| AP50–95 small (<32² px) | 0.1195 | 0.1444 |
| Recall small | 0.5343 | 0.5787 |
| AP50–95 medium (32²–96²) | 0.2831 | 0.3250 |
| Recall medium | 0.8326 | 0.8545 |
| AP50–95 large (≥96²) | 0.2903 | 0.3214 |
| Recall large | 0.8211 | 0.8568 |
| Objects (all / tiny / small / medium / large) | 75,101 / 27,345 / 50,836 / 21,856 / 2,409 | same |
| Runtime mean (ms/image) | 68.44 | 174.79 |
| Runtime p95 (ms/image) | 105.84 | 262.82 |

## 5. Tiling vs 640 full-image baseline

Baseline rows are from
`runs/baseline/baseline_scores_test_640.csv`; deltas are `tiling − baseline`
(positive is better for accuracy, worse for runtime). Full table:
`runs/tiling/tiling_vs_baseline_test.csv`.

| Metric | yolo26m base | yolo26m tiled | Δ m | yolo26x base | yolo26x tiled | Δ x |
|---|---:|---:|---:|---:|---:|---:|
| mAP50–95 | 0.1846 | 0.1948 | **+0.0103** | 0.2065 | 0.2252 | **+0.0187** |
| mAP50 | 0.3080 | 0.3393 | +0.0313 | 0.3420 | 0.3857 | +0.0437 |
| Recall@IoU0.5/conf0.25 | 0.5160 | 0.6303 | +0.1143 | 0.5542 | 0.6679 | +0.1137 |
| AP50–95 tiny | 0.0373 | 0.0643 | **+0.0270** | 0.0430 | 0.0857 | **+0.0428** |
| Recall tiny | 0.1965 | 0.3849 | **+0.1884** | 0.2374 | 0.4406 | **+0.2031** |
| AP50–95 small | 0.0822 | 0.1195 | +0.0372 | 0.0969 | 0.1444 | +0.0475 |
| Recall small | 0.3711 | 0.5343 | +0.1633 | 0.4141 | 0.5787 | +0.1647 |
| AP50–95 medium | 0.2813 | 0.2831 | +0.0018 | 0.3146 | 0.3250 | +0.0104 |
| Recall medium | 0.8100 | 0.8326 | +0.0226 | 0.8394 | 0.8545 | +0.0151 |
| AP50–95 large | 0.5049 | 0.2903 | **−0.2146** | 0.5048 | 0.3214 | **−0.1834** |
| Recall large | 0.9066 | 0.8211 | −0.0855 | 0.9240 | 0.8568 | −0.0672 |
| Runtime mean (ms/image) | 12.68 | 68.44 | +55.76 (×5.40) | 23.31 | 174.79 | +151.46 (×7.50) |
| Runtime p95 (ms/image) | 15.58 | 105.84 | +90.27 | 28.84 | 262.82 | +233.98 |

Relative gains: tiny AP **×1.72 (m)** / **×2.00 (x)**, tiny recall
**×1.96 (m)** / **×1.86 (x)**; runtime **×5.40 (m)** / **×7.50 (x)**.

## 6. Overlap sensitivity sweep — yolo26m

Command: sections 3.1/3.2 of [`notebooks/tiling.ipynb`](../notebooks/tiling.ipynb),
saving `runs/tiling/tiling_overlap_sweep_test_yolo26m.csv`.

| Metric | 0 % | 20 % | 40 % |
|---|---:|---:|---:|
| mAP50–95 | 0.1892 | 0.1948 | 0.1953 |
| mAP50 | 0.3356 | 0.3393 | 0.3390 |
| Recall@IoU0.5/conf0.25 | 0.6203 | 0.6303 | 0.6359 |
| AP50–95 tiny | 0.0627 | 0.0643 | 0.0644 |
| Recall tiny | 0.3760 | 0.3849 | 0.3884 |
| AP50–95 small | 0.1169 | 0.1195 | 0.1197 |
| Recall small | 0.5257 | 0.5343 | 0.5395 |
| AP50–95 medium | 0.2741 | 0.2831 | 0.2820 |
| Recall medium | 0.8205 | 0.8326 | 0.8383 |
| AP50–95 large | 0.2783 | 0.2903 | 0.2894 |
| Recall large | 0.8020 | 0.8211 | 0.8340 |
| Runtime mean (ms/image) | 63.34 | 68.44 | 85.79 |
| Runtime p95 (ms/image) | 80.36 | 105.84 | 196.70 |

## 7. Observations

* **Tiling sharply improves small-object metrics.** Tiny AP roughly doubles
  (0.037→0.064 for m, 0.043→0.086 for x) and tiny recall roughly doubles
  (0.197→0.385 for m, 0.237→0.441 for x). Overall recall rises by ~0.114 for both
  models and mAP50 by ~0.03–0.04. This confirms the primary hypothesis: the
  gain from tiling is concentrated on the smallest objects.
* **The whole mAP nudges up, but the gain is diluted by large objects.**
  mAP50–95 improves only +0.010 (m) and +0.019 (x) because **large-object AP
  collapses** (0.505→0.290 for m, 0.505→0.321 for x) and large recall drops
  0.86–0.94→0.82–0.86. Objects that no longer fit inside a single 640 crop are
  split/clipped across tiles, and the merged box localizes them worse.
* **Medium objects are essentially unchanged** (+0.002 m, +0.010 x AP), so the
  native-resolution benefit is confined to tiny/small objects, not a uniform
  resolution effect.
* **yolo26x ≥ yolo26m under tiling** (mAP50–95 +0.030, tiny AP +0.021, tiny
  recall +0.056), i.e. added capacity helps small objects even at fixed tiling —
  but it costs ~2.55× the tiled runtime of yolo26m (174.8 vs 68.4 ms).
* **Runtime is the price.** Tiling costs ~5.4× (m) / ~7.5× (x) the 640 baseline,
  because each image becomes 2–15 forward passes plus merging; the heavier
  yolo26x multiplies this further. p95 is much higher than the mean at 20–40 %
  overlap, reflecting the images that need the most tiles.
* **Overlap: more helps marginally, at rising cost.** 0 %→20 % buys tiny AP
  +0.0016 and tiny recall +0.0089; 20 %→40 % adds only +0.0001 tiny AP and
  +0.0035 tiny recall while runtime p95 nearly doubles (105.8→196.7 ms). The
  20 % setting in the proposal is a good accuracy/cost compromise; 40 % is not
  justified by the gains.

## 8. Failure analysis

Three failure modes grounded in the measured numbers and in the qualitative
figures produced in section 5 of [`notebooks/tiling.ipynb`](../notebooks/tiling.ipynb)
(selected deterministically: most-crowded image, image with the largest object,
first image with a large object spanning the `x=640` tile edge).

1. **Large-object regression (new, caused by tiling).** Large AP drops by
   ~0.18–0.21 and large recall by ~0.07–0.09. A large object spanning tile
   borders is detected as two partial boxes that NMS cannot merge into the true
   extent, or is missed because neither tile contains it whole. This is the main
   regression introduced by tiling and is invisible in the 640 baseline.
2. **Tiny objects (primary failure, still).** Despite roughly doubling, tiny AP
   reaches only 0.064–0.086 and tiny recall 0.385–0.441, far below medium/large
   recall. Native-resolution crops recover many more tiny objects than 640
   resizing, but most tiny objects are still missed or mislocalized.
3. **Occlusion / crowding.** Overall recall (0.63–0.67) stays below large-object
   recall, and misses concentrate in dense small/medium clusters. Class-agnostic
   recall hides class confusion: mAP50–95 (0.19–0.23) is far below mAP50
   (0.34–0.39), showing that many matched boxes carry the wrong class.

Interaction worth noting: `yolo26x` regresses less on large objects than
`yolo26m` under tiling (large AP 0.321 vs 0.290) and gains more on tiny objects,
but pays the highest runtime — so the model-size comparison does not fully
decouple from the resolution effect.

## 9. Reproduce

```bash
# Full tiling notebook (scores both models at 20% overlap and sweeps 0/20/40% for yolo26m)
uv run jupyter nbconvert --to notebook --execute notebooks/tiling.ipynb \
    --inplace --ExecutePreprocessor.timeout=-1
```

Or score one condition directly with the shared evaluator:

```python
from src.evaluation.metrics import evaluate_split, load_split_paths
from ultralytics import YOLO

images, labels = load_split_paths("test")  # test-dev, not val
model = YOLO("runs/baseline/yolo26m_s640/weights/best.pt")
metrics = evaluate_split(
    labels=labels,
    images=images,
    model=model,
    inference="tiled",  # native-resolution 640x640 tiles
    tile=640,
    overlap=0.20,  # or 0.0 / 0.40 for the sweep
    runtime_images=200,
)
```

To use later fine-tuned weights, point `evaluate_split`'s `model` (or the
notebook's `WEIGHTS` dict) at the new checkpoint — no other change is needed.

## 10. Artifacts

| Artifact | Path |
|---|---|
| Shared evaluator (tiled path added) | `src/evaluation/metrics.py` |
| Notebook (sections 2–6) | `notebooks/tiling.ipynb` |
| Main tiling scores (test-dev, 20 %) | `runs/tiling/tiling_scores_test_ov20.csv` |
| Overlap sweep, 0 % | `runs/tiling/tiling_scores_test_ov00.csv` |
| Overlap sweep, 40 % | `runs/tiling/tiling_scores_test_ov40.csv` |
| Overlap sweep table (yolo26m) | `runs/tiling/tiling_overlap_sweep_test_yolo26m.csv` |
| Baseline vs tiling table | `runs/tiling/tiling_vs_baseline_test.csv` |
| Accuracy–runtime plot | `runs/tiling/tiling_tradeoff.png` |
| Overlap-sweep plot | `runs/tiling/tiling_overlap_sweep.png` |
| Checkpoints (unchanged) | `runs/baseline/{yolo26m,yolo26x}_s640/weights/best.pt` |
