# Baseline results — VisDrone2019-DET (test-dev)

Stage-1 reference results for the small-object-detection study
([docs/proposal.md](proposal.md)). Reported scores are computed on the official
VisDrone2019-DET **test-dev** split (1,610 images) with the single shared
evaluator [`src/evaluation/metrics.py`](../src/evaluation/metrics.py) at
`imgsz=640`. The official **validation** split is used only for
checkpoint/hyperparameter selection and is never reported as a result.

Two baselines are reported:

1. **Vanilla** — the COCO-pretrained `yolo26m.pt` / `yolo26x.pt` weights scored
   zero-shot (80 COCO classes), i.e. **no VisDrone fine-tuning**;
2. **Fine-tuned (stage-1)** — the two models after the short 5-epoch VisDrone
   fine-tune (checkpoint selected on val), for reference and to quantify what
   fine-tuning buys.

The vanilla baseline is the primary result of this document; it is the
untrained reference point that all later experiments (1280 full-image, tiling,
overlap sweep) will be compared against.

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

All timings come from the same machine with batch size 1, 10 warm-up images
and explicit `torch.cuda.synchronize()` around each timed call.

## 2. Data

Official splits are preserved and images are never resampled on disk. Official
classes 1–10 are mapped to YOLO ids 0–9
(`pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus,
motor`); classes 0 (ignored regions) and 11 (don't care) are dropped from
targets and handled by the evaluator.

| Split | Images | Boxes | Ignored regions | Role |
|---|---:|---:|---:|---|
| train | 6,471 | 343,197 | 10,345 | fine-tune only |
| val | 548 | 38,759 | 1,410 | checkpoint / HP selection only |
| **test-dev** | **1,610** | **75,101** | **2,445** | **all reported scores** |

Test-dev object counts by size group (original-image coordinates):

| Group | Definition | Objects |
|---|---|---:|
| all | any area | 75,101 |
| tiny | area < 16² px (= 256 px²) | 27,345 |
| small | area < 32² px (= 1024 px²) | 50,836 |
| medium | 32² px ≤ area < 96² px | 21,856 |
| large | area ≥ 96² px | 2,409 |

`tiny` is a subset of `small`, so the counts overlap by design.

## 3. Models

| Model | Parameters | GFLOPs | Initialization |
|---|---:|---:|---|
| yolo26m | 21,788,308 (21.8 M) | 75.0 | COCO-pretrained (`yolo26m.pt`, 80 classes) |
| yolo26x | 58,831,668 (58.8 M) | 209.2 | COCO-pretrained (`yolo26x.pt`, 80 classes) |

### Vanilla class mapping

COCO-pretrained models emit 80 COCO classes, while the protocol scores the 10
VisDrone classes. For the vanilla baseline, predictions are mapped **by class
name** and unmapped categories are dropped:

| COCO class | → VisDrone class |
|---|---|
| person | pedestrian |
| bicycle | bicycle |
| car | car |
| motorcycle | motor |
| bus | bus |
| truck | truck |
| *(all other 74 COCO classes)* | **dropped** |

Consequences that must be kept in mind when reading the vanilla table:

* VisDrone `van`, `tricycle`, `awning-tricycle` have no COCO counterpart, and
  COCO `person` cannot be split into VisDrone `pedestrian` + `people`.
  **Class-aware AP for those four VisDrone classes is therefore 0 in the
  vanilla baseline** (no prediction can ever carry those category ids).
* Recall is matched **class-agnostically** (IoU only), so the same objects can
  still be counted as found. This is why vanilla recall (e.g. tiny 2.8–3.8 %)
  is higher than vanilla AP (tiny 0.8–1.2 %).

The fine-tuned models use the identity mapping (their own 10 class ids).

## 4. Fixed protocol / all parameters

All constants are fixed before any result is examined and are defined in
[`src/evaluation/metrics.py`](../src/evaluation/metrics.py).

| Parameter | Value |
|---|---|
| Seed (training, manifests, subsets) | 42 |
| Training image size | 640 |
| Inference image size (baseline condition) | 640 |
| Epochs (stage-1 short fine-tune) | 5 |
| Batch size (training) | 4 |
| LR schedule | cosine (`cos_lr=True`), AdamW chosen by `optimizer=auto`, lr0 ≈ 7.14e-4 |
| Early stopping (stage-1) | `patience=0` (disabled; full protocol later: max 50, patience 15) |
| `close_mosaic` | 1 |
| Augmentation | Ultralytics defaults (mosaic, HSV, horizontal flip, deterministic) |
| AP confidence threshold | 0.001 |
| Recall confidence threshold | 0.25 |
| Recall IoU threshold | 0.5 |
| NMS IoU | 0.7 |
| Ignored-region IoU | 0.5 (predictions overlapping ignored regions are dropped) |
| Tiny threshold | area < 16² px = 256 px² |
| Small threshold | area < 32² px = 1024 px² |
| Medium/large boundary | 96² px = 9216 px² |
| Runtime protocol | batch size 1, warm-up 10 images, CUDA sync, all tiles + merge counted |
| Number of classes | 10 (VisDrone) |

### Training time

| Model | Wall-clock (5 epochs, incl. validation) |
|---|---:|
| yolo26m | 875.8 s (14.6 min) |
| yolo26x | 1,845.6 s (30.8 min) |

## 5. Vanilla (COCO-pretrained) baseline — primary result

Command: `run_vanilla_baseline(base_models, split="test", imgsz=640)` in
[`notebooks/baseline.ipynb`](../notebooks/baseline.ipynb) (section 7), saving
`runs/baseline/baseline_scores_vanilla_test_640.csv`. Scores are on the 1,610
test-dev images.

| Metric | yolo26m | yolo26x |
|---|---:|---:|
| mAP50–95 | 0.0586 | 0.0797 |
| mAP50 | 0.0970 | 0.1297 |
| Recall@IoU0.5/conf0.25 | 0.2072 | 0.2421 |
| AP50–95 tiny (<16² px) | 0.0084 | 0.0117 |
| Recall tiny | 0.0281 | 0.0380 |
| AP50–95 small (<32² px) | 0.0215 | 0.0295 |
| Recall small | 0.0902 | 0.1087 |
| AP50–95 medium (32²–96²) | 0.0993 | 0.1294 |
| Recall medium | 0.4238 | 0.4947 |
| AP50–95 large (≥96²) | 0.2416 | 0.2777 |
| Recall large | 0.7115 | 0.7646 |
| Objects (all / tiny / small / medium / large) | 75,101 / 27,345 / 50,836 / 21,856 / 2,409 | same |
| Runtime mean (ms/image) | 12.462 | 23.172 |
| Runtime p95 (ms/image) | 15.230 | 28.734 |

## 6. Fine-tuned (stage-1) reference

Command: `run_baseline_inference(checkpoints, split="test", imgsz=640)` (section 8),
saving `runs/baseline/baseline_scores_test_640.csv`. Checkpoints:
`runs/baseline/{yolo26m,yolo26x}_s640/weights/best.pt` (selected on val).

| Metric | yolo26m | yolo26x |
|---|---:|---:|
| mAP50–95 | 0.1846 | 0.2065 |
| mAP50 | 0.3080 | 0.3420 |
| Recall@IoU0.5/conf0.25 | 0.5160 | 0.5542 |
| AP50–95 tiny (<16² px) | 0.0373 | 0.0430 |
| Recall tiny | 0.1965 | 0.2374 |
| AP50–95 small (<32² px) | 0.0822 | 0.0969 |
| Recall small | 0.3711 | 0.4141 |
| AP50–95 medium (32²–96²) | 0.2813 | 0.3146 |
| Recall medium | 0.8100 | 0.8394 |
| AP50–95 large (≥96²) | 0.5049 | 0.5048 |
| Recall large | 0.9066 | 0.9240 |
| Runtime mean (ms/image) | 12.680 | 23.307 |
| Runtime p95 (ms/image) | 15.575 | 28.838 |

### Effect of fine-tuning (fine-tuned − vanilla)

| Metric | Δ yolo26m | Δ yolo26x |
|---|---:|---:|
| mAP50–95 | +0.1260 | +0.1268 |
| mAP50 | +0.2110 | +0.2122 |
| Recall@IoU0.5/conf0.25 | +0.3088 | +0.3121 |
| AP50–95 tiny | +0.0289 | +0.0312 |
| Recall tiny | +0.1684 | +0.1995 |
| AP50–95 small | +0.0607 | +0.0674 |
| Recall small | +0.2808 | +0.3054 |
| AP50–95 medium | +0.1820 | +0.1852 |
| Recall medium | +0.3863 | +0.3447 |
| AP50–95 large | +0.2633 | +0.2271 |
| Recall large | +0.1951 | +0.1594 |

## 7. Observations

* **Fine-tuning dominates for small objects.** Vanilla tiny recall is 2.8–3.8 %;
  fine-tuning lifts it to 19.7–23.7 %, and tiny AP from 0.8–1.2 % to
  3.7–4.3 %. The medium/small groups gain the most in absolute recall
  (+0.31–0.39).
* **Large objects transfer well even zero-shot** (vanilla recall 71.2–76.5 %),
  so the difficulty is concentrated in small/tiny objects, not in the backbone.
* **yolo26x ≥ yolo26m on most metrics**, but the gap is modest (mAP50–95 +0.021
  vanilla, +0.022 fine-tuned) and at large-object AP they are tied (0.5049 vs
  0.5048). The larger model costs ~1.85× runtime.
* **Runtime at 640 is stable and cheap** (~12.5 ms m / ~23.2 ms x mean), which
  is the reference cost that tiling and 1280 will be compared against.
* Test-dev scores are systematically lower than validation scores (same
  protocol), so val must not be used for reported numbers — only for selection.

## 8. Metric implementation verification

Every metric required by the proposal is implemented once in
[`src/evaluation/metrics.py`](../src/evaluation/metrics.py) and reached through
the single entry point `evaluate_split` (`metrics.py:437`).

| Required metric | Implementation |
|---|---|
| mAP50–95 and mAP50 at conf 0.001 | `compute_map50_95` (`metrics.py:332`) via pycocotools `COCOeval` |
| Recall at IoU 0.5 / conf 0.25 | `compute_recall` (`metrics.py:381`) |
| Tiny-object AP50–95 and recall, reported together | tiny area range in `AREA_RANGES` + `recall_tiny` |
| Small/medium/large AP + recall + object counts | `AREA_RANGES` groups + `n_{group}` counts |
| Runtime mean and p95 per original image | `benchmark_runtime` (`metrics.py:245`), batch 1, warm-up, CUDA sync |
| Single evaluator for all conditions | `evaluate_split` (`metrics.py:437`) |
| Ignored regions respected | `filter_ignored` (`metrics.py:165`) at IoU ≥ 0.5 |
| Failure analysis | section 9 (report level) |

Verified gaps addressed while factoring the evaluator out of the notebook:

* the vanilla baseline required a **class-mapping** path, added as
  `coco_class_map` (`metrics.py:207`) + `remap_classes` (`metrics.py:191`);
  fine-tuned models pass `class_map=None` and behave exactly as before;
* the notebook's inline evaluator (section 6) was **replaced by an import**, so
  there is now exactly one evaluator implementation, as required by the README.

## 9. Failure analysis (baseline evidence)

Three observed failure modes, grounded in the measured numbers above; the
qualitative tiling case is deferred because the baseline condition runs
full-image only.

1. **Tiny objects (primary failure).** Even after fine-tuning, tiny
   (<16² px) AP reaches only 0.037–0.043 while large-object AP reaches
   0.505. Tiny recall (~20–24 %) is far below medium/large recall
   (~81–92 %), showing both missed detections and poor localization/class of
   the few tiny objects that are found. This is the effect the study targets.
2. **Occlusion / crowding.** Overall recall (0.52–0.55 fine-tuned) is much
   lower than large-object recall (0.91–0.92), and most misses concentrate in
   the dense small/medium clusters. Class-agnostic matching hides class
   confusion here: a nearby vehicle box can satisfy IoU for the wrong class, so
   AP drops faster than recall (e.g. mAP50–95 0.18–0.21 vs recall 0.52–0.55).
3. **Class shift / label mapping.** In the vanilla baseline the COCO→VisDrone
   mapping makes `van`, `tricycle`, `awning-tricycle`, and `people`
   unreachable for class-aware AP, and collapses `pedestrian`/`people` into
   one category. This is a protocol-documented detection limit, not a model
   error, and it disappears after fine-tuning.

Tiling-boundary regressions are not observable in the 640 full-image baseline;
they will be measured once the tiling condition is implemented.

## 10. Reproduce

```bash
# Install
uv sync

# Full stage-1 notebook (downloads data, fine-tunes both models, scores test-dev, writes CSVs)
uv run jupyter nbconvert --to notebook --execute notebooks/baseline.ipynb \
    --inplace --ExecutePreprocessor.timeout=-1
```

Or run the two scoring sections independently once `data/` is prepared and the
checkpoints exist (vanilla scoring needs no training):

```python
from src.evaluation.metrics import coco_class_map, evaluate_split, load_split_paths
from ultralytics import YOLO

images, labels = load_split_paths("test")  # test-dev, not val
for name in ("yolo26m", "yolo26x"):
    model = YOLO(f"{name}.pt")  # COCO weights for the vanilla baseline
    metrics = evaluate_split(
        labels=labels,
        images=images,
        model=model,
        imgsz=640,
        class_map=coco_class_map(model),
    )
```

## 11. Artifacts

| Artifact | Path |
|---|---|
| Shared evaluator | `src/evaluation/metrics.py` |
| Notebook (sections 6–9) | `notebooks/baseline.ipynb` |
| Vanilla scores (test-dev) | `runs/baseline/baseline_scores_vanilla_test_640.csv` |
| Fine-tuned scores (test-dev) | `runs/baseline/baseline_scores_test_640.csv` |
| Fine-tuned checkpoints | `runs/baseline/{yolo26m,yolo26x}_s640/weights/best.pt` |
| Training logs | `runs/baseline/{yolo26m,yolo26x}_s640/results.csv` |
