# Small-Object Detection in Aerial Images

Does tiled or higher-resolution inference improve small-object detection on aerial
images compared with standard 640px resizing, and what is the accuracy–runtime
trade-off? Full project definition: [docs/proposal.md](docs/proposal.md).

We fine-tune **YOLO26m** and **YOLO26x** (COCO-pretrained) on **VisDrone2019-DET**
(10 classes) at `imgsz=640` and evaluate each checkpoint under three inference
conditions: 640 full-image baseline, 1280 full-image, and 640×640 tiling at native
resolution with 20% overlap.

## Development setup

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12 (managed by uv),
an NVIDIA GPU with recent drivers for training/inference.

```bash
git clone <repo-url>
cd small-object-detection
uv sync                  # installs all runtime + dev dependencies into .venv
uv run pre-commit install   # enable ruff + mypy hooks on every commit
```

Run everything with `uv run ...` so the project environment is used, e.g.
`uv run python -c "import torch; print(torch.cuda.is_available())"`.

### Pre-commit hooks

Hooks are configured in [.pre-commit-config.yaml](.pre-commit-config.yaml):

* **ruff** — lint (with `--fix`) and format; also lints notebook code cells
* **mypy** — static type checking (notebooks are excluded, config in `pyproject.toml`)

```bash
uv run pre-commit run --all-files   # manual check of the whole repo
uv run ruff check .                 # lint only
uv run ruff format .                # format only
uv run mypy src                     # type check only
```

Dependencies are pinned/ranged in [pyproject.toml](pyproject.toml); lockfile: `uv.lock`
(commit it whenever dependencies change).

## Data

`data/` is git-ignored except for `.gitkeep`. The notebook
[notebooks/baseline.ipynb](notebooks/baseline.ipynb) downloads the official
VisDrone2019-DET zips (train 6,471 / val 548 / test-dev 1,610 images) into
`data/raw/` and converts them once to:

```
data/
├── raw/VisDrone2019-DET-{train,val,test-dev}/   # official images + annotations (kept)
├── images/{train,val,test}/                     # symlinks to native-resolution images
├── labels/{train,val,test}/                     # YOLO labels, classes 0–9
├── ignore/{train,val,test}/                     # ignored/don't-care boxes (evaluator only)
├── manifests/{train,val,test}.csv               # seed-42 manifests
└── visdrone.yaml                                # shared Ultralytics dataset config
```

Class mapping: VisDrone 1–10 → YOLO 0–9
(`pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`).
VisDrone classes 0 (ignored regions) and 11 (don't care) are dropped from training
targets and respected by the evaluator. Images are never resampled on disk.

### Exploratory data analysis

[notebooks/eda.ipynb](notebooks/eda.ipynb) verifies whether the official splits
are stratified (class and object-size distributions via total variation distance)
and displays sample photos. Key finding: they are only approximately stratified —
test-dev is tiny-heavy (36.4 % of objects < 16² px vs 26.0 % in train) and val is
enriched in `people`/`motor`, which is why val is used **only** for checkpoint
selection and all reported scores use test-dev.

## Running the baseline (stage 1)

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/baseline.ipynb \
    --inplace --ExecutePreprocessor.timeout=-1
```

or open it interactively with `uv run jupyter lab notebooks/baseline.ipynb`.

The notebook downloads the data, converts annotations, scores the vanilla
(COCO-pretrained) models, fine-tunes both models (short stage-1 run, `EPOCHS=5`),
and writes baseline scores on the **test-dev** split to
`runs/baseline/baseline_scores_vanilla_test_640.csv` and
`runs/baseline/baseline_scores_test_640.csv` (validation is used only for
checkpoint/hyperparameter selection). Training artifacts land in
`runs/baseline/<model>_s640/` (git-ignored). See
[docs/baseline_results.md](docs/baseline_results.md) for the full baseline report.

### Shared evaluator

Every experiment must be scored with
`evaluate_split(labels, images, model, imgsz=..., conf=...)` from
[`src/evaluation/metrics.py`](src/evaluation/metrics.py), which returns:
mAP50–95, mAP50 (conf 0.001), recall@IoU0.5/conf0.25, tiny (<16² px) AP + recall,
COCO small/medium/large AP + recall with per-group object counts, the disjoint
`small_only` group (16²–32² px, so tiny / small_only / medium / large partition all
objects), AP50 per size group (`ap50_<group>`), and mean/p95 runtime per original
image (batch 1, warm-up, CUDA sync). All metrics are computed in original-image
coordinates; ignored regions are respected. Every condition keeps at most
`MAX_DET = 1000` detections per original image (tiled: after merging); before this,
Ultralytics' default `max_det=300` capped full-image inference per image but tiling
per tile, which favoured tiling on crowded images. Fine-tuned models
use the identity class mapping; zero-shot COCO models use `coco_class_map`. Do not
introduce a second evaluator — extend this one.

## Running the full fine-tune (stage 2a)

[notebooks/finetune.ipynb](notebooks/finetune.ipynb) implements the proposal's
full training protocol (max 50 epochs, `patience=15`, cosine LR, `close_mosaic=10`,
all layers trainable) with one shared config for both models and a short pilot
that fixes the effective batch size. It writes validation-selected checkpoints to
`runs/finetune/<model>_s640/weights/best.pt`. Expect several hours per model; run
it on a GPU/Kaggle session. Set `VISDRONE_DATA` if the prepared `data/` tree lives
elsewhere. Env vars select the run: `FT_MODELS` (e.g. `yolo26m`), `FT_DEVICE`
(GPU index), `FT_BATCH` (pin the batch, skips the pilot), `FT_WORKERS`.

On a 2-GPU Kaggle session (2× T4), `bash scripts/kaggle_finetune.sh` installs the
pinned dependencies, prepares the data with
[`scripts/prepare_data.py`](scripts/prepare_data.py) (only the data cells of the
baseline notebook), runs the pilot once and then trains `yolo26m` on GPU 0 and
`yolo26x` on GPU 1 in parallel with the same batch. See
[docs/finetuning.md](docs/finetuning.md).

## Final evaluation (stage 2b): 640 / 1280 / tiling

All three inference conditions are scored on the **same checkpoints**. `EVAL_STAGE`
(env var) selects them: `finetune` (default, `runs/finetune/`) or `baseline`
(stage-1, `runs/baseline/`). Run, in order, with the same `EVAL_STAGE`:

1. [notebooks/highres.ipynb](notebooks/highres.ipynb) — 640 and 1280 full-image
   (`evaluate_split(..., imgsz=640|1280)`) → `runs/highres/<stage>/`. Its appendix
   scores every condition per native image size (960×540, 1360×765, 1920×1080; images
   with a long side < 1280 are upscaled at 1280, not enriched) via the standalone
   [`src/evaluation/native_size.py`](src/evaluation/native_size.py); `RUN_APPENDIX=0`
   skips it.
2. [notebooks/tiling.ipynb](notebooks/tiling.ipynb) — 640×640 native-resolution tiles
   with 20 % overlap, the 0/20/40 % overlap sensitivity sweep for `yolo26m` and
   failure cases (`evaluate_split(..., inference="tiled")`) → `runs/tiling/<stage>/`.
3. [notebooks/results.ipynb](notebooks/results.ipynb) — no inference; merges the CSVs
   into the final tables → `runs/results/<stage>/` (CSV, `results_tables.md`, PNG):
   overall metrics + runtime with Δ vs 640, per disjoint object-size group (AP50–95,
   AP50, recall, counts), yolo26x − yolo26m per condition, overlap sweep.

On Kaggle: `FINETUNE_SRC=<path to runs/finetune of the training output> bash
scripts/kaggle_eval.sh` runs all three. Stage-1 reports:
[docs/baseline_results.md](docs/baseline_results.md),
[docs/tiling_results.md](docs/tiling_results.md).

### Conventions

* Seed 42 everywhere (training, manifests, subsets).
* NMS IoU 0.7, eval conf 0.001 (AP) / 0.25 (recall), recall IoU 0.5 — fixed
  before results are examined.
* One shared training YAML (`data/visdrone.yaml`) for both models; the only
  difference between runs is model size and its pretrained weights.
* Validation is used only for checkpoint/hyperparameter selection; all reported
  scores are computed on test-dev with the shared evaluator.
