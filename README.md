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

## Running the baseline (stage 1)

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/baseline.ipynb \
    --inplace --ExecutePreprocessor.timeout=-1
```

or open it interactively with `uv run jupyter lab notebooks/baseline.ipynb`.

The notebook downloads the data, converts annotations, fine-tunes both models
(short stage-1 run, `EPOCHS=5`), and writes baseline scores to
`runs/baseline/baseline_scores_val_640.csv`. Training artifacts land in
`runs/baseline/<model>_s640/` (git-ignored).

### Shared evaluator

Every experiment must be scored with the notebook's
`evaluate_split(labels, images, model, imgsz=..., conf=...)`, which returns:
mAP50–95, mAP50 (conf 0.001), recall@IoU0.5/conf0.25, tiny (<16² px) AP + recall,
COCO small/medium/large AP + recall with per-group object counts, and mean/p95
runtime per original image (batch 1, warm-up, CUDA sync). All metrics are computed
in original-image coordinates; ignored regions are respected. Do not introduce a
second evaluator — extend this one.

### Conventions

* Seed 42 everywhere (training, manifests, subsets).
* NMS IoU 0.7, eval conf 0.001 (AP) / 0.25 (recall), recall IoU 0.5 — fixed
  before results are examined.
* One shared training YAML (`data/visdrone.yaml`) for both models; the only
  difference between runs is model size and its pretrained weights.
* Test-dev is evaluated only for the final six experiments, never during
  development.
