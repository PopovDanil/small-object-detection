# Fine-tuning — full protocol (stage 2a)

How the two reported models are trained for the study
([docs/proposal.md](proposal.md) §"Models and training strategy"). The baseline
notebook ([notebooks/baseline.ipynb](../notebooks/baseline.ipynb)) ran a
deliberately short **5-epoch stage-1** fine-tune to validate the pipeline; this
stage runs the **real** training whose validation-selected checkpoints feed every
reported result ([docs/baseline_results.md](baseline_results.md),
[docs/tiling_results.md](tiling_results.md)).

Implemented in [`notebooks/finetune.ipynb`](../notebooks/finetune.ipynb).

## 1. What is trained

Exactly **two** models, both initialized from COCO-pretrained weights and
fine-tuned for the 10 VisDrone classes:

| Run | Initialization | Dataset | Image size | Purpose |
|---|---|---|---|---|
| yolo26m | `yolo26m.pt` (COCO, 80 classes) | VisDrone2019-DET train | 640 | medium model |
| yolo26x | `yolo26x.pt` (COCO, 80 classes) | VisDrone2019-DET train | 640 | high-capacity model |

No third model is trained. The resulting checkpoints are evaluated **unchanged**
at 640 full-image, 1280 full-image and tiling inference — there is **no
1280-resolution training stage**, because the study is about inference-time
resolution.

## 2. Fixed training protocol

All constants are fixed before any result is examined and are defined at the top
of [`notebooks/finetune.ipynb`](../notebooks/finetune.ipynb):

| Parameter | Value |
|---|---|
| Seed | 42 |
| Image size (`imgsz`) | 640 |
| Max epochs | 50 |
| Early stopping | `patience=15` on validation mAP50–95 |
| Best checkpoint | validation-selected (`weights/best.pt`) |
| Trainable layers | all (`freeze=None`), from epoch 1 |
| Optimizer | default YOLO (`optimizer="auto"`) |
| Base LR | `lr0=0.01`, scaled to effective batch via `nbs=64` |
| LR schedule | cosine (`cos_lr=True`), `warmup_epochs=3` |
| Weight decay | `5e-4` |
| Mosaic close | `close_mosaic=10` (final 10 epochs on clean geometry) |
| Augmentation | Ultralytics defaults (mosaic, HSV, horizontal flip), deterministic |
| Batch size | fixed once by the pilot (see §3), identical for both runs |
| Dataset config | one shared YAML (`data/visdrone.yaml`) for both runs |
| Validation | used only for checkpoint selection, never for reported scores |

Because `optimizer="auto"` is used, Ultralytics chooses AdamW (as in the stage-1
run) and derives the LR from `lr0`/`nbs`; the observed stage-1 LR at effective
batch 4 was `7.14e-4`, consistent with `0.01 × 4 / 64`. The effective batch size,
`nbs`, seed and the full augmentation schedule are written to
`runs/finetune/training_config.json` alongside the run.

## 3. Effective batch size (pilot)

The proposal fixes the batch size with a short pilot before training. The
notebook's `run_pilot()` trains the **memory-heaviest** model (`yolo26x`) for one
epoch on 2 % of the training set for each candidate batch (`[4, 8, 16]`), stops
at the first out-of-memory failure, and records peak GPU memory and wall time.
The largest batch that fits becomes the effective batch size, used unchanged for
both fine-tuning runs.

* Result: `runs/pilot/pilot_result.json`.
* Pinned into the run: `runs/finetune/training_config.json` (`effective_batch`,
  `nbs`).

Set `BATCH_OVERRIDE` in the notebook to skip the pilot and pin a known batch
(e.g. on a known Kaggle GPU).

## 4. How to run

### Kaggle (recommended; the proposal targets T4/P100-class sessions)

1. Prepare the dataset once with [notebooks/baseline.ipynb](../notebooks/baseline.ipynb)
   (or upload the prepared `data/` tree as a Kaggle Dataset).
2. Upload/clone the repo, install with `uv sync` (or `pip install -r` the runtime
   dependencies).
3. Set the data root and run:

```bash
export VISDRONE_DATA=/kaggle/input/<your-visdrone-dataset>/data
uv run jupyter nbconvert --to notebook --execute notebooks/finetune.ipynb \
    --inplace --ExecutePreprocessor.timeout=-1
```

### Locally

```bash
# from the repo root, after data/ is prepared and a GPU is available
uv run jupyter nbconvert --to notebook --execute notebooks/finetune.ipynb \
    --inplace --ExecutePreprocessor.timeout=-1
```

> Expect several hours for the full 50-epoch protocol on a single consumer GPU
> (~2.5 h for `yolo26m`, ~5 h for `yolo26x` on the 8 GB laptop used for the
> baseline, by extrapolation from the stage-1 run). On a Kaggle GPU this is
> substantially faster.

## 5. Artifacts produced

| Artifact | Path |
|---|---|
| Best checkpoint, yolo26m | `runs/finetune/yolo26m_s640/weights/best.pt` |
| Best checkpoint, yolo26x | `runs/finetune/yolo26x_s640/weights/best.pt` |
| Per-epoch training log | `runs/finetune/<model>_s640/results.csv` |
| Validation curves | `runs/finetune/finetune_curves.png` |
| Summary table | `runs/finetune/finetune_summary.csv` |
| Run record (batch, wall time) | `runs/finetune/<model>_s640/finetune_record.json` |
| Full shared config | `runs/finetune/training_config.json` |
| Pilot result | `runs/pilot/pilot_result.json` |

All `runs/` artifacts are git-ignored.

## 6. Evaluating the fine-tuned checkpoints

The fine-tuned models use the identity 10-class mapping (`class_map=None`).
Point the evaluation notebook at the new weights by editing its `WEIGHTS` dict:

```python
# notebooks/tiling.ipynb
WEIGHTS = {
    "yolo26m": ROOT / "runs" / "finetune" / "yolo26m_s640" / "weights" / "best.pt",
    "yolo26x": ROOT / "runs" / "finetune" / "yolo26x_s640" / "weights" / "best.pt",
}
```

or score directly with the shared evaluator
([`src/evaluation/metrics.py`](../src/evaluation/metrics.py)):

```python
from src.evaluation.metrics import evaluate_split, load_split_paths
from ultralytics import YOLO

images, labels = load_split_paths("test")  # test-dev only for reporting
model = YOLO("runs/finetune/yolo26m_s640/weights/best.pt")
evaluate_split(labels=labels, images=images, model=model, imgsz=640)  # 640 full-image
evaluate_split(labels=labels, images=images, model=model, imgsz=1280)  # 1280 full-image
evaluate_split(
    labels=labels, images=images, model=model, inference="tiled", tile=640, overlap=0.20
)  # tiling
```

## 7. Results

To be filled once the full 50-epoch runs complete. The notebook writes
`runs/finetune/finetune_summary.csv` with, per model, the number of epochs run,
the selected epoch, best validation mAP50–95/mAP50 and training wall time; the
final test-dev scores for all inference conditions are then produced by the
shared evaluator (see [docs/tiling_results.md](tiling_results.md) for the tiling
tables, which are re-runnable against these checkpoints).

## 8. Reproducibility notes

* Seed 42 is fixed for training, manifests and subsets.
* One shared config per run; the only difference between `yolo26m` and `yolo26x`
  is model size and pretrained weights.
* `training_config.json` records the effective batch size next to every run, as
  required by the proposal.
* Validation is used only for checkpoint/hyperparameter selection; all reported
  scores are computed on test-dev with the single shared evaluator.
