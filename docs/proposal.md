# Small-Object Detection in Aerial Images

## Motivation and project question

Aerial and drone imagery contains many objects that occupy only a few dozen pixels after standard resizing, and detectors typically lose these objects or localize them poorly. Our project asks: **does tiled or higher-resolution inference improve small-object detection on aerial images compared with standard resizing, and what is the accuracy–runtime trade-off?** As a secondary question, we compare a medium and a high-capacity detector under identical inference conditions to see whether added model capacity helps small objects as much as added input resolution does.

Our working hypotheses are that both tiling and 1280-pixel inference raise small-object recall relative to a 640-pixel baseline, that tiling gains the most on the smallest objects, and that each gain comes at a measurable runtime cost that we quantify rather than assume.

## Project scope

The objective is to measure whether inference-time resolution — tiled inference or 1280-pixel full-image inference — improves small-object detection on aerial images over a standard 640-pixel baseline, and to quantify the accuracy–runtime trade-off; secondarily, to compare a medium and a high-capacity detector under identical inference conditions.

- **Task:** single-stage object detection (bounding box + class label) on individual drone images, scored in original-image coordinates with ignored regions respected.
- **Classes:** all 10 VisDrone detection classes — pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.
- **Models:** exactly two are fine-tuned (two training runs total) — YOLO26m (medium) and YOLO26x (highest reported COCO AP of the released YOLO26 detection sizes), both from COCO-pretrained weights, for the 10 VisDrone classes; no other model is trained.
- **Experimental settings:** two models × three inference conditions (640 baseline, 1280 full-image, 640×640 tiling with 20% overlap at native resolution) = six main experiments, plus one sensitivity sweep over tile overlap (0%, 20%, 40%) for YOLO26m. All weights are trained at `imgsz=640` and reused unchanged at both inference resolutions; NMS/IoU settings are fixed before any results are examined.

## Dataset preparation

We will use **VisDrone2019-DET** with all 10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor. The task is to predict object bounding boxes and class labels in individual drone images. Official split boundaries will be preserved, original annotations will be retained, and ignored regions will be respected during evaluation.

| Split | Images used | Purpose |
|---|---:|---|
| Train | All 6,471 official training images | Fine-tuning |
| Validation | All 548 official validation images | Checkpoint and parameter selection |
| Test | All 1,610 annotated official test-dev images | Final evaluation only |

Using the complete training split avoids discarding half of the available data and removes any need to justify a subset. Should inference on the full test split prove too expensive during the pilot, we will evaluate on a 500-image subset selected once with seed 42; in that case exactly the same saved test manifest will be used for all six experiments. All subsets and manifests will be generated with seed 42 and saved for reproducibility. Before training we will check class coverage and duplicate images. Every tile will stay with its source image.

**Preprocessing.** Official VisDrone annotations (`.txt` in VisDrone format) will be converted once to YOLO format for training, mapping the 10 classes to ids 0–9; `ignored regions` and `don't care` areas are dropped from training targets and handled by the evaluator instead. Images are stored at native resolution and are never resampled on disk: letterbox padding and resizing happen on the fly, to 640 during training and to 640 or 1280 at inference. Tiling slices native-resolution images into 640 × 640 crops with 20% overlap, clips boxes to each crop, and maps predictions back to original-image coordinates before merging. Training augmentation follows the Ultralytics defaults with a fixed seed, and the identical configuration file (class names, image size, augmentation, and hyperparameters) is reused for both models so that only the weights differ.

Dataset and downloads: [official VisDrone repository](https://github.com/VisDrone/VisDrone-Dataset#download); [dataset configuration](https://docs.ultralytics.com/datasets/detect/visdrone/).

## Experimental plan

### Models and training strategy

We will fine-tune **exactly two models — YOLO26m (medium) and YOLO26x (highest reported COCO AP among the released YOLO26 detection sizes)** — both initialized from COCO-pretrained weights, for the 10 VisDrone classes. This means two fine-tuning runs in total (one per model); no third model is trained, and every inference condition reuses one of these two checkpoints. This is a practical model selection, not a claim of best performance on VisDrone. [Model documentation](https://docs.ultralytics.com/models/yolo26/)

Both models will be fine-tuned for a maximum of 50 epochs with early stopping (fixed `patience=15`) based on validation mAP50–95, using identical data and training protocol and validation-selected checkpoints. We will run a short Kaggle pilot to check GPU memory and throughput, then fix the batch settings and record the effective batch size.

**Fine-tuning strategy (fixed for both models).** Because COCO→VisDrone is a large label shift, we use a single-stage full fine-tune rather than staged freezing:

- **All layers trainable from epoch 1** — no backbone freezing, no discriminative learning rates; the detection head is re-initialized for the 10 VisDrone classes while all other weights start from COCO.
- **Optimizer and schedule:** default YOLO optimizer with `lr0≈0.01` scaled to the effective batch (`norm_batch=64`), `warmup_epochs=3`, cosine LR decay (`cos_lr=True`), `weight_decay=5e-4`.
- **Stopping rule:** `max_epochs=50`, `patience=15` on validation mAP50–95, best checkpoint kept by validation selection; seed fixed so runs are comparable.
- **Augmentation:** Ultralytics defaults (mosaic, HSV, horizontal flip), with `close_mosaic=10` so the final 10 epochs train on clean geometry. Mosaic is kept deliberately: it injects small objects into scenes, which suits our goal.
- **Fairness constraint:** one shared training YAML — class names, image size, augmentation, epochs, LR schedule, seed — is used for both runs, so the only difference between the two checkpoints is model size and its pretrained weights. Effective batch size is fixed once by the pilot and recorded with every result.
- **Explicitly out of scope:** a second 1280-resolution training stage. It would confound the study, since our claim is about *inference-time* resolution; all weights come from 640 training only.

**Both YOLO26m and YOLO26x are trained with `imgsz=640`. The resulting checkpoints are evaluated at both 640 and 1280 inference resolutions without retraining.** This is what makes the experiment a test of inference-time resolution rather than of training resolution.

### Inference conditions

Each selected checkpoint will be evaluated under three inference conditions, yielding six main experiments. For each model, the same selected checkpoint will be used across all three inference modes, and weights will remain fixed across inference conditions.

1. **Baseline:** full-image resizing/letterboxing at inference size 640.
2. **Higher resolution:** full-image inference at inference size 1280.
3. **Tiling:** 640 × 640 crops at native resolution with 20% overlap. Predictions are transformed to original-image coordinates and merged using documented NMS/IoU settings that will be fixed before any results are examined and held identical across all experiments.

All conditions use the same detector branch and evaluation settings. We will compare inference methods within each model and model sizes within each inference method. One additional sensitivity experiment will vary tile overlap at 0%, 20%, and 40% for YOLO26m, with every other factor fixed.

### Evaluation metrics

- **mAP50–95 and mAP50:** localization and classification quality, computed with a low prediction threshold (initially 0.001).
- **Recall:** IoU = 0.5 and confidence = 0.25, fixed across all conditions.
- **Tiny-object performance (primary result):** AP50–95 and recall for tiny boxes with original-image area below 16² pixels, reported together. Recall alone only shows whether objects were found; AP shows whether their localization and classification were correct.
- **Small/medium/large objects:** AP50–95 and recall per COCO size group (small < 32² px, medium 32²–96² px, large > 96² px), computed in original-image coordinates, together with the object count in each group.
- **Runtime:** mean and p95 milliseconds per original image, including preprocessing, all tiles, and merging; same GPU, batch size 1, with warm-up and GPU synchronization. Training time and the runtime environment will also be reported.
- **Single evaluator:** one shared evaluator will score all six conditions, respecting ignored regions and documenting detection limits. Results apply to our protocol and are not official leaderboard scores.
- **Failure analysis:** at least three observed failure cases will be discussed, covering tiny objects, occlusion/crowding, and tile boundaries, including any regressions introduced by tiling.

## Implementation plan

Pipeline steps, in order:

1. **Data setup (Member 1):** download VisDrone2019-DET, convert annotations to YOLO format, generate seed-42 manifests for train/val/test (and the optional 500-image test subset), verify class coverage and duplicates, and write environment plus run instructions.
2. **Training (Member 2):** run a short Kaggle pilot to fix batch size and record effective batch size, then fine-tune YOLO26m and YOLO26x with the fixed strategy above (`imgsz=640`, all layers trainable, max 50 epochs, `patience=15` on validation mAP50–95, shared YAML and seed); save logs and the shared training configuration.
3. **Inference harness (Member 3):** implement the three modes — 640 full-image, 1280 full-image, and native-resolution tiling with overlap — including coordinate mapping, NMS/IoU merging, and a timing loop with warm-up, batch size 1, and GPU synchronization; then run the 0/20/40% overlap sweep.
4. **Evaluation (Member 4):** build one shared evaluator that computes mAP50–95, mAP50, recall, tiny-object (<16² px) and COCO size-group metrics with per-group object counts, respecting ignored regions; emit a single metric table and plots for all six conditions.
5. **Analysis and delivery (all):** run failure-case analysis (tiny objects, occlusion/crowding, tile boundaries), assemble the two-page summary, demo, and reproducible command list, and prepare for individual Q&A.

### Main libraries and resources

All package versions are pinned in a `requirements.txt`, and every run starts from the same seed-42 configuration.

| Library / resource | Role in the pipeline |
|---|---|
| **Ultralytics (YOLO26)** | Fine-tuning, checkpoint selection, dataset/annotation plumbing, and the 640/1280 full-image inference paths; fetches the COCO-pretrained `yolo26m.pt` and `yolo26x.pt` weights. |
| **PyTorch (CUDA backend)** | Training and inference execution, mixed-precision flags, and explicit CUDA synchronization around timed regions. |
| **OpenCV, Pillow** | Image I/O, native-resolution tiling into 640×640 crops, and coordinate mapping of boxes back to original images. |
| **NumPy, Pandas** | Prediction merging, per-condition metric tables, manifests, and CSV exports. |
| **pycocotools** | COCO-style mAP50–95/mAP50 computation on converted annotations, extended with our tiny (<16² px) and small/medium/large grouping. |
| **Matplotlib, Seaborn** | Accuracy–runtime trade-off plots and failure-case figures. |
| **Git, `requirements.txt`, shared drive** | Reproducibility: versioned configs, seed-42 manifests, checkpoints, per-condition prediction files, and timing logs in a documented directory layout. |

**Compute:** Kaggle GPU sessions (T4/P100 class) for the pilot and both fine-tuning runs; one fixed local GPU for all timed inference and evaluation, with GPU model, driver, precision mode, and batch size recorded alongside every result. Any result must be regenerable from a single documented command.

## Team responsibilities

| Member | Main responsibility | Concrete output |
|---|---|---|
| Member 1 | Dataset and reproducibility | Download/conversion, split manifests, annotation checks, environment and run instructions |
| Member 2 | Model training | Kaggle training for both models, checkpoints, logs and shared training configuration |
| Member 3 | Inference and timing | Three inference modes, tile merging, overlap sensitivity and runtime measurements |
| Member 4 | Evaluation and communication | Shared evaluator, metric table, failure analysis, two-page summary and demo assembly |

All members review the metrics and failure cases, contribute to the report and demo, and prepare for individual Q&A.
