#!/usr/bin/env bash
# Final evaluation of the fine-tuned checkpoints on test-dev (one GPU is enough):
# highres.ipynb (640 + 1280) -> tiling.ipynb (tiling 20 % + overlap sweep + failure
# cases) -> results.ipynb (final tables). Executed notebooks land in runs/notebooks/.
#
#   FINETUNE_SRC=/kaggle/input/<finetune-notebook-output>/small-object-detection/runs/finetune \
#     bash scripts/kaggle_eval.sh
#
# FINETUNE_SRC: folder holding yolo26m_s640/ and yolo26x_s640/ from the fine-tune run
#               (skip it if runs/finetune/ is already in place).
# EVAL_STAGE:   finetune (default) | baseline.  RUN_APPENDIX=0 skips the per-native-size
#               appendix of highres.ipynb (~30-60 min).
set -euo pipefail
cd "$(dirname "$0")/.."
export EVAL_STAGE="${EVAL_STAGE:-finetune}"

pip install -q "ultralytics==8.4.175" "pycocotools==2.0.11"
python scripts/prepare_data.py

# Kaggle: without FINETUNE_SRC, look for the fine-tune output attached as an input.
if [[ -z "${FINETUNE_SRC:-}" && ! -f "runs/${EVAL_STAGE}/yolo26m_s640/weights/best.pt" ]]; then
  found=$(find /kaggle/input -path "*/runs/${EVAL_STAGE}/yolo26m_s640/weights/best.pt" 2>/dev/null | head -n 1)
  [[ -n "$found" ]] && FINETUNE_SRC=$(dirname "$(dirname "$(dirname "$found")")")
  echo "FINETUNE_SRC=${FINETUNE_SRC:-<not found>}"
fi

if [[ -n "${FINETUNE_SRC:-}" ]]; then
  mkdir -p "runs/${EVAL_STAGE}"
  for model in yolo26m yolo26x; do
    mkdir -p "runs/${EVAL_STAGE}/${model}_s640/weights"
    cp "${FINETUNE_SRC}/${model}_s640/weights/best.pt" "runs/${EVAL_STAGE}/${model}_s640/weights/"
  done
fi
ls -la runs/"${EVAL_STAGE}"/*_s640/weights/best.pt

mkdir -p runs/notebooks
for notebook in highres tiling results; do
  echo "[$(date +%H:%M)] running ${notebook}.ipynb (EVAL_STAGE=${EVAL_STAGE})"
  python -m jupyter nbconvert --to notebook --execute "notebooks/${notebook}.ipynb" \
    --ExecutePreprocessor.timeout=-1 --output-dir runs/notebooks \
    --output "${notebook}_${EVAL_STAGE}.ipynb"
done
echo "[$(date +%H:%M)] done: tables in runs/results/${EVAL_STAGE}/"
