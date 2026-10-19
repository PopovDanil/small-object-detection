#!/usr/bin/env bash
# Full fine-tune on a 2-GPU Kaggle session (2x T4): yolo26m on GPU 0 and yolo26x on
# GPU 1 in parallel, same shared config and the same batch for both runs.
#
#   bash scripts/kaggle_finetune.sh
#
# Steps: install deps -> prepare data -> pilot once (fixes the batch) -> two
# parallel runs of notebooks/finetune.ipynb. Executed notebooks + logs land in
# runs/finetune/. Set FT_BATCH to skip the pilot, FT_WORKERS to change workers.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -q "ultralytics==8.4.175" "pycocotools==2.0.11"

# 1. data (download + YOLO conversion), skipped parts are cached on re-runs
python scripts/prepare_data.py

mkdir -p runs/finetune runs/pilot
run_notebook() {  # $1 = output name; env selects model/device/batch
  python -m jupyter nbconvert --to notebook --execute notebooks/finetune.ipynb \
    --ExecutePreprocessor.timeout=-1 --output-dir runs/finetune --output "$1"
}

# 2. pilot once on GPU 0 (yolo26x decides the batch), unless the batch is pinned
if [[ -z "${FT_BATCH:-}" ]]; then
  FT_MODELS="" FT_DEVICE=0 run_notebook finetune_pilot.ipynb > runs/pilot/pilot.log 2>&1
  FT_BATCH=$(python -c "import json; print(json.load(open('runs/pilot/pilot_result.json'))['effective_batch'])")
fi
echo "effective batch = ${FT_BATCH}"

# 3. both models in parallel, one per GPU; 4 Kaggle CPUs are shared by both runs
WORKERS="${FT_WORKERS:-2}"
FT_MODELS=yolo26m FT_DEVICE=0 FT_BATCH="$FT_BATCH" FT_WORKERS="$WORKERS" \
  run_notebook finetune_yolo26m.ipynb > runs/finetune/yolo26m.log 2>&1 &
PID_M=$!
FT_MODELS=yolo26x FT_DEVICE=1 FT_BATCH="$FT_BATCH" FT_WORKERS="$WORKERS" \
  run_notebook finetune_yolo26x.ipynb > runs/finetune/yolo26x.log 2>&1 &
PID_X=$!

# nbconvert only writes cell output at the end, so print the last epoch row of each
# run's results.csv every 10 minutes (visible in the Kaggle session log).
(
  while kill -0 "$PID_M" 2>/dev/null || kill -0 "$PID_X" 2>/dev/null; do
    sleep 600
    for model in yolo26m yolo26x; do
      csv="runs/finetune/${model}_s640/results.csv"
      [[ -f "$csv" ]] && echo "[$(date +%H:%M)] ${model}: $(tail -n 1 "$csv" | cut -c1-120)"
    done
  done
) &

STATUS=0
wait "$PID_M" || { echo "yolo26m FAILED (see runs/finetune/yolo26m.log)"; STATUS=1; }
wait "$PID_X" || { echo "yolo26x FAILED (see runs/finetune/yolo26x.log)"; STATUS=1; }
ls -la runs/finetune/*/weights/ 2>/dev/null || true
exit "$STATUS"
