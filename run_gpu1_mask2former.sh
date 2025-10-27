#!/usr/bin/env bash
set -euo pipefail

# (Opcional) activar conda dentro del script
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate lightning || true
fi

CUDA_VISIBLE_DEVICES=1  :  # fija la GPU

# === Config base ===
PY=python
MAIN=main.py
DATASET=liver
ROOT=/home/exx/Documents/nnUNetFrame/dataset/nnUNet_raw
SPLIT=./data/splits_final.json
SIZE=large
DINO=dinov3
BATCH=8
EPOCHS=100
WARMUP=20
FOLDS=(3 4)

mkdir -p output/logs

run_job () {
  local exp="$1"; shift
  echo ">>> [GPU1] ${exp}"
  CUDA_VISIBLE_DEVICES=1 ${PY} ${MAIN} --exp_name "${exp}" "$@" 2>&1 | tee "output/logs/${exp}.log"
}

for FOLD in "${FOLDS[@]}"; do
  # 1) HEAD (linear)
  EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_mask2former"
  run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
    --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" --debug --wandb --use_mask2former \
    --warmup_epochs 10 --lr 1e-4 --weight_decay 0.1


done

echo "=== GPU1 listo. Logs en output/logs ==="

