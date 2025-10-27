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
BATCH=256
EPOCHS=100
WARMUP=20
FOLDS=(1 3)
LORA_RANKS=(4 8)

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p output/logs

run_job () {
  local exp="$1"; shift
  echo ">>> [GPU1] ${exp}"
  CUDA_VISIBLE_DEVICES=1 ${PY} ${MAIN} --exp_name "${exp}" "$@" 2>&1 | tee "output/logs/${exp}.log"
}

for FOLD in "${FOLDS[@]}"; do
  # 1) HEAD (linear)
  EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_HEAD_${STAMP}"
  run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
    --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" --debug

  # 2) FPN
  EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_FPN_${STAMP}"
  run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
    --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" --use_fpn --debug
done

for FOLD in "${FOLDS[@]}"; do
  # 3) LoRA (HEAD) r=4,8
  for r in "${LORA_RANKS[@]}"; do
    EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_LoRA_r${r}_${STAMP}"
    run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
      --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
      --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" \
      --use_lora --r "${r}"
  done

  # 4) FPN + LoRA r=4,8
  for r in "${LORA_RANKS[@]}"; do
    EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_FPN_LoRA_r${r}_${STAMP}"
    run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
      --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
      --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" \
      --use_fpn --use_lora --r "${r}"
  done
done

echo "=== GPU1 listo. Logs en output/logs ==="
