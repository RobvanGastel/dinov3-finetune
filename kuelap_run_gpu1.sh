#!/usr/bin/env bash
#SBATCH --account=bcastane_lab
#SBATCH --partition=kuelap
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x-%j.out


set -eo pipefail

# ---- Activar conda (deshabilitar -u temporalmente) ----
set +u
source /software/anaconda3/5.3.0b/etc/profile.d/conda.sh
conda activate castane_lab
set -u
# -------------------------------------------------------

# Variables de entorno (protegidas si están vacías)
: "${LD_LIBRARY_PATH:=}"
export LD_LIBRARY_PATH="/scratch/bcastane_lab/lab-conda/envs/castane_lab/lib:${LD_LIBRARY_PATH}"
export CUDA_HOME="/scratch/bcastane_lab/lab-conda/envs/castane_lab"

# === Config base ===
PY=python
MAIN=main.py
DATASET=liver
ROOT=/scratch/bcastane_lab/eochoaal
SPLIT=./data/splits_final.json
SIZE=large
DINO=dinov3
BATCH=32
EPOCHS=100
WARMUP=20
N_WORKERS=18
FOLDS=(1 3)
LORA_RANKS=(4 8)

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p output/logs logs

run_job () {
  local exp="$1"; shift
  echo ">>> [$SLURM_JOB_ID] ${exp}"
  # Slurm 24.05 tip: pass --mem to srun to avoid env conflicts
  srun --exclusive --mem=64G ${PY} ${MAIN} --exp_name "${exp}" "$@" \
      2>&1 | tee "output/logs/${exp}.log"
}

for FOLD in "${FOLDS[@]}"; do
  # 1) HEAD (linear)
  EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_HEAD_${STAMP}"
  run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
    --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" --debug \
    --n_workers "${N_WORKERS}" --wandb

  # 2) FPN
  EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_FPN_${STAMP}"
  run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
    --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" --use_fpn --debug \
    --n_workers "${N_WORKERS}" --wandb
done

for FOLD in "${FOLDS[@]}"; do
  # 3) LoRA (HEAD) r=4,8
  for r in "${LORA_RANKS[@]}"; do
    EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_LoRA_r${r}_${STAMP}"
    run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
      --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
      --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" \
      --use_lora --r "${r}" --n_workers "${N_WORKERS}" --debug --wandb
  done

  # 4) FPN + LoRA r=4,8
  for r in "${LORA_RANKS[@]}"; do
    EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_FPN_LoRA_r${r}_${STAMP}"
    run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
      --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
      --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" \
      --use_fpn --use_lora --r "${r}" --n_workers "${N_WORKERS}" --debug --wandb
  done
done

echo "=== GPU1 listo. Logs en output/logs y logs/%x-%j.out ==="
