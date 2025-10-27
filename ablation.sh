#!/usr/bin/env bash
set -euo pipefail

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
FOLDS=(0 1 2 3 4)
LORA_RANKS=(4 8)

# Concurrency: 2 GPUs -> 2 procesos a la vez
MAX_JOBS=2
GPUS=(0 1)

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p output

# === Helper para lanzar un experimento en una GPU concreta ===
launch () {
  local gpu_id="$1"; shift
  local exp_name="$1"; shift
  echo ">>> [GPU ${gpu_id}] ${exp_name}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  ${PY} ${MAIN} \
    --exp_name "${exp_name}" \
    "$@"
}

# === Cola de comandos a ejecutar (se llenan en el orden solicitado) ===
declare -a CMDS_GPU
declare -a CMDS_ARGS
i=0

add_cmd () {
  local fold="$1"; shift
  local exp_suffix="$1"; shift
  local extra_args=("$@")

  local exp_name="dino_${DATASET}_fold${fold}_bs${BATCH}_${exp_suffix}_${STAMP}"

  CMDS_GPU[i]=$(( i % ${#GPUS[@]} ))     # asigna GPU 0/1 alternando
  CMDS_ARGS[i]="--dataset ${DATASET} --root ${ROOT} --split_json ${SPLIT} \
--size ${SIZE} --dino_type ${DINO} --batch_size ${BATCH} \
--epochs ${EPOCHS} --warmup_epochs ${WARMUP} --fold ${fold} ${extra_args[*]} \
--exp_name ${exp_name}"
  i=$((i + 1))
}

# ====== FASE 1: SIN LoRA (HEAD -> FPN) ======
for FOLD in "${FOLDS[@]}"; do
  add_cmd "${FOLD}" "HEAD"                   # head only
  add_cmd "${FOLD}" "FPN"  "--use_fpn"       # FPN sin LoRA
done

# ====== FASE 2: CON LoRA (HEAD r=4,8 -> FPN r=4,8) ======
for FOLD in "${FOLDS[@]}"; do
  for r in "${LORA_RANKS[@]}"; do
    add_cmd "${FOLD}" "LoRA_r${r}"               "--use_lora" "--r" "${r}"
  done
  for r in "${LORA_RANKS[@]}"; do
    add_cmd "${FOLD}" "FPN_LoRA_r${r}"           "--use_fpn" "--use_lora" "--r" "${r}"
  done
done

# === Ejecuta con hasta MAX_JOBS en paralelo ===
active_jobs=0
for idx in "${!CMDS_ARGS[@]}"; do
  gpu="${GPUS[ ${CMDS_GPU[$idx]} ]}"
  args_str="${CMDS_ARGS[$idx]}"

  # Lanza en background usando la función launch de ESTA shell
  eval "launch ${gpu} EXP ${args_str}" &
  ((active_jobs++))

  if (( active_jobs >= MAX_JOBS )); then
    wait -n
    ((active_jobs--))
  fi
done

wait
echo "=== Listo. Revisa 'output/' para logs, pesos y métricas. ==="
