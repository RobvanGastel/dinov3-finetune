#!/usr/bin/env bash
#SBATCH --account=bcastane_lab
#SBATCH --partition=kuelap
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --hint=nomultithread
#SBATCH --output=logs/compile_ops-%j.out

set -euo pipefail

CONDA_SH="/software/anaconda3/5.3.0b/etc/profile.d/conda.sh"
CONDA_ENV="castane_lab"
OPS_DIR="/scratch/bcastane_lab/eochoaal/dinov3-finetune/dinov3/eval/segmentation/models/utils/ops"

export MAX_JOBS="${SLURM_CPUS_PER_TASK:-18}"
# export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

cd "$OPS_DIR"

# Puedes compilar directo o vía srun (equivalen aquí)
python setup.py build install
# srun --exclusive python setup.py build install

echo ">> Compilación/instalación completada."
