#!/usr/bin/env bash
#SBATCH --account=bcastane_lab
#SBATCH --partition=kuelap
#SBATCH --time=00:10:00
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

echo "GPUs: $SLURM_GPUS | CPUs/GPU: $SLURM_CPUS_PER_TASK | Mem: $SLURM_MEM_PER_NODE"
nvidia-smi
free -h

srun --mem=64G python main.py --exp_name test_dino --dataset liver --root /scratch/bcastane_lab/eochoaal --split_json ./data/splits_final.json --size large --dino_type dinov3 --batch_size 256 --epochs 10 --warmup_epochs 1 --fold 0 --debug --n_workers 18