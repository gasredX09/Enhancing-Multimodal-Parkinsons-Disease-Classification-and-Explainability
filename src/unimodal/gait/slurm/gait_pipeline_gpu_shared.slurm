#!/bin/bash
#SBATCH --job-name=gait-pipeline-gpu
#SBATCH --output=/ocean/projects/med260006p/shared/biomedAI/project/logs/slurm/%x-%j.out
#SBATCH --error=/ocean/projects/med260006p/shared/biomedAI/project/logs/slurm/%x-%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gpus=v100-32:1
#SBATCH --time=10:00:00
#SBATCH --chdir=/ocean/projects/med260006p/shared/biomedAI/project
#SBATCH --account=med260006p

set -euo pipefail

PROJECT_ROOT="/ocean/projects/med260006p/shared/biomedAI/project"
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm

CONDA_ENV_NAME="${CONDA_ENV:-chiu-lab}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/.conda/envs/$CONDA_ENV_NAME}"

if [[ -d "${CONDA_ENV_PREFIX}" ]]; then
  export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
  export CONDA_PREFIX="${CONDA_ENV_PREFIX}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "[ERROR] Conda environment not found: ${CONDA_ENV_PREFIX}"
  exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Python: $(command -v python)"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"
nvidia-smi || true

python src/unimodal/gait/gait_ensemble_orchestrator.py --tasks all --force
python src/unimodal/gait/ensemble_fusion.py --strategy all

echo "[INFO] Completed gait pipeline run at $(date)"
