#!/bin/bash
#SBATCH --job-name=hw-bench
#SBATCH --output=/ocean/projects/med260006p/shared/biomedAI/project/logs/slurm/%x-%j.out
#SBATCH --error=/ocean/projects/med260006p/shared/biomedAI/project/logs/slurm/%x-%j.err
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2000M
#SBATCH --chdir=/ocean/projects/med260006p/shared/biomedAI/project
#SBATCH --account=cis260039p

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
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -c "import sys; print('[INFO] Python executable:', sys.executable)"

python src/unimodal/handwriting/benchmark_handwriting_models.py \
  --n-splits 5 \
  --embedding-dim 4 \
  --primary-metric roc_auc

echo "[INFO] Handwriting benchmark completed at $(date)"
