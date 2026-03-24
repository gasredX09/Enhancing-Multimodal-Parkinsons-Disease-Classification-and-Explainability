#!/bin/bash
#SBATCH --job-name=gait-pipeline-cpu
#SBATCH --output=/ocean/projects/med260006p/shared/biomedAI/project/logs/slurm/%x-%j.out
#SBATCH --error=/ocean/projects/med260006p/shared/biomedAI/project/logs/slurm/%x-%j.err
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --chdir=/ocean/projects/med260006p/shared/biomedAI/project
#SBATCH --account=bio230007p

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

python src/unimodal/gait/gait_ensemble_orchestrator.py --tasks rf --force
python src/unimodal/gait/ensemble_fusion.py --strategy all

echo "[INFO] Completed CPU gait pipeline run at $(date)"