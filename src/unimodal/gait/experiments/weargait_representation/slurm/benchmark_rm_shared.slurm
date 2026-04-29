#!/bin/bash
#SBATCH --job-name=wg-repr-bench
#SBATCH --output=/ocean/projects/med260006p/aguda1/biomedAI/project/logs/slurm/%x-%j.out
#SBATCH --error=/ocean/projects/med260006p/aguda1/biomedAI/project/logs/slurm/%x-%j.err
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --chdir=/ocean/projects/med260006p/aguda1/biomedAI/project
#SBATCH --account=med260006p

set -euo pipefail

PROJECT_ROOT="/ocean/projects/med260006p/aguda1/biomedAI/project"
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

RUN_NAME="${RUN_NAME:-weargait_representation_benchmark_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="outputs/unimodal_gait/runs/${RUN_NAME}"

echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Run dir: ${RUN_DIR}"

echo "[INFO] Starting benchmark at $(date)"
python src/unimodal/gait/experiments/weargait_representation/run_experiment.py --run-dir "${RUN_DIR}"
echo "[INFO] Finished benchmark at $(date)"
