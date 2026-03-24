#!/bin/bash
#SBATCH --job-name=weargait-sep-gpu
#SBATCH --output=/ocean/projects/med260006p/aguda1/biomedAI/project/logs/slurm/%x-%j.out
#SBATCH --error=/ocean/projects/med260006p/aguda1/biomedAI/project/logs/slurm/%x-%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=v100-32:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/ocean/projects/med260006p/aguda1/biomedAI/project
#SBATCH --account=med260006p

set -euo pipefail

PROJECT_ROOT="/ocean/projects/med260006p/aguda1/biomedAI/project"
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm
mkdir -p outputs/unimodal_gait/weargait_dl_embeddings
mkdir -p outputs/unimodal_gait/weargait_concat_embeddings

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

FORCE_FLAG=""
if [[ "${FORCE:-1}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python: $(command -v python)"
echo "[INFO] Started: $(date)"

python - <<'PY'
import torch
print('[INFO] CUDA available:', torch.cuda.is_available())
print('[INFO] CUDA devices:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('[INFO] Device 0:', torch.cuda.get_device_name(0))
PY

nvidia-smi || true

python src/unimodal/gait/prepare_weargait_index.py
python src/unimodal/gait/gait_ensemble_orchestrator.py --tasks weargait ${FORCE_FLAG}
python src/unimodal/gait/concat_weargait_task_embeddings.py \
  --input-root outputs/unimodal_gait/weargait_dl_embeddings \
  --output-dir outputs/unimodal_gait/weargait_concat_embeddings \
  --tasks SelfPace HurriedPace TUG

echo "[INFO] Finished: $(date)"
