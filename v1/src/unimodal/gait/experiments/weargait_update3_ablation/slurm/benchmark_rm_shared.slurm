#!/bin/bash
#SBATCH -J wg-u3-ablate
#SBATCH -p RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH --account=med260006p
#SBATCH -o /ocean/projects/med260006p/aguda1/biomedAI/project/logs/slurm/%x-%j.out
#SBATCH -e /ocean/projects/med260006p/aguda1/biomedAI/project/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/ocean/projects/med260006p/aguda1/biomedAI/project
RUN_NAME=${RUN_NAME:-weargait_update3_ablation_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${PROJECT_ROOT}/outputs/unimodal_gait/runs/${RUN_NAME}

mkdir -p ${PROJECT_ROOT}/logs/slurm

cd /ocean/projects/med260006p/aguda1/biomedAI

echo "Starting WearGait Update-3 gait ablation"
echo "Run dir: ${RUN_DIR}"
date

python ${PROJECT_ROOT}/src/unimodal/gait/experiments/weargait_update3_ablation/run_experiment.py \
  --run-dir ${RUN_DIR}

date
echo "Finished WearGait Update-3 gait ablation"
