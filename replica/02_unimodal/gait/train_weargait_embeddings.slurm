#!/bin/bash
#SBATCH --job-name=weargait_emb
#SBATCH --output=logs/train_weargait_%j.out
#SBATCH --error=logs/train_weargait_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-shared
#SBATCH --time=08:00:00
#SBATCH --mem=20GB
#SBATCH --account=med260006p

# Print job info
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# Load modules
module load anaconda3

# Activate conda environment
source activate chiu-lab

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Print Python/PyTorch info
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to script directory
cd /ocean/projects/med260006p/shared/biomedAI/project/replica/02_unimodal/gait

# Optional: rebuild index to guarantee latest file manifest
python prepare_weargait_index.py

# Run training
echo "Starting WearGait embedding training..."
python train_weargait_embeddings.py

# Print completion info
echo ""
echo "Job finished at: $(date)"
