#!/bin/bash
#SBATCH --job-name=roberta_abstention
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=output/evaluation_%j.out
#SBATCH --error=output/evaluation_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agolbaghi1@sheffield.ac.uk

set -euo pipefail

source ~/.bashrc
mamba activate abstention_rl

module load CUDA/12.1.1
module load GCC/12.3.0
module load CMake/3.26.3-GCCcore-12.3.0

cd /mnt/parscratch/users/ach21ag/private/roberta_classifier_finetuning_for-abstention

echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version:"
python --version

echo "Checking GPU..."
nvidia-smi

echo "Checking PyTorch CUDA..."
python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Starting eight-class training..."

python train_roberta_classifier.py \
  --data_dir eight_class \
  --output_dir roberta_eight_class \
  --model_name FacebookAI/roberta-base \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5

echo "Finished eight-class training."

echo "Starting seven-class training..."

python train_roberta_classifier.py \
  --data_dir seven_class \
  --output_dir roberta_seven_class \
  --model_name FacebookAI/roberta-base \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5

echo "Finished seven-class training."
