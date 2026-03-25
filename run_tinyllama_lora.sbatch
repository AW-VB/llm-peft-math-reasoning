#!/bin/bash
#SBATCH -J tinyllama-lora
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 06:00:00
#SBATCH -o /home/msds/sirui005/SIRUI005/llm-peft-math-reasoning/logs/%x-%j.out
#SBATCH -e /home/msds/sirui005/SIRUI005/llm-peft-math-reasoning/logs/%x-%j.err

set -euo pipefail

REPO_DIR="/home/msds/sirui005/SIRUI005/llm-peft-math-reasoning"

mkdir -p "${REPO_DIR}/logs"
cd "${SLURM_SUBMIT_DIR:-${REPO_DIR}}"

source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate llm

export TOKENIZERS_PARALLELISM=false

python finetune.py \
  --base_model "TinyLlama/TinyLlama_v1.1" \
  --data_path "./ft-training_set/math_7k.json" \
  --output_dir "./trained_models/tinyllama-v1.1-lora-epoch1" \
  --adapter_name "lora" \
  --batch_size 4 \
  --micro_batch_size 4 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120
