#!/bin/bash
#SBATCH -J openllama-base-eval
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

python evaluate.py \
  --model "LLaMA-7B" \
  --adapter "Base" \
  --baseline \
  --dataset "AddSub" \
  --base_model "openlm-research/open_llama_3b_v2"

python evaluate.py \
  --model "LLaMA-7B" \
  --adapter "Base" \
  --baseline \
  --dataset "AQuA" \
  --base_model "openlm-research/open_llama_3b_v2"

echo "Recent evaluation summary:"
tail -n 4 "${REPO_DIR}/experiment/eval_summary.tsv"
