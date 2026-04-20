#!/bin/bash
#SBATCH -J gsm8k-rerun-all
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 6:00:00
#SBATCH -o /home/msds/sirui005/SIRUI005/llm-peft-math-reasoning/logs/%x-%j.out
#SBATCH -e /home/msds/sirui005/SIRUI005/llm-peft-math-reasoning/logs/%x-%j.err
set -euo pipefail

REPO_DIR="/home/msds/sirui005/SIRUI005/llm-peft-math-reasoning"
OPENLLAMA_BASE_MODEL="openlm-research/open_llama_3b_v2"
TINYLLAMA_BASE_MODEL="TinyLlama/TinyLlama_v1.1"
DATASET_NAME="gsm8k"
EVAL_BATCH_SIZE=8
SUMMARY_FILE="${REPO_DIR}/experiment/gsm8k_rerun_openllama_tinyllama_all.tsv"

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/experiment"
cd "${REPO_DIR}"

set +u
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh
conda activate llm
set -u

DATASET_TOTAL=$(python - <<PY
import json
with open("${REPO_DIR}/dataset/${DATASET_NAME}/test.json", "r") as f:
    print(len(json.load(f)))
PY
)

printf 'run_name\tmodel\tadapter\tepoch\ttotal\tcorrect\taccuracy\telapsed_seconds\tresult_json\n' > "${SUMMARY_FILE}"
JOB_START_TIME=$(date +%s)

append_summary_row() {
  local run_name="$1"
  local model_name="$2"
  local adapter_name="$3"
  local epoch_label="$4"
  local result_json="$5"
  local elapsed_seconds="$6"
  python - <<PY
import json
run_name = ${run_name@Q}
model_name = ${model_name@Q}
adapter_name = ${adapter_name@Q}
epoch_label = ${epoch_label@Q}
result_json = ${result_json@Q}
summary_file = ${SUMMARY_FILE@Q}
elapsed_seconds = ${elapsed_seconds}
with open(result_json, "r") as f:
    data = json.load(f)
total = len(data)
correct = sum(1 for item in data if item.get("flag"))
accuracy = correct / total if total else 0.0
with open(summary_file, "a") as f:
    f.write(
        f"{run_name}\t{model_name}\t{adapter_name}\t{epoch_label}\t"
        f"{total}\t{correct}\t{accuracy:.6f}\t{elapsed_seconds}\t{result_json}\n"
    )
print(
    f"SUMMARY_ROW\t{run_name}\t{model_name}\t{adapter_name}\t{epoch_label}\t"
    f"{total}\t{correct}\t{accuracy:.6f}\t{elapsed_seconds}\t{result_json}"
)
PY
}

result_count() {
  local result_json="$1"
  python - <<PY
import json
path = ${result_json@Q}
try:
    with open(path, "r") as f:
        data = json.load(f)
    print(len(data))
except FileNotFoundError:
    print(0)
except Exception:
    print(-1)
PY
}

run_eval() {
  local run_name="$1"
  local model_name="$2"
  local adapter_name="$3"
  local epoch_label="$4"
  local result_json="$5"
  shift 5
  local start_time
  local end_time
  local elapsed_seconds
  local existing_count

  echo "============================================================"
  echo "[$(date '+%F %T')] Starting ${run_name}"
  echo "============================================================"

  existing_count=$(result_count "${result_json}")
  if [[ "${existing_count}" == "${DATASET_TOTAL}" ]]; then
    echo "Existing completed result found for ${run_name}: ${existing_count}/${DATASET_TOTAL}. Skipping evaluation."
    append_summary_row "${run_name}" "${model_name}" "${adapter_name}" "${epoch_label}" "${result_json}" "0"
    echo "============================================================"
    echo "[$(date '+%F %T')] Finished ${run_name} (skipped, already complete)"
    echo "============================================================"
    return
  fi

  if [[ "${existing_count}" =~ ^[0-9]+$ ]] && [[ "${existing_count}" -gt 0 ]]; then
    echo "Resuming ${run_name} from ${existing_count}/${DATASET_TOTAL} saved samples."
  elif [[ "${existing_count}" == "-1" ]]; then
    echo "Existing result file for ${run_name} is unreadable JSON; evaluate.py will overwrite or fail accordingly."
  fi

  start_time=$(date +%s)
  python "${REPO_DIR}/evaluate.py" --resume "$@"
  end_time=$(date +%s)
  elapsed_seconds=$((end_time - start_time))
  append_summary_row "${run_name}" "${model_name}" "${adapter_name}" "${epoch_label}" "${result_json}" "${elapsed_seconds}"
  echo "============================================================"
  echo "[$(date '+%F %T')] Finished ${run_name} (elapsed: ${elapsed_seconds}s)"
  echo "============================================================"
}

run_eval \
  "openllama-base-gsm8k" \
  "OpenLlama" \
  "Base" \
  "baseline" \
  "${REPO_DIR}/experiment/LLaMA-7B-Base-gsm8k-rerun-openllama-base.json" \
  --model "LLaMA-7B" \
  --adapter "Base" \
  --dataset "${DATASET_NAME}" \
  --base_model "${OPENLLAMA_BASE_MODEL}" \
  --baseline \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-openllama-base"

run_eval \
  "openllama-lora-epoch1-gsm8k" \
  "OpenLlama" \
  "LoRA" \
  "1" \
  "${REPO_DIR}/experiment/LLaMA-7B-LoRA-gsm8k-rerun-openllama-lora-epoch1.json" \
  --model "LLaMA-7B" \
  --adapter "LoRA" \
  --dataset "${DATASET_NAME}" \
  --base_model "${OPENLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/openllama-3b-lora-epoch1" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-openllama-lora-epoch1"

run_eval \
  "openllama-lora-epoch2-gsm8k" \
  "OpenLlama" \
  "LoRA" \
  "2" \
  "${REPO_DIR}/experiment/LLaMA-7B-LoRA-gsm8k-rerun-openllama-lora-epoch2.json" \
  --model "LLaMA-7B" \
  --adapter "LoRA" \
  --dataset "${DATASET_NAME}" \
  --base_model "${OPENLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/openllama-3b-lora-epoch2" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-openllama-lora-epoch2"

run_eval \
  "openllama-lora-epoch3-gsm8k" \
  "OpenLlama" \
  "LoRA" \
  "3" \
  "${REPO_DIR}/experiment/LLaMA-7B-LoRA-gsm8k-rerun-openllama-lora-epoch3.json" \
  --model "LLaMA-7B" \
  --adapter "LoRA" \
  --dataset "${DATASET_NAME}" \
  --base_model "${OPENLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/openllama-3b-lora-epoch3" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-openllama-lora-epoch3"

run_eval \
  "openllama-adapterp-epoch1-gsm8k" \
  "OpenLlama" \
  "AdapterP" \
  "1" \
  "${REPO_DIR}/experiment/LLaMA-7B-AdapterP-gsm8k-rerun-openllama-adapterp-epoch1.json" \
  --model "LLaMA-7B" \
  --adapter "AdapterP" \
  --dataset "${DATASET_NAME}" \
  --base_model "${OPENLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/openllama-3b-adapterp-epoch1" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-openllama-adapterp-epoch1"

run_eval \
  "openllama-adapterp-epoch2-gsm8k" \
  "OpenLlama" \
  "AdapterP" \
  "2" \
  "${REPO_DIR}/experiment/LLaMA-7B-AdapterP-gsm8k-rerun-openllama-adapterp-epoch2.json" \
  --model "LLaMA-7B" \
  --adapter "AdapterP" \
  --dataset "${DATASET_NAME}" \
  --base_model "${OPENLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/openllama-3b-adapterp-epoch2" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-openllama-adapterp-epoch2"

run_eval \
  "openllama-adapterp-epoch3-gsm8k" \
  "OpenLlama" \
  "AdapterP" \
  "3" \
  "${REPO_DIR}/experiment/LLaMA-7B-AdapterP-gsm8k-rerun-openllama-adapterp-epoch3.json" \
  --model "LLaMA-7B" \
  --adapter "AdapterP" \
  --dataset "${DATASET_NAME}" \
  --base_model "${OPENLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/openllama-3b-adapterp-epoch3" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-openllama-adapterp-epoch3"

run_eval \
  "tinyllama-base-gsm8k" \
  "TinyLlama" \
  "Base" \
  "baseline" \
  "${REPO_DIR}/experiment/TinyLlama-Base-gsm8k-rerun-tinyllama-base.json" \
  --model "TinyLlama" \
  --adapter "Base" \
  --dataset "${DATASET_NAME}" \
  --base_model "${TINYLLAMA_BASE_MODEL}" \
  --baseline \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-tinyllama-base"

run_eval \
  "tinyllama-lora-epoch1-gsm8k" \
  "TinyLlama" \
  "LoRA" \
  "1" \
  "${REPO_DIR}/experiment/TinyLlama-LoRA-gsm8k-rerun-tinyllama-lora-epoch1.json" \
  --model "TinyLlama" \
  --adapter "LoRA" \
  --dataset "${DATASET_NAME}" \
  --base_model "${TINYLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/tinyllama-v1.1-lora-epoch1" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-tinyllama-lora-epoch1"

run_eval \
  "tinyllama-lora-epoch2-gsm8k" \
  "TinyLlama" \
  "LoRA" \
  "2" \
  "${REPO_DIR}/experiment/TinyLlama-LoRA-gsm8k-rerun-tinyllama-lora-epoch2.json" \
  --model "TinyLlama" \
  --adapter "LoRA" \
  --dataset "${DATASET_NAME}" \
  --base_model "${TINYLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/tinyllama-v1.1-lora-epoch2" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-tinyllama-lora-epoch2"

run_eval \
  "tinyllama-lora-epoch3-gsm8k" \
  "TinyLlama" \
  "LoRA" \
  "3" \
  "${REPO_DIR}/experiment/TinyLlama-LoRA-gsm8k-rerun-tinyllama-lora-epoch3.json" \
  --model "TinyLlama" \
  --adapter "LoRA" \
  --dataset "${DATASET_NAME}" \
  --base_model "${TINYLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/tinyllama-v1.1-lora-epoch3" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-tinyllama-lora-epoch3"

run_eval \
  "tinyllama-adapterp-epoch1-gsm8k" \
  "TinyLlama" \
  "AdapterP" \
  "1" \
  "${REPO_DIR}/experiment/TinyLlama-AdapterP-gsm8k-rerun-tinyllama-adapterp-epoch1.json" \
  --model "TinyLlama" \
  --adapter "AdapterP" \
  --dataset "${DATASET_NAME}" \
  --base_model "${TINYLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/tinyllama-v1.1-adapterp-epoch1" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-tinyllama-adapterp-epoch1"

run_eval \
  "tinyllama-adapterp-epoch2-gsm8k" \
  "TinyLlama" \
  "AdapterP" \
  "2" \
  "${REPO_DIR}/experiment/TinyLlama-AdapterP-gsm8k-rerun-tinyllama-adapterp-epoch2.json" \
  --model "TinyLlama" \
  --adapter "AdapterP" \
  --dataset "${DATASET_NAME}" \
  --base_model "${TINYLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/tinyllama-v1.1-adapterp-epoch2" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-tinyllama-adapterp-epoch2"

run_eval \
  "tinyllama-adapterp-epoch3-gsm8k" \
  "TinyLlama" \
  "AdapterP" \
  "3" \
  "${REPO_DIR}/experiment/TinyLlama-AdapterP-gsm8k-rerun-tinyllama-adapterp-epoch3.json" \
  --model "TinyLlama" \
  --adapter "AdapterP" \
  --dataset "${DATASET_NAME}" \
  --base_model "${TINYLLAMA_BASE_MODEL}" \
  --lora_weights "${REPO_DIR}/trained_models/tinyllama-v1.1-adapterp-epoch3" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --output_tag "rerun-tinyllama-adapterp-epoch3"

echo "Accuracy summary file: ${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"

JOB_END_TIME=$(date +%s)
echo "Total job elapsed seconds: $((JOB_END_TIME - JOB_START_TIME))"
