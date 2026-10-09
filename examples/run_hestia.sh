#!/usr/bin/env bash
# End-to-end HESTIA example: data -> Hessian calibration -> QAT -> evaluation.
#
#   bash examples/run_hestia.sh
#
# Every variable below can be overridden from the environment, e.g.
#   MODEL=/path/to/Llama-3.2-1B NUM_GPUS=4 MAX_STEPS=200 bash examples/run_hestia.sh
# A short budget is used by default so that the example finishes quickly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL=${MODEL:-meta-llama/Llama-3.2-1B}
DATASET=${DATASET:-openbmb/Ultra-FineWeb}
DATASET_SPLIT=${DATASET_SPLIT:-en}
TEXT_FIELD=${TEXT_FIELD:-content}
WORK_DIR=${WORK_DIR:-${ROOT}/outputs/example}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
SEQ_LEN=${SEQ_LEN:-1024}
MAX_STEPS=${MAX_STEPS:-100}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
EVAL_LIMIT=${EVAL_LIMIT:-200}

DATA_DIR="${WORK_DIR}/data"
TRACES="${WORK_DIR}/hessian_traces.json"
OUTPUT_DIR="${WORK_DIR}/checkpoint"
GRAD_ACCUM=$(( GLOBAL_BATCH_SIZE / (NUM_GPUS * PER_DEVICE_BATCH_SIZE) ))
GRAD_ACCUM=$(( GRAD_ACCUM > 0 ? GRAD_ACCUM : 1 ))
mkdir -p "${WORK_DIR}"

# 1) Tokenize and pack just enough sequences for this run.
if [[ ! -d "${DATA_DIR}" ]]; then
    python "${ROOT}/scripts/prepare_data.py" \
        --tokenizer "${MODEL}" \
        --dataset "${DATASET}" --split "${DATASET_SPLIT}" --text-field "${TEXT_FIELD}" \
        --seq-len "${SEQ_LEN}" \
        --max-sequences $(( MAX_STEPS * GLOBAL_BATCH_SIZE )) \
        --output-dir "${DATA_DIR}"
fi

# 2) One-time offline Hutch++ estimation of tensor-wise Hessian traces.
if [[ ! -f "${TRACES}" ]]; then
    torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${ROOT}/scripts/calibrate.py" \
        --model-path "${MODEL}" \
        --dataset-path "${DATA_DIR}" \
        --output "${TRACES}"
fi

# 3) Quantization-aware training with Hessian-guided annealing (ternary, group size 128).
torchrun --standalone --nproc_per_node "${NUM_GPUS}" "${ROOT}/scripts/train.py" \
    --model_name_or_path "${MODEL}" \
    --dataset_path "${DATA_DIR}" \
    --sensitivity_path "${TRACES}" \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed "${ROOT}/examples/ds_zero2.json" \
    --bf16 \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --lr_scheduler_type warmup_stable_decay \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none

# 4) Evaluate the hard-quantized ternary model.
python "${ROOT}/scripts/evaluate.py" \
    --model-path "${OUTPUT_DIR}" \
    --limit "${EVAL_LIMIT}"
