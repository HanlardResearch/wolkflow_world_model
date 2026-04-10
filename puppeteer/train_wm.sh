#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/userhome/Research_HUB/ChatDev/puppeteer}"
DATA_ROOT="${DATA_ROOT:-./results/world_model_dataset-0409-1754/GAIA/validation}"
DATASET_FILENAME="${DATASET_FILENAME:-workflow_world_model}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoint/workflow_world_model_gaia_validation}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VAL_RATIO="${VAL_RATIO:-0.2}"
SEED="${SEED:-42}"
GRADIENT_CLIP="${GRADIENT_CLIP:-1.0}"
MODEL_DIM="${MODEL_DIM:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LATENT_DIM="${LATENT_DIM:-256}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.15}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-3}"
MAX_FILES="${MAX_FILES:--1}"
MAX_RECORDS="${MAX_RECORDS:--1}"
USE_QWEN_TEXT_ENCODER="${USE_QWEN_TEXT_ENCODER:-1}"
QWEN_TEXT_ENCODER_MODEL_NAME="${QWEN_TEXT_ENCODER_MODEL_NAME:-/extrahome0/HF_models/Qwen/Qwen3-Embedding-0.6B}"
QWEN_TEXT_ENCODER_BATCH_SIZE="${QWEN_TEXT_ENCODER_BATCH_SIZE:-16}"
QWEN_TEXT_ENCODER_DEVICES="${QWEN_TEXT_ENCODER_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
PRECOMPUTE_QWEN_TEXT_CACHE="${PRECOMPUTE_QWEN_TEXT_CACHE:-1}"
QWEN_TEXT_CACHE_DIR="${QWEN_TEXT_CACHE_DIR:-${OUTPUT_DIR}/qwen_text_cache}"
USE_SWANLAB="${USE_SWANLAB:-1}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-SSM-world-model}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_RUN_NAME="${SWANLAB_RUN_NAME:-gaia-validation-wm}"
SWANLAB_TAGS="${SWANLAB_TAGS:-gaia,world-model}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"

cd "${ROOT_DIR}"

if [[ ! -f "train_workflow_world_model.py" ]]; then
  echo "train_workflow_world_model.py not found under ROOT_DIR=${ROOT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

echo "Root dir      : ${ROOT_DIR}"
echo "Data root     : ${DATA_ROOT}"
echo "Output dir    : ${OUTPUT_DIR}"
echo "Device        : ${DEVICE}"
echo "Epochs        : ${EPOCHS}"
echo "Batch size    : ${BATCH_SIZE}"
echo "Learning rate : ${LEARNING_RATE}"
echo "Val ratio     : ${VAL_RATIO}"
echo "Model dim     : ${MODEL_DIM}"
echo "Hidden dim    : ${HIDDEN_DIM}"
echo "Latent dim    : ${LATENT_DIM}"
echo "Dropout       : ${DROPOUT}"
echo "Python        : ${PYTHON_BIN}"
echo "Use Qwen text : ${USE_QWEN_TEXT_ENCODER}"
echo "Qwen-Embedding model    : ${QWEN_TEXT_ENCODER_MODEL_NAME}"
echo "Qwen devices  : ${QWEN_TEXT_ENCODER_DEVICES}"
echo "Precompute cache : ${PRECOMPUTE_QWEN_TEXT_CACHE}"
echo "Qwen cache dir   : ${QWEN_TEXT_CACHE_DIR}"
echo "Use SwanLab   : ${USE_SWANLAB}"
echo "SwanLab proj  : ${SWANLAB_PROJECT}"
echo "SwanLab run   : ${SWANLAB_RUN_NAME}"

if [[ "${USE_QWEN_TEXT_ENCODER}" == "1" && "${PRECOMPUTE_QWEN_TEXT_CACHE}" == "1" ]]; then
  mkdir -p "${QWEN_TEXT_CACHE_DIR}"
  IFS=',' read -r -a QWEN_DEVICE_ARRAY <<< "${QWEN_TEXT_ENCODER_DEVICES}"
  NUM_SHARDS="${#QWEN_DEVICE_ARRAY[@]}"
  if [[ "${NUM_SHARDS}" -le 0 ]]; then
    echo "No Qwen devices configured for precompute." >&2
    exit 1
  fi
  for ((i=0; i<NUM_SHARDS; i++)); do
    DEVICE_NAME="${QWEN_DEVICE_ARRAY[$i]}"
    CACHE_SHARD_PATH="${QWEN_TEXT_CACHE_DIR}/shard_${i}.jsonl"
    if [[ -s "${CACHE_SHARD_PATH}" ]]; then
      echo "Reuse Qwen cache shard: ${CACHE_SHARD_PATH}"
      continue
    fi
    echo "Start Qwen cache shard ${i}/${NUM_SHARDS} on ${DEVICE_NAME}"
    "${PYTHON_BIN}" precompute_qwen_text_cache.py \
      --data-root "${DATA_ROOT}" \
      --dataset-filename "${DATASET_FILENAME}" \
      --model-name "${QWEN_TEXT_ENCODER_MODEL_NAME}" \
      --output-path "${CACHE_SHARD_PATH}" \
      --device "${DEVICE_NAME}" \
      --batch-size "${QWEN_TEXT_ENCODER_BATCH_SIZE}" \
      --max-files "${MAX_FILES}" \
      --max-records "${MAX_RECORDS}" \
      --shard-index "${i}" \
      --num-shards "${NUM_SHARDS}" &
  done
  wait
fi

CMD=(
  "${PYTHON_BIN}" train_workflow_world_model.py
  --data-root "${DATA_ROOT}"
  --dataset-filename "${DATASET_FILENAME}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --val-ratio "${VAL_RATIO}"
  --seed "${SEED}"
  --gradient-clip "${GRADIENT_CLIP}"
  --model-dim "${MODEL_DIM}"
  --hidden-dim "${HIDDEN_DIM}"
  --latent-dim "${LATENT_DIM}"
  --num-heads "${NUM_HEADS}"
  --num-layers "${NUM_LAYERS}"
  --dropout "${DROPOUT}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --early-stop-min-epochs "${EARLY_STOP_MIN_EPOCHS}"
  --max-files "${MAX_FILES}"
  --max-records "${MAX_RECORDS}"
)

if [[ "${USE_QWEN_TEXT_ENCODER}" == "1" ]]; then
  CMD+=(
    --use-qwen-text-encoder
    --qwen-text-encoder-model-name "${QWEN_TEXT_ENCODER_MODEL_NAME}"
    --qwen-text-encoder-batch-size "${QWEN_TEXT_ENCODER_BATCH_SIZE}"
    --qwen-text-encoder-devices "${QWEN_TEXT_ENCODER_DEVICES}"
    --qwen-text-cache-path "${QWEN_TEXT_CACHE_DIR}"
  )
fi

if [[ "${USE_SWANLAB}" == "1" ]]; then
  CMD+=(
    --use-swanlab
    --swanlab-project "${SWANLAB_PROJECT}"
    --swanlab-run-name "${SWANLAB_RUN_NAME}"
    --swanlab-tags "${SWANLAB_TAGS}"
    --swanlab-mode "${SWANLAB_MODE}"
  )
  if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
    CMD+=(
      --swanlab-workspace "${SWANLAB_WORKSPACE}"
    )
  fi
fi

if [[ "${USE_SWANLAB}" == "1" ]]; then
  echo "Tracker args  : --use-swanlab --swanlab-project ${SWANLAB_PROJECT} --swanlab-run-name ${SWANLAB_RUN_NAME}"
else
  echo "Tracker args  : none"
fi

CMD+=("$@")
"${CMD[@]}"
