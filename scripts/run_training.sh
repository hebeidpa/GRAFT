#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_CSV:?Set TRAIN_CSV to the labelled training-cohort CSV}"
: "${PILLAR_MODEL_DIR:?Set PILLAR_MODEL_DIR to Pillar0-AbdomenCT}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mask_guided_internal_validation}"
DEVICE="${DEVICE:-cuda:0}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

ARGS=(
  --train-csv "$TRAIN_CSV"
  --model-dir "$PILLAR_MODEL_DIR"
  --project-src src
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --early-stopping-fraction 0.20
  --epochs 15
  --patience 4
)

if [[ -n "${VALIDATION_CSV:-}" ]]; then
  ARGS+=(--validation-csv "$VALIDATION_CSV")
fi

exec graft-train-mask-guided "${ARGS[@]}"
