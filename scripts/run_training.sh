#!/usr/bin/env bash
set -euo pipefail

: "${COHORT_CSV:?Set COHORT_CSV to the labelled cohort CSV}"
: "${PILLAR_MODEL_DIR:?Set PILLAR_MODEL_DIR to Pillar0-AbdomenCT}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mask_guided_nested_cv}"
DEVICE="${DEVICE:-cuda:0}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec graft-train-mask-guided \
  --csv "$COHORT_CSV" \
  --model-dir "$PILLAR_MODEL_DIR" \
  --project-src src \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --fold all \
  --n-splits 5 \
  --inner-validation-fraction 0.20 \
  --epochs 15 \
  --patience 4 \
  --empty-cache-each-case
