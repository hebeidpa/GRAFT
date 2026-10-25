#!/usr/bin/env bash
set -euo pipefail

COHORT_CSV="${COHORT_CSV:-examples/deidentified_four_cases/cohort.csv}"
MODEL_DIR="${MODEL_DIR:-models/Pillar0-AbdomenCT}"
CHECKPOINT="${CHECKPOINT:-weights/graft_v4_final_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/prediction}"
DEVICE="${DEVICE:-cuda:0}"

graft-infer \
  --cohort-csv "$COHORT_CSV" \
  --checkpoint "$CHECKPOINT" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE"
