#!/usr/bin/env bash
set -euo pipefail

python -m graft_pillar.train_mask_guided \
  --train-csv /path/to/training_cohort.csv \
  --validation-csv /path/to/internal_validation_cohort.csv \
  --model-dir /path/to/Pillar0-AbdomenCT \
  --project-src /path/to/GRAFT/src \
  --output-dir /path/to/output \
  --device cuda:0 \
  --early-stopping-fraction 0.20 \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --lambda-recurrence 1.0 \
  --lambda-complication 0.75 \
  --epochs 15 \
  --patience 4
