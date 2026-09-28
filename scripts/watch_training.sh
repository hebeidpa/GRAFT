#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:-outputs/mask_guided_nested_cv/training.log}"
exec tail -n 60 -F "$LOG_FILE"
