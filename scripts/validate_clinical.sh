#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config.yaml}"
graft-validate-clinical --config "$CONFIG"
