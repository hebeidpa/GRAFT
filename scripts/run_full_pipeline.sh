#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config.yaml}"
graft-validate --config "$CONFIG"
graft-extract --config "$CONFIG" --max-cases 2
graft-extract --config "$CONFIG"
graft-validate-clinical --config "$CONFIG"
graft-train --config "$CONFIG"
