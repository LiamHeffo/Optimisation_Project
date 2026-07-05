#!/usr/bin/env bash
# Run replot_l1d_outputs.py from the project root.
# Usage: ./scripts/replot_l1d_outputs.sh [input_path] [--suffix SUFFIX]
#   input_path  Single run dir or parent of condition_* dirs (default: src/L1d_Outputs/DEAP_0)
#   --suffix    Suffix appended to output PNG names (default: _replot)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${PROJECT_ROOT}/src/L1d_Outputs/replot_l1d_outputs.py" "$@"
