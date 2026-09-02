#!/usr/bin/env bash
# Conservative 50-step TextWorld BehR pilot built on the verified smoke launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"

export TRAIN_DATA="${TRAIN_DATA:-${MAIN_ROOT}/data/processed/textworld_grpo_task_split_v1/train/pilot.parquet}"
export VAL_DATA="${VAL_DATA:-${MAIN_ROOT}/data/processed/textworld_grpo_task_split_v1/val/pilot.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/checkpoints/textworld_behr_pilot}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-textworld-behr-pilot}"
export GROUP_SIZE="${GROUP_SIZE:-2}"
export TOTAL_STEPS="${TOTAL_STEPS:-50}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export VAL_FREQ="${VAL_FREQ:-10}"

# Local model APIs must not be routed through the cluster HTTP proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

exec bash "${SCRIPT_DIR}/run_grpo_textworld_smoke.sh" "$@"
