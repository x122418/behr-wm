#!/usr/bin/env bash
# Matched 50-step Union-JS pilot with an auxiliary real-observation SFT loss.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export REWARD_MODE="${REWARD_MODE:-union_js}"
export SFT_LOSS_COEF="${SFT_LOSS_COEF:-0.1}"
export GROUP_SIZE="${GROUP_SIZE:-4}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export TOTAL_STEPS="${TOTAL_STEPS:-50}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export VAL_FREQ="${VAL_FREQ:-10}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-unionjs-k64-g4-t07-sft01-textworld-pilot1k-50steps-seed42}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/checkpoints/textworld_unionjs_k64_g4_t07_sft01_pilot1k_50steps_seed42}"

exec bash "${SCRIPT_DIR}/run_grpo_textworld_pilot.sh" "$@"
