#!/usr/bin/env bash
# One-command launcher for the fixed seven-model TextWorld pilot evaluation.

set -euo pipefail

STAGE="${1:-all}"
case "${STAGE}" in
    preflight|merge|generate|score|verify|all) shift || true ;;
    -h|--help)
        echo "Usage: LWM_HDD=/shared/root $0 [preflight|merge|generate|score|verify|all] [runner options]"
        exit 0
        ;;
    *) echo "ERROR: invalid stage: ${STAGE}" >&2; exit 2 ;;
esac

: "${LWM_HDD:?Set LWM_HDD to the shared project root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LWM_MAIN="${LWM_MAIN:-${LWM_HDD}/workspace/LWM_project}"
TEXTWORLD_GRPO_VENV="${TEXTWORLD_GRPO_VENV:-${LWM_HDD}/envs/lwm-grpo}"
PILOT_ROOT="${PILOT_ROOT:-${LWM_HDD}/lwm_runs/textworld_lora_pilot100_seed42_retry1}"
MERGED_ROOT="${MERGED_ROOT:-${LWM_HDD}/lwm_runs/merged_models/pilot100_unified}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${LWM_HDD}/lwm_runs/evaluation/pilot100_unified}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${TEXTWORLD_GRPO_VENV}/bin/python" \
    "${PROJECT_ROOT}/scripts/run_textworld_pilot_eval_matrix.py" \
    --stage "${STAGE}" \
    --base-model "${BASE_MODEL:-${LWM_MAIN}/models/WorldModel-Textworld-Qwen2.5-7B}" \
    --actor-model "${ACTOR_MODEL:-${LWM_MAIN}/models/Qwen3-8B}" \
    --pilot-root "${PILOT_ROOT}" \
    --validation-data "${VALIDATION_DATA:-${LWM_MAIN}/data/processed/textworld_grpo_task_split_v1/val/pilot.parquet}" \
    --merged-root "${MERGED_ROOT}" \
    --evaluation-root "${EVALUATION_ROOT}" \
    --merge-gpus "${MERGE_GPUS:-1,3,5}" \
    --server-gpus "${SERVER_GPUS:-1,3,5}" \
    --score-gpus "${SCORE_GPUS:-6,7}" \
    --ports "${WM_PORTS:-8101,8102,8103}" \
    "$@"
