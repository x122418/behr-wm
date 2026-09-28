#!/usr/bin/env bash
# Evaluate the base model and the three preserved formal step-3500 adapters.

set -euo pipefail

STAGE="${1:-all}"
case "${STAGE}" in
    preflight|merge|generate|score|verify|all) shift || true ;;
    -h|--help)
        echo "Usage: LWM_MAIN=/path/to/LWM_project FORMAL_ADAPTER_ROOT=/path/to/adapters $0 [preflight|merge|generate|score|verify|all] [runner options]"
        exit 0
        ;;
    *) echo "ERROR: invalid stage: ${STAGE}" >&2; exit 2 ;;
esac

: "${LWM_MAIN:?Set LWM_MAIN to the LWM_project directory}"
: "${FORMAL_ADAPTER_ROOT:?Set FORMAL_ADAPTER_ROOT to the directory containing the three arms}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEXTWORLD_GRPO_VENV="${TEXTWORLD_GRPO_VENV:-${PROJECT_ROOT}/.venv}"
RUN_ROOT="${RUN_ROOT:-${LWM_MAIN}/../lwm_runs}"
MERGED_ROOT="${MERGED_ROOT:-${RUN_ROOT}/merged_models/formal3500_unified}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${RUN_ROOT}/evaluation/formal3500_unified}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${TEXTWORLD_GRPO_VENV}/bin/python" \
    "${PROJECT_ROOT}/scripts/run_textworld_pilot_eval_matrix.py" \
    --stage "${STAGE}" \
    --base-model "${BASE_MODEL:-${LWM_MAIN}/models/WorldModel-Textworld-Qwen2.5-7B}" \
    --actor-model "${ACTOR_MODEL:-${LWM_MAIN}/models/Qwen3-8B}" \
    --adapter-spec "behr_step3500=${FORMAL_ADAPTER_ROOT}/behr/global_step_3500" \
    --adapter-spec "union_js_step3500=${FORMAL_ADAPTER_ROOT}/union_js/global_step_3500" \
    --adapter-spec "union_js_sft_step3500=${FORMAL_ADAPTER_ROOT}/union_js_sft/global_step_3500" \
    --validation-data "${VALIDATION_DATA:-${LWM_MAIN}/data/processed/textworld_grpo_task_split_v1/val/pilot.parquet}" \
    --merged-root "${MERGED_ROOT}" \
    --evaluation-root "${EVALUATION_ROOT}" \
    --merge-gpus "${MERGE_GPUS:-1,3,5}" \
    --server-gpus "${SERVER_GPUS:-1,3,5}" \
    --score-gpus "${SCORE_GPUS:-6,7}" \
    --ports "${WM_PORTS:-8101,8102,8103}" \
    "$@"
