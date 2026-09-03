#!/usr/bin/env bash
# Resumable single-step baseline on the task-disjoint TextWorld validation pilot.

set -euo pipefail

DRY_RUN=false
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help) echo "Usage: $0 [--dry-run]"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"

INPUT_DATA="${INPUT_DATA:-${MAIN_ROOT}/data/processed/textworld_grpo_task_split_v1/val/pilot.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/evaluation/textworld_sft_val_pilot_1000}"
ACTOR_MODEL="${ACTOR_MODEL:-/DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B}"
WM_API_BASE="${WM_API_BASE:-http://127.0.0.1:8001}"
ACTOR_GPU="${ACTOR_GPU:-5}"
LIMIT="${LIMIT:-1000}"
CONCURRENCY="${CONCURRENCY:-8}"
TOP_KS="${TOP_KS:-32,64}"

export CUDA_VISIBLE_DEVICES="${ACTOR_GPU}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND=(
    "${PROJECT_ROOT}/.venv/bin/python"
    "${PROJECT_ROOT}/src/data/evaluate_textworld_transition_baseline.py"
    --input "${INPUT_DATA}"
    --output-dir "${OUTPUT_DIR}"
    --actor-model-path "${ACTOR_MODEL}"
    --wm-api-base "${WM_API_BASE}"
    --limit "${LIMIT}"
    --concurrency "${CONCURRENCY}"
    --max-tokens 512
    --top-ks "${TOP_KS}"
)

echo "TextWorld validation baseline"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  NO_PROXY=127.0.0.1,localhost"
echo "  Input: ${INPUT_DATA}"
echo "  Output/resume directory: ${OUTPUT_DIR}"
echo "  Actor: ${ACTOR_MODEL}"
echo "  World-model API: ${WM_API_BASE}"
printf '  %s\n' "${COMMAND[@]}"

if "$DRY_RUN"; then
    echo "Dry run only; no model was loaded and no output was created."
    exit 0
fi

for path in "${INPUT_DATA}" "${ACTOR_MODEL}" "${PROJECT_ROOT}/.venv/bin/python"; do
    [ -e "${path}" ] || { echo "ERROR: required path not found: ${path}" >&2; exit 1; }
done
curl --noproxy 127.0.0.1,localhost -fsS --connect-timeout 10 \
    "${WM_API_BASE}/v1/models" >/dev/null || {
    echo "ERROR: world-model API is not healthy at ${WM_API_BASE}" >&2
    exit 1
}

exec "${COMMAND[@]}"
