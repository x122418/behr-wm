#!/usr/bin/env bash
# Start one frozen-actor TextWorld consistency scorer service.

set -euo pipefail

MODEL=""
GPU=""
HOST="0.0.0.0"
PORT="8002"
TOP_K="64"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --gpu) GPU="${2:-}"; shift 2 ;;
        --host) HOST="${2:-}"; shift 2 ;;
        --port) PORT="${2:-}"; shift 2 ;;
        --top-k) TOP_K="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 --model PATH --gpu ID [--port 8002] [--top-k 64] [--dry-run]"
            exit 0
            ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL}" || -z "${GPU}" ]]; then
    echo "ERROR: --model and --gpu are required" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND=(
    "${PROJECT_ROOT}/.venv/bin/python"
    -m src.reward.textworld_consistency_server
    --model "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --top-k "${TOP_K}"
)

echo "TextWorld actor consistency service"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  NO_PROXY=127.0.0.1,localhost"
echo "  Model: ${MODEL}"
echo "  Host: ${HOST}"
echo "  Port: ${PORT}"
echo "  Top-k: ${TOP_K}"
printf '  %s\n' "${COMMAND[@]}"

if "$DRY_RUN"; then
    echo "Dry run only; no model was loaded."
    exit 0
fi

[[ -e "${MODEL}" ]] || { echo "ERROR: model path not found: ${MODEL}" >&2; exit 1; }
[[ -x "${PROJECT_ROOT}/.venv/bin/python" ]] || {
    echo "ERROR: project Python not found" >&2
    exit 1
}

exec "${COMMAND[@]}"
