#!/usr/bin/env bash
# Matched 2,000-step TextWorld formal experiment launcher.

set -euo pipefail

DRY_RUN=false
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help)
        echo "Usage: FORMAL_ARM=behr|union_js|union_js_sft $0 [--dry-run]"
        exit 0
        ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"

FORMAL_ARM="${FORMAL_ARM:-}"
case "${FORMAL_ARM}" in
    behr)
        REWARD_MODE=cauchy
        SFT_LOSS_COEF=0.0
        ;;
    union_js)
        REWARD_MODE=union_js
        SFT_LOSS_COEF=0.0
        ;;
    union_js_sft)
        REWARD_MODE=union_js
        SFT_LOSS_COEF=0.1
        ;;
    *)
        echo "ERROR: FORMAL_ARM must be one of: behr, union_js, union_js_sft" >&2
        exit 2
        ;;
esac

FORMAL_SEED="${FORMAL_SEED:-42}"
FORMAL_RESUME="${FORMAL_RESUME:-0}"
case "${FORMAL_RESUME}" in
    0)
        RESUME_MODE=disable
        PREFLIGHT_MODE=(--new-run)
        ;;
    1)
        RESUME_MODE=auto
        PREFLIGHT_MODE=(--resume)
        ;;
    *) echo "ERROR: FORMAL_RESUME must be 0 or 1" >&2; exit 2 ;;
esac

TRAIN_DATA="${TRAIN_DATA:-${MAIN_ROOT}/data/processed/textworld_grpo_task_split_v1/train/full.parquet}"
VAL_DATA="${VAL_DATA:-${MAIN_ROOT}/data/processed/textworld_grpo_task_split_v1/val/pilot.parquet}"
WORLD_MODEL="${WORLD_MODEL:-${MAIN_ROOT}/models/WorldModel-Textworld-Qwen2.5-7B}"
ACTOR_MODEL="${ACTOR_MODEL:-/DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8000}"
CONSISTENCY_URL="${CONSISTENCY_URL:-http://127.0.0.1:8002}"
if [[ "${REWARD_MODE}" == "cauchy" ]]; then
    SCORER_URL="${JUDGE_URL}"
else
    SCORER_URL="${CONSISTENCY_URL}"
fi

GROUP_SIZE="${GROUP_SIZE:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
VAL_FREQ="${VAL_FREQ:-250}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/checkpoints/textworld_formal_${FORMAL_ARM}_steps${TOTAL_STEPS}_seed${FORMAL_SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-formal-${FORMAL_ARM}-textworld-steps${TOTAL_STEPS}-seed${FORMAL_SEED}}"

export TRAIN_DATA VAL_DATA WORLD_MODEL REWARD_MODE SFT_LOSS_COEF
export GROUP_SIZE ROLLOUT_TEMPERATURE TOTAL_STEPS SAVE_FREQ VAL_FREQ
export OUTPUT_DIR EXPERIMENT_NAME JUDGE_URL CONSISTENCY_URL
export DATA_SEED="${FORMAL_SEED}"
export ACTOR_DATA_LOADER_SEED="${FORMAL_SEED}"
export MAX_ACTOR_CKPT_TO_KEEP RESUME_MODE

echo "TextWorld formal experiment"
echo "  Arm: ${FORMAL_ARM}"
echo "  Data/actor seed: ${FORMAL_SEED}"
echo "  Reward mode: ${REWARD_MODE}"
echo "  Auxiliary SFT coefficient: ${SFT_LOSS_COEF}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Resume: ${FORMAL_RESUME}"

if ! "$DRY_RUN"; then
    for path in "${TRAIN_DATA}" "${VAL_DATA}" "${WORLD_MODEL}" "${ACTOR_MODEL}"; do
        [[ -e "${path}" ]] || { echo "ERROR: required path not found: ${path}" >&2; exit 1; }
    done
    curl --noproxy 127.0.0.1,localhost -fsS --connect-timeout 10 \
        "${SCORER_URL}/health" >/dev/null || {
        echo "ERROR: reward scorer is not healthy at ${SCORER_URL}" >&2
        exit 1
    }
    "${PROJECT_ROOT}/.venv/bin/python" \
        "${PROJECT_ROOT}/src/training/textworld_formal_run.py" \
        "${PREFLIGHT_MODE[@]}" \
        --output-dir "${OUTPUT_DIR}" \
        --arm "${FORMAL_ARM}" \
        --reward-mode "${REWARD_MODE}" \
        --sft-loss-coef "${SFT_LOSS_COEF}" \
        --train-data "${TRAIN_DATA}" \
        --val-data "${VAL_DATA}" \
        --world-model "${WORLD_MODEL}" \
        --actor-model "${ACTOR_MODEL}" \
        --scorer-url "${SCORER_URL}" \
        --seed "${FORMAL_SEED}" \
        --actor-data-loader-seed "${FORMAL_SEED}" \
        --group-size "${GROUP_SIZE}" \
        --rollout-temperature "${ROLLOUT_TEMPERATURE}" \
        --total-steps "${TOTAL_STEPS}" \
        --save-freq "${SAVE_FREQ}" \
        --val-freq "${VAL_FREQ}" \
        --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP}"
fi

if "$DRY_RUN"; then
    exec bash "${SCRIPT_DIR}/run_grpo_textworld_smoke.sh" --dry-run
fi
exec bash "${SCRIPT_DIR}/run_grpo_textworld_smoke.sh"
