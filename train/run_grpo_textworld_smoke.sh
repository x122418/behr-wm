#!/usr/bin/env bash
# Two-step TextWorld BehR GRPO smoke launcher. Use --dry-run without GPUs.

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

TRAIN_DATA="${TRAIN_DATA:-${MAIN_ROOT}/data/processed/textworld_grpo_task_split_v1/train/smoke.parquet}"
VAL_DATA="${VAL_DATA:-${MAIN_ROOT}/data/processed/textworld_grpo_task_split_v1/val/pilot.parquet}"
WORLD_MODEL="${WORLD_MODEL:-${MAIN_ROOT}/models/WorldModel-Textworld-Qwen2.5-7B}"
REWARD_FN_PATH="${REWARD_FN_PATH:-${PROJECT_ROOT}/src/reward/behr_reward_textworld.py}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8000}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-Qwen/Qwen3-8B}"
CONSISTENCY_URL="${CONSISTENCY_URL:-http://127.0.0.1:8002}"
CONSISTENCY_TOP_K="${CONSISTENCY_TOP_K:-64}"
REWARD_MODE="${REWARD_MODE:-cauchy}"
case "${REWARD_MODE}" in
    cauchy|union_js|full_vocab_js) ;;
    *) echo "ERROR: unsupported REWARD_MODE: ${REWARD_MODE}" >&2; exit 2 ;;
esac
GPU_IDS="${CUDA_VISIBLE_DEVICES:-5,6,7}"
N_GPUS="${N_GPUS:-$(awk -F, '{print NF}' <<<"${GPU_IDS}")}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/checkpoints/textworld_${REWARD_MODE}_smoke}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-textworld-behr-smoke}"
GROUP_SIZE="${GROUP_SIZE:-2}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.3}"
SFT_LOSS_COEF="${SFT_LOSS_COEF:-0.0}"
DATA_SEED="${DATA_SEED:-}"
ACTOR_DATA_LOADER_SEED="${ACTOR_DATA_LOADER_SEED:-}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-}"
RESUME_MODE="${RESUME_MODE:-}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-}"
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-True}"
case "${RESUME_MODE}" in
    ""|auto|disable|resume_path) ;;
    *) echo "ERROR: unsupported RESUME_MODE: ${RESUME_MODE}" >&2; exit 2 ;;
esac
case "${FILTER_OVERLONG_PROMPTS}" in
    True|False) ;;
    *) echo "ERROR: FILTER_OVERLONG_PROMPTS must be True or False" >&2; exit 2 ;;
esac
TOTAL_STEPS="${TOTAL_STEPS:-2}"
SAVE_FREQ="${SAVE_FREQ:--1}"
VAL_FREQ="${VAL_FREQ:--1}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"

COMMAND=(
    "${PROJECT_ROOT}/.venv/bin/python" -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "data.train_files=${TRAIN_DATA}"
    "data.val_files=${VAL_DATA}"
    data.train_batch_size=4
    data.max_prompt_length=4096
    data.max_response_length=512
    "data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS}"
    data.truncation=left
    "actor_rollout_ref.model.path=${WORLD_MODEL}"
    actor_rollout_ref.actor.optim.lr=5e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=4
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    "++actor_rollout_ref.actor.sft_loss_coef=${SFT_LOSS_COEF}"
    actor_rollout_ref.actor.entropy_coeff=0
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.max_model_len=4608
    actor_rollout_ref.rollout.enforce_eager=True
    "actor_rollout_ref.rollout.n=${GROUP_SIZE}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_p=1.0
    "custom_reward_function.path=${REWARD_FN_PATH}"
    custom_reward_function.name=compute_score
    ++custom_reward_function.reward_kwargs.use_http_judge=True
    "++custom_reward_function.reward_kwargs.judge_api_url=${JUDGE_URL}"
    "++custom_reward_function.reward_kwargs.judge_model_path=${JUDGE_MODEL_PATH}"
    "++custom_reward_function.reward_kwargs.reward_mode=${REWARD_MODE}"
    "++custom_reward_function.reward_kwargs.consistency_api_url=${CONSISTENCY_URL}"
    "++custom_reward_function.reward_kwargs.consistency_top_k=${CONSISTENCY_TOP_K}"
    ++custom_reward_function.reward_kwargs.behavior_scale_coef=1.0
    ++custom_reward_function.reward_kwargs.behavior_weight=1.0
    ++custom_reward_function.reward_kwargs.facts_weight=0.0
    ++custom_reward_function.reward_kwargs.format_penalty=-1.0
    ++custom_reward_function.reward_kwargs.max_workers=4
    'trainer.logger=["console","tensorboard"]'
    trainer.project_name=behr-wm-textworld
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS}"
    trainer.nnodes=1
    trainer.val_before_train=False
    "trainer.test_freq=${VAL_FREQ}"
    "trainer.save_freq=${SAVE_FREQ}"
    trainer.total_epochs=1
    "trainer.total_training_steps=${TOTAL_STEPS}"
    "trainer.default_local_dir=${OUTPUT_DIR}"
)
if [[ -n "${DATA_SEED}" ]]; then
    COMMAND+=("data.seed=${DATA_SEED}")
fi
if [[ -n "${ACTOR_DATA_LOADER_SEED}" ]]; then
    COMMAND+=("actor_rollout_ref.actor.data_loader_seed=${ACTOR_DATA_LOADER_SEED}")
fi
if [[ -n "${MAX_ACTOR_CKPT_TO_KEEP}" ]]; then
    COMMAND+=("trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}")
fi
if [[ -n "${RESUME_MODE}" ]]; then
    COMMAND+=("trainer.resume_mode=${RESUME_MODE}")
fi
if [[ -n "${RAY_TEMP_DIR}" ]]; then
    COMMAND+=("++ray_kwargs.ray_init._temp_dir=${RAY_TEMP_DIR}")
fi
echo "TextWorld BehR GRPO smoke configuration"
echo "  GPUs: ${GPU_IDS} (${N_GPUS})"
echo "  World model: ${WORLD_MODEL}"
echo "  Train: ${TRAIN_DATA}"
echo "  Validation: ${VAL_DATA}"
echo "  Judge: ${JUDGE_URL}"
echo "  Consistency scorer: ${CONSISTENCY_URL} (top-k=${CONSISTENCY_TOP_K})"
echo "  Reward mode: ${REWARD_MODE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  TENSORBOARD_DIR=${TENSORBOARD_DIR}"
printf '  %s\n' "${COMMAND[@]}"

if "$DRY_RUN"; then
    echo "Dry run only; no output directory was created and no training was started."
    exit 0
fi

if [[ "${SFT_LOSS_COEF}" != "0" && "${SFT_LOSS_COEF}" != "0.0" ]]; then
    "${PROJECT_ROOT}/.venv/bin/python" \
        "${PROJECT_ROOT}/scripts/install_verl_sft_aux_patch.py" --check
fi

for path in "$TRAIN_DATA" "$VAL_DATA" "$REWARD_FN_PATH" "$WORLD_MODEL" "${PROJECT_ROOT}/.venv/bin/python"; do
    [ -e "$path" ] || { echo "ERROR: required path not found: $path" >&2; exit 1; }
done
if [[ "${REWARD_MODE}" == "cauchy" ]]; then
    SCORER_URL="${JUDGE_URL}"
else
    SCORER_URL="${CONSISTENCY_URL}"
fi
curl --noproxy 127.0.0.1,localhost -fsS --connect-timeout 10 \
    "${SCORER_URL}/health" >/dev/null || {
    echo "ERROR: reward scorer is not healthy at ${SCORER_URL}" >&2
    exit 1
}

mkdir -p "${OUTPUT_DIR}/logs"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=disabled
export TENSORBOARD_DIR
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/logs/train.log"
