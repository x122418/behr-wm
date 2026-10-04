#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${1:-Qwen3-8B}"
MODEL="${2:-vllm_agent}"
API_KEY="${3:-EMPTY}"
API_BASE_URL="${4:-http://localhost:8000/v1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MAX_ROUND="${MAX_ROUND:-50}"
NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$MODEL_PATH")}"
ENV_PORT="${ENV_PORT:-$((30000 + RANDOM % 30000))}"

SERVER_BIN="$PROJECT_ROOT/venv/textworld-eval/bin/textworld"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
INFERENCE_FILE="$PROJECT_ROOT/data/eval/textworld_test.json"
OUTPUT_DIR="$PROJECT_ROOT/outputs/task_success_rate/real/textworld/$EXPERIMENT_NAME"

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

SERVER_COMMAND=(
    "$SERVER_BIN" --host 127.0.0.1 --port "$ENV_PORT"
)
EVAL_COMMAND=(
    "$PYTHON_BIN" "$SCRIPT_DIR/interact_with_real.py"
    --api_key "$API_KEY"
    --base_url "$API_BASE_URL"
    --model "$MODEL"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --inference_file "$INFERENCE_FILE"
    --output_dir "$OUTPUT_DIR"
    --max_round "$MAX_ROUND"
    --max_concurrency "$MAX_CONCURRENCY"
    --max_tokens "$MAX_TOKENS"
    --num_examples "$NUM_EXAMPLES"
    --task_name textworld
    --env_server_base "http://127.0.0.1:$ENV_PORT"
)

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "NO_PROXY=127.0.0.1,localhost"
    print_command "${SERVER_COMMAND[@]}"
    print_command "${EVAL_COMMAND[@]}"
    exit 0
fi

for required in "$SERVER_BIN" "$PYTHON_BIN" "$INFERENCE_FILE"; do
    if [[ ! -e "$required" ]]; then
        echo "Error: required TextWorld evaluation input is missing: $required" >&2
        exit 1
    fi
done

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
SERVER_LOG="/tmp/lwm_textworld_server_${ENV_PORT}.log"
"${SERVER_COMMAND[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$ENV_PORT/" >/dev/null; then
        "${EVAL_COMMAND[@]}"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: TextWorld server exited. See $SERVER_LOG" >&2
        tail -40 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

echo "Error: TextWorld server did not become ready. See $SERVER_LOG" >&2
exit 1
