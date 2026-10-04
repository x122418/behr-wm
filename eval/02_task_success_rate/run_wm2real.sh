#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK="${TASK:-webshop}"
WM_OUTPUT_DIR="${1:-$PROJECT_ROOT/outputs/task_success_rate/wm/$TASK}"
N_SAMPLES="${N_SAMPLES:--1}"
ENV_PORT="${ENV_PORT:-$((30000 + RANDOM % 30000))}"

if [[ "$TASK" == "textworld" ]]; then
    MAX_WORKERS="${MAX_WORKERS:-1}"
    SERVER_COMMAND=(
        "$PROJECT_ROOT/venv/textworld-eval/bin/textworld"
        --host 127.0.0.1
        --port "$ENV_PORT"
    )
else
    MAX_WORKERS="${MAX_WORKERS:-50}"
    case "$TASK" in
        webshop)
            SERVER_COMMAND=(webshop --host 127.0.0.1 --port "$ENV_PORT")
            ;;
        alfworld|alfworld_valid_seen|alfworld_valid_unseen)
            SERVER_COMMAND=(python -m agentenv.envs.alfworld_server --host 127.0.0.1 --port "$ENV_PORT")
            ;;
        sciworld)
            SERVER_COMMAND=(python -m agentenv.envs.sciworld_server --host 127.0.0.1 --port "$ENV_PORT")
            ;;
        *)
            echo "Error: unsupported task: $TASK" >&2
            exit 1
            ;;
    esac
fi

REPLAY_COMMAND=(
    "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/cal_wm2real.py"
    --task "$TASK"
    --test_file_root "$WM_OUTPUT_DIR"
    --port "$ENV_PORT"
    --max_workers "$MAX_WORKERS"
    --n_samples "$N_SAMPLES"
)

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "NO_PROXY=127.0.0.1,localhost"
    print_command "${SERVER_COMMAND[@]}"
    print_command "${REPLAY_COMMAND[@]}"
    exit 0
fi

if [[ ! -d "$WM_OUTPUT_DIR" ]]; then
    echo "Error: WM output directory not found: $WM_OUTPUT_DIR" >&2
    exit 1
fi
if [[ "$TASK" == "textworld" && ! -x "${SERVER_COMMAND[0]}" ]]; then
    echo "Error: TextWorld evaluation runtime is missing: ${SERVER_COMMAND[0]}" >&2
    echo "Run: bash scripts/env_setup/install_textworld_eval_runtime.sh" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
SERVER_LOG="/tmp/lwm_${TASK}_server_${ENV_PORT}.log"
"${SERVER_COMMAND[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$ENV_PORT/" >/dev/null; then
        "${REPLAY_COMMAND[@]}"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: $TASK server exited. See $SERVER_LOG" >&2
        tail -40 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

echo "Error: $TASK server did not become ready. See $SERVER_LOG" >&2
exit 1
