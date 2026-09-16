#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORD2WORLD_REPO="https://github.com/X1AOX1A/Word2World.git"
WORD2WORLD_REVISION="e8dc240269fb222de3c242b09c70bfbf0b4ac1c9"
MAIN_PYTHON="$PROJECT_ROOT/.venv/bin/python"
EVAL_VENV="$PROJECT_ROOT/venv/textworld-eval"

print_plan() {
    printf '%s\n' \
        "TextWorld trajectory-evaluation runtime" \
        "source: $WORD2WORLD_REPO" \
        "revision: $WORD2WORLD_REVISION" \
        "server venv: venv/textworld-eval" \
        "server engine: textworld==1.6.2" \
        "main .venv: agentenv client only"
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    print_plan
    exit 0
fi

for command_name in git uv; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required command not found: $command_name" >&2
        exit 1
    fi
done
if [[ ! -x "$MAIN_PYTHON" ]]; then
    echo "Error: project Python is missing: $MAIN_PYTHON" >&2
    exit 1
fi

RUNTIME_SOURCE="$(mktemp -d /tmp/lwm-word2world-runtime.XXXXXX)"
cleanup() {
    rm -rf "$RUNTIME_SOURCE"
}
trap cleanup EXIT INT TERM

print_plan
git clone --filter=blob:none --no-checkout "$WORD2WORLD_REPO" "$RUNTIME_SOURCE"
git -C "$RUNTIME_SOURCE" sparse-checkout init --cone
git -C "$RUNTIME_SOURCE" sparse-checkout set \
    AgentGym/agentenv \
    AgentGym/agentenv-textworld
git -C "$RUNTIME_SOURCE" checkout --detach "$WORD2WORLD_REVISION"

uv venv "$EVAL_VENV" --python 3.10
UV_LINK_MODE=copy uv pip install \
    --python "$EVAL_VENV/bin/python" \
    "textworld==1.6.2"
UV_LINK_MODE=copy uv pip install \
    --python "$EVAL_VENV/bin/python" \
    "$RUNTIME_SOURCE/AgentGym/agentenv-textworld"

UV_LINK_MODE=copy uv pip install \
    --python "$MAIN_PYTHON" \
    --no-deps \
    "$RUNTIME_SOURCE/AgentGym/agentenv"
UV_LINK_MODE=copy uv pip install \
    --python "$MAIN_PYTHON" \
    "azure.identity<2"

"$MAIN_PYTHON" -c \
    "from agentenv.envs import TextworldEnvClient, TextworldTask; print('agentenv TextWorld client: OK')"
"$EVAL_VENV/bin/python" -c \
    "import textworld, agentenv_textworld; print('textworld', textworld.__version__); print('agentenv_textworld: OK')"
