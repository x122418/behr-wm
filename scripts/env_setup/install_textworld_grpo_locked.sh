#!/usr/bin/env bash
# Install the minimal, project-local environment used by TextWorld GRPO.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCK_FILE="${PROJECT_ROOT}/requirements-textworld-grpo.lock"
VENV_PATH="${TEXTWORLD_GRPO_VENV:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${TEXTWORLD_GRPO_PYTHON:-/usr/bin/python3}"
DRY_RUN=false

usage() {
    printf 'Usage: %s [--dry-run]\n' "$0"
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
    esac
done

[ -f "$LOCK_FILE" ] || {
    printf 'ERROR: lock file not found: %s\n' "$LOCK_FILE" >&2
    exit 1
}

printf 'TextWorld GRPO locked environment\n'
printf '  Python: 3.10\n'
printf '  Environment: %s\n' "$VENV_PATH"
printf '  Lock file: %s\n' "$LOCK_FILE"
printf '  Native wheels: CUDA 12.6\n'
sed -n '/^[[:alnum:]][^#]*==/p' "$LOCK_FILE" | sed 's/^/  /'

if "$DRY_RUN"; then
    printf 'Dry run only; no environment was created and no package was installed.\n'
    exit 0
fi

command -v uv >/dev/null 2>&1 || {
    printf 'ERROR: uv is required but was not found in PATH.\n' >&2
    exit 1
}
[ -x "$PYTHON_BIN" ] || {
    printf 'ERROR: Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
}

python_version="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "$python_version" = "3.10" ] || {
    printf 'ERROR: Python 3.10 is required, found %s at %s\n' "$python_version" "$PYTHON_BIN" >&2
    exit 1
}

uv venv "$VENV_PATH" --python "$PYTHON_BIN"
VENV_PYTHON="${VENV_PATH}/bin/python"

# Install the ABI-defining CUDA stack first from the official cu126 index.
uv pip install --python "$VENV_PYTHON" \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Resolve the remaining direct dependencies while preserving the native pins.
uv pip install --python "$VENV_PYTHON" -r "$LOCK_FILE"

# Use the repository-tested prebuilt wheel; never compile Flash Attention here.
uv pip install --python "$VENV_PYTHON" \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'

"$VENV_PYTHON" - <<'PY'
import torch
import transformers
import vllm
import verl

print("Environment import check passed")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("verl", getattr(verl, "__version__", "unknown"))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

printf 'Environment ready: source %s/bin/activate\n' "$VENV_PATH"
