#!/usr/bin/env python3
"""Reject vLLM CuMem wheels that corrupt None on CPython 3.10/3.11."""

import importlib.util
from pathlib import Path
import subprocess
import sys


def main() -> int:
    if sys.version_info >= (3, 12):
        print("safe vLLM CuMem runtime: CPython None is immortal")
        return 0

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        print("ERROR: vLLM is not installed", file=sys.stderr)
        return 2

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    extensions = list(package_dir.glob("cumem_allocator*.so"))
    if len(extensions) != 1:
        print(
            f"ERROR: expected one vLLM CuMem extension, found {extensions}",
            file=sys.stderr,
        )
        return 2

    try:
        symbols = subprocess.run(
            ["nm", "-D", str(extensions[0])],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: could not inspect {extensions[0]}: {exc}", file=sys.stderr)
        return 2

    if "Py_IncRef" not in symbols:
        print(
            "ERROR: unsafe vLLM CuMem runtime: the abi3 extension returns "
            "None without a runtime Py_IncRef",
            file=sys.stderr,
        )
        return 1

    print(f"safe vLLM CuMem runtime: {extensions[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
