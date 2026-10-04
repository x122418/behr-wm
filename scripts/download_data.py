#!/usr/bin/env python3
"""Fetch pinned upstream assets required by the BehR-WM evaluations."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

HF_REPO_ID = "X1AOX1A/LLMasWorldModels"
REVISION = "ff6ae2b924d1a49e4b89825913887f2ea96cb282"
ENVS = ("webshop", "textworld")
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEBSHOP_BACKEND_DIR = ROOT / "AgentGym" / "agentenv-webshop" / "webshop"

TEST_FILES = {
    "webshop": [
        "llama_factory/webshop_test_109.json",
        "eval/webshop_test.json",
        "init_contexts/webshop/agent_instruct_test.json",
        "init_contexts/webshop/wm_instruct_test.json",
    ],
    "textworld": [
        "llama_factory/textworld_test_173.json",
        "eval/textworld_test.json",
        "init_contexts/textworld/agent_instruct_test.json",
        "init_contexts/textworld/wm_instruct_test.json",
    ],
}

ARCHIVES = {
    "textworld.zip": (DATA_DIR / "textworld", DATA_DIR / "textworld" / "games"),
    "webshop.zip": (WEBSHOP_BACKEND_DIR, WEBSHOP_BACKEND_DIR / "data"),
    "webshop_index.zip": (
        WEBSHOP_BACKEND_DIR,
        WEBSHOP_BACKEND_DIR / "search_index",
    ),
}


def _require_hub():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(
            "[behr-wm] huggingface_hub is required. Install it with:\n"
            "              pip install 'huggingface_hub>=0.24.0'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return snapshot_download


def _plan(envs: list[str], webshop_backend: bool) -> list[str]:
    paths: list[str] = []
    for env in envs:
        paths.extend(TEST_FILES[env])
    if "textworld" in envs:
        paths.append("textworld.zip")
    if webshop_backend:
        paths.extend(["webshop.zip", "webshop_index.zip"])
    return list(dict.fromkeys(paths))


def _extract(archive_rel: str, force: bool) -> None:
    destination, sentinel = ARCHIVES[archive_rel]
    archive = DATA_DIR / archive_rel
    if not archive.is_file():
        print(f"[behr-wm] expected archive missing: {archive}", file=sys.stderr)
        raise SystemExit(1)
    if sentinel.exists() and not force:
        print(f"[behr-wm] {sentinel.relative_to(ROOT)} already present, skipping extract")
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[behr-wm] extracting {archive_rel} -> {destination.relative_to(ROOT)}/ ...")
    with zipfile.ZipFile(archive) as archive_file:
        archive_file.extractall(destination)
    if not sentinel.exists():
        print(
            f"[behr-wm] WARNING: expected {sentinel.relative_to(ROOT)} after "
            f"extracting {archive_rel}; the upstream archive layout may have changed.",
            file=sys.stderr,
        )


def _download(envs: list[str], webshop_backend: bool, force_extract: bool) -> int:
    snapshot_download = _require_hub()
    paths = _plan(envs, webshop_backend)
    print(f"[behr-wm] source:   {HF_REPO_ID}")
    print(f"[behr-wm] revision: {REVISION}")
    print(f"[behr-wm] fetching {len(paths)} path(s) into data/ ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(DATA_DIR),
        allow_patterns=paths,
    )
    for archive_rel in ARCHIVES:
        if archive_rel in paths:
            _extract(archive_rel, force=force_extract)
    print("[behr-wm] done. Next: docs/EVALUATION.md")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download pinned BehR-WM evaluation assets from "
            f"{HF_REPO_ID} (arXiv:2512.18832)."
        )
    )
    parser.add_argument(
        "--env",
        choices=(*ENVS, "all"),
        default="all",
        help="Environment subset (default: all).",
    )
    parser.add_argument(
        "--webshop-backend",
        action="store_true",
        help=(
            "Also fetch and unpack the WebShop product corpus and search index "
            "needed for real-environment WebShop evaluation."
        ),
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract archives even if the target directory already exists.",
    )
    args = parser.parse_args()
    envs = list(ENVS) if args.env == "all" else [args.env]
    if args.webshop_backend and "webshop" not in envs:
        envs.append("webshop")
    return _download(envs, args.webshop_backend, args.force_extract)


if __name__ == "__main__":
    sys.exit(main())
