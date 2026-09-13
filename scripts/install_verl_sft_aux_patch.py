#!/usr/bin/env python3
"""Apply or verify the project auxiliary-SFT patch for VERL 0.7.1."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import py_compile
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = PROJECT_ROOT / "patches" / "verl-0.7.1-textworld-sft-aux.patch"
REQUIRED_VERSION = "0.7.1"
TARGETS = (
    Path("verl/workers/config/actor.py"),
    Path("verl/workers/fsdp_workers.py"),
    Path("verl/workers/actor/dp_actor.py"),
)
PATCHED_MARKERS = {
    TARGETS[0]: "sft_loss_coef: float = 0.0",
    TARGETS[1]: "tokenizer=self.tokenizer",
    TARGETS[2]: "data.batch = attach_aux_sft_batch(data.batch, sft_batch)",
}


def _default_site_packages() -> Path:
    for entry in sys.path:
        candidate = Path(entry)
        if (candidate / "verl").is_dir() and list(candidate.glob("verl-*.dist-info")):
            return candidate.resolve()
    raise RuntimeError("could not locate the installed VERL site-packages directory")


def _installed_version(site_packages: Path) -> str:
    metadata_files = sorted(site_packages.glob("verl-*.dist-info/METADATA"))
    if len(metadata_files) != 1:
        raise RuntimeError(
            f"expected exactly one VERL distribution in {site_packages}, found {len(metadata_files)}"
        )
    metadata = Parser().parsestr(metadata_files[0].read_text(encoding="utf-8"))
    return metadata.get("Version", "")


def _patch_state(site_packages: Path) -> str:
    states = []
    for relative_path, marker in PATCHED_MARKERS.items():
        target = site_packages / relative_path
        if not target.is_file():
            raise RuntimeError(f"missing VERL patch target: {target}")
        states.append(target.read_text(encoding="utf-8").count(marker) == 1)
    if all(states):
        return "patched"
    if not any(states):
        return "unpatched"
    raise RuntimeError("VERL auxiliary SFT patch is only partially applied")


def _compile_targets(site_packages: Path) -> None:
    for relative_path in TARGETS:
        py_compile.compile(str(site_packages / relative_path), doraise=True)


def apply_patch(site_packages: Path) -> None:
    state = _patch_state(site_packages)
    if state == "patched":
        _compile_targets(site_packages)
        print("VERL auxiliary SFT patch already applied")
        return
    backups = {
        relative_path: (site_packages / relative_path).read_bytes()
        for relative_path in TARGETS
    }
    result = subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "--fuzz=0",
            "-p1",
            "-d",
            str(site_packages),
            "-i",
            str(PATCH_PATH),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        for relative_path, content in backups.items():
            (site_packages / relative_path).write_bytes(content)
        raise RuntimeError(f"failed to apply VERL patch:\n{result.stdout}{result.stderr}")
    try:
        if _patch_state(site_packages) != "patched":
            raise RuntimeError("patch command completed without all expected markers")
        _compile_targets(site_packages)
    except Exception:
        for relative_path, content in backups.items():
            (site_packages / relative_path).write_bytes(content)
        raise
    print("VERL auxiliary SFT patch applied")


def check_patch(site_packages: Path) -> None:
    if _patch_state(site_packages) != "patched":
        raise RuntimeError("VERL auxiliary SFT patch is not installed")
    _compile_targets(site_packages)
    print("VERL auxiliary SFT patch is installed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--site-packages", type=Path)
    args = parser.parse_args()

    try:
        site_packages = (args.site_packages or _default_site_packages()).resolve()
        version = _installed_version(site_packages)
        if version != REQUIRED_VERSION:
            raise RuntimeError(
                f"auxiliary SFT patch requires VERL {REQUIRED_VERSION}; found {version or 'unknown'}"
            )
        if not PATCH_PATH.is_file():
            raise RuntimeError(f"missing patch file: {PATCH_PATH}")
        if args.apply:
            apply_patch(site_packages)
        else:
            check_patch(site_packages)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
