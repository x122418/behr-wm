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
SFT_LOSS_BLOCK = '''                    if sft_loss_coef > 0:
                        sft_inputs = {
                            key.removeprefix("sft_"): value
                            for key, value in model_inputs.items()
                            if key.startswith("sft_")
                        }
                        sft_inputs["pad_token_id"] = pad_token_id
                        sft_outputs = self._forward_micro_batch(
                            sft_inputs, temperature=1.0, calculate_entropy=False
                        )
                        sft_loss = token_mean_nll(
                            sft_outputs["log_probs"], sft_inputs["response_mask"]
                        )
                        zero_kl = pg_loss.new_zeros(())
                        effective_kl_loss = kl_loss if self.config.use_kl_loss else zero_kl
                        policy_loss = combine_actor_losses(
                            pg_loss=pg_loss,
                            kl_loss=effective_kl_loss,
                            kl_coef=self.config.kl_loss_coef if self.config.use_kl_loss else 0.0,
                            sft_loss=sft_loss,
                            sft_coef=sft_loss_coef,
                        )
                        if calculate_entropy and entropy is not None and entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff
                        metrics["actor/sft_loss"] += sft_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/sft_loss_coef"] = sft_loss_coef

'''
SFT_LOSS_ANCHOR = '''                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
'''


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
        _validate_patched_structure(site_packages)
        return "patched"
    if not any(states):
        return "unpatched"
    raise RuntimeError("VERL auxiliary SFT patch is only partially applied")


def _validate_patched_structure(site_packages: Path) -> None:
    actor_source = (site_packages / TARGETS[0]).read_text(encoding="utf-8")
    fsdp_source = (site_packages / TARGETS[1]).read_text(encoding="utf-8")
    dp_actor_source = (site_packages / TARGETS[2]).read_text(encoding="utf-8")
    expected_counts = {
        "actor sft_loss_coef": (actor_source, "sft_loss_coef: float = 0.0", 1),
        "actor validation": (actor_source, "sft_loss_coef must be nonnegative", 1),
        "FSDP tokenizer forwarding": (fsdp_source, "tokenizer=self.tokenizer", 1),
        "auxiliary SFT import": (dp_actor_source, "from src.training.textworld_sft_aux import (", 1),
        "actor tokenizer": (dp_actor_source, "self.tokenizer = tokenizer", 1),
        "auxiliary batch attachment": (
            dp_actor_source,
            "data.batch = attach_aux_sft_batch(data.batch, sft_batch)",
            1,
        ),
        "SFT coefficient branches": (dp_actor_source, "if sft_loss_coef > 0:", 3),
        "SFT loss combination": (dp_actor_source, "policy_loss = combine_actor_losses(", 1),
        "SFT loss metric": (
            dp_actor_source,
            'metrics["actor/sft_loss"] += sft_loss.detach().item() * loss_scale_factor',
            1,
        ),
        "total loss metric": (
            dp_actor_source,
            'metrics["actor/total_loss"] += policy_loss.detach().item() * loss_scale_factor',
            1,
        ),
    }
    for label, (source, marker, expected) in expected_counts.items():
        actual = source.count(marker)
        if actual != expected:
            raise RuntimeError(
                f"invalid patched VERL structure: {label} occurs {actual} times; expected {expected}"
            )

    kl_index = dp_actor_source.index('micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef')
    sft_index = dp_actor_source.index(SFT_LOSS_BLOCK.rstrip())
    dynamic_index = dp_actor_source.index(SFT_LOSS_ANCHOR, sft_index)
    if not kl_index < sft_index < dynamic_index:
        raise RuntimeError(
            "invalid patched VERL structure: auxiliary SFT loss must be between KL and dynamic batching"
        )


def _restore_targets(site_packages: Path, backups: dict[Path, bytes]) -> None:
    for relative_path, content in backups.items():
        (site_packages / relative_path).write_bytes(content)


def _cleanup_patch_artifacts(site_packages: Path) -> None:
    for relative_path in TARGETS:
        target = site_packages / relative_path
        for suffix in (".rej", ".orig"):
            artifact = Path(f"{target}{suffix}")
            if artifact.exists():
                artifact.unlink()


def _build_context_drift_candidate(
    partial_contents: dict[Path, bytes],
) -> dict[Path, bytes]:
    candidate = dict(partial_contents)
    actor_path = TARGETS[2]
    source = candidate[actor_path].decode("utf-8")
    if source.count("policy_loss = combine_actor_losses(") != 0:
        raise RuntimeError("compatibility path expected the auxiliary SFT loss block to be absent")
    anchor_count = source.count(SFT_LOSS_ANCHOR)
    if anchor_count != 1:
        raise RuntimeError(
            f"compatibility anchor occurs {anchor_count} times; expected exactly one"
        )
    source = source.replace(SFT_LOSS_ANCHOR, SFT_LOSS_BLOCK + SFT_LOSS_ANCHOR)
    candidate[actor_path] = source.encode("utf-8")
    return candidate


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
        partial_contents = {
            relative_path: (site_packages / relative_path).read_bytes()
            for relative_path in TARGETS
        }
        _restore_targets(site_packages, backups)
        _cleanup_patch_artifacts(site_packages)
        try:
            candidate = _build_context_drift_candidate(partial_contents)
            for relative_path, content in candidate.items():
                (site_packages / relative_path).write_bytes(content)
            if _patch_state(site_packages) != "patched":
                raise RuntimeError("compatibility path completed without all expected markers")
            _compile_targets(site_packages)
        except Exception as compatibility_error:
            _restore_targets(site_packages, backups)
            _cleanup_patch_artifacts(site_packages)
            raise RuntimeError(
                "failed to apply VERL patch:\n"
                f"{result.stdout}{result.stderr}"
                f"compatibility path failed: {compatibility_error}"
            ) from compatibility_error
        print("VERL auxiliary SFT patch applied via context-drift compatibility path")
        return
    try:
        if _patch_state(site_packages) != "patched":
            raise RuntimeError("patch command completed without all expected markers")
        _compile_targets(site_packages)
    except Exception:
        _restore_targets(site_packages, backups)
        _cleanup_patch_artifacts(site_packages)
        raise
    _cleanup_patch_artifacts(site_packages)
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
