#!/usr/bin/env python3
"""Archive LoRA adapters before pruning large TextWorld checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import uuid

from src.training.textworld_checkpoint_retention import (
    prune_checkpoints,
    validated_checkpoints,
)


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
CHECKSUMS_NAME = "SHA256SUMS"


@dataclass(frozen=True)
class ArchivePruneResult:
    archived_steps: list[int]
    verified_steps: list[int]
    removed_checkpoints: list[Path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_files(adapter_dir: Path) -> dict[str, Path]:
    if not adapter_dir.is_dir() or adapter_dir.is_symlink():
        raise ValueError(f"LoRA adapter directory is missing or unsafe: {adapter_dir}")
    files = {name: adapter_dir / name for name in ADAPTER_FILES}
    for path in files.values():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"LoRA adapter file is missing or unsafe: {path}")
    return files


def _checksums(files: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256(files[name]) for name in ADAPTER_FILES}


def _checksum_text(checksums: dict[str, str]) -> str:
    return "".join(f"{checksums[name]}  {name}\n" for name in ADAPTER_FILES)


def _verify_archive(destination: Path, source_checksums: dict[str, str]) -> None:
    destination_files = _adapter_files(destination)
    destination_checksums = _checksums(destination_files)
    if destination_checksums != source_checksums:
        raise ValueError(f"archived LoRA adapter does not match source: {destination}")
    checksum_file = destination / CHECKSUMS_NAME
    expected = _checksum_text(source_checksums)
    if not checksum_file.is_file() or checksum_file.is_symlink():
        raise ValueError(f"archived LoRA adapter checksum file is missing: {checksum_file}")
    if checksum_file.read_text(encoding="utf-8") != expected:
        raise ValueError(f"archived LoRA adapter checksum file does not match: {checksum_file}")


def _archive_adapter(
    source: Path, destination: Path, source_checksums: dict[str, str]
) -> None:
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False)
    try:
        for name in ADAPTER_FILES:
            shutil.copy2(source / name, temporary / name)
        (temporary / CHECKSUMS_NAME).write_text(
            _checksum_text(source_checksums), encoding="utf-8"
        )
        _verify_archive(temporary, source_checksums)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def archive_and_prune_checkpoints(
    output_dir: Path, archive_dir: Path, keep: int
) -> ArchivePruneResult:
    """Archive every tracked LoRA adapter, then prune old full checkpoints."""
    if not isinstance(keep, int) or isinstance(keep, bool) or keep <= 0:
        raise ValueError("checkpoint retention must be a positive integer")

    checkpoints = validated_checkpoints(output_dir)
    unverified_sources: list[tuple[int, Path, dict[str, Path]]] = []
    for step, checkpoint in checkpoints:
        adapter = checkpoint / "actor" / "lora_adapter"
        files = _adapter_files(adapter)
        unverified_sources.append((step, adapter, files))
    sources = [
        (step, adapter, _checksums(files))
        for step, adapter, files in unverified_sources
    ]

    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_steps: list[int] = []
    verified_steps: list[int] = []
    for step, source, source_checksums in sources:
        destination = archive_dir / f"global_step_{step}"
        if destination.exists():
            if not destination.is_dir() or destination.is_symlink():
                raise ValueError(f"archived LoRA adapter path is unsafe: {destination}")
            _verify_archive(destination, source_checksums)
            verified_steps.append(step)
            continue
        _archive_adapter(source, destination, source_checksums)
        archived_steps.append(step)

    removed = prune_checkpoints(output_dir, keep=keep)
    return ArchivePruneResult(
        archived_steps=archived_steps,
        verified_steps=verified_steps,
        removed_checkpoints=removed,
    )
