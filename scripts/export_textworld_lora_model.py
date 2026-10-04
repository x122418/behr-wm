#!/usr/bin/env python3
"""Merge a TextWorld PEFT adapter into its base model reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence


PROVENANCE_NAME = "LWM_MERGE_PROVENANCE.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_export_inputs(base_model: Path, adapter: Path, output_dir: Path) -> None:
    required = (
        base_model / "config.json",
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required file not found: {path}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")


def model_artifact_manifest(model_path: Path) -> dict[str, object]:
    config = model_path / "config.json"
    index = model_path / "model.safetensors.index.json"
    weights = sorted(model_path.glob("*.safetensors"))
    if not config.is_file() or not weights:
        raise FileNotFoundError(f"incomplete model artifacts: {model_path}")
    manifest: dict[str, object] = {
        "path": str(model_path.resolve()),
        "config_sha256": sha256_file(config),
        "weights": [
            {"name": path.name, "size": path.stat().st_size} for path in weights
        ],
    }
    if index.is_file():
        manifest["index_sha256"] = sha256_file(index)
    revision = model_path / "LWM_SOURCE_REVISION.txt"
    if revision.is_file():
        manifest["source_revision"] = revision.read_text(encoding="utf-8").strip()
    return manifest


def build_provenance(base_model: Path, adapter: Path) -> dict[str, object]:
    adapter_config = adapter / "adapter_config.json"
    adapter_weights = adapter / "adapter_model.safetensors"
    return {
        "schema_version": 1,
        "base_model": str(base_model.resolve()),
        "base_model_manifest": model_artifact_manifest(base_model),
        "adapter": str(adapter.resolve()),
        "adapter_config_sha256": sha256_file(adapter_config),
        "adapter_model_sha256": sha256_file(adapter_weights),
    }


def merge_model(base_model: Path, adapter: Path, output_dir: Path) -> None:
    # Heavy imports stay below validation so --dry-run works on CPU-only hosts.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        print(f"Loading base model: {base_model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        print(f"Loading adapter: {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        print("Merging adapter into base weights", flush=True)
        model = model.merge_and_unload(safe_merge=True)
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.save_pretrained(temporary)
        provenance = build_provenance(base_model, adapter)
        (temporary / PROVENANCE_NAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_model = args.base_model.resolve()
    adapter = args.adapter.resolve()
    output_dir = args.output_dir.resolve()
    try:
        validate_export_inputs(base_model, adapter, output_dir)
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    provenance = build_provenance(base_model, adapter)
    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter}")
    print(f"Adapter SHA256: {provenance['adapter_model_sha256']}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        print("Dry run only; no model was loaded and no output was created.")
        return 0

    merge_model(base_model, adapter, output_dir)
    print(f"Merge complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
