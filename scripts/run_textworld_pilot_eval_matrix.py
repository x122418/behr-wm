#!/usr/bin/env python3
"""Run the fixed seven-model TextWorld pilot evaluation matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterable, Sequence
from urllib.request import Request, urlopen

from scripts.export_textworld_lora_model import (
    PROVENANCE_NAME,
    build_provenance,
    model_artifact_manifest,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTER = PROJECT_ROOT / "scripts" / "export_textworld_lora_model.py"
EVALUATOR = PROJECT_ROOT / "src" / "data" / "evaluate_textworld_transition_baseline.py"
SERVER_LAUNCHER = PROJECT_ROOT / "scripts" / "servers" / "start_wm_server.sh"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_path: Path
    adapter_path: Path | None


def build_model_matrix(
    base_model: Path, pilot_root: Path, merged_root: Path
) -> list[ModelSpec]:
    specs = [ModelSpec("base", base_model, None)]
    for arm in ("behr", "union_js", "union_js_sft"):
        for step in (50, 100):
            name = f"{arm}_step{step}"
            adapter = (
                pilot_root
                / "checkpoints"
                / arm
                / f"global_step_{step}"
                / "actor"
                / "lora_adapter"
            )
            specs.append(ModelSpec(name, merged_root / name, adapter))
    return specs


def read_unique_jsonl(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSONL file: {path}")
    rows: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = row.get("item_id")
        if not item_id:
            raise ValueError(f"missing item_id in {path}:{line_number}")
        if item_id in rows:
            raise ValueError(f"duplicate item_id {item_id!r} in {path}")
        rows[item_id] = row
    return rows


def _assert_finite(value: object, location: str) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite numeric value at {location}: {value}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{location}[{index}]")


def verify_evaluation_matrix(
    evaluation_root: Path, model_names: Sequence[str], expected_count: int
) -> dict[str, object]:
    reference_ids: set[str] | None = None
    for name in model_names:
        model_dir = evaluation_root / name
        rows = read_unique_jsonl(model_dir / "results.jsonl")
        if len(rows) != expected_count:
            raise ValueError(
                f"{name}: expected {expected_count} unique results, found {len(rows)}"
            )
        failed = [item_id for item_id, row in rows.items() if row.get("status") != "ok"]
        if failed:
            raise ValueError(f"{name}: {len(failed)} result rows are not successful")
        item_ids = set(rows)
        if reference_ids is None:
            reference_ids = item_ids
        elif item_ids != reference_ids:
            raise ValueError(f"{name}: item IDs differ from the base evaluation")

        summary_path = model_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_triplet = (expected_count, expected_count, 0)
        actual_triplet = (
            summary.get("total"),
            summary.get("successful"),
            summary.get("errors"),
        )
        if actual_triplet != expected_triplet:
            raise ValueError(
                f"{name}: invalid summary counts {actual_triplet}, expected {expected_triplet}"
            )
        _assert_finite(summary, f"{name}.summary")
    return {"models": list(model_names), "item_count": expected_count, "status": "ok"}


def _csv_values(raw: str, label: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    return values


def _chunks(values: Sequence[ModelSpec], size: int) -> Iterable[Sequence[ModelSpec]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _runtime_env(gpu: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (
        f":{current_pythonpath}" if current_pythonpath else ""
    )
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def _model_files_exist(model_path: Path) -> bool:
    has_weights = (model_path / "model.safetensors").is_file() or (
        model_path / "model.safetensors.index.json"
    ).is_file()
    return (model_path / "config.json").is_file() and has_weights


def validate_preflight_paths(
    specs: Sequence[ModelSpec],
    base_model: Path,
    actor_model: Path,
    validation_data: Path,
) -> None:
    for label, model_path in (("base", base_model), ("actor", actor_model)):
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"{label} model config missing: {model_path}")
        if not _model_files_exist(model_path):
            raise FileNotFoundError(f"{label} model weights missing: {model_path}")
    if not validation_data.is_file():
        raise FileNotFoundError(f"validation parquet missing: {validation_data}")
    for path in (EXPORTER, EVALUATOR, SERVER_LAUNCHER):
        if not path.is_file():
            raise FileNotFoundError(f"required runtime script missing: {path}")
    for spec in specs:
        if spec.adapter_path is None:
            continue
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            path = spec.adapter_path / filename
            if not path.is_file():
                raise FileNotFoundError(f"{spec.name} adapter file missing: {path}")


def run_preflight_stage(specs: Sequence[ModelSpec], args: argparse.Namespace) -> None:
    validate_preflight_paths(
        specs, args.base_model, args.actor_model, args.validation_data
    )
    checks = (
        ["bash", "-n", str(SERVER_LAUNCHER)],
        [
            sys.executable,
            "-c",
            (
                "import pandas, peft, requests, torch, transformers, verl, vllm; "
                "print('runtime imports OK', torch.__version__, vllm.__version__, "
                "verl.__version__, transformers.__version__, peft.__version__)"
            ),
        ],
        [sys.executable, str(PROJECT_ROOT / "scripts" / "env_setup" / "check_vllm_cumem_runtime.py")],
    )
    for command in checks:
        subprocess.run(command, cwd=PROJECT_ROOT, env=_runtime_env(), check=True)
    print("[preflight] paths, runtime imports, shell syntax, and CuMem guard are OK")


def _merged_model_is_current(spec: ModelSpec, base_model: Path) -> bool:
    if spec.adapter_path is None or not _model_files_exist(spec.model_path):
        return False
    provenance_path = spec.model_path / PROVENANCE_NAME
    if not provenance_path.is_file():
        return False
    actual = json.loads(provenance_path.read_text(encoding="utf-8"))
    return actual == build_provenance(base_model, spec.adapter_path)


def ensure_evaluation_contract(
    output_dir: Path, contract: dict[str, object], *, create: bool
) -> None:
    contract_path = output_dir / "evaluation_contract.json"
    if contract_path.is_file():
        actual = json.loads(contract_path.read_text(encoding="utf-8"))
        if actual != contract:
            raise RuntimeError(f"evaluation contract mismatch: {contract_path}")
        return
    if not create:
        raise RuntimeError(f"evaluation contract missing: {contract_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"refusing non-empty evaluation directory without contract: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _evaluation_contract(spec: ModelSpec, args: argparse.Namespace) -> dict[str, object]:
    if spec.adapter_path is None:
        model_identity: object = model_artifact_manifest(spec.model_path)
    else:
        provenance_path = spec.model_path / PROVENANCE_NAME
        if not provenance_path.is_file():
            raise FileNotFoundError(f"merged model provenance missing: {provenance_path}")
        model_identity = json.loads(provenance_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "model_name": spec.name,
        "model_path": str(spec.model_path.resolve()),
        "model_identity": model_identity,
        "actor_model": model_artifact_manifest(args.actor_model),
        "validation_data": str(args.validation_data.resolve()),
        "validation_data_sha256": sha256_file(args.validation_data),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "limit": args.limit,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "top_ks": args.top_ks,
    }


def _wait_jobs(jobs: list[tuple[str, subprocess.Popen, object]]) -> None:
    failures: list[str] = []
    try:
        for name, process, log_handle in jobs:
            return_code = process.wait()
            log_handle.close()
            if return_code:
                failures.append(f"{name} (exit {return_code})")
    except BaseException:
        for _, process, log_handle in jobs:
            _stop_process_group(process)
            if not log_handle.closed:
                log_handle.close()
        raise
    if failures:
        raise RuntimeError("subprocess failures: " + ", ".join(failures))


def run_merge_stage(
    specs: Sequence[ModelSpec], base_model: Path, merged_root: Path, gpus: Sequence[str]
) -> None:
    merged_root.mkdir(parents=True, exist_ok=True)
    pending: list[ModelSpec] = []
    for spec in specs:
        if spec.adapter_path is None:
            continue
        if spec.model_path.exists():
            if _merged_model_is_current(spec, base_model):
                print(f"[merge] reuse verified model: {spec.name}")
                continue
            raise RuntimeError(
                f"refusing unverified or stale merged directory: {spec.model_path}"
            )
        pending.append(spec)

    log_root = merged_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for wave in _chunks(pending, len(gpus)):
        jobs = []
        for spec, gpu in zip(wave, gpus):
            command = [
                sys.executable,
                str(EXPORTER),
                "--base-model",
                str(base_model),
                "--adapter",
                str(spec.adapter_path),
                "--output-dir",
                str(spec.model_path),
            ]
            log_handle = (log_root / f"{spec.name}.log").open("a", encoding="utf-8")
            print(f"[merge] start {spec.name} on GPU {gpu}")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=_runtime_env(gpu),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            jobs.append((spec.name, process, log_handle))
        _wait_jobs(jobs)


def _server_command(spec: ModelSpec, gpu: str, port: str, args: argparse.Namespace) -> list[str]:
    return [
        "bash",
        str(SERVER_LAUNCHER),
        "--model",
        str(spec.model_path),
        "--port",
        port,
        "--gpu",
        gpu,
        "--len",
        str(args.max_model_len),
        "--gpu-mem-util",
        str(args.server_gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]


def _evaluation_command(
    spec: ModelSpec, args: argparse.Namespace, stage: str, api_base: str
) -> list[str]:
    return [
        sys.executable,
        str(EVALUATOR),
        "--input",
        str(args.validation_data),
        "--output-dir",
        str(args.evaluation_root / spec.name),
        "--actor-model-path",
        str(args.actor_model),
        "--wm-api-base",
        api_base,
        "--limit",
        str(args.limit),
        "--concurrency",
        str(args.concurrency),
        "--max-tokens",
        str(args.max_tokens),
        "--top-ks",
        args.top_ks,
        "--stage",
        stage,
    ]


def _wait_for_server(spec: ModelSpec, port: str, process: subprocess.Popen, timeout: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{spec.name} server exited with {process.returncode}")
        try:
            request = Request(url, headers={"Accept": "application/json"})
            with urlopen(request, timeout=5) as response:
                payload = json.load(response)
            models = payload.get("data") or []
            root = models[0].get("root") if models else None
            if root and Path(root).resolve() == spec.model_path.resolve():
                print(f"[generate] server healthy: {spec.name} at {url}")
                return
            last_error = f"server model root mismatch: {root!r}"
        except Exception as exc:  # readiness failures are retried until timeout
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for {spec.name}: {last_error}")


def _stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_generate_stage(specs: Sequence[ModelSpec], args: argparse.Namespace) -> None:
    args.evaluation_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        ensure_evaluation_contract(
            args.evaluation_root / spec.name,
            _evaluation_contract(spec, args),
            create=True,
        )
    server_log_root = args.evaluation_root / "server_logs"
    server_log_root.mkdir(parents=True, exist_ok=True)
    width = min(len(args.server_gpus), len(args.ports))
    if width < 1:
        raise ValueError("generation requires at least one GPU and port")
    for wave in _chunks(specs, width):
        servers: list[tuple[ModelSpec, str, subprocess.Popen, object]] = []
        try:
            for spec, gpu, port in zip(wave, args.server_gpus, args.ports):
                if not _model_files_exist(spec.model_path):
                    raise FileNotFoundError(f"incomplete model directory: {spec.model_path}")
                log_handle = (server_log_root / f"{spec.name}.log").open(
                    "a", encoding="utf-8"
                )
                process = subprocess.Popen(
                    _server_command(spec, gpu, port, args),
                    cwd=PROJECT_ROOT,
                    env=_runtime_env(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                servers.append((spec, port, process, log_handle))
            for spec, port, process, _ in servers:
                _wait_for_server(spec, port, process, args.server_timeout)

            jobs = []
            for spec, port, _, _ in servers:
                output_dir = args.evaluation_root / spec.name
                log_handle = (output_dir / "generate.log").open("a", encoding="utf-8")
                process = subprocess.Popen(
                    _evaluation_command(
                        spec, args, "generate", f"http://127.0.0.1:{port}"
                    ),
                    cwd=PROJECT_ROOT,
                    env=_runtime_env(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                jobs.append((spec.name, process, log_handle))
            _wait_jobs(jobs)
        finally:
            for _, _, process, log_handle in servers:
                _stop_process_group(process)
                log_handle.close()


def run_score_stage(specs: Sequence[ModelSpec], args: argparse.Namespace) -> None:
    for spec in specs:
        ensure_evaluation_contract(
            args.evaluation_root / spec.name,
            _evaluation_contract(spec, args),
            create=False,
        )
    for wave in _chunks(specs, len(args.score_gpus)):
        jobs = []
        for spec, gpu in zip(wave, args.score_gpus):
            output_dir = args.evaluation_root / spec.name
            if not (output_dir / "generations.jsonl").is_file():
                raise FileNotFoundError(f"missing generation cache for {spec.name}")
            log_handle = (output_dir / "score.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                _evaluation_command(spec, args, "score", "http://127.0.0.1:1"),
                cwd=PROJECT_ROOT,
                env=_runtime_env(gpu),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(f"[score] start {spec.name} on GPU {gpu}")
            jobs.append((spec.name, process, log_handle))
        _wait_jobs(jobs)


def print_dry_run(specs: Sequence[ModelSpec], args: argparse.Namespace) -> None:
    selected = (
        ("preflight", "merge", "generate", "score", "verify")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in selected:
        print(f"Stage: {stage}")
        for spec in specs:
            if stage == "merge" and spec.adapter_path is None:
                print(f"  {spec.name}: reuse base model {spec.model_path}")
            elif stage == "merge":
                print(f"  {spec.name}: {spec.adapter_path} -> {spec.model_path}")
            else:
                print(f"  {spec.name}: {args.evaluation_root / spec.name}")
    print("Dry run only; no directories or subprocesses were created.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "merge", "generate", "score", "verify", "all"),
        default="all",
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--actor-model", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--merge-gpus", default="1,3,5")
    parser.add_argument("--server-gpus", default="1,3,5")
    parser.add_argument("--score-gpus", default="6,7")
    parser.add_argument("--ports", default="8101,8102,8103")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-ks", default="32,64")
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--server-gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for key in ("base_model", "actor_model", "pilot_root", "validation_data", "merged_root", "evaluation_root"):
        setattr(args, key, getattr(args, key).resolve())
    args.merge_gpus = _csv_values(args.merge_gpus, "--merge-gpus")
    args.server_gpus = _csv_values(args.server_gpus, "--server-gpus")
    args.score_gpus = _csv_values(args.score_gpus, "--score-gpus")
    args.ports = _csv_values(args.ports, "--ports")
    if args.limit < 1 or args.concurrency < 1:
        parser.error("--limit and --concurrency must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = build_model_matrix(args.base_model, args.pilot_root, args.merged_root)
    if args.dry_run:
        print_dry_run(specs, args)
        return 0
    try:
        if args.stage in {"preflight", "all"}:
            run_preflight_stage(specs, args)
        if args.stage in {"merge", "all"}:
            run_merge_stage(specs, args.base_model, args.merged_root, args.merge_gpus)
        if args.stage in {"generate", "all"}:
            run_generate_stage(specs, args)
        if args.stage in {"score", "all"}:
            run_score_stage(specs, args)
        if args.stage in {"verify", "all"}:
            for spec in specs:
                ensure_evaluation_contract(
                    args.evaluation_root / spec.name,
                    _evaluation_contract(spec, args),
                    create=False,
                )
            report = verify_evaluation_matrix(
                args.evaluation_root, [spec.name for spec in specs], args.limit
            )
            report_path = args.evaluation_root / "verification.json"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(report, indent=2, sort_keys=True))
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TimeoutError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
