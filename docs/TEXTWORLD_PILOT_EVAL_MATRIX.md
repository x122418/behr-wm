# TextWorld Pilot Evaluation Matrix

This runbook evaluates the frozen base world model and all six LoRA pilot
checkpoints on the same ordered validation rows:

- `base`
- `behr_step50`, `behr_step100`
- `union_js_step50`, `union_js_step100`
- `union_js_sft_step50`, `union_js_sft_step100`

The runner is resumable. Generation and scoring append per-item JSONL caches,
verified merged models are reused, and each world-model server is placed in its
own process group so cleanup cannot kill unrelated Ray or vLLM jobs.

## 1. Online release gate

Run this on the online GPU instance before scheduling offline H100 work. The
online and offline instances must see the same `LWM_HDD` filesystem.

```bash
export LWM_HDD=/inspire/hdd/project/project-public/czxs253130170
export LWM_MAIN="$LWM_HDD/workspace/LWM_project"
export BEHR_REPO="$LWM_MAIN/repos/behr-wm"
cd "$BEHR_REPO"

bash scripts/run_textworld_pilot_eval_matrix.sh preflight
bash scripts/run_textworld_pilot_eval_matrix.sh all --dry-run
```

The preflight checks all base/actor files, all six adapters, the validation
parquet, runtime imports, launcher syntax, and the patched vLLM CuMem runtime.
The dry run prints the complete seven-model schedule without writing files.

Use the 4090 to run a two-row end-to-end gate. It serializes all GPU work on
GPU 0 while producing the same merged models that the H100 run will reuse:

```bash
export MERGE_GPUS=0 SERVER_GPUS=0 SCORE_GPUS=0 WM_PORTS=8101
export EVALUATION_ROOT="$LWM_HDD/lwm_runs/evaluation/pilot100_release_gate_2"

bash scripts/run_textworld_pilot_eval_matrix.sh all --limit 2
```

Do not proceed until that command exits zero and writes
`$EVALUATION_ROOT/verification.json` with `"status": "ok"`.

## 2. Offline H100 run

After pulling the exact tested Git commit, use the default three server GPUs
and two scoring GPUs. Runtime files remain under `lwm_runs` and are not tracked
by Git.

```bash
export LWM_HDD=/inspire/hdd/project/project-public/czxs253130170
export LWM_MAIN="$LWM_HDD/workspace/LWM_project"
export BEHR_REPO="$LWM_MAIN/repos/behr-wm"
cd "$BEHR_REPO"

unset MERGE_GPUS SERVER_GPUS SCORE_GPUS WM_PORTS EVALUATION_ROOT
bash scripts/run_textworld_pilot_eval_matrix.sh preflight

nohup bash scripts/run_textworld_pilot_eval_matrix.sh all \
  > "$LWM_HDD/lwm_runs/evaluation/pilot100_unified.runner.log" 2>&1 &
echo $! > "$LWM_HDD/lwm_runs/evaluation/pilot100_unified.runner.pid"
```

Live status:

```bash
tail -f "$LWM_HDD/lwm_runs/evaluation/pilot100_unified.runner.log"
```

If the instance is reclaimed, rerun the same `all` command. Completed merges,
generations, and scores are reused. Existing merged directories are reused only
when their provenance matches the current base model and adapter hashes;
unknown or partial directories cause a hard failure instead of silent reuse.

## 3. Verify and compare

`all` already performs the strict matrix verification. It requires exactly
1,000 successful unique rows for every model, identical item IDs, zero errors,
and finite summary metrics.

Generate the paired comparison report after verification:

```bash
export EVAL_ROOT="$LWM_HDD/lwm_runs/evaluation/pilot100_unified"

PYTHONPATH="$BEHR_REPO" "$LWM_HDD/envs/lwm-grpo/bin/python" \
  scripts/analysis/compare_textworld_pilot_matrix.py \
  --evaluation-root "$EVAL_ROOT" \
  --bootstrap-samples 10000 \
  --seed 42
```

This writes `analysis/comparison.json` and `analysis/comparison.csv`. Every
comparison is paired by `item_id`; `improvement` is oriented so positive is
always better, while `raw_delta` remains `model_mean - base_mean`.

## 4. Individual resumable stages

For diagnosis or controlled scheduling, replace `all` with one of:

```text
preflight  merge  generate  score  verify
```

Never delete a partial cache merely to resume. Fix the underlying problem and
rerun its stage; the evaluator skips item IDs already present in its JSONL.
