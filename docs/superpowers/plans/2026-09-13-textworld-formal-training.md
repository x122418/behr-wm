# TextWorld Formal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, reproducible launcher and runbook for matched 2,000-step TextWorld BehR, Union-JS, and Union-JS-plus-SFT experiments.

**Architecture:** A small Python preflight owns immutable run manifests and resume validation. The existing smoke launcher gains environment-controlled seed, resume, and checkpoint-retention settings, while a formal wrapper maps one arm name to the only two settings allowed to differ: reward mode and auxiliary SFT coefficient.

**Tech Stack:** Bash, Python 3.10, JSON, unittest, VERL 0.7.1, Hydra, TensorBoard, vLLM, FSDP.

**Spec:** `docs/superpowers/specs/2026-09-13-textworld-formal-training-design.md`

## Global Constraints

- Use `train/full.parquet`, 2,000 steps, batch size 4, `G=4`, temperature 0.7, validation every 250 steps, and saving every 1,000 steps.
- Set both `data.seed` and `actor_rollout_ref.actor.data_loader_seed` to the declared seed; do not claim control of a nonexistent legacy rollout seed.
- Retain at most one actor checkpoint per run.
- Run only one formal arm at a time with two training GPUs and one scorer GPU.
- Never delete an FSDP checkpoint until its merged model and evaluation outputs pass integrity checks.
- Existing smoke and pilot behavior must remain unchanged when new environment variables are omitted.

---

### Task 1: Immutable Formal Run Manifest

**Files:**
- Create: `src/training/textworld_formal_run.py`
- Test: `tests/test_textworld_formal_run.py`

**Interfaces:**
- Consumes: resolved formal-run configuration passed as CLI flags.
- Produces: `prepare_run(output_dir: Path, manifest: dict, resume: bool) -> Path`, which writes or validates `formal_run_manifest.json` and returns its path.

- [ ] **Step 1: Write failing tests**

Cover these real filesystem behaviors:

```python
def test_new_run_writes_canonical_manifest(self): ...
def test_new_run_rejects_nonempty_output(self): ...
def test_resume_requires_checkpoint_tracker(self): ...
def test_resume_rejects_manifest_mismatch(self): ...
def test_resume_accepts_identical_manifest_and_tracker(self): ...
```

The canonical manifest must include schema version, arm, reward mode, SFT coefficient, train and validation paths, seed, group size, rollout temperature, total steps, save/validation frequencies, model paths, and scorer URL.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=. .venv/bin/python tests/test_textworld_formal_run.py
```

Expected: import failure because `textworld_formal_run.py` does not exist.

- [ ] **Step 3: Implement the manifest preflight**

Implement canonical JSON serialization with sorted keys and indentation. For a new run, require an absent or empty output directory, create it, and write the manifest atomically through a sibling temporary file followed by `Path.replace`. For resume, require both an identical existing manifest and `latest_checkpointed_iteration.txt`; report every mismatching key in the exception.

Expose a CLI with explicit flags and mutually exclusive `--new-run` / `--resume`. It prints the resolved manifest path on success and exits nonzero on validation failure.

- [ ] **Step 4: Run tests and compile check**

```bash
PYTHONPATH=. .venv/bin/python tests/test_textworld_formal_run.py
.venv/bin/python -m py_compile src/training/textworld_formal_run.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/textworld_formal_run.py tests/test_textworld_formal_run.py
git commit -m "feat: validate formal TextWorld run manifests"
```

---

### Task 2: Common Seed and Checkpoint Controls

**Files:**
- Modify: `train/run_grpo_textworld_smoke.sh`
- Modify: `tests/test_textworld_smoke_launcher.py`

**Interfaces:**
- Consumes optional environment variables `DATA_SEED`, `ACTOR_DATA_LOADER_SEED`, `MAX_ACTOR_CKPT_TO_KEEP`, and `RESUME_MODE`.
- Produces Hydra overrides `data.seed`, `actor_rollout_ref.actor.data_loader_seed`, `trainer.max_actor_ckpt_to_keep`, and `trainer.resume_mode`.

- [ ] **Step 1: Add failing dry-run tests**

Invoke the smoke launcher with values 42, 42, 1, and `disable`, then assert that all four exact Hydra overrides appear. Add a second assertion that omitted variables produce none of these overrides, preserving the previous VERL defaults.

- [ ] **Step 2: Run the launcher test and verify RED**

```bash
PYTHONPATH=. .venv/bin/python tests/test_textworld_smoke_launcher.py
```

Expected: the four new overrides are absent.

- [ ] **Step 3: Add the environment defaults and command overrides**

Default all four variables to empty and append their Hydra overrides only when explicitly set. Validate a nonempty `RESUME_MODE` against `auto|disable|resume_path` before any output is created. The formal wrapper sets 42, 42, 1, and `disable`; legacy launchers retain their prior defaults.

- [ ] **Step 4: Run tests and shell syntax checks**

```bash
PYTHONPATH=. .venv/bin/python tests/test_textworld_smoke_launcher.py
PYTHONPATH=. .venv/bin/python tests/test_textworld_pilot_launcher.py
bash -n train/run_grpo_textworld_smoke.sh
```

Expected: all tests pass and shell syntax is valid.

- [ ] **Step 5: Commit**

```bash
git add train/run_grpo_textworld_smoke.sh tests/test_textworld_smoke_launcher.py
git commit -m "feat: control TextWorld GRPO reproducibility and retention"
```

---

### Task 3: Matched Three-Arm Formal Launcher

**Files:**
- Create: `train/run_grpo_textworld_formal.sh`
- Create: `tests/test_textworld_formal_launcher.py`

**Interfaces:**
- Consumes `FORMAL_ARM=behr|union_js|union_js_sft`, optional `FORMAL_SEED`, `FORMAL_RESUME`, and common GPU/service environment variables.
- Produces one resolved formal run, its immutable manifest, and an invocation of `run_grpo_textworld_smoke.sh`.

- [ ] **Step 1: Write failing launcher tests**

For each arm, call `--dry-run` and assert common values:

```text
train/full.parquet
data.seed=42
actor_rollout_ref.actor.data_loader_seed=42
actor_rollout_ref.rollout.n=4
actor_rollout_ref.rollout.temperature=0.7
trainer.total_training_steps=2000
trainer.test_freq=250
trainer.save_freq=1000
trainer.max_actor_ckpt_to_keep=1
trainer.resume_mode=disable
```

Assert arm mappings exactly:

```text
behr         -> reward_mode=cauchy,   sft_loss_coef=0.0
union_js     -> reward_mode=union_js, sft_loss_coef=0.0
union_js_sft -> reward_mode=union_js, sft_loss_coef=0.1
```

Also test that an unknown/missing arm fails before creating output, dry-run has no side effects, the launcher source invokes manifest preflight for real runs, and `FORMAL_RESUME=1` maps to resume validation plus `trainer.resume_mode=auto`.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. .venv/bin/python tests/test_textworld_formal_launcher.py
```

Expected: launcher file not found.

- [ ] **Step 3: Implement the formal wrapper**

Map arm settings in a closed `case` statement. Resolve all common settings once, derive a unique output/experiment name containing arm, step count, and seed, print a concise configuration summary, then invoke the manifest CLI for non-dry runs. Finally `exec` the existing smoke launcher with all values exported.

The wrapper must not start or stop services. It checks the applicable scorer through the existing smoke launcher and preserves `--dry-run` behavior without creating directories.

- [ ] **Step 4: Run tests and syntax checks**

```bash
PYTHONPATH=. .venv/bin/python tests/test_textworld_formal_launcher.py
PYTHONPATH=. .venv/bin/python tests/test_textworld_smoke_launcher.py
PYTHONPATH=. .venv/bin/python tests/test_textworld_sft_aux_pilot_launcher.py
bash -n train/run_grpo_textworld_formal.sh
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add train/run_grpo_textworld_formal.sh tests/test_textworld_formal_launcher.py
git commit -m "feat: launch matched formal TextWorld experiments"
```

---

### Task 4: Formal Experiment Runbook and Gates

**Files:**
- Modify: `docs/TRAINING.md`
- Test: `tests/test_textworld_formal_launcher.py`

**Interfaces:**
- Consumes the formal launcher and existing scorer, merger, and evaluator scripts.
- Produces an operator sequence for smoke, training, merge, validation/test evaluation, integrity checks, and checkpoint deletion.

- [ ] **Step 1: Add the runbook**

Document the exact sequential arm order (`behr`, `union_js`, `union_js_sft`), three-GPU allocation, scorer health/provenance check, two-step override, 2,000-step launch, TensorBoard directory, resume command, FSDP merge command, common 1,000-row validation and 1,820-row test commands, result integrity requirements, and deletion gate.

State the expected disk lifecycle: one 43 GB FSDP checkpoint, one 15 GB merged model, at least 130 GB free before a run, and deletion only after successful merge and evaluation.

- [ ] **Step 2: Verify all CPU gates**

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_textworld*.py' -v
PYTHONPATH=. .venv/bin/python tests/test_install_verl_sft_aux_patch.py
PYTHONPATH=. .venv/bin/python tests/test_verl_sft_aux_integration.py
bash -n train/run_grpo_textworld_smoke.sh
bash -n train/run_grpo_textworld_formal.sh
git diff --check
```

Expected: zero failures.

- [ ] **Step 3: Run formal dry-runs for every arm**

```bash
for arm in behr union_js union_js_sft; do
  FORMAL_ARM="$arm" bash train/run_grpo_textworld_formal.sh --dry-run
done
```

Expected: identical common settings and only the declared reward/SFT differences.

- [ ] **Step 4: Commit**

```bash
git add docs/TRAINING.md
git commit -m "docs: add formal TextWorld experiment runbook"
```

- [ ] **Step 5: GPU execution gate**

With three confirmed free H100 GPUs and at least 130 GB free disk, start the applicable frozen-actor scorer on one GPU. Run a fresh two-step `union_js_sft` formal smoke using two other GPUs and a distinct output directory. Require two finite training steps, zero scorer failures, no OOM, and a clean exit before launching a 2,000-step arm.

- [ ] **Step 6: Start only the first formal arm**

Begin with `behr` so the comparison baseline is established before the new method. Record session name, GPUs, scorer endpoint, output directory, and log path. Monitor the first two steps and the first validation before leaving the process unattended.
