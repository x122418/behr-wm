# TextWorld Union-JS Auxiliary SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible auxiliary teacher-forced next-observation SFT loss to the stable TextWorld Union-JS GRPO path.

**Architecture:** Keep SFT tensor/loss logic in `src/training`, and carry a narrow, version-guarded downstream patch for VERL 0.7.1 that calls it from the legacy FSDP PPO actor. The existing GRPO rollout and frozen Qwen3-8B scorer remain unchanged; one optimizer update receives `GRPO + KL + lambda_sft * SFT`.

**Tech Stack:** Python 3.10, PyTorch/FSDP, VERL 0.7.1, Hugging Face tokenizers, unittest, Bash, vLLM.

**Spec:** `docs/superpowers/specs/2026-09-13-textworld-union-js-sft-aux-design.md`

## Global Constraints

- Only the world model receives gradients; the reference actor remains frozen.
- Do not modify global Python, CUDA, or drivers.
- Do not leave manual `.venv` edits: every VERL change is represented by a checked-in patch and installer.
- `actor.sft_loss_coef` defaults to `0.0`, preserving current BehR and pure Union-JS behavior.
- First pilot: `lambda_sft=0.1`, `beta=0.001`, top-64+OTHER JS, `G=4`, temperature `0.7`, 1,000 train rows, 1,000 task-disjoint validation rows, 50 steps.
- Save only step 50.

---

### Task 1: Project-owned auxiliary SFT helpers

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/textworld_sft_aux.py`
- Create: `tests/test_textworld_sft_aux.py`

**Interfaces:**
- Consumes padded GRPO prompts, prompt masks, `reward_model["ground_truth"]`, tokenizer, and target length limit.
- Produces `build_aux_sft_batch -> dict[str, torch.Tensor]`, `token_mean_nll`, and `combine_actor_losses`.

- [ ] **Step 1: Write failing mask/padding tests**

```python
class FakeTokenizer:
    eos_token_id, pad_token_id = 99, 0
    def encode(self, text, add_special_tokens=False):
        return {"real one": [11, 12], "real two": [21]}[text]

def test_build_aux_sft_batch_masks_prompt_and_target_padding():
    result = build_aux_sft_batch(
        prompts=torch.tensor([[0, 7, 8], [5, 6, 7]]),
        prompt_attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        ground_truths=["real one", "real two"],
        tokenizer=FakeTokenizer(), max_response_length=4,
    )
    assert result["responses"].tolist() == [[11, 12, 99, 0], [21, 99, 0, 0]]
    assert result["response_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]

def test_rollout_duplication_does_not_change_token_mean_loss():
    base = token_mean_nll(LOG_PROBS, MASK)
    repeated = token_mean_nll(
        LOG_PROBS.repeat_interleave(4, 0), MASK.repeat_interleave(4, 0)
    )
    torch.testing.assert_close(repeated, base)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python tests/test_textworld_sft_aux.py`

Expected: FAIL because `src.training.textworld_sft_aux` does not exist.

- [ ] **Step 3: Implement the minimal helpers**

```python
def token_mean_nll(log_probs, response_mask):
    return -(log_probs * response_mask).sum() / response_mask.sum()

def combine_actor_losses(*, pg_loss, kl_loss, kl_coef, sft_loss, sft_coef):
    return pg_loss + kl_coef * kl_loss + sft_coef * sft_loss
```

Implement `build_aux_sft_batch` with keyword-only arguments `prompts`, `prompt_attention_mask`, `ground_truths`, `tokenizer`, and `max_response_length`. Append EOS to every real observation, right-pad targets, preserve left-padded prompts, build position IDs using VERL's `compute_position_id_with_mask`, reject an empty target mask, and retain EOS when truncating a target.

- [ ] **Step 4: Run focused regressions**

```bash
.venv/bin/python tests/test_textworld_sft_aux.py
.venv/bin/python tests/test_prepare_textworld_grpo_data.py
.venv/bin/python tests/test_textworld_actor_inputs.py
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training tests/test_textworld_sft_aux.py
git commit -m "feat: build auxiliary TextWorld SFT batches"
```

---

### Task 2: Version-locked VERL downstream patch

**Files:**
- Create: `patches/verl-0.7.1-textworld-sft-aux.patch`
- Create: `scripts/install_verl_sft_aux_patch.py`
- Create: `tests/test_install_verl_sft_aux_patch.py`
- Patch target: `.venv/lib/python3.10/site-packages/verl/workers/config/actor.py`
- Patch target: `.venv/lib/python3.10/site-packages/verl/workers/fsdp_workers.py`
- Patch target: `.venv/lib/python3.10/site-packages/verl/workers/actor/dp_actor.py`

**Interfaces:**
- Consumes Task 1 helpers.
- Produces `ActorConfig.sft_loss_coef: float = 0.0`, a `tokenizer=None` constructor argument on `DataParallelPPOActor`, metrics `actor/sft_loss`, `actor/sft_loss_coef`, `actor/total_loss`, and installer CLI `--check|--apply`.

- [ ] **Step 1: Write failing installer tests on a temporary fake VERL tree**

```python
def test_check_rejects_wrong_verl_version():
    result = run_installer("--check", version="0.7.0")
    assert result.returncode != 0
    assert "requires VERL 0.7.1" in result.stderr

def test_apply_is_idempotent_and_check_passes_after_apply():
    assert run_installer("--apply", version="0.7.1").returncode == 0
    assert run_installer("--apply", version="0.7.1").returncode == 0
    assert run_installer("--check", version="0.7.1").returncode == 0
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python tests/test_install_verl_sft_aux_patch.py`

Expected: FAIL because installer and patch are absent.

- [ ] **Step 3: Implement guarded installer**

Support exactly one of `--check` and `--apply`, plus optional `--site-packages PATH` for tests. Require VERL `0.7.1`; verify each unpatched or patched context marker occurs exactly once; write atomically; make repeated `--apply` a no-op; make `--check` compile all three targets.

- [ ] **Step 4: Create and apply the narrow patch**

The patch must only:

1. add and validate nonnegative `sft_loss_coef`;
2. pass `self.tokenizer` into `DataParallelPPOActor`;
3. retain `reward_model` only when the coefficient is positive;
4. build auxiliary tensors from prompt tensors and real observations before micro-batch splitting;
5. perform a second `_forward_micro_batch`, calculate token-mean NLL, and combine losses before one backward call;
6. skip all SFT work when coefficient is zero;
7. aggregate the three new metrics with the existing `loss_scale_factor` convention.

```bash
.venv/bin/python scripts/install_verl_sft_aux_patch.py --apply
.venv/bin/python scripts/install_verl_sft_aux_patch.py --check
```

- [ ] **Step 5: Run tests and compile patched targets**

```bash
.venv/bin/python tests/test_install_verl_sft_aux_patch.py
.venv/bin/python tests/test_textworld_sft_aux.py
.venv/bin/python -m py_compile .venv/lib/python3.10/site-packages/verl/workers/config/actor.py .venv/lib/python3.10/site-packages/verl/workers/fsdp_workers.py .venv/lib/python3.10/site-packages/verl/workers/actor/dp_actor.py
```

Expected: all exit 0.

- [ ] **Step 6: Commit**

```bash
git add patches scripts/install_verl_sft_aux_patch.py tests/test_install_verl_sft_aux_patch.py
git commit -m "feat: patch VERL for auxiliary SFT updates"
```

---

### Task 3: Prove loss integration without GPUs

**Files:**
- Create: `tests/test_verl_sft_aux_integration.py`

**Interfaces:**
- Consumes patched `ActorConfig` and Task 1 loss helper.
- Produces regression proof for the zero and positive coefficient paths.

- [ ] **Step 1: Write integration assertions**

```python
def test_zero_sft_coefficient_preserves_existing_total_loss():
    total = combine_actor_losses(pg_loss=torch.tensor(2.0), kl_loss=torch.tensor(3.0),
        kl_coef=0.001, sft_loss=torch.tensor(5.0), sft_coef=0.0)
    torch.testing.assert_close(total, torch.tensor(2.003))

def test_positive_coefficient_adds_weighted_sft_loss():
    total = combine_actor_losses(pg_loss=torch.tensor(2.0), kl_loss=torch.tensor(3.0),
        kl_coef=0.001, sft_loss=torch.tensor(5.0), sft_coef=0.1)
    torch.testing.assert_close(total, torch.tensor(2.503))
```

Also assert `ActorConfig` accepts `0.0` and `0.1` but rejects a negative value, and patched source contains the metric names.

- [ ] **Step 2: Verify the test fails for any missing marker**

Run: `.venv/bin/python tests/test_verl_sft_aux_integration.py`

- [ ] **Step 3: Make only the minimal correction needed by failing assertions**

Keep the public key exactly `actor_rollout_ref.actor.sft_loss_coef`; add no additional optimizer modes.

- [ ] **Step 4: Run all TextWorld CPU tests**

Run: `for f in tests/test_textworld_*.py tests/test_behr_reward_textworld.py; do .venv/bin/python "$f" || exit 1; done`

Expected: every file exits 0.

- [ ] **Step 5: Commit**

```bash
git add tests/test_verl_sft_aux_integration.py
git commit -m "test: verify auxiliary SFT loss integration"
```

---

### Task 4: Reproducible auxiliary-SFT launcher

**Files:**
- Modify: `train/run_grpo_textworld_smoke.sh`
- Create: `train/run_grpo_textworld_sft_aux_pilot.sh`
- Modify: `tests/test_textworld_smoke_launcher.py`
- Create: `tests/test_textworld_sft_aux_pilot_launcher.py`
- Modify: `docs/TRAINING.md`

**Interfaces:**
- Consumes patched key and installer `--check`.
- Produces `SFT_LOSS_COEF`, the already approved `ROLLOUT_TEMPERATURE`, and a matched 50-step launcher.

- [ ] **Step 1: Write failing launcher tests**

```python
def test_aux_pilot_forwards_matched_configuration():
    result = run_aux_launcher("--dry-run")
    assert result.returncode == 0
    for expected in ["sft_loss_coef=0.1", "rollout.n=4",
                     "rollout.temperature=0.7", "reward_mode=union_js",
                     "total_training_steps=50", "save_freq=50"]:
        assert expected in result.stdout
```

Also prove a non-dry run fails before Ray initialization when installer `--check` fails.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python tests/test_textworld_smoke_launcher.py
.venv/bin/python tests/test_textworld_sft_aux_pilot_launcher.py
```

- [ ] **Step 3: Forward the coefficient in the base launcher**

Add `SFT_LOSS_COEF="${SFT_LOSS_COEF:-0.0}"` and `"++actor_rollout_ref.actor.sft_loss_coef=${SFT_LOSS_COEF}"`. For nonzero coefficient, run installer `--check` before scorer health checks or Ray initialization.

- [ ] **Step 4: Implement the matched wrapper**

Export defaults `REWARD_MODE=union_js`, `SFT_LOSS_COEF=0.1`, `GROUP_SIZE=4`, `ROLLOUT_TEMPERATURE=0.7`, `TOTAL_STEPS=50`, `SAVE_FREQ=50`, and `VAL_FREQ=10`. Use experiment/output name `textworld_unionjs_k64_g4_t07_sft01_pilot1k_50steps_seed42`.

- [ ] **Step 5: Document exact patch, scorer, smoke, and pilot commands**

Explain in `docs/TRAINING.md` that `actor/sft_loss` is a training metric while held-out EM/JS come from the common evaluator.

- [ ] **Step 6: Run launcher regressions**

```bash
.venv/bin/python tests/test_textworld_smoke_launcher.py
.venv/bin/python tests/test_textworld_pilot_launcher.py
.venv/bin/python tests/test_textworld_sft_aux_pilot_launcher.py
bash -n train/run_grpo_textworld_smoke.sh
bash -n train/run_grpo_textworld_sft_aux_pilot.sh
```

Expected: all exit 0 and dry-run creates no output.

- [ ] **Step 7: Commit**

```bash
git add train tests/test_textworld_smoke_launcher.py tests/test_textworld_sft_aux_pilot_launcher.py docs/TRAINING.md
git commit -m "feat: launch Union-JS GRPO with auxiliary SFT"
```

---

### Task 5: GPU gates, pilot, and common evaluation

**Files:**
- Runtime: `outputs/checkpoints/textworld_unionjs_k64_g4_t07_sft01_pilot1k_50steps_seed42/`
- Runtime: `outputs/merged_models/textworld_unionjs_k64_g4_t07_sft01_pilot50/`
- Runtime: `outputs/evaluation/textworld_unionjs_k64_g4_t07_sft01_pilot50_val_pilot1000/`

**Interfaces:**
- Consumes Task 4 launcher, frozen actor scorer, VERL merger, and common evaluator.
- Produces final checkpoint and `summary.json` comparable to SFT, BehR, and pure Union-JS.

- [ ] **Step 1: Verify resources**

Run `nvidia-smi` and `df -h /DATA/disk1 /tmp`. Require two training GPUs, at least 20GB free for scorer, and at least 70GB disk headroom.

- [ ] **Step 2: One-step GPU probe**

Start scorer, then run:

```bash
REWARD_MODE=union_js SFT_LOSS_COEF=0.1 GROUP_SIZE=4 ROLLOUT_TEMPERATURE=0.7 TOTAL_STEPS=1 SAVE_FREQ=-1 VAL_FREQ=-1 OUTPUT_DIR=outputs/checkpoints/textworld_unionjs_sft_aux_probe bash train/run_grpo_textworld_smoke.sh
```

Require finite PG/KL/SFT/total loss and gradients, zero scorer failures, and exit 0.

- [ ] **Step 3: Two-step smoke**

Repeat with `TOTAL_STEPS=2` and a new directory. Require two completed steps, finite metrics, no OOM, and no scorer errors.

- [ ] **Step 4: Matched 50-step pilot**

Run `bash train/run_grpo_textworld_sft_aux_pilot.sh`. Record validation JS/KL/BehR, response length, scorer failure rate, and each loss at steps 10/20/30/40/50. Stop on non-finite loss or persistent scorer failure.

- [ ] **Step 5: Verify and merge step 50**

Require both FSDP rank shards and `latest_checkpointed_iteration.txt`, then run:

```bash
PYTHONPATH=. .venv/bin/python -m verl.model_merger merge --backend fsdp --local_dir outputs/checkpoints/textworld_unionjs_k64_g4_t07_sft01_pilot1k_50steps_seed42/global_step_50/actor --target_dir outputs/merged_models/textworld_unionjs_k64_g4_t07_sft01_pilot50
```

Verify the merged config loads locally.

- [ ] **Step 6: Common deterministic 1,000-row evaluation**

Serve the merged model, then run:

```bash
OUTPUT_DIR=outputs/evaluation/textworld_unionjs_k64_g4_t07_sft01_pilot50_val_pilot1000 ACTOR_GPU=<different-free-gpu> LIMIT=1000 CONCURRENCY=8 bash scripts/evaluate_textworld_validation_baseline.sh
```

Require `total=successful=1000`, `errors=0`, 1,000 unique IDs, and the same input parquet and ID set as existing pilot results.

- [ ] **Step 7: Report and release services**

Compare EM, BehR reward, full-vocabulary KL/JS, and top-64+OTHER JS across SFT, BehR, stable pure Union-JS, and Union-JS+SFT. Stop only services created here and verify their GPUs are released.
