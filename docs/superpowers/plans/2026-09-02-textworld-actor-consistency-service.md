# TextWorld Actor-Distribution Consistency Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-model GPU service that computes exact and union-top-k actor-distribution consistency rewards and connect it to TextWorld GRPO without changing the original BehR baseline.

**Architecture:** Move scientific-critical prompt alignment and distribution metrics into small shared reward modules, then place a thin FastAPI service around one frozen Qwen3-8B scorer engine. TextWorld `compute_score` selects the unchanged vLLM BehR client or the new consistency client by explicit reward mode; full logits never cross HTTP.

**Tech Stack:** Python 3.10, PyTorch, Transformers, FastAPI/Pydantic, Uvicorn, requests, unittest, verl/Ray, Bash.

**Spec:** `docs/superpowers/specs/2026-09-02-textworld-actor-consistency-service-design.md`

## Global Constraints

- Do not change the WebShop reward path or the original `reward_mode=cauchy` behavior.
- Do not modify vLLM internals or send full-vocabulary logits over HTTP.
- The scorer process owns exactly one tokenizer and one frozen actor model.
- The first version serializes inference and does not implement cross-request dynamic batching or a full-logit cache.
- Use float32 for probabilities and divergence calculations.
- Never silently fall back from a JS objective to original BehR.
- CPU tests must not load a real checkpoint or require a GPU.
- Use repository-local Git identity only if the user supplies it; never modify global Git configuration.

---

### Task 1: Shared TextWorld Teacher-Forcing Inputs

**Files:**
- Create: `src/reward/textworld_actor_inputs.py`
- Modify: `src/data/evaluate_textworld_transition_baseline.py`
- Test: `tests/test_textworld_actor_inputs.py`
- Modify test: `tests/test_evaluate_textworld_transition_baseline.py`

**Interfaces:**
- Produces: `build_teacher_forced_actor_inputs(tokenizer, history, observation, expert_action) -> tuple[torch.Tensor, torch.Tensor]`.
- The first tensor contains causal-LM input IDs ending immediately before the final action token; the second contains all logged action token IDs.
- Later scorer tasks consume this exact function.

- [ ] **Step 1: Write failing shared-input tests**

Create a recording character tokenizer and assert the existing TextWorld role conversion, thinking-disabled chat template, final-token shift, empty-action rejection, and deterministic action IDs:

```python
def test_builds_textworld_teacher_forced_inputs():
    tokenizer = RecordingCharacterTokenizer()
    history = [
        {"role": "system", "content": "initial room"},
        {"role": "user", "content": "open door"},
        {"role": "assistant", "content": "door opens"},
    ]
    model_ids, action_ids = build_teacher_forced_actor_inputs(
        tokenizer, history, "current room", "go east"
    )
    assert tokenizer.messages[1:] == [
        {"role": "user", "content": "initial room"},
        {"role": "assistant", "content": "open door"},
        {"role": "user", "content": "door opens"},
        {"role": "user", "content": "current room"},
    ]
    assert action_ids.tolist() == [ord(c) for c in "go east"]
    assert model_ids.numel() + 1 == len("CHAT\nASSISTANT:\ngo east")

def test_rejects_empty_expert_action():
    with self.assertRaisesRegex(ValueError, "expert_action"):
        build_teacher_forced_actor_inputs(tokenizer, [], "room", "")
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_actor_inputs.py' -v
```

Expected: import failure because `src.reward.textworld_actor_inputs` does not exist.

- [ ] **Step 3: Implement the shared input builder**

Create a focused module with:

```python
def build_teacher_forced_actor_inputs(tokenizer, history, observation, expert_action):
    if not isinstance(expert_action, str) or not expert_action.strip():
        raise ValueError("expert_action must be a non-empty string")
    messages = build_textworld_actor_messages(history, observation)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not prompt.endswith("\n"):
        prompt += "\n"
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + expert_action, add_special_tokens=False)
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("tokenization merged the action with its prompt boundary")
    action_ids = full_ids[len(prefix_ids):]
    if not action_ids:
        raise ValueError("expert_action tokenized to zero tokens")
    return (
        torch.tensor(full_ids[:-1], dtype=torch.long),
        torch.tensor(action_ids, dtype=torch.long),
    )
```

Use the exact system prompt and history role conversion currently encoded by
`TextWorldHTTPJudgeAgent._build_prompt_with_action`; do not invent a second
prompt format.

- [ ] **Step 4: Refactor the offline evaluator to use the shared function**

Keep `build_actor_inputs` as a compatibility wrapper:

```python
def build_actor_inputs(tokenizer, history, observation, logged_action):
    return build_teacher_forced_actor_inputs(
        tokenizer, history, observation, logged_action
    )
```

- [ ] **Step 5: Run focused and evaluator tests**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_actor_inputs.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_evaluate_textworld_transition_baseline.py' -v
```

Expected: all pass with no checkpoint load.

- [ ] **Step 6: Commit the independently testable input contract**

```bash
git add src/reward/textworld_actor_inputs.py src/data/evaluate_textworld_transition_baseline.py tests/test_textworld_actor_inputs.py tests/test_evaluate_textworld_transition_baseline.py
git commit -m "refactor: share TextWorld actor input alignment"
```

---

### Task 2: Shared Distribution Metrics and Normalized Reward

**Files:**
- Create: `src/reward/actor_distribution_metrics.py`
- Modify: `src/data/score_textworld_actor_consistency.py`
- Modify: `src/data/evaluate_textworld_transition_baseline.py`
- Create test: `tests/test_actor_distribution_metrics.py`
- Modify test: `tests/test_score_textworld_actor_consistency.py`

**Interfaces:**
- Produces: `compute_actor_distribution_metrics(real_logits, predicted_logits, action_ids, top_ks=(64,)) -> dict[str, float]`.
- Produces: `js_consistency_reward(js_divergence: float) -> float`.
- All logits have shape `[T, V]`; `action_ids` has shape `[T]`.

- [ ] **Step 1: Write failing metric tests**

Cover identity, a hand-computed two-token vocabulary example, union+OTHER,
invalid shapes, invalid top-k, non-finite logits, and reward normalization:

```python
def test_js_reward_maps_bounds_to_one_and_zero():
    self.assertAlmostEqual(js_consistency_reward(0.0), 1.0)
    self.assertAlmostEqual(js_consistency_reward(math.log(2.0)), 0.0)

def test_identical_logits_have_zero_divergence():
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    metrics = compute_actor_distribution_metrics(
        logits, logits.clone(), torch.tensor([0]), top_ks=(2,)
    )
    self.assertAlmostEqual(metrics["full_vocab_js"], 0.0, places=7)
    self.assertAlmostEqual(metrics["top2_union_other_js"], 0.0, places=7)
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_actor_distribution_metrics.py' -v
```

Expected: import failure for the new module.

- [ ] **Step 3: Move and harden the existing validated calculations**

Move the existing offline `compute_consistency_metrics` implementation into
the reward module, rename it to the produced interface, and validate before
calculation:

```python
if real_logits.ndim != 2 or predicted_logits.shape != real_logits.shape:
    raise ValueError("real and predicted logits must share [T, V] shape")
if action_ids.ndim != 1 or action_ids.numel() != real_logits.shape[0]:
    raise ValueError("action_ids must align with logits positions")
if not torch.isfinite(real_logits).all() or not torch.isfinite(predicted_logits).all():
    raise ValueError("logits must be finite")
if any(k < 1 or k > real_logits.shape[-1] for k in top_ks):
    raise ValueError("top_k must be within vocabulary size")
```

Implement normalized reward as:

```python
def js_consistency_reward(js_divergence):
    if not math.isfinite(js_divergence) or js_divergence < 0:
        raise ValueError("js_divergence must be finite and non-negative")
    return min(1.0, max(0.0, 1.0 - js_divergence / math.log(2.0)))
```

- [ ] **Step 4: Preserve the old import surface**

In `src/data/score_textworld_actor_consistency.py`, import the new function and
provide:

```python
compute_consistency_metrics = compute_actor_distribution_metrics
```

This keeps existing callers and result field names stable.

- [ ] **Step 5: Run all metric/evaluator tests**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_actor_distribution_metrics.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_score_textworld_actor_consistency.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_evaluate_textworld_transition_baseline.py' -v
```

Expected: all pass and existing fixed numerical expectations remain unchanged.

- [ ] **Step 6: Commit the shared metric core**

```bash
git add src/reward/actor_distribution_metrics.py src/data/score_textworld_actor_consistency.py src/data/evaluate_textworld_transition_baseline.py tests/test_actor_distribution_metrics.py tests/test_score_textworld_actor_consistency.py
git commit -m "refactor: share actor distribution metrics"
```

---

### Task 3: Single-Model Batched Scorer Engine

**Files:**
- Create: `src/reward/textworld_consistency_engine.py`
- Create test: `tests/test_textworld_consistency_engine.py`

**Interfaces:**
- Produces: `TextWorldConsistencyEngine(model, tokenizer, model_name, top_k=64)`.
- Produces: `engine.score(history, real_observation, predicted_observation, expert_action, reward_metric) -> dict[str, Any]`.
- Consumes Task 1 input builder and Task 2 metric/reward functions.

- [ ] **Step 1: Write a failing fake-model batching test**

Use a fake model that records call count and batch size and returns deterministic
`[2, max_T, V]` logits:

```python
result = engine.score(
    history=history,
    real_observation="real room",
    predicted_observation="predicted room",
    expert_action="go east",
    reward_metric="union_topk_other_js",
)
self.assertEqual(fake_model.call_count, 1)
self.assertEqual(fake_model.last_batch_size, 2)
self.assertEqual(result["reward_metric"], "union_topk_other_js")
self.assertIn("top2_union_other_js", result)
self.assertGreaterEqual(result["score"], 0.0)
self.assertLessEqual(result["score"], 1.0)
```

Also test mismatched action IDs, unsupported reward metric, inference exception,
and that `model.eval()` plus `torch.inference_mode()` are used.

- [ ] **Step 2: Run the engine test and verify RED**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_consistency_engine.py' -v
```

Expected: missing engine module.

- [ ] **Step 3: Implement padding and one-call inference**

Build the real and predicted inputs independently, verify action IDs, left-pad
or right-pad consistently with an attention mask, and call the model once:

```python
batch_ids = pad_sequence([real_ids, predicted_ids], batch_first=True,
                         padding_value=tokenizer.pad_token_id)
attention_mask = batch_ids.ne(tokenizer.pad_token_id)
with self._inference_lock, torch.inference_mode():
    outputs = self.model(
        input_ids=batch_ids.to(self.device),
        attention_mask=attention_mask.to(self.device),
        use_cache=False,
    )
```

Right-pad the batch, retain each unpadded input length `L_i`, and select
`outputs.logits[i, L_i - T:L_i, :]` for the `T` causal positions predicting
the logged action. Do not assume the two full prompts have equal lengths.
Convert only the aligned `[T, V]` slices to float32 before invoking Task 2
metrics.

- [ ] **Step 4: Map explicit reward metrics**

```python
if reward_metric == "union_topk_other_js":
    divergence = metrics[f"top{self.top_k}_union_other_js"]
elif reward_metric == "full_vocab_js":
    divergence = metrics["full_vocab_js"]
else:
    raise ValueError(f"unsupported reward_metric: {reward_metric}")
metrics["score"] = js_consistency_reward(divergence)
```

Include resolved model name, action token count, inference seconds, and queue
wait seconds. Never catch model exceptions inside the engine.

- [ ] **Step 5: Run engine and shared scientific tests**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_consistency_engine.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_actor_inputs.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_actor_distribution_metrics.py' -v
```

- [ ] **Step 6: Commit the scorer engine**

```bash
git add src/reward/textworld_consistency_engine.py tests/test_textworld_consistency_engine.py
git commit -m "feat: add batched TextWorld consistency engine"
```

---

### Task 4: FastAPI Consistency Service and Launcher

**Files:**
- Create: `src/reward/textworld_consistency_server.py`
- Create: `scripts/servers/start_textworld_consistency_server.sh`
- Create test: `tests/test_textworld_consistency_server.py`
- Create test: `tests/test_textworld_consistency_server_launcher.py`

**Interfaces:**
- Produces: `create_app(engine=None, loader=None) -> FastAPI`.
- Produces endpoints `GET /health`, `GET /v1/models`, and `POST /v1/behavior-consistency`.
- Launcher requires `--model` and `--gpu`, defaults to port `8002` and top-k `64`.

- [ ] **Step 1: Write failing API tests with an injected fake engine**

Use FastAPI `TestClient` and avoid model loading:

```python
response = client.post("/v1/behavior-consistency", json={
    "history": [{"role": "system", "content": "room"}],
    "real_observation": "real",
    "predicted_observation": "predicted",
    "expert_action": "go east",
    "top_k": 64,
    "reward_metric": "union_topk_other_js",
})
self.assertEqual(response.status_code, 200)
self.assertEqual(response.json()["score"], 0.9)
self.assertEqual(fake_engine.calls, 1)
```

Add 422 tests for empty action/invalid top-k/alignment `ValueError`, 503 before
readiness, and 500 without a numeric score for unexpected inference failure.

- [ ] **Step 2: Run API tests and verify RED**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_consistency_server.py' -v
```

Expected: missing server module.

- [ ] **Step 3: Implement schemas and app factory**

Define strict Pydantic request models with `top_k >= 1`, non-empty strings, and
literal reward metrics. App state owns one engine. Translate `ValueError` to
422 and unexpected exceptions to 500 with a request ID but no fabricated
metrics.

The production loader uses:

```python
tokenizer = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=True,
    dtype=torch.bfloat16, device_map={"": 0}
)
model.eval()
```

- [ ] **Step 4: Write failing launcher dry-run tests**

Assert `--dry-run --model /model --gpu 5` prints port 8002, GPU5, top-k64,
`NO_PROXY=127.0.0.1,localhost`, and creates no files. Assert missing model/GPU
arguments exit 2 before importing Transformers.

- [ ] **Step 5: Implement the launcher and server CLI**

The launcher exports only the selected CUDA device and calls:

```bash
.venv/bin/python -m src.reward.textworld_consistency_server \
  --model "$MODEL" --host 0.0.0.0 --port "$PORT" --top-k "$TOP_K"
```

It checks that model and Python paths exist only after handling `--dry-run`.

- [ ] **Step 6: Run service and launcher tests**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_consistency_server.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_consistency_server_launcher.py' -v
bash -n scripts/servers/start_textworld_consistency_server.sh
```

- [ ] **Step 7: Commit the service boundary**

```bash
git add src/reward/textworld_consistency_server.py scripts/servers/start_textworld_consistency_server.sh tests/test_textworld_consistency_server.py tests/test_textworld_consistency_server_launcher.py
git commit -m "feat: serve TextWorld actor consistency rewards"
```

---

### Task 5: TextWorld GRPO Reward Client

**Files:**
- Modify: `src/reward/behr_reward_textworld.py`
- Modify test: `tests/test_behr_reward_textworld.py` (create if absent)

**Interfaces:**
- Produces: `TextWorldConsistencyHTTPClient(api_url, timeout, top_k)`.
- Produces: `client.compute_reward(predicted_state, real_state, expert_action, history, reward_mode) -> dict[str, Any]`.
- Consumes Task 4 endpoint and maps `union_js` / `full_vocab_js` to service reward metrics.

- [ ] **Step 1: Write failing HTTP-client tests**

Patch `requests.Session.post` and assert exact payload mapping:

```python
result = client.compute_reward(
    predicted_state="pred", real_state="real", expert_action="go east",
    history=history, reward_mode="union_js"
)
self.assertEqual(sent_json["reward_metric"], "union_topk_other_js")
self.assertEqual(sent_json["top_k"], 64)
self.assertEqual(result["score"], 0.98)
```

Test timeout, 500, malformed JSON, non-finite score, and verify each returns
`api_failed=True` without invoking the original BehR client.

- [ ] **Step 2: Run the focused reward tests and verify RED**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_behr_reward_textworld.py' -v
```

Expected: missing consistency client or unsupported mode.

- [ ] **Step 3: Add configuration without altering original defaults**

Extend `PivotGRPOConfig` and `compute_score` keyword arguments:

```python
consistency_api_url: str = "http://127.0.0.1:8002"
consistency_top_k: int = 64
consistency_api_timeout: float = 300.0
```

Retain `reward_mode="cauchy"` and existing judge defaults.

- [ ] **Step 4: Implement explicit backend selection**

Use one branch at the current behavioral-fidelity call site:

```python
if reward_mode in {"union_js", "full_vocab_js"}:
    fidelity_result = consistency_client.compute_reward(...)
else:
    fidelity_result = judge.compute_behavioral_fidelity_reward(...)
```

On JS service failure, apply the configured failure/format penalty and expose
`api_failed`, `api_error`, and requested mode in the returned diagnostics. Do
not call `compute_behavioral_fidelity_reward` in that branch.

- [ ] **Step 5: Add original BehR regression test**

Assert `reward_mode="cauchy"` calls only the existing judge and returns the same
score and diagnostic keys as before this task.

- [ ] **Step 6: Run focused and full reward tests**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_behr_reward_textworld.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

- [ ] **Step 7: Commit the GRPO integration**

```bash
git add src/reward/behr_reward_textworld.py tests/test_behr_reward_textworld.py
git commit -m "feat: add JS consistency reward backend"
```

---

### Task 6: Matched Pilot Configuration

**Files:**
- Modify: `train/run_grpo_textworld_pilot.sh`
- Modify: `train/run_grpo_textworld_smoke.sh`
- Modify test: `tests/test_textworld_pilot_launcher.py`
- Modify test: `tests/test_textworld_smoke_launcher.py`

**Interfaces:**
- Consumes environment `REWARD_MODE` with values `cauchy`, `union_js`, or `full_vocab_js`.
- Consumes `CONSISTENCY_URL` defaulting to `http://127.0.0.1:8002` and `CONSISTENCY_TOP_K` defaulting to `64`.
- Dry-run remains side-effect free.

- [ ] **Step 1: Write failing matched-config tests**

Run dry-run with `REWARD_MODE=union_js` and assert:

```python
self.assertIn("reward_kwargs.reward_mode=union_js", result.stdout)
self.assertIn("reward_kwargs.consistency_api_url=http://127.0.0.1:8002", result.stdout)
self.assertIn("reward_kwargs.consistency_top_k=64", result.stdout)
```

Run default dry-run and assert it remains `cauchy`. Assert an unknown mode exits
2 before creating output.

- [ ] **Step 2: Run launcher tests and verify RED**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_*launcher.py' -v
```

Expected: JS overrides absent or invalid mode accepted.

- [ ] **Step 3: Parameterize the existing verified launcher**

Validate `REWARD_MODE` with a shell `case`, append the consistency URL/top-k
Hydra overrides, and keep all train data, validation data, group size, batch
size, learning rate, seed-sensitive settings, save frequency, and step count
identical across modes.

For non-dry runs, health-check port 8000 for `cauchy` and port 8002 for JS
modes. Set `NO_PROXY` before either health check.

- [ ] **Step 4: Run launcher tests and syntax checks**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_textworld_*launcher.py' -v
bash -n train/run_grpo_textworld_smoke.sh train/run_grpo_textworld_pilot.sh
```

- [ ] **Step 5: Commit matched launch configuration**

```bash
git add train/run_grpo_textworld_smoke.sh train/run_grpo_textworld_pilot.sh tests/test_textworld_smoke_launcher.py tests/test_textworld_pilot_launcher.py
git commit -m "feat: configure matched TextWorld reward pilots"
```

---

### Task 7: Equivalence Probe and End-to-End Verification

**Files:**
- Create: `scripts/probes/compare_textworld_consistency_service.py`
- Create test: `tests/test_compare_textworld_consistency_service.py`
- Modify: `docs/TRAINING.md`

**Interfaces:**
- Probe reads a fixed parquet prefix, compares local offline metrics with HTTP service metrics, and exits non-zero if absolute error exceeds `1e-6`.
- It does not train or modify checkpoints.

- [ ] **Step 1: Write failing probe comparison tests**

Factor a pure comparison function and test exact agreement, tolerance-bound
agreement, missing metrics, and a difference greater than `1e-6`:

```python
compare_metric_rows(
    {"full_vocab_js": 0.1},
    {"full_vocab_js": 0.1000005},
    tolerance=1e-6,
)
with self.assertRaisesRegex(ValueError, "full_vocab_js"):
    compare_metric_rows(
        {"full_vocab_js": 0.1},
        {"full_vocab_js": 0.10001},
        tolerance=1e-6,
    )
```

- [ ] **Step 2: Run probe tests and verify RED**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_compare_textworld_consistency_service.py' -v
```

- [ ] **Step 3: Implement the fixed-sample equivalence probe**

The CLI accepts `--input`, `--limit`, `--service-url`, `--model-path`, and
`--tolerance`. It reuses shared Task 1/2 code for local results, calls the
service for the same rows, prints maximum absolute error per metric, and exits
1 on any failed or missing row.

- [ ] **Step 4: Document exact launch and verification commands**

Add commands for:

```bash
bash scripts/servers/start_textworld_consistency_server.sh \
  --model /DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B --gpu 5

.venv/bin/python scripts/probes/compare_textworld_consistency_service.py \
  --input ../../data/processed/textworld_grpo_task_split_v1/val/pilot.parquet \
  --limit 8 --service-url http://127.0.0.1:8002 \
  --model-path /DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B \
  --tolerance 1e-6

REWARD_MODE=union_js bash train/run_grpo_textworld_smoke.sh
```

State explicitly that the GPU equivalence probe precedes the two-step JS smoke,
and the smoke precedes the 50-step matched pilot.

- [ ] **Step 5: Run the complete CPU verification suite**

```bash
.venv/bin/python -m unittest discover -s tests -v
bash -n scripts/servers/start_textworld_consistency_server.sh
bash -n train/run_grpo_textworld_smoke.sh train/run_grpo_textworld_pilot.sh
git diff --check
```

Expected: all CPU tests pass; no model loads and no GPU is required.

- [ ] **Step 6: When one scorer GPU is available, run the equivalence probe**

Start the service, run eight fixed validation rows, and require every listed
metric to agree within `1e-6`. Record service model path, dtype, GPU, and output
under `outputs/probes/textworld_consistency_service/`.

- [ ] **Step 7: When the GRPO GPU allocation is available, run two-step JS smoke**

Use the existing 64-row train smoke and task-disjoint validation pilot. Verify
two optimizer steps, finite rewards/advantages, no scorer API failures, and a
saved log. Do not begin the 50-step pilot if any acceptance check fails.

- [ ] **Step 8: Commit probe and documentation**

```bash
git add scripts/probes/compare_textworld_consistency_service.py tests/test_compare_textworld_consistency_service.py docs/TRAINING.md
git commit -m "test: verify consistency service equivalence"
```

---

## Final acceptance checkpoint

- [ ] Re-run the complete CPU test suite and record the exact pass count.
- [ ] Confirm `reward_mode=cauchy` dry-run matches the original BehR backend.
- [ ] Confirm JS dry-run names port 8002 and never names the vLLM BehR endpoint as its scorer.
- [ ] Confirm the scorer process owns one actor model and full logits never appear in HTTP responses.
- [ ] Confirm fixed-row offline/service differences are at most `1e-6`.
- [ ] Confirm the two-step JS GRPO smoke has finite rewards and zero scorer API failures.
- [ ] Only then schedule matched 50-step `cauchy` and `union_js` pilots.
