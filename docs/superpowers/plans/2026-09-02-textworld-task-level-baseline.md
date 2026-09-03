# TextWorld Task-Level Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible small-scale TextWorld SFT baseline for `SR_Real`, `SR_WM`, `SR_W2R`, aggregate `CR`, and pairwise `CR_pw`, then make the same pipeline resumable for all 200 bundled BehR evaluation tasks.

**Architecture:** Keep the frozen Qwen3-8B actor and Qwen2.5-7B SFT world model as separate OpenAI-compatible services. Run the same task IDs through the real environment and the world model, replay WM actions in the real environment, then join results by task ID with the existing pairwise analyzer. Add only the missing TextWorld launcher plumbing and bounded-sample controls; do not change reward or model code.

**Tech Stack:** Bash, Python 3.10, AgentGym/agentenv, TextWorld, vLLM, existing BehR evaluation scripts.

**Spec:** `docs/EVALUATION.md` and `eval/02_task_success_rate/README.md`, constrained to TextWorld and the SFT checkpoint.

## Global Constraints

- Do not run the full 200-task evaluation until a 2-task end-to-end smoke passes.
- Use the same frozen Qwen3-8B actor for Real and WM evaluation.
- Use `models/WorldModel-Textworld-Qwen2.5-7B` as the first WM baseline.
- Preserve task IDs across Real, WM, and W2R so `CR_pw` is a true paired metric.
- Do not modify reward functions, install global packages, or stop other users' GPU processes.
- Every launcher must support bounded samples, low concurrency, resumable outputs, and `--dry-run` where practical.

---

### Task 1: Restore and verify the TextWorld evaluation runtime

**Files:**
- Inspect: `scripts/env_setup/install_env.sh`
- Inspect: `AgentGym/agentenv/`
- Create only if needed: a project-local evaluation environment or restore missing editable packages in `.venv`
- Test: `tests/test_textworld_evaluation_preflight.py`

**Interfaces:**
- Consumes: bundled `data/init_contexts/textworld/{agent,wm}_instruct_test.json`
- Produces: importable `agentenv.controller`, `agentenv.envs.TextworldTask`, a resolvable TextWorld server module, and the exact games directory used by all real-environment stages

- [ ] **Step 1: Write a failing preflight test** that asserts the two 200-task init-context files have matching IDs and that the configured TextWorld games directory exists and is non-empty.
- [ ] **Step 2: Run** `.venv/bin/python -m unittest tests/test_textworld_evaluation_preflight.py -v` and record the missing import/assets as the expected failure.
- [ ] **Step 3: Restore only project-local AgentGym/agentenv dependencies and TextWorld game assets** using the repository installer or upstream asset source; do not alter the global Python environment.
- [ ] **Step 4: Run import checks** for `TextworldTask`, the TextWorld server module, and one server reset against task ID 0.
- [ ] **Step 5: Re-run the preflight test** and require PASS.

### Task 2: Add a bounded TextWorld Real baseline launcher

**Files:**
- Create: `eval/02_task_success_rate/run_real_textworld.sh`
- Test: `tests/test_textworld_task_success_launchers.py`

**Interfaces:**
- Consumes: actor URL/model, TextWorld test IDs, games directory, `NUM_EXAMPLES`, `MAX_CONCURRENCY`, and `MAX_ROUND`
- Produces: `outputs/task_success_rate/real/textworld/<experiment>/textworld_<id>.json`

- [ ] **Step 1: Write a failing dry-run test** asserting `task_name=textworld`, the bundled TextWorld test file, the games directory, and `num_examples=2` are present in the rendered command.
- [ ] **Step 2: Run the launcher test** and require FAIL because the launcher does not exist.
- [ ] **Step 3: Implement the minimal launcher** by starting the TextWorld server, waiting on health with a bounded timeout, invoking `interact_with_real.py`, and trapping only its own server PID.
- [ ] **Step 4: Re-run the launcher test** and require PASS.
- [ ] **Step 5: Run two Real tasks** with Qwen3-8B, concurrency 1, deterministic decoding, and verify two valid result JSON files.

### Task 3: Add bounded WM and W2R launch controls

**Files:**
- Modify: `eval/02_task_success_rate/run_wm.sh`
- Modify: `eval/02_task_success_rate/run_wm2real.sh`
- Modify: `eval/02_task_success_rate/cal_wm2real.py`
- Test: `tests/test_textworld_task_success_launchers.py`

**Interfaces:**
- Consumes: the same first `N` task IDs used by Real evaluation
- Produces: paired WM files and W2R files under one experiment directory

- [ ] **Step 1: Add failing tests** for `N_SAMPLES=2`, low concurrency, configurable max steps, and W2R replay restricted to exactly the selected WM files.
- [ ] **Step 2: Run the tests** and require the new assertions to fail.
- [ ] **Step 3: Pass `N_SAMPLES` through `run_wm.sh` to the existing `--n_samples` option**, without changing the default full-run behavior.
- [ ] **Step 4: Add an optional replay limit or explicit manifest to `cal_wm2real.py`** so smoke evaluation cannot accidentally replay stale/full outputs.
- [ ] **Step 5: Re-run launcher/unit tests** and require PASS.

### Task 4: Execute the two-task paired baseline smoke

**Files:**
- Output: `outputs/task_success_rate/real/textworld/sft_qwen3_8b_smoke/`
- Output: `outputs/task_success_rate/wm/textworld/sft_qwen3_8b_smoke/`
- Output: `outputs/task_success_rate/wm/textworld/sft_qwen3_8b_smoke/valid_on_real_env/`
- Output: `outputs/task_success_rate/textworld_sft_smoke_metrics.json`

**Interfaces:**
- Consumes: Tasks 1–3 and healthy actor/WM services
- Produces: paired two-task `SR_Real`, `SR_WM`, `SR_W2R`, `CR`, and `CR_pw`

- [ ] **Step 1: Verify GPU ownership and actor/WM health** without touching unrelated processes.
- [ ] **Step 2: Run Real for exactly two task IDs** and validate JSON schema plus zero API errors.
- [ ] **Step 3: Run WM for the same two IDs** and validate saved action trajectories.
- [ ] **Step 4: Replay only those trajectories in TextWorld** and validate task-ID equality across all three result sets.
- [ ] **Step 5: Run `analyze_pairwise_cr.py`** with explicit Real, WM, and W2R directories and save the JSON report.

### Task 5: Promote the verified pipeline to the 200-task SFT baseline

**Files:**
- Output: the same directory layout with experiment name `sft_qwen3_8b_full`
- Document: `docs/EVALUATION.md`

**Interfaces:**
- Consumes: the smoke-verified commands and all 200 matching task IDs
- Produces: the SFT row used as the denominator/reference for later BehR and union-top-k JS comparisons

- [ ] **Step 1: Record smoke resource usage and choose conservative concurrency** from observed actor, WM, CPU RAM, and environment-server utilization.
- [ ] **Step 2: Run/resume Real over 200 tasks** and require 200 valid JSON files with zero silent omissions.
- [ ] **Step 3: Run/resume WM and W2R over the identical 200 IDs** and reject any ID-set mismatch before analysis.
- [ ] **Step 4: Save final metrics and a contingency table** containing both-success, Real-only, W2R-only, and both-fail counts.
- [ ] **Step 5: Run the full unit suite** with `.venv/bin/python -m unittest discover -s tests -v` and document the exact commands and model provenance.
