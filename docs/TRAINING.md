# Training Guide

This guide documents the verl GRPO configuration used to produce the BehR-WM
checkpoints reported in the paper. We deliberately **do not ship cluster-specific
launch scripts**; instead we provide the reward code plus a reference
invocation that you can adapt to your own environment.

## 1. Prerequisites

- Finish [Installation](INSTALL.md).
- Install the training framework:

  ```bash
  bash scripts/env_setup/install_verl.sh   # or: pip install verl
  ```

- Hardware: 4× A100-80GB (minimum) or 8× A100-80GB (recommended).
- Base world models used in the paper (all on HuggingFace):
  - `X1AOX1A/WorldModel-Webshop-Qwen2.5-7B`
  - `X1AOX1A/WorldModel-Webshop-Llama3.1-8B`
  - `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B`
  - `X1AOX1A/WorldModel-Textworld-Llama3.1-8B`
- Reference Agent: a frozen instruction-tuned LLM exposed via OpenAI-compatible
  HTTP (we use `Qwen/Qwen3-8B` in the paper).

## 2. Architecture Overview

GRPO training co-locates three components:

```
Training node
├── FSDP actor + reference policy           (all GPUs)
├── vLLM rollout                            (shared GPUs, TP=2 for 8-GPU / TP=1 for 4-GPU)
└── Reference Agent HTTP server (judge)     (remote or same-node, queried over HTTP)
```

The BehR reward implementation in [`src/reward/`](../src/reward/) calls the
Reference Agent via HTTP to compute per-token log-probabilities of the logged
action under both the WM-predicted state and the real state.

## 3. Start the Reference Agent server

```bash
# Co-located on training GPUs (tensor-parallel, memory-shared with rollout)
bash scripts/servers/start_reference_agent_server.sh \
     -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3 --shared
```

Health-check:

```bash
curl http://localhost:8000/health
```

## 4. Plug BehR into verl

verl supports user-supplied reward functions through
`custom_reward_function.path` / `custom_reward_function.name`. Point it at the
environment-specific module:

| Environment | Module |
|-------------|--------|
| WebShop | [`src/reward/behr_reward_webshop.py`](../src/reward/behr_reward_webshop.py) |
| TextWorld | [`src/reward/behr_reward_textworld.py`](../src/reward/behr_reward_textworld.py) |

Both export `compute_score(data_source, solution_str, ground_truth, extra_info)`
and return a dict with a `"score"` field plus BehR diagnostics.

### Reward hyper-parameters

| Key | WebShop | TextWorld | Meaning |
|-----|---------|-----------|---------|
| `reward_mode` | `cauchy` (recommended) or `exponential` | same | Shape of the BehR transform |
| `behavior_scale_coef` ($\alpha$) | `1.0` | `1.0` | Sensitivity of BehR to $|\Delta|$ |
| `behavior_weight` | `0.8` (BehR+FactR) / `1.0` (BehR-only) | `1.0` | Weight of BehR in the total reward |
| `facts_weight` | `0.2` / `0.0` | `0.0` | Physical-facts reward (WebShop only) |
| `length_penalty_weight` | `0.0` | `0.0` | Disabled in the final paper setup |
| `format_penalty` | `-2.0` | `-1.0` | Penalty for malformed WM outputs |
| `use_http_judge` | `True` | `True` | Use the Reference Agent HTTP server |
| `judge_api_url` | `http://localhost:8000` | same | Reference Agent endpoint |
| `api_timeout` | `2400.0` | `2400.0` | Seconds |
| `max_workers` | `16` | `8` | Parallel reward-compute workers |

## 5. Reference verl invocation (WebShop, 4× A100)

The command below reproduces the BehR setup used for LLaMA-3.1-8B on WebShop.
Edit `TRAIN_DATA`, `VAL_DATA`, `WORLD_MODEL`, `REWARD_FN_PATH`, and output paths
for your environment.

```bash
TRAIN_DATA=/path/to/webshop_train.parquet
VAL_DATA=/path/to/webshop_test.parquet
WORLD_MODEL=X1AOX1A/WorldModel-Webshop-Llama3.1-8B
REWARD_FN_PATH=$(pwd)/src/reward/behr_reward_webshop.py
JUDGE_URL=http://localhost:8000
OUTPUT_DIR=./outputs/checkpoints/behr_llama_$(date +%Y%m%d_%H%M)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=true

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=32 \
    data.max_prompt_length=14336 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    +data.num_proc=64 \
    data.truncation='left' \
    actor_rollout_ref.model.path=${WORLD_MODEL} \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_seqs=12 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.3 \
    actor_rollout_ref.rollout.top_p=1.00 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=compute_score \
    ++custom_reward_function.reward_kwargs.use_http_judge=True \
    ++custom_reward_function.reward_kwargs.judge_api_url="${JUDGE_URL}" \
    ++custom_reward_function.reward_kwargs.api_timeout=2400.0 \
    ++custom_reward_function.reward_kwargs.max_workers=16 \
    ++custom_reward_function.reward_kwargs.reward_mode="cauchy" \
    ++custom_reward_function.reward_kwargs.behavior_scale_coef=1.0 \
    ++custom_reward_function.reward_kwargs.format_penalty=-2.0 \
    ++custom_reward_function.reward_kwargs.facts_weight=0.2 \
    ++custom_reward_function.reward_kwargs.behavior_weight=0.8 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='behr-wm-webshop' \
    trainer.experiment_name="wm-llama3.1-8b-behr-$(date +%Y%m%d_%H%M)" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=20 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=5 \
    trainer.default_local_dir=${OUTPUT_DIR}
```

### 8-GPU adjustments

Keep all reward-side parameters identical; change only the capacity knobs:

```
actor_rollout_ref.rollout.tensor_model_parallel_size=2
trainer.n_gpus_per_node=8
data.train_batch_size=128
actor_rollout_ref.rollout.n=8
```

### TextWorld adjustments

Swap the reward module and drop the physical-facts term:

```
custom_reward_function.path=$(pwd)/src/reward/behr_reward_textworld.py
++custom_reward_function.reward_kwargs.behavior_weight=1.0
++custom_reward_function.reward_kwargs.facts_weight=0.0
++custom_reward_function.reward_kwargs.format_penalty=-1.0
++custom_reward_function.reward_kwargs.max_workers=8
```

## 6. Data format

Training consumes parquet files whose rows expose the fields documented in
[`src/data/prepare_data.py`](../src/data/prepare_data.py) — essentially a
prompt (agent-visible context), a target next state (real observation), and
enough metadata for BehR to re-construct the logged action. Released dataset
artifacts (see [Release Timeline](../README.md#release-timeline)) ship in this
format.

## 7. Monitoring

- Reward diagnostics are logged per step under `custom_reward_function` keys —
  `score`, `behavior_reward`, `facts_reward`, `mean_log_prob_pred`,
  `mean_log_prob_real`.
- Checkpoints land in `OUTPUT_DIR`; use `scripts/servers/start_wm_server.sh`
  to serve them for evaluation (see [EVALUATION.md](EVALUATION.md)).

## 8. Reproducibility notes

The paper numbers were obtained with:
- verl as of the v0.2 release (pinned revision documented in
  `scripts/env_setup/install_verl.sh`),
- vLLM 0.6.x with `enforce_eager=True` (disables CUDA-graph capture, removes
  a class of non-determinism in rollout),
- temperature 1.3, top-p 1.0, group size 5 (4-GPU) or 8 (8-GPU),
- 5 training epochs, checkpoints saved every 20 steps.

We observed that `cauchy` reward mode converges more stably than `exponential`
at large $|\Delta|$ (better gradient preservation); `exponential` remains a
valid choice for smoother reward landscapes.

## 9. TextWorld actor-distribution consistency reward

The JS reward uses a dedicated frozen-actor service. Full-vocabulary logits
remain inside that process; GRPO workers receive scalar metrics only.

Start one Qwen3-8B scorer on its own GPU:

```bash
bash scripts/servers/start_textworld_consistency_server.sh \
  --model /DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B \
  --gpu 5 \
  --port 8002 \
  --top-k 64
```

Before training, compare the service against locked offline results. This
reuses existing exact metrics and therefore does not load a second actor model:

```bash
NO_PROXY=127.0.0.1,localhost PYTHONPATH=. \
.venv/bin/python scripts/probes/compare_textworld_consistency_service.py \
  --input outputs/evaluation/textworld_sft_val_pilot_1000/results.jsonl \
  --limit 8 \
  --service-url http://127.0.0.1:8002 \
  --top-k 64 \
  --tolerance 1e-6 \
  --output outputs/probes/textworld_consistency_service/equivalence.json
```

Only after equivalence passes, run the two-step JS smoke:

```bash
REWARD_MODE=union_js \
CONSISTENCY_URL=http://127.0.0.1:8002 \
bash train/run_grpo_textworld_smoke.sh
```

The 50-step matched pilot comes last:

```bash
REWARD_MODE=union_js \
CONSISTENCY_URL=http://127.0.0.1:8002 \
bash train/run_grpo_textworld_pilot.sh
```

Use `REWARD_MODE=cauchy` for the original BehR control. Do not start either
50-step pilot unless the service equivalence probe and the corresponding
two-step smoke both pass with finite rewards and zero scorer API failures.

## 10. TensorBoard metrics

The TextWorld smoke and pilot launchers enable both console and TensorBoard
logging. Events are written to `${OUTPUT_DIR}/tensorboard` by default; override
that location with `TENSORBOARD_DIR` when needed.

```bash
tensorboard --logdir outputs/checkpoints --port 6006
```

GRPO training curves include `actor/pg_loss`, `actor/kl_loss`, entropy,
gradient norm, reward, advantage, memory, and throughput metrics. Validation
uses rollout reward metrics rather than a supervised cross-entropy
`val_loss`. Report held-out reward/consistency and downstream EM or
trajectory metrics separately.

### Auxiliary SFT + Union-JS pilot

The pinned VERL 0.7.1 PPO actor needs the repository-owned downstream patch
before it can combine GRPO and teacher-forced real-observation SFT losses:

```bash
PYTHONPATH=. .venv/bin/python scripts/install_verl_sft_aux_patch.py --apply
PYTHONPATH=. .venv/bin/python scripts/install_verl_sft_aux_patch.py --check
```

Start the frozen actor consistency scorer on a dedicated GPU:

```bash
bash scripts/servers/start_textworld_consistency_server.sh \
  --model /DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B \
  --gpu 7 --port 8002 --top-k 64
```

Use two other GPUs for a two-step smoke:

```bash
CUDA_VISIBLE_DEVICES=5,6 N_GPUS=2 \
REWARD_MODE=union_js SFT_LOSS_COEF=0.1 GROUP_SIZE=4 \
ROLLOUT_TEMPERATURE=0.7 TOTAL_STEPS=2 SAVE_FREQ=-1 VAL_FREQ=-1 \
OUTPUT_DIR=outputs/checkpoints/textworld_unionjs_sft_aux_smoke \
bash train/run_grpo_textworld_smoke.sh
```

After the smoke passes, run the matched 50-step pilot with the same GPU
allocation:

```bash
CUDA_VISIBLE_DEVICES=5,6 N_GPUS=2 \
bash train/run_grpo_textworld_sft_aux_pilot.sh
```

`actor/sft_loss`, `actor/pg_loss`, `actor/kl_loss`, and `actor/total_loss` are
training diagnostics. They are not substitutes for the task-disjoint
held-out exact match, full-vocabulary KL/JS, or trajectory metrics.

For a run started with console-only logging, convert its completed log once:

```bash
PYTHONPATH=. .venv/bin/python \
  scripts/analysis/console_log_to_tensorboard.py \
  --input outputs/checkpoints/<run>/logs/train.log \
  --output-dir outputs/checkpoints/<run>/tensorboard
```

## 10. Formal 2,000-step TextWorld comparison

The LoRA formal scale-up uses the full task-disjoint training parquet as a
deterministically shuffled pool. Each trained arm receives the same first
2,000 batches (`data.seed=42`, batch size 4), `G=4`, rollout temperature 0.7,
actor learning rate `5e-6`, and LoRA configuration `r=32`, `alpha=32`,
`target_modules=all-linear`. Run the arms sequentially in this order:

1. `behr`
2. `union_js`
3. `union_js_sft`

The formal launcher enables LoRA by default. The smoke launcher remains
backward compatible: omitting `LORA_RANK` keeps the existing full-parameter
path. All three formal arms must use the same `ACTOR_LR`, `LORA_RANK`,
`LORA_ALPHA`, `LORA_TARGET_MODULES`, and `REWARD_NUM_WORKERS`. These values are
stored in the immutable run manifest, so a resume with a changed value is
rejected. The only intended arm differences are the reward mode and auxiliary
SFT coefficient.

Reserve one otherwise empty H100 GPU for LoRA training plus colocated vLLM
rollout and one H100 for the frozen actor scorer. The one-card VERL allocation
is accepted only after the two-step gate below records finite losses, no OOM,
and an updated LoRA adapter. If that gate does not fit, use two VERL GPUs for
all three arms rather than changing only one arm.
Require at least 130 GB free on `/DATA/disk1` before starting an arm.
The formal launcher also exports `HF_DATASETS_CACHE` to
`/DATA/disk1/huangjiaqi_cache/lwm_hf_datasets` by default. This keeps the
expanded Arrow cache off the nearly full system disk and lets all three arms
reuse one cache. Set `HF_DATASETS_CACHE` explicitly only when moving the run to
another data disk. Ray runtime files and any object-store spill are similarly
kept under `/DATA/disk1/huangjiaqi_cache/lwm_ray` through the launcher's
`RAY_TEMP_DIR` setting.

The formal launcher sets `data.filter_overlong_prompts=False` because the
frozen task-disjoint data was already exhaustively token-audited with the same
world-model tokenizer: all 570,010 training prompts are at most 2,822 tokens
and all validation prompts are at most 3,008 tokens, below the 4,096-token
limit. Legacy smoke/pilot launchers retain runtime filtering by default.

For BehR, start the reference completion server on the scorer GPU:

```bash
bash scripts/servers/start_reference_agent_server.sh \
  -m /DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B \
  -p 8000 -gpu 7 -l 4608 --shared --util 0.30
curl --noproxy '*' -fsS http://127.0.0.1:8000/health
```

For either Union-JS arm, use the consistency scorer instead:

```bash
bash scripts/servers/start_textworld_consistency_server.sh \
  --model /DATA/disk1/huangjiaqi_data/qwen_model/Qwen3-8B \
  --gpu 7 --port 8002 --top-k 64
curl --noproxy '*' -fsS http://127.0.0.1:8002/health
```

The consistency server defaults to a 5 ms request-coalescing window and a
maximum microbatch of 32 requests. Concurrent GRPO samples with identical
history, real observation, and expert action are grouped so the frozen actor
computes the real branch once. Real and predicted actor inputs are then batched
only when their token lengths are exactly equal; the engine never pads a batch,
because changing the padded sequence length causes material BF16/SDPA drift on
the frozen Qwen3 actor. This does not change the reward definition. Set
`--batch-wait-ms 0` to restore the legacy unbatched path. Before a formal run
with a new actor/runtime, compare the batched and legacy paths on locked
examples and require a maximum absolute metric difference of `1e-6` with
identical within-group reward ordering.

JS is non-negative mathematically, but float32 cancellation can produce tiny
negative values when the two actor distributions are nearly identical. The
metric implementation clamps these roundoff artifacts to zero, and the reward
boundary independently accepts negative values only within `1e-6`; larger
negative or non-finite values remain errors. Rejected scorer requests log the
underlying validation reason instead of exposing only an HTTP 422 access line.

Inspect the complete resolved command without creating output:

```bash
FORMAL_ARM=behr CUDA_VISIBLE_DEVICES=6 N_GPUS=1 \
  bash train/run_grpo_textworld_formal.sh --dry-run
```

Before a long run, use the exact formal data, sampling, and seed settings for a
fresh two-step GPU gate while disabling checkpointing and validation:

```bash
FORMAL_ARM=union_js_sft CUDA_VISIBLE_DEVICES=6 N_GPUS=1 \
TOTAL_STEPS=2 SAVE_FREQ=-1 VAL_FREQ=-1 \
OUTPUT_DIR=outputs/checkpoints/textworld_formal_union_js_sft_lora_r32_smoke2_seed42 \
  bash train/run_grpo_textworld_formal.sh
```

Start a formal arm only after the smoke reports two finite steps, no OOM, and
no scorer failure:

```bash
FORMAL_ARM=behr CUDA_VISIBLE_DEVICES=6 N_GPUS=1 \
  bash train/run_grpo_textworld_formal.sh
```

The default run validates every 250 steps, saves every 1,000 steps, and sets
`trainer.max_actor_ckpt_to_keep=1`. To resume the exact same manifest and
latest checkpoint:

```bash
FORMAL_ARM=behr FORMAL_RESUME=1 CUDA_VISIBLE_DEVICES=6 N_GPUS=1 \
  bash train/run_grpo_textworld_formal.sh
```

Do not set `FORMAL_RESUME=1` for a different arm, seed, model, scorer, training
budget, learning rate, reward-worker count, or LoRA configuration; the
immutable manifest rejects such a mismatch.

VERL 0.7.1 does not repopulate its in-memory checkpoint history after a new
process resumes, so its built-in retention limit can leave the loaded
checkpoint beside the newly saved one. After a successful formal launcher
exit, `src/training/textworld_checkpoint_retention.py` reads the atomic latest
checkpoint tracker and removes only older `global_step_N` directories. It
refuses to delete anything when the manifest or tracker is invalid, the
tracked checkpoint is missing, or a directory is newer than the tracker.

`reward.num_workers=8` is explicit in the launcher. This is the Ray reward
manager concurrency used by VERL. The old
`custom_reward_function.reward_kwargs.max_workers=4` override was not consumed
by the TextWorld reward function and has been removed; this makes the existing
effective concurrency explicit rather than changing the reward computation.

After step 2,000, verify every expected FSDP rank shard and merge the actor:

```bash
PYTHONPATH=. .venv/bin/python -m verl.model_merger merge --backend fsdp \
  --local_dir outputs/checkpoints/textworld_formal_<arm>_lora_r32_steps2000_seed42/global_step_2000/actor \
  --target_dir outputs/merged_models/textworld_formal_<arm>_lora_r32_steps2000_seed42
```

For a LoRA checkpoint, also require a loadable
`global_step_2000/actor/lora_adapter/adapter_model.safetensors` and matching
`adapter_config.json`. The installed VERL merger can export a
`lora_adapter/` directory, but the repository's current world-model server
accepts only one model path. Therefore, do not start the full evaluation or
delete checkpoint shards until the two-step export gate has verified either a
PEFT `merge_and_unload()` export or an explicitly LoRA-enabled vLLM launch.
LoRA reduces trainable optimizer state, but a VERL FSDP resume checkpoint may
still contain base-model shards; keep `trainer.max_actor_ckpt_to_keep=1` and
continue monitoring disk usage.

Serve the merged model on port 8001, then run the common deterministic
transition evaluator on validation and test. Use a separate GPU for the frozen
actor:

```bash
OUTPUT_DIR=outputs/evaluation/textworld_formal_<arm>_lora_r32_steps2000_seed42_val1000 \
ACTOR_GPU=5 LIMIT=1000 CONCURRENCY=8 \
  bash scripts/evaluate_textworld_validation_baseline.sh

INPUT_DATA=../../data/processed/textworld_grpo_task_split_v1/test/test.parquet \
OUTPUT_DIR=outputs/evaluation/textworld_formal_<arm>_lora_r32_steps2000_seed42_test1820 \
ACTOR_GPU=5 LIMIT=1820 CONCURRENCY=8 \
  bash scripts/evaluate_textworld_validation_baseline.sh
```

Before deleting an FSDP checkpoint, require `total == successful`,
`errors == 0`, unique item IDs, finite metrics, a loadable merged config and
tokenizer, and matching input provenance. Keep the merged model, manifest,
training/TensorBoard logs, and evaluation JSONL/JSON outputs.
