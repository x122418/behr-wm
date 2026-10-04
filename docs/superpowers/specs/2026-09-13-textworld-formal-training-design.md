# TextWorld Formal Training Design

## Goal

Run the first controlled, larger-scale TextWorld experiment for actor-distribution
consistency rewards. The experiment tests whether Union-top-64 + OTHER JS, with
an auxiliary real-observation SFT loss, improves both world-model state fidelity
and frozen-actor behavioral consistency beyond the SFT checkpoint, original
BehR, and pure Union-JS.

This is the first formal scale-up from the 50-step pilot. It is not a full epoch
over every available transition and is not yet a multi-seed paper claim.

## Experimental arms

All trained arms start from the same
`WorldModel-Textworld-Qwen2.5-7B` checkpoint and use the same training data
order, optimizer settings, rollout group size, temperature, training budget,
validation schedule, and frozen Qwen3-8B actor.

1. Original BehR GRPO (`reward_mode=cauchy`, auxiliary SFT coefficient 0).
2. Union-top-64 + OTHER JS GRPO (`reward_mode=union_js`, auxiliary SFT
   coefficient 0).
3. Union-top-64 + OTHER JS GRPO with auxiliary SFT coefficient 0.1.
4. The unchanged SFT checkpoint is evaluated as a zero-training baseline.

The three GRPO arms use `G=4` and rollout temperature 0.7. Matching these
settings is necessary because the earlier pilots did not all use identical
sampling configurations.

## Data and deterministic sampling

Training reads
`data/processed/textworld_grpo_task_split_v1/train/full.parquet`, containing
570,010 transitions from training tasks. Each run is limited to 2,000 optimizer
steps with batch size 4, so it consumes approximately 8,000 transition draws
from the deterministic shuffled sampler without requiring a separate subset
file.

The launcher sets the data sampler seed and actor data-loader seed explicitly
to 42. VERL 0.7.1 does not expose a seed field in its legacy `RolloutConfig`, so
the pinned vLLM backend's recorded default is held constant rather than
pretending that the launcher controls it. The data-order seed and complete
configuration are recorded in the experiment name and console log. The methods
therefore see the same training-row order; their generated observations may
diverge because their parameters and rewards diverge during optimization.

Online validation uses the existing fixed 1,000-row task-disjoint validation
pilot every 250 steps. Final evaluation uses both that common 1,000-row set and
the complete 1,820-row held-out test transition set. Online validation never
selects or changes training examples.

## Training configuration

The common configuration is:

- world model: Qwen2.5-7B TextWorld SFT checkpoint;
- frozen reference actor: Qwen3-8B;
- two H100 GPUs for VERL FSDP actor/rollout workers;
- one separate H100 GPU for the applicable frozen-actor reward service;
- GRPO batch size 4 and group size 4;
- rollout temperature 0.7 and top-p 1.0;
- maximum prompt length 4,096 and response length 512;
- learning rate `5e-6`;
- actor KL coefficient 0.001;
- 2,000 total training steps;
- validation every 250 steps;
- checkpoint saving every 1,000 steps;
- at most one retained actor checkpoint per run.

Only the reward mode and auxiliary SFT coefficient differ between trained
arms. TensorBoard and console logs include reward, validation consistency,
policy-gradient loss, KL loss, auxiliary SFT loss, total loss, gradient norm,
response length, scorer failure rate, timing, and memory metrics.

## Resource and checkpoint lifecycle

An FSDP training checkpoint is approximately 43 GB and a merged Hugging Face
model is approximately 15 GB. With about 308 GB free at design time, experiments
run sequentially rather than concurrently.

For each arm:

1. train with `trainer.max_actor_ckpt_to_keep=1`;
2. save at steps 1,000 and 2,000, retaining only the latest completed save;
3. verify both rank shards and the checkpoint tracker;
4. merge step 2,000 to a Hugging Face model directory;
5. verify that the merged config and tokenizer load locally;
6. complete the common validation and test evaluations;
7. verify row counts, unique IDs, finite metrics, and zero errors;
8. delete the 43 GB FSDP checkpoint only after the merged model and evaluation
   artifacts pass all checks.

Merged models, TensorBoard logs, training logs, configuration manifests, and
evaluation JSON/JSONL files are retained. A failed or interrupted run keeps its
latest valid checkpoint until it is resumed or explicitly abandoned.

## Execution gates

Before every long run:

- confirm the exact three GPU IDs are free and remain unclaimed;
- confirm the scorer health endpoint and model provenance;
- require at least 130 GB free disk headroom, allowing a transient second
  checkpoint during rotation plus logs;
- run launcher `--dry-run` and a two-step smoke with the exact formal settings;
- refuse to reuse a non-empty output directory unless resume mode and checkpoint
  provenance match the intended run.

During training, stop the run for non-finite loss, OOM, persistent scorer
failures, model-provenance mismatch, or another user's process appearing on an
allocated GPU. A single transient scorer failure is recorded and investigated;
it does not silently become a valid reward observation.

## Evaluation and success criteria

Every model is evaluated with the same frozen actor, input parquet, generation
settings, and item-ID set. The primary transition-level report contains:

- exact match (higher is better);
- original BehR Cauchy reward (higher is better);
- full-vocabulary KL from real to predicted observation (lower is better);
- full-vocabulary JS (lower is better);
- Union-top-64 + OTHER JS (lower is better);
- generation/scoring success counts and latency.

The 2,000-step Union-JS + SFT run is considered successful for scale-up if it
retains or improves exact match relative to the matched BehR run while improving
full-vocabulary JS, with 100% successful final evaluation rows and finite
training metrics. Results are reported at the predetermined final step rather
than selecting the best point after inspecting the test set.

Trajectory-level TextWorld evaluation remains required before a paper-level
claim about downstream policy success. It is a later evaluation stage and does
not change this transition-level training protocol.

## Follow-up stages

If the 2,000-step single-seed result preserves the pilot advantage:

1. extend the strongest arms to 5,000-10,000 steps using a predeclared budget;
2. repeat the decisive comparison with at least three declared data-order and
   actor-loader seeds while keeping the pinned rollout backend fixed;
3. add trajectory-level CR and CR-pw evaluation;
4. run reward and auxiliary-SFT coefficient ablations only after the main
   matched comparison is stable.

## Non-goals

- No 570,010-transition full epoch in this stage.
- No concurrent long-running GRPO arms.
- No actor fine-tuning.
- No full-vocabulary KL training reward in the main comparison.
- No automatic deletion before merge and evaluation verification.
- No claim that the 2,000-step single-seed result alone establishes statistical
  significance.
