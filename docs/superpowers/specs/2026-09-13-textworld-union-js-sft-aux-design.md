# TextWorld Union-JS with Auxiliary SFT Loss

## Goal

Train the TextWorld world model with the stable Union-JS GRPO objective while
preserving its supervised next-observation behavior.  The target objective is

\[
L_{total}=L_{GRPO-JS}+\beta L_{KL}+\lambda_{SFT}L_{SFT}.
\]

The frozen reference actor remains a scorer only.  Gradients from all three
terms update only the world model.

## Motivation and success criterion

The stable Union-JS pilot (`G=4`, rollout temperature `0.7`) achieved the best
held-out actor-distribution consistency so far (full-vocabulary JS `0.01366`,
KL `0.1393`) but reduced exact match from the SFT checkpoint's `65.5%` to
`64.9%`.  The auxiliary SFT term should retain the distribution improvement
without losing supervised next-state fidelity.

The first 50-step pilot is considered promising if it:

- keeps full-vocabulary JS below the SFT baseline (`0.02308`);
- restores exact match to at least the SFT baseline (`65.5%`); and
- completes with finite GRPO, KL, SFT, and total losses and no scorer errors.

These pilot thresholds select the next experiment; they are not final paper
claims.

## Data contract

Each existing GRPO row already contains both branches required by the loss:

- `prompt`: history before the next observation;
- `reward_model.ground_truth`: the real next observation for SFT;
- `extra_info.history` and `extra_info.expert_action`: inputs used by the
  frozen actor consistency scorer.

No new dataset is required.  Before the actor update, the trainer tokenizes
`prompt + ground_truth` with the world-model tokenizer and constructs:

- teacher-forced input IDs and attention/position IDs;
- an SFT loss mask that is zero over the prompt and one over ground-truth
  observation tokens;
- one SFT target per original prompt, not one separately weighted target per
  rollout sample.

The GRPO rollout tensors and masks remain unchanged.

## Training flow

For every optimizer micro-batch:

1. Run the existing GRPO forward pass on sampled WM observations.
2. Compute the existing clipped policy-gradient loss and reference-model KL.
3. Run a second teacher-forced forward pass on the real observation.
4. Compute token-mean cross entropy only on ground-truth observation tokens.
5. Backpropagate the weighted sum and take one optimizer step.

The two forward passes are sequential so their activation graphs do not need
to coexist longer than necessary.  The implementation must preserve FSDP
gradient accumulation and must not send gradients to the frozen actor scorer.

The first pilot uses:

- `lambda_sft = 0.1`;
- `beta = 0.001` (unchanged);
- Union-top-64 + OTHER JS reward;
- `G = 4`, rollout temperature `0.7`;
- 1,000 training transitions, 1,000 task-disjoint validation transitions;
- 50 steps, saving only the final checkpoint.

## Integration boundary

VERL 0.7.1 has an SFT loss implementation, but its legacy PPO actor has no
configuration hook for combining that loss with PPO/GRPO.  We will not make
untracked edits to `.venv`.

The repository will carry a version-locked downstream patch for VERL 0.7.1
plus an idempotent installer/checker.  The patch will be intentionally narrow:

- preserve the real observation through the PPO batch;
- build the auxiliary teacher-forced tensors;
- add the SFT term inside the legacy FSDP actor update;
- expose `actor.sft_loss_coef` with default `0.0`, so existing experiments are
  behaviorally unchanged;
- log `actor/sft_loss`, `actor/sft_loss_coef`, and `actor/total_loss`.

The patch application must verify the installed VERL version and expected
source context, fail loudly on drift, and be reproducible from a fresh local
environment.  Training launchers must refuse auxiliary-SFT mode when the patch
is not installed.

## Validation and tests

CPU tests must cover:

- prompt tokens are excluded from the SFT mask;
- padding tokens are excluded;
- rollout duplication does not change the effective SFT weighting;
- `sft_loss_coef=0` reproduces the existing loss path;
- a positive coefficient changes the total loss by exactly
  `lambda_sft * sft_loss`;
- launcher dry-run forwards `lambda_sft`, `G=4`, and temperature `0.7`;
- the patch installer is version-checked and idempotent.

GPU validation proceeds in gates:

1. one-batch forward/backward probe with finite loss and gradients;
2. two-step smoke with healthy Union-JS scorer;
3. 50-step pilot;
4. merge the FSDP checkpoint and run the same deterministic offline evaluator
   used for SFT, BehR, and pure Union-JS.

## Comparison table

The pilot report will use the same held-out 1,000 rows and frozen Qwen3-8B
actor and will report exact match, original BehR reward, full-vocabulary KL,
full-vocabulary JS, and top-64 Union+OTHER JS.  It will compare:

- SFT checkpoint;
- BehR GRPO;
- stable pure Union-JS GRPO;
- Union-JS GRPO + auxiliary SFT.

## Non-goals

- No full 632,525-transition training run in this iteration.
- No actor fine-tuning.
- No change to the Union-JS mathematical definition.
- No claim that exact match is the final trajectory-level TextWorld metric.
- No modification of global Python, CUDA, or driver installations.
