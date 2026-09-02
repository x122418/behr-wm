# TextWorld Actor-Distribution Consistency Service Design

Date: 2026-09-02

## Goal

Add a training-time scorer for TextWorld world-model GRPO that compares the
frozen reference actor's teacher-forced token distributions under the real and
world-model-predicted observations. The first supported rewards are exact
full-vocabulary Jensen–Shannon consistency and union-top-64 plus OTHER
Jensen–Shannon consistency.

The scorer must reproduce the already validated offline metrics without
returning full-vocabulary logits over HTTP or loading the actor into verl/Ray
reward workers.

## Non-goals

- Do not change the WebShop reward path.
- Do not replace the original BehR reward mode.
- Do not modify vLLM internals.
- Do not add cross-request dynamic batching in the first version.
- Do not add an unbounded full-logit cache.
- Do not run a full GRPO experiment as part of the implementation.

## Architecture

Run one independent HTTP service on the GPU assigned to the frozen Qwen3-8B
reference actor. The service owns exactly one tokenizer and one model instance.
GRPO reward workers send histories, observations, and logged actions; the
service constructs a two-example real/predicted batch, performs one
teacher-forced actor forward pass, computes divergences locally, and returns
only scalar metrics.

```text
verl/Ray compute_score workers
           |
           | HTTP: history, real state, predicted state, expert action
           v
actor consistency service (one process, one frozen actor GPU)
           |
           +-- shared prompt construction and action-token alignment
           +-- batched real/predicted forward pass
           +-- full-vocabulary KL/JS
           +-- union-top-k plus OTHER JS
           |
           v
HTTP: scalar divergences, reward, token count, scorer provenance
```

The original vLLM BehR server remains available for the original baseline.
Matched experiments select one reward backend at launch time rather than
mixing the two servers in a single run.

## Shared teacher-forcing contract

Prompt construction and action-token alignment are scientific-critical logic.
They must have one shared implementation used by:

- the existing offline transition evaluator;
- the new actor consistency service;
- service/offline equivalence tests.

For every action position `t`, both inputs use the same logged action prefix
`y_<t`. The only difference between the two inputs is the current observation.
The implementation must verify that real and predicted inputs produce exactly
the same action token IDs. A mismatch is a request error, not a value to
silently truncate or pad.

The shared function accepts the TextWorld history, current observation, and
expert action. It applies the Qwen chat template with thinking disabled and
returns model input IDs plus aligned action IDs. Existing offline behavior is
the compatibility reference.

## Metrics and reward

Let `p_t` be the reference actor distribution for the real observation and
`q_t` the distribution for the predicted observation at logged action position
`t`.

The service computes:

- original logged-action diagnostics, including the mean log-probability
  difference and Cauchy BehR reward;
- `full_vocab_kl_real_to_candidate`;
- `full_vocab_js`;
- `top64_union_other_js` (with configurable `top_k`);
- `action_token_count`.

For natural logarithms, the normalized JS consistency reward is:

```text
reward = 1 - mean_t(JS(p_t, q_t)) / log(2)
```

Numerical rules:

- compute logits in model dtype but probabilities/divergences in float32;
- use stable `log_softmax`/`softmax` operations;
- clamp the final normalized reward to `[0, 1]` only for floating-point drift;
- aggregate only logged action positions;
- reject empty actions, non-finite outputs, or token-alignment failures.

Union-top-k plus OTHER uses the union of each side's top-k vocabulary IDs. The
probabilities for every union token are gathered from both full distributions.
All probability mass outside the union is placed in one OTHER bucket. This is
not the less rigorous independently-renormalized truncated distribution.

## HTTP API

### `GET /health`

Returns readiness only after the tokenizer and frozen actor are loaded.

```json
{
  "status": "ok",
  "model": "/resolved/model/path",
  "device": "cuda:0",
  "dtype": "torch.bfloat16"
}
```

### `GET /v1/models`

Returns scorer provenance so launchers and reward workers can verify that the
configured frozen actor is the intended checkpoint.

### `POST /v1/behavior-consistency`

Request:

```json
{
  "history": [{"role": "system", "content": "..."}],
  "real_observation": "...",
  "predicted_observation": "...",
  "expert_action": "go east",
  "top_k": 64,
  "reward_metric": "union_topk_other_js"
}
```

Successful response:

```json
{
  "score": 0.9794,
  "reward_metric": "union_topk_other_js",
  "full_vocab_js": 0.0143,
  "full_vocab_kl_real_to_candidate": 0.2529,
  "top64_union_other_js": 0.0143,
  "original_behr_cauchy_reward": 0.9223,
  "action_token_count": 3,
  "model": "/resolved/model/path"
}
```

`reward_metric` initially accepts `union_topk_other_js` and `full_vocab_js`.
The service returns HTTP 422 for invalid request fields or alignment failures,
HTTP 503 before model readiness, and HTTP 500 for unexpected inference errors.
It never substitutes a fabricated numeric reward.

## Concurrency and memory

The first version serializes model forward passes with an inference lock. This
prevents concurrent HTTP requests from causing uncontrolled activation-memory
spikes. Each forward pass contains the real and predicted examples as one
padded batch of two.

FastAPI may accept concurrent connections, but requests wait at the inference
lock. This prioritizes correctness and bounded memory for the pilot. Timing and
queue-wait fields are returned or logged so later profiling can justify
dynamic batching.

No full-vocabulary tensor cache is included initially. A bounded real-state
cache may be introduced only after profiling demonstrates that actor forward
time dominates and after defining an explicit byte budget and eviction policy.

## GRPO integration

Extend the TextWorld reward configuration with:

- `reward_mode=cauchy`: unchanged original BehR path;
- `reward_mode=union_js`: call the scorer with
  `reward_metric=union_topk_other_js`;
- `reward_mode=full_vocab_js`: call the scorer with
  `reward_metric=full_vocab_js`;
- `consistency_api_url` and `consistency_top_k`.

The new client sends exactly one request per generated WM observation. The
returned `score` feeds the existing behavior-weight aggregation. Additional
metrics are included in the reward result for logging.

On timeout, non-2xx response, malformed JSON, or non-finite score, the reward
function marks `api_failed=true` and applies the existing configured format/
failure penalty. It must not silently fall back to original BehR because that
would mix objectives inside one experiment.

The global client instance remains lightweight and contains only HTTP session
state; the actor model exists exclusively in the scorer service process.

## Launch and configuration

Add a launcher that requires an explicit model path and GPU ID and defaults to
port 8002. It sets `NO_PROXY=127.0.0.1,localhost`, uses local model files, and
prints resolved model/GPU/port/dtype before loading.

The pilot launcher receives a reward-backend selector and scorer URL while
preserving all other matched GRPO hyperparameters. Dry-run output must make the
selected objective and endpoint unambiguous.

## Testing

All production changes follow red-green-refactor.

1. Pure metric tests:
   - identical distributions produce zero divergence and reward one;
   - a hand-computed example matches full KL, full JS, and union+OTHER JS;
   - action masks average only valid positions;
   - invalid `top_k`, empty actions, non-finite values, and alignment mismatch
     fail explicitly.

2. Shared prompt tests:
   - preserve the existing TextWorld history-role conversion;
   - match the current offline evaluator token-for-token;
   - real and predicted observations yield identical action IDs.

3. Service tests with a tiny fake tokenizer/model:
   - API schema and status codes;
   - real/predicted examples are passed in one batch;
   - response metrics equal the pure scorer output;
   - health is false before readiness and includes provenance after readiness;
   - inference failures do not return a numeric score.

4. Reward-client tests with mocked HTTP transport:
   - correct payload and reward mode mapping;
   - scalar score and diagnostic propagation;
   - timeout/error behavior and no silent BehR fallback;
   - original `cauchy` behavior remains unchanged.

5. Equivalence smoke test on GPU when available:
   - run fixed validation records through both the offline scorer and service;
   - require metric agreement within an explicit float32 tolerance;
   - then run a two-step GRPO JS smoke before any pilot training.

## Acceptance criteria

- The scorer service loads exactly one frozen actor model.
- Full logits remain inside the scorer process.
- Offline and service metrics agree on fixed examples within tolerance.
- `reward_mode=cauchy` remains regression-compatible.
- JS reward failures are explicit and never silently change objectives.
- A two-step JS GRPO smoke completes before the 50-step matched pilot.
- The complete CPU test suite passes without requiring a GPU or model load.

