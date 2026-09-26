# ROSA Surrogate Experiments

**Status:** exploratory prototype; the 100-digit addition task is not solved yet. | **Last updated:** 2026-09-26

This note records the motivation, implementation, measurements, and limitations of the experiments around using a differentiable model as a training-time surrogate for the discrete ROSA operator. It includes both the small binary-sequence validation and the RWKV7-based 100-digit addition toy task.

## Summary

The core idea is to preserve ROSA's exact discrete value in the forward pass while using a learned differentiable surrogate to provide a backward gradient. A small binary-sequence experiment provided an initial proof of concept: the causal Transformer surrogate matched the exact ROSA output on 96.289% of held-out positions, and a fixed-pattern upstream value projection improved from 0% to 100% exact-task accuracy using the surrogate gradient. A focused check of the addition ROSA branch confirms bitwise exact forward values and exact gradient agreement with a direct proxy path upstream.

The addition prototype has a four-example overfit harness and a configurable random-stream trainer. The 100-digit streaming run performed 5,000 updates on 40,000 fresh additions; its best held-out token accuracy was 11.700%, and every greedy checkpoint remained 0/4 exact sums. The follow-up 10-digit run was stopped after its step-4,400 metrics: held-out token accuracy peaked at 14.506% and finished at 13.768%, while every greedy check remained 0/16 exact sums. A matched 1,500-step trial with proxy-gradient scale `0.5` peaked at 14.036% held-out token accuracy and also remained 0/16 on greedy checks. Its same-code scale-1 control had two skipped non-finite-gradient updates; the scale-0.5 run had none. These are single-seed exploratory comparisons.

These results show that the prototype can train on a continuing stream, but do not show that it has learned 100-digit addition. The streaming run used only 16 fixed validation examples, one seed, and a randomly initialized small model; it did not load a pretrained RWKV checkpoint.

## Hypothesis and gradient construction

ROSA's binary matching operation is discrete, so its exact output does not provide a useful ordinary gradient to upstream model parameters. The experiment uses a differentiable proxy during training while keeping the exact operator's output as the forward value.

For exact output `y_exact` and proxy output `y_proxy`, the straight-through construction is:

```python
y = y_exact + (y_proxy - y_proxy.detach())
```

The parenthesized proxy subtraction is exactly zero in the forward pass because both operands have identical values. This keeps the forward value bitwise equal to `y_exact`; the backward pass sees the derivative of `y_proxy`. The proxy is also trained directly with an auxiliary cross-entropy loss against exact ROSA targets:

```text
total_loss = task_loss + lambda_rosa * rosa_distillation_loss
```

The small binary toy uses a one-hot exact output in this construction. The addition prototype uses an exact binary ROSA result encoded as `-1/+1`, with the proxy's two-class softmax converted to a signed signal. In both cases, the ROSA distillation loss is computed over proxy logits and exact targets.

The intended deployment idea is to use the exact ROSA operator at inference. The current prototype already uses the exact result as its forward value, but it still evaluates the proxy network and constructs its proxy logits in `eval()` mode. Replacing or bypassing the proxy computation in an inference-only path has not been implemented.

### Addition STE implementation check

The addition ROSA branch originally computed `exact_signal + proxy_signal - proxy_signal.detach()` from left to right. A floating-point check found that this expression was not always bitwise exact: among one million random `tanh` proxy values for each exact signal (`-1` and `+1`), 117,016 and 115,789 values respectively differed from the exact signal by one FP32 ULP (maximum absolute error `5.96e-8`). The implementation now computes `exact_signal + (proxy_signal - proxy_signal.detach())`. The detached subtraction is exactly zero for the forward pass while retaining the proxy derivative. The same parenthesization was applied to the binary toy's STE expressions.

The focused check in [`rosa_add_ste_check.py`](rosa_add_ste_check.py) ran the actual addition `ROSAQKV` module on CPU with a fixed `[2, 16, 8]` input, dropout and sign flips disabled, and an identity output projection to expose the STE activation. It verified that:

- The forward activation was bitwise equal to the signed exact ROSA target, with both target classes present.
- With task loss only and no distillation loss, gradients reached the input, q/k/v projections, and proxy parameters; the proxy head gradient norm was `32.0311`.
- Input and all compared upstream/proxy parameter gradients matched the direct differentiable-proxy reference exactly (maximum absolute difference `0`).

This validates the STE wiring and exact-forward behavior in the addition ROSA branch. It does not establish that the proxy approximates ROSA well or that STE solves the addition task.

#### Gradient-scaled STE trial

A simple STE variant scales only the task gradient through the proxy path:

```python
y = y_exact + alpha * (y_proxy - y_proxy.detach())
```

The forward value remains exactly `y_exact`; the task gradient sent through the proxy and its q/k/v inputs is multiplied by `alpha`. The auxiliary ROSA distillation loss is unchanged. The stream trainer now exposes this as `--ste-gradient-scale` (default `1.0`). The focused STE check was rerun with `alpha=0.5`; exact forward remained bitwise equal, and upstream/proxy gradients matched half of the direct proxy-path gradients exactly.

For a matched comparison, two fresh 10-digit streams used the same seed (`321`), 48-token context, batch size 32, fixed validation set of 128 examples, learning rate `1e-4`, weight decay `0.01` on linear matrices, no ROSA dropout, 5% sign flipping, and 1,500 steps. Validation was recorded every 100 steps; greedy exact accuracy used the same 16 examples at steps 500, 1,000, and 1,500.

| STE gradient scale | Step 500 validation loss / token accuracy | Step 1,000 validation loss / token accuracy | Step 1,500 validation loss / token accuracy | Best token accuracy through step 1,500 | Greedy at 500 / 1,000 / 1,500 | Skipped non-finite gradients |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 1.0 | 2.4896 / 13.163% | 2.4997 / 12.827% | 2.5008 / 12.559% | 13.700% at step 300 | 0/16, 0/16, 0/16 | 2 |
| 0.5 | 2.5829 / 11.954% | 2.4937 / 12.760% | 2.5030 / 12.895% | **14.036% at steps 1,100 and 1,300** | 0/16, 0/16, 0/16 | 0 |

The half-scale run reached a slightly higher best token accuracy, while the scale-1 run had a lower best task loss (`2.4803` at step 300 versus `2.4852` at step 1,100). At step 1,500 their task losses were close, and neither configuration produced a correct greedy sum. The lower scale also had fewer observed non-finite-gradient skips in this pair, but one seed is not enough to conclude that it is more stable or generally better.

The runs are [`scale 1 metrics`](runs/add10_stream_stecontrol_wd_20260926/metrics.csv), [`scale 0.5 metrics`](runs/add10_stream_stehalf_wd_20260926/metrics.csv), and their respective [`scale 1 log`](runs/add10_stream_stecontrol_wd_20260926.log) and [`scale 0.5 log`](runs/add10_stream_stehalf_wd_20260926.log). Both checkpoints are saved in the corresponding run directories.

## Experiment files

| File | Purpose |
| --- | --- |
| [`rosa_surrogate_toy.py`](rosa_surrogate_toy.py) | Small binary-sequence experiment for exact-forward/surrogate-backward behavior, plus a controlled upstream value-projection check. |
| [`rosa_add_ste_check.py`](rosa_add_ste_check.py) | Focused check that the addition ROSA branch is bitwise exact in the forward pass and sends the same task gradient upstream as the differentiable proxy path. |
| [`rosa_add100_toy.py`](rosa_add100_toy.py) | Configurable generator, token layout, and self-check for fixed-width addition examples (default: 100 digits). |
| [`rosa_numba.py`](rosa_numba.py) | Numba-compiled exact binary ROSA implementation used to generate targets efficiently. |
| [`rosa_add100_smoke.py`](rosa_add100_smoke.py) | One-batch CUDA smoke model: four RWKV7+ROSA blocks, a causal Transformer proxy per ROSA block, and the custom WKV7 CUDA kernel. |
| [`rosa_add100_overfit.py`](rosa_add100_overfit.py) | Repeated-batch optimizer test with teacher-forced metrics and autoregressive greedy decoding. |
| [`rosa_add100_stream.py`](rosa_add100_stream.py) | Configurable random-stream trainer with held-out evaluation, greedy checks, CSV logging, checkpoints, resume support, and proxy STE gradient scaling. |

The four-example overfit scripts start from a fresh initialization and regenerate the same examples from seed 321. The streaming trainer saves model, optimizer, and random-number-generator state so a run can resume from its last checkpoint.

## Binary-sequence surrogate experiment

### Setup

The first validation uses binary `q`, `k`, and `v` sequences of length 12 with two channels. Half of the examples contain a shifted `q/k` substring so the data include longer matches as well as matches from independent random sequences. The Python suffix-automaton reference produces exact targets; its no-match value `-1` and a matched value of `0` are both mapped to class 0, matching the 1-bit operator's output semantics.

The proxy is a causal Transformer with width 64, two layers, and four attention heads. Its input contains the `q/k/v` values, and it predicts two output classes per channel and position. Training uses 4,096 examples, validation uses 512 examples, AdamW with learning rate `2e-3` and weight decay `1e-4`, batch size 128, and 1,000 steps. The task label is an XOR of the two ROSA output bits at the final position. The objective combines the downstream task loss with ROSA-output distillation at weight 1.0.

### Results

The recorded held-out results were:

| Measurement | Result |
| --- | ---: |
| Per-position hard proxy match to exact ROSA | 96.289% |
| Downstream task accuracy when supplied exact ROSA output | 100% |
| Downstream task accuracy when supplied hard proxy output | about 83.984% |

This showed that the proxy could approximate many ROSA outputs and that exact-forward/surrogate-backward training was viable in the tiny setup. It also showed the remaining gap between exact and proxy activations.

### Upstream gradient check

The same script includes a deliberately small check that freezes the Transformer proxy and learns a scalar projection before binary `v`. A fixed `q/k` pattern makes the final ROSA output read `v` from position 6. A one-hot probe verifies that source position. The learned scalar started at `alpha=-2.0`; after optimization it reached approximately `alpha=1.804`, and exact downstream task accuracy on the controlled check moved from 0% to 100%.

This is a useful gradient-path sanity check, not evidence that the proxy learns arbitrary ROSA behavior or that a large recurrent model will train successfully. The `q/k` pattern is fixed, and only a scalar `v` projection is learned.

## Exact 100-digit addition data

### Task and tokens

Each example is addition of two uniformly sampled, nonzero-leading 100-digit integers. The sum has either 100 or 101 digits. The vocabulary follows the arithmetic demo convention:

| Token IDs | Meaning |
| --- | --- |
| `0` through `9` | Decimal digits |
| `10` | `+` |
| `11` | `-` (available in the vocabulary, unused in this addition task) |
| `12` | `=` |

The prompt is `A+B=` and has 202 tokens. The target is the decimal sum followed by `=`; additional `=` tokens pad the sequence. The loss mask includes only the sum digits and the first `=` after the sum. It excludes the prompt and padding positions.

The RWKV7 CUDA kernel uses a chunk length of 16. The chosen input context is 304 tokens, divisible by 16. The generator allocates a 305-token full sequence so the 304-token model input has a next-token target at every position.

### Data validation

The data self-check was run with 512 generated examples and seed 42. It verified operand width, no leading zero, arithmetic correctness, target alignment, and that the loss mask covers the sum plus its terminator. The check produced 214 100-digit sums and 298 101-digit sums, as expected for random 100-digit operands.

Run it with:

```bash
python RWKV-v8/rosa_add100_toy.py --self-check --samples 512 --seed 42
```

The four-example overfit runs use a separate fixed batch generated with `random.Random(321)`.

## RWKV7+ROSA prototype

### Architecture

The smoke model is a small, randomly initialized architecture adapted from the existing RWKV7+ROSA code path; it does not load the external arithmetic checkpoint. Its relevant dimensions are:

| Component | Configuration |
| --- | --- |
| Vocabulary | 13 tokens |
| Input context | 304 tokens |
| Model width | 128 |
| RWKV7 head size | 16 (8 heads) |
| RWKV7 CUDA chunk length | 16 |
| Blocks | 4, each with RWKV7 time mixing, ROSA, and feed-forward sublayers |
| ROSA proxy | One-layer causal Transformer, width 16, two attention heads |
| Exact ROSA inputs | `q`, `k`, and `v` thresholded at zero to binary symbols |

The model computes exact ROSA targets from detached binary `q/k/v`, runs the causal proxy, and uses exact targets in the forward activation. The exact operator's no-match result and a matched zero both become binary output 0; binary output 1 becomes a signed `+1` activation and output 0 a signed `-1` activation. The proxy loss is averaged over the ROSA blocks.

The addition objective is masked next-token cross-entropy on answer digits and the terminator, plus the ROSA distillation loss at weight 0.1. The four-example optimizer runs use AdamW, zero weight decay, learning rate `1e-4`, and global gradient clipping at 1.0. Earlier exploratory runs used `5e-4` before the learning rate was reduced.

### One-step CUDA smoke test

The one-batch smoke test used a single exact 100-digit example and completed a forward pass, backward pass, and optimizer step. The recorded task loss was 2.7628 and ROSA auxiliary loss was 0.7730. Gradients were finite and nonzero in the ROSA `q` projection, RWKV7 output projection, and proxy output head. The RWKV7 receptance gradient was finite but zero on the first step because the RWKV7 output projection is initialized to zero. After warmup, recorded timing was about 0.04 seconds forward and 0.06 seconds backward. Peak memory in the later four-example runs was about 0.88 GiB.

## Numba exact ROSA implementation

`rosa_numba.py` implements the online suffix-automaton lookup with Numba. Instead of Python transition dictionaries, it stores binary transitions in a dense array of shape `[2*T+1, 2]`, along with suffix links, maximum lengths, and rightmost end positions. Inputs are contiguous `uint8` arrays shaped `[rows, T]`, where each row represents one batch/channel sequence.

Validation compared the Numba operator to the Python reference on random sequences of lengths `1, 2, 3, 8, 32, 128, 304`, exhaustively checked all 4,680 binary `(q,k,v)` triples for lengths up to 4, and checked a batch of shape `128 x 304`. The tested outputs matched under the prototype's binary ROSA semantics.

Recorded warmed timing for a `128 x 304` batch was approximately:

| Work | Python reference | Numba | Approximate speedup |
| --- | ---: | ---: | ---: |
| Exact ROSA target generation | 0.0682 s | 0.0026 s | 26x |
| Full model forward, exact target backend swapped | 0.3046 s | 0.0250 s | 12x |
| Full model backward | 0.0416 s | 0.0288 s | 1.4x |

The warmed full-model logits matched when the exact target backend was swapped. Numba JIT compilation was excluded from steady-state timing. The current integration transfers binary `q/k/v` from CUDA to CPU NumPy for target generation and copies targets back to CUDA, so transfer costs and scaling at larger batch sizes remain to be measured.

## Four-example overfit experiments

### Metric definitions

The reported `task_loss` and `token_acc` are teacher-forced metrics evaluated only at the masked sum and terminator positions. Later input positions contain the correct preceding answer digits. These metrics measure next-token prediction under the gold prefix; they do **not** measure whether a generated answer is correct.

At each checkpoint the script also performs greedy autoregressive decoding. It repeatedly predicts the next token from the model's current generated prefix, stopping at `=` or after allowing 101 answer digits plus a terminator. `greedy_exact` counts an example only when every answer digit is correct and the model emits `=`.

`rosa_loss` is the proxy's two-class cross-entropy against exact ROSA outputs. It is not an addition metric.

### Initial run at learning rate `5e-4`

The initial fixed batch contains four examples. All configurations share the initial evaluation `task_loss=2.7030`, `rosa_loss=0.8338`, and `token_acc=7.389%`.

With no dropout and no sign flips, task loss initially fell, then rose again:

| Step | Task loss | ROSA loss | Teacher-forced token accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 2.7030 | 0.8338 | 7.389% |
| 1 | 2.4974 | 0.8306 | 11.084% |
| 5 | 2.2129 | 0.8247 | 19.951% |
| 10 | 2.1131 | 0.8170 | 26.108% |
| 20 | 1.9287 | 0.8046 | 32.266% |
| 50 | 2.1298 | 0.8167 | 28.818% |
| 100 | 2.2910 | 0.8206 | 25.369% |

At step 20, greedy exact-sum accuracy was 0/4. The longer run indicated that this learning rate was unstable for this setup: teacher-forced task metrics worsened after step 20.

Adding ROSA-branch dropout 0.1 while retaining `5e-4` did not produce a usable 100-step run. At step 20 the task loss was 2.0682 and token accuracy 29.557%; at step 50 they were 2.1194 and 24.631%. At the step-100 checkpoint the losses were `NaN`. The first non-finite step was not recorded in that run. The run's greedy result was 0/4 and did not terminate with `=`.

### Controlled 100-step comparisons at learning rate `1e-4`

After reducing the learning rate, four regularization combinations were run with the same seed and fixed examples. Dropout is applied after the ROSA branch's output projection and is disabled in `eval()` mode. Sign flipping independently negates ROSA signal elements with the configured probability before the branch's output projection; it is enabled only while training.

| ROSA dropout | Sign-flip probability | Task loss at steps 20 / 50 / 100 | ROSA loss at 20 / 50 / 100 | Token accuracy at 20 / 50 / 100 | Greedy exact at 100 |
| ---: | ---: | --- | --- | --- | ---: |
| 0.1 | 0 | 2.1916 / 1.9571 / 1.3205 | 0.8272 / 0.8150 / 0.8250 | 22.167% / 33.005% / 59.852% | 0/4 |
| 0 | 0 | 2.1810 / 1.9067 / 1.4199 | 0.8263 / 0.8167 / 0.8256 | 21.921% / 37.931% / 54.433% | 0/4 |
| 0.1 | 0.05 | 2.2211 / 2.0550 / 1.4960 | 0.8273 / 0.8172 / 0.8116 | 20.936% / 30.788% / 54.680% | 0/4 |
| 0 | 0.05 | 2.2072 / 1.9364 / 1.3837 | 0.8270 / 0.8131 / 0.8187 | 20.936% / 37.192% / 58.128% | 0/4 |

Every configuration had 0/4 exact greedy sums at step 100. Most outputs failed to emit a terminator; in the no-dropout/sign-flip run, one of four examples emitted `=` but the answer was still wrong.

These single-seed results do not establish that dropout or sign flips improve training. At step 100, dropout 0.1/no flip was somewhat ahead of no dropout/no flip, and no dropout/5% flip was somewhat ahead of no dropout/no flip. The trajectories crossed at earlier checkpoints, and no repeated-seed estimate was run.

### Requested 200-step comparison

The most recent comparison held seed, examples, and learning rate constant and compared:

| Setting | Task loss, steps 20 / 50 / 100 / 200 | ROSA loss, steps 20 / 50 / 100 / 200 | Token accuracy, steps 20 / 50 / 100 / 200 | Greedy exact at 200 |
| --- | --- | --- | --- | ---: |
| Dropout 0.1, flip 0 | 2.1916 / 1.9571 / 1.3205 / **0.9221** | 0.8272 / 0.8150 / 0.8250 / 0.8513 | 22.167% / 33.005% / 59.852% / **70.690%** | 0/4 |
| Dropout 0, flip 0.05 | 2.2072 / 1.9364 / 1.3837 / **0.7515** | 0.8270 / 0.8131 / 0.8187 / 0.8436 | 20.936% / 37.192% / 58.128% / **81.281%** | 0/4 |

The dropout-0/flip-0.05 run fit the four training examples better under teacher forcing at step 200: lower task loss and 10.591 percentage points higher masked token accuracy than dropout-0/flip-0. (That exact 200-step no-dropout/no-flip control was not run; the available no-dropout/no-flip comparison ends at step 100.) Both requested configurations still had 0/4 exact greedy sums. In the flip-only run, one prediction terminated early with an incorrect partial answer; the other three did not emit the terminator within the decoding limit.

The ROSA proxy loss rose between steps 100 and 200 for both configurations, even as task loss fell. Thus a lower addition loss did not coincide with improving proxy distillation loss in this run.

### Current interpretation

The strongest conclusion is that teacher-forced fit on four examples can improve substantially while autoregressive addition remains unsolved. The model's token accuracy and task loss do not capture error propagation through a 100-digit generated prefix, nor do they demonstrate generalization to new operands. The 200-step comparison favors dropout 0/flip 0.05 for fitting this one batch, but because this pair of runs changes both regularizers and uses one seed, it does not isolate a general causal effect.

### Random-stream 100-digit addition run

The streaming trainer draws a new batch of eight random 100-digit additions for every update. A separate fixed validation set of 16 additions is used for teacher-forced loss and token accuracy every 100 steps. Greedy decoding checks four validation examples every 500 steps. The run used seed 321, no ROSA dropout, a 5% ROSA sign-flip probability, and AdamW weight decay 0.01 on 2-D linear projection matrices only. Embeddings, normalization parameters, and vectors were excluded from decay.

The run first trained to step 250 at learning rate `5e-5`. A non-finite gradient occurred in an exploratory continuation; training resumed from the step-250 checkpoint at `2e-5`, then paused at a step-1,000 checkpoint for the NaN replay. The completed run contains 5,000 optimizer updates and 40,000 generated training examples. The final 4,000-step segment (step 1,000 to 5,000) took 25.1 minutes on an RTX 3060 and peaked at 3.42 GiB CUDA memory; the earlier segments were run in separate sessions.

| Step | Validation task loss | Validation ROSA loss | Validation token accuracy | Greedy exact |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.6807 | 0.8333 | 8.374% | — |
| 1,000 | 2.3405 | 0.7807 | 9.483% | 0/4 |
| 2,000 | 2.3375 | 0.7454 | 9.975% | 0/4 |
| 3,000 | 2.3301 | 0.7466 | 10.099% | 0/4 |
| 4,000 | 2.3348 | 0.7511 | 10.776% | 0/4 |
| 5,000 | 2.3362 | 0.7508 | 10.222% | 0/4 |

The lowest validation task loss was 2.3296 at step 2,600; the best token accuracy was 11.700% at step 2,500. Greedy exact accuracy was 0/4 at every checkpoint from step 500 through 5,000. The proxy's distillation loss improved from its initial value, while addition metrics stayed near their initial range. These results do not show that the model learned the addition algorithm.

The final metrics and resumable checkpoint are [`metrics.csv`](runs/add100_stream_wd_resume_20260926/metrics.csv) and [`last.pt`](runs/add100_stream_wd_resume_20260926/last.pt).

### Non-finite loss replay

An earlier streaming run with seed 321, batch size 8, learning rate `1e-4`, no weight decay, and 5% sign flipping stopped with a non-finite loss at step 692. Replaying the same settings on an otherwise idle GPU matched the recorded step-100 metrics but diverged later and completed 750 steps without a NaN. A prior replay overlapped with the weight-decay run and also stayed finite through 1,000 steps. The failure is therefore not yet reliably reproducible; no root cause is established.

The successful no-weight-decay replay metrics are in [`metrics.csv`](runs/add100_nan_repro_solo_20260926/metrics.csv). The first failed run's partial log is in [`metrics.csv`](runs/add100_stream_20260926-160413/metrics.csv).

### 10-digit random-stream addition run (stopped at step 4,400)

To reduce sequence length, the generator and stream trainer were parameterized by operand width. The 10-digit run used a 48-token input context, the smallest multiple of the RWKV7 chunk length (16) that fits the prompt and longest possible sum. The 48-token input is within the ROSA proxy's 304-position embedding table. Data generation was checked on 512 examples; both 10-digit and 11-digit sum lengths appeared, arithmetic and target alignment were correct, and the original 100-digit data self-check still passed.

The stream used seed 321, batches of 32 fresh additions, and a fixed 128-example validation set. AdamW used learning rate `1e-4`, weight decay `0.01` on 2-D linear projection matrices only, no ROSA dropout, and 5% ROSA sign flipping. Validation loss and teacher-forced token accuracy were recorded every 100 steps. Greedy exact-sum accuracy was checked on 16 fixed validation examples every 500 steps. The run was explicitly stopped after recording step 4,400 rather than completing the planned 5,000 steps. The metrics CSV records the run through step 4,400, representing 140,800 training examples; the most recent checkpoint was saved at step 4,250.

| Step | Validation task loss | Validation ROSA loss | Validation token accuracy | Greedy exact |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.7376 | 0.8403 | 7.858% | — |
| 500 | 2.5779 | 0.8334 | 12.626% | 0/16 |
| 1,000 | 2.4640 | 0.8476 | 12.626% | 0/16 |
| 1,400 | 2.4395 | 0.8397 | **14.506% (best)** | — |
| 2,000 | 2.4203 | 0.8515 | 13.230% | 0/16 |
| 3,000 | 2.4192 | 0.8511 | 13.499% | 0/16 |
| 4,000 | 2.4172 | 0.8508 | 13.163% | 0/16 |
| 4,400 | **2.4145 (best)** | 0.8506 | 13.768% | — |

The best held-out task loss was the final measured value, but the best token accuracy occurred earlier and declined afterward. All eight greedy checks from steps 500 through 4,000 produced 0/16 exact sums. Thus the shorter context improved held-out next-token metrics over initialization, but this run did not demonstrate reliable autoregressive addition or generalization. The ROSA distillation loss also ended slightly above its initial value. This is one seed and one fixed validation set; the run stopped before its planned final 600 updates.

The interrupted run's metrics, last checkpoint, and console log are [`metrics.csv`](runs/add10_stream_wd_20260926/metrics.csv), [`last.pt`](runs/add10_stream_wd_20260926/last.pt), and [`add10_stream_wd_20260926.log`](runs/add10_stream_wd_20260926.log).

## Reproduction commands

From the repository root:

```bash
# Binary surrogate experiment
python RWKV-v8/rosa_surrogate_toy.py

# Verify exact-forward/proxy-backward behavior in the addition ROSA branch
python RWKV-v8/rosa_add_ste_check.py

# Validate 100-digit data and target alignment
python RWKV-v8/rosa_add100_toy.py --self-check --samples 512 --seed 42

# Validate the 10-digit version with its 48-token context
python RWKV-v8/rosa_add100_toy.py --self-check --digits 10 --context-len 48 --samples 512 --seed 42

# Train a 10-digit random stream (the documented run stopped early at step 4,400)
python RWKV-v8/rosa_add100_stream.py --digits 10 --context-len 48 --steps 5000 --batch-size 32 --validation-size 128 --eval-every 100 --greedy-every 500 --greedy-size 16 --save-every 250 --learning-rate 1e-4 --weight-decay 0.01 --rosa-dropout 0 --rosa-sign-flip 0.05 --seed 321 --run-dir RWKV-v8/runs/add10_stream_wd_20260926

# Matched 1,500-step comparison with half-scale proxy STE gradients
python RWKV-v8/rosa_add100_stream.py --digits 10 --context-len 48 --steps 1500 --batch-size 32 --validation-size 128 --eval-every 100 --greedy-every 500 --greedy-size 16 --save-every 250 --learning-rate 1e-4 --weight-decay 0.01 --rosa-dropout 0 --rosa-sign-flip 0.05 --ste-gradient-scale 0.5 --seed 321 --run-dir RWKV-v8/runs/add10_stream_stehalf_wd_20260926

# Matched scale-1 control (default behavior)
python RWKV-v8/rosa_add100_stream.py --digits 10 --context-len 48 --steps 1500 --batch-size 32 --validation-size 128 --eval-every 100 --greedy-every 500 --greedy-size 16 --save-every 250 --learning-rate 1e-4 --weight-decay 0.01 --rosa-dropout 0 --rosa-sign-flip 0.05 --ste-gradient-scale 1 --seed 321 --run-dir RWKV-v8/runs/add10_stream_stecontrol_wd_20260926

# One-batch CUDA forward/backward/optimizer smoke test
python RWKV-v8/rosa_add100_smoke.py

# Requested 200-step comparison, first condition
python RWKV-v8/rosa_add100_overfit.py \
  --rosa-dropout 0.1 --rosa-sign-flip 0 --steps 200

# Requested 200-step comparison, second condition
python RWKV-v8/rosa_add100_overfit.py \
  --rosa-dropout 0 --rosa-sign-flip 0.05 --steps 200

# Fresh random-stream 100-digit addition training
python RWKV-v8/rosa_add100_stream.py \
  --steps 5000 --batch-size 8 --weight-decay 0.01

# Reproduce the staged learning-rate schedule used in the recorded run
python RWKV-v8/rosa_add100_stream.py \
  --steps 250 --batch-size 8 --learning-rate 5e-5 --weight-decay 0.01 \
  --run-dir RWKV-v8/runs/add100_stream_repro
python RWKV-v8/rosa_add100_stream.py \
  --resume RWKV-v8/runs/add100_stream_repro/last.pt --steps 5000 \
  --batch-size 8 --learning-rate 2e-5 --weight-decay 0.01 \
  --run-dir RWKV-v8/runs/add100_stream_repro

# Replay the earlier no-weight-decay numerical failure conditions
python RWKV-v8/rosa_add100_stream.py \
  --steps 750 --batch-size 8 --learning-rate 1e-4 --weight-decay 0 \
  --rosa-sign-flip 0.05 --reproduce-unsafe-nan
```

The 100-digit training scripts require CUDA and compile the repository's WKV7 CUDA extension on first use. The recorded environment was PyTorch `2.14.0+cu130`, CUDA `13.0`, Numba `0.67.0`, and an NVIDIA GeForce RTX 3060.

## Limitations and next measurements

1. **No exact autoregressive success yet.** Every greedy checkpoint reported in the 100-digit experiments had 0/4 exact sums.
2. **Small generalization set.** The overfit harness evaluates on its four training examples. The streaming run used a fixed held-out set of only 16 additions, too small for a robust generalization estimate.
3. **Single-seed results.** The streaming experiment used one seed. It does not isolate the effects of weight decay, sign flipping, or learning rate.
4. **Small random-initialized model.** The prototype uses four blocks at width 128 and does not load the L4 arithmetic checkpoint or train a full RWKV model.
5. **Proxy quality is not established at scale.** The small binary toy had a separate 96.289% held-out match result, but the addition experiment reports no held-out ROSA match accuracy for learned `q/k/v` sequences.
6. **Inference path is not optimized.** `eval()` disables dropout and sign flips, and the forward value is exact ROSA, but the proxy is still evaluated. An exact-only inference path remains to be implemented and benchmarked.
7. **Numba target generation remains CPU-side.** The CUDA-to-CPU-to-CUDA transfer path was practical for the toy experiment; larger-scale throughput and batching need measurement.

Useful next evaluations are larger held-out random addition sets, multiple seeds for the streaming run, and per-position greedy diagnostics (first error position, answer length, and terminator accuracy). Those would distinguish proxy approximation from carry and termination failures.
