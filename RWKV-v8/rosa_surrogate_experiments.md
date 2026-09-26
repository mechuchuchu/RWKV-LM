# ROSA Surrogate Experiments

**Status:** exploratory prototype; the 100-digit addition task is not solved yet. | **Last updated:** 2026-09-26

This note records the motivation, implementation, measurements, and limitations of the experiments around using a differentiable model as a training-time surrogate for the discrete ROSA operator. It includes both the small binary-sequence validation and the RWKV7-based 100-digit addition toy task.

## Summary

The core idea is to preserve ROSA's exact discrete value in the forward pass while using a learned differentiable surrogate to provide a backward gradient. A small binary-sequence experiment provided an initial proof of concept: the causal Transformer surrogate matched the exact ROSA output on 96.289% of held-out positions, and a fixed-pattern upstream value projection improved from 0% to 100% exact-task accuracy using the surrogate gradient.

The 100-digit addition prototype now runs a four-block, randomly initialized RWKV7+ROSA model on exact 100-digit operands. Numba made the exact binary ROSA teacher practical for this toy run. The model can reduce teacher-forced loss on four repeated examples, but none of the configurations tested has produced a correct autoregressive sum. The best of the two 200-step configurations compared most recently—dropout 0 with a 5% ROSA sign-flip probability—reached 81.281% teacher-forced token accuracy, while greedy exact-sum accuracy remained 0/4.

These results are evidence that the prototype and gradient path execute, not evidence that the model has learned general 100-digit addition. The 100-digit runs use four training examples, have no held-out arithmetic set, and do not load a pretrained RWKV checkpoint.

## Hypothesis and gradient construction

ROSA's binary matching operation is discrete, so its exact output does not provide a useful ordinary gradient to upstream model parameters. The experiment uses a differentiable proxy during training while keeping the exact operator's output as the forward value.

For exact output `y_exact` and proxy output `y_proxy`, the straight-through construction is:

```python
y = y_exact + y_proxy - y_proxy.detach()
```

The forward value is `y_exact`, because the last two terms cancel numerically. The backward pass sees the derivative of `y_proxy`. The proxy is also trained directly with an auxiliary cross-entropy loss against exact ROSA targets:

```text
total_loss = task_loss + lambda_rosa * rosa_distillation_loss
```

The small binary toy uses a one-hot exact output in this construction. The addition prototype uses an exact binary ROSA result encoded as `-1/+1`, with the proxy's two-class softmax converted to a signed signal. In both cases, the ROSA distillation loss is computed over proxy logits and exact targets.

The intended deployment idea is to use the exact ROSA operator at inference. The current prototype already uses the exact result as its forward value, but it still evaluates the proxy network and constructs its proxy logits in `eval()` mode. Replacing or bypassing the proxy computation in an inference-only path has not been implemented.

## Experiment files

| File | Purpose |
| --- | --- |
| [`rosa_surrogate_toy.py`](rosa_surrogate_toy.py) | Small binary-sequence experiment for exact-forward/surrogate-backward behavior, plus a controlled upstream value-projection check. |
| [`rosa_add100_toy.py`](rosa_add100_toy.py) | Generator, token layout, and self-check for exact 100-digit addition examples. |
| [`rosa_numba.py`](rosa_numba.py) | Numba-compiled exact binary ROSA implementation used to generate targets efficiently. |
| [`rosa_add100_smoke.py`](rosa_add100_smoke.py) | One-batch CUDA smoke model: four RWKV7+ROSA blocks, a causal Transformer proxy per ROSA block, and the custom WKV7 CUDA kernel. |
| [`rosa_add100_overfit.py`](rosa_add100_overfit.py) | Repeated-batch optimizer test with teacher-forced metrics and autoregressive greedy decoding. |

The 100-digit scripts do not save checkpoints. Each invocation starts from a fresh initialization and regenerates the same four examples from seed 321.

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

## Reproduction commands

From the repository root:

```bash
# Binary surrogate experiment
python RWKV-v8/rosa_surrogate_toy.py

# Validate 100-digit data and target alignment
python RWKV-v8/rosa_add100_toy.py --self-check --samples 512 --seed 42

# One-batch CUDA forward/backward/optimizer smoke test
python RWKV-v8/rosa_add100_smoke.py

# Requested 200-step comparison, first condition
python RWKV-v8/rosa_add100_overfit.py \
  --rosa-dropout 0.1 --rosa-sign-flip 0 --steps 200

# Requested 200-step comparison, second condition
python RWKV-v8/rosa_add100_overfit.py \
  --rosa-dropout 0 --rosa-sign-flip 0.05 --steps 200
```

The 100-digit training scripts require CUDA and compile the repository's WKV7 CUDA extension on first use. The recorded environment was PyTorch `2.14.0+cu130`, CUDA `13.0`, Numba `0.67.0`, and an NVIDIA GeForce RTX 3060.

## Limitations and next measurements

1. **No exact autoregressive success yet.** Every greedy checkpoint reported in the 100-digit experiments had 0/4 exact sums.
2. **No generalization set.** The overfit harness trains and evaluates on the same four examples. A held-out random addition set has not been measured.
3. **Tiny and stochastic comparison.** Each regularization condition has one seed and a single four-example batch. Dropout and sign flipping add stochasticity.
4. **Only a small random-initialized model.** The prototype uses four blocks at width 128 and does not load the L4 arithmetic checkpoint or train the full RWKV model.
5. **Proxy quality is not yet established at scale.** The small binary toy had a separate 96.289% held-out match result, but the 100-digit experiment reports only training-time ROSA distillation cross-entropy, not held-out ROSA match accuracy for learned `q/k/v` sequences.
6. **Inference path is not optimized.** `eval()` disables dropout and sign flips, and the forward value is exact ROSA, but the proxy is still evaluated. An exact-only inference path remains to be implemented and benchmarked.
7. **Numba target generation remains CPU-side.** The current CUDA-to-CPU-to-CUDA transfer path was practical for the toy experiment; larger-scale throughput and batching need measurement.

Useful next evaluations are a held-out random addition set, multiple seeds for the two 200-step settings, and per-position greedy diagnostics (first error position, answer length, and terminator accuracy). Those would distinguish memorization, proxy approximation, and autoregressive carry/termination behavior.
