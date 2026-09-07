# Architecture

This document explains every non-obvious design decision in this codebase
and how the pieces fit together. It assumes you've read the Quickstart in
[README.md](README.md).

## 1. The task

> Given a graph, find a diametric path -- any one of the graph's longest
> shortest paths.

Formally: for a connected graph `G`, its diameter is
`max over all (s, t) of dist_G(s, t)` (BFS distance, unweighted). A
*diametric path* is any shortest `s->t` path whose length equals that
maximum. The model is given only the graph (no source/target); it must
output a full path, and that path's own endpoints and length are graded
against the graph.

## 2. Reference: ELF

[ELF](https://github.com/lillian039/ELF) ("Embedded Language Flows") is a
continuous diffusion language model: a frozen pretrained T5 encoder maps
text to a continuous embedding sequence, a transformer denoises that
sequence via flow matching (with a secondary decoder head that maps the
final denoised embeddings to discrete tokens via cross-entropy), and Muon is
the optimizer. The official repo's `main` branch is JAX; this project is
built against its `pytorch_elf` branch, which is what "follow the ELF setup
exactly (T5, Muon, etc.)" is grounded in here. Every module in `src/spelf/`
that has a close ELF analog says so in its docstring and names what changed
and why; this document is the higher-level version of the same story.

**What's preserved exactly:** the transformer block design (RMSNorm, RoPE +
QK-norm attention, SwiGLU FFN), the flow-matching objective (logit-normal
time sampling, `v = (x0 - z) / (1 - t)`), the dual decoder(CE)/denoiser(L2)
training branch selected per-example by a Bernoulli draw, self-conditioning
with a learned self-cond-CFG guidance target, prefix conditioning tokens
(time / self-cond-cfg / model-mode), the ODE/SDE sampler, and the Muon
optimizer's Newton-Schulz-orthogonalized update rule with its bias-corrected
Nesterov-Adam side channel for non-2D params.

**What's adapted for this domain:** everything about *what text* flows
through the system (Sections 3-4), the model's size (Section 5), and the
training operational surface -- bf16 autocast, DDP, gradient-accumulation
`no_sync()`, HF-Hub checkpoint upload, and PPL/BLEU/ROUGE online eval are
all dropped, since they exist in ELF for multi-host TPU/GPU-scale runs and
have no work to do at this project's CPU/MPS, single-process, ~1M-parameter
scale. Where a simplification changes behavior, the affected module's
docstring says so explicitly.

## 3. Data: graphs, splits, and the node-id label pool

### Generation (`spelf/graphgen.py`, `spelf/data_cache.py`)

Graphs are Erdos-Renyi `G(n, p)`: `n` nodes, each of the `n*(n-1)/2`
possible edges present independently with probability `p`. Two properties
this task needs that plain `G(n, p)` doesn't guarantee:

- **Connectivity** (a diameter is undefined otherwise): `p` is drawn per
  graph as a random multiple (`edge_prob_min_factor`..`edge_prob_max_factor`,
  default 1.2x-2.5x) of the Erdos-Renyi connectivity threshold `ln(n)/n`,
  clipped to `[edge_prob_floor, edge_prob_ceil]`. If the sampled graph still
  turns out disconnected (or has more than `max_edges` edges, to bound
  sequence length), it's rejected and resampled -- rejection sampling, not a
  different generative model, so the result is still "Erdos-Renyi" graphs,
  just conditioned on being usable.
- **A well-defined, reproducible ground-truth path** even when several
  pairs tie for the diameter, or several shortest paths connect the same
  pair (`find_diametric_example`, `shortest_path`): the diametric pair is
  the lexicographically-smallest `(s, t)` achieving the maximum distance,
  and the reported path is the lexicographically-smallest shortest path
  between them (greedily walk from `s`, always taking the smallest-id
  neighbor whose distance-to-`t` decreased by exactly one). Both rules are
  independent of adjacency-list iteration order, so the same graph always
  produces the same ground truth. (Ground truth is only used to build
  training targets -- eval scores the *model's* output against the graph
  directly, since several different paths can be equally correct diametric
  paths; see Section 8.)

### Node-id label pool

The task spec fixes ID = 6-10 nodes, OOD = 11-14 nodes, and separately
requires that node ids be drawn from the *full* OOD range so every id
appears during ID training. Concretely: the node universe is
`range(ood_max_nodes)` (default `range(14)`), and *every* graph -- ID or OOD
-- picks its `n` node labels as a random subset of that full range, not of
`range(n)`. An ID graph with 8 nodes might be labeled `{1, 4, 9, 11, 12, 13,
2, 7}` -- ids up to 13 appear routinely in 8-node graphs.

This is a deliberate control: it isolates *OOD by graph size/structure* from
*OOD by unseen vocabulary/labels*. Without it, an ID-trained model failing on
OOD graphs could conflate "it never saw node 12 used this way" with "it can't
reason about 14-node structures" -- this design removes that confound.
(With the real T5 tokenizer -- Section 4 -- individual digit/number *tokens*
like `"12"` are common regardless of graph size, since T5 was pretrained on
huge amounts of text containing numbers; what this control actually isolates
is whether the model has seen a given node *label* playing a structural role
-- an endpoint, a hub, a leaf -- during ID training, not merely whether its
token embedding is initialized. That's a meaningful ablation either way.)
`tests/test_label_pool.py` checks the coverage property directly.

### Splits

`scripts/generate_data.py` writes three JSONL files under `data_dir`, each
from an independent RNG stream (`data_seed`, `data_seed+1`, `data_seed+2`):
`train.jsonl` (ID), `id_val.jsonl` (ID, held out -- the in-distribution eval
set), `ood_test.jsonl` (OOD). Nothing is generated on the fly during
training or eval; every script reads these cached files.

### Serialization and tokenizer (`spelf/dataset.py`, `spelf/tokenizer.py`)

```
condition:  nodes: <n0> <n1> ... edges: <u0> - <v0> <u1> - <v1> ... find diametric path
target:     <p0> <p1> ... <pk>                          (+ EOS, appended at encode time)
```

The encoder is a real pretrained T5 (Section 4), so tokenization uses T5's
own vocabulary -- `transformers.T5TokenizerFast.from_pretrained("t5-small")`
-- rather than a bespoke one. Grammar keywords are lowercase (`nodes:`,
`edges:`, `find diametric path`) so common words tokenize as single pieces
under T5's SentencePiece vocab (e.g. `"▁edges"`, `"▁find"`) instead of
splitting the way their uppercase forms do; domain-specific words like
"diametric" still split into several subword pieces regardless of case
(`"▁di|a|metric"`), which is expected and harmless.

**Keeping node ids separate.** Every node id is single-space-delimited from
its neighbors on both sides -- as its own entry in the node list, and as the
two space-separated operands of `-` in each edge (`u - v`, never `u-v`).
Under a SentencePiece tokenizer, leading whitespace is *part of* a token
(`" 12"` -> `"▁12"`), so this spacing is what actually keeps adjacent node
ids from merging into one token or bleeding into each other:
`tokenizer("11 12")` -> `["▁11", "▁12"]`, two clean tokens, never
`["▁1112"]` or any token mixing digits from both. `scripts/
inspect_t5_tokenization.py` prints exactly this (`|` between tokens) for a
handful of examples and saves them to `data_sample.txt`;
`tests/test_tokenizer_roundtrip.py::test_adjacent_multi_digit_node_ids_stay_separate_tokens`
checks it holds for every node-id width (1 and 2 digits) programmatically.
One real quirk visible in that sample file: `"0"` specifically tokenizes as
two pieces (`"▁"` + `"0"`) rather than one, unlike other single digits --
harmless (decoding still reconstructs the exact original text), just a
reminder that "one token per id" isn't universally true even though ids stay
cleanly separated from each other.

Both condition and target are tokenized with `add_special_tokens=False`
(ELF's own minimal data-prep recipe), and an explicit EOS is appended to the
target afterward, since the model needs a learnable stopping signal at
generation time (`sampling.mask_after_eos` truncates each decoded sequence
at its first predicted EOS). The same tokenizer instance is used for the
condition (fed to the encoder) and the target (the CE decoder head's output
space over the *full* T5 vocabulary, ~32k tokens) -- matching ELF's own
setup, where encoder and decoder always share one vocabulary.

Batching (`make_collate_fn`) concatenates condition + target ids into one
`max_length`-padded sequence and derives three masks, matching ELF's
`data_utils.py` exactly: `cond_seq_mask` (1 at condition positions),
`attention_mask` (1 at any valid, non-pad position -- also the loss mask),
and `encoder_attention_mask` (condition tokens attend only to condition
tokens; target tokens attend to everything valid). That last mask is what
the frozen T5 encoder actually sees (Section 4) -- and it's run over the
*whole* concatenated sequence, condition and target together, which is the
detail that makes the next section make sense. `max_input_length=240` /
`max_length=264` are sized with margin above the worst case at
`max_edges=46`, `ood_max_nodes=14` (~222 condition tokens, ~16 target
tokens under T5's tokenizer -- see the largest example in `data_sample.txt`).

## 4. Encoder: a real pretrained, frozen T5

A subtlety worth stating explicitly, because it's easy to misread ELF's
`train_step.py`: the frozen T5 encoder is not merely a "read the prompt"
module. It's called once per training step over the *entire* concatenated
(condition + target) sequence, using `encoder_attention_mask` above. Its
output over the condition region becomes the pinned conditioning embeddings
(`cond_seq`); its output over the *target* region becomes `x0`, the clean
latent that the diffusion process is trained to denoise toward, and that the
CE decoder head is trained to reconstruct as discrete tokens. The encoder
thus defines the entire continuous embedding space the diffusion model
operates in, for both halves of the sequence.

**This is a real pretrained T5, not a custom one.** An earlier version of
this project pretrained its own small T5 from scratch (span corruption on
this project's own data) because it started from a closed, bespoke
vocabulary with no pretrained checkpoint to match. Once the tokenizer became
T5's own real vocabulary (Section 3), there both *is* a matching pretrained
checkpoint and no reason not to use it -- `transformers.T5EncoderModel.
from_pretrained("t5-small")`, frozen, exactly the call ELF's own `modules/
t5_encoder.py` makes (`spelf/t5_encoder.py::build_pretrained_encoder`). No
training happens here at all; weights are downloaded/cached by
`transformers` on first use, identically to real ELF.

**Latent normalization.** `encode_text` normalizes encoder outputs by
`(x - latent_mean) / latent_std`, matching ELF's `encoder_utils.encode_text`.
Since there's no local pretraining step to compute these as a byproduct of
anymore, `scripts/prepare_encoder.py` computes them directly: it runs the
frozen encoder over the *actual training collate pipeline*
(`PathDataset`/`get_dataloader`, the same concatenated condition+target
sequences and `encoder_attention_mask` that `train_step.py` will really use,
not condition text in isolation) for up to `latent_stats_sample_size`
examples, and takes the scalar mean/std of its outputs at valid positions.
These, plus which encoder/tokenizer names were used, are cached to
`config.encoder_profile_path` as a small JSON file -- not a weights
checkpoint, since the weights are just re-downloaded via `from_pretrained`
every run. `train.py`/`eval.py` read the profile back out and verify
`encoder_model_name` still matches the current config before trusting its
stats.

**Sizing.** `t5-small`: `d_model=512`, 6 encoder layers, ~35M frozen
parameters -- ELF's own default encoder. Since the diffusion backbone itself
is tiny (Section 5), most of the *trainable* ELF-XS model's parameters
aren't in the backbone at all but in the pieces sized off the encoder's
`d_model=512` and the ~32k-token vocabulary -- `bottleneck_dim`'s text
projection and especially `unembed_kernel` (`hidden_size x vocab`) -- the
same proportion real ELF-B has relative to its own T5 encoder and vocabulary.

## 5. Model: the ELF diffusion transformer (`spelf/dlm.py`, `spelf/modules.py`)

Ported block-for-block from ELF's `pytorch_elf` branch (`modules/model.py`,
`modules/layers.py`): pre-norm transformer blocks (RMSNorm -> QK-normed
multi-head attention with 1D RoPE -> RMSNorm -> SwiGLU FFN), learned prefix
tokens carrying time / self-cond-cfg / model-mode conditioning (prepended to
the sequence, RoPE-exempt via `num_empty_token`), a zero-initialized final
flow-matching output head, and a factored CE decoder head
(`hidden -> text_encoder_dim -> vocab`, sharing the backbone with the flow
head). The one dependency dropped is `einops` (`modules.py`'s `rotate_half`
and RoPE frequency doubling are each a one-line `reshape`/`repeat_interleave`
without it).

**A load-bearing contract, not obvious from the reference:** the RoPE
table's prefix budget (`num_empty_token`) is fixed at construction time from
`num_model_mode_tokens + num_time_tokens + num_self_cond_cfg_tokens`. Every
forward call must therefore supply *exactly* `max_length` non-prefix
positions, and if the model was built with `num_self_cond_cfg_tokens > 0`,
every call must pass `self_cond_cfg_scale` (never `None`) -- omitting it
prepends a shorter prefix than the RoPE table expects and produces a shape
mismatch several layers deep. `dlm.py` raises a clear `ValueError` for the
second case instead of letting it surface as a confusing RoPE broadcast
error (`tests/test_dlm.py::test_forward_requires_self_cond_cfg_scale_when_configured`
covers this). Every real call site (`train_step.py`, `sampling.py`) already
satisfies both halves of this contract, since batches are always padded to
`config.max_length` and `self_cond_cfg_scale` is computed unconditionally
whenever `num_self_cond_cfg_tokens > 0`.

**Model sizing.** ELF-B/M/L (105M/342M/652M params) target document-length
English text; the T5 vocabulary here is the same one ELF-B uses
(`t5-small`, ~32k tokens, Section 4), but this task's sequences are far
shorter (<=~264 positions vs. document-length OWT) -- so `ELF_models`
defines much smaller *backbone* presets instead:

| name | depth | hidden | heads | (ELF-B for reference) |
|---|---|---|---|---|
| `ELF-XS` (default) | 4 | 128 | 4 | depth 12, hidden 768, heads 12 |
| `ELF-S` | 6 | 192 | 6 | |
| `ELF-M` | 8 | 256 | 8 | |

`ELF-XS` at the default config is ~18M parameters total -- but, per
Section 4's sizing note, the vast majority of that is the vocabulary-sized
`unembed_kernel`, not the backbone itself; the backbone is fast enough to
iterate on CPU/MPS, which is the point.

## 6. Training objective (`spelf/train_step.py`, `spelf/sampling.py`)

One forward pass per step computes both heads on a mixed input; each
example in the batch independently draws **decoder** (CE) or **denoiser**
(L2) mode via a per-example Bernoulli at `decoder_prob` (default 0.5), and
the two losses are masked to their respective rows and combined with a
single shared denominator. This means every step trains both heads
(smoother gradients than alternating whole-batch mode), matching ELF's own
`train_step.py` exactly:

- **Denoiser (L2) branch**: flow-matching. `t ~ logit-normal` (or uniform);
  `z = t*x0 + (1-t)*noise`; target `v = (x0 - z) / clamp(1-t, t_eps)`; loss
  is `(v_pred - v_target)^2`. Condition positions are pinned to their clean
  embedding throughout (never noised, never predicted).
- **Decoder (CE) branch**: the input is a *separately* logit-normal-noised
  latent (`decoder_z`, always evaluated at `t=1` so the backbone knows it's
  in "decode" mode) and the loss is per-token cross-entropy against the true
  token id, from the factored decoder head.
- **Self-conditioning**: with probability `self_cond_prob` (default 0.5),
  the model additionally sees its own (no-grad) prediction of `x0` from a
  shared unconditional forward pass, concatenated as a second half of the
  input channel dimension -- this is why the model accepts inputs of width
  `C` or `2C`.
- **Self-cond-CFG guidance target**: when `num_self_cond_cfg_tokens > 0`
  (default on), the L2 target is further adjusted by a guidance term
  `(1 - 1/w) * (v_cond - v_uncond)` for a randomly sampled guidance strength
  `w` (log-uniform in `[1+self_cond_cfg_min, 1+self_cond_cfg_max]`) -- this
  trains the backbone to internalize a *range* of guidance strengths, so a
  single trained model supports classifier-free-guidance-style sampling at
  an arbitrary strength chosen at inference time (`config.self_cond_cfg_scale`).
- **Classifier-free guidance (label dropping)**: with probability
  `label_drop_prob` (default 0.1, nonzero unlike ELF's base default, so CFG
  sampling is actually meaningful for this always-conditional task), the
  condition is masked from the target's view *before* encoding, so the
  target's latent for dropped examples is genuinely unconditional -- what
  makes `cfg_scale > 1` sampling extrapolation valid.

Optimizer: Muon by default (Section 7); gradient clipping at global norm 1.0;
EMA of trainable parameters (`ema_decay1`, default 0.999) is what periodic
eval and `scripts/eval.py` actually evaluate (`_build_eval_model` loads the
EMA weights into an eval-mode copy), since EMA weights are standard practice
for evaluating diffusion models and is what ELF does too.

### Sampling (`spelf/sampling.py`)

Flow-matching ODE (deterministic Euler) or SDE (stochastic, `sde_gamma`
churn) rollout from Gaussian noise to a final latent, with condition
positions restored to their pinned embedding after every step; the *last*
step is always a plain ODE step regardless of `sampling_method`, matching
ELF. Self-conditioning and CFG at sampling time reuse the exact same
guided-forward machinery as training (`_forward_sample_self_cond`,
`_forward_sample`). The final latent is decoded to tokens by one extra
forward pass with `decoder_step_active=True` and `t=1` (`decode_batch`),
argmax over the CE head; `mask_after_eos` then truncates each sequence at
its first predicted EOS.

## 7. Optimizer: Muon (`spelf/muon.py`)

A from-scratch, single-device PyTorch implementation of Muon (2D parameters
get Newton-Schulz-orthogonalized momentum updates; everything else gets
bias-corrected Nesterov-Adam), rather than a port of ELF's
`utils/muon_utils.py`. That file wraps and monkey-patches an external `muon`
PyPI package plus `torch.distributed` all-gather logic for multi-host
training; this project runs single-device, so that machinery would be
dead weight and an external dependency to keep in sync. The algorithmic
core is preserved: 5-step quintic Newton-Schulz orthogonalization in fp32,
Nesterov momentum with bias correction, and `sqrt(max(1, fan_out/fan_in))`
shape-scaling of the update. That last detail matters here specifically:
`dlm.py`'s `proj_kernel`/`unembed_kernel` are bare 2D `nn.Parameter`s stored
`(in, out)` (so `x @ proj_kernel` works), the opposite convention from
`nn.Linear.weight`'s `(out, in)` -- `muon_with_aux_adam` detects which
convention each 2D parameter follows (by checking whether it *is* some
module's `nn.Linear.weight`) and flips the fan-in/fan-out ratio accordingly,
so the shape-scaling is correct for both layouts. `AdamW` is available as a
config alternative (`optimizer: adamw`) for comparison.

Note on Newton-Schulz orthogonalization: the quintic iteration used here is
tuned to pull singular values *toward* 1 within a handful of steps (Muon
uses 5); it is not an algorithm that converges to exact orthogonality as
`steps -> inf` the way a Newton iteration for the matrix sign function
would. `tests/test_muon.py` tests the properties that actually hold --
finite/shape-preserving output, singular-vector preservation (the update is
a matrix polynomial in `X X^T` applied to `X`, so it can only rescale
singular values, never rotate singular vectors), and that badly-scaled
singular values move toward 1 -- rather than asserting near-perfect
orthogonality, which the algorithm doesn't actually guarantee at this step
count.

## 8. Evaluation (`spelf/metrics.py`, `spelf/viz.py`)

Both `scripts/train.py` (every `eval_freq` epochs) and `scripts/eval.py`
(standalone, given a checkpoint) run the identical pipeline over the ID
(`id_val.jsonl`) and OOD (`ood_test.jsonl`) splits: generate via the full
sampler, decode to text (`tokenizer.decode(..., skip_special_tokens=True)`),
parse back into a candidate path (`dataset.parse_path_text` -- returns
`None`, not a best-effort partial parse, if any whitespace-separated piece
of the decoded body isn't a bare node-id, since a stray word or punctuation
mid-path is a formatting error, not something to silently paper over), then
score against the graph directly with four
independent boolean checks (`metrics.evaluate_path`):

1. **valid_path**: every node exists in the graph, no node repeats, and
   every consecutive pair is an actual edge.
2. **shortest_path**: `valid_path` AND the output's length equals the BFS
   distance between its own two endpoints -- it doesn't have to match the
   *stored* ground-truth path (several may exist), only genuinely be *a*
   shortest path between wherever it says it starts and ends.
3. **correct_length**: the output's length equals the graph's diameter.
   Deliberately independent of validity -- a structurally wrong sequence
   that happens to have the right length still scores here, which is useful
   for distinguishing "the length head is right but the path head isn't"
   from "nothing is right."
4. **optimal_path**: `valid_path AND shortest_path AND correct_length` --
   the output is a genuine diametric path of the graph (not necessarily
   the specific path stored as ground truth, any one that qualifies).

Rates over the eval batch are logged as `eval_id/*_rate` /
`eval_ood/*_rate` (`final_eval_id/*` / `final_eval_ood/*` from
`scripts/eval.py`, plus `wandb.summary` entries since those are one-off
scores rather than a training-time series). `eval_num_examples` controls how
many examples are scored per split (`-1` = full split); `eval_num_viz`
(default 5, matching the task spec) controls how many of those are also
rendered as images.

Visualization (`viz.render_graph_path`, `networkx` for layout only) draws
the full graph in gray, with the predicted path's edges/interior nodes
highlighted and its first/last nodes marked distinctly, titled with the
diameter and the four metric values; `wandb_images_for_examples` wraps each
as a `wandb.Image` **captioned with the model's raw decoded text output**,
per the task spec, and logs them under `eval_{id,ood}/samples`.

## 9. Checkpointing & wandb resume (`spelf/checkpoint.py`, `spelf/train_state.py`)

Checkpoints (`checkpoint_<step>.pt` under `output_dir`, `keep_last=3`
retained) hold the model state dict, EMA params, optimizer state, LR
scheduler state, step/epoch, the training RNG's state, and **the wandb run
id**. That last field is the mechanism behind "saving and resuming synced
with the wandb ID": `resolve_run` auto-detects the latest checkpoint in
`output_dir` (or an explicit `--config_override resume=<path>`), and if one
exists, `peek_wandb_run_id` reads its stored id *before* `wandb.init` is
called, which then passes `id=<that id>, resume="allow"`. Resuming a run
therefore reattaches to the exact same wandb run automatically -- nothing
for the user to track by hand, unlike ELF's reference training script, which
requires manually passing the same `--wandb_run_name` on every resume for
its `id=` to line up. A fresh run (no checkpoint found) mints a new id via
`generate_wandb_run_id()` (wraps `wandb.util.generate_id`, with a fallback
for older/newer wandb versions where that moved), and it's saved into every
checkpoint from that point on. `scripts/eval.py` reattaches the same way
when `use_wandb` is on, so a standalone evaluation's summary metrics land in
the training run's history rather than opening a disconnected run.

One CPU-vs-device wrinkle worth flagging: `torch.Generator()` is CPU-only,
but `load_checkpoint(..., device=...)` uses that same `device` as
`map_location` for the whole checkpoint payload. The saved RNG state tensor
is explicitly moved back to CPU (`.cpu()`) before `generator.set_state()`,
which requires a CPU `ByteTensor` specifically --
`tests/test_checkpoint.py::test_load_checkpoint_generator_state_survives_non_cpu_map_location`
is a regression test for this (skipped when no non-CPU device is available).

Resume granularity is epoch-level, not mid-epoch: `state.epoch` records
*completed* epochs, and `save_freq < 1` (fractional, intra-epoch saving) is
still supported, but resuming from a fractional-epoch checkpoint restarts
that epoch from its beginning rather than replaying ELF's mid-epoch
batch-skip logic. Since training data is IID-sampled per epoch anyway, this
costs at most one epoch's worth of redundant compute on resume, in exchange
for real simplicity -- worth it at this project's scale (default
`save_freq=1`, whole epochs, where the distinction doesn't even arise).

## 10. Configuration (`spelf/common.py::Config`)

A single flat dataclass (rather than ELF's base-`Config` + YAML-overlay +
separate `SamplingConfig` list), since this project has one model family and
one task rather than ELF's several (OWT/WMT/XSum, three model sizes, a
sampling-config sweep). Every field is documented inline in `common.py`
next to its default, grouped by the same sections as this document (graphs,
serialization, encoder, model, denoiser/decoder objective, conditioning/CFG,
optimization, sampling, eval, logging/checkpointing, wandb, misc).
`configs/default.yml` is a template overlay -- every field in it already
matches the dataclass default; it exists to show which knobs are worth
touching first, and as something to copy and edit for a new experiment.
`--config_override field=value` (repeatable) on every script applies ad hoc
overrides on top of a YAML file, coercing `value` to the field's current
type.
