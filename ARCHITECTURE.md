# Architecture & Reference

This is the systematic companion to [`README.md`](README.md) (which tells the narrative
story of the project). This document is a reference: every file's role, the exact data
flow from raw graph to model batch, every design decision with its rationale, and every
tunable and non-tunable variable in the codebase. Line numbers are omitted (files change);
function/class names are given so you can jump to them with your editor's symbol search.

## 1. Pipeline overview

Three independent stages, run in this order, each producing artifacts the next stage
reads from disk:

```
scripts/generate_data.py                    scripts/pretrain_encoder.py
        |                                            |
        v                                            v
  data/*.pt, data/meta.json  ------------->  runs/encoder/checkpoint_*.pt
        |                                            |
        |                     +----------------------+
        |                     |
        v                     v
scripts/train_dlm.py    scripts/train_arlm.py
        |                     |
        v                     v
  runs/dlm/checkpoint_*.pt  runs/arlm/checkpoint_*.pt
        |                     |
        +----------+----------+
                   |
                   v
   scripts/eval_only.py | eval_venn.py | dataset_stats.py
```

`scripts/dataset_stats.py` only depends on stage 1's output and can be run any time
after `generate_data.py`. Everything downstream of stage 2 (`pretrain_encoder.py`)
depends on a completed encoder checkpoint, because both `train_dlm.py` and
`train_arlm.py` load it via `--encoder_ckpt` and freeze it (`spelf/encoder.py`'s
`load_frozen_encoder`) before their own training loop starts.

## 2. Repository map

| File | Role |
|---|---|
| [`src/spelf/tokenizer.py`](src/spelf/tokenizer.py) | Vocabulary, sequence encode/decode, fixed-length formulas. Single source of truth for parsing token ids, ground-truth or model-generated alike. |
| [`src/spelf/graphgen.py`](src/spelf/graphgen.py) | Random connected graph sampling, diametric-pair enumeration, deterministic shortest-path tie-break. |
| [`src/spelf/data_cache.py`](src/spelf/data_cache.py) | Orchestrates graph sampling → diametric-path selection → tokenization → cached tensors + `meta.json`. |
| [`src/spelf/dataset.py`](src/spelf/dataset.py) | Thin `torch.utils.data.Dataset` wrappers around the cached tensors. |
| [`src/spelf/modules.py`](src/spelf/modules.py) | Shared building blocks: `SharedEmbedding`, `LearnedPosEnc`, `TimeEmbedding`, `EncoderLayer`, `DecoderLayer`. |
| [`src/spelf/encoder.py`](src/spelf/encoder.py) | `GraphEncoder` + its masked-language-model (MLM) pretraining objective. |
| [`src/spelf/t5_encoder.py`](src/spelf/t5_encoder.py) | Optional alternative conditioning encoder: frozen pretrained HuggingFace T5 over a text serialization of the graph. |
| [`src/spelf/dlm.py`](src/spelf/dlm.py) | `DLMDecoder` — ELF-style rectified-flow diffusion decoder (loss + Euler sampler). |
| [`src/spelf/arlm.py`](src/spelf/arlm.py) | `GPTDecoder` — causal autoregressive decoder (loss + greedy sampler). |
| [`src/spelf/metrics.py`](src/spelf/metrics.py) | Decodes raw generations and scores token-accuracy / exact-match / valid / shortest / correct-length / optimal rates. Model-agnostic. |
| [`src/spelf/viz.py`](src/spelf/viz.py) | Ground-truth-vs-generated two-panel graph plots, for wandb image logging. |
| [`src/spelf/common.py`](src/spelf/common.py) | Seeding, device selection, optimizer/LR-schedule construction, checkpoint save/load/resume, early stopping, wandb init, uniform `(context, context_mask)` dispatch across encoder backends. |
| [`scripts/generate_data.py`](scripts/generate_data.py) | CLI wrapper around `data_cache.generate_all`. |
| [`scripts/pretrain_encoder.py`](scripts/pretrain_encoder.py) | Trains `GraphEncoder` via MLM. |
| [`scripts/train_dlm.py`](scripts/train_dlm.py) | Trains `DLMDecoder` against a frozen encoder. |
| [`scripts/train_arlm.py`](scripts/train_arlm.py) | Trains `GPTDecoder` against a frozen encoder (same shape as `train_dlm.py`, minus diffusion-specific flags). |
| [`scripts/eval_only.py`](scripts/eval_only.py) | Standalone checkpoint evaluation: prints/logs rates, optionally saves example plots. |
| [`scripts/eval_venn.py`](scripts/eval_venn.py) | Standalone checkpoint evaluation rendered as a Venn diagram (valid/shortest/correct-length/optimal), ID + OOD. |
| [`scripts/dataset_stats.py`](scripts/dataset_stats.py) | Computes and writes per-split token-count and graph-connectivity statistics for a generated dataset. |
| [`tests/`](tests) | Unit tests: tokenizer round-trip, BFS tie-break, graph-generator connectivity, node-label-pool coverage. |

## 3. Data flow, end to end

**Stage A — graph → tensors** (`generate_data.py` → `data_cache.generate_all`):

1. For each split, `graphgen.sample_graph(n, avg_degree, rng, label_pool_size=tok.MAX_NODES)`
   draws an Erdős–Rényi graph on `n` nodes, reject-sampled until connected, then relabels
   its nodes with a random `n`-subset of `{0..MAX_NODES-1}` (see §5.3 for why).
2. `graphgen.graph_diameter` and `graphgen.diametric_pairs` compute the graph's diameter
   and every node pair that achieves it; `graphgen.sample_diametric_paths` picks up to
   `max_paths_per_graph` of those pairs and computes each one's canonical path via
   `graphgen.shortest_path_lexsmallest` (deterministic tie-break, see §5.2).
3. `data_cache.generate_labeled_split` turns each `(graph, path)` pair into an example
   dict (`n`, shuffled `edges`, `node_list`, `path`, `diameter`, `graph_id`);
   `generate_unlabeled_inputs` does the same without a path, for encoder pretraining.
4. `data_cache.build_labeled_tensors` / `build_unlabeled_tensors` call
   `tokenizer.encode_input` / `encode_target` / `pad_input` on every example and stack
   the results into fixed-shape tensors (`input_ids`, `input_mask`, `target_ids`,
   `n_nodes`, `graph_id`).
5. Tensors are `torch.save`d as `train.pt` / `val_id.pt` / `val_ood.pt` /
   `encoder_pretrain_extra.pt`; generation config + realized sizes go to `meta.json`
   (read back everywhere via `dataset.load_meta`).

**Stage B — tensors → training batch** (`dataset.py`, used by all three training
scripts and both eval scripts):

- `dataset.PathDataset` wraps one of the three labeled `.pt` files; each `__getitem__`
  returns one example's `input_ids`/`input_mask`/`target_ids`/`n_nodes`/`graph_id`.
- `dataset.EncoderPretrainDataset` concatenates `train.pt`'s inputs with
  `encoder_pretrain_extra.pt`'s inputs (no path fields) — this is the *only* place those
  two files' inputs are combined; the decoders never see `encoder_pretrain_extra.pt`.
- `dataset.collate_fn` is the default per-key stack (examples are already fixed-shape and
  pre-padded), named explicitly for clarity at every `DataLoader(..., collate_fn=...)`
  call site.

**Stage C — batch → conditioning context** (`common.encode_context`, used by every
training/eval loop):

- Runs the frozen encoder (`GraphEncoder` or `T5GraphEncoder`) on `(input_ids,
  input_mask)` and returns `(context, context_mask)` uniformly regardless of which
  encoder backend is in play (`GraphEncoder`'s output sequence is length-aligned with
  `input_ids`, so `context_mask := input_mask`; `T5GraphEncoder` re-tokenizes the graph
  as text and returns its own mask — see §6.5).
- Always wraps the call in `torch.no_grad()` **except** when the encoder declares
  `context_requires_grad = True` (only `T5GraphEncoder`, whose frozen T5 stack still
  needs gradient to flow to its trainable projection/embedding).

**Stage D — context + target → loss, or context → generation:**

- Training: `dlm.loss(model, context, context_mask, target_ids, ...)` or
  `arlm.loss(model, context, context_mask, target_ids)` — see §6.6/§6.7.
- Inference/eval: `model.generate(context, context_mask, **sample_kwargs)` — both
  `DLMDecoder` and `GPTDecoder` expose this identical interface, so `metrics.run_eval`
  and every eval script call it without knowing which decoder it is.

**Stage E — generation → metrics/plots** (`metrics.py`, `viz.py`):

- `metrics.evaluate_generation` decodes a raw generated id sequence with
  `tokenizer.decode_target` (returns `None` on anything malformed, never raises), rebuilds
  the graph from the input with `tokenizer.decode_input`, and scores `valid` /
  `shortest` / `correct_length` / `optimal` / `exact_match` / `token_accuracy` (see §7 for
  exact definitions).
- `metrics.run_eval` batches this over a `DataLoader`, returning aggregate rates plus a
  list of per-example dicts.
- `viz.plot_example` takes one such per-example dict and renders a two-panel
  ground-truth-vs-generated `networkx.spring_layout` figure, laid out deterministically
  from the graph's own content (`viz._content_seed`) so the same graph always looks the
  same across training steps.

## 4. Vocabulary & sequence format (non-tunable)

Defined entirely in [`tokenizer.py`](src/spelf/tokenizer.py); nothing here is a CLI flag.

```
Token ids:  0 <PAD>  1 <G>  2 <E>  3 <N>  4 <P>  5 <EOS>  6 <MASK>   7..20 node-id 0..13
Input:      <G> u1 v1 <E> u2 v2 <E> ... <N> n0 n1 ... n(k-1)
Target:     <P> p0 p1 ... p(L-1) <EOS> <PAD> <PAD> ...
```

- `MAX_NODES = 14` — the largest graph size the OOD split uses; fixes the vocabulary so
  ID and OOD share every node-id token (no OOD-vocab problem).
- `VOCAB_SIZE = 7 + MAX_NODES = 21`.
- `TARGET_LENGTH = MAX_NODES + 2 = 16` — one global fixed target canvas (`<P>` + up to
  `MAX_NODES` path nodes + `<EOS>`) used for *every* graph size, ID or OOD. This isn't
  just a padding convenience: the DLM diffuses over a fixed-size tensor `[L_TGT, D]`, so
  the target length must be a single constant across the whole dataset, not a
  per-example `n + 2`.
- Input length is per-example and varies: `input_length(n, m) = n + 3m + 2`
  (`<G>` + 3 tokens/edge + `<N>` + `n` node-id tokens). `data_cache.max_input_length`
  takes the max over *all four splits* once at generation time and records it as `l_in`
  in `meta.json`; every input is padded out to that one shared value so encoder tensors
  have one constant shape too.
- `decode_input`/`decode_target` are the **single source of truth** for parsing ids back
  into a graph/path — used identically for ground truth (at data-generation time) and
  for possibly-malformed model output (at eval time). Both return `None` on any
  structural violation (missing/duplicated markers, non-node tokens in the wrong place,
  content after `<EOS>`/`<N>`-list, etc.) rather than raising, so a broken generation is
  just scored as invalid, never crashes an eval loop.

## 5. Data-generation design decisions

Source: [`graphgen.py`](src/spelf/graphgen.py), [`data_cache.py`](src/spelf/data_cache.py).

### 5.1 The task itself: diametric path, not shortest path

The input encodes *only* a graph — no query pair. The model must find *some* pair of
nodes and a path between them whose length equals the graph's diameter. This is checked
independently by `graphgen.diametric_pairs` (all pairs at maximum pairwise distance) —
there is often more than one, so a graph can validly train on (and be scored against)
several different correct answers.

### 5.2 Deterministic shortest-path tie-break

`graphgen.shortest_path_lexsmallest(G, start, end)`: BFS distances computed *from `end`*,
then the path is reconstructed from `start` by always stepping to the smallest-id
neighbor whose distance-to-`end` is exactly one less than the current node's. This is a
greedy algorithm that provably yields the lexicographically-smallest shortest path (every
candidate considered at each step lies on *some* shortest path by the distance
invariant, and picking the smallest can never rule out a smaller full path later). It
exists purely so that training targets are **reproducible** given a seed — without a
tie-break, "the" shortest path between a diametric pair would be an arbitrary function of
BFS traversal order.

### 5.3 Node labels drawn from a pool wider than the graph

`sample_graph(n, avg_degree, rng, label_pool_size=tok.MAX_NODES)` relabels a freshly
sampled `n`-node graph with a random `n`-subset of `{0, ..., MAX_NODES-1}` rather than
contiguous `0..n-1`. Consequence: **graph size is the only axis that distinguishes ID
from OOD** — every node-id *token value* the model can be asked to emit already appears
in ID-sized (6–10 node) training graphs, so OOD evaluation (11–14 nodes) tests
generalization to larger search spaces, not to literally unseen output tokens.
`data_cache.assert_full_node_id_coverage` asserts this holds for `train.pt` after every
generation run — a bug here would produce a `sample_graph`-not-`generate_all` assertion
failure with an explicit message, so it's treated as a real bug if it ever fires rather
than expected sampling variance (with `label_pool_size=14` drawn independently across
50,000 training graphs, the chance any single node-id is never drawn is astronomically
small).

Because `_rebuild_graph`/`viz.plot_example` iterate a graph's real node set as
`node_list` (not `range(n)`), this labeling scheme required threading `node_list`
explicitly through the tokenizer, data cache, metrics, and viz code — see the comments
in `metrics._rebuild_graph` and `viz.plot_example` for the specific failure mode this
avoids (spurious "phantom" nodes if `range(n)` were used instead).

### 5.4 Held-out splits generated *first*, train resampled around them

`data_cache.generate_all` samples `val_id`/`val_ood` before `train`, collects their
`graph_key()`s into `held_out_keys`, and passes that as `forbidden_keys` into
`generate_labeled_split`/`generate_unlabeled_inputs` for `train` and
`encoder_pretrain_extra` — each reject-resamples (up to `max_resample_tries=200`) any
graph that collides with a held-out key. This ordering matters: for small `n` (e.g. 6
nodes has only 15 possible edges), the space of likely Erdős–Rényi outcomes is small
enough that generating train first and merely *checking* disjointness afterward would
fail in practice — birthday-paradox collisions between a 50,000-graph train split and a
500-graph val split are near-guaranteed at that scale. Resampling around a reserved set
this small succeeds within a handful of tries even though the underlying space is small.
`data_cache.assert_disjoint_graphs` re-verifies disjointness explicitly after generation
as a hard assertion, independent of whether resampling worked as intended.

`graph_key(n, edges)` is the canonical, order/direction-independent graph identity used
throughout: `(n, frozenset(sorted edge tuples))`.

### 5.5 Multiple paths per graph, capped

Each graph contributes one training example per distinct diametric pair, up to
`max_paths_per_graph` (subsampled if more exist, all of them if fewer). A graph whose
diameter is achieved by several pairs (e.g. every antipodal pair on an even cycle) thus
trains the model on all of its correct answers, while a graph with a unique diametric
pair contributes exactly one example — never padded to look like it has `k` paths. The
same cap, reinterpreted, controls how many independently edge-order-shuffled copies of
each graph appear in the *unlabeled* encoder-pretraining corpus (`generate_unlabeled_inputs`),
where there's no notion of "path" to multiply by at all.

Consequence documented and verified in `dataset_stats.py`'s output (see
`data/dataset_stats.txt`): because low-diameter graphs systematically have more
diametric pairs than high-diameter graphs, an **example-weighted** statistic (e.g. mean
diameter implied by target lengths) differs from the corresponding **graph-weighted**
statistic (mean diameter over unique graphs) — not a bug, an expected artifact of this
per-graph example multiplicity.

### 5.6 Encoder pretraining's extra unlabeled OOD corpus

`encoder_pretrain_extra.pt` (unlabeled, 11–14 node graphs) exists solely because
`GraphEncoder` uses **learned** positional encodings: if pretraining only ever saw 6–10
node inputs, positions past the 6–10-node range would be untrained garbage when the
encoder is later asked (at DLM/ARLM eval time) to encode an 11–14 node OOD graph — an
artifact that would contaminate exactly the ID-vs-OOD comparison this project measures.
Critically, this corpus carries **no path/answer labels** — the encoder never sees
diametric-path supervision, so this doesn't leak OOD reasoning-task signal into the
comparison, only positional/structural exposure. `dataset.EncoderPretrainDataset`
concatenates it with `train.pt`'s inputs (paths dropped) at load time; it's never
touched by `train_dlm.py`/`train_arlm.py`.

### 5.7 Seeding scheme

One `--seed` fans out into four independent `numpy.random.default_rng` streams via fixed
large offsets (`data_cache.SEED_OFFSETS`: train `+0`, val_id `+1_000_000`, val_ood
`+2_000_000`, encoder_pretrain_extra `+3_000_000`) so each split's sampling is
reproducible and independent of the others under a shared top-level seed. `graph_id`
values are similarly kept in disjoint numeric ranges per split (train starts at 0,
val_id at `10**9`, val_ood at `2*10**9`) purely so ids never collide across splits if
ever concatenated — not otherwise meaningful.

## 6. Model architecture design decisions

Source: [`modules.py`](src/spelf/modules.py), [`encoder.py`](src/spelf/encoder.py),
[`dlm.py`](src/spelf/dlm.py), [`arlm.py`](src/spelf/arlm.py),
[`t5_encoder.py`](src/spelf/t5_encoder.py).

### 6.1 One shared embedding table across encoder + both decoders

`modules.SharedEmbedding` wraps a single `nn.Embedding(vocab_size, d_model)`. It is
constructed once (inside `GraphEncoder.__init__` or standalone in
`pretrain_encoder.py`), pretrained as part of the encoder's MLM objective, and then
passed **by object reference** into whichever decoder is being trained
(`DLMDecoder(..., embedding=encoder.embedding)` / same for `GPTDecoder`) — not copied.
This mirrors how ELF diffuses directly inside a frozen pretrained embedding space
(there, T5's; here, the from-scratch graph encoder's own).

The wrinkle: `<P>` and `<EOS>` never appear in the encoder's own input (only in
decoder path targets), so encoder pretraining would never touch those two rows.
`SharedEmbedding.freeze_pretrained_rows(trainable_token_ids=(tok.P, tok.EOS))` installs a
gradient hook that zeros the gradient for every row *except* those two, called once
by `GraphEncoder.freeze()` right after encoder pretraining finishes. Combined with
excluding embeddings from weight decay in `common.build_optimizer` (necessary — AdamW's
weight decay would otherwise keep shrinking the "frozen" rows despite their zeroed
gradient), this makes the shared table a true frozen copy of what the encoder learned,
except for two rows that keep training downstream.

Embedding init: `nn.init.normal_(weight, mean=0.0, std=0.02)` — the conventional
small-std init (BERT/GPT-2 style). Chosen explicitly over PyTorch's default `N(0,1)` per
element, because a 128-dim row at unit variance has norm ≈√128, and dotting two such rows
in `SharedEmbedding.unembed` (`x @ weight.T`) would produce huge, poorly-calibrated
logits before any training happens.

### 6.2 `EncoderLayer` vs `DecoderLayer`: one shared conditioning mechanism

Both `DLMDecoder` and `GPTDecoder` use the identical `modules.DecoderLayer` (self-attn →
cross-attn to `context` → MLP, Pre-LN), differing only in `causal` (`False` for DLM,
`True` for ARLM, applied via a `-inf`-filled upper-triangular additive attention mask).
This is a deliberate experimental control: since both decoders condition on the graph via
the exact same cross-attention block wired to the exact same frozen encoder output, the
decoder's *own* architecture (diffusion vs. autoregressive) is isolated as the one
variable under study — a difference in results can't be attributed to a different
conditioning mechanism.

`modules.EncoderLayer` (used only by `GraphEncoder`) is simpler: bidirectional
self-attention + MLP, no cross-attention, no causal mask.

### 6.3 Attention dropout hardcoded to 0, MLP/residual dropout tunable

Every `nn.MultiheadAttention` in this codebase (`EncoderLayer.self_attn`,
`DecoderLayer.self_attn`/`cross_attn`) is constructed with `dropout=0.0` **regardless**
of the `--dropout` CLI flag — a hard platform constraint, not a design choice:
`nn.MultiheadAttention`'s scaled-dot-product-attention fast path raises
`NotImplementedError` for `dropout_p>0` during training on MPS. Regularization for these
sublayers still comes from `self.dropout` (the CLI-tunable `--dropout`, default 0.1)
applied to each sublayer's *output* before the residual add.

### 6.4 GraphEncoder pretraining objective (masked node-token prediction)

`encoder.mlm_forward`: masks a random `~mlm_prob` fraction of **node-id token
positions only** (edge endpoints, `<N>`-list entries) — structural markers
(`<G>`/`<E>`/`<N>`) are never masked, since the useful self-supervised signal here is
relational understanding of node identity/connectivity, not of the fixed, trivially
predictable grammar. Masked positions get replaced with `<MASK>`; loss is cross-entropy
via the tied unembedding (`model.embedding.unembed`) at masked positions only
(`ignore_index=-100` elsewhere).

`GraphEncoder.freeze()` sets `requires_grad=False` on every parameter except the
embedding table (which keeps `requires_grad=True` module-wide so `<P>`/`<EOS>` can still
train downstream, gated instead by the gradient hook from §6.1) and calls `.eval()`.
Callers (`common.encode_context`) must still wrap the encoder's forward pass in
`torch.no_grad()` during downstream DLM/ARLM training even though the module's own
`requires_grad` flags say frozen — because the embedding table's `<P>`/`<EOS>` rows are
*not* frozen, and this conditioning path should never contribute gradient to them (they
should only train via the decoder's own target-embedding lookups).

### 6.5 Optional T5 conditioning encoder (`T5GraphEncoder`)

An alternative to `GraphEncoder`, selected via `--encoder_kind t5`. Closer to ELF's
actual paper setup (a frozen pretrained *text* encoder) than this project's default
from-scratch graph encoder, letting the comparison ask "is a strong pretrained text
encoder a better or worse conditioning source than a small from-scratch graph encoder?"
while holding the decoder fixed.

Key differences from `GraphEncoder`, all consequences of T5 operating on natural-language
subwords rather than this project's own 22-token graph vocab:
- `T5GraphEncoder.forward` first serializes the decoded graph to an English sentence
  (`graph_to_text`) and retokenizes with T5's own tokenizer — so its output
  sequence length/mask are **unrelated** to `input_mask`, unlike `GraphEncoder` (whose
  output is length-aligned with its own input). This is exactly why
  `common.encode_context` exists as a uniform dispatch point rather than every call site
  assuming `context_mask := input_mask`.
- It owns its *own* fresh `SharedEmbedding` over the graph vocab (trained from scratch
  alongside the decoder) plus a trainable `nn.Linear` projection from T5's hidden size
  down to `d_model` — only those two pieces train; the pretrained T5 stack itself stays
  frozen (`T5GraphEncoder.freeze`) and its `.train()` is overridden so a stray recursive
  `.train()` call from a parent module can never toggle its dropout back on.
- `context_requires_grad = True` (checked by `common.encode_context`) because gradient
  must still flow to `proj`/`embedding` even though the T5 forward pass itself runs
  under `torch.no_grad()` internally.
- `trainable_state_dict()`/`load_trainable_state_dict()` checkpoint only `proj` and
  `embedding` — the frozen T5 stack is fully reproducible from `--t5_model_name` alone,
  so re-saving tens of millions of frozen params on every checkpoint would be pure waste.

Requires `transformers`/`sentencepiece` (see [`requirements.txt`](requirements.txt)),
otherwise unused.

### 6.6 DLMDecoder: ELF-style rectified-flow decoder

`dlm.DLMDecoder.forward(z_t, t, context, context_mask, mode, self_cond)` predicts a
clean embedding `x_hat` (x-prediction, not noise-prediction) from a noisy input `z_t`,
diffusion time `t`, cross-attention context, a `mode` flag (`DENOISE_MODE=0` /
`DECODE_MODE=1`, added as a learned per-mode embedding so one network serves both
training branches), and an optional self-conditioning input.

`dlm.loss` (one training step):
1. **Context dropout** (`cfg_dropout_prob`, default 0.1): per-example, replace `context`
   with the model's learned `null_context` — enables optional classifier-free guidance
   (CFG) at sampling time.
2. **Branch assignment**: each example is randomly assigned to the **denoise branch**
   (probability `1 - decode_branch_prob`, default 0.8) with `t ~ Uniform(0, 1-eps)`, or
   the **decode branch** (probability `decode_branch_prob`, default 0.2) with
   `t ~ Uniform(0.5, 1.0)` (near-clean).
3. **Forward-process interpolation**: `z_t = t·x + (1-t)·ε`, `x` = clean target
   embeddings, `ε ~ N(0,I)` — this is the *rectified flow* interpolation (linear path
   from noise to data), not a diffusion-SDE forward process.
4. **Self-conditioning** (`selfcond_prob`, default 0.0 = off): with this probability, an
   extra no-grad forward pass with `self_cond=0` produces `x_hat'`, which is fed
   (detached) as the real forward pass's `self_cond` input.
5. **Single shared forward pass**, `mode` varying per example, produces `x_hat` for
   every example regardless of its assigned branch.
6. **Loss routing**: denoise-branch examples get reweighted MSE
   `(1/((1-t)^2+eps))·‖x_hat - x‖^2` (the reweighting emphasizes small `t`, i.e. noisier
   inputs — standard for x-prediction rectified-flow training); decode-branch examples
   get cross-entropy on `unembed(x_hat)` against `target_ids`. **Both branches' losses
   are computed over all `L_TGT` positions, including `<PAD>`** — this is the direct
   answer to "why doesn't the DLM have a per-step token accuracy like the ARLM does": the
   DLM has no autoregressive stopping mechanism, so it must learn to place
   `<EOS>`/`<PAD>` itself, and there's no natural "predict the next token" framing to
   score a training-time accuracy against (see the eval-time `token_accuracy`/
   `token_accuracy_nopad` in `metrics.py` instead — those work identically for both
   models because they score real `.generate()` output, not training-step logits).

`dlm.sample` (inference): `z_0 ~ N(0,I)` over `[B, L_TGT, D]`; Euler-integrates
`dz/dt = (x_hat - z)/(1 - t + eps)` for `num_sample_steps` (default 32) equal steps from
`t=0` to `t≈1`, always running the network in `DENOISE_MODE`; if `guidance_scale != 1.0`
each step also runs an unconditional (`null_context`) pass and linearly combines them
(`guidance_scale=1.0`, the default, skips this entirely — CFG is off by default because
generation here is always meant to be graph-conditioned, unlike ELF's unconditional
text-generation use case, but the mechanism is wired for sweeps). A final `DECODE_MODE`
forward pass + `argmax(unembed(x_hat))` converts the final continuous state to discrete
tokens. `use_self_cond` must match how the checkpoint was *trained* (`selfcond_prob > 0`)
— threaded explicitly from training config in `train_dlm.py`/`eval_only.py`/
`eval_venn.py` rather than left as an independent CLI default, since a model trained
with self-conditioning always zeroed never learned to use a nonzero self-conditioning
input.

### 6.7 GPTDecoder: standard causal decoder

`arlm.GPTDecoder.forward` is an ordinary decoder-only transformer stack (causal
self-attention via `DecoderLayer(causal=True)` + cross-attention to the same frozen
context) producing token logits directly (no diffusion machinery). `arlm.loss` is
standard teacher-forced next-token cross-entropy with `<PAD>` excluded via
`ignore_index=tok.PAD` — unlike the DLM, the ARLM naturally learns to stop via `<EOS>`,
so it never needs to learn to predict trailing `<PAD>`. This is also the source of the
per-training-step `train/token_accuracy` scalar that only `train_arlm.py` logs (computed
as a byproduct of `arlm.loss`'s teacher-forced logits, which `dlm.loss` has no equivalent
of).

`arlm.sample`: greedy (`argmax`) autoregressive decoding starting from `<P>`, one token
at a time, stopping per-example at the first `<EOS>` (tracked via a `finished` mask so
already-finished examples in a batch get forced to emit `<PAD>` rather than continuing to
decode past their own `<EOS>`), or at `max_len` (defaults to `l_tgt`). No KV-cache — the
full forward pass is recomputed every step; explicitly noted as negligible cost at this
scale (2 layers, `d_model=128`, `L_TGT<=16`).

## 7. Evaluation semantics

Source: [`metrics.py`](src/spelf/metrics.py).

`metrics.evaluate_generation(input_ids_row, gen_ids_row, target_ids_row)` computes, per
example:

| Metric | Definition | Notes |
|---|---|---|
| `token_accuracy` | elementwise match, generated vs. target, over the full `L_TGT` canvas | inflated by trivially-correct trailing `<PAD>` positions |
| `token_accuracy_nopad` | elementwise match restricted to positions the **target** doesn't pad | masks by the target's own `<PAD>` span, not the generation's, so a generation that mispredicts *where* `<EOS>`/`<PAD>` starts is still penalized correctly |
| `exact_match` | full-sequence equality with the one canonical target used in that example | a strict diagnostic — the model can find a *different*, equally valid, diametric path and score `optimal=True` while `exact_match=False` |
| `valid` | decodes to a real simple path (no repeated nodes) using real edges of the reconstructed graph | prerequisite for `shortest`/`correct_length`/`optimal` — all three are `False` if this is `False` |
| `shortest` | `valid` and the path is the actual shortest path between its own two endpoints | necessary but not sufficient for being diametric — the endpoints need not be a diametric pair |
| `correct_length` | `valid` and the path's edge-length equals the graph's diameter | also not sufficient alone — a non-shortest walk between two close nodes could coincidentally have `diameter` edges |
| `optimal` | `shortest` **and** `correct_length` | equivalent to "endpoints are a diametric pair and this is a shortest path between them" — exactly the definition of a genuine diametric path. This is the ID early-stopping signal in all three training scripts. |

`shortest` and `correct_length` are both subsets of `valid` by construction (only ever
set `True` when `valid` is `True`), and their intersection is exactly `optimal` — this
is the structure `eval_venn.py` visualizes (§9.6).

`metrics.run_eval(decoder, encoder, loader, device, sample_kwargs, max_batches)` is the
one function both `.generate()`-exposing decoders share: it dispatches purely on
`decoder.generate(context, context_mask, **sample_kwargs)`, so everything past that call
is identical for DLM and ARLM. Called with `max_batches` set for the cheap subsampled
per-`--eval_every` eval during training, and `max_batches=None` for full-dataset eval
(`--full_eval_every`) and both standalone eval scripts.

## 8. Training infrastructure

Source: [`common.py`](src/spelf/common.py).

- **Optimizer** (`build_optimizer`): AdamW with the conventional exclusion of
  embeddings/LayerNorm/bias/1-D params (and `null_context`) from weight decay —
  necessary, not cosmetic, for `SharedEmbedding.freeze_pretrained_rows` to behave as a
  true freeze (§6.1). Accepts a single module or a list of modules
  (`[decoder, t5_encoder]` when the conditioning encoder itself has trainable params);
  `trainable_parameters` dedupes by parameter object identity so a param shared between
  two modules (the embedding table, encoder ↔ decoder) is never double-listed,
  double-clipped, or double-stepped.
- **LR schedule** (`build_lr_schedule`): linear warmup → cosine decay to 0, as a
  `LambdaLR` multiplier.
- **Checkpointing** (`save_checkpoint`/`load_checkpoint`): each checkpoint bundles model
  state, optimizer state, scheduler state, an arbitrary `extra_state` dict (early-stopper
  state, wandb run id, T5-encoder trainable state when applicable), the run's own config
  dict, and full RNG state (python/numpy/torch, plus CUDA/MPS if available) so a resumed
  run's data order and dropout masks continue exactly where they left off, not just its
  weights. Every run directory accumulates `checkpoint_latest.pt`, `checkpoint_best.pt`,
  and (`also_snapshot=True`) a rolling window of `checkpoint_step_N.pt` files pruned to
  the last `keep_last_k` (default 3) by `_prune_old_snapshots`.
- **Resume safety** (`check_checkpoint_config`): asserts specific config keys (currently
  just `encoder_kind`) match between a resumed checkpoint and the current CLI invocation
  *before* attempting to load optimizer state — without this, resuming a `--run_dir` that
  was trained with a different `--encoder_kind` would surface later as a cryptic
  `optimizer.load_state_dict` "parameter group doesn't match" error, since the two
  encoder kinds have disjoint trainable-parameter sets.
- **Early stopping** (`EarlyStopper`): tracks a metric (ID `optimal_rate`, `mode="max"`
  in every training script) across eval rounds; `step()` returns `True` once it has
  failed to improve by more than `--tolerance` for `--patience` consecutive rounds.
  Encoder pretraining does *not* use this — it has no path-label signal to early-stop on,
  so it runs a fixed `--steps` budget instead, tracking held-out MLM loss purely for
  "did I diverge" monitoring and best-checkpoint selection.
- **wandb** (`wandb_init`): logs the full CLI config (plus derived fields like `l_in`/
  `l_tgt`/`vocab_size`/`d_model`) under a shared `--wandb_group` so encoder/DLM/ARLM runs
  are comparable side-by-side in the UI. Reattaches to a previous run via a saved
  `run.id` (in `extra_state["wandb_run_id"]`) on resume, so a locally-resumed run
  continues its existing history instead of forking a fresh, empty-looking run at the
  same step.

## 9. Scripts (command flow and I/O)

| Script | Reads | Writes | Notes |
|---|---|---|---|
| [`generate_data.py`](scripts/generate_data.py) | nothing | `{out_dir}/{train,val_id,val_ood,encoder_pretrain_extra}.pt`, `{out_dir}/meta.json` | Run once per dataset config; everything downstream reads `meta.json` for shapes/vocab. |
| [`pretrain_encoder.py`](scripts/pretrain_encoder.py) | `data/train.pt`, `data/encoder_pretrain_extra.pt`, `data/meta.json` (via `dataset.EncoderPretrainDataset`) | `runs/encoder/checkpoint_{latest,best,step_N}.pt` | Fixed `--steps` budget, no early stop (§8). A 10% (up to `--n_eval_holdout`) slice of the pretraining corpus is held out via `random_split` for eval-loss/accuracy tracking. |
| [`train_dlm.py`](scripts/train_dlm.py) | `data/{train,val_id,val_ood}.pt`, `data/meta.json`, `--encoder_ckpt` (or a fresh T5 encoder if `--encoder_kind t5`) | `runs/dlm/checkpoint_{latest,best,step_N}.pt` | Trains `DLMDecoder`; early-stops on ID `optimal_rate`. |
| [`train_arlm.py`](scripts/train_arlm.py) | same as `train_dlm.py` | `runs/arlm/checkpoint_{latest,best,step_N}.pt` | Same shape as `train_dlm.py`, no diffusion-specific args (§6.7). |
| [`eval_only.py`](scripts/eval_only.py) | a trained checkpoint (`--checkpoint`), the encoder it references (`--encoder_ckpt` or the value baked into the checkpoint's own config), `data/val_{id,ood}.pt` | optional PNGs (`--save_viz_dir`), optional wandb log | Reconstructs the encoder+decoder purely from the checkpoint's saved config — no training-script CLI args need to be re-supplied by hand. |
| [`eval_venn.py`](scripts/eval_venn.py) | same as `eval_only.py` | one PNG (`--out`, defaults to `venn_{model_kind}_{checkpoint_stem}.png`) | Same checkpoint-driven reconstruction pattern as `eval_only.py` (`load_encoder_from_checkpoint`/`build_decoder` duplicate that logic locally rather than importing it, since `eval_only.py` doesn't expose it as a reusable function). See §9.6 for the diagram's design. |
| [`dataset_stats.py`](scripts/dataset_stats.py) | `data/{train,val_id,val_ood,encoder_pretrain_extra}.pt`, `data/meta.json` | one text report (`--out`, defaults to `{data_dir}/dataset_stats.txt`) | No model/checkpoint involved at all — pure dataset introspection. See §9.7. |

### 9.1–9.5 Training-script control flow (shared shape across `pretrain_encoder.py`/`train_dlm.py`/`train_arlm.py`)

All three follow the same explicit (not abstracted into a shared `Trainer` class) loop
shape, by design, so each script reads top-to-bottom on its own:

1. Parse args → `common.set_seed` → `common.get_device`.
2. Load `meta.json`, build datasets/loaders.
3. Build model(s), optimizer, LR schedule, (for DLM/ARLM) `EarlyStopper`.
4. If `--resume` resolves to an existing checkpoint: validate config compatibility
   (`check_checkpoint_config`), load all state, recover `start_step`.
5. `wandb_init`, reattaching to a previous run id if resumed.
6. Infinite-loader training loop (`infinite_loader` just re-iterates the `DataLoader`
   forever — no notion of "epoch" is tracked or logged anywhere in this codebase, only
   `step`): forward → loss → backward → grad-clip → optimizer/scheduler step → periodic
   console + wandb logging (`--log_every`) → periodic cheap eval (`--eval_every`) →
   periodic full eval (`--full_eval_every`) → checkpoint save (best + latest, every cheap
   eval round) → early-stop check (DLM/ARLM only).
7. Final full eval + wandb image log, `run.finish()`.

### 9.6 `eval_venn.py`'s Venn-diagram design

Given the four-region structure from §7 (`shortest`/`correct_length` both subsets of
`valid`, intersection = `optimal`), the diagram draws `shortest` and `correct_length` as
two **fixed-size** overlapping circles (`_draw_venn`, radius `r=1.4`, center distance
`d=1.4` — constants, not derived from the counts) enclosed in a larger dashed `valid`
boundary, with the actual count and percentage of the split's total written as text in
each of the four regions (`shortest`-only, `correct_length`-only, `optimal`, `valid`-only)
plus `invalid` reported separately outside the boundary. This was a deliberate
simplification over an earlier area-proportional version (circle sizes/overlap solved to
match true counts via a closed-form circle-intersection-area formula): the proportional
version produced illegible, near-zero-radius, overlapping labels whenever a checkpoint's
counts were degenerate (e.g. an undertrained smoke-test model scoring near-zero on every
metric), and exact areas add little when the same information is already printed as
text. The uniform layout is schematic only — never read circle/overlap *size* as
meaningful, only the text.

### 9.7 `dataset_stats.py`'s per-example vs. per-unique-graph split

Token-count statistics (`input_tokens`, `target_tokens`,
`implied_diameter_from_target`) are computed **per example** — that's what actually
varies example-to-example (padding, which diametric path was chosen) and what a
model/dataloader actually sees. Graph-connectivity statistics (`n_nodes`, `n_edges`,
`avg_degree`, `density`, `diameter`, `avg_shortest_path_length`) are computed **once per
unique graph** — deduplicated by `graph_id` for the three labeled splits
(`_unique_labeled_graphs`) or by `data_cache.graph_key(n, edges)` for the unlabeled
`encoder_pretrain_extra` split, which has no `graph_id` field
(`_unique_unlabeled_graphs`). Without this dedup, connectivity stats would be skewed
toward whichever graphs happen to contribute more examples (more diametric paths, or
more edge-shuffled copies) — see §5.5's `examples_per_graph`-vs-diameter correlation for
why that skew is systematic, not noise. `implied_diameter_from_target` is a decode-free
cross-check computed directly from `target_ids != PAD` counts (`target_len - 3 ==
diameter`, since every labeled target is `<P>` + path_nodes + `<EOS>` and
`len(path_nodes) - 1 == diameter` by construction) — it's example-weighted (like the
token-count stats), so it's *expected* to differ slightly from the graph-weighted
`diameter` connectivity stat, confirmed in `data/dataset_stats.txt` (train: 2.799 vs.
3.074) and verified as a real weighting effect, not a bug, via a one-off script during
development (0 mismatches between per-example implied diameter and the example's own
graph's true `nx.diameter` across 2000 sampled examples).

## 10. Tunable variables (CLI arguments)

Every `argparse` flag in the codebase, grouped by concern. Script column shows which
script(s) expose the flag; where a flag exists in more than one script with the same
name, its default may differ between them (noted inline).

### Data generation (`generate_data.py`)

| Flag | Default | Effect |
|---|---|---|
| `--seed` | 42 | Top-level seed; fans out into 4 independent per-split RNG streams (§5.7). |
| `--out_dir` | `data` | Output directory for `.pt` files + `meta.json`. |
| `--n_train_graphs` | 50000 | Number of distinct graphs sampled for `train.pt`. |
| `--max_paths_per_graph` | 5 | Cap on diametric-pair examples per graph (labeled splits); exact edge-shuffle repeat count for the unlabeled corpus (§5.5). |
| `--n_val_id_graphs` | 500 | Graphs in `val_id.pt`. |
| `--n_val_ood_graphs` | 300 | Graphs in `val_ood.pt`. |
| `--n_encoder_pretrain_ood_graphs` | 10000 | Graphs in `encoder_pretrain_extra.pt` (§5.6). |
| `--avg_degree` | 3.0 | Target average node degree, drives the Erdős–Rényi edge probability (`graphgen._sample_graph_contiguous`). |

### Architecture (shared shape across `pretrain_encoder.py`/`train_dlm.py`/`train_arlm.py`; T5 mode overrides `d_model`)

| Flag | Default | Effect |
|---|---|---|
| `--d_model` | 128 | Hidden size. In `train_dlm.py`/`train_arlm.py`, only used when `--encoder_kind t5` — custom mode infers `d_model` from the loaded `--encoder_ckpt` instead. |
| `--n_layers` | 2 | Transformer layers (encoder or decoder, per script). |
| `--n_heads` | 8 | Attention heads. |
| `--d_mlp` | 512 | MLP hidden size. |
| `--dropout` | 0.1 | Post-attention/MLP dropout (attention's own internal dropout is hardcoded to 0, §6.3). |

### Conditioning encoder selection (`train_dlm.py`, `train_arlm.py`)

| Flag | Default | Effect |
|---|---|---|
| `--encoder_kind` | `custom` | `custom` = this project's `GraphEncoder` (`--encoder_ckpt` required); `t5` = frozen pretrained `T5GraphEncoder` (§6.5). |
| `--encoder_ckpt` | `None` | Path to a `pretrain_encoder.py` checkpoint; required iff `--encoder_kind custom`. |
| `--t5_model_name` | `t5-small` | HuggingFace T5 checkpoint name; used iff `--encoder_kind t5`. |

### Optimization (all four training-capable scripts: `pretrain_encoder.py`, `train_dlm.py`, `train_arlm.py`; `eval_*.py` don't train)

| Flag | Default (`pretrain_encoder.py` / `train_dlm.py` / `train_arlm.py`) | Effect |
|---|---|---|
| `--seed` | 0 / 0 / 0 | Seeds `common.set_seed` for this run (independent of the data-generation seed). |
| `--lr` | 3e-4 / 2e-4 / 3e-4 | AdamW learning rate. DLM's default is lower — noted in-code as wanting more training stability for the diffusion objective. |
| `--warmup_steps` | 1000 / 2000 / 1000 | Linear-warmup length before cosine decay begins. |
| `--weight_decay` | 0.01 / 0.01 / 0.01 | AdamW weight decay (excluded for embeddings/norms/bias, §8). |
| `--grad_clip` | 1.0 / 1.0 / 1.0 | Global-norm gradient clipping threshold. |
| `--batch_size` | 256 / 128 / 128 | Training batch size. |
| `--steps` (encoder) / `--max_steps` (DLM/ARLM) | 30000 / 150000 / 100000 | Total step budget (encoder: hard budget; DLM/ARLM: budget or early stop, whichever first). |

### Early stopping (`train_dlm.py`, `train_arlm.py` only — encoder pretraining has no label signal to stop on, §8)

| Flag | Default | Effect |
|---|---|---|
| `--patience` | 5 | Consecutive non-improving eval rounds tolerated before stopping. |
| `--tolerance` | 0.005 | Minimum ID `optimal_rate` improvement to reset the patience counter. |

### DLM-specific (`train_dlm.py`, plus a read-only echo in `eval_only.py`/`eval_venn.py` via the checkpoint's saved config)

| Flag | Default | Effect |
|---|---|---|
| `--cfg_dropout` | 0.1 | Per-example probability of replacing context with `null_context` during training (§6.6 step 1). |
| `--decode_branch_prob` | 0.2 | Fraction of each batch routed to the near-clean cross-entropy branch instead of the noisy-MSE branch. |
| `--selfcond_prob` | 0.0 | Probability of using a real (detached) self-conditioning input instead of zeros; 0.0 = vanilla (no self-conditioning at train *or* sample time, since `eval_only.py`/`eval_venn.py`/`train_dlm.py`'s own `sample_kwargs` all derive `use_self_cond` from this). |
| `--lambda_ce` | 1.0 | Weight on the decode-branch cross-entropy term relative to the denoise-branch MSE term in the total loss. |
| `--num_sample_steps` | 32 | Euler-integration steps at sampling/inference time (also exposed identically in `eval_only.py`/`eval_venn.py`). |
| `--guidance_scale` | 1.0 | Classifier-free-guidance strength at sampling time; 1.0 disables CFG entirely (also in `eval_only.py`/`eval_venn.py`). |

### Encoder pretraining objective (`pretrain_encoder.py`)

| Flag | Default | Effect |
|---|---|---|
| `--mlm_prob` | 0.15 | Fraction of eligible (node-id) token positions masked per example. |
| `--n_eval_holdout` | 2000 | Size of the held-out slice of the pretraining corpus used for eval loss/accuracy (capped at 10% of the corpus). |

### Eval cadence & subsampling (`train_dlm.py`, `train_arlm.py`)

| Flag | Default | Effect |
|---|---|---|
| `--log_every` | 100 | Console + wandb scalar logging interval (steps). |
| `--eval_every` | 1000 | Cheap subsampled ID+OOD eval interval; drives early-stop checks and `checkpoint_latest`/`checkpoint_best` saves. |
| `--full_eval_every` | 5000 | Full (unsubsampled) ID+OOD eval interval. |
| `--n_id_subsample` | 500 | Approx. example count used for the cheap eval's ID subset (converted to a batch count). |
| `--n_ood_subsample` | 300 | Same, OOD. |
| `--n_viz_examples` | 4 (training scripts) / 8 (`eval_only.py`) | Number of example plots logged to wandb (and/or saved, `eval_only.py`) per eval round. |

### Run management (every script that trains or evaluates)

| Flag | Default | Effect |
|---|---|---|
| `--data_dir` | `data` | Where to read cached `.pt`/`meta.json` from. |
| `--run_dir` | script-specific (`runs/encoder`, `runs/dlm`, `runs/arlm`) | Checkpoint directory. |
| `--resume` | `latest` | `latest` \| `best` \| an explicit path \| `none` (fresh start); resolved by `common.resolve_checkpoint_path`. |
| `--num_workers` | 0 | `DataLoader` worker count. |
| `--wandb_project` | `shortest-path-elf` | wandb project name. |
| `--wandb_run_name` | script-specific | wandb run display name. |
| `--wandb_group` | `graph-shortest-path` | Groups encoder/DLM/ARLM runs together in the wandb UI. |
| `--wandb_mode` | `online` (training scripts) / `disabled` (`eval_only.py`) | `online` \| `offline` \| `disabled`. |

### Standalone eval (`eval_only.py`, `eval_venn.py`)

| Flag | Default | Effect |
|---|---|---|
| `--checkpoint` | *(required)* | Path to a trained DLM/ARLM checkpoint. |
| `--model_kind` | *(required)* | `dlm` \| `arlm` — must match the checkpoint. |
| `--encoder_ckpt` | `None` | Overrides the encoder path baked into the checkpoint's own config, if set. |
| `--split` | `both` | (`eval_only.py` only) `id` \| `ood` \| `both`. `eval_venn.py` always does both, side by side. |
| `--batch_size` | 128 | Eval batch size. |
| `--save_viz_dir` | `None` | (`eval_only.py` only) also save example plots as local PNGs. |
| `--out` | `venn_{model_kind}_{checkpoint_stem}.png` | (`eval_venn.py` only) output image path. |

### Dataset statistics (`dataset_stats.py`)

| Flag | Default | Effect |
|---|---|---|
| `--data_dir` | `data` | Dataset directory to analyze. |
| `--out` | `{data_dir}/dataset_stats.txt` | Report output path. |

## 11. Non-tunable variables (hardcoded constants)

These are not exposed as CLI flags; changing them means editing source, and several have
correctness implications elsewhere in the codebase if changed carelessly.

| Constant | Value | File | Why fixed |
|---|---|---|---|
| `MAX_NODES` | 14 | `tokenizer.py` | Largest OOD graph size; fixes the shared ID/OOD vocabulary (§4). Changing it requires regenerating all data and retraining, since `VOCAB_SIZE`/`TARGET_LENGTH` derive from it. |
| `PAD,G,E,N,P,EOS,MASK` | `range(7)` | `tokenizer.py` | Fixed special-token id assignment; baked into every cached tensor on disk. |
| `NODE_OFFSET` | 7 | `tokenizer.py` | First node-id token id; node token `k` is `NODE_OFFSET + k`. |
| `VOCAB_SIZE` | `NODE_OFFSET + MAX_NODES` = 21 | `tokenizer.py` | Derived, not independently settable. |
| `TARGET_LENGTH` | `MAX_NODES + 2` = 16 | `tokenizer.py` | Global fixed diffusion-canvas length (§4) — must be one constant across every graph size for the DLM's fixed-shape tensors. |
| `ID_NODE_RANGE` | `(6, 10)` | `data_cache.py` | In-distribution graph size range; not exposed via CLI (unlike `--avg_degree` etc.) since the whole ID/OOD split design assumes exactly this boundary. |
| `OOD_NODE_RANGE` | `(11, 14)` | `data_cache.py` | Out-of-distribution graph size range; upper bound must equal `MAX_NODES`. |
| `SEED_OFFSETS` | `{train:0, val_id:1e6, val_ood:2e6, encoder_pretrain_extra:3e6}` | `data_cache.py` | Per-split RNG-stream independence under one top-level `--seed` (§5.7). |
| graph_id offset scheme | train starts at 0; val_id at `10**9`; val_ood at `2*10**9` | `data_cache.generate_all` | Keeps `graph_id` numerically disjoint across splits; not otherwise meaningful. |
| `DENOISE_MODE` / `DECODE_MODE` | 0 / 1 | `dlm.py` | Learned mode-embedding indices distinguishing the DLM's two training branches (§6.6). |
| Attention dropout | `0.0` (always, regardless of `--dropout`) | `modules.py` (`EncoderLayer`, `DecoderLayer`) | MPS backend does not support `dropout_p>0` in `nn.MultiheadAttention`'s fast path during training (§6.3) — a platform limitation, not a tuning choice. |
| Embedding init std | 0.02 | `modules.SharedEmbedding.__init__` | Conventional small-std init to keep unembedding logits well-scaled from the start (§6.1). |
| Frozen-row exceptions | `(tok.P, tok.EOS)` | `modules.SharedEmbedding.freeze_pretrained_rows` default arg | The only two vocab tokens absent from the encoder's own input space (§6.1); every call site in this codebase uses the default. |
| MLM-maskable tokens | node-id tokens only (`MASKABLE_TOKENS_ONLY_NODES = True`, documentation flag) | `encoder.py` | Structural markers (`<G>/<E>/<N>`) are never masked (§6.4). |
| `eval_venn.py` circle geometry | `r=1.4`, `d=1.4` | `scripts/eval_venn.py` | Purely schematic layout constants, deliberately not derived from data (§9.6). |
| `viz.py` node sizes/colors | `node_size=300/400`; blue=path, green=start, orange=end, red=invalid | `viz.py` | Cosmetic, not configurable via CLI. |
| `keep_last_k` (checkpoint snapshot pruning) | 3 | `common.save_checkpoint` default arg | Not exposed via any script's CLI; every call site uses the default. |
| optimizer no-decay rule | param `ndim<=1` or name contains `"embedding"`/`"norm"`/`"null_context"` | `common.build_optimizer` | Fixed heuristic, not parameterized. |

## 12. Checkpoint schema

Every checkpoint written by `common.save_checkpoint` (encoder, DLM, or ARLM alike) is a
single `torch.save`d dict:

```python
{
  "step": int,
  "model_state": <model.state_dict()>,
  "optimizer_state": <optimizer.state_dict() or None>,
  "scheduler_state": <scheduler.state_dict() or None>,
  "extra_state": {
      # encoder: {"best_eval_loss": float | None, "wandb_run_id": str}
      # DLM/ARLM: {"early_stopper": EarlyStopper.state_dict(), "wandb_run_id": str,
      #            "encoder_trainable": {...} }  # only present if --encoder_kind t5
  },
  "config": <vars(args) | derived fields (l_in, l_tgt, vocab_size, d_model, ...)>,
  "rng_state": {"python": ..., "numpy": ..., "torch": ..., "torch_cuda"?: ..., "torch_mps"?: ...},
}
```

`config` is what makes `eval_only.py`/`eval_venn.py` able to reconstruct a matching
encoder+decoder from nothing but `--checkpoint` — every architecture/`encoder_kind`
field they need is read back from here rather than re-supplied on the eval CLI.
`common.check_checkpoint_config` guards `--resume` against attaching to a checkpoint
whose `config` disagrees on `encoder_kind` with the current invocation (§8).

## 13. Testing

[`tests/`](tests) covers the deterministic, non-model-training parts of the pipeline
(nothing here trains a model — that's checked by hand via wandb curves, per the
project's original plan):

- `test_tokenizer_roundtrip.py` — `encode_input`/`decode_input`,
  `encode_target`/`decode_target` round-trip identity (including the exact example from
  `data_sample.txt`), padding behavior, and that malformed sequences decode to `None`
  rather than raising.
- `test_bfs_tiebreak.py` — `shortest_path_lexsmallest` against hand-worked small graphs,
  including cases specifically constructed to distinguish "any valid shortest path" from
  "the lexicographically smallest one" (§5.2).
- `test_graphgen_connected.py` — `sample_graph` always returns a connected graph across
  both the ID and OOD size ranges, average degree lands near the target, `diametric_pairs`
  correctly enumerates ties, `sample_diametric_paths` respects its cap, and the
  spanning-tree fallback (`_random_spanning_tree_fallback`) is itself always connected.
- `test_label_pool.py` — the wide node-label pool (§5.3): labels are drawn from the full
  pool (not just `0..n-1`), `label_pool_size < n` is rejected, default (no pool) behavior
  is unchanged, and ID-sized training graphs really do exercise high node-id tokens
  (directly exercises `assert_full_node_id_coverage`).

Run with `pytest tests/ -q` from the repo root (after `pip install -r requirements.txt`).
