# Diametric-Path DLM vs ARLM

Compares an **ELF-style continuous-embedding diffusion language model (DLM)** against a
**GPT-style autoregressive language model (ARLM)** on a synthetic search/reasoning task:
given a small random graph, find *a* path whose length equals the graph's **diameter**
(the longest shortest-path distance between any two nodes) — a "diametric path". Unlike
a shortest-path task, there is no query naming which two nodes to connect: the model
sees only the graph and must find some pair (and a path between them) that achieves the
diameter, of which there may be several. Both decoders condition on the same frozen,
pretrained bidirectional graph encoder, so the only variable under study is the decoder
architecture (diffusion vs. autoregressive).

"ELF" = [*Embedded Language Flows*](https://arxiv.org/html/2605.10938) — a diffusion LM
that operates via **rectified-flow matching in continuous embedding space** (x-prediction,
not masked/discrete token diffusion), with a two-branch denoise/decode training scheme,
self-conditioning, and classifier-free-guidance dropout. This codebase adapts that recipe
to condition on a from-scratch graph encoder instead of a frozen T5 encoder, and to
generate a fixed-length path non-autoregressively instead of free-form text.

Models are intentionally small (2 transformer layers, hidden size 128, MLP size 512
throughout) so the whole pipeline runs on a Mac (MPS/CPU) in plain PyTorch — no GPU or
HuggingFace dependency required.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the systematic reference: full data flow,
every design decision with its rationale, and a complete catalog of tunable (CLI) vs.
non-tunable (hardcoded) variables.

## Repo layout

```
shortest-path/
  data_sample.txt              example of the raw input/target text format
  requirements.txt
  src/spelf/
    tokenizer.py                vocab, encode/decode, fixed-length formulas
    graphgen.py                 random connected graphs, deterministic shortest path
    data_cache.py                orchestrates generation -> tokenization -> cached tensors
    dataset.py                  thin PyTorch Dataset wrappers around the cached tensors
    modules.py                  shared embedding table + transformer building blocks
    encoder.py                  GraphEncoder + its MLM pretraining objective
    dlm.py                      ELF-style diffusion decoder (train loss + sampler)
    arlm.py                     GPT-style causal decoder (train loss + sampler)
    muon.py                     Muon optimizer (canonical ELF's DLM optimizer)
    t5_encoder.py                T5GraphEncoder (conditioning-only, ARLM) +
                                 T5DiffusionEncoder (diffuses in T5's own embedding
                                 space, DLM) -- see "T5 as an alternative encoder" above
    metrics.py                  decode generations -> exact-match/valid/optimal rates
    viz.py                      ground-truth-vs-generated graph plots for wandb
    common.py                   seeding, optimizer/LR schedule, checkpoint save/resume,
                                 early stopping, wandb init
  scripts/
    generate_data.py            build and cache all data splits
    pretrain_encoder.py         self-supervised graph encoder pretraining
    train_dlm.py                train the DLM decoder
    train_arlm.py                train the ARLM decoder
    eval_only.py                evaluate a saved checkpoint without training
    eval_venn.py                evaluate a checkpoint and render a valid/shortest/
                                 correct-length/optimal Venn diagram, ID + OOD
    dataset_stats.py            report per-split token-count and graph-connectivity
                                 statistics for a generated dataset
    inspect_t5_tokenization.py  show (and verify) how T5's real tokenizer segments a
                                 batch of real graph/path examples, token boundaries
                                 marked with '|'
  tests/                       unit tests for tokenizer + graph generator
  data/                        generated datasets (gitignored)
  runs/                        checkpoints + configs per run (gitignored)
```

## Data format

The raw input describes only a graph — there is no query. The model must output *some*
path whose length (in edges) equals the graph's diameter.

```
<G> u1 v1 <E> u2 v2 <E> ... <N> n0 n1 ... n(k-1)      (input, "prompt")
<P> p0 p1 ... pk <EOS> <PAD> <PAD> ...                 (target, "path")
```

- `<G>` starts the edge list, each edge written as `u v <E>` in shuffled order (so the
  model must actually parse the list, not exploit position).
- `<N>` starts the sorted node-id list; the input ends there.
- The target starts with `<P>`, lists the path's node ids in order, ends with `<EOS>`,
  and is padded with `<PAD>` out to a **fixed global length** `L_TGT = MAX_NODES + 2 = 16`
  (`MAX_NODES=14`, the largest graph size used anywhere) — one constant shape across
  every graph size, which the DLM needs for its fixed-size diffusion canvas.
- Vocab is 21 tokens total: 7 special tokens (`<PAD> <G> <E> <N> <P> <EOS> <MASK>`)
  plus node-id tokens `0..13`. See `tokenizer.py` for the exact encode/decode logic —
  `decode_input`/`decode_target` are the single source of truth used everywhere
  (training-data prep, eval, and interpreting raw model output alike), so a malformed
  model generation decodes to `None`/partial results rather than crashing anything.

A graph's diameter can be achieved by several distinct pairs of nodes (e.g. every
antipodal pair on an even cycle) — `graphgen.diametric_pairs` enumerates all of them,
and for each one, a canonical path is computed by BFS with a **deterministic
tie-break**: BFS distances are computed *from the larger-id endpoint*, then the path is
reconstructed from the smaller-id endpoint by always stepping to the smallest-id
neighbor whose distance-to-target is one less than the current node's
(`graphgen.shortest_path_lexsmallest`). This gives the unique lexicographically-smallest
shortest path between that pair, so training targets are reproducible given a seed.

## Data generation

Graphs are Erdős–Rényi (`p` tuned so avg degree ≈ 3), reject-sampled until connected
(with a random-spanning-tree fallback so generation can never hang). Splits:

| split | graphs | max paths/graph | graph size (# nodes) | purpose |
|---|---|---|---|---|
| train | 50,000 | 5 | 6–10 | training (~150K+ examples, varies with diametric-pair multiplicity) |
| val_id | 500 | 5 | 6–10 | in-distribution validation (held out from train) |
| val_ood | 300 | 5 | 11–14 | out-of-distribution validation |
| encoder_pretrain_extra | 10,000 | 5 | 11–14 | **unlabeled** (no paths), encoder-only |

Each graph contributes one training example per distinct **diametric pair** (a pair of
nodes whose shortest-path distance equals the graph's diameter), up to
`--max_paths_per_graph` (default 5) — so a graph with several equally-long "longest
shortest paths" trains the model on all of them (subsampled down to the cap if there are
more), while a graph with a unique diametric pair contributes just one example
(`graphgen.sample_diametric_paths`). This is how "use multiple paths per graph, if they
exist" is implemented. `meta.json` records the realized `avg_train_paths_per_graph`.
For the unlabeled encoder-pretraining corpus (no notion of a "path" at all), the same
knob instead controls how many independently edge-shuffled copies of each graph are
generated.

`val_id`/`val_ood` graphs are always generated **before** train and reserved, so train is
reject-resampled around them — the in-distribution evaluation split's graphs are
guaranteed disjoint from every training graph (`data_cache.assert_disjoint_graphs`,
checked automatically after every `generate_data.py` run).

Node *labels*, not just graph size, matter here: `graphgen.sample_graph(..., label_pool_size=tok.MAX_NODES)`
labels each graph's `n` nodes with a random `n`-subset of `{0, ..., 13}` rather than
contiguous `0..n-1`. This means every node-id token the model can be asked to emit is
exercised by training examples at every graph size — an ID-sized (6–10 node) training
graph can and does use node-id tokens up to 13. **Graph size alone is the ID/OOD axis**;
node-id token *values* are shared between splits, so OOD evaluation tests generalization
to larger search spaces, not to literally unseen output tokens. `data_cache.py` asserts
this coverage holds (`assert_full_node_id_coverage`) after generating `train.pt`.

`encoder_pretrain_extra` exists to close a specific gap: the encoder uses **learned**
positional encodings, and if it only ever saw 6–10 node inputs during pretraining, its
higher input positions would be untrained garbage when asked to encode 11–14 node OOD
graphs at eval time — an artifact that would contaminate exactly the OOD comparison this
project is trying to measure. So the encoder's self-supervised (no path labels, ever)
pretraining corpus is `train.pt`'s inputs concatenated with this extra unlabeled OOD-sized
set (`dataset.EncoderPretrainDataset`), while the DLM/ARLM decoders themselves only ever
train on the labeled `train.pt`.

## Models

**GraphEncoder** (`encoder.py`) — 2-layer bidirectional transformer, learned positional
encoding, pretrained standalone via masked node-id-token prediction (structural markers
like `<G>/<E>/<N>` are never masked — only node-id tokens are, since that's the
relational signal worth learning). After pretraining it is **frozen** and reused
identically as the conditioning encoder for both decoders, via cross-attention, so both
see exactly the same representation of the graph.

**Shared embedding table** (`modules.SharedEmbedding`) — there is only **one** token
embedding table in the whole system, owned by the encoder and pretrained with it, then
shared **by object reference** into whichever decoder is being trained (not copied). This
mirrors how ELF diffuses directly in a frozen pretrained embedding space. The wrinkle:
`<P>` and `<EOS>` never appear in the encoder's own input (only in path targets), so
those two embedding rows would never get trained by encoder pretraining. The fix is a
gradient hook (`SharedEmbedding.freeze_pretrained_rows`) that zeros gradients for every
row *except* `<P>`/`<EOS>` — so the rest of the table is a true frozen copy of what the
encoder learned, while those two rows keep training during DLM/ARLM training. (Paired
with zero weight-decay on embeddings in `common.build_optimizer`, since AdamW's weight
decay would otherwise shrink "frozen" rows regardless of their zeroed gradient.)

**DLMDecoder** (`dlm.py`) — the ELF-style diffusion decoder, 2 layers. Generates all
`L_TGT` path positions **jointly** (non-autoregressive):
- Forward process is a rectified-flow interpolation `z_t = t·x + (1-t)·ε` between clean
  token embeddings `x` and Gaussian noise `ε`.
- The network does **x-prediction**: given noisy `z_t` and time `t`, predict the clean
  embedding `x̂`, trained with a reweighted MSE `(1/(1-t)²)·‖x̂-x‖²` (denoise branch, ~80%
  of each batch) or, on the other ~20% ("decode branch", near-clean `t∈[0.5,1]`),
  cross-entropy after unembedding `x̂` back to token logits — so the same network serves
  as both the flow-matching denoiser and the final embedding→token decoder.
- **Self-conditioning**: half the time, an extra no-grad forward pass' prediction is fed
  back in as auxiliary input to the real forward pass (standard diffusion trick).
- **CFG dropout**: 10% of examples have their conditioning context replaced with a
  learned null vector during training, enabling optional classifier-free guidance at
  sampling time (default guidance scale 1.0 = off, since generation here is always meant
  to be graph-conditioned).
- **Sampling**: start from pure noise, Euler-integrate the ODE `dz/dt = (x̂-z)/(1-t)` for
  `num_sample_steps` (default 32), then one final decode-mode forward + argmax.

**GPTDecoder** (`arlm.py`) — standard causal-attention decoder, 2 layers, cross-attending
to the same frozen encoder context via the identical `DecoderLayer` class the DLM uses
(only `causal=True` differs), so the conditioning mechanism is apples-to-apples. Trained
with ordinary teacher-forced next-token cross-entropy (`<PAD>` excluded from the loss,
since the ARLM naturally stops at `<EOS>`); sampled greedily, autoregressively, no
KV-cache (negligible cost at this scale).

## T5 as an alternative encoder (`--encoder_kind t5`)

Both `train_dlm.py`/`train_arlm.py` accept `--encoder_kind t5` in place of the default
`custom` (this project's own from-scratch `GraphEncoder`). Both serialize the graph to
English text via `t5_encoder.graph_to_text` and encode it with a frozen pretrained
HuggingFace T5 (`--t5_model_name`, default `t5-small`) — but the *DLM* and the *ARLM* use
T5 in two structurally different ways (`t5_encoder.py`):

- **`GPTDecoder` + T5** (`T5GraphEncoder`): T5 is purely a *conditioning source*. A
  trainable `nn.Linear` projects T5's frozen hidden states down to a freely-chosen
  `d_model` (128 by default); the decoder still generates in this project's own small
  21-token vocab, embedded by a separate, freshly-trained `SharedEmbedding`. An entirely
  ordinary design for an autoregressive model.
- **`DLMDecoder` + T5** (`T5DiffusionEncoder`): matches what the canonical ELF
  implementation (arXiv:2605.10938) actually does with T5 — the DLM diffuses *directly in
  T5's own frozen token embedding space*, not a separately-trained one. Concretely: the
  target for training is T5's own tokenization of `t5_encoder.path_to_text(path)` (e.g.
  `"Path : 3 - 7 - 4 - 5 ."`), padded to a fixed `l_tgt_t5` computed once from the
  worst-case 14-node path; the diffusion embedding/unembedding is tied to T5's own
  (frozen) ~32k-token embedding table (`T5TiedEmbedding`); and there is no projection
  layer at all — T5's contextualized hidden states and its embedding table share the same
  dimension throughout its stack, so once the decoder's own `d_model` is set to T5's
  (512 for t5-small), T5's `last_hidden_state` is used directly as cross-attention
  context. This also means `T5DiffusionEncoder` has *zero* trainable parameters of its
  own — every trainable weight lives in `DLMDecoder`.

  `metrics.py`/`viz.py` stay completely unaware any of this happened: `eval_only.py` /
  `eval_venn.py` / `train_dlm.py`'s own eval calls wrap the raw model in
  `t5_encoder.T5SpaceDLMAdapter`, which detokenizes a T5-space generation back to a path
  and re-encodes it in the project's own vocab (`tokenizer.encode_target`) before handing
  it to `metrics.run_eval` — an unparseable generation just re-encodes to an all-`<PAD>`
  row, which `tokenizer.decode_target` already treats as invalid.

**Node-id token-boundary safety**: every number in `graph_to_text`/`path_to_text` is
bounded by a literal space on both sides (including around `-`, `,`, and before `.`) —
not written bare (`"3-7"`) — because a space is a hard token boundary for
SentencePiece/BPE tokenizers, so no node id can ever end up sharing a single token with a
different node id. This was verified directly against T5's real tokenizer, not just
assumed: the old unspaced format really does fuse some adjacent node ids into one token
(e.g. `"1-4"` → a single token); the spaced format never does, across every case checked.
Run `scripts/inspect_t5_tokenization.py` to see this for real, per-token, against a batch
of actual dataset examples (see below).

## Evaluation

`metrics.run_eval` decodes every generated sequence via `tokenizer.decode_target` and
scores, separately for ID and OOD:
- **token accuracy** — elementwise match against the ground-truth sequence
- **exact-match rate** — full-sequence match against the one canonical target path used
  in that example (a diagnostic; the model may find a *different*, equally valid,
  diametric path and be correctly scored as `optimal` while missing `exact_match`)
- **valid rate** — decodes to a real simple path using real edges of the graph (no
  repeated nodes) — the path's own two endpoints, whatever they are
- **shortest rate** — valid *and* the path is actually the shortest path between its own
  two endpoints (no shortcut exists between them) — necessary but not sufficient for
  being diametric, since those endpoints need not be a diametric pair
- **correct-length rate** — valid *and* the path's length (in edges) equals the graph's
  diameter (recomputed via `networkx.diameter` on the decoded graph) — also not
  sufficient alone, since a non-shortest walk between two close nodes could
  coincidentally have `diameter` edges
- **optimal rate** — shortest *and* correct-length at once, i.e. a genuine diametric
  path: the endpoints are a diametric pair and this is a shortest path between them.
  This is the **"id_optimal rate"** when computed on the ID split: the percentage of
  graphs for which the model found *any* diametric path — the primary early-stopping
  signal

`viz.plot_example` renders a two-panel graph plot (ground truth | generated) via
`networkx.spring_layout`, seeded per-example so the same example's layout stays visually
stable across training. Invalid generations are shown in red, with the raw generated
token sequence printed underneath if it wasn't even parseable as a path.

## Training workflow

1. **Generate data** once (`generate_data.py`) — cheap, a few seconds.
2. **Pretrain the encoder** (`pretrain_encoder.py`) — fixed step budget (no path labels
   to early-stop on), tracks loss/accuracy on a held-out slice of the pretraining corpus.
3. **Train the DLM and/or ARLM** (`train_dlm.py` / `train_arlm.py`), pointing
   `--encoder_ckpt` at the frozen encoder checkpoint from step 2. Each does a cheap
   subsampled eval (default 500 ID / 300 OOD examples) every `--eval_every` steps and a
   full-dataset eval every `--full_eval_every` steps, logging scalars + a handful of
   graph-visualization images to wandb each cheap-eval round. Training stops when ID
   optimal-rate fails to improve by more than `--tolerance` for `--patience` consecutive
   eval rounds (or at `--max_steps`, whichever comes first).
4. **Compare** the two decoders' final ID/OOD exact-match, valid-path, and optimal-path
   rates — that comparison is the actual research result.

All three training scripts share the same checkpoint/resume/early-stop machinery
(`common.py`): each run directory keeps `checkpoint_latest.pt`, `checkpoint_best.pt`, and
a few rolling numbered snapshots, and `--resume latest|best|<path>` picks up exactly
where a run left off (model, optimizer, LR schedule, early-stopper state, and full
RNG state — python/numpy/torch/mps — are all restored).

## CLI reference

### `scripts/generate_data.py`
```
--seed 42                        top-level seed (per-split streams are derived from it)
--out_dir data
--n_train_graphs 50000
--max_paths_per_graph 5           # cap on diametric-pair paths per graph (labeled splits);
                                    # exact edge-shuffle repeat count for the unlabeled corpus
--n_val_id_graphs 500
--n_val_ood_graphs 300
--n_encoder_pretrain_ood_graphs 10000
--avg_degree 3.0
```

### `scripts/pretrain_encoder.py`
```
--data_dir data  --run_dir runs/encoder  --resume latest   # 'latest' | 'best' | <path> | 'none'
--seed 0  --steps 30000  --batch_size 256  --lr 3e-4  --warmup_steps 1000
--weight_decay 0.01  --grad_clip 1.0  --mlm_prob 0.15  --n_eval_holdout 2000
--d_model 128  --n_layers 2  --n_heads 8  --d_mlp 512  --dropout 0.1
--log_every 100  --eval_every 500  --num_workers 0
--wandb_project shortest-path-elf  --wandb_run_name encoder-pretrain
--wandb_group graph-shortest-path  --wandb_mode online   # online | offline | disabled
```

### `scripts/train_dlm.py`
```
--data_dir data  --encoder_ckpt runs/encoder/checkpoint_best.pt   # required
--run_dir runs/dlm  --resume latest
--seed 0  --max_steps 150000  --batch_size 128  --warmup_steps 2000  --grad_clip 1.0
--optimizer muon                                      # 'muon' (default) | 'adamw'; matches the
                                                       # canonical ELF implementation
                                                       # (arXiv:2605.10938): Muon over the decoder's
                                                       # own >=2D hidden weight matrices, AdamW
                                                       # (--lr/--weight_decay) over everything else
                                                       # (embeddings/norms/biases/null_context)
--lr 2e-4  --weight_decay 0.01                        # AdamW group, both optimizer choices
--muon_lr 0.02  --muon_momentum 0.95  --muon_weight_decay 0.0
                                                       # Muon group; only used when --optimizer muon.
                                                       # The paper reports muon_lr=0.002 -- pass that
                                                       # explicitly to match it exactly
--patience 5  --tolerance 0.005                      # early stopping on ID optimal-rate
--cfg_dropout 0.1  --decode_branch_prob 0.2  --selfcond_prob 0.0  --lambda_ce 1.0
                                                       # selfcond_prob=0 (default) is vanilla: no
                                                       # self-conditioning at train OR sample time
--num_sample_steps 32  --guidance_scale 1.0           # Euler steps / CFG at sampling time
--n_layers 2  --n_heads 8  --d_mlp 512  --dropout 0.1
--log_every 100  --eval_every 1000  --full_eval_every 5000
--n_id_subsample 500  --n_ood_subsample 300  --n_viz_examples 4  --num_workers 0
--wandb_project shortest-path-elf  --wandb_run_name dlm-run1
--wandb_group graph-shortest-path  --wandb_mode online
```

### `scripts/train_arlm.py`
Same shape as `train_dlm.py` minus the diffusion-specific flags
(`cfg_dropout`/`decode_branch_prob`/`selfcond_prob`/`lambda_ce`/`num_sample_steps`/
`guidance_scale`); defaults differ slightly: `--max_steps 100000  --lr 3e-4
--warmup_steps 1000`.

### `scripts/eval_only.py`
```
--checkpoint runs/dlm/checkpoint_best.pt  --model_kind dlm   # 'dlm' | 'arlm', both required
--data_dir data
--encoder_ckpt ...        # optional; defaults to the path stored in the checkpoint's own config
--split both               # 'id' | 'ood' | 'both'
--batch_size 128
--num_sample_steps 32  --guidance_scale 1.0     # DLM only
--n_viz_examples 8
--save_viz_dir ...         # optional; also save example PNGs locally
--num_workers 0
--wandb_project shortest-path-elf  --wandb_run_name ...  --wandb_group graph-shortest-path
--wandb_mode disabled      # online | offline | disabled (default disabled, unlike the training scripts)
```

### `scripts/eval_venn.py`
Runs the same `metrics.run_eval` as `eval_only.py`, but instead of printing/logging rates
it renders a two-panel (ID | OOD) Venn diagram of `valid`/`shortest`/`correct_length`/
`optimal` (see the Evaluation section above for what those mean): "shortest" and "correct
length" are drawn as two fixed-size overlapping circles (schematic layout, not sized or
positioned by count) enclosed in a dashed "valid path" boundary, with the actual count and
percentage of the split reported as text in each of the four regions -- their overlap is
exactly `optimal`; invalid generations are reported as a count/percentage outside the
"valid path" boundary.
```
--checkpoint runs/dlm/checkpoint_best.pt  --model_kind dlm   # 'dlm' | 'arlm', both required
--data_dir data
--encoder_ckpt ...        # optional; defaults to the path stored in the checkpoint's own config
--batch_size 128
--num_sample_steps 32  --guidance_scale 1.0     # DLM only
--num_workers 0
--out ...                  # optional; defaults to venn_{model_kind}_{checkpoint_stem}.png
```

### `scripts/dataset_stats.py`
Reports, per split (train / val_id / val_ood / encoder_pretrain_extra) and written to a
text file: example and unique-graph counts, unpadded input/target token-count statistics
(mean/std/min/max), node-count statistics, and graph-connectivity statistics -- edge
count, average degree, density, diameter, average shortest-path length -- each computed
once per *unique* graph in the split (not per example, which would double-count a graph
once per diametric-path example or per edge-shuffled copy it contributes). Also reports
`implied_diameter_from_target`, a decode-free cross-check computed directly from target
sequence lengths; it's the *example*-weighted diameter mean (graphs with more diametric
paths, hence more examples, count more), so it's expected to differ slightly from the
*graph*-weighted `diameter` connectivity stat, not a bug.
```
--data_dir data
--out ...                  # optional; defaults to {data_dir}/dataset_stats.txt
```

### `scripts/inspect_t5_tokenization.py`
Shows exactly how T5's real tokenizer segments a batch of real `graph_to_text`/
`path_to_text` examples -- token boundaries marked with `|` -- and automatically checks
(via the tokenizer's own char-offset mapping) that no single token ever spans two
different node-id numbers, exiting non-zero if that check ever fails. See "T5 as an
alternative encoder" above for why this matters.
```
--data_dir data
--split val_id             # 'train' | 'val_id' | 'val_ood'
--t5_model_name t5-small
--n_samples 20
--seed 0
--out ...                  # optional; defaults to {data_dir}/t5_tokenization_samples.txt
```

## Running it

```bash
pip install -r requirements.txt

python scripts/generate_data.py --seed 42 --out_dir data/
python scripts/dataset_stats.py --data_dir data/ --out data/dataset_stats.txt
python scripts/inspect_t5_tokenization.py --data_dir data/ --n_samples 20
python scripts/pretrain_encoder.py --data_dir data/ --run_dir runs/encoder/
python scripts/train_dlm.py  --data_dir data/ --encoder_ckpt runs/encoder/checkpoint_best.pt --run_dir runs/dlm/  --wandb_run_name dlm-run1
python scripts/train_arlm.py --data_dir data/ --encoder_ckpt runs/encoder/checkpoint_best.pt --run_dir runs/arlm/ --wandb_run_name arlm-run1

# T5 as the encoder instead (see "T5 as an alternative encoder" above) -- no --encoder_ckpt needed:
python scripts/train_dlm.py  --data_dir data/ --encoder_kind t5 --run_dir runs/dlm_t5/  --wandb_run_name dlm-t5-run1
python scripts/train_arlm.py --data_dir data/ --encoder_kind t5 --run_dir runs/arlm_t5/ --wandb_run_name arlm-t5-run1

# later, standalone:
python scripts/eval_only.py --checkpoint runs/dlm/checkpoint_best.pt  --model_kind dlm  --data_dir data/ --split both
python scripts/eval_only.py --checkpoint runs/arlm/checkpoint_best.pt --model_kind arlm --data_dir data/ --split both
python scripts/eval_venn.py --checkpoint runs/dlm/checkpoint_best.pt  --model_kind dlm  --data_dir data/
```

Run `pytest tests/` to check the tokenizer and graph generator's unit tests.
