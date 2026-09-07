# spelf: Shortest-Path ELF

An [ELF](https://arxiv.org/abs/2605.10938v2)-style continuous diffusion
language model, evaluated on a graph-reasoning task: **given a graph, find a
diametric path** (any one of its longest shortest paths). Follows the ELF
recipe exactly -- a real frozen pretrained T5 text encoder, a flow-matching +
decoder-CE diffusion transformer, the Muon optimizer -- adapted from
natural-language generation to this small, structured domain. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full design writeup and every
decision's rationale.

## Setup

```bash
pip install -r requirements.txt
wandb login   # logging to wandb is on by default; see "Weights & Biases" below
```

## Quickstart

Three steps, each reading only what the previous step wrote to disk:

```bash
# 1. Generate the ID/OOD Erdos-Renyi datasets (once)
python scripts/generate_data.py --config configs/default.yml

# 2. Load the pretrained T5 encoder and cache its latent normalization stats (once)
python scripts/prepare_encoder.py --config configs/default.yml

# 3. Train the ELF diffusion LM (logs to wandb + checkpoints by default)
python scripts/train.py --config configs/default.yml
```

Resuming is automatic: re-running step 3 with the same `output_dir` picks up
the latest checkpoint and reattaches to the same wandb run (see
"Checkpointing & wandb resume" in ARCHITECTURE.md).

Standalone evaluation of a saved checkpoint (ID + OOD, same four metrics +
sample images as periodic in-training eval):

```bash
python scripts/eval.py --config configs/default.yml --checkpoint runs/dlm/checkpoint_12000.pt
```

Override any config field from the command line:

```bash
python scripts/train.py --config configs/default.yml \
  --config_override epochs=100 --config_override batch_size=256
```

## Task

Erdos-Renyi graphs are generated offline (`scripts/generate_data.py`) and
cached to JSONL. Node ids are drawn from a fixed universe
(`range(ood_max_nodes)`, default 14) regardless of graph size, so ID
training exercises every node-id token even though ID graphs only have
6-10 nodes -- OOD (11-14 nodes) tests generalization to larger *structures*,
not to unseen *vocabulary*. See ARCHITECTURE.md, "Node-id label pool".

Each example is serialized as text, encoded with the real pretrained T5
tokenizer -- every node id is single-space-delimited from its neighbors, so
it always lands in its own token(s) and never bleeds into an adjacent id
(see ARCHITECTURE.md, "Serialization and tokenizer"; run
`scripts/inspect_t5_tokenization.py` to see the exact token split, `|`
between tokens -- `data_sample.txt` at the repo root has a saved example):

```
condition: nodes: 3 7 1 12 edges: 3 - 7 7 - 1 1 - 12 3 - 12 find diametric path
target:    3 7 1 12
```

## Evaluation metrics

Logged to wandb every `eval_freq` epochs, for both the ID (held-out) and OOD
splits, as `eval_id/*` / `eval_ood/*` (and `final_eval_id/*` /
`final_eval_ood/*` from `scripts/eval.py`):

- **valid_path_rate** -- output is an actual path in the graph (real edges,
  no repeated nodes).
- **shortest_path_rate** -- output is *a* shortest path between its own two
  endpoints.
- **correct_length_rate** -- output's length equals the graph's diameter.
- **optimal_path_rate** -- all three at once: a genuine diametric path.

5 sample graphs per split per eval are rendered with the predicted path
highlighted and logged as `wandb.Image`s, captioned with the model's raw
text output.

## Weights & Biases

On by default (`use_wandb: true`). Checkpointing and wandb resume are
synced through the run id, which is stored inside every checkpoint: resuming
from a checkpoint automatically reattaches to that same wandb run, with
nothing for you to track by hand. Disable with
`--config_override use_wandb=false`, or run offline with
`--config_override wandb_mode=offline`.

## Tests

```bash
pytest tests/ -q
```

## Repository layout

```
configs/default.yml        Editable template config (see spelf/common.py::Config for every field)
data_sample.txt             A saved run of inspect_t5_tokenization.py -- example serializations
scripts/
  generate_data.py         Pre-generate + cache the train/id_val/ood_test JSONL splits
  prepare_encoder.py         Load the pretrained T5 encoder + compute its latent norm stats
  inspect_t5_tokenization.py  Print example serializations with `|` between T5 tokens
  train.py                     Train the ELF diffusion LM; periodic ID+OOD eval; wandb + checkpoints
  eval.py                        Standalone ID+OOD eval of a saved checkpoint
src/spelf/
  common.py                 Config dataclass, YAML/CLI overrides, seeding, device resolution
  graphgen.py                Erdos-Renyi sampling, BFS, diameter search (all graph math)
  tokenizer.py                 Serialization grammar + the real T5 tokenizer loader
  dataset.py                     Graph<->text serialization, Dataset, batch collation
  data_cache.py                    Generate-and-cache-to-disk (JSONL) for the three splits
  t5_encoder.py                      Frozen pretrained T5 encoder wrapper + latent-stats I/O
  modules.py                           ELF transformer layer primitives (RoPE attn, SwiGLU, RMSNorm, ...)
  dlm.py                                 The ELF diffusion transformer + ELF-XS/S/M size presets
  muon.py                                  Self-contained Muon optimizer
  sampling.py                                Flow-matching schedules, ODE/SDE sampler, decode head
  train_step.py                                One training step (flow-matching + decoder CE loss)
  train_state.py                                 TrainState, optimizer/LR-schedule builders, EMA
  checkpoint.py                                    Save/load/discover checkpoints; wandb-run-id sync
  metrics.py                                         The four eval metrics + generation/scoring loop
  viz.py                                               Graph+path rendering -> captioned wandb.Image
tests/                                                 pytest suite (unit + one integration smoke test)
ARCHITECTURE.md                                        Full design writeup and rationale
```
