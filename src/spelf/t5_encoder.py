"""T5-based conditioning encoders, selected via --encoder_kind t5 in train_dlm.py /
train_arlm.py / eval_only.py / eval_venn.py.

Two distinct classes live here, because "use T5" means two different things for the
two decoders:

- `T5GraphEncoder`: T5 as a *conditioning source* for `GPTDecoder` (train_arlm.py),
  plus -- since there's otherwise no reason left to keep this project's own small
  vocabulary around -- the decoder generates directly *into T5's own vocabulary* too.
  T5's own contextualized hidden states are used directly as cross-attention context --
  no projection layer, so this encoder's own `d_model` is always T5's own hidden size,
  tying the two systems' dimension exactly as `T5DiffusionEncoder` does below. Unlike
  `T5DiffusionEncoder`, though, the decoder's own target-token embedding and its output
  unembedding (`UntiedEmbedding`, sized to `len(t5_tokenizer)`) are two ordinary,
  independently-trained matrices with no weight tying at all -- not to each other, and
  not to T5's own (frozen) embedding table, even though they now share T5's
  vocabulary -- exactly what a from-scratch GPT decoder head normally looks like, just
  aimed at a bigger, T5-shaped target. (This is a deliberate design correction, made
  after the DLM's own T5 redesign below prompted revisiting this class too -- see the
  conversation that changed it.)

- `T5DiffusionEncoder`: matches the canonical ELF implementation's actual use of T5
  (arXiv:2605.10938) -- T5's own frozen token embedding table *is* the space the DLM
  diffuses into (x, z_t, x_hat all live there; the final unembed is tied to that same
  frozen table), and T5's own contextualized hidden states are used directly as
  cross-attention context, with no projection layer at all (T5's embedding dimension
  and hidden-state dimension are the same throughout its stack, so no projection is
  needed once the decoder's own d_model is set to match). This is what `DLMDecoder`
  (train_dlm.py) uses when --encoder_kind t5 -- diffusing in a small from-scratch
  vocab while only *conditioning* on T5 (the old design) doesn't reflect what "ELF with
  a T5 encoder" actually means, so this replaces that for the DLM specifically. See
  T5DiffusionEncoder's own docstring for the practical consequences (a much wider
  decoder, and target sequences that are T5's own tokenization of a serialized path).

Both classes reuse `graph_to_text`: node-id numbers are always bounded by literal
whitespace on both sides (not just written next to each other with a bare "-" or ","),
because a space is a hard token boundary for SentencePiece/BPE tokenizers -- text can
never merge across it into a single token. Verified directly against T5's real
tokenizer (not just assumed): "3-7" tokenizes as a single fused token "1-4" for some
adjacent node-id pairs (silently merging the two node ids into one indivisible id --
exactly the failure mode this format avoids), while "3 - 7" reliably tokenizes as
"_3 | _ | - | _7", three clean tokens with the numbers on either side never touching. See
scripts/inspect_t5_tokenization.py to see this for real (not just this docstring's
claim) against a batch of real generated examples.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import torch
import torch.nn as nn

from . import tokenizer as tok


def graph_to_text(decoded: dict) -> str:
    """Deterministic text serialization of a decoded input (see tokenizer.decode_input)
    for T5 to encode. Every node-id number is bounded by a literal space on both sides
    -- including around "-" and "," and before the closing "." -- so no node id can
    ever share a token with a different node id or with adjacent punctuation, no matter
    how T5's specific BPE merges happen to fall (see this module's docstring)."""
    edges = " , ".join(f"{u} - {v}" for u, v in decoded["edges"])
    nodes = " ".join(str(x) for x in decoded["node_list"])
    return (
        f"Graph with nodes : {nodes} . Edges : {edges} . "
        f"Find a path whose length equals the graph's diameter "
        f"(the longest shortest path between any two nodes) ."
    )


def path_to_text(path: Sequence[int]) -> str:
    """Deterministic text serialization of a path (a list of node ids) -- the
    T5DiffusionEncoder analogue of tokenizer.encode_target: this is the text whose T5
    tokenization becomes the DLM's actual diffusion target when diffusing in T5's own
    embedding space. Same node-id spacing discipline as graph_to_text, for the same
    reason -- here it matters even more, since a merged token would make two distinct
    path nodes literally indistinguishable to the model's own target embedding."""
    return "Path : " + " - ".join(str(n) for n in path) + " ."


_PATH_TEXT_RE = re.compile(r"^Path\s*:\s*(\d+(?:\s*-\s*\d+)*)\s*\.$")


def decode_t5_path_text(text: str) -> Optional[list[int]]:
    """Parse decoded T5 text back into a project-vocab node-id path list, or None if it
    doesn't match the "Path : n - n - ... - n ." template -- the T5-text analogue of
    tokenizer.decode_target. Deliberately tolerant of exactly how much whitespace
    surrounds each token (T5's own tokenizer.decode() does not perfectly reproduce the
    spacing path_to_text wrote -- e.g. it collapses "0 ." to "0." -- so this matches on
    the numbers/hyphens/structure, not exact whitespace), but strict about everything
    else: a missing "Path :" prefix, a missing trailing ".", or any non-numeric,
    non-hyphen content in between all cause this to return None rather than guess.
    Does NOT check node-id range or duplicate nodes -- same division of responsibility
    as tokenizer.decode_target: this is format-parsing only; semantic validity (real
    node ids, no repeats, real edges) is metrics.evaluate_generation's job."""
    m = _PATH_TEXT_RE.match(text.strip())
    if not m:
        return None
    return [int(x) for x in re.findall(r"\d+", m.group(1))]


def decode_t5_path_ids(t5_tokenizer, ids: Sequence[int]) -> Optional[list[int]]:
    """T5-token-id-space analogue of tokenizer.decode_target: parse a (possibly
    model-generated) sequence of T5 subword ids back into a project-vocab node-id path,
    or None if malformed. Truncates at the first </s> (eos) and ignores everything
    after it, mirroring decode_target's own relaxation (see that function's docstring):
    dlm.loss excludes pad positions from both its losses when diffusing in T5's vocab
    too, so nothing trains the model on what belongs after </s>, and a generation is
    read by finding it, not by validating its tail."""
    ids = list(ids)
    eos_id = t5_tokenizer.eos_token_id
    if eos_id in ids:
        ids = ids[:ids.index(eos_id)]
    text = t5_tokenizer.decode(ids, skip_special_tokens=False)
    return decode_t5_path_text(text)


def compute_l_tgt_t5(t5_tokenizer, margin: int = 8) -> int:
    """Fixed T5-token-space length for a decoder's fixed-size canvas when generating
    directly in T5's own vocabulary -- used by both T5DiffusionEncoder (the DLM's
    diffusion canvas) and T5GraphEncoder (ARLM's `l_tgt_t5`, which adds 1 more for a
    prepended decoder-start marker, see T5GraphEncoder.tokenize_path_targets). Every
    batch's target sequence must be padded/truncated to one shared constant, exactly
    like tokenizer.TARGET_LENGTH is for the project's own vocab.

    Computed from the true worst case rather than scanning the dataset: a path visits
    at most tok.MAX_NODES distinct nodes (it can't repeat one), so the longest possible
    path text is path_to_text(<all MAX_NODES node ids>); empirically (see the
    conversation that added this function) every node-id number contributes exactly
    one T5 token regardless of digit count, so no other node-id subset or ordering can
    produce a longer tokenization -- verified against 2000+ random paths, all <= this
    worst case. `margin` is pure safety headroom against tokenizer quirks not covered
    by that verification, not a correction to a known gap."""
    worst_path = list(range(tok.MAX_NODES))
    ids = t5_tokenizer(path_to_text(worst_path))["input_ids"]
    return len(ids) + margin


class UntiedEmbedding(nn.Module):
    """An ordinary, independently-trained input-embedding + output-unembedding pair --
    no weight tying at all, unlike modules.SharedEmbedding (tied embed/unembed, and
    *shared* -- the literal same object -- into a decoder from whichever encoder
    pretrained it) or T5TiedEmbedding (tied to T5's own frozen table below). This is
    what a from-scratch GPT decoder head normally looks like: `nn.Embedding(vocab_size,
    d_model)` for input, a separate `nn.Linear(d_model, vocab_size)` for output logits.
    Used by T5GraphEncoder over T5's own vocabulary (`vocab_size = len(t5_tokenizer)`,
    ARLM+T5 decodes *into* T5's vocab, not the project's own 21-token one -- see that
    class's docstring), where there is no pretrained embedding space worth tying into --
    tying was never load-bearing there, just borrowed convention from
    GraphEncoder/T5DiffusionEncoder, where it actually is."""

    def __init__(self, vocab_size: int = tok.VOCAB_SIZE, d_model: int = 128):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_embedding.weight, mean=0.0, std=0.02)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embedding(ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.tok_embedding.weight

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)


class T5GraphEncoder(nn.Module):
    """T5 as a pure conditioning source -- see this module's docstring for how this
    differs from T5DiffusionEncoder. Used by train_arlm.py's --encoder_kind t5, and
    available for eval_only.py/eval_venn.py against ARLM+T5 checkpoints.

    T5's contextualized hidden states are used directly as cross-attention context, with
    no projection layer -- T5's embedding dimension and hidden-state dimension are the
    same throughout its stack, so `self.d_model` is always T5's own `config.d_model`
    (512 for t5-small), not a free choice; `GPTDecoder`'s own d_model must match it
    exactly for cross-attention's shapes to line up.

    `GPTDecoder` also decodes *into* T5's own vocabulary, not this project's own
    21-token one: `self.embedding` is an UntiedEmbedding sized to
    `len(self.t5_tokenizer)` (T5's real, tight vocabulary bound -- see
    `tokenize_path_targets`'s docstring for why this specific value), so training
    targets are T5's own tokenization of `path_to_text(path)` rather than
    `tokenizer.encode_target(path)`. `embedding`'s embed and unembed remain fully
    untied from each other AND from T5's own (frozen) embedding table -- see
    UntiedEmbedding's docstring for why tying isn't meaningful here even though the
    vocabulary now happens to be the same one T5 itself uses. Only `embedding` is
    trainable; the T5 stack itself stays frozen throughout, and (since nothing else
    here is trainable either) its forward pass never needs gradient to flow through it
    at all.
    """

    context_requires_grad = False  # nothing trainable on the T5 side -- see class docstring

    def __init__(self, model_name: str = "t5-small", max_text_len: int = 256):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.model_name = model_name
        self.max_text_len = max_text_len
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.t5 = T5EncoderModel.from_pretrained(model_name)

        self.d_model = self.t5.config.d_model
        self.embedding = UntiedEmbedding(vocab_size=len(self.t5_tokenizer), d_model=self.d_model)
        # +1 beyond compute_l_tgt_t5's own bound for the prepended decoder-start marker
        # -- see tokenize_path_targets.
        self.l_tgt_t5 = compute_l_tgt_t5(self.t5_tokenizer) + 1

        self.freeze()

    def freeze(self) -> None:
        """Freeze the pretrained T5 stack only; `embedding` stays trainable (see class
        docstring). Kept as an explicit method, mirroring GraphEncoder.freeze(), so
        load_t5_encoder reads the same as load_frozen_encoder."""
        for p in self.t5.parameters():
            p.requires_grad = False
        self.t5.eval()

    def train(self, mode: bool = True):
        """Override so an accidental `t5_graph_encoder.train()` (e.g. via a parent
        module's recursive .train()) can never put the frozen T5 stack into train mode
        (dropout etc.) -- `embedding` still switches normally since it's a plain
        submodule without its own train()/eval() semantics beyond dropout, which it
        doesn't have."""
        super().train(mode)
        self.t5.eval()
        return self

    def trainable_state_dict(self) -> dict:
        """The only weights this encoder needs checkpointed -- the frozen T5 stack is
        reproducible from `model_name` alone, so re-saving its ~tens-of-millions of
        frozen params on every checkpoint would be pure waste."""
        return {"embedding": self.embedding.state_dict()}

    def load_trainable_state_dict(self, state: dict) -> None:
        self.embedding.load_state_dict(state["embedding"])

    def _texts_from_batch(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> list[str]:
        texts = []
        for row, m in zip(input_ids.cpu().tolist(), input_mask.cpu().tolist()):
            real = [t for t, keep in zip(row, m) if keep]
            decoded = tok.decode_input(real)
            assert decoded is not None, "T5GraphEncoder requires well-formed (unmasked) inputs"
            texts.append(graph_to_text(decoded))
        return texts

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        texts = self._texts_from_batch(input_ids, input_mask)
        enc = self.t5_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_len,
        ).to(input_ids.device)
        with torch.no_grad():
            hidden = self.t5(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        return hidden, enc["attention_mask"].bool()

    def tokenize_path_targets(self, target_ids: torch.Tensor, device) -> torch.Tensor:
        """Batch of project-vocab target_ids (as stored in PathDataset, decoded via
        tokenizer.decode_target) -> T5's own tokenization of path_to_text(path), with
        T5's own `pad_token_id` prepended as a decoder-start marker -- this is what
        actually gets passed as `target_ids` to arlm.loss, and what arlm.sample's
        `start_id`/`pad_id` must also be set to (see train_arlm.py). T5 has no
        dedicated BOS token, so using pad_token_id as the decoder-start marker mirrors
        the standard T5/seq2seq convention (`decoder_start_token_id = pad_token_id`,
        e.g. HF's own T5ForConditionalGeneration default) rather than inventing a new
        one. Padded/truncated to this encoder's fixed `self.l_tgt_t5`; asserts (rather
        than silently truncating) if a real path's tokenization plus the prepended
        marker would exceed it."""
        texts = []
        for row in target_ids.cpu().tolist():
            path = tok.decode_target(row)
            assert path is not None, "ground-truth target should always be well-formed"
            texts.append(path_to_text(path))
        pad_id = self.t5_tokenizer.pad_token_id
        unpadded = self.t5_tokenizer(texts, padding=False, truncation=False)["input_ids"]
        longest = max(len(ids) for ids in unpadded) + 1  # +1 for the prepended start marker
        assert longest <= self.l_tgt_t5, (
            f"a target path's T5 tokenization plus start marker ({longest} tokens) exceeds "
            f"l_tgt_t5={self.l_tgt_t5} -- compute_l_tgt_t5's margin needs increasing"
        )
        enc = self.t5_tokenizer(
            texts, return_tensors="pt", padding="max_length", max_length=self.l_tgt_t5 - 1, truncation=False,
        )["input_ids"]
        start = torch.full((enc.shape[0], 1), pad_id, dtype=torch.long)
        return torch.cat([start, enc], dim=1).to(device)


def load_t5_encoder(model_name: str, device) -> T5GraphEncoder:
    encoder = T5GraphEncoder(model_name=model_name).to(device)
    encoder.eval()
    return encoder


class T5TiedEmbedding(nn.Module):
    """Adapts T5's own frozen input embedding table to the same interface
    (forward/unembed/.weight) modules.SharedEmbedding exposes, so DLMDecoder can be
    handed `T5DiffusionEncoder.embedding` exactly the way it's handed
    GraphEncoder.embedding or T5GraphEncoder.embedding -- no caller needs to know which
    kind of embedding it received. Holds no parameters of its own; `t5_embedding` is a
    submodule reference to T5's own (already-frozen, as part of freezing the whole T5
    stack) embedding table, so this is a view, not a copy."""

    def __init__(self, t5_embedding: nn.Embedding):
        super().__init__()
        self.t5_embedding = t5_embedding

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.t5_embedding(ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.t5_embedding.weight

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.t5_embedding.weight.t()


class T5DiffusionEncoder(nn.Module):
    """T5 as the DLM's actual diffusion space, matching the canonical ELF
    implementation's use of a frozen T5 encoder (arXiv:2605.10938) -- see this module's
    docstring for why this differs from T5GraphEncoder. Concretely:

    - `self.embedding` is a `T5TiedEmbedding` wrapping T5's own frozen token embedding
      table (~32k subwords) -- not a separate, freshly-trained small table. The DLM's
      diffusion target `x`, its noisy `z_t`, its prediction `x_hat`, and its final
      unembedded logits are all in this space. This also means the DLM's decoder must
      operate at `d_model = t5.config.d_model` (512 for t5-small), not this project's
      usual 128 -- a genuinely larger decoder, not just a different embedding source.
    - There is no `proj` layer: T5's contextualized `last_hidden_state` and its
      embedding table share the same dimension throughout T5's stack, so once the
      decoder's own d_model matches T5's, `last_hidden_state` can be used directly as
      cross-attention context with no projection needed.
    - Because of the above, this encoder has *zero* of its own trainable parameters --
      the whole T5 stack (embedding included) is frozen; only DLMDecoder's own weights
      train. `trainable_state_dict()`/`load_trainable_state_dict()` are kept as
      no-op-shaped stubs purely so train_dlm.py's checkpoint code doesn't need a special
      case distinguishing this from T5GraphEncoder.
    - Training targets are no longer tokenizer.encode_target(path) (this project's own
      21-token vocab) -- they're T5's own tokenization of path_to_text(path), produced
      by `tokenize_path_targets`. `l_tgt_t5` (see compute_l_tgt_t5) is this encoder's
      fixed target-canvas length, computed once at construction time and then fixed for
      the life of the run (and saved into the checkpoint config, so eval scripts don't
      need to recompute it).
    """

    context_requires_grad = False  # nothing trainable here at all -- see class docstring

    def __init__(self, model_name: str = "t5-small", max_text_len: int = 256, l_tgt_t5: Optional[int] = None):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.model_name = model_name
        self.max_text_len = max_text_len
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.t5 = T5EncoderModel.from_pretrained(model_name)

        self.d_model = self.t5.config.d_model
        self.embedding = T5TiedEmbedding(self.t5.get_input_embeddings())
        self.l_tgt_t5 = l_tgt_t5 if l_tgt_t5 is not None else compute_l_tgt_t5(self.t5_tokenizer)

        self.freeze()

    def freeze(self) -> None:
        for p in self.t5.parameters():
            p.requires_grad = False
        self.t5.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.t5.eval()
        return self

    def trainable_state_dict(self) -> dict:
        return {}

    def load_trainable_state_dict(self, state: dict) -> None:
        pass

    def _texts_from_batch(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> list[str]:
        texts = []
        for row, m in zip(input_ids.cpu().tolist(), input_mask.cpu().tolist()):
            real = [t for t, keep in zip(row, m) if keep]
            decoded = tok.decode_input(real)
            assert decoded is not None, "T5DiffusionEncoder requires well-formed (unmasked) inputs"
            texts.append(graph_to_text(decoded))
        return texts

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        texts = self._texts_from_batch(input_ids, input_mask)
        enc = self.t5_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_len,
        ).to(input_ids.device)
        with torch.no_grad():
            hidden = self.t5(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        return hidden, enc["attention_mask"].bool()

    def tokenize_path_targets(self, target_ids: torch.Tensor, device) -> torch.Tensor:
        """Batch of project-vocab target_ids (as stored in PathDataset, decoded via
        tokenizer.decode_target) -> T5's own tokenization of path_to_text(path),
        padded/truncated to this encoder's fixed self.l_tgt_t5 -- this is what actually
        gets passed as `target_ids` to dlm.loss when diffusing in T5 space. Asserts
        (rather than silently truncating) if a real path's T5 tokenization somehow
        exceeds l_tgt_t5 -- see compute_l_tgt_t5's docstring for why that shouldn't
        happen for any path this dataset can produce."""
        texts = []
        for row in target_ids.cpu().tolist():
            path = tok.decode_target(row)
            assert path is not None, "ground-truth target should always be well-formed"
            texts.append(path_to_text(path))
        unpadded = self.t5_tokenizer(texts, padding=False, truncation=False)["input_ids"]
        longest = max(len(ids) for ids in unpadded)
        assert longest <= self.l_tgt_t5, (
            f"a target path's T5 tokenization ({longest} tokens) exceeds l_tgt_t5="
            f"{self.l_tgt_t5} -- compute_l_tgt_t5's margin needs increasing"
        )
        enc = self.t5_tokenizer(
            texts, return_tensors="pt", padding="max_length", max_length=self.l_tgt_t5, truncation=False,
        )
        return enc["input_ids"].to(device)


def load_t5_diffusion_encoder(model_name: str, device, l_tgt_t5: Optional[int] = None) -> T5DiffusionEncoder:
    encoder = T5DiffusionEncoder(model_name=model_name, l_tgt_t5=l_tgt_t5).to(device)
    encoder.eval()
    return encoder


class T5SpaceDecoderAdapter:
    """Wraps a decoder (DLMDecoder or GPTDecoder) that generates in T5's own token-id
    space (T5DiffusionEncoder for the DLM, T5GraphEncoder for the ARLM) so it exposes
    the same `.generate(context, context_mask, **kwargs) -> LongTensor[B,
    tokenizer.TARGET_LENGTH]` interface, in the project's own 21-token vocab, that every
    other decoder in this codebase exposes -- this is what lets metrics.run_eval and
    viz.plot_example stay completely unaware T5 was ever involved, rather than needing a
    parallel T5-aware scoring/plotting path. A generation this can't parse back into a
    well-formed path (decode_t5_path_ids returns None, or the path doesn't fit
    tokenizer.TARGET_LENGTH) re-encodes to an all-<PAD> row, which
    tokenizer.decode_target already treats as invalid (its first required token is
    <P>), so it flows through exactly like any other malformed generation everywhere
    downstream.

    `drop_first_token`: set True for the ARLM (only) -- arlm.sample always writes a
    fixed decoder-start marker (T5's pad_token_id, see T5GraphEncoder.tokenize_path_targets)
    into position 0 of every generation, which isn't part of the actual path text and
    must be stripped before decode_t5_path_ids tries to parse it (otherwise the decoded
    text starts with a literal "<pad>" and never matches the "Path : ..." template). The
    DLM has no such marker -- its raw generation starts directly with content."""

    def __init__(self, model, t5_tokenizer, drop_first_token: bool = False):
        self.model = model
        self.t5_tokenizer = t5_tokenizer
        self.drop_first_token = drop_first_token

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    @torch.no_grad()
    def generate(self, context: torch.Tensor, context_mask: torch.Tensor, **sample_kwargs) -> torch.Tensor:
        raw = self.model.generate(context, context_mask, **sample_kwargs)  # [B, l_tgt_t5], T5 vocab ids
        device = raw.device
        out = torch.zeros(raw.shape[0], tok.TARGET_LENGTH, dtype=torch.long, device=device)  # all-<PAD> default
        for i, row in enumerate(raw.tolist()):
            if self.drop_first_token:
                row = row[1:]
            path = decode_t5_path_ids(self.t5_tokenizer, row)
            if path is None or any(not (0 <= n < tok.MAX_NODES) for n in path):
                continue  # leave as all-<PAD> -> tok.decode_target(...) treats it as invalid
            try:
                out[i] = torch.tensor(tok.encode_target(path), dtype=torch.long, device=device)
            except ValueError:
                pass  # path too long to fit TARGET_LENGTH -> also left as invalid
        return out
