"""T5-based conditioning encoders, selected via --encoder_kind t5 in train_dlm.py /
train_arlm.py / eval_only.py / eval_venn.py.

Two distinct classes live here, because "use T5" means two different things for the
two decoders:

- `T5GraphEncoder`: T5 purely as a *conditioning source* (frozen T5 -> trainable
  `proj` down to a freely-chosen `d_model`), while the decoder still embeds/unembeds
  its own targets in a small, separately-trained vocabulary (`SharedEmbedding` over
  this project's 21-token graph vocab). This is what `GPTDecoder` (train_arlm.py) uses,
  and is a perfectly ordinary, unproblematic design for an autoregressive model.

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
from .modules import SharedEmbedding


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
    or None if malformed. Mirrors decode_target's strictness about trailing content:
    everything after the first </s> must be only <pad>, else this returns None instead
    of silently ignoring garbage after the intended content."""
    ids = list(ids)
    eos_id = t5_tokenizer.eos_token_id
    pad_id = t5_tokenizer.pad_token_id
    if eos_id in ids:
        i = ids.index(eos_id)
        if any(t != pad_id for t in ids[i + 1:]):
            return None
        ids = ids[:i]
    text = t5_tokenizer.decode(ids, skip_special_tokens=False)
    return decode_t5_path_text(text)


class T5GraphEncoder(nn.Module):
    """T5 as a pure conditioning source -- see this module's docstring for how this
    differs from T5DiffusionEncoder. Used by train_arlm.py's --encoder_kind t5, and
    available for eval_only.py/eval_venn.py against ARLM+T5 checkpoints.

    T5's own embedding space is pretrained on natural-language subwords and has nothing
    to do with this project's 22-token graph vocab, so -- unlike GraphEncoder, whose
    embedding table *is* the pretrained conditioning signal and is shared verbatim into
    the decoder -- this encoder owns a fresh SharedEmbedding over the graph vocab that
    trains from scratch alongside the decoder, plus a trainable linear projection from
    T5's hidden size down to this project's d_model. Only those two pieces are
    trainable; the T5 stack itself stays frozen throughout.
    """

    context_requires_grad = True  # see common.encode_context

    def __init__(self, model_name: str = "t5-small", d_model: int = 128, max_text_len: int = 256):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.model_name = model_name
        self.max_text_len = max_text_len
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.t5 = T5EncoderModel.from_pretrained(model_name)

        self.d_model = d_model
        self.proj = nn.Linear(self.t5.config.d_model, d_model)
        self.embedding = SharedEmbedding(vocab_size=tok.VOCAB_SIZE, d_model=d_model)

        self.freeze()

    def freeze(self) -> None:
        """Freeze the pretrained T5 stack only; `proj` and `embedding` stay trainable
        (see class docstring). Kept as an explicit method, mirroring
        GraphEncoder.freeze(), so load_t5_encoder reads the same as load_frozen_encoder."""
        for p in self.t5.parameters():
            p.requires_grad = False
        self.t5.eval()

    def train(self, mode: bool = True):
        """Override so an accidental `t5_graph_encoder.train()` (e.g. via a parent
        module's recursive .train()) can never put the frozen T5 stack into train mode
        (dropout etc.) -- proj/embedding still switch normally since they're plain
        submodules without their own train()/eval() semantics beyond dropout, which
        neither has."""
        super().train(mode)
        self.t5.eval()
        return self

    def trainable_state_dict(self) -> dict:
        """The only weights this encoder needs checkpointed -- the frozen T5 stack is
        reproducible from `model_name` alone, so re-saving its ~tens-of-millions of
        frozen params on every checkpoint would be pure waste."""
        return {"proj": self.proj.state_dict(), "embedding": self.embedding.state_dict()}

    def load_trainable_state_dict(self, state: dict) -> None:
        self.proj.load_state_dict(state["proj"])
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
        context = self.proj(hidden)  # trainable projection -> gradient flows here even though `hidden` came from no_grad
        context_mask = enc["attention_mask"].bool()
        return context, context_mask


def load_t5_encoder(model_name: str, d_model: int, device) -> T5GraphEncoder:
    encoder = T5GraphEncoder(model_name=model_name, d_model=d_model).to(device)
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


def compute_l_tgt_t5(t5_tokenizer, margin: int = 8) -> int:
    """Fixed T5-token-space length for the DLM's diffusion canvas when diffusing
    directly in T5's embedding space (T5DiffusionEncoder) -- every batch's target
    sequence must be padded/truncated to one shared constant, exactly like
    tokenizer.TARGET_LENGTH is for the project's own vocab, since the DLM's positional
    encoding and fixed-size diffusion tensors assume one constant shape.

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


class T5SpaceDLMAdapter:
    """Wraps a DLMDecoder that was trained to diffuse/generate in T5's own token-id
    space (see T5DiffusionEncoder) so it exposes the same
    `.generate(context, context_mask, **kwargs) -> LongTensor[B, tokenizer.TARGET_LENGTH]`
    interface, in the project's own 21-token vocab, that every other decoder in this
    codebase exposes -- this is what lets metrics.run_eval and viz.plot_example stay
    completely unaware T5 was ever involved, rather than needing a parallel T5-aware
    scoring/plotting path. A generation this can't parse back into a well-formed path
    (decode_t5_path_ids returns None, or the path doesn't fit tokenizer.TARGET_LENGTH)
    re-encodes to an all-<PAD> row, which tokenizer.decode_target already treats as
    invalid (its first required token is <P>), so it flows through exactly like any
    other malformed generation everywhere downstream."""

    def __init__(self, model, t5_tokenizer):
        self.model = model
        self.t5_tokenizer = t5_tokenizer

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
            path = decode_t5_path_ids(self.t5_tokenizer, row)
            if path is None or any(not (0 <= n < tok.MAX_NODES) for n in path):
                continue  # leave as all-<PAD> -> tok.decode_target(...) treats it as invalid
            try:
                out[i] = torch.tensor(tok.encode_target(path), dtype=torch.long, device=device)
            except ValueError:
                pass  # path too long to fit TARGET_LENGTH -> also left as invalid
        return out
